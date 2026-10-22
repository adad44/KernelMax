"""One reviewed execution session. No numerical grading or verdicts here.

This is a reliability boundary, NOT an adversarial sandbox. The worker shares
the user's filesystem permissions. Only explicitly reviewed snapshots may run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import types

# -I excludes the scratch cwd/user paths; bootstrap only this trusted checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.candidate_protocol import (
    VERSION,
    file_digest,
    pack_tensor,
    read_json,
    unpack_tensor,
)
from orchestrator.candidate_manager import (
    FrozenCandidate,
    validate_candidate,
    verify_frozen_candidate,
)
from target.adapter import TrustedExpertAdapter
from target.reference_moe import ReferenceMoE


def _frozen(request):
    info = request["snapshot"]
    candidate = validate_candidate(info["directory"], Path(info["directory"]).parent)
    frozen = FrozenCandidate(
        **vars(candidate), sha256=info["sha256"], file_sha256=info["file_sha256"]
    )
    if frozen.sha256 != request["approved_sha256"]:
        raise ValueError("snapshot is not approved")
    verify_frozen_candidate(frozen)
    if request["identity"]["candidate"] != {
        "candidate_id": frozen.candidate_id,
        "implementation_type": frozen.implementation_type,
        "sha256": frozen.sha256,
    }:
        raise ValueError("snapshot does not match request identity")
    return frozen


def _binding(request):
    runtime = request["runtime"]
    if runtime["mode"] == "numpy_fixture":

        def expert(x, ids):
            return x[:, :, None, :] + ids[:, :, :, None].astype(x.dtype)

        return (
            None,
            lambda layer, x, ids: expert(x, ids),
            {i: expert for i in (0, 12, 23)},
        )
    if runtime["mode"] != "mlx_native":
        raise ValueError("unknown execution mode")
    root = Path(runtime["model_path"])
    manifest = runtime["model_files"]
    if not manifest:
        raise ValueError("native execution requires a checkpoint hash manifest")
    required = {"config.json", "model.safetensors.index.json"}
    index = read_json(root / "model.safetensors.index.json")
    required.update(index["weight_map"].values())
    if not required <= manifest.keys():
        raise ValueError("checkpoint manifest omits configuration/index/weight shards")
    for name, digest in manifest.items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or file_digest(path) != digest
        ):
            raise ValueError(f"checkpoint integrity failure: {name}")
    config = read_json(root / "config.json")
    if (
        config.get("model_type") != "gpt_oss"
        or config.get("quantization_config", {}).get("quant_method") != "mxfp4"
    ):
        raise ValueError("checkpoint must preserve native GPT-OSS MXFP4 weights")
    for field, expected in (
        ("hidden_size", runtime["hidden_size"]),
        ("num_local_experts", runtime["num_experts"]),
        ("num_experts_per_tok", 4),
    ):
        if config.get(field) != expected:
            raise ValueError(f"checkpoint configuration mismatch: {field}")
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten
    from evaluator.native_baseline import materialize_parameters

    def progress(completed, total, name, size):
        if completed % 25 == 0 or completed == total:
            print(f"native_load {completed}/{total}", flush=True)

    # Avoid the known large-model Metal loading timeout; untimed CPU materialization.
    with mx.stream(mx.cpu):
        model, _ = load(str(root), lazy=True)
        parameters = [
            (f"layer{layer}.{name}", value)
            for layer in (0, 12, 23)
            for name, value in tree_flatten(
                model.model.layers[layer].mlp.experts.parameters()
            )
        ]
        materialize_parameters(parameters, mx, progress)
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp.experts, name)
            if (module.mode, module.bits, module.group_size) != ("mxfp4", 4, 32):
                raise ValueError("native expert weight format changed")
    reference = ReferenceMoE(model)
    if runtime["expert_factory"]:
        module, symbol = runtime["expert_factory"].split(":")
        adapter = getattr(importlib.import_module(module), symbol)(model, mx=mx)
        if not isinstance(adapter, TrustedExpertAdapter):
            raise TypeError("expert factory must return TrustedExpertAdapter")
        if (
            adapter.hidden_size != runtime["hidden_size"]
            or adapter.num_experts != runtime["num_experts"]
            or adapter.mx is not mx
        ):
            raise ValueError("expert factory disagrees with runtime configuration")
        experts = dict(adapter.experts)
    else:
        experts = {i: model.model.layers[i].mlp.experts for i in (0, 12, 23)}
    return mx, reference, experts


def _candidate_entry(snapshot):
    # A private package supports relative helper imports without adding submissions
    # to sys.path or importing them in the parent/reference process.
    name = "_reviewed_candidate"
    package = types.ModuleType(name)
    package.__path__ = [str(snapshot.directory)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        name + ".wrapper", snapshot.directory / "wrapper.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    entry = getattr(module, "run", None)
    if not callable(entry):
        raise ValueError("wrapper.py must export run(experts, x, selected_expert_ids)")
    return entry


def main(request_path: Path, response_path: Path) -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = read_json(request_path)
    limit = request["max_response_bytes"]
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    with response_path.open("w", encoding="utf-8") as response:

        def emit(value):
            response.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
            response.flush()

        try:
            if request["version"] != VERSION or request["phase"] not in (
                "reference",
                "candidate",
            ):
                raise ValueError("invalid worker request")
            snapshot = _frozen(request)
            mx, reference, experts = _binding(request)  # Once per entire session.
            emit(
                {
                    "type": "session",
                    "version": VERSION,
                    "phase": request["phase"],
                    "identity": request["identity"],
                    "mode": request["runtime"]["mode"],
                    "pid": os.getpid(),
                    "materialized_expert_layers": [0, 12, 23],
                    "python": sys.version,
                    "executable": sys.executable,
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in (
                            ("numpy",) if mx is None else ("numpy", "mlx", "mlx-lm")
                        )
                    },
                }
            )
            entry = (
                _candidate_entry(snapshot) if request["phase"] == "candidate" else None
            )
            for item in request["inputs"]:
                row = {
                    "type": "result",
                    "case_name": item["case_name"],
                    "layer_index": item["layer_index"],
                    "seed": item["seed"],
                    "output": None,
                    "error": None,
                }
                try:
                    x, ids = unpack_tensor(item["x"]), unpack_tensor(item["ids"])
                    if mx is not None:
                        x, ids = (
                            mx.array(x, dtype=mx.bfloat16),
                            mx.array(ids, dtype=mx.int32),
                        )
                    before = (pack_tensor(x, mx), pack_tensor(ids, mx))
                    layer = item["layer_index"]
                    output = (
                        reference(layer, x, ids)
                        if entry is None
                        else entry(experts[layer], x, ids)
                    )
                    packed = pack_tensor(
                        output, mx
                    )  # Materialize while execution errors are attributable.
                    if before != (pack_tensor(x, mx), pack_tensor(ids, mx)):
                        raise ValueError("candidate mutated evaluator-owned inputs")
                    row["output"] = packed
                    row["input_sha256"] = hashlib.sha256(
                        json.dumps(before, sort_keys=True).encode()
                    ).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                emit(row)
                if mx is not None:
                    mx.clear_cache()
            verify_frozen_candidate(snapshot)
            emit({"type": "complete", "rows": len(request["inputs"])})
            return 0
        except Exception as exc:
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            return 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
