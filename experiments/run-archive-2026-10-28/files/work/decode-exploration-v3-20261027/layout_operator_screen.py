"""Pinned-checkpoint dense operator exploration; no full-model speedup claim."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics as st
import struct
import subprocess
import sys
import threading
import time
import traceback

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent / "kernelmax-first-loop"
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import environment, swap_observation
from evaluator.baseline_validation import thermal_state
from evaluator.measurement import (
    assess_environment,
    BoundedExploration,
    ExplorationComplete,
)
from orchestrator.candidate_manager import (
    FrozenCandidate,
    validate_candidate,
    verify_frozen_candidate,
)
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def bootstrap(values, seed):
    rng = random.Random(seed)
    values = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [values[249], values[9749]]


def load_bf16(root, index, key, mx, np):
    """Read precisely one pinned safetensor's original words, no whole-shard load."""
    path = root / index["weight_map"][key]
    with path.open("rb") as source:
        header_bytes = source.read(8)
        if len(header_bytes) != 8:
            raise ValueError("Truncated safetensor size")
        length = struct.unpack("<Q", header_bytes)[0]
        header = json.loads(source.read(length))
        info = header[key]
        if info["dtype"] != "BF16":
            raise ValueError("Dense checkpoint tensor must remain BF16")
        start, end = info["data_offsets"]
        if end - start != int(np.prod(info["shape"])) * 2:
            raise ValueError("Safetensor shape/data range mismatch")
        source.seek(8 + length + start)
        raw = source.read(end - start)
    if len(raw) != end - start:
        raise ValueError("Truncated tensor payload")
    words = np.frombuffer(raw, dtype="<u2").reshape(info["shape"])
    with mx.stream(mx.cpu):
        value = mx.array(words, dtype=mx.uint16).view(mx.bfloat16)
        mx.eval(value)
    return value, {
        "key": key,
        "shape": info["shape"],
        "shard": path.name,
        "native_tensor_bytes": len(raw),
        "original_words_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Fresh operator-screen directory required")
    # Check actual command lines rather than trusting an old state-file status.
    commands = subprocess.check_output(["ps", "-axo", "pid,command"], text=True)
    for line in commands.splitlines():
        if str(os.getpid()) == line.split(maxsplit=1)[0]:
            continue
        if any(
            token in line
            for token in (
                "decode-exploration-v3-20261027/profile.py",
                "decode-exploration-v3-20261027/layout_gpu_fixture.py",
            )
        ):
            raise RuntimeError("A prior GPU worker is still live")
    contract = load_contract(ROOT / "kernelmaxxing.yaml")
    session = load_authorization(WORK / "session.json", contract)
    info = json.loads((WORK / "layout-frozen.json").read_text())
    candidate = validate_candidate(info["directory"], Path(info["directory"]).parent)
    frozen = FrozenCandidate(
        **vars(candidate), sha256=info["sha256"], file_sha256=info["file_sha256"]
    )
    verify_frozen_candidate(frozen)
    fixture_path = WORK / "native-gpu-fixture-001/state.json"
    fixture = json.loads(fixture_path.read_text())
    if (
        fixture["status"] != "passed_native_synthetic_fixture"
        or len(fixture["records"]) != 26
        or fixture["snapshot_sha256"] != frozen.sha256
        or not all(r["passed"] for r in fixture["records"])
        or not fixture.get("protected_unchanged")
        or not fixture.get("snapshot_unchanged")
    ):
        raise RuntimeError("Complete matching native fixture pass required")
    runtime = native_runtime(Path(session["model_path"]), contract)
    receipt = json.loads(Path(session["checkpoint_cache"]).read_text())
    validate_receipt(
        receipt, runtime.model_path, runtime.model_revision, runtime.model_files
    )
    import mlx.core as mx
    import numpy as np

    extra = [
        __file__,
        WORK / "layout-operator-plan.json",
        WORK / "layout-frozen.json",
        WORK / "session.json",
        fixture_path,
        mx.__file__,
    ]
    protected = _protected_files(extra)
    output.mkdir(parents=True)
    state = {
        "status": "loading_dense_weights",
        "pid": os.getpid(),
        "mode": "exploration",
        "official": False,
        "verdict": None,
        "acceptance_enabled": False,
        "speedup_claim_eligible": False,
        "contract_sha256": contract.sha256,
        "candidate_id": frozen.candidate_id,
        "snapshot_sha256": frozen.sha256,
        "protected_before": protected,
        "checkpoint_receipt": receipt,
        "started_unix": time.time(),
        "scope": "13 actual checkpoint BF16 dense matrices; varied synthetic BF16 vectors; operator timing only",
        "cases": [],
        "samples": [],
        "calibration": {},
        "summaries": {},
        "errors": [],
    }
    save = lambda: write(output / "state.json", state)
    budget = BoundedExploration(
        contract.measurement.exploration, max_requests=128, max_seconds=1800
    )
    observer_stop = threading.Event()
    thermal = []

    def observe():
        while not observer_stop.wait(0.5):
            thermal.append(thermal_state())

    observer = threading.Thread(target=observe, daemon=True)
    save()
    observer.start()

    def check():
        verify_frozen_candidate(frozen)
        validate_receipt(
            receipt, runtime.model_path, runtime.model_revision, runtime.model_files
        )
        if protected != _protected_files(extra):
            raise RuntimeError("Protected sources changed")

    def boundary():
        check()
        budget.begin_request()
        before = environment()
        if not assess_environment(before, before, mode="exploration")["continue_pass"]:
            raise RuntimeError("Hard preflight power/thermal/observation failure")
        return before, len(thermal)

    def assessment(before, first):
        after = environment()
        result = assess_environment(
            before, after, mode="exploration", thermal_states=thermal[first:]
        )
        return after, result

    try:
        name = "frozen_operator_" + frozen.candidate_id
        spec = importlib.util.spec_from_file_location(
            name, frozen.directory / "implementation.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        index = json.loads(
            (runtime.model_path / "model.safetensors.index.json").read_text()
        )
        keys = [
            f"model.layers.{layer}.self_attn.{projection}_proj"
            for layer in (0, 12, 23)
            for projection in ("q", "k", "v", "o")
        ] + ["lm_head"]
        prepared = []
        rng = np.random.default_rng(390712)
        for key in keys:
            check()
            start = time.perf_counter()
            weight, identity = load_bf16(
                runtime.model_path, index, key + ".weight", mx, np
            )
            bias = None
            if key + ".bias" in index["weight_map"]:
                bias, bias_identity = load_bf16(
                    runtime.model_path, index, key + ".bias", mx, np
                )
                identity["bias"] = bias_identity
            encode_start = time.perf_counter()
            projection = module.prepare_projection(weight)
            preparation_ms = (time.perf_counter() - encode_start) * 1000
            n, k = weight.shape
            inputs = [
                mx.array(
                    rng.standard_normal((1, 1, k), dtype=np.float32) * np.float32(m),
                    dtype=mx.bfloat16,
                )
                for m in (1.0, 0.5, 2.0, 64.0)
            ]
            mx.eval(inputs)
            record = {
                "key": key,
                "tensor_identity": identity,
                "codec": projection.statistics,
                "prepare_ms": preparation_ms,
                "load_and_prepare_ms": (time.perf_counter() - start) * 1000,
            }
            state["cases"].append(record)
            save()
            prepared.append(
                {
                    "key": key,
                    "weight": weight,
                    "bias": bias,
                    "candidate": projection,
                    "inputs": inputs,
                }
            )
            print(
                "prepared",
                key,
                "codec ratio",
                projection.statistics["encoded_ratio"],
                flush=True,
            )
        state["resident_native_weight_bytes"] = sum(
            p["weight"].nbytes + (p["bias"].nbytes if p["bias"] is not None else 0)
            for p in prepared
        )
        state["resident_sidecar_bytes"] = sum(
            p["candidate"].statistics["allocated_sidecar_bytes"] for p in prepared
        )
        state["resident_allocator_active_bytes"] = mx.get_active_memory()
        save()

        def operation(case, x, lane):
            if lane == "candidate":
                return case["candidate"](x, case["bias"])
            return (
                x @ case["weight"].T
                if case["bias"] is None
                else mx.addmm(case["bias"], x, case["weight"].T)
            )

        state["status"] = "checkpoint_operator_correctness"
        save()
        for case in prepared:
            before, first = boundary()
            rows = []
            for number, x in enumerate(case["inputs"]):
                reference = operation(case, x, "baseline")
                actual = operation(case, x, "candidate")
                mx.eval(reference, actual)
                mx.synchronize()
                a = np.array(actual.view(mx.uint16))
                b = np.array(reference.view(mx.uint16))
                exact = bool(np.array_equal(a, b))
                finite = bool(mx.all(mx.isfinite(actual)).item())
                path = output / (case["key"] + "-" + str(number) + ".npz")
                np.savez_compressed(
                    path,
                    reference_words=b,
                    candidate_words=a,
                    input_words=np.array(x.view(mx.uint16)),
                )
                rows.append(
                    {
                        "input": number,
                        "passed": exact
                        and finite
                        and actual.dtype == reference.dtype == mx.bfloat16
                        and actual.shape == reference.shape,
                        "bit_exact": exact,
                        "finite": finite,
                        "arrays": str(path),
                        "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            after, flags = assessment(before, first)
            result = {
                "key": case["key"],
                "checks": rows,
                "environment_before": before,
                "environment_after": after,
                "environment_assessment": flags,
            }
            state.setdefault("correctness", []).append(result)
            save()
            print(
                "correctness", case["key"], all(r["passed"] for r in rows), flush=True
            )
            if not all(r["passed"] for r in rows):
                raise ValueError("Pinned-checkpoint projection correctness failed")
            if not flags["continue_pass"]:
                raise RuntimeError("Hard environment failure")

        def sample(lane, phase, pair):
            # Fixed rest avoids a long continuously heated operator loop. All
            # kernel compilation, encoding, readback checks are already outside.
            time.sleep(2)
            before, first = boundary()
            timings = []
            for case in prepared:
                mx.synchronize()
                start = time.perf_counter()
                calls = 0
                for _ in range(4):
                    for x in case["inputs"]:
                        mx.eval(operation(case, x, lane))
                        calls += 1
                mx.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if calls != 16:
                    raise ValueError("Operator-call count mismatch")
                timings.append(
                    {
                        "key": case["key"],
                        "calls": calls,
                        "elapsed_ms": elapsed,
                        "ms_per_call": elapsed / calls,
                    }
                )
            after, flags = assessment(before, first)
            row = {
                "lane": lane,
                "phase": phase,
                "pair": pair,
                "timings": timings,
                "environment_before": before,
                "environment_after": after,
                "environment_assessment": flags,
                "swap_activity": swap_observation(before, after, "paging-aware"),
                "confirmation_eligible": False,
            }
            state["samples"].append(row)
            save()
            print(phase, pair, lane, "flags", flags["flags"], flush=True)
            if not flags["continue_pass"]:
                raise RuntimeError("Hard environment failure")
            return row

        for phase in ("aa_warmup", "aa", "candidate_warmup", "candidate"):
            state["status"] = phase
            save()
            for pair in range(1, (6 if "warmup" in phase else 21)):
                lanes = (
                    ("A", "B") if phase.startswith("aa") else ("baseline", "candidate")
                )
                for lane in lanes if pair % 2 else tuple(reversed(lanes)):
                    sample(lane, phase, pair)
            if phase == "aa":
                rows = [r for r in state["samples"] if r["phase"] == "aa"]
                for index, key in enumerate(keys):
                    a = {
                        r["pair"]: r["timings"][index]["ms_per_call"]
                        for r in rows
                        if r["lane"] == "A"
                    }
                    b = {
                        r["pair"]: r["timings"][index]["ms_per_call"]
                        for r in rows
                        if r["lane"] == "B"
                    }
                    ratios = [a[p] / b[p] for p in range(1, 21)]
                    ci = bootstrap(ratios, 42 + index)
                    state["calibration"][key] = {
                        "ratios": ratios,
                        "median_ratio": st.median(ratios),
                        "bootstrap_95pct": ci,
                        "passed": ci[0] <= 1 <= ci[1] and ci[1] - ci[0] <= 0.04,
                        "swap_affected_rows": sum(
                            r["environment_assessment"]["swap_affected"] for r in rows
                        ),
                    }
                # Exploration preserves this calibration even if noisy, and
                # finishes its bounded pass. It is not fresh confirmation.
                save()
        rows = [r for r in state["samples"] if r["phase"] == "candidate"]
        for index, key in enumerate(keys):
            a = {
                r["pair"]: r["timings"][index]["ms_per_call"]
                for r in rows
                if r["lane"] == "baseline"
            }
            b = {
                r["pair"]: r["timings"][index]["ms_per_call"]
                for r in rows
                if r["lane"] == "candidate"
            }
            savings = [a[p] - b[p] for p in range(1, 21)]
            ratios = [a[p] / b[p] for p in range(1, 21)]
            state["summaries"][key] = {
                "median_saved_ms_per_call": st.median(savings),
                "all_pair_saved_ms": savings,
                "bootstrap_95pct_saved_ms": bootstrap(savings, 100 + index),
                "median_ratio": st.median(ratios),
                "bootstrap_95pct_ratio": bootstrap(ratios, 200 + index),
                "baseline_median_ms": st.median(a.values()),
                "candidate_median_ms": st.median(b.values()),
                "all_pairs_included": True,
                "confirmation_eligible": False,
                "swap_affected_rows": sum(
                    r["environment_assessment"]["swap_affected"] for r in rows
                ),
            }
        attention = [
            state["summaries"][key]["median_saved_ms_per_call"]
            for key in keys
            if key != "lm_head"
        ]
        state["conditional_model_planning_ms_per_token"] = (
            sum(attention) * 8
            + state["summaries"]["lm_head"]["median_saved_ms_per_call"]
        )
        state["planning_assumptions"] = (
            "Only if sampled layers/vectors represent all24layers and isolated savings add on the model critical path. Not measured full-model speedup; prototype sidecars inflate residency."
        )
        state["status"] = "completed_checkpoint_operator_exploration"
    except ExplorationComplete as error:
        state["status"] = "completed_at_exploration_bound"
        state["bound_reason"] = str(error)
    except BaseException:
        state["status"] = "failed"
        state["errors"].append(traceback.format_exc())
        raise
    finally:
        observer_stop.set()
        observer.join()
        state["protected_unchanged"] = protected == _protected_files(extra)
        try:
            verify_frozen_candidate(frozen)
            validate_receipt(
                receipt, runtime.model_path, runtime.model_revision, runtime.model_files
            )
            state["checkpoint_and_snapshot_unchanged"] = True
        except Exception:
            state["checkpoint_and_snapshot_unchanged"] = False
            state["errors"].append(traceback.format_exc())
        state["requests_started"] = budget.requests
        state["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
