"""Controller-owned native synthetic checks for the frozen exact-layout reader.

This is a correctness fixture, not checkpoint/full-model validation or timing.
Run only after the complete-model exploration worker has exited.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent / "kernelmax-first-loop"
sys.path.insert(0, str(ROOT))
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.native_baseline import environment
from evaluator.baseline_validation import thermal_state
from evaluator.measurement import assess_environment, BoundedExploration
from orchestrator.candidate_manager import (
    FrozenCandidate,
    validate_candidate,
    verify_frozen_candidate,
)
from scripts.loop_control import load_authorization


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    process = subprocess.run(
        ["ps", "-p", "81514", "-o", "lstart=,command="], capture_output=True, text=True
    )
    if (
        process.returncode == 0
        and "decode-exploration-v3-20261027/profile.py" in process.stdout
    ):
        raise RuntimeError(
            "Whole-model exploration still live; no overlapping GPU fixture"
        )
    contract = load_contract(ROOT / "kernelmaxxing.yaml")
    load_authorization(WORK / "session.json", contract)
    info = json.loads((WORK / "layout-frozen.json").read_text())
    candidate = validate_candidate(info["directory"], Path(info["directory"]).parent)
    frozen = FrozenCandidate(
        **vars(candidate), sha256=info["sha256"], file_sha256=info["file_sha256"]
    )
    verify_frozen_candidate(frozen)
    import mlx.core as mx
    import numpy as np

    extra = [__file__, WORK / "layout-frozen.json", WORK / "session.json", mx.__file__]
    protected = _protected_files(extra)
    state = {
        "status": "starting",
        "mode": "exploration",
        "scope": "native synthetic word-decoder and projection checks only",
        "candidate_id": frozen.candidate_id,
        "snapshot_sha256": frozen.sha256,
        "contract_sha256": contract.sha256,
        "pid": os.getpid(),
        "official": False,
        "verdict": None,
        "speedup_claim_eligible": False,
        "records": [],
        "errors": [],
        "protected_before": protected,
        "started_unix": time.time(),
    }
    save = lambda: write(output / "state.json", state)
    budget = BoundedExploration(
        contract.measurement.exploration, max_requests=32, max_seconds=600
    )
    observed = []
    stop = threading.Event()

    def observe():
        while not stop.wait(0.5):
            observed.append(thermal_state())

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    save()

    def begin(label):
        budget.begin_request()
        verify_frozen_candidate(frozen)
        before = environment()
        assessment = assess_environment(before, before, mode="exploration")
        if not assessment["continue_pass"]:
            raise RuntimeError("Hard preflight failure: " + str(assessment))
        state["status"] = label
        save()
        return before, len(observed)

    def retain(row, before, index):
        after = environment()
        row["environment_assessment"] = assess_environment(
            before, after, mode="exploration", thermal_states=observed[index:]
        )
        row["environment_before"] = before
        row["environment_after"] = after
        state["records"].append(row)
        save()
        print(
            row["case"],
            row.get("passed"),
            row["environment_assessment"]["flags"],
            flush=True,
        )
        if not row["environment_assessment"]["continue_pass"]:
            raise RuntimeError("Hard environment failure")
        if not row["passed"]:
            raise ValueError("Native exact-layout fixture failed: " + row["case"])

    try:
        name = "frozen_lossless_dense_" + frozen.candidate_id
        spec = importlib.util.spec_from_file_location(
            name, frozen.directory / "implementation.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        decoder = mx.fast.metal_kernel(
            name="fixture_lossless_words_" + frozen.candidate_id,
            input_names=["payload", "headers", "escapes"],
            output_names=["decoded"],
            header=module._HEADER,
            source="""
                const uint start = thread_position_in_grid.x * 4;
                if (start >= WORDS) return;
                float4 values = km_load_bf16_4(payload, headers, escapes, start / 64, start & 63u);
                #pragma unroll
                for (uint q=0; q<4; ++q) decoded[start+q] = ushort(as_type<uint>(values[q]) >> 16);
            """,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
        rng = np.random.default_rng(390712)
        all_words = np.arange(65536, dtype=np.uint16)
        for label, words in (
            ("all_words_ordered", all_words),
            ("all_words_shuffled", rng.permutation(all_words)),
        ):
            before, index = begin(label)
            weight = mx.array(words.reshape(512, 128), dtype=mx.uint16).view(
                mx.bfloat16
            )
            mx.eval(weight)
            prepared = module.prepare_projection(weight)
            actual = decoder(
                inputs=[prepared.payload, prepared.headers, prepared.escapes],
                template=[("WORDS", 65536)],
                grid=(16384, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(65536,)],
                output_dtypes=[mx.uint16],
            )[0]
            mx.eval(actual)
            mx.synchronize()
            values = np.array(actual)
            row = {
                "case": label,
                "words": 65536,
                "passed": bool(np.array_equal(values, words)),
                "statistics": prepared.statistics,
                "comparison": "original uint16 words, including zero signs and NaN payload bits",
            }
            np.savez_compressed(
                output / (label + ".npz"), reference_words=words, actual_words=values
            )
            retain(row, before, index)
            del weight, prepared, actual, values
            mx.clear_cache()
        shapes = (
            (128, 128),
            (512, 192),
            (512, 2880),
            (2880, 4096),
            (4096, 2880),
            (201088, 128),
        )
        for shape_index, (n, k) in enumerate(shapes):
            raw = rng.standard_normal((n, k), dtype=np.float32)
            raw *= np.float32(0.02)
            weight = mx.array(raw, dtype=mx.bfloat16)
            mx.eval(weight)
            del raw
            # Force finite raw escapes as well as compressed normal blocks.
            words = np.array(weight.view(mx.uint16))
            words[::17, 0] = 0
            words[1::31, 64] = 0x8000
            weight = mx.array(words, dtype=mx.uint16).view(mx.bfloat16)
            mx.eval(weight)
            weight_hash = hashlib.sha256(words.tobytes()).hexdigest()
            del words
            prepared = module.prepare_projection(weight)
            for magnitude in (1.0, 64.0):
                x = mx.array(
                    rng.standard_normal((1, 1, k), dtype=np.float32)
                    * np.float32(magnitude),
                    dtype=mx.bfloat16,
                )
                for has_bias in (False, True):
                    label = f"n{n}-k{k}-m{int(magnitude)}-bias{int(has_bias)}"
                    before, index = begin(label)
                    bias = (
                        mx.array(
                            rng.standard_normal(n, dtype=np.float32), dtype=mx.bfloat16
                        )
                        if has_bias
                        else None
                    )
                    reference = (
                        x @ weight.T if bias is None else mx.addmm(bias, x, weight.T)
                    )
                    actual = prepared(x, bias)
                    mx.eval(reference, actual)
                    mx.synchronize()
                    reference_words = np.array(reference.view(mx.uint16))
                    actual_words = np.array(actual.view(mx.uint16))
                    finite = bool(mx.all(mx.isfinite(actual)).item())
                    exact = bool(np.array_equal(reference_words, actual_words))
                    error = float(
                        mx.max(
                            mx.abs(
                                reference.astype(mx.float32) - actual.astype(mx.float32)
                            )
                        ).item()
                    )
                    payload = {
                        "reference_words": reference_words,
                        "actual_words": actual_words,
                        "input_words": np.array(x.view(mx.uint16)),
                    }
                    if bias is not None:
                        payload["bias_words"] = np.array(bias.view(mx.uint16))
                    np.savez_compressed(output / (label + ".npz"), **payload)
                    row = {
                        "case": label,
                        "shape": [n, k],
                        "magnitude": magnitude,
                        "bias": has_bias,
                        "passed": exact
                        and finite
                        and actual.shape == reference.shape
                        and actual.dtype == reference.dtype == mx.bfloat16,
                        "all_output_words_bit_exact": exact,
                        "finite": finite,
                        "max_abs_error": error,
                        "weight_words_sha256": weight_hash,
                        "statistics": prepared.statistics,
                    }
                    retain(row, before, index)
                    del actual, reference, bias, payload, reference_words, actual_words
            del weight, prepared, x
            mx.clear_cache()
        if len(state["records"]) != 26:
            raise ValueError("Missing native fixture records")
        state["status"] = "passed_native_synthetic_fixture"
    except BaseException:
        state["status"] = "failed"
        state["errors"].append(traceback.format_exc())
        raise
    finally:
        stop.set()
        observer.join()
        state["protected_unchanged"] = protected == _protected_files(extra)
        try:
            verify_frozen_candidate(frozen)
            state["snapshot_unchanged"] = True
        except Exception:
            state["snapshot_unchanged"] = False
            state["errors"].append(traceback.format_exc())
        state["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
