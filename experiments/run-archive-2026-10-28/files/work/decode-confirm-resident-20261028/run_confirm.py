"""One changed residency protocol for an existing frozen vocabulary candidate.

Controller command files are a sequential one-off work queue, not an automation.
Candidate code, contract and runtime inputs remain frozen per operation.
"""

import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics as st
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
from evaluator.correctness import compare_outputs
from evaluator.native_baseline import (
    environment,
    fixed_inputs,
    materialize_parameters,
    request,
    swap_observation,
)
from evaluator.baseline_validation import aa_noise, thermal_state
from evaluator.measurement import assess_environment, PhaseMemoryObserver, ObservedModel
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


def interval(values, seed=42):
    rng = random.Random(seed)
    boot = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [boot[249], boot[9749]]


class NumericalFailure(RuntimeError):
    pass


class EnvironmentFailure(RuntimeError):
    pass


class ConfirmationStop(RuntimeError):
    pass


class Binding:
    """Trusted runtime-only placement. Originals are restored at every exit."""

    def __init__(self, model, adapter, mx):
        self.model = model
        self.adapter = adapter
        self.mx = mx
        self.linear_cls = type(model.lm_head)
        self.block_cls = type(model.model.layers[0])
        self.expert_cls = type(model.model.layers[0].mlp.experts)
        self.linear = self.linear_cls.__call__
        self.block = self.block_cls.__call__
        self.expert = self.expert_cls.__call__
        self.layer_ids = {id(l): i for i, l in enumerate(model.model.layers)}
        self.expert_ids = {
            id(l.mlp.experts): i for i, l in enumerate(model.model.layers)
        }
        self.enabled = False
        self.audit = None
        self.counts = {}

    @contextlib.contextmanager
    def lane(self, enabled=False, audit=None):
        self.enabled = enabled
        self.audit = audit
        self.counts = {
            "projection_specialized": 0,
            "projection_fallback": 0,
            "block_specialized": 0,
            "block_fallback": 0,
        }

        def linear(layer, x):
            if (
                layer is self.model.lm_head
                and enabled
                and self.adapter.kind in ("vocabulary", "combined")
            ):
                supported = x.ndim == 3 and x.shape[:2] == (1, 1)
                self.counts[
                    "projection_specialized" if supported else "projection_fallback"
                ] += 1
                return self.adapter.call_projection(self.linear, layer, x)
            return self.linear(layer, x)

        def block(layer, x, mask, cache=None):
            if enabled and self.adapter.kind in ("moe", "combined"):
                if x.ndim == 3 and x.shape[:2] == (1, 1):
                    self.counts["block_specialized"] += 1

                    def capture(ids):
                        if audit is not None:
                            audit.current.append((self.layer_ids[id(layer)], ids))

                    return self.adapter.call_block(
                        self.block, layer, x, mask, cache, capture
                    )
                self.counts["block_fallback"] += 1
            return self.block(layer, x, mask, cache)

        def expert(layer, x, ids, *args, **kwargs):
            if audit is not None:
                audit.current.append((self.expert_ids[id(layer)], ids))
            return self.expert(layer, x, ids, *args, **kwargs)

        self.linear_cls.__call__ = linear
        self.block_cls.__call__ = block
        if audit is not None:
            self.expert_cls.__call__ = expert
        try:
            yield
        finally:
            self.linear_cls.__call__ = self.linear
            self.block_cls.__call__ = self.block
            self.expert_cls.__call__ = self.expert
            self.enabled = False
            self.audit = None


class AuditModel:
    def __init__(self, model):
        self.original = model
        self.calls = []
        self.current = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, tokens, cache=None):
        self.calls.append(list(tokens.shape))
        self.current = []
        return self.original(tokens, cache=cache)


def main():
    output = WORK / "resident-state.json"
    if output.exists():
        raise ValueError(
            "Fresh resident worker only; never restart a live or completed run"
        )
    contract = load_contract(ROOT / "kernelmaxxing.yaml")
    session = load_authorization(WORK / "session.json", contract)
    plan = json.loads((WORK / "plan.json").read_text())
    runtime = native_runtime(Path(session["model_path"]), contract)
    receipt = json.loads(Path(session["checkpoint_cache"]).read_text())
    validate_receipt(
        receipt, runtime.model_path, runtime.model_revision, runtime.model_files
    )
    import mlx.core as mx
    import numpy as np
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten

    extra = [
        __file__,
        WORK / "session.json",
        WORK / "plan.json",
        WORK / "correctness-parent.json",
        gpt_oss.__file__,
        switch_layers.__file__,
        mx.__file__,
    ]
    protected = _protected_files(extra)
    state = {
        "status": "loading",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "contract_sha256": contract.sha256,
        "official": False,
        "verdict": None,
        "acceptance_enabled": False,
        "candidate_ids": list(plan["existing_candidate_ids"]),
        "commands": [],
        "protected_before": protected,
        "errors": [],
        "five_percent_improvement_verified": False,
    }
    save = lambda: write(output, state)
    save()
    binding = None

    def check(frozen=None):
        if time.time() - state["started_unix"] > 36000:
            raise TimeoutError("Ten-hour resident-session cap")
        validate_receipt(
            receipt, runtime.model_path, runtime.model_revision, runtime.model_files
        )
        if protected != _protected_files(extra):
            raise RuntimeError("Protected source changed")
        if frozen is not None:
            verify_frozen_candidate(frozen)

    try:
        with mx.stream(mx.cpu):
            model, _ = load(str(runtime.model_path), lazy=True)

            def progress(done, total, name, size):
                if done % 25 == 0 or done == total:
                    state["load_progress"] = {"done": done, "total": total}
                    save()
                    print("load", done, total, flush=True)

            materialize_parameters(tree_flatten(model.parameters()), mx, progress)
        inputs = fixed_inputs([512, 2048, 4096], model.args.vocab_size, 42)
        current_id = None
        adapter = None
        frozen = None

        def run_command(command, number):
            nonlocal binding, current_id, adapter, frozen
            mode = command["mode"]
            if mode != "confirm" or state["commands"]:
                raise ValueError("One confirmation operation only; no exploration, reroll or new candidate")
            frozen_info = json.loads(Path(command["frozen_info"]).read_text())
            c = validate_candidate(
                frozen_info["directory"], Path(frozen_info["directory"]).parent
            )
            selected = FrozenCandidate(
                **vars(c),
                sha256=frozen_info["sha256"],
                file_sha256=frozen_info["file_sha256"],
            )
            check(selected)
            if selected.candidate_id != plan["confirmation_candidate_id"]:
                raise ValueError("Only the already-frozen vocabulary placement is authorized")
            if selected.sha256 != plan["confirmation_candidate_sha256"]:
                raise ValueError("Frozen vocabulary snapshot differs from the declared plan")
            if selected.candidate_id not in state["candidate_ids"]:
                if len(state["candidate_ids"]) >= plan["candidate_cap"]:
                    raise ValueError("Three-candidate cap reached")
                state["candidate_ids"].append(selected.candidate_id)
                save()
            if current_id != selected.candidate_id:
                binding = None
                adapter = None
                gc.collect()
                mx.clear_cache()
                module_name = "sprint_" + selected.candidate_id
                spec = importlib.util.spec_from_file_location(
                    module_name, selected.directory / "implementation.py"
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                prepare_start = time.perf_counter()
                adapter = module.prepare(model)
                mx.synchronize()
                current_id = selected.candidate_id
                frozen = selected
                binding = Binding(model, adapter, mx)
                preparation = {
                    "wall_ms": (time.perf_counter() - prepare_start) * 1000,
                    "statistics": getattr(adapter, "statistics", {}),
                    "resident_active_bytes": mx.get_active_memory(),
                    "resident_native_model_bytes": sum(
                        t.nbytes for _, t in tree_flatten(model.parameters())
                    ),
                }
            else:
                if selected.sha256 != frozen.sha256:
                    raise ValueError("Candidate identity changed")
                preparation = {
                    "reused_same_frozen_adapter": True,
                    "statistics": getattr(adapter, "statistics", {}),
                }
            directory = WORK / f"command-{number:03d}-{mode}-{current_id}"
            directory.mkdir()
            cs = {
                "mode": mode,
                "status": "starting",
                "candidate_id": current_id,
                "snapshot_sha256": frozen.sha256,
                "contract_sha256": contract.sha256,
                "command": command,
                "preparation": preparation,
                "started_unix": time.time(),
                "correctness": [],
                "requests": [],
                "phase_memory_requests": [],
                "resident_primers": [],
                "calibrations": {},
                "summaries": {},
                "errors": [],
                "five_percent_improvement_verified": False,
            }

            def store():
                write(directory / "state.json", cs)

            store()
            state["status"] = f"{mode}:{current_id}"
            state["active_directory"] = str(directory)
            save()
            command_start = time.monotonic()
            request_count = 0
            max_requests = 800
            reference_tokens = {}

            def operation_check():
                check(frozen)
                if time.monotonic() - command_start > (
                    28800
                ):
                    raise TimeoutError("Command wall-time cap")

            def trace(length, steps, enabled, forced=None):
                operation_check()
                audit = AuditModel(model)
                cache = make_prompt_cache(model)
                prompt = mx.array([inputs[str(length)]], dtype=mx.int32)
                mx.eval(prompt)
                logits_list = []
                routes = []
                tokens = []
                dtypes = []
                with binding.lane(enabled, audit):
                    audit(prompt[:, :-1], cache=cache)
                    mx.eval([c.state for c in cache])
                    mx.clear_cache()
                    remaining = prompt[:, -1:]
                    for step in range(steps):
                        logits = audit(remaining, cache=cache)[:, -1, :]
                        mx.eval(logits)
                        captured = sorted(audit.current, key=lambda item: item[0])
                        if [i for i, _ in captured] != list(range(24)):
                            raise NumericalFailure("Incomplete route audit")
                        routes.append(
                            np.stack([np.array(v).reshape(4) for _, v in captured])
                        )
                        logits_list.append(np.array(logits.astype(mx.float32)))
                        dtypes.append(str(logits.dtype))
                        token = mx.argmax(logits, axis=-1)
                        mx.eval(token)
                        tokens.append(token.item())
                        next_token = tokens[-1] if forced is None else forced[step]
                        remaining = mx.array([[next_token]], dtype=mx.int32)
                expected = [[1, length - 1]] + [[1, 1]] * steps
                if audit.calls != expected:
                    raise NumericalFailure("Unexpected trace forward count")
                cache_values = [
                    np.array(v.astype(mx.float32)) for c in cache for v in c.state
                ]
                metadata = [str(c.meta_state) for c in cache]
                cache_types = sorted({str(v.dtype) for c in cache for v in c.state})
                return {
                    "logits": np.concatenate(logits_list),
                    "routes": np.stack(routes),
                    "tokens": tokens,
                    "cache": cache_values,
                    "meta": metadata,
                    "cache_dtypes": cache_types,
                    "logit_dtypes": sorted(set(dtypes)),
                }

            def correctness(length, steps):
                ref = trace(length, steps, False)
                cand = trace(length, steps, True, ref["tokens"])
                comparison = compare_outputs(
                    ref["logits"],
                    cand["logits"],
                    case_name="full_model_logits",
                    policy=contract.correctness,
                )
                cache_ok = (
                    len(ref["cache"]) == len(cand["cache"])
                    and ref["meta"] == cand["meta"]
                )
                cache_errors = []
                for a, b in zip(ref["cache"], cand["cache"]):
                    if a.shape != b.shape or not np.all(np.isfinite(b)):
                        cache_ok = False
                        cache_errors.append(None)
                        continue
                    error = float(np.max(np.abs(a - b)))
                    cache_errors.append(error)
                    if not np.all(
                        np.abs(a - b)
                        <= contract.correctness.atol
                        + contract.correctness.rtol * np.abs(a)
                    ):
                        cache_ok = False
                path = directory / f"correctness-{length}-{steps}.npz"
                payload = {
                    "reference_logits": ref["logits"],
                    "candidate_logits": cand["logits"],
                    "reference_routes": ref["routes"],
                    "candidate_routes": cand["routes"],
                    "reference_tokens": np.array(ref["tokens"]),
                    "candidate_tokens": np.array(cand["tokens"]),
                }
                np.savez_compressed(path, **payload)
                row = {
                    "length": length,
                    "outputs": steps,
                    "numerical_passed": comparison.passed,
                    "detail": comparison.detail,
                    "all_logits_bit_exact": bool(
                        np.array_equal(ref["logits"], cand["logits"])
                    ),
                    "route_slot_differences": int(
                        np.count_nonzero(ref["routes"] != cand["routes"])
                    ),
                    "route_set_differences": int(
                        np.count_nonzero(
                            np.sort(ref["routes"], axis=-1)
                            != np.sort(cand["routes"], axis=-1)
                        )
                    ),
                    "token_differences": sum(
                        a != b for a, b in zip(ref["tokens"], cand["tokens"])
                    ),
                    "cache_within_tolerance": cache_ok,
                    "cache_max_abs_errors": cache_errors,
                    "cache_metadata_equal": ref["meta"] == cand["meta"],
                    "reference_cache_dtypes": ref["cache_dtypes"],
                    "candidate_cache_dtypes": cand["cache_dtypes"],
                    "arrays": str(path),
                    "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                row["passed"] = (
                    comparison.passed
                    and cache_ok
                    and ref["cache_dtypes"]
                    == cand["cache_dtypes"]
                    == ["mlx.core.bfloat16"]
                    and ref["logit_dtypes"]
                    == cand["logit_dtypes"]
                    == ["mlx.core.bfloat16"]
                )
                cs["correctness"].append(row)
                store()
                print(
                    "correctness",
                    mode,
                    current_id,
                    length,
                    steps,
                    row["passed"],
                    flush=True,
                )
                del ref, cand, payload
                gc.collect()
                mx.clear_cache()
                if not row["passed"]:
                    raise NumericalFailure(
                        "Full-model numerical/shape/dtype/cache comparison failed"
                    )
                return row

            def collect(length, lane, phase, pair, instrument=False):
                nonlocal request_count
                operation_check()
                if request_count + 2 > max_requests:
                    raise TimeoutError("Command request bound reached")
                time.sleep(
                    plan["screen" if mode == "screen" else "confirmation"][
                        "before_each_request_idle_seconds"
                    ]
                )
                # Unconditional same-lane128-output hydration after the fixed
                # cooldown. Keep every primer and its paging transitions;
                # never choose/retry primers based on a favorable outcome.
                primer_before = environment()
                primer_preflight = assess_environment(primer_before, primer_before, mode="exploration")
                if not primer_preflight["continue_pass"]:
                    raise EnvironmentFailure("Primer hard preflight: " + str(primer_preflight))
                primer_seen = [primer_before["thermal_state"]]
                primer_stop = threading.Event()
                def observe_primer():
                    while not primer_stop.wait(.5):
                        primer_seen.append(thermal_state())
                primer_observer = threading.Thread(target=observe_primer, daemon=True)
                primer_audit = AuditModel(model)
                primer_observer.start()
                try:
                    with binding.lane(lane == "candidate"):
                        request_count += 1
                        primer = request(primer_audit, inputs[str(length)], 4096, mx, make_prompt_cache)
                        primer_counts = dict(binding.counts)
                finally:
                    primer_stop.set(); primer_observer.join()
                primer_after = environment()
                primer_flags = assess_environment(primer_before, primer_after, mode="exploration", thermal_states=primer_seen)
                primer.update(prompt_tokens=length, lane=lane, phase=phase, trial=pair, untimed=True,
                    environment_before=primer_before, environment_after=primer_after,
                    environment_assessment=primer_flags, thermal_states=primer_seen,
                    swap_activity=swap_observation(primer_before, primer_after, "paging-aware"),
                    actual_forward_calls=len(primer_audit.calls), placement_counts=primer_counts,
                    tokens_match_frozen_reference=primer['output_token_ids']==reference_tokens[length])
                cs['resident_primers'].append(primer)
                # Do not serialize the growing state between hydration and the
                # measured fresh-cache request. Its normal environment boundary
                # follows immediately; both rows persist together or in finally.
                if primer_audit.calls != [[1,length-1]]+[[1,1]]*128 or not primer['tokens_match_frozen_reference']:
                    raise NumericalFailure('Primer call-count or deterministic token mismatch')
                if lane=='candidate' and primer_counts['projection_specialized'] != 128:
                    raise NumericalFailure('Primer vocabulary placement count mismatch')
                if not primer_flags['continue_pass']:
                    raise EnvironmentFailure('Primer power/thermal/observation failure: '+str(primer_flags))
                before = environment()
                first = assess_environment(
                    before,
                    before,
                    mode="exploration" if mode == "screen" else "confirmation",
                )
                if not first["continue_pass"]:
                    raise EnvironmentFailure("Preflight environment: " + str(first))
                seen = [before["thermal_state"]]
                stop = threading.Event()

                def observe():
                    while not stop.wait(0.5):
                        seen.append(thermal_state())

                observer = threading.Thread(target=observe, daemon=True)
                audit = AuditModel(model)
                memory = PhaseMemoryObserver(length, mx) if instrument else None
                proxy = ObservedModel(audit, memory) if memory else audit
                if memory:
                    memory.start()
                observer.start()
                try:
                    with binding.lane(lane == "candidate"):
                        request_count += 1
                        result = request(
                            proxy, inputs[str(length)], 4096, mx, make_prompt_cache
                        )
                        counts = dict(binding.counts)
                finally:
                    stop.set()
                    observer.join()
                    memory_result = memory.finish() if memory else None
                if audit.calls != [[1, length - 1]] + [[1, 1]] * 128:
                    raise NumericalFailure(
                        "Extra/missing model forward in timed driver"
                    )
                if lane == "candidate":
                    if (
                        adapter.kind in ("vocabulary", "combined")
                        and counts["projection_specialized"] != 128
                    ):
                        raise NumericalFailure(
                            "Vocabulary placement call count mismatch"
                        )
                    if (
                        adapter.kind in ("moe", "combined")
                        and counts["block_specialized"] != 128 * 24
                    ):
                        raise NumericalFailure("MoE placement call count mismatch")
                after = environment()
                flags = assess_environment(
                    before,
                    after,
                    mode="exploration" if mode == "screen" else "confirmation",
                    thermal_states=seen,
                )
                row = {
                    **result,
                    "prompt_tokens": length,
                    "lane": lane,
                    "phase": phase,
                    "trial": pair,
                    "environment_before": before,
                    "environment_after": after,
                    "environment_assessment": flags,
                    "thermal_states": seen,
                    "invalid_reasons": flags["flags"],
                    "actual_forward_calls": len(audit.calls),
                    "placement_counts": counts,
                    "swap_activity": swap_observation(before, after, "paging-aware"),
                    "instrumented": instrument,
                    "phase_memory": memory_result,
                    "tokens_match_frozen_reference": result['output_token_ids']==reference_tokens[length],
                    "primer_index": len(cs['resident_primers'])-1,
                }
                cs["phase_memory_requests" if instrument else "requests"].append(row)
                store()
                print(
                    mode,
                    length,
                    phase,
                    pair,
                    lane,
                    f"{result['metrics']['decode_tokens_per_second']:.3f}tok/s",
                    flags["flags"],
                    flush=True,
                )
                if not flags["continue_pass"]:
                    raise EnvironmentFailure("Environment flags: " + str(flags))
                if not row['tokens_match_frozen_reference']:
                    raise NumericalFailure('Timed request changed deterministic reference tokens')
                return row

            def summarize(length, rows, key):
                a = {r["trial"]: r for r in rows if r["lane"] == "baseline"}
                b = {r["trial"]: r for r in rows if r["lane"] == "candidate"}
                if set(a) != set(b) or len(a) < 3:
                    raise ValueError("Incomplete paired comparison")
                ratios = [
                    b[p]["metrics"]["decode_tokens_per_second"]
                    / a[p]["metrics"]["decode_tokens_per_second"]
                    for p in sorted(a)
                ]
                saves = [
                    1000 / a[p]["metrics"]["decode_tokens_per_second"]
                    - 1000 / b[p]["metrics"]["decode_tokens_per_second"]
                    for p in sorted(a)
                ]
                summary = {
                    "length": length,
                    "pairs": len(a),
                    "paired_ratios": ratios,
                    "median_ratio": st.median(ratios),
                    "bootstrap_95pct_ratio": interval(ratios, length),
                    "median_saved_ms_per_token": st.median(saves),
                    "all_pair_saved_ms_per_token": saves,
                    "swap_affected_rows": sum(
                        r["environment_assessment"]["swap_affected"] for r in rows
                    ),
                    "mode": mode,
                    "metrics": {},
                }
                for metric in a[next(iter(a))]["metrics"]:
                    summary["metrics"][metric] = {}
                    for lane, values in [("baseline", a), ("candidate", b)]:
                        v = [r["metrics"][metric] for r in values.values()]
                        m = st.median(v)
                        summary["metrics"][metric][lane] = {
                            "median": m,
                            "p90": float(np.percentile(v, 90)),
                            "mad": st.median(abs(x - m) for x in v),
                        }
                cs["summaries"][key] = summary
                store()
                return summary

            try:
                if mode == "screen":
                    correctness(512, plan["screen"]["short_correctness_outputs"])
                    for lane in ("baseline", "candidate"):
                        collect(512, lane, "phase_memory", 0, True)
                    for lane in ("baseline", "candidate"):
                        collect(512, lane, "warmup", 0)
                    rows = []
                    for pair in range(1, 4):
                        for lane in (
                            ("baseline", "candidate")
                            if pair % 2
                            else ("candidate", "baseline")
                        ):
                            rows.append(collect(512, lane, "screen", pair))
                    summary = summarize(512, rows, "512-screen")
                    cs["promising_for_confirmation"] = (
                        summary["median_ratio"]
                        >= plan["screen"]["promotion_signal_ratio"]
                    )
                    cs["status"] = "completed_exploratory_screen"
                else:
                    # Reuse completed mathematical evidence for exactly the
                    # same frozen code, checkpoint, precision and model sources.
                    # The only intervention is untimed residency preparation.
                    parent_path = WORK/'correctness-parent.json'
                    parent = json.loads(parent_path.read_text())
                    old = parent['failed_confirmation_005']
                    for path, digest in parent['parent_resident_state']['protected_before'].items():
                        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                            raise RuntimeError('Numerical evidence source changed: '+path)
                    old_path = Path(old['state_path'])
                    if hashlib.sha256(old_path.read_bytes()).hexdigest() != old['state_sha256']:
                        raise RuntimeError('Numerical evidence artifact changed')
                    old_state = json.loads(old_path.read_text())
                    if (old_state['snapshot_sha256'] != frozen.sha256 or old_state['contract_sha256'] != contract.sha256
                        or parent['checkpoint_identity']['receipt_sha256'] != hashlib.sha256(Path(session['checkpoint_cache']).read_bytes()).hexdigest()
                        or any(parent['checkpoint_identity'][key] != receipt[key] for key in
                            ('byte_verification','file_identity','model_files','model_path','model_revision'))
                        or not all(old_state.get(k) is True for k in ('protected_unchanged','checkpoint_and_snapshot_unchanged','original_methods_restored'))):
                        raise RuntimeError('Numerical evidence identity/guards mismatch')
                    old_rows = {row['length']:row for row in old_state['correctness']}
                    if set(old_rows) != {512,2048,4096}:
                        raise RuntimeError('Full numerical evidence coverage missing')
                    for length,row in old_rows.items():
                        if (row['outputs']!=128 or not row['passed'] or not row['all_logits_bit_exact']
                            or row['token_differences'] or row['route_set_differences'] or not row['cache_within_tolerance']):
                            raise NumericalFailure('Completed full-length numerical pass required')
                        path = Path(row['arrays'])
                        if hashlib.sha256(path.read_bytes()).hexdigest() != row['arrays_sha256']:
                            raise RuntimeError('Numerical arrays changed')
                        with np.load(path,allow_pickle=False) as arrays:
                            tokens=arrays['reference_tokens'].tolist()
                            if len(tokens)!=128 or tokens!=arrays['candidate_tokens'].tolist():
                                raise NumericalFailure('Preserved reference token evidence inconsistent')
                            reference_tokens[length]=tokens
                        cs['correctness'].append({**row,'reused_unchanged_evidence':True,'source_state':str(old_path)})
                    cs['correctness_reuse']={'parent_manifest_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                        'source_state_sha256':old['state_sha256'],'basis':'Same frozen candidate/checkpoint/model sources and precision; only untimed residency preparation changes. No old timing samples reused.'}
                    store()
                    for length in (512, 2048, 4096):
                        aa = []
                        for warm in range(1, 6):
                            for lane in ("A", "B"):
                                collect(length, lane, "aa_warmup", warm)
                        for pair in range(1, 21):
                            for lane in ("A", "B") if pair % 2 else ("B", "A"):
                                aa.append(collect(length, lane, "aa", pair))
                        cs["calibrations"][str(length)] = aa_noise(aa, [length])[
                            str(length)
                        ]
                        store()
                        if not cs["calibrations"][str(length)]["passed"]:
                            raise ConfirmationStop("stopped_confirmation_calibration")
                        blocks = 2 if length == 512 else 1
                        for block in range(1, blocks + 1):
                            for warm in range(1, 6):
                                for lane in ("baseline", "candidate"):
                                    collect(length, lane, f"warmup-{block}", warm)
                            rows = []
                            for pair in range(1, 21):
                                for lane in (
                                    ("baseline", "candidate")
                                    if pair % 2
                                    else ("candidate", "baseline")
                                ):
                                    rows.append(
                                        collect(length, lane, f"confirm-{block}", pair)
                                    )
                            summary = summarize(length, rows, f"{length}-block-{block}")
                            noise = max(
                                abs(x - 1)
                                for x in cs["calibrations"][str(length)][
                                    "bootstrap_95pct_ci"
                                ]
                            )
                            summary["five_percent_verified"] = summary[
                                "median_ratio"
                            ] >= 1.05 and summary["bootstrap_95pct_ratio"][0] > max(
                                1.05, 1 + noise
                            )
                            store()
                            if not summary["five_percent_verified"]:
                                raise ConfirmationStop(
                                    "stopped_confirmation_target_not_demonstrated"
                                )
                    cs["five_percent_improvement_verified"] = True
                    cs["status"] = "completed_repeatable_five_percent_confirmation"
            except ConfirmationStop as error:
                cs["status"] = str(error)
            except (NumericalFailure, EnvironmentFailure) as error:
                cs["status"] = "stopped_" + type(error).__name__
                cs["errors"].append(traceback.format_exc())
            except BaseException:
                cs["status"] = "failed"
                cs["errors"].append(traceback.format_exc())
                raise
            finally:
                cs["requests_started"] = request_count
                cs["protected_unchanged"] = protected == _protected_files(extra)
                try:
                    check(frozen)
                    cs["checkpoint_and_snapshot_unchanged"] = True
                except Exception:
                    cs["checkpoint_and_snapshot_unchanged"] = False
                    cs["errors"].append(traceback.format_exc())
                cs["original_methods_restored"] = (
                    binding.linear_cls.__call__ is binding.linear
                    and binding.block_cls.__call__ is binding.block
                    and binding.expert_cls.__call__ is binding.expert
                )
                cs["finished_unix"] = time.time()
                store()
                if not cs["original_methods_restored"]:
                    raise RuntimeError("Runtime binding restoration failed")
                if (
                    not cs["protected_unchanged"]
                    or not cs["checkpoint_and_snapshot_unchanged"]
                ):
                    cs["five_percent_improvement_verified"] = False
                    cs["status"] = "failed_integrity"
                    store()
                    raise RuntimeError("Final source/checkpoint/snapshot guard failed")
            if cs["five_percent_improvement_verified"]:
                state["five_percent_improvement_verified"] = True
                save()
            state["commands"].append(
                {
                    "number": number,
                    "mode": mode,
                    "candidate_id": current_id,
                    "status": cs["status"],
                    "directory": str(directory),
                    "five_percent_improvement_verified": cs[
                        "five_percent_improvement_verified"
                    ],
                }
            )
            save()

        with wired_limit(model):
            number = 1
            while True:
                state["status"] = "waiting_command"
                state["next_command"] = number
                save()
                command_path = WORK / "commands" / f"{number:03d}.json"
                deadline = time.monotonic() + 1800
                while not command_path.exists():
                    check()
                    if time.monotonic() > deadline:
                        state["status"] = "resident_wait_expired"
                        save()
                        return
                    time.sleep(2)
                raw = command_path.read_bytes()
                command = json.loads(raw)
                state["active_command_sha256"] = hashlib.sha256(raw).hexdigest()
                save()
                if command["mode"] == "finish":
                    state["status"] = "finished_by_controller"
                    save()
                    return
                run_command(command, number)
                number += 1
    except BaseException:
        state["status"] = "failed"
        state["errors"].append(traceback.format_exc())
        raise
    finally:
        state["protected_unchanged"] = protected == _protected_files(extra)
        try:
            validate_receipt(
                receipt, runtime.model_path, runtime.model_revision, runtime.model_files
            )
            state["checkpoint_unchanged"] = True
        except Exception:
            state["checkpoint_unchanged"] = False
            state["errors"].append(traceback.format_exc())
        state["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
