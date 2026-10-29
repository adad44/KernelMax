"""Collect a baseline-only calibration attempt; never issue candidate verdicts.

The unchanged MLX-LM model is driven synchronously (no speculative/lookahead
token). This is not the throughput reported by mlx_lm.stream_generate. Preserve
the driver hash when comparing candidates; reviewer sign-off is still required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import struct
import subprocess
import time
import uuid
from pathlib import Path
import threading

from evaluator.schemas import load_contract
from evaluator.baseline_validation import thermal_state, driver_checks, expert_checks, aa_noise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_inputs(lengths, vocab_size, seed):
    rng = random.Random(seed)
    return {str(n): [rng.randrange(vocab_size) for _ in range(n)] for n in lengths}


def materialize_parameters(parameters, mx, progress):
    """Bound load-time evaluations; do not change arrays or inference execution.

    Each native tensor (including the upstream sanitize layout operations) is
    evaluated once, then synchronized before advancing to the next tensor.
    """
    parameters = list(parameters)
    for index, (name, tensor) in enumerate(parameters):
        progress(index, len(parameters), name, tensor.nbytes)
        mx.eval(tensor)
        mx.synchronize()
    progress(len(parameters), len(parameters), None, 0)


def metrics(prompt_length, start, prefill_end, tokens_at, peak):
    intervals = [b - a for a, b in zip(tokens_at, tokens_at[1:])]
    if len(intervals) != 127 or any(x <= 0 for x in intervals):
        raise ValueError("A request must retain all 127 positive decode intervals")
    result = {
        "decode_tokens_per_second": 127 / (tokens_at[-1] - tokens_at[0]),
        "time_to_first_token_ms": (tokens_at[0] - start) * 1000,
        "prefill_tokens_per_second": prompt_length / (prefill_end - start),
        "inter_token_latency_ms": statistics.median(intervals) * 1000,
        "peak_memory_bytes": peak,
        "full_model_wall_time_ms": (tokens_at[-1] - start) * 1000,
    }
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Invalid metric")
    return result, [x * 1000 for x in intervals]


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def environment():
    swap = command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used\s*=\s*([\d.]+)M", swap)
    if not match:
        raise RuntimeError("Cannot observe swap usage")
    vm = command("vm_stat")
    result = {
        "observed_at": time.time(),
        "power": command("pmset", "-g", "batt"),
        "thermal": command("pmset", "-g", "therm"),
        "power_settings": command("pmset", "-g", "custom"),
        "swap_used_mib": float(match.group(1)),
        "swap_raw": swap,
        "vm_stat": vm,
        "thermal_state": thermal_state(),
        "process_cpu_snapshot": command("ps", "-Ao", "pid,pcpu,comm", "-r")[:8000],
    }
    for label in ('Swapins', 'Swapouts'):
        counter = re.search(rf'^{label}:\s*(\d+)\.', vm, re.MULTILINE)
        if counter:
            result[label.lower()] = int(counter.group(1))
    return result


def environment_issues(before, after):
    issues = []
    if "'AC Power'" not in before["power"] or "'AC Power'" not in after["power"]:
        issues.append("AC power unavailable")
    if after["swap_used_mib"] > before["swap_used_mib"]:
        issues.append("swap grew during the request")
    if before["thermal"] != after["thermal"]:
        issues.append("thermal/performance observation changed")
    if before["power_settings"] != after["power_settings"]:
        issues.append("power settings changed")
    if 'thermal_state' in before and 'thermal_state' in after:
        if before['thermal_state'] != 0 or after['thermal_state'] != 0:
            issues.append('native thermal state is not nominal')
    for counter in ('swapins', 'swapouts'):
        if counter in before and counter in after and after[counter] > before[counter]:
            issues.append(f'{counter} increased during request')
    for state in (before, after):
        if "lowpowermode         1" in state["power_settings"]:
            issues.append("low power mode enabled")
        if not all(line in state['thermal'] for line in (
                'No thermal warning level has been recorded',
                'No performance warning level has been recorded')):
            issues.append("thermal state requires review")
    return sorted(set(issues))


def native_headers(model_path):
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    dtypes = {}
    for name in shards:
        with (model_path / name).open("rb") as source:
            size = struct.unpack("<Q", source.read(8))[0]
            if size > 32 * 1024 * 1024:
                raise ValueError("Unexpected safetensors header size")
            header = json.loads(source.read(size))
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            expected = "U8" if key.endswith(("_blocks", "_scales")) else "BF16"
            if tensor["dtype"] != expected:
                raise ValueError(f"Unexpected native precision: {key}")
            dtypes[key] = tensor["dtype"]
    if set(dtypes) != set(index["weight_map"]):
        raise ValueError("Checkpoint index/header mismatch")
    return shards, dtypes


def request(model, ids, step_size, mx, make_prompt_cache):
    cache = make_prompt_cache(model)
    prompt = mx.array([ids], dtype=mx.int32)
    mx.eval(prompt)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    start = time.perf_counter()
    # Match MLX-LM's cache-prefill layout: leave one token for the first logit.
    remaining = prompt
    while remaining.shape[1] > 1:
        n = min(step_size, remaining.shape[1] - 1)
        model(remaining[:, :n], cache=cache)
        mx.eval([c.state for c in cache])
        remaining = remaining[:, n:]
        mx.clear_cache()
    logits = model(remaining, cache=cache)[:, -1, :]
    mx.eval(logits)
    mx.synchronize()
    prefill_end = time.perf_counter()
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    token_ids = [token.item()]
    timestamps = [time.perf_counter()]
    for _ in range(127):
        logits = model(token[:, None], cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        token_ids.append(token.item())
        timestamps.append(time.perf_counter())
    mx.synchronize()
    peak = mx.get_peak_memory()
    # Validation is outside the interval. The baseline has no candidate to compare.
    finite = bool(mx.all(mx.isfinite(logits)).item())
    cache_dtypes = sorted({str(t.dtype) for c in cache for t in c.state})
    if not finite or cache_dtypes != ["mlx.core.bfloat16"]:
        raise ValueError(f"Native output/cache check failed: {finite}, {cache_dtypes}")
    values, intervals = metrics(len(ids), start, prefill_end, timestamps, peak)
    return {
        "metrics": values, "inter_token_intervals_ms": intervals,
        "output_token_ids": token_ids, "cache_dtypes": cache_dtypes,
        "last_logits_finite": finite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--validate-reference", action='store_true')
    parser.add_argument("--paired-aa", action='store_true')
    parser.add_argument("--controlled", action='store_true', help='Require controller acknowledgement between timed pairs')
    parser.add_argument("--rehash-checkpoint", action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    contract_path = repo / "kernelmaxxing.yaml"
    contract = load_contract(contract_path)
    spec = contract.to_dict()
    # Explicit guard: changing this workload needs a reviewed new driver.
    workload = spec["workload"]
    if (workload["generation_tokens"], workload["batch_size"],
            workload["sampling"], workload["eos_stopping"],
            workload["reuse_prompt_cache"]) != (128, 1, "greedy", False, False):
        raise ValueError("Unsupported workload")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "schema": "native_baseline_calibration_attempt_v1", "run_id": str(uuid.uuid4()),
        "status": "preflight", "official": False, "verdict": None,
        "contract": spec, "contract_sha256": contract.sha256,
        "driver_sha256": digest(__file__), "errors": [], "samples": [], "warmups": [],
        "loading_strategy": "upstream lazy load on CPU stream; native byte-layout conversion and one-parameter eval on CPU; Metal inference; outside timing",
        "mode": {'validate_reference': args.validate_reference, 'paired_aa': args.paired_aa,
                 'controlled': args.controlled, 'rehash_checkpoint': args.rehash_checkpoint},
        'primary_baseline_lane': 'A',
        "calibration_policy": {'bootstrap_replicates': 10000, 'confidence': .95,
                               'aa_ci_contains_one': True, 'maximum_ratio_ci_width': .04,
                               'native_thermal_state': 'nominal throughout request',
                               'swap_policy': 'no swap growth, swapins or swapouts during any request',
                               'candidate_acceptance': 'disabled; unchanged contract'},
        "review_required": ["synchronous driver and timing boundaries", "background CPU/GPU isolation",
                            "thermal telemetry resolution", "held-out/operator correctness suite",
                            "baseline noise and tolerance calibration"],
        "timing_definition": {
            "driver": "unchanged MLX-LM model, synchronous greedy, no lookahead",
            "prefill": "forward start through materialized final prompt logits, before argmax",
            "decode": "127 materialized token intervals; no EOS stop",
            "inter_token_trial_metric": "median of the 127 retained intervals",
            "peak_memory": "MLX allocator peak including resident weights; not total system memory",
        },
    }

    def save():
        temporary = args.output / "attempt.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "attempt.json")

    def checkpoint(detail):
        if not args.controlled:
            return
        token = str(uuid.uuid4())
        state['controller_checkpoint'] = {'token': token, 'detail': detail}
        save()
        gate = args.output / 'gate.json'
        gate.write_text(json.dumps({'token': token, 'detail': detail, 'status': 'paused',
                                    'samples': len(state['samples']), 'warmups': len(state['warmups'])}))
        acknowledgement = args.output / 'resume.json'
        while True:
            if acknowledgement.exists():
                receipt = json.loads(acknowledgement.read_text())
                if receipt['token'] == token:
                    state.setdefault('controller_receipts', []).append(receipt)
                    break
            time.sleep(.5)
        gate.write_text(json.dumps({'token': token, 'status': 'running'}))
        time.sleep(5)  # Let the controller's update/UI activity settle before timing.

    def collect(model, ids, length, trial, lane):
        before = environment()
        if environment_issues(before, before):
            raise RuntimeError(f'Pre-request environment invalid: {environment_issues(before, before)}')
        observed_thermal = [before['thermal_state']]
        stop = threading.Event()
        def observe():
            while not stop.wait(.5):
                observed_thermal.append(thermal_state())
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            sample = request(model, ids, workload['prefill_step_size'], mx, make_prompt_cache)
        finally:
            stop.set()
            observer.join()
        after = environment()
        sample.update(prompt_tokens=length, trial=trial, lane=lane,
                      environment_before=before, environment_after=after,
                      thermal_states_during_request=observed_thermal)
        sample['invalid_reasons'] = environment_issues(before, after)
        if any(t != 0 for t in observed_thermal):
            sample['invalid_reasons'].append('native thermal state changed during request')
        return sample

    try:
        save()
        verification = json.loads(args.verification.read_text())
        shards, dtypes = native_headers(args.model)
        if (verification["revision"] != spec["model"]["revision"]
                or verification["exit_code"] != 0 or verification["result"]["checked"] != 10
                or verification["result"]["repo_id"] != spec["model"]["repository"]):
            raise ValueError("Pinned checkpoint verification did not pass")
        state["checkpoint_verification"] = verification
        state["checkpoint_dtypes"] = dtypes
        state["checkpoint_files"] = {}
        for name in shards:
            metadata = args.model / ".cache/huggingface/download" / (name + ".metadata")
            commit, sha, *_ = metadata.read_text().splitlines()
            if commit != spec["model"]["revision"] or not re.fullmatch("[0-9a-f]{64}", sha):
                raise ValueError("Missing checkpoint hash provenance")
            state["checkpoint_files"][name] = {"sha256": sha, "bytes": (args.model / name).stat().st_size}
        if args.rehash_checkpoint:
            state['status'] = 'hashing_checkpoint'
            save()
            for name in shards:
                print(f'Verifying actual SHA256: {name}', flush=True)
                hasher = hashlib.sha256()
                with (args.model / name).open('rb') as source:
                    while block := source.read(4 * 1024 * 1024):
                        hasher.update(block)
                actual = hasher.hexdigest()
                state['checkpoint_files'][name]['observed_sha256'] = actual
                save()
                if actual != state['checkpoint_files'][name]['sha256']:
                    raise ValueError(f'Checkpoint hash mismatch: {name}')
        config = json.loads((args.model / "config.json").read_text())
        if config["quantization_config"]["quant_method"] != "mxfp4":
            raise ValueError("Not the native MXFP4 checkpoint")
        state["inputs"] = fixed_inputs(workload["prompt_lengths"], config["vocab_size"], workload["seed"])
        state["inputs_sha256"] = hashlib.sha256(json.dumps(state["inputs"], sort_keys=True).encode()).hexdigest()
        state["hardware"] = {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "model_identifier": command("sysctl", "-n", "hw.model"),
            "memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
        }
        hardware = spec["hardware"]
        if state["hardware"] != {"chip": hardware["chip"], "model_identifier": hardware["model_identifier"],
                                "memory_bytes": hardware["memory_gib"] * 2**30}:
            raise ValueError("Hardware differs from contract")
        from mlx_lm import load
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import wired_limit
        from mlx.utils import tree_flatten
        import mlx_lm
        import mlx_lm.models.gpt_oss as implementation
        import mlx_lm.utils as loader
        state["runtime"] = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm", "numpy", "transformers")}
        state["runtime"].update(python=platform.python_version(), macos=platform.mac_ver()[0])
        state["runtime_source_sha256"] = {str(p): digest(p) for p in Path(mlx_lm.__file__).parent.rglob('*.py')}
        state['runtime_binary_sha256'] = {str(p): digest(p) for p in Path(mx.__file__).parent.rglob('*')
                                          if p.is_file() and p.suffix in ('.so', '.dylib', '.metallib')}
        state['evaluator_source_sha256'] = {str(p): digest(p) for p in Path(__file__).parent.glob('*.py')}
        state["environment_before_load"] = environment()
        if "'AC Power'" not in state["environment_before_load"]["power"]:
            raise ValueError("Plug into AC power before the baseline")
        state["status"] = "loading"
        save()
        print("Loading verified native checkpoint; model loading is excluded from timing.", flush=True)
        started = time.perf_counter()
        def loading_progress(completed, total, name, size):
            state["loading_progress"] = {
                "completed_tensors": completed, "total_tensors": total,
                "current_tensor": name, "current_tensor_bytes": size,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if completed % 25 == 0 or completed == total:
                save()
                print(f"Materializing native tensors: {completed}/{total}; next={name}", flush=True)

        # Native sanitize consists of packed-weight views/slices/contiguous copies.
        # Build these operations on CPU so the SSD-backed byte copies do not
        # occupy a Metal command buffer. Unified-memory arrays are subsequently
        # consumed by unchanged Metal inference after exiting this context.
        with mx.stream(mx.cpu):
            model, _ = load(str(args.model), lazy=True)
            state["status"] = "materializing"
            materialize_parameters(tree_flatten(model.parameters()), mx, loading_progress)
        mx.synchronize()
        state["load_seconds"] = time.perf_counter() - started
        for layer in model.model.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                module = getattr(layer.mlp.experts, name)
                if (module.mode, module.bits, module.group_size) != ("mxfp4", 4, 32):
                    raise ValueError("Unexpected expert weight representation")
            if layer.self_attn.q_proj.weight.dtype != mx.bfloat16:
                raise ValueError("Attention weights are not BF16")
        if model.model.embed_tokens.weight.dtype != mx.bfloat16 or model.lm_head.weight.dtype != mx.bfloat16:
            raise ValueError("Embedding/output weights are not BF16")
        if args.validate_reference:
            state['status'] = 'correctness'
            save()
            print('Running stock generation and independent selected-expert reference checks.', flush=True)
            with wired_limit(model):
                held_out = fixed_inputs([257], config['vocab_size'], workload['seed'] + 1009)['257']
                state['held_out_input_ids'] = held_out
                state['driver_correctness'] = driver_checks(model, state['inputs'], held_out,
                    workload['prefill_step_size'], request, mx, make_prompt_cache)
                save()
                if not all(r['passed'] for r in state['driver_correctness']):
                    raise RuntimeError('Stock-generation output comparison failed')
                state['expert_correctness'] = expert_checks(model, mx,
                    spec['correctness']['atol'], spec['correctness']['rtol'])
                save()
                if len(state['expert_correctness']) != 21 or not all(r['passed'] for r in state['expert_correctness']):
                    raise RuntimeError('Selected-expert reference correctness failed')
        state["status"] = "settling"
        save()
        print(f"Native load passed. Settling {args.settle_seconds}s; close unrelated heavy apps.", flush=True)
        time.sleep(args.settle_seconds)
        with wired_limit(model):
            # Tiny untimed precision/finite-output probe before allocating large requests.
            state["precision_probe"] = request(model, [1, 2], workload["prefill_step_size"], mx, make_prompt_cache)
            save()
            for phase, repeats in (("warmups", spec["benchmark"]["warmups"]),
                                   ("samples", spec["benchmark"]["trials_per_implementation"])):
                state["status"] = phase
                save()
                for trial in range(repeats):
                    lengths = workload["prompt_lengths"]
                    # Counterbalance context-length order. There is no fake candidate A/B.
                    order = lengths[trial % len(lengths):] + lengths[:trial % len(lengths)]
                    for length in order:
                        checkpoint(f'{phase} round {trial+1}/{repeats}, prompt {length}')
                        lanes = (('A', 'B') if trial % 2 == 0 else ('B', 'A')) if args.paired_aa and phase == 'samples' else ('A',)
                        for lane in lanes:
                            sample = collect(model, state['inputs'][str(length)], length, trial+1, lane)
                            state[phase].append(sample)
                            save()
                            print(f"{phase} {trial+1}/{repeats} prompt={length} lane={lane}: {sample['metrics']['decode_tokens_per_second']:.2f} tok/s; invalid={sample['invalid_reasons']}", flush=True)
                            if sample["invalid_reasons"]:
                                raise RuntimeError("Environment invalidated request; retained evidence and stopped")
        if digest(__file__) != state["driver_sha256"] or load_contract(contract_path).sha256 != state["contract_sha256"]:
            raise ValueError("Protected input changed during collection")
        for path, expected in {**state['runtime_source_sha256'], **state['runtime_binary_sha256'], **state['evaluator_source_sha256']}.items():
            if digest(path) != expected:
                raise ValueError(f'Protected source changed: {path}')
        if args.paired_aa:
            state['aa_calibration'] = aa_noise(state['samples'], workload['prompt_lengths'], workload['seed'])
            save()
            if not all(v['passed'] for v in state['aa_calibration'].values()):
                raise RuntimeError('A/A noise calibration did not pass predeclared gates')
        state["summary"] = {}
        for length in workload["prompt_lengths"]:
            selected = [s for s in state["samples"] if s["prompt_tokens"] == length]
            values = {}
            for key in selected[0]["metrics"]:
                xs = sorted(s["metrics"][key] for s in selected)
                median = statistics.median(xs)
                values[key] = {"median": median, "p90": xs[math.ceil(.9 * len(xs)) - 1],
                               "mad": statistics.median(abs(x - median) for x in xs)}
            state["summary"][str(length)] = values
        state["status"] = "completed_pending_review"
        print("Collection completed. Provisional until reviewer/environment/correctness gates pass.", flush=True)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["errors"].append("Interrupted by user")
        raise
    except BaseException as error:
        state["status"] = "failed"
        state["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        (args.output / 'gate.json').write_text(json.dumps({'status': state['status'], 'errors': state['errors']}))


if __name__ == "__main__":
    main()
