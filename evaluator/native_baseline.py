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
import os
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
import copy
import shutil


class EnvironmentInvalid(RuntimeError):
    """Only this failure class permits a fresh attempt in the same worker."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def resident_retry_loop(run, new_attempt, retain_failure, verify, cooldown, retries):
    """Retain entire failed attempts; never retry statistical or GPU failures."""
    for index in range(1, retries + 2):
        try:
            return run(index)
        except EnvironmentInvalid as error:
            retain_failure(error, index)
            if index > retries:
                raise
            verify()
            cooldown()
            new_attempt(index + 1)


def verify_resident_inputs(state, contract_path):
    if load_contract(contract_path).sha256 != state['contract_sha256']:
        raise ValueError('Protected contract changed; resident reuse refused')
    for path, expected in {**state['runtime_source_sha256'], **state['runtime_binary_sha256'],
                           **state['evaluator_source_sha256'], **state.get('model_support_sha256', {})}.items():
        if digest(path) != expected:
            raise ValueError(f'Protected input changed; resident reuse refused: {path}')
    for path, expected in state.get('checkpoint_stat_at_verification', {}).items():
        observed = Path(path).stat()
        actual = [observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns]
        if actual != expected:
            raise ValueError(f'Checkpoint changed since verified load; resident reuse refused: {path}')

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
        'process_cpu_time_seconds': time.process_time(),
    }
    for label in ('Swapins', 'Swapouts'):
        counter = re.search(rf'^{label}:\s*(\d+)\.', vm, re.MULTILINE)
        if counter:
            result[label.lower()] = int(counter.group(1))
    return result


def environment_issues(before, after, swap_policy='strict'):
    if swap_policy not in ('strict', 'paging-aware'):
        raise ValueError('Unknown swap policy')
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
        if counter in before and counter in after:
            if after[counter] < before[counter]:
                issues.append(f'{counter} counter decreased unexpectedly')
            elif after[counter] > before[counter] and (counter == 'swapouts' or swap_policy == 'strict'):
                issues.append(f'{counter} increased during request')
        elif swap_policy == 'paging-aware':
            issues.append(f'{counter} observation missing')
    for state in (before, after):
        if "lowpowermode         1" in state["power_settings"]:
            issues.append("low power mode enabled")
        if not all(line in state['thermal'] for line in (
                'No thermal warning level has been recorded',
                'No performance warning level has been recorded')):
            issues.append("thermal state requires review")
    return sorted(set(issues))


def swap_observation(before, after, swap_policy):
    """Keep allowed paging visible; it is never removed from the evidence."""
    return {'policy': swap_policy, 'swapin_pages': after['swapins'] - before['swapins'],
            'swapout_pages': after['swapouts'] - before['swapouts'],
            'swap_growth_mib': after['swap_used_mib'] - before['swap_used_mib'],
            'swapins_permitted': swap_policy == 'paging-aware'}


def quiet_preflight(observe=environment, clock=time.monotonic, sleep=time.sleep,
                    quiet_seconds=10, timeout_seconds=60, swap_policy='strict'):
    """Wait for an untimed quiet window; never discard/retry a measured sample."""
    start = quiet_start = clock()
    previous = observe()
    observations = []
    while True:
        sleep(2)
        current = observe()
        now = clock()
        issues = environment_issues(previous, current, swap_policy)
        observations.append({'elapsed_seconds': now - start,
                             'swapins': current['swapins'], 'swapouts': current['swapouts'],
                             'swap_used_mib': current['swap_used_mib'],
                             'thermal_state': current['thermal_state'], 'issues': issues,
                             'swap_activity': swap_observation(previous, current, swap_policy)})
        if issues:
            quiet_start = now
        if now - quiet_start >= quiet_seconds and not issues:
            return {'quiet_seconds_required': quiet_seconds, 'waited_seconds': now - start,
                    'observations': observations, 'swap_policy': swap_policy}
        if now - start >= timeout_seconds:
            raise EnvironmentInvalid('Untimed quiet preflight did not settle; no request started',
                {'quiet_seconds_required': quiet_seconds, 'waited_seconds': now - start,
                 'observations': observations, 'swap_policy': swap_policy})
        previous = current


def resident_precondition(lengths, run, observe=environment, retain=lambda row: None,
                          max_rounds=5, required_clean_rounds=2, prepare=lambda: None,
                          swap_policy='strict'):
    """Untimed resident-state setup, before the contract's five warmups.

    Retain every setup request, including paging. Require two complete clean
    rounds; never reinterpret these requests as official timing samples.
    """
    records, clean_rounds = [], 0
    for round_index in range(1, max_rounds + 1):
        clean = True
        for length in lengths:
            ready = prepare()
            before = observe()
            sample = run(length)
            after = observe()
            issues = environment_issues(before, after, swap_policy)
            row = {'round': round_index, 'prompt_tokens': length,
                   'excluded_from_baseline': True, 'purpose': 'untimed resident-state priming',
                   'quiet_preflight': ready,
                   'environment_before': before, 'environment_after': after,
                   'environment_events': issues,
                   'swap_activity': swap_observation(before, after, swap_policy), **sample}
            records.append(row)
            retain(row)
            clean = clean and not issues
        clean_rounds = clean_rounds + 1 if clean else 0
        if clean_rounds >= required_clean_rounds:
            return records
    raise EnvironmentInvalid('Resident-state priming did not produce two clean rounds; no official warmups started')


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
    parser.add_argument('--dense-diagnostic', type=Path)
    parser.add_argument('--resident-retries', type=int, default=0,
                        help='Additional complete attempts after environment-only failure; no reloading')
    parser.add_argument('--swap-policy', choices=('strict', 'paging-aware'), default='strict',
                        help='Paging-aware permits logged swap-ins, never swap-outs or swap growth')
    parser.add_argument('--per-request-control', action='store_true',
                        help='Require the same untimed controlled settling/quiet window before both lanes')
    args = parser.parse_args()
    if not 0 <= args.resident_retries <= 2:
        raise ValueError('Resident retry limit must be between zero and two')
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
    session_root = args.output
    session = {'session_id': str(uuid.uuid4()), 'worker_pid': os.getpid(),
               'status': 'running', 'phase': 'preflight', 'current_attempt': str(args.output),
               'model_loads': 0, 'max_environment_retries': args.resident_retries, 'failed_attempts': []}
    state = {
        "schema": "native_baseline_calibration_attempt_v1", "run_id": str(uuid.uuid4()),
        "status": "preflight", "official": False, "verdict": None,
        "contract": spec, "contract_sha256": contract.sha256,
        "driver_sha256": digest(__file__), "errors": [], "samples": [], "warmups": [],
        'swap_policy': args.swap_policy,
        "loading_strategy": "upstream lazy load on CPU stream; native byte-layout conversion and one-parameter eval on CPU; Metal inference; outside timing",
        "mode": {'validate_reference': args.validate_reference, 'paired_aa': args.paired_aa,
                 'controlled': args.controlled, 'rehash_checkpoint': args.rehash_checkpoint,
                 'per_request_control': args.per_request_control},
        'primary_baseline_lane': 'A',
        'resident_session': {'session_id': session['session_id'], 'worker_pid': session['worker_pid'],
                             'attempt_index': 1, 'model_load_run_id': None, 'load_reused': False,
                             'max_environment_retries': args.resident_retries, 'retained_failed_attempts': []},
        "calibration_policy": {'bootstrap_replicates': 10000, 'confidence': .95,
                               'aa_ci_contains_one': True, 'maximum_ratio_ci_width': .04,
                               'native_thermal_state': 'nominal throughout request',
                               'swap_policy': ('no swap growth, swapins or swapouts during any request'
                                   if args.swap_policy == 'strict' else
                                   'swap-ins allowed and retained; no swap-outs or swap growth during any request'),
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
        pointer = session_root / 'resident-session.json.tmp'
        pointer.write_text(json.dumps(session, indent=2) + '\n')
        pointer.replace(session_root / 'resident-session.json')

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
        preflight = quiet_preflight(swap_policy=args.swap_policy)
        state.setdefault('quiet_preflights', []).append({'checkpoint': token, **preflight})
        save()

    def collect(model, ids, length, trial, lane):
        before = environment()
        if environment_issues(before, before, args.swap_policy):
            raise EnvironmentInvalid(f'Pre-request environment invalid: {environment_issues(before, before, args.swap_policy)}', before)
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
        sample['swap_activity'] = swap_observation(before, after, args.swap_policy)
        sample['invalid_reasons'] = environment_issues(before, after, args.swap_policy)
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
        state['model_support_sha256'] = {str(args.model / name): digest(args.model / name) for name in
            ('config.json', 'generation_config.json', 'model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')}
        state['checkpoint_stat_at_verification'] = {}
        for name in shards:
            stat = (args.model / name).stat()
            state['checkpoint_stat_at_verification'][str(args.model / name)] = [
                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
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
        state["runtime"] = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm", "numpy", "transformers")}
        state["runtime"].update(python=platform.python_version(), macos=platform.mac_ver()[0])
        state["runtime_source_sha256"] = {str(p): digest(p) for p in Path(mlx_lm.__file__).parent.rglob('*.py')}
        state['runtime_binary_sha256'] = {str(p): digest(p) for p in Path(mx.__file__).parent.rglob('*')
                                          if p.is_file() and p.suffix in ('.so', '.dylib', '.metallib')}
        state['evaluator_source_sha256'] = {str(p): digest(p) for p in Path(__file__).parent.glob('*.py')}
        snapshot_root = args.output / 'protected_sources'
        for label, sources, root in (('evaluator', state['evaluator_source_sha256'], Path(__file__).parent),
                                     ('mlx_lm', state['runtime_source_sha256'], Path(mlx_lm.__file__).parent)):
            for path in sources:
                source = Path(path)
                target = snapshot_root / label / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        if args.dense_diagnostic:
            diagnostic_bytes = args.dense_diagnostic.read_bytes()
            diagnostic = json.loads(diagnostic_bytes)
            if diagnostic['contract_sha256'] != contract.sha256:
                raise ValueError('Dense diagnostic contract mismatch')
            state['dense_reference_diagnostic'] = {
                'artifact': str(args.dense_diagnostic),
                'sha256': hashlib.sha256(diagnostic_bytes).hexdigest(),
                'failed_cases': [r for r in diagnostic.get('expert_correctness', []) if not r['passed']],
                'scope': 'alternate dense BF16 kernel; unresolved numerical discrepancy; not the native contract reference'}
            (args.output / 'dense_reference_diagnostic.json').write_bytes(diagnostic_bytes)
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
        session['model_loads'] = 1
        state['resident_session']['model_load_run_id'] = state['run_id']
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
            print('Running stock generation comparisons and native selected-expert dispatch-invariance checks.', flush=True)
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
        frozen_base = copy.deepcopy(state)

        def retain_failure(error, index):
            state['status'] = 'failed'
            state['failure_category'] = 'environment'
            state['errors'].append(f'EnvironmentInvalid: {error}')
            if error.evidence is not None:
                state['failure_environment_evidence'] = error.evidence
            session['phase'] = 'environment_failure_retained'
            save()
            artifact = args.output / 'attempt.json'
            session['failed_attempts'].append({'run_id': state['run_id'], 'artifact': str(artifact),
                'sha256': digest(artifact), 'failure_category': 'environment', 'attempt_index': index})
            save()
            (args.output / 'gate.json').write_text(json.dumps({'status': 'failed', 'errors': state['errors']}))
            print(f"Attempt {index} environment-invalid; retained at {artifact}. Model stays resident.", flush=True)

        def cooldown():
            mx.synchronize()
            mx.clear_cache()
            session['phase'] = 'resident_retry_cooldown'
            save()
            print('Resident worker cooldown: 60 seconds; no checkpoint rehash or model reload.', flush=True)
            time.sleep(60)

        def new_attempt(index):
            nonlocal state
            args.output = session_root.with_name(session_root.name + f'-retry{index:02d}')
            args.output.mkdir(exist_ok=False)
            state = copy.deepcopy(frozen_base)
            state.update(run_id=str(uuid.uuid4()), status='resident_retry_preflight', errors=[], warmups=[], samples=[])
            state['resident_session'].update(attempt_index=index, load_reused=True,
                retained_failed_attempts=copy.deepcopy(session['failed_attempts']))
            shutil.copytree(session_root / 'protected_sources', args.output / 'protected_sources')
            if (session_root / 'dense_reference_diagnostic.json').exists():
                shutil.copy2(session_root / 'dense_reference_diagnostic.json', args.output / 'dense_reference_diagnostic.json')
            session.update(current_attempt=str(args.output), phase='resident_retry_preflight')
            save()
            print(f"Fresh attempt {index}: {args.output}; run_id={state['run_id']}; model_loads=1", flush=True)

        def run_attempt(index):
            verify_resident_inputs(state, contract_path)
            state['status'] = 'settling'
            session['phase'] = 'settling'
            save()
            print(f"Resident model ready for attempt {index}. Settling {args.settle_seconds}s; loading excluded.", flush=True)
            time.sleep(args.settle_seconds)
            # Tiny untimed precision/finite-output probe before allocating large requests.
            state["precision_probe"] = request(model, [1, 2], workload["prefill_step_size"], mx, make_prompt_cache)
            save()
            state['status'] = 'resident_preconditioning'
            session['phase'] = state['status']
            state['resident_preconditioning'] = []
            save()
            def retain_priming(row):
                state['resident_preconditioning'].append(row)
                save()
                print(f"Untimed resident priming round {row['round']} prompt={row['prompt_tokens']}: events={row['environment_events']}", flush=True)
            resident_precondition(workload['prompt_lengths'],
                                  lambda length: request(model, state['inputs'][str(length)],
                                                         workload['prefill_step_size'], mx, make_prompt_cache),
                                  retain=retain_priming,
                                  prepare=lambda: quiet_preflight(swap_policy=args.swap_policy),
                                  swap_policy=args.swap_policy)
            for phase, repeats in (("warmups", spec["benchmark"]["warmups"]),
                                   ("samples", spec["benchmark"]["trials_per_implementation"])):
                state["status"] = phase
                session['phase'] = phase
                save()
                for trial in range(repeats):
                    lengths = workload["prompt_lengths"]
                    # Counterbalance context-length order. There is no fake candidate A/B.
                    order = lengths[trial % len(lengths):] + lengths[:trial % len(lengths)]
                    for length in order:
                        if not args.per_request_control or phase == 'warmups':
                            checkpoint(f'{phase} round {trial+1}/{repeats}, prompt {length}')
                        lanes = (('A', 'B') if trial % 2 == 0 else ('B', 'A')) if args.paired_aa and phase == 'samples' else ('A',)
                        for lane in lanes:
                            if args.per_request_control and phase == 'samples':
                                checkpoint(f'{phase} round {trial+1}/{repeats}, prompt {length}, lane {lane}')
                            sample = collect(model, state['inputs'][str(length)], length, trial+1, lane)
                            state[phase].append(sample)
                            save()
                            print(f"{phase} {trial+1}/{repeats} prompt={length} lane={lane}: {sample['metrics']['decode_tokens_per_second']:.2f} tok/s; invalid={sample['invalid_reasons']}", flush=True)
                            if sample["invalid_reasons"]:
                                raise EnvironmentInvalid("Environment invalidated request; retained evidence and stopped")
            verify_resident_inputs(state, contract_path)
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
            session.update(status='completed_pending_review', phase='completed_pending_review')
            print("Collection completed. Provisional until reviewer/environment/correctness gates pass.", flush=True)

        with wired_limit(model):
            resident_retry_loop(run_attempt, new_attempt, retain_failure,
                                lambda: verify_resident_inputs(state, contract_path),
                                cooldown, args.resident_retries)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["errors"].append("Interrupted by user")
        session.update(status='interrupted', phase='interrupted', errors=state['errors'])
        raise
    except BaseException as error:
        state["status"] = "failed"
        message = f"{type(error).__name__}: {error}"
        if message not in state['errors']:
            state["errors"].append(message)
        session.update(status='failed', phase='failed', errors=state['errors'])
        raise
    finally:
        save()
        (args.output / 'gate.json').write_text(json.dumps({'status': state['status'], 'errors': state['errors']}))


if __name__ == "__main__":
    main()
