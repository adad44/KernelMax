"""Exploration triage and passive phase-memory observations; no verdicts.

Phase observations are instrumentation for hypothesis discovery. Confirmation
uses fresh matched comparisons and the unchanged strict environment checks.
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import threading
import time

from evaluator.native_baseline import environment_issues


_SWAP_FLAGS = frozenset(
    {
        "swap grew during the request",
        "swapouts increased during request",
        "swapins increased during request",
    }
)


def assess_environment(before, after, *, mode, thermal_states=()):
    if mode not in {"exploration", "confirmation"}:
        raise ValueError("measurement mode must be exploration or confirmation")
    flags = environment_issues(before, after, swap_policy="paging-aware")
    if any(value != 0 for value in thermal_states):
        flags.append("nonnominal thermal state during request")
    flags = sorted(set(flags))
    swap_flags = [value for value in flags if value in _SWAP_FLAGS]
    stop_reasons = [
        value for value in flags if mode == "confirmation" or value not in _SWAP_FLAGS
    ]
    return {
        "mode": mode,
        "flags": flags,
        "swap_affected": bool(swap_flags),
        "continue_pass": not stop_reasons,
        "stop_reasons": stop_reasons,
        "confirmation_sample_eligible": mode == "confirmation" and not flags,
        "attribution": "system-wide counters; no identification of model pages",
    }


class ExplorationComplete(RuntimeError):
    """A predeclared pass bound was reached; retain completed work normally."""


class BoundedExploration:
    def __init__(
        self, policy, *, clock=time.monotonic, max_requests=None, max_seconds=None
    ):
        self.clock = clock
        self.started = clock()
        self.max_requests = (
            policy.max_requests if max_requests is None else max_requests
        )
        self.max_seconds = (
            policy.max_wall_time_seconds if max_seconds is None else max_seconds
        )
        if (
            type(self.max_requests) is not int
            or not 0 < self.max_requests <= policy.max_requests
            or type(self.max_seconds) not in (int, float)
            or not 0 < self.max_seconds <= policy.max_wall_time_seconds
        ):
            raise ValueError("pass bounds must be positive and within the contract")
        self.requests = 0

    def begin_request(self):
        if self.requests >= self.max_requests:
            raise ExplorationComplete("request bound reached")
        if self.clock() - self.started >= self.max_seconds:
            raise ExplorationComplete("wall-time bound reached")
        self.requests += 1


def parse_vm_stat(raw):
    header = re.search(r"page size of (\d+) bytes", raw or "")
    counters = {
        name.strip(): int(value)
        for name, value in re.findall(r"^([^:\n]+):\s*(\d+)\.", raw or "", re.MULTILINE)
    }
    return {
        "page_size_bytes": int(header.group(1)) if header else None,
        "counts": counters,
    }


def _read(args):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.PIPE, timeout=2
    ).strip()


def memory_snapshot(mx=None, *, read=_read):
    started = time.monotonic()
    errors = []

    def command(args):
        try:
            return read(args)
        except Exception as error:
            errors.append({"command": args, "error": str(error)[:300]})
            return None

    vm_raw = command(["vm_stat"])
    swap_raw = command(["sysctl", "-n", "vm.swapusage"])
    pressure = command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    rss = command(["ps", "-p", str(os.getpid()), "-o", "rss="])
    vm = parse_vm_stat(vm_raw)
    size = vm["page_size_bytes"]
    used = re.search(r"used\s*=\s*([\d.]+)M", swap_raw or "")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    gauges = {}
    for name in (
        "Pages free",
        "Pages active",
        "Pages inactive",
        "Pages speculative",
        "Pages wired down",
        "File-backed pages",
        "Pages purgeable",
        "Pages occupied by compressor",
        "Pages stored in compressor",
    ):
        count = vm["counts"].get(name)
        gauges[name] = count * size if count is not None and size is not None else None
    allocator = None
    if mx is not None:
        allocator = {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        }
    return {
        "observed_at": time.time(),
        "monotonic": time.monotonic(),
        "vm": vm,
        "memory_bytes": gauges,
        "memory_pressure_level_raw": int(pressure)
        if pressure and pressure.isdigit()
        else None,
        "swap_used_bytes": round(float(used.group(1)) * 1024**2) if used else None,
        "swap_raw": swap_raw,
        "vm_stat_raw": vm_raw,
        "process": {
            "pid": os.getpid(),
            "rss_bytes": int(rss) * 1024 if rss and rss.strip().isdigit() else None,
            "max_rss_bytes": usage.ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "minor_faults": usage.ru_minflt,
            "major_faults": usage.ru_majflt,
        },
        "mlx_allocator": allocator,
        "observation_wall_seconds": time.monotonic() - started,
        "errors": errors,
    }


def memory_delta(before, after):
    seconds = after["monotonic"] - before["monotonic"]
    size = before["vm"]["page_size_bytes"]
    valid = seconds > 0 and size is not None and size == after["vm"]["page_size_bytes"]
    result = {
        "seconds": seconds,
        "swap_used_delta_bytes": None,
        "counter_errors": [],
        "swap_attribution": "system-wide; process faults are not swap-specific",
        "rate_byte_basis": "VM page-equivalent bytes, not measured disk I/O",
    }
    if before["swap_used_bytes"] is not None and after["swap_used_bytes"] is not None:
        result["swap_used_delta_bytes"] = (
            after["swap_used_bytes"] - before["swap_used_bytes"]
        )
    for label in (
        "Swapins",
        "Swapouts",
        "Pageins",
        "Pageouts",
        "Compressions",
        "Decompressions",
    ):
        a = before["vm"]["counts"].get(label)
        b = after["vm"]["counts"].get(label)
        delta = b - a if a is not None and b is not None else None
        if delta is not None and delta < 0:
            result["counter_errors"].append(label + " decreased")
        rate_valid = valid and delta is not None and delta >= 0
        result[label] = {
            "pages": delta,
            "pages_per_second": delta / seconds if rate_valid else None,
            "bytes_per_second": delta * size / seconds if rate_valid else None,
        }
    return result


class PhaseMemoryObserver:
    """Passive read-only signals. Boundary/periodic overhead is retained explicitly.

    Prefill includes production/readback of the first token. Decode begins before
    the first generated-token forward, and ends after the request's normal drain.
    This instrumentation is never used to establish a confirmation speedup.
    """

    def __init__(
        self, prompt_tokens, mx=None, *, interval_seconds=1.0, snapshot=memory_snapshot
    ):
        if prompt_tokens < 1 or interval_seconds <= 0:
            raise ValueError("positive prompt length and observation interval required")
        self.remaining_prompt = prompt_tokens
        self.mx = mx
        self.interval = interval_seconds
        self.snapshot = snapshot
        self.phase = "before_request"
        self.samples = []
        self.errors = []
        self.lock = threading.Lock()
        self.sample_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def sample(self, reason):
        with self.sample_lock:
            with self.lock:
                phase = self.phase
            value = self.snapshot(self.mx)
            value.update(phase=phase, reason=reason)
            with self.lock:
                self.samples.append(value)

    def transition(self, phase):
        with self.lock:
            self.phase = phase
        self.sample("boundary")

    def start(self):
        self.sample("boundary")

        def observe():
            while not self.stop.wait(self.interval):
                try:
                    self.sample("periodic")
                except Exception as error:
                    with self.lock:
                        self.errors.append(str(error))

        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()

    def before_model_call(self, token_count):
        phase = "prefill" if self.remaining_prompt > 0 else "decode"
        if phase != self.phase:
            self.transition(phase)
        self.remaining_prompt = max(0, self.remaining_prompt - token_count)

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("memory observer did not stop")
        self.transition("after_decode")
        boundaries = {r["phase"]: r for r in self.samples if r["reason"] == "boundary"}
        phases = {}
        for phase, next_phase in (("prefill", "decode"), ("decode", "after_decode")):
            if phase in boundaries and next_phase in boundaries:
                phases[phase] = memory_delta(boundaries[phase], boundaries[next_phase])
        return {
            "samples": self.samples,
            "phase_deltas": phases,
            "errors": self.errors,
            "observation_wall_seconds": sum(
                r["observation_wall_seconds"] for r in self.samples
            ),
            "confirmation_eligible": False,
            "limits": "Boundary reads instrument exploration; global swap is not process attribution. Process faults/RSS and MLX allocator usage are complementary signals, not complete model residency. File-backed pages do not reconstruct Activity Monitor Cached Files or its pressure formula.",
        }


class ObservedModel:
    def __init__(self, model, observer):
        self.model = model
        self.observer = observer

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, tokens, *args, **kwargs):
        self.observer.before_model_call(tokens.shape[1])
        return self.model(tokens, *args, **kwargs)
