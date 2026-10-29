#!/usr/bin/env python3
"""Seeded, resident-model baseline using unmodified MLX-LM stream_generate."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def conditions():
    return {
        "battery": command("pmset", "-g", "batt"),
        "thermal": command("pmset", "-g", "therm"),
        "swap": command("sysctl", "vm.swapusage"),
        "vm_stat": command("vm_stat"),
        "memory_pressure": command("memory_pressure", "-Q"),
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/Volumes/BankOfSouls/models/gpt-oss-20b-mlx")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.prompt_lengths + [args.generation_tokens, args.trials, args.prefill_step_size]) < 1:
        parser.error("Lengths, trials, and prefill step size must be positive.")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("x")

    def announce(message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    model_path = Path(args.model).resolve()
    package_path = Path(mlx_lm.__file__).parent
    files = [model_path / "config.json", model_path / "model.safetensors.index.json",
             package_path / "benchmark.py", package_path / "generate.py",
             package_path / "models" / "gpt_oss.py", Path(__file__).resolve()]
    report = {
        "status": "loading", "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            **vars(args), "batch_size": 1, "sampling": "greedy_argmax",
            "warmups_per_length": 1, "kv_cache": "fresh each request; default precision",
            "prefix_cache": False, "speculative_decoding": False,
            "quantize_activations": False, "prompt_type": "synthetic_uniform_random_token_ids",
            "prompt_relationship": "prefixes of one seeded longest prompt",
            "trial_order": "rotating prompt lengths each round",
            "eos_stopping": "disabled as in mlx_lm.benchmark, forcing fixed generation length",
            "model_load_in_timing": False, "prefill_chunk_size": args.prefill_step_size,
        },
        "environment": {
            "python": platform.python_version(), "machine": platform.machine(),
            "macos": command("sw_vers"),
            "hardware": command("sysctl", "-n", "hw.memsize", "hw.model", "machdep.cpu.brand_string"),
            "packages": {p: importlib.metadata.version(p)
                         for p in ["mlx", "mlx-metal", "mlx-lm", "transformers", "numpy"]},
        },
        "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "checkpoint_files": [], "conditions_before": conditions(),
        "warmups": [], "trials": [],
        "metric_notes": {
            "prefill_tps": "MLX-LM prompt_tps; prompt evaluation through first-token sampling",
            "decode_tps": "MLX-LM generation_tps (stock definition)",
            "core_ttft_seconds": "First stream response; excludes loading, tokenization, server, and network",
            "total_seconds": "Synchronized entire stream iteration including generation cleanup",
            "peak_memory_gb": "Reset per trial; MLX allocator active-memory peak in decimal GB, not system RAM/RSS",
        },
    }
    for shard in sorted(model_path.glob("*.safetensors")):
        metadata = model_path / ".cache" / "huggingface" / "download" / (shard.name + ".metadata")
        report["checkpoint_files"].append({
            "name": shard.name, "bytes": shard.stat().st_size,
            "huggingface_download_metadata": metadata.read_text().splitlines() if metadata.exists() else None,
            "note": "Download metadata provenance, not a fresh full-shard checksum verification",
        })

    def save():
        temp = out / "results.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(out / "results.json")

    save()
    announce(f"Loading model once from {model_path}; load time is excluded from trials.")
    try:
        load_start = time.perf_counter()
        model, tokenizer, config = load(args.model, return_config=True,
                                        model_config={"quantize_activations": False})
        mx.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_start
        # Same EOS override as the installed upstream benchmark.
        tokenizer._eos_token_ids = set()
        mx.random.seed(args.seed)
        vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
        full_prompt = mx.random.randint(0, vocab_size, (max(args.prompt_lengths),)).tolist()
        prompts = {length: full_prompt[:length] for length in args.prompt_lengths}
        report["prompts"] = {str(k): {"token_ids": v, "sha256": digest(v)} for k, v in prompts.items()}
        report["status"] = "warming_up"
        announce(f"Model loaded in {report['model_load_seconds']:.2f}s; seed={args.seed}, greedy sampling.")
        save()

        def run(length, trial):
            mx.synchronize()
            mx.reset_peak_memory()
            mx.random.seed(args.seed)
            started = time.perf_counter()
            first_token = None
            output_tokens = []
            response = None
            for response in stream_generate(model, tokenizer, prompts[length],
                                            max_tokens=args.generation_tokens,
                                            prefill_step_size=args.prefill_step_size):
                if first_token is None:
                    first_token = time.perf_counter() - started
                output_tokens.append(response.token)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            if response is None or response.generation_tokens != args.generation_tokens:
                raise RuntimeError("Generation did not match the fixed output-token budget.")
            if len(output_tokens) != args.generation_tokens:
                raise RuntimeError("Stream token count did not match the output-token budget.")
            return {
                "prompt_tokens": length, "generation_tokens": response.generation_tokens,
                "trial": trial, "prefill_tps": response.prompt_tps,
                "decode_tps": response.generation_tps, "core_ttft_seconds": first_token,
                "total_seconds": elapsed, "peak_memory_gb": mx.get_peak_memory() / 1e9,
                "output_token_ids": output_tokens, "output_sha256": digest(output_tokens),
                "finish_reason": response.finish_reason,
            }

        for length in args.prompt_lengths:
            announce(f"Warmup: {length} input / {args.generation_tokens} output tokens.")
            report["warmups"].append(run(length, 0))
            save()
        report["status"] = "measuring"
        for index in range(args.trials):
            offset = index % len(args.prompt_lengths)
            order = args.prompt_lengths[offset:] + args.prompt_lengths[:offset]
            for length in order:
                result = run(length, index + 1)
                result["finished_utc"] = datetime.now(timezone.utc).isoformat()
                result["conditions_after"] = conditions()
                report["trials"].append(result)
                announce(f"Trial {index+1}/{args.trials}, input={length}: "
                         f"prefill={result['prefill_tps']:.2f} tok/s, "
                         f"decode={result['decode_tps']:.2f} tok/s, "
                         f"TTFT={result['core_ttft_seconds']:.3f}s, "
                         f"peak={result['peak_memory_gb']:.3f} GB.")
                save()

        report["summary"] = {}
        metrics = ["prefill_tps", "decode_tps", "core_ttft_seconds", "total_seconds", "peak_memory_gb"]
        for length in args.prompt_lengths:
            trials = [r for r in report["trials"] if r["prompt_tokens"] == length]
            summary = {k: {"median": statistics.median([r[k] for r in trials]),
                           "mean": statistics.mean([r[k] for r in trials]),
                           "p10": percentile([r[k] for r in trials], .1),
                           "p90": percentile([r[k] for r in trials], .9),
                           "min": min(r[k] for r in trials), "max": max(r[k] for r in trials)}
                       for k in metrics}
            summary["trials"] = len(trials)
            summary["distinct_output_hashes"] = len({r["output_sha256"] for r in trials})
            report["summary"][str(length)] = summary
        report["conditions_after"] = conditions()
        report["status"] = "complete"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        announce("COMPLETE: 30 measured trials saved to " + str(out / "results.json"))
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        announce("FAILED: " + repr(error))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
