#!/usr/bin/env python3
"""Validate a complete baseline artifact and render its measured summary."""

import hashlib
import json
from pathlib import Path
import re
import sys


def counters(snapshot):
    text = snapshot["vm_stat"]["stdout"]
    return {name: int(re.search(rf"{name}:\s+(\d+)", text).group(1))
            for name in ("Swapins", "Swapouts")}


path = Path(sys.argv[1]).resolve()
report = json.loads(path.read_text())
assert report["status"] == "complete", "Benchmark is not complete"
configuration = report["configuration"]
lengths = configuration["prompt_lengths"]
trials = report["trials"]
assert len(trials) == len(lengths) * configuration["trials"]
for length in lengths:
    selected = [t for t in trials if t["prompt_tokens"] == length]
    assert len(selected) == configuration["trials"]
    assert sorted(t["trial"] for t in selected) == list(range(1, configuration["trials"] + 1))
    for trial in selected:
        assert trial["generation_tokens"] == configuration["generation_tokens"]
        assert len(trial["output_token_ids"]) == configuration["generation_tokens"]
        assert trial["finish_reason"] == "length"
        for key in ["prefill_tps", "decode_tps", "core_ttft_seconds", "total_seconds", "peak_memory_gb"]:
            assert trial[key] > 0
    prompt = report["prompts"][str(length)]
    assert len(prompt["token_ids"]) == length
    assert hashlib.sha256(json.dumps(prompt["token_ids"], separators=(",", ":")).encode()).hexdigest() == prompt["sha256"]

lines = ["# Measured GPT-OSS-20B MLX baseline", "",
         "Apple M4, 24 GiB unified memory; model stored on BankOfSouls.", "",
         f"Seed **{configuration['seed']}**; batch **1**; **{configuration['generation_tokens']} generated tokens**; "
         f"**{configuration['trials']} measured trials per length**, after one warmup per length.", "",
         "| Input tokens | Median prefill tok/s | Median decode tok/s | Decode range | Median core TTFT (s) | Peak MLX GB |",
         "|---:|---:|---:|---:|---:|---:|"]
for length in lengths:
    summary = report["summary"][str(length)]
    decode = summary["decode_tps"]
    lines.append(f"| {length:,} | {summary['prefill_tps']['median']:.2f} | {decode['median']:.2f} | "
                 f"{decode['min']:.2f}–{decode['max']:.2f} | {summary['core_ttft_seconds']['median']:.3f} | "
                 f"{summary['peak_memory_gb']['max']:.3f} |")
lines += ["", "## Reproducibility and limitations", "",
          f"- Packages: `{report['environment']['packages']}`.",
          f"- Model loading took {report['model_load_seconds']:.2f} seconds and is excluded from timing.",
          "- Synthetic uniform random token IDs, nested prompt prefixes; seed controls the workload.",
          "- Greedy sampling; fresh KV cache; no prefix cache, speculation, or activation quantization.",
          f"- Prefill chunk size {configuration['prefill_step_size']}; trial lengths rotate each round.",
          "- Core TTFT excludes loading, tokenization, API scheduling, and networking.",
          "- Peak memory is the MLX active-allocation peak (decimal GB), not total process/system memory.",
          "- Repeated output hashes verify repeatability of this baseline, not correctness of a future optimization."]
for length in lengths:
    selected = [t for t in trials if t["prompt_tokens"] == length]
    initial, final = selected[0]["decode_tps"], selected[-1]["decode_tps"]
    lines.append(f"- {length:,} tokens: first-to-last decode {initial:.2f} → {final:.2f} tok/s "
                 f"({100 * (final / initial - 1):+.1f}%); "
                 f"{report['summary'][str(length)]['distinct_output_hashes']} distinct measured output hash(es).")
first_counters = counters(trials[0]["conditions_after"])
last_counters = counters(trials[-1]["conditions_after"])
lines += [f"- System swap counter changes between the first and last measured-request snapshots: "
          f"{last_counters['Swapins'] - first_counters['Swapins']:,} swap-in pages; "
          f"{last_counters['Swapouts'] - first_counters['Swapouts']:,} swap-out pages. "
          "These are system-wide and include background/inter-trial activity, not isolated model I/O.",
          "- This is a working-machine baseline. Throughput drift and background memory activity "
          "mean small optimization wins need controlled paired reruns, not comparison with a single saved number.",
          "", "## Artifacts", "", "- [All raw trials and provenance](results.json)",
          "- [Progress log](run.log)", "- [Harness](../../baseline-tools/mlx_baseline.py)", ""]
summary_path = path.parent / "summary.md"
summary_path.write_text("\n".join(lines))
print("\n".join(lines))
print(f"Validated {len(trials)} measured trials. Saved {summary_path}")
