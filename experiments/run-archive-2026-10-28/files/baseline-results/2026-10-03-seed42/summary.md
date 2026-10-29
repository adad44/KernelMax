# Measured GPT-OSS-20B MLX baseline

Apple M4, 24 GiB unified memory; model stored on BankOfSouls.

Seed **42**; batch **1**; **256 generated tokens**; **10 measured trials per length**, after one warmup per length.

| Input tokens | Median prefill tok/s | Median decode tok/s | Decode range | Median core TTFT (s) | Peak MLX GB |
|---:|---:|---:|---:|---:|---:|
| 512 | 213.36 | 26.71 | 24.59–32.09 | 2.573 | 12.502 |
| 2,048 | 242.64 | 25.37 | 23.26–30.62 | 8.590 | 12.859 |
| 4,096 | 242.67 | 25.35 | 23.72–30.11 | 17.054 | 12.922 |

## Reproducibility and limitations

- Packages: `{'mlx': '0.32.3', 'mlx-metal': '0.32.3', 'mlx-lm': '0.31.3', 'transformers': '5.17.0', 'numpy': '2.5.3'}`.
- Model loading took 303.15 seconds and is excluded from timing.
- Synthetic uniform random token IDs, nested prompt prefixes; seed controls the workload.
- Greedy sampling; fresh KV cache; no prefix cache, speculation, or activation quantization.
- Prefill chunk size 2048; trial lengths rotate each round.
- Core TTFT excludes loading, tokenization, API scheduling, and networking.
- Peak memory is the MLX active-allocation peak (decimal GB), not total process/system memory.
- Repeated output hashes verify repeatability of this baseline, not correctness of a future optimization.
- 512 tokens: first-to-last decode 32.09 → 24.59 tok/s (-23.4%); 1 distinct measured output hash(es).
- 2,048 tokens: first-to-last decode 30.62 → 23.26 tok/s (-24.0%); 1 distinct measured output hash(es).
- 4,096 tokens: first-to-last decode 30.11 → 24.94 tok/s (-17.2%); 1 distinct measured output hash(es).
- System swap counter changes between the first and last measured-request snapshots: 27,108 swap-in pages; 7,293 swap-out pages. These are system-wide and include background/inter-trial activity, not isolated model I/O.
- This is a working-machine baseline. Throughput drift and background memory activity mean small optimization wins need controlled paired reruns, not comparison with a single saved number.

## Artifacts

- [All raw trials and provenance](results.json)
- [Progress log](run.log)
- [Harness](../../baseline-tools/mlx_baseline.py)
