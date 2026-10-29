# Local MLX-LM baseline

This harness measures the unchanged installed MLX-LM `stream_generate` path.
It does not patch runtime code, model weights, or KernelMax's evaluator.

Run from `/Users/alandiaz/Downloads/KernelMaxxxing` with other model sessions stopped:

```sh
caffeinate -i runtime/gpt-oss/.venv/bin/python -u baseline-tools/mlx_baseline.py \
  --prompt-lengths 512 2048 4096 \
  --generation-tokens 256 --trials 10 --seed 42 \
  --output baseline-results/2026-10-03-seed42
```

Use a new output directory for each run; existing directories are never overwritten.

## Workload and timing

- Batch size 1; default seed 42; greedy argmax sampling.
- One model load, excluded from trial timing. One full warmup per input length.
- A seeded 4,096-token uniform random prompt supplies nested prefixes for all lengths.
- Fresh KV cache for each request; no prefix cache, speculation, or activation quantization.
- EOS stopping disabled, matching the stock benchmark, so every request generates 256 tokens.
- Default 2,048-token prefill chunks. This is fixed across all lengths.
- Ten measured trials per length, rotating length order each round to reduce order bias.
- Synchronization before timing and after generation; no extra per-token synchronization.
- Raw prompt/output IDs, hashes, software versions, source hashes, checkpoint download metadata,
  battery/thermal/swap conditions, and all trials are retained in `results.json`.
- `run.log` records progress. `results.json` is updated atomically after each completed trial.

`prefill_tps` and `decode_tps` use MLX-LM's own definitions, for comparison with its stock
benchmark. `core_ttft_seconds` measures the first stream response, excluding model loading,
tokenization, server scheduling, and networking; it is not end-to-end serving TTFT.
`peak_memory_gb` is the per-request MLX active-allocation peak in decimal GB, not total
process/system RAM. Summary p10/p90 values use linearly interpolated sample percentiles.

This is a synthetic execution baseline, not a quality evaluation or a serving/concurrency
benchmark. Ten repeats of one prompt measure timing variability, not prompt diversity.
Add fixed realistic prompts and correctness checks before accepting an optimization.
Existing allocated swap is not evidence of active swapping; compare recorded cumulative
swap-in/swap-out counters between trials. Thermal warnings and background work can affect
results, so preserve the environment notes and repeat paired baseline/candidate tests.
