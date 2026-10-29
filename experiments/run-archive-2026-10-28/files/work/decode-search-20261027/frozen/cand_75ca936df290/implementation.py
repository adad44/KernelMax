"""Complete native-model decode with one true greedy step queued ahead.

There is no speculative or discarded forward. The only change is when the CPU
submits model graphs and observes tokens. Numerical operations remain native.
"""
import time


def decode_request(model, ids, step_size, mx, make_prompt_cache):
    from evaluator.native_baseline import metrics

    if not ids or step_size < 1:
        raise ValueError('A nonempty prompt and positive prefill step are required')
    cache = make_prompt_cache(model)
    prompt = mx.array([ids], dtype=mx.int32)
    mx.eval(prompt)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    start = time.perf_counter()
    remaining = prompt
    while remaining.shape[1] > 1:
        n = min(step_size, remaining.shape[1] - 1)
        model(remaining[:, :n], cache=cache)
        mx.eval([c.state for c in cache])
        remaining = remaining[:, n:]
        mx.clear_cache()

    # Same first-token boundary as the synchronous reference.
    logits = model(remaining, cache=cache)[:, -1, :]
    mx.eval(logits)
    mx.synchronize()
    prefill_end = time.perf_counter()
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    token_ids = [token.item()]
    timestamps = [time.perf_counter()]

    pending_logits = model(token[:, None], cache=cache)[:, -1, :]
    pending_token = mx.argmax(pending_logits, axis=-1)
    mx.async_eval(pending_token)
    submissions = 1

    for step in range(1, 128):
        if step < 127:
            # The next graph depends on the actual pending GPU token, without
            # waiting for Python to read that token back first.
            next_logits = model(pending_token[:, None], cache=cache)[:, -1, :]
            next_token = mx.argmax(next_logits, axis=-1)
            mx.async_eval(next_token)
            submissions += 1
        token_ids.append(pending_token.item())
        timestamps.append(time.perf_counter())
        if step < 127:
            pending_token = next_token
            pending_logits = next_logits

    last_token_observed_at = timestamps[-1]
    # All submitted work, including the final drain, belongs to this request.
    mx.synchronize()
    gpu_drained_at = time.perf_counter()
    timestamps[-1] = gpu_drained_at
    peak = mx.get_peak_memory()

    # Validation runs outside the measured generation interval, like reference.
    finite = bool(mx.all(mx.isfinite(pending_logits)).item())
    cache_dtypes = sorted({str(t.dtype) for c in cache for t in c.state})
    if not finite or cache_dtypes != ['mlx.core.bfloat16']:
        raise ValueError('Native output/cache check failed')
    if submissions != 127 or len(token_ids) != 128:
        raise ValueError('Unexpected queued-step or emitted-token count')
    values, intervals = metrics(len(ids), start, prefill_end, timestamps, peak)
    return {
        'metrics': values,
        'inter_token_intervals_ms': intervals,
        'output_token_ids': token_ids,
        'cache_dtypes': cache_dtypes,
        'last_logits_finite': finite,
        'timing_observations': {
            'start': start,
            'prefill_end': prefill_end,
            'tokens_at': timestamps,
            'last_token_observed_at': last_token_observed_at,
            'gpu_drained_at': gpu_drained_at,
            'final_gpu_drain_ms': (gpu_drained_at-last_token_observed_at)*1000,
            'async_submissions': submissions,
            'peak_memory_bytes': peak,
        },
    }
