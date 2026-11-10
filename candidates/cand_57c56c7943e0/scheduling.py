"""Queue one actual greedy step while the CPU observes its predecessor.

The trace hook retains tensor references only for correctness qualification.
The timed callable always emits exactly 128 tokens with 127 submissions.
"""
import time


def _decode(model, ids, step_size, mx, make_prompt_cache, outputs, capture):
    if not ids or step_size < 1 or outputs not in (8,128):
        raise ValueError('Locked decode supports nonempty prompts and 8/128-output qualification')
    cache=make_prompt_cache(model)
    prompt=mx.array([ids],dtype=mx.int32)
    mx.eval(prompt)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    start=time.perf_counter()
    remaining=prompt
    while remaining.shape[1]>1:
        n=min(step_size,remaining.shape[1]-1)
        model(remaining[:,:n],cache=cache)
        mx.eval([c.state for c in cache])
        remaining=remaining[:,n:]
        mx.clear_cache()

    logits=model(remaining,cache=cache)[:,-1,:]
    mx.eval(logits)
    mx.synchronize()
    prefill_end=time.perf_counter()
    token=mx.argmax(logits,axis=-1)
    mx.eval(token)
    token_ids=[token.item()]
    timestamps=[time.perf_counter()]
    if capture is not None:
        capture(0,logits,token,cache)

    pending_logits=model(token[:,None],cache=cache)[:,-1,:]
    pending_token=mx.argmax(pending_logits,axis=-1)
    if capture is not None:
        capture(1,pending_logits,pending_token,cache)
    mx.async_eval(pending_token)
    submissions=1
    for step in range(1,outputs):
        if step<outputs-1:
            next_logits=model(pending_token[:,None],cache=cache)[:,-1,:]
            next_token=mx.argmax(next_logits,axis=-1)
            if capture is not None:
                capture(step+1,next_logits,next_token,cache)
            mx.async_eval(next_token)
            submissions+=1
        token_ids.append(pending_token.item())
        timestamps.append(time.perf_counter())
        if step<outputs-1:
            pending_token=next_token
            pending_logits=next_logits

    last_observed=timestamps[-1]
    mx.synchronize()
    drained=time.perf_counter()
    timestamps[-1]=drained
    peak=mx.get_peak_memory()
    finite=bool(mx.all(mx.isfinite(pending_logits)).item())
    cache_dtypes=sorted({str(t.dtype) for c in cache for t in c.state})
    if not finite or cache_dtypes!=['mlx.core.bfloat16']:
        raise ValueError('Output/cache validation failed')
    if submissions!=outputs-1 or len(token_ids)!=outputs:
        raise ValueError('Unexpected queued steps or emitted tokens')
    return {'output_token_ids':token_ids,'cache_dtypes':cache_dtypes,
        'last_logits_finite':finite,'timing_observations':{
            'start':start,'prefill_end':prefill_end,'tokens_at':timestamps,
            'last_token_observed_at':last_observed,'gpu_drained_at':drained,
            'final_gpu_drain_ms':(drained-last_observed)*1000,
            'async_submissions':submissions,'peak_memory_bytes':peak}}


def decode_request(model,ids,step_size,mx,make_prompt_cache):
    from evaluator.native_baseline import metrics
    result=_decode(model,ids,step_size,mx,make_prompt_cache,128,None)
    observed=result['timing_observations']
    values,intervals=metrics(len(ids),observed['start'],observed['prefill_end'],
        observed['tokens_at'],observed['peak_memory_bytes'])
    result.update(metrics=values,inter_token_intervals_ms=intervals)
    return result


def trace_request(model,ids,step_size,mx,make_prompt_cache,outputs,capture):
    if capture is None:
        raise ValueError('Qualification requires an explicit capture hook')
    return _decode(model,ids,step_size,mx,make_prompt_cache,outputs,capture)
