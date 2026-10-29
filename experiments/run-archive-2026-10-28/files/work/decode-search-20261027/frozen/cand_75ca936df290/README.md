# One-step-ahead complete decode scheduling

Entry point: `decode_request(model, ids, step_size, mx, make_prompt_cache)`.
This is a complete decode scheduler and cannot be evaluated by the expert-only
`run(experts, x, ids)` adapter.

It calls the unchanged model for prompt prefill and128single-token outputs. After
releasing the first token, it queues the next true greedy step with `async_eval`
and builds the following graph before reading the pending token into Python.
Every queued step produces an emitted token; no extra forward or future token
is omitted from timing. The last interval includes the final GPU drain.

Weights, tensor layouts, routing, mathematical operations, cache types and
prefill handling are unchanged. Each request creates its own cache. The trusted
controller must verify every generated logit, expert route, greedy token, final
cache value/metadata and actual model-call count before matched timing. Numerical
or scheduling failures stop evaluation; this submission contains no verdict.
