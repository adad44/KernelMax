# First decode candidate

The installed MLX-LM 0.31.3 `QuantizedSwitchLinear` calls native `gather_qmm`,
then adds the selected expert's output bias. `SwitchGLU` does this independently
for up/gate before calling GPT-OSS's already compiled SwiGLU. This candidate
moves those two bias additions into the activation's compilation boundary.
The hypothesis is reduced GPU dispatch/intermediate overhead during decode;
weight bandwidth may dominate enough that the change provides no benefit.

Only batch-one, one-token, four-selected-expert BF16 decode is specialized.
The same packed MXFP4 weights/scales, expert IDs, clipping constants, SwiGLU
equation, and native down-projection are used. Nothing replaces or mutates a
model module. No weight concatenation, precision conversion, or tensor cache is
introduced. Other shapes explicitly use `experts(x, ids)` unchanged.

For the protected seven-case suite, `decode` exercises this implementation at
layers 0/12/23. The other six cases exercise the declared reference fallback;
passing those does not establish an optimized prefill path. Compiled floating
point behavior must still pass the existing evaluator tolerances. A complete
model/task comparison is separate from these operator checks.

Execution requires human approval of the exact frozen snapshot under the
reviewed-candidate evaluator policy. Candidate metadata is not approval.
No correctness result, benchmark result, acceptance or speedup is asserted.

After operator correctness passes, a controller must measure alternating
baseline/candidate trials on the designated M4, with warmups and synchronization
outside each timed interval. An operator diagnostic alone cannot establish
end-to-end speedup. The historical full-model baseline is not a matched
candidate trial. Task-quality and acceptance gates remain disabled/planned.
