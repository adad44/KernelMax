# Opt-in confirmation under desktop load

The checked-in `kernelmaxxing.yaml` retains the original strict policy. Its
normalized contract identity remains unchanged. Legacy runners continue using
strict assessment unless they explicitly select another policy.

For a separate desktop-load experiment, freeze a new contract with:

```yaml
measurement:
  confirmation:
    swap_policy: retain_and_flag_desktop_load
    require_fresh_matched_comparisons: true
benchmark:
  invalidate_on_swap_growth: false
  invalidate_on_thermal_change: true
```

All other locked model, precision, numerical, prompt/output, pairing, control,
repeatability and bound requirements remain. Freeze the trusted controller and
new contract before collecting fresh timing. The controller must explicitly pass
the contract's confirmation policy to `assess_environment`; changing the schema
does not activate this mode in existing runners.

Safari and ordinary background applications may remain open. Keep that workload
approximately consistent between implementations and record substantial known
changes. Retain all samples, paging flags, counts/rates, memory-pressure data,
allocator usage and available process-memory observations. System-wide paging
does not identify which model or process pages were swapped.

Under this mode, swap-ins, swap-outs and swap growth are flags rather than stop
reasons or automatic sample exclusions. AC/thermal failures and unexpected
counter resets remain hard assessment failures; numerical, generation, runtime,
integrity and declared request/time failures remain controller hard stops.

Analyze all predeclared eligible control/comparison observations. Keep failed
statistical controls and comparisons and report an inconclusive result; do not
choose a clean subset or repeat unchanged runs until favorable. Warmups and any
instrumented diagnostics have their roles fixed before collection.

A supported performance claim applies under the recorded desktop workload,
including Safari and observed paging. It is not an isolated-machine or
paging-free claim. Preserve the original reference, historical strict artifacts
and smaller independently verified checkpoints. Keeping a useful checkpoint is
separate from reaching the overall 5% goal, and neither implies task-quality
acceptance.
