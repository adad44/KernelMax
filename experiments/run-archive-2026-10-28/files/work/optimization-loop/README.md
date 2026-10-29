# Current optimization loop

Use the streamlined controller for subsequent candidates. No new model run was
performed to configure this loop. Prior run history is untouched and uncommitted.

```sh
cd /Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop
/Users/alandiaz/Downloads/KernelMaxxxing/runtime/gpt-oss/.venv/bin/python -B scripts/run_candidate_loop.py \
  --session /Users/alandiaz/Downloads/KernelMaxxxing/work/optimization-loop/session.json \
  --candidate candidates/cand_<12-hex-digits> \
  --work-dir /Users/alandiaz/Downloads/KernelMaxxxing/work/<new-attempt>/native-correctness
```

The first run hashes checkpoint bytes once. Both workers and later candidates
reuse that receipt only while pins and file identities match. Every changed
candidate still runs every required native correctness case. After a pass, the
paired timing controller consumes that exact native report without another
approval or repeated native session. Keep workload-specific real-input checks,
AC/thermal/paging validity, A/A calibration, raw samples and call-count checks.

Do not rerun parked candidates just to exercise this entry point. The next
candidate needs a measurable bottleneck and an estimated latency contribution.

Session source hashes cover the updated harness. Historical measurements retain
their own source snapshots. Formal acceptance remains disabled and no new
baseline certification or model speedup is claimed by this tooling change.

Repository workflow: ../../kernelmax-first-loop/scripts/CANDIDATE_LOOP.md
