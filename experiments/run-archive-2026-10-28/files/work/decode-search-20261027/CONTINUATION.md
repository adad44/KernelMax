# Active whole-decode search

Original goal: run a NEW candidate after identifying where the COMPLETE decode
step spends time; target a plausible saving of several ms/token instead of minor
expert kernel variants. Goal remains ACTIVE. A new candidate has NOT yet been
selected, created or run in this search.

The user explicitly expanded the contract on 2026-10-27 to permit complete decode
CPU/GPU scheduling and unnecessary-wait investigation; graph fusion across
normalization/routing/attention/experts; exact layouts, buffer reuse and KV-cache
updates that preserve precision and behavior. User has standing authorization
for local execution; do not ask for per-snapshot approval. Do not merge/publish.

## Authoritative checkout and changes

- Active experiment checkout: `/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop`.
- `kernelmaxxing.yaml`: experiment ID `gpt-oss-20b-m4-complete-decode`, boundary
  `complete_decode`; normalized hash `5fc5d80bf357fd02a8977b10da7896692c79aa9c30019ab85ba6296c9bd56af4`.
- Schema accepts explicit legacy `moe_expert_execution` and new `complete_decode`,
  still rejects arbitrary `whole_model` and typo scopes.
- AGENTS and candidates/README describe scopes and broader entry point, require
  full-model correctness, cache/route/token checks and no hidden extra work.
- Changed protected sources under explicit maintainer authorization. Fresh
  controller session is `session.json` HERE; original old optimization-loop
  session/history untouched and stale for the new contract. New source snapshot
  recorded. Do not recertify old benchmarks under this contract.
- 72 targeted unittest tests passed. Full isolated pytest: **345 passed, 8 optional
  MLX tests skipped**. `validation.json` retains the first unittest attempt that
  failed imports because pinned benchmark Python lacks pytest. Isolated uv deps
  were used; benchmark runtime/dependencies were not modified.

## Running profiling process — verify live, do not restart on a polling timeout

- launchd handle `gui/501/com.kernelmax.decodeprofile.cooled.20261027`.
- PID **71655**, process birth **Tue Oct 27 18:10:46 2026**.
- Script `cooled-profile/profile.py`; logs `cooled-profile/profile-stdout.log`,
  `cooled-profile/profile-stderr.log`; authoritative `cooled-profile/profile-state.json`.
- Last verified live at elapsed2:15, loading175/531 tensors. Re-poll THIS exact
  handle/PID. Native CPU-stream materialization typically takes ~6 minutes.
- Profile retains all data, uses pinned native checkpoint receipt, AC/nominal
  thermals/no swap-out or growth, full128outputs and fresh caches.
- Cooled cadence: fixed12s hold +5s settle +10s quiet before EVERY measured
  request. Five warmups then20A/A pairs with existing bootstrap gate. After a
  pass, exact-logit capture then5warmups+5alternating pairs per deletion region:
  experts, attention+cache GPU work, lm_head, small_ops. These are coarse,
  nonadditive optimistic planning budgets, NOT deployable candidates or GPU
  hardware shares. Per-step wall AND process CPU time separate build/eval.
- Three-hour profile cap. No statistical rerolls. No concurrent benchmarks.
- Previous continuous-cadence attempt in this parent directory failed thermal
  validity during A/A pair15, request34; no A/A or deletion result. It is retained
  with protected/checkpoint guards. Its job was unloaded. New cooled run is a
  distinct thermal intervention, not an unchanged reroll.

## Findings and next action

Fresh first attempt measured about **0.74ms Python graph build / 41ms total**,
readback ~0.002ms. Graph build ALONE is too small for the requested several-ms
goal. Eval/wait includes C++ scheduling, GPU execution and waiting; do not call
it GPU kernel time. Per-request process CPU included prefill and is not a
per-decode CPU share. Cooled run adds per-step CPU timing to resolve that gap.

Past optimization-loop evidence is authoritative but under old contracts: small
expert variants implied only~0.1ms/token model saving; one fresh20pair512model
test of cand_02e5f860f143 found no gain above A/A noise. Latest cand_35fff69ba1e1
was deadline-partial12pairs; do not finish/relabel it as the requested NEW candidate.

1. Inspect cooled run and measured CPU/region budgets. If CPU evaluation plus
   graph construction supports several ms of plausible unoverlapped headroom,
   test a NEW one-token asynchronous scheduling candidate. Graph-build-only
   0.74ms does not justify that claim. Otherwise target a measured large region
   with a concrete mechanism (larger fusion/exact memory execution now allowed).
2. `evaluate_schedule.py` here is a PREPARED, never-run full-model controller for
   a scheduling candidate. It has no candidate implementation/snapshot yet.
   Must adjust its per-request cooling cadence before freezing/running; currently
   it still uses continuous resident timing and will likely overheat. It loads
   source/receipt/session, freezes validation, traces every128logits/routes and
   final cache arrays/meta on512/2048/4096, counts exactly129forward calls
   (oneprefill+128singletoken), then fresh20pairA/A and20candidate pairs.
   It stops on environmental/statistical failure and includes no acceptance.
3. Candidate scheduling idea: queue next true greedy step on GPU while CPU
   builds following graph, exactly128outputs/no extra forward. Record first-token
   release and final GPU drain. Preserve all logits/routes/cache exactly.
4. Freeze new candidate with create_candidate_id/freeze_candidate, record actual
   source hash and hypothesis before measurements. Broader candidate cannot be
   certified by old expert-only evaluator; full-model diagnostic owns evidence.
5. If a broader candidate requires another controller, keep it external here;
   do not mutate existing profiler/source snapshots or old failed evidence.
6. Completion requires a new candidate actually executed after measured target
   selection and honest retained outcome. No verified improvement is required
   by wording, but a plausible several-ms mechanism is; do not substitute minor
   variants. Acceptance/task-quality remains disabled/planned. Audit then call
   update_goal(complete) only after the original objective is truly satisfied.

No subagents: current developer instructions forbid proactive delegation. No
background recurring automation has been created. No commit/push/merge occurred.

## Continuation turn: prepared timestamp observer, profiler still running

Previous goal turn was progress (scope/schema/tests/new live profile). This turn
also made progress: implemented an additional command-buffer timestamp probe
and tightened the prepared candidate evaluator. No NEW candidate yet.

Latest verified profiler: PID71655 / same birth, live at elapsed19:41, status
`aa_calibration`,23requests retained (five warmups+18A/A rows through pair9).
Every recorded cooled request so far had no environmental flags. Do not treat
incomplete A/A as a pass. `latest-progress.json` captures latest descriptive data:
median per-request mean build wall~0.679ms, build CPU~0.721ms, **eval CPU~5.100ms**,
eval/wait wall~40.286ms, readback~0.005ms. Process CPU includes other runtime
threads; CPU evaluation may already overlap GPU. This is a plausible broader
scheduling budget to investigate, not a proven recoverable5ms.

`timeline/` contains a NEW PREPARED observer and follow-up controller:

- `command_timeline.m` and compiled `command_timeline.dylib`: process-local
  Objective-C commit-method override records CPU submission times and retains
  command buffers. AFTER normal GPU drain it reads public MTLCommandBuffer
  GPUStartTime/GPUEndTime, then restores the exact original method. No completion
  callbacks, per-kernel callbacks, inserted GPU commands or between-stage syncs.
  Retaining command buffers can perturb memory; paired native observer/control
  measurements and environment remain necessary. These timestamps are whole
  command-buffer intervals, not kernel-active time or GPU hardware counters.
- C-only helper compiled successfully. First C++ build failed missing default
  libc++ headers (SDK has them; no environment was changed); switched to pure
  Objective-C. `build.json` keeps first failure; `build-v2.json` keeps successful
  build. Compilation was scheduled within the running profiler's fixed12second
  hold after a completed request, with timestamps retained, and finished<1second.
- `probe.py` bindings + interval-union arithmetic. Offline checks passed64byte
  ABI and overlapping intervals/gaps/empty timeline validation, recorded in
  `offline-validation.json`. This is NOT native observer validation.
- **Next after cooled profiler COMPLETES and its exact process has EXITED:**
  run pinned Python `-B timeline/validate.py` from this work directory. It refuses
  while PID71655 is the profiler, exercises a tiny GPU expression, requires real
  aligned GPU timestamps/exact output/completed status/method restoration, and
  records `timeline/validation.json` bound to actual helper source/binary hashes.
  Fix any observer error before loading another complete model.
- `timeline/run.py`, `timeline/plan.json`, `timeline/job.plist` prepared but **NOT
  started**. `run.py` refuses a still-live profiler or incomplete/failed profile,
  requires the current native observer fixture pass, then loads the native model
  once, collects three alternating native control/observer pairs with same cooled
  cadence and retains every command buffer/CPU step/environment observation.
  It restores observer method after EACH trace and stops on any failure.
- After valid probes, it waits at most20minutes while retaining the resident
  model for controller selection. Inspect `timeline/state.json`, choose a NEW
  candidate from CPU/region/command-gap evidence and freeze it. Only then create
  this parent's `run-selected-candidate.json` with `{"action":"run_schedule"}`
  plus optional descriptive identity; the actual evaluator independently reads
  `candidate-frozen.json`/`candidate-plan.json`. This hands the resident model to
  the prepared full-model evaluator, avoiding another6minute checkpoint load.
- Do NOT start a second benchmark while profiler/timeline/candidate is live.
  `timeline/job.plist` label `com.kernelmax.decodetimeline.20261027` is ready for
  manual launch only AFTER native observer validation. No recurring automation.

`evaluate_schedule.py` is still PREPARED/unexecuted, now updated BEFORE any run:
accepts `main(preloaded_model=model)`, uses fixed12s+5s+10s quiet before every
measured request, independently checks actual cache/logit BF16 dtype, finite
logits, all128trace entries and exact129forward shapes; keeps original full
logit/routes/token/cache-array/meta checks. Cap now5hours for all3cooled contexts.
Its source is included in the timestamp runner's protected snapshot, so do not
edit it AFTER starting that runner; finish review/preparation beforehand.

Native MLX binary/source inspection revealed configured command-buffer splitting
by operation count and referenced memory. Installed0.32.3 native source uses
~40operations/40MB defaults on base/pro. Large expert-weight inputs can trigger
many commits; actual counts/gaps are unmeasured until the timestamp probe.
Do not claim these are unnecessary waits or that larger buffers will save time
without measurement. Python-only build0.68ms is still insufficient for several
ms/token. Async scheduling could overlap some of ~5.1ms CPU evaluation, but
critical-path overlap must be tested. Avoid another minor expert kernel variant.
