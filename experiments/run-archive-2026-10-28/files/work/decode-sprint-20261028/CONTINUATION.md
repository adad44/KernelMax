# Decode sprint continuation

## Objective and authority

Active goal: achieve repeatable full-model decode improvement of at least 5%, aiming for 10–20% through fusion, with at most three substantive candidates and no broad profiling. A short or noisy screen is exploratory and cannot establish the goal. Luna high owns only prepared procedural execution, monitoring, and artifacts; primary owns code design and interpretation. Do not mark the goal complete based on local signal.

Schema 3 is active. Exploration retains and flags swap/noisy samples within prepared bounds. Confirmation is strict: complete numerical checks and fresh matched measurements with swap growth rejected. Acceptance remains disabled. No broad profiling is authorized.

## Immutable inputs and candidate slots

- Contract normalized SHA256: `2a31bbda3389e91f12e1e17b1cff261b96c5f12cd3ce9af5dc818fd21a50c895`.
- Runner SHA256: `a9116f4a37d559037e58d5d519e8bdb16911a368f8ae22452fdc6b28b7200367`.
- Plan SHA256: `dbd808ffe1ec0f46521d5d635e1fc2fe4b01fd918dd1631916a474663b9b574d`.
- Session SHA256: `e5d96b0a1b2fb86e7e77f5df567c2bc69f29af477935ceb799a4f9e4d4b7354d`.
- Vocabulary slot 1: `cand_5785998944e5`, frozen bundle `b6dd528b50b4d3d89db496fb021bbcd21d2043f701c36196e39ea08457283918`.
- MoE slot 2: `cand_4a02c3189b7d`, frozen bundle `e4c96fc022079e58d7987f1bec0bdd4defb36589b4fc303efc5a41a5b76e1ed3`.
- Slot 3 is unallocated. Two unique candidates have been counted; C2 has not been executed or correctness tested.

## Live resident worker (last verified 2026-10-28)

Launchd label `com.kernelmax.sprint.vocab.20261028`, PID 48619, born Oct 28 10:57:36 local. One-off sleep assertion: `com.kernelmax.sprint.vocab.caffeinate.20261028`, PID 48639, running `caffeinate -i -s -w 48619`. Worker loads the model once and waits between sequential command files, with a 10-hour total cap. Do not start another GPU worker.

`commands/001.json` is the authorized vocabulary-only screen. Last observation: resident state `screen:cand_5785998944e5`, model load 531/531; short 512-prompt/8-token full-model numerical check passed bit-exact for logits, cache metadata and tolerance, with no route/token differences. The first phase-memory request was observed; screen trials had not yet completed. No reported errors. Continue monitoring `resident-state.json`, command state, logs, launchd/PID birth, and guards. Preserve all samples and flags.

After C1 screen ends, create `commands/002.json` for the frozen MoE candidate only if worker is authoritatively `waiting_command`, source/checkpoint/snapshot/method-restoration guards pass, and no hard GPU, integrity, thermal, or power failure occurred. This C2 screen is explicitly authorized by primary. If only C1 numerical check fails while worker/guards remain healthy, report before advancing; do not advance on GPU/integrity uncertainty. Do not write `003` or any confirmation command without primary instruction.

## Screen and confirmation scope

Each screen is exploratory: 8-token full-model correctness, two instrumented requests excluded from timing, then one warmup per lane and three alternating 128-output pairs at prompt 512. Retain and flag swap/noise. Report ratios, ms/token, sample flags, all correctness differences and guards; do not claim speedup or 5% progress from a screen.

A confirmation requires full 128-logit/route/token/cache numerical checks on contexts 512, 2048, and 4096; fresh 20-pair A/A calibration plus 20 candidate pairs using ABBA; and an independent second 512 block. Strict swap-growth rejection applies. For a 5% claim, each context's paired median ratio must be at least 1.05 and its bootstrap 95% lower bound above 1.05, with the second 512 block meeting the same threshold and exceeding fresh A/A noise. Primary interprets both screen results and decides whether to compose slot 3 or prepare a strict confirmation. No such action is automatic.

The sequential resident controller accepts prepared `{mode: "screen"|"confirm", frozen_info: absolute_path}` commands; `finish` is for graceful closure when primary authorizes it. Keep runner, plan, session, protected sources, and frozen snapshots unchanged while resident worker is live.

## Latest verified milestone

C1 timing began after the passing short check and two phase-memory rows. Warmups were baseline 24.366 and candidate 25.573 tok/s. The first timed pair was baseline 24.358 tok/s with +0.19 MiB swap growth and 1,548 swapout pages (flagged and retained under exploration), versus candidate 26.403 tok/s with no swapouts. This single flagged pair is not interpreted as speedup. PID 48619 remains live; the C1 screen continues.

C1 second pair is also flagged and retained: candidate 25.699 tok/s, +98.87 MiB swap growth and 30,908 swapout pages; baseline 23.932 tok/s, zero swapouts. No hard error was reported. The third pair and final guards are pending; the screen remains exploratory.

C1 command 001 completed with `completed_exploratory_screen`; worker reached `waiting_command` for 002. Its three paired ratios were [1.08396, 1.07384, 1.10803], median 1.08396, bootstrap interval [1.07384, 1.10803], and paired saved latency [3.180, 2.873, 4.131] ms/token. Three of six timed rows had swap growth/swapouts (the others had zero swapouts); all thermal states were 0 and power was AC. Short numerical check passed bit-exact. Protected source, checkpoint/snapshot, and method restoration guards passed; no errors. `five_percent_improvement_verified` remains false, and this is not confirmation.

After those guards passed and the resident worker was verified waiting, the explicitly authorized C2 screen was queued as `commands/002.json`, pointing to `moe-frozen.json` (candidate `cand_4a02c3189b7d`). Queue SHA256 `aa017139f666188897ebf78164a64f605e35b11266142d175d81b06c9a637348`. C2 is not yet executed or correctness tested. Do not write a third command or confirmation command; primary must interpret both screens and decide next steps.

Primary's optional future composition scaffolding was checked without preparation or ID allocation during an inter-request hold: isolated Ruff 0.17.3 formatted only `prepare_combined.py`; AST passed for it and `combined_implementation.py.in`. Hash record is `combined-preparation-check.json`. Slot 3 remains unallocated; do not run the preparer without primary's later decision.

C2 has passed its short check and both phase-memory rows; warmups and first pair show no swapouts. Pair 1: baseline 24.412, candidate 24.862 tok/s. Pair 2 candidate: 24.810 tok/s; its baseline half and final pair remain pending.

C2 command 002 completed with `completed_exploratory_screen`; short numerical check passed bit-exact with no route/token/cache differences. Three timed pairs were unflagged, 0/6 swap-affected, with thermal 0 and AC power. Ratios [1.01845, 1.01633, 1.02034], median 1.01845, bootstrap interval [1.01633, 1.02034], and saved latency [0.742, 0.658, 0.818] ms/token. No errors; protected, checkpoint/snapshot, and original-method-restoration guards passed. `five_percent_improvement_verified=false`.

Both raw command records, per-row timing/environment/swap fields, correctness summaries, guard results, and before/after hashes are indexed in `screen-results.json`. Both screens are exploratory; C1 has three swap-affected rows and C2's unflagged signal is about 1.8%. Resident PID 48619 remains live at `waiting_command` for 003, with no 003 file queued. Primary must interpret the two results and explicitly decide whether to prepare slot 3 or strict confirmation. No goal-completion or speedup claim is supported here.

Primary authorized the third and final snapshot after reviewing C1/C2 screens. Prepared `prepare_combined.py` was run once under the pinned Python while the resident worker waited; it allocated `cand_3f4c02fcfc71`, frozen bundle SHA256 `fe5ae0dd947e383f92c4ea10abd884838551ba74f6da205381e707935d3caaf5`. Both parent provenance sets were verified byte-for-byte; no new arithmetic was added. Exactly three IDs are allocated, and no fourth candidate/variant is authorized.

After verifying C2's completed screen, passing protected/checkpoint/snapshot/method-restoration guards, and the same resident PID waiting for command 003, queued only the authorized combined exploratory screen in `commands/003.json` (SHA256 `ce7b645058ba10522360f2a5138921810931b7cc2fd6a466041b7fd76098a163`). C3 has not yet been correctness-tested or timed. Continue monitoring its short numerical result and three pairs. Do not queue confirmation; primary reviews the full outcome first.

C3 command 003 completed healthy and met the primary's confirmation-selection rule. It was selected for the single prepared strict confirmation: 8-token correctness passed, median screen ratio 1.101605 (bootstrap interval [1.100149, 1.103459]), all six timed rows unflagged, all three command guards true, no errors, and resident worker verified waiting for 004. Selection record: `confirmation-selection.json`. Strict confirmation command 004 points to `combined-frozen.json`, SHA256 `e6ae71d0c012eef8ecc00e0ee145912af3081dfbd7af6a8cbc00dfe0ed780e9f`. No 5% claim is justified until the full confirmation/calibration/guards complete; do not queue another confirmation or finish command.

## Strict confirmation outcome

C3 confirmation command 004 failed and stopped before timing. Prompt 512 passed the full 128-output check exactly for logits, routes, tokens and cache. Prompt 2048 failed: 14 route-set/slot differences, 0 token differences, max logit absolute error 0.75 with 200,071/25,739,264 tolerance violations, cache metadata equal but cache values outside tolerance (maximum listed cache absolute error 1.25). Prompt 4096 was not attempted. No A/A calibration, timed rows, or summaries were run. The correctness NPZ token arrays were equal for both attempted contexts; there were zero timed confirmation requests, so request-level output-token/forward-count/placement audits do not apply.

Failure state is retained as `stopped_NumericalFailure`. Protected source, checkpoint/snapshot, and original method restoration guards passed. Worker PID 48619 (birth 10:57:36 local) remains live at `waiting_command` for 005, with caffeinate PID 48639 attached. No retry or further confirmation command is queued. `confirmation-audit.json` records the raw failure hashes and status. The repeatable 5% objective remains unproven; there is no official verdict or acceptance claim. Primary must interpret and decide closure.

Primary parked C3 and C2 for confirmation after C3's substantive 2048-context mismatch. C3 raw confirmation state and failure are preserved in `command-004-confirm-cand_3f4c02fcfc71/state.json` and `confirmation-audit.json`. After verifying the same PID 48619 waiting for command 005 and all C3 guard flags true, queued the primary-authorized fresh strict confirmation for already-screened C1 vocabulary-only candidate `cand_5785998944e5`. This is a distinct candidate from failed C3, not a C3 reroll. Selection/disposition records: `confirmation-selection-005.json`, `confirmation-disposition-005.json`; queue hash `c63740be3bb8206cc3101161293fdce546926cce82870f677ecf4f1b750fc53d`. No frozen code changed and no fourth candidate was allocated.

C1 strict confirmation command 005 has now passed full 128-output contexts 512 and 2048: exact logits, zero route/token differences, cache within tolerance. Prompt 4096 remains pending; no AA or timed requests have begun. Same resident PID 48619 remains live on command 005.

Optional post-run diagnostic `fusion-route-differences.json` reads only the C3 failed 2048 NPZ's two route arrays. There were 9 step/layer vectors with any route-ID order/set difference (14 individual slots); first vector difference was step108/layer18 with identical ID sets reordered, while first differing set was step109/layer15, reference [1,22,19,3] vs candidate [22,12,19,3]. Root cause is not established; stable tie behavior is only a hypothesis. This CPU-side artifact did not run a GPU probe or alter any source.

Current confirmation is command 005 for C1 vocabulary-only, not C3. C1 has passed full 128-output numerical checks at all three contexts; its 512 A/A group is now in warmup/pairing. C3's separate 004 confirmation failed at 2048 with route/cache/logit tolerance violations and remains parked. The C3 exploratory ~10.16% screen was never confirmation and must not be conflated with C1 strict evidence.

C1 confirmation 005 passed all three 128-output numerical checks at 512/2048/4096 with exact logits/routes/tokens and cache within tolerance. It then stopped during the third 512 A/A warmup: system-wide swap grew 210.31 MiB and swapouts increased by 14,484 pages. Strict policy stopped immediately; only five A/A warmup requests were recorded, with zero A/A pairs, calibrations, or candidate timing rows. C1 command status is `stopped_EnvironmentFailure`; no retry/finish/new command is queued. Protected/checkpoint/snapshot/method restoration guards all passed. Worker PID 48619 remains live at `waiting_command` for 006; caffeinate PID 48639 is attached. Detailed audit: `confirmation-audit-005.json`. The 5% objective remains unproven. Primary must decide whether to close the resident worker.

## Resident retirement and protocol fork

After command 005 stopped on strict system-wide swap growth, primary authorized graceful closure. Command 006 `finish` was consumed while waiting; old resident PID 48619 and attached caffeinate PID 48639 both exited 0. Final resident protected/checkpoint guards are true with no errors. Original raw states and artifacts remain in this sprint directory.

A fresh sibling `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-confirm-resident-20261028/` now contains byte-identical copies of the old runner (`run_confirm.py`), plan (`plan-before.json`), and session (`session-before.json`), plus `correctness-parent.json` with references to C1's three exact 128-output check NPZs and hashes, candidate/contract/checkpoint identity, old protected hashes and command 005 stop evidence. This sibling is preparation only: no model process was started. Primary must patch and review this copy's new residency/primer protocol before any execution.
