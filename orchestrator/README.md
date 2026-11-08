# Orchestrator

The proposed [search-policy improvement loop](SEARCH_POLICY_DESIGN.md) describes
how a future outer loop could evaluate better experiment-selection strategies.
Its first two foundation steps now have session accounting and a scheduling
deadline. Policy generation, policy comparisons, promotion and rollback remain
proposals; this code establishes no measured policy or inference improvement.

`run_candidate` connects candidate validation/freezing, evaluator handoff, the
experiment budget, the state machine and the local journal. The runner does not
import or execute submission files. It passes the controller-created
`RunIdentity`, read-only `FrozenCandidate`, and validated `ExperimentContract` to
an injected evaluator matching `TrustedPassThroughEvaluator`'s interface.

`run_session` processes an ordered list of submissions through that runner and
shares one `ExperimentBudget` across the whole session. Candidate IDs are unique
within a session: later occurrences are marked `duplicate` and never evaluated.
Invalid submissions and ordinary evaluator failures are reported per submission;
the session proceeds to the next distinct submission while the shared budget
allows. Journal failures, unexpected runner exceptions, and invalid evaluator
report or diagnostic identities stop the shared budget. Later unique submissions
are skipped with that stop reason. A failed final journal write retains the
returned report in memory and does not claim that its terminal state was saved.
When no attempts remain, later submissions are marked `budget_exhausted` without
being frozen or evaluated. The session summary distinguishes an explicit
`ACCEPTED` evaluator verdict from a completed run with no verdict. The latter is
reported as `completed_no_verdict`, so diagnostic success is never presented as
candidate approval.

The runner validates returned report provenance against the reserved identity,
keeps that exact `EvaluationReport` object, lets the budget observe it, and
records the transition in the journal. It does not write official report files;
those remain evaluator-owned. The trusted pass-through evaluator returns
diagnostic evidence only: `on_diagnostic_passed` is journaled as a signal but
does not create `CORRECTNESS_PASSED`, run candidate benchmarking, or claim that
the submission produced the checked outputs. Candidate execution still needs a
separate isolated evaluator/runtime.

The journal appends each transition, evaluator diagnostic signal and captured
failure to `history.jsonl`. It atomically replaces one state snapshot per
experiment under `states/`, syncing both file and parent directory. Its hashed
state filenames avoid treating experiment IDs as paths. Journal files are
controller-owned and should be placed outside agent-writable candidate space.

## Bounded search sessions and costs

Extend the existing trusted `run_session` call with an elapsed-time limit:

```python
from orchestrator.session import run_session

result = run_session(
    candidate_dirs,  # Finite, trusted iterable of already selected directories.
    evaluator=evaluator,  # Existing trusted/injected evaluator with its approvals.
    contract=contract,
    budget=ExperimentBudget(max_attempts=5, max_retries_per_candidate=1),
    journal=journal,
    frozen_root=frozen_root,
    candidates_root=candidates_root,
    max_elapsed_seconds=120,
    configuration_ids={
        "generator": "externally-recorded-generator-version-or-hash",
        "evaluator_configuration": "externally-recorded-config-hash",
        "controller_revision": "externally-recorded-git-commit",
    },
)
print(result.summary())
accounting = result.to_dict()
```

The fixed strategy is `ordered_unique_submissions_v1`: input order, first
occurrence of each candidate ID, no reranking. One shared monotonic deadline
and the supplied `ExperimentBudget` apply across the session, including retries;
previously consumed attempts are not reset. `max_elapsed_seconds=None` preserves
the original behavior with attempt limits only. A unique `search_id` identifies
each invocation even when a caller reuses a display `session_id`; frozen files
are namespaced by this ID under the controller's frozen root. Use a fresh budget
for an independent search. Reusing a budget intentionally shares its limits.

There is no candidate-generation or automatic transient-error classifier in the
current repository. Default behavior continues to move to the next distinct
candidate after failures. A trusted controller may pass
`retry_classifier=its_existing_classifier`, a callable receiving an unchanged
failed `EvaluationReport` and returning an actual bool. The existing
`ExperimentBudget.can_retry/start_retry` rules remain authoritative: only
observed failed reports explicitly classified as transient can retry, each
retry consumes a total attempt, and completed verdicts/interrupted runs cannot
retry. Exceptions without an evaluator report do not gain a synthetic report
or a retry. Classification errors stop the session and preserve prior evidence.
Retries reuse the same frozen files/hash with a fresh run ID through
`run_candidate(retry_of=..., retryable=True)`; live submissions are not refrozen.

The deadline is a **scheduling boundary, not a hard runtime cutoff**. The
controller checks it before submissions and between stages/retries; it never
terminates Python threads. For the existing `ReviewedCandidateEvaluator`, it
uses an immutable copy whose per-worker timeout is capped at half the remaining
session time, because reference and candidate workers run sequentially. Its
existing process-group cleanup still handles worker timeouts. Model hashing,
setup, cleanup, finite iterable consumption and arbitrary injected Python
evaluators lack safe cancellation and can overrun the deadline. Late results
are retained with `within_deadline=False`; no late result can improve a score.

The returned immutable observations retain all attempts, original report
objects, relative start/finish offsets and controller stage durations. Existing
run transitions, evaluator reports and verdicts stay unchanged. The append-only
journal gains `kind="search_session"` records with `session_event` values
`started`, `submission` and `completed`. They link the session/search IDs,
fixed strategy identity/hash, contract hash, workload seed, available external
configuration identities, evaluator/run/candidate identities, observed report
environment, retry classifications, remaining attempt budget and stop reason.
Run report contents remain in the existing transition/signal records; official
reports remain evaluator-owned. Session-journal failures stop new work and are
retained in `result.journal_error`, including final-flush failures. Skipped and
duplicate submissions remain in the final session summary.

Elapsed search time begins on entry to `run_session` and includes collection,
validation, freezing, evaluator calls, controller decisions, retries, failures
and journaling. The runner reports `validation`, `freezing`, `evaluation_call`
and `runner_total` durations. **Do not sum runner_total with its components.**
Profiling, compilation, warmups and benchmarking inside an evaluator call are
included in that aggregate wall time, but their individual durations are
unavailable and remain null. Work performed by a trusted iterable is charged to
`submission_collection`; this is not a new generation API. Pre-generated
candidate costs and contract/environment initialization before entry remain
outside the boundary. Generation resource usage and unavailable provenance are
null, never invented zero costs. Accounting is explicitly
`partial_search_boundary`, not complete end-to-end search cost.

The returned elapsed total includes the final journal flush. The completed
journal record captures the pre-flush total: recording the cost of that very
write would require another write. This distinction is explicit in accounting.
All of these durations are search-level wall time, separate from timed model
inference samples, and cannot establish GPU performance.

Sessions are currently **unscored**: `score` and `best_run_id` are null with a
reason. The report schema supports only planned task-quality execution, the
locked contract disables acceptance with provisional gates, and no authoritative
search-score aggregation exists. Even a synthetic ACCEPTED verdict is preserved
as an evaluator outcome without becoming a verified best search result. No
reference score of 1.0 is invented for incomplete diagnostic searches.

Remaining prerequisites from `SEARCH_POLICY_DESIGN.md` include isolated native
candidate execution, complete model/performance/quality integration, calibrated
acceptance and workload aggregation, and complete search cost/provenance
coverage. Versioned policy proposals, paired validation/confirmation searches,
statistical comparisons, promotion and rollback are later steps. This controller
grants no additional candidate-agent write access or approval authority.

Run the fake-clock session and existing orchestration regressions offline:

```sh
python -B -m unittest \
  tests.test_bounded_search_sessions tests.test_orchestrator_session \
  tests.test_orchestrator_run tests.test_orchestrator_journal \
  tests.test_experiment_budget tests.test_orchestrator_cli -v
python -B -m pytest -q -p no:cacheprovider
```

Use an existing Python 3.12 environment with NumPy and PyYAML (plus pytest for
the full suite). The new tests use the real runner and controlled submissions,
fake evaluators and an injectable clock; they download no models, execute no
optimization candidates and require neither MLX nor M4 hardware.

If an append is interrupted, the next read or append preserves the unfinished
final bytes in a private `history-incomplete-*.bin` recovery file before trimming
that tail. Complete history records are preserved. Recovery holds the journal
lock and syncs the recovery file and its directory before truncating history;
if preservation fails, the original history remains intact and the operation
raises `JournalError`. Newline-terminated corrupt records remain visible as
errors rather than being silently removed.

A submission with an invalid candidate ID gets a `submission_rejected` history
entry containing its experiment ID, supplied path/name, and error. Because it
cannot form a valid candidate state, it has no state snapshot; the returned
result retains `state=None`. No freezing or evaluator budget is used.

## Offline example

This example uses NumPy fake experts and the real trusted pass-through evaluator.
It needs no model download, MLX installation, or candidate execution. Save the
code as `offline_orchestrator_example.py` in the repository root and run it with
the evaluator's NumPy and PyYAML dependencies installed:

```python
import json
import tempfile
from pathlib import Path

from evaluator.pass_through import TrustedPassThroughEvaluator
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import create_candidate_id
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.run import run_candidate
from target.adapter import TrustedExpertAdapter


def expert(x, selected_expert_ids):
    return x[:, :, None, :] + selected_expert_ids[:, :, :, None].astype(x.dtype)


def reference(layer_index, x, selected_expert_ids):
    # Real use binds a trusted native expert wrapper for the given layer.
    return expert(x, selected_expert_ids)


contract = load_contract("kernelmaxxing.yaml")
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    candidates = root / "candidates"
    candidate_id = create_candidate_id()
    candidate = candidates / candidate_id
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text(json.dumps({
        "candidate_id": candidate_id,
        "name": "Offline diagnostic fixture",
        "implementation_type": "mlx",
        "description": "Provenance only; no submission code is executed",
    }))
    for filename in ("wrapper.py", "implementation.py"):
        (candidate / filename).write_text(
            'raise AssertionError("submission code was imported")\n'
        )

    adapter = TrustedExpertAdapter(
        {layer: expert for layer in (0, 12, 23)},
        hidden_size=8,
        num_experts=32,
    )
    evaluator = TrustedPassThroughEvaluator(reference, adapter)
    journal = ExperimentJournal(root / "journal")
    result = run_candidate(
        candidate,
        evaluator=evaluator,
        contract=contract,
        budget=ExperimentBudget(max_attempts=1),
        journal=journal,
        frozen_root=root / "controller" / "frozen",
        candidates_root=candidates,
    )

    print(result.state.phase.value)
    print(f"diagnostic cases: {len(result.report.correctness)}")
    print(f"benchmark samples: {len(result.report.performance)}")
    print(f"official verdict: {result.report.verdict}")
    print(f"journal entries: {len(journal.read_history())}")
    assert result.report is result.state.evaluator_report
    assert result.report.performance == ()
    assert result.report.verdict is None
    assert not any(
        event.type.value == "correctness_passed" for event in result.state.history
    )

    # The snapshot is intentionally read-only; restore test permissions before
    # TemporaryDirectory cleans up its scratch files.
    frozen = root / "controller" / "frozen"
    for path in frozen.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    for path in frozen.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
```

The diagnostic callback is wired automatically by `run_candidate`; its signal
adds a journal record in the `correctness` phase without changing experiment
state. The report's 7 cases summarize 21 fake-expert checks. Passing them only
confirms the adapter/reference test wiring.

## Multi-submission CLI demonstration

Run the offline session demonstration from the repository root:

```sh
python -m orchestrator.main demo
```

It creates temporary fake submissions, detects a repeated candidate ID, injects
one evaluator failure, and then runs the trusted pass-through evaluator with
NumPy fake experts. The default shared budget is two attempts, so the final
unique submission is reported as budget-exhausted. The summary labels the
trusted pass-through result `completed_no_verdict`; the candidate code is never
executed and the diagnostic does not imply candidate approval. Change the
shared limit with `--max-attempts`, for example:

```sh
python -m orchestrator.main demo --max-attempts 3
```

Run the offline runner and journal tests from the repository root:

```sh
python -B -m unittest \
  tests.test_orchestrator_run \
  tests.test_orchestrator_journal \
  tests.test_orchestrator_session \
  tests.test_orchestrator_cli -v
```
