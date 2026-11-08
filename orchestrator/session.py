"""Run a sequence of candidate submissions with one shared attempt budget."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from evaluator.schemas import EvaluationReport, ExperimentContract, Verdict
from orchestrator.candidate_manager import DEFAULT_CANDIDATES_ROOT
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal, JournalError
from orchestrator.run import CandidateRunResult, Evaluator, StageDuration, run_candidate
from orchestrator.state_machine import EventType, ExperimentPhase


class SubmissionOutcome(str, Enum):
    """Session-level disposition; diagnostic completion is not approval."""

    APPROVED = "approved"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    COMPLETED_NO_VERDICT = "completed_no_verdict"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    BUDGET_EXHAUSTED = "budget_exhausted"


STRATEGY_ID = "ordered_unique_submissions_v1"
STRATEGY_SHA256 = hashlib.sha256(
    b"ordered_unique_submissions_v1:input-order:first-id-wins:controller-classified-retries"
).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionProvenance:
    """Frozen controller configuration; absent external identities stay null."""

    contract_sha256: str
    workload_seed: int
    evaluator_type: str
    configuration_ids: tuple[tuple[str, str | None], ...]
    strategy_id: str = STRATEGY_ID
    strategy_sha256: str = STRATEGY_SHA256


@dataclass(frozen=True, slots=True)
class AttemptObservation:
    """Relative monotonic offsets plus the original, unchanged runner result."""

    result: CandidateRunResult
    started_seconds: float
    finished_seconds: float
    within_deadline: bool
    retry_classification: bool | None = None

    def to_dict(self) -> dict:
        attempt, report = self.result.attempt, self.result.report
        state = self.result.state
        return {
            "run_id": attempt.run_id if attempt else None,
            "candidate": attempt.run.candidate.to_dict() if attempt else None,
            "experiment_id": state.experiment_id if state else None,
            "attempt_number": attempt.number if attempt else None,
            "retry_number": attempt.retry_number if attempt else None,
            "started_seconds": self.started_seconds,
            "finished_seconds": self.finished_seconds,
            "within_deadline": self.within_deadline,
            "retry_classification": self.retry_classification,
            "durations": [
                {"stage": item.stage, "elapsed_seconds": item.elapsed_seconds}
                for item in self.result.durations
            ],
            "report_run_id": report.run_id if report else None,
            "environment": report.environment.to_dict()
            if report and report.environment
            else None,
            "verdict": report.verdict.to_dict() if report and report.verdict else None,
            "errors": list(report.errors) if report else None,
            "controller_error": self.result.error,
        }


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Outcome for one submitted directory, retaining its runner result."""

    index: int
    candidate_id: str
    directory: Path
    outcome: SubmissionOutcome
    result: CandidateRunResult | None = None
    detail: str = ""
    attempts: tuple[AttemptObservation, ...] = ()

    @property
    def report(self) -> EvaluationReport | None:
        return self.result.report if self.result is not None else None

    @property
    def candidate_approved(self) -> bool:
        """Only an explicit evaluator ACCEPTED verdict counts as approval."""

        state = self.result.state if self.result is not None else None
        report = state.evaluator_report if state is not None else None
        return (
            state is not None
            and state.phase is ExperimentPhase.COMPLETED
            and report is not None
            and report.verdict is not None
            and report.verdict.verdict is Verdict.ACCEPTED
        )


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Ordered outcomes and budget counters for a complete submission batch."""

    session_id: str
    submissions: tuple[SubmissionResult, ...]
    attempts_started: int
    max_attempts: int
    retries_started: int
    stop_reason: str | None
    search_id: str = ""
    elapsed_seconds: float | None = None
    max_elapsed_seconds: float | None = None
    provenance: SessionProvenance | None = None
    durations: tuple[StageDuration, ...] = ()
    attempts_remaining: int | None = None
    max_retries_per_candidate: int | None = None
    journal_error: str | None = None
    score: float | None = None
    best_run_id: str | None = None
    scoring_reason: str = (
        "unscored: authoritative search scoring is unavailable; acceptance is "
        "disabled/provisional and task-quality execution is planned"
    )

    def to_dict(self) -> dict:
        """JSON-safe accounting with links to evaluator evidence in run history."""
        provenance = self.provenance
        return {
            "search_id": self.search_id,
            "session_id": self.session_id,
            "provenance": {
                "strategy_id": provenance.strategy_id,
                "strategy_sha256": provenance.strategy_sha256,
                "contract_sha256": provenance.contract_sha256,
                "workload_seed": provenance.workload_seed,
                "evaluator_type": provenance.evaluator_type,
                "configuration_ids": dict(provenance.configuration_ids),
            }
            if provenance
            else None,
            "elapsed_seconds": self.elapsed_seconds,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "remaining_seconds": (
                max(0.0, self.max_elapsed_seconds - self.elapsed_seconds)
                if self.max_elapsed_seconds is not None
                and self.elapsed_seconds is not None
                else None
            ),
            "attempts_started": self.attempts_started,
            "attempts_remaining": self.attempts_remaining,
            "max_attempts": self.max_attempts,
            "retries_started": self.retries_started,
            "max_retries_per_candidate": self.max_retries_per_candidate,
            "stop_reason": self.stop_reason,
            "journal_error": self.journal_error,
            "score": self.score,
            "best_run_id": self.best_run_id,
            "scoring_reason": self.scoring_reason,
            "accounting": {
                "coverage": "partial_search_boundary",
                "included": "submission collection, runner work, evaluator calls, retries, failures and journaling within run_session",
                "outside_boundary": "contract/environment setup and pre-generated candidate production",
                "stage_breakdown_unavailable": {
                    name: None
                    for name in (
                        "generation",
                        "profiling",
                        "compilation",
                        "validation_inside_evaluator",
                        "warmups",
                        "benchmarking",
                        "generation_resource_usage",
                    )
                },
                "note": "Stages inside an evaluator call are included in its aggregate duration but cannot be separated; runner_total overlaps its component stages. Final journal flush is included in returned elapsed time, not in the event it writes.",
                "durations": [
                    {"stage": item.stage, "elapsed_seconds": item.elapsed_seconds}
                    for item in self.durations
                ],
                "cancellation": "cooperative scheduling deadline only; reviewed subprocess worker timeouts are capped, generic evaluators and setup/cleanup cannot be safely interrupted",
            },
            "submissions": [
                {
                    "index": item.index,
                    "candidate_id": item.candidate_id,
                    "directory": str(item.directory),
                    "outcome": item.outcome.value,
                    "detail": item.detail,
                    "attempts": [attempt.to_dict() for attempt in item.attempts],
                }
                for item in self.submissions
            ],
        }

    def summary(self) -> str:
        counts = {outcome: 0 for outcome in SubmissionOutcome}
        for submission in self.submissions:
            counts[submission.outcome] += 1
        outcome_summary = (
            ", ".join(
                f"{outcome.value}={count}" for outcome, count in counts.items() if count
            )
            or "none"
        )
        lines = [
            f"Session {self.session_id}",
            (
                f"Submissions: {len(self.submissions)} | evaluator attempts: "
                f"{self.attempts_started}/{self.max_attempts} | "
                f"retries: {self.retries_started}"
            ),
            f"Outcomes: {outcome_summary}",
        ]
        for item in self.submissions:
            line = f"{item.index}. {item.candidate_id}: {item.outcome.value}"
            if item.outcome is SubmissionOutcome.COMPLETED_NO_VERDICT:
                line += " (diagnostics recorded; no candidate approval)"
            elif item.outcome is SubmissionOutcome.APPROVED:
                line += " (explicit evaluator approval)"
            if item.detail:
                line += f" — {item.detail}"
            lines.append(line)
        if self.stop_reason:
            lines.append(f"Session stop reason: {self.stop_reason}")
        if self.elapsed_seconds is not None:
            lines.append(
                f"Elapsed search boundary: {self.elapsed_seconds:.6f}s (partial accounting)"
            )
        lines.append(self.scoring_reason)
        return "\n".join(lines)


def _outcome(result: CandidateRunResult) -> SubmissionOutcome:
    state = result.state
    if state is None:
        return SubmissionOutcome.INVALID
    if state.history and state.history[-1].type is EventType.CANDIDATE_REJECTED:
        return SubmissionOutcome.INVALID
    if state.phase is ExperimentPhase.INTERRUPTED:
        return SubmissionOutcome.INTERRUPTED
    if state.phase is ExperimentPhase.FAILED:
        return SubmissionOutcome.FAILED
    # Use only the report accepted into the state machine. CandidateRunResult
    # deliberately retains mismatched evaluator reports for forensic evidence.
    report = state.evaluator_report
    if report is not None and report.verdict is not None:
        verdict = report.verdict.verdict
        if verdict is Verdict.ACCEPTED:
            return SubmissionOutcome.APPROVED
        if verdict is Verdict.REJECTED:
            return SubmissionOutcome.REJECTED
        return SubmissionOutcome.INCONCLUSIVE
    if state.phase is ExperimentPhase.COMPLETED:
        return SubmissionOutcome.COMPLETED_NO_VERDICT
    return SubmissionOutcome.FAILED


def run_session(
    candidate_dirs: Iterable[str | Path],
    *,
    evaluator: Evaluator,
    contract: ExperimentContract,
    budget: ExperimentBudget,
    journal: ExperimentJournal,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
    session_id: str | None = None,
    max_elapsed_seconds: float | None = None,
    retry_classifier: Callable[[EvaluationReport], bool] | None = None,
    configuration_ids: Mapping[str, str | None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SessionResult:
    """Process submissions in order through ``run_candidate`` with one budget.

    Candidate IDs are unique within a session. A repeated ID is reported as a
    duplicate and never reaches the runner. Validation/evaluator failures are
    retained as per-submission results; later distinct submissions continue
    while the shared budget permits. Explicit evaluator acceptance is the only
    outcome reported as candidate approval.
    """

    if not callable(clock):
        raise TypeError("clock must be callable")
    started = clock()
    if not math.isfinite(started):
        raise ValueError("clock must return finite monotonic seconds")
    if not isinstance(contract, ExperimentContract):
        raise TypeError("contract must be a validated ExperimentContract")
    if not isinstance(budget, ExperimentBudget):
        raise TypeError("budget must be an ExperimentBudget")
    if not isinstance(journal, ExperimentJournal):
        raise TypeError("journal must be an ExperimentJournal")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(frozen_root, (str, Path)):
        raise TypeError("frozen_root must be a path")
    if not isinstance(candidates_root, (str, Path)):
        raise TypeError("candidates_root must be a path")
    if session_id is None:
        session_id = f"session-{uuid.uuid4().hex}"
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if max_elapsed_seconds is not None and (
        type(max_elapsed_seconds) not in (int, float)
        or not math.isfinite(max_elapsed_seconds)
        or max_elapsed_seconds <= 0
    ):
        raise ValueError("max_elapsed_seconds must be finite and positive")
    if retry_classifier is not None and not callable(retry_classifier):
        raise TypeError("retry_classifier must be callable or None")
    identities = {
        "generator": None,
        "environment": None,
        "starting_state": None,
        "evaluator_configuration": None,
        "controller_revision": None,
        "retry_classifier": (
            f"{retry_classifier.__module__}.{retry_classifier.__qualname__}"
            if retry_classifier is not None
            and hasattr(retry_classifier, "__qualname__")
            else None
        ),
    }
    if configuration_ids is not None:
        if not isinstance(configuration_ids, Mapping) or any(
            not isinstance(k, str)
            or not k.strip()
            or (v is not None and (not isinstance(v, str) or not v.strip()))
            for k, v in configuration_ids.items()
        ):
            raise ValueError(
                "configuration_ids must map nonempty names to identities or None"
            )
        identities.update(configuration_ids)
    provenance = SessionProvenance(
        contract.sha256,
        contract.workload.seed,
        f"{getattr(evaluator, '__module__', type(evaluator).__module__)}."
        f"{getattr(evaluator, '__qualname__', type(evaluator).__qualname__)}",
        tuple(sorted(identities.items())),
    )
    last = started

    def now():
        nonlocal last
        value = clock()
        if not math.isfinite(value) or value < last:
            raise ValueError("clock must return finite monotonic seconds")
        last = value
        return value

    deadline = (
        started + max_elapsed_seconds if max_elapsed_seconds is not None else None
    )
    search_id = uuid.uuid4().hex
    durations = []
    outcomes: list[SubmissionResult] = []
    journal_error = None

    def record(event, payload):
        nonlocal journal_error
        try:
            journal.record_session(
                session_id, event, {"search_id": search_id, **payload}
            )
        except JournalError as exc:
            journal_error = f"session journal failed: {exc}"
            budget.stop(journal_error)

    def check_deadline():
        if deadline is not None and now() >= deadline and budget.stop_reason is None:
            budget.stop("session deadline reached")

    record(
        "started",
        {
            "contract_sha256": provenance.contract_sha256,
            "strategy_id": provenance.strategy_id,
            "strategy_sha256": provenance.strategy_sha256,
            "workload_seed": provenance.workload_seed,
            "evaluator_type": provenance.evaluator_type,
            "configuration_ids": dict(provenance.configuration_ids),
            "max_attempts": budget.max_attempts,
            "attempts_started": budget.attempts_started,
            "max_retries_per_candidate": budget.max_retries_per_candidate,
            "max_elapsed_seconds": max_elapsed_seconds,
            "journal_history": str(journal.history_path.absolute()),
        },
    )
    collection_started = now()
    durations.append(StageDuration("session_setup", collection_started - started))
    try:
        # Existing interface accepts a finite trusted iterable, not a cancellable
        # candidate generator. Its consumption is charged to this boundary.
        directories = tuple(candidate_dirs)
    except Exception as exc:  # noqa: BLE001 -- Retain trusted iterable failures.
        directories = ()
        budget.stop(f"submission collection failed: {type(exc).__name__}: {exc}")
    durations.append(StageDuration("submission_collection", now() - collection_started))
    if any(not isinstance(path, (str, Path)) for path in directories):
        raise TypeError("candidate_dirs must contain only paths")

    seen_ids: set[str] = set()
    for index, raw_directory in enumerate(directories, start=1):
        check_deadline()
        directory = Path(raw_directory)
        candidate_id = directory.name
        if candidate_id in seen_ids:
            outcomes.append(
                SubmissionResult(
                    index,
                    candidate_id,
                    directory,
                    SubmissionOutcome.DUPLICATE,
                    detail="candidate ID already appeared in this session",
                )
            )
            continue
        seen_ids.add(candidate_id)

        if budget.must_stop:
            reason = budget.stop_reason or "maximum total attempts reached"
            outcomes.append(
                SubmissionResult(
                    index,
                    candidate_id,
                    directory,
                    SubmissionOutcome.BUDGET_EXHAUSTED,
                    detail=reason,
                )
            )
            continue

        attempts = []
        previous = None
        while True:
            attempt_started = now()
            try:
                result = run_candidate(
                    directory,
                    evaluator=evaluator,
                    contract=contract,
                    budget=budget,
                    journal=journal,
                    frozen_root=Path(frozen_root) / search_id,
                    candidates_root=candidates_root,
                    experiment_id=f"{session_id}-{search_id}-{index}-{len(attempts)}-{candidate_id}",
                    retry_of=previous,
                    retryable=previous is not None,
                    deadline=deadline,
                    clock=now,
                )
                finished = now()
                attempts.append(
                    AttemptObservation(
                        result,
                        attempt_started - started,
                        finished - started,
                        deadline is None or finished <= deadline,
                    )
                )
                outcome, detail = _outcome(result), result.error or ""
                check_deadline()
                report = result.state.evaluator_report if result.state else None
                # No exception-string heuristics: preserve the existing explicit
                # controller classification and Budget.can_retry finality gates.
                if budget.must_stop or report is None or retry_classifier is None:
                    break
                if not budget.can_retry(report, retryable=True).allowed:
                    break
                classified = retry_classifier(report)
                if type(classified) is not bool:
                    raise ValueError("retry_classifier must return a bool")
                attempts[-1] = replace(attempts[-1], retry_classification=classified)
                check_deadline()
                if not classified or budget.must_stop:
                    break
                previous = result
            except Exception as exc:  # noqa: BLE001 -- Stop on controller failures.
                result = attempts[-1].result if attempts else None
                outcome = SubmissionOutcome.FAILED
                detail = f"session runner failed: {type(exc).__name__}: {exc}"
                budget.stop(detail)
                break

        submission = SubmissionResult(
            index, candidate_id, directory, outcome, result, detail, tuple(attempts)
        )
        outcomes.append(submission)
        record(
            "submission",
            {
                "index": index,
                "candidate_id": candidate_id,
                "outcome": outcome.value,
                "detail": detail,
                "attempts": [item.to_dict() for item in attempts],
                "elapsed_seconds": now() - started,
                "attempts_remaining": budget.attempts_remaining,
                "retries_started": budget.retries_started,
            },
        )

    check_deadline()
    result = SessionResult(
        session_id=session_id,
        submissions=tuple(outcomes),
        attempts_started=budget.attempts_started,
        max_attempts=budget.max_attempts,
        retries_started=budget.retries_started,
        stop_reason=budget.stop_reason
        or (
            "maximum total attempts reached"
            if budget.must_stop
            else "submissions exhausted"
        ),
        search_id=search_id,
        elapsed_seconds=now() - started,
        max_elapsed_seconds=max_elapsed_seconds,
        provenance=provenance,
        durations=tuple(durations),
        attempts_remaining=budget.attempts_remaining,
        max_retries_per_candidate=budget.max_retries_per_candidate,
        journal_error=journal_error,
    )
    record("completed", result.to_dict())
    return replace(
        result,
        elapsed_seconds=now() - started,
        journal_error=journal_error,
        stop_reason=budget.stop_reason or result.stop_reason,
    )
