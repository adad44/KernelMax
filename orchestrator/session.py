"""Run a sequence of candidate submissions with one shared attempt budget."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from evaluator.schemas import EvaluationReport, ExperimentContract, Verdict
from orchestrator.candidate_manager import DEFAULT_CANDIDATES_ROOT
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.run import CandidateRunResult, Evaluator, run_candidate
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


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Outcome for one submitted directory, retaining its runner result."""

    index: int
    candidate_id: str
    directory: Path
    outcome: SubmissionOutcome
    result: CandidateRunResult | None = None
    detail: str = ""

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

    def summary(self) -> str:
        counts = {outcome: 0 for outcome in SubmissionOutcome}
        for submission in self.submissions:
            counts[submission.outcome] += 1
        outcome_summary = ", ".join(
            f"{outcome.value}={count}"
            for outcome, count in counts.items()
            if count
        ) or "none"
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
) -> SessionResult:
    """Process submissions in order through ``run_candidate`` with one budget.

    Candidate IDs are unique within a session. A repeated ID is reported as a
    duplicate and never reaches the runner. Validation/evaluator failures are
    retained as per-submission results; later distinct submissions continue
    while the shared budget permits. Explicit evaluator acceptance is the only
    outcome reported as candidate approval.
    """

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

    directories = tuple(candidate_dirs)
    if any(not isinstance(path, (str, Path)) for path in directories):
        raise TypeError("candidate_dirs must contain only paths")

    seen_ids: set[str] = set()
    outcomes: list[SubmissionResult] = []
    for index, raw_directory in enumerate(directories, start=1):
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

        try:
            result = run_candidate(
                directory,
                evaluator=evaluator,
                contract=contract,
                budget=budget,
                journal=journal,
                frozen_root=frozen_root,
                candidates_root=candidates_root,
                experiment_id=f"{session_id}-{index}-{candidate_id}",
            )
            outcome = _outcome(result)
            detail = result.error or ""
        except Exception as exc:  # Keep a single infrastructure failure local.
            result = None
            outcome = SubmissionOutcome.FAILED
            detail = f"session runner failed: {type(exc).__name__}: {exc}"

        outcomes.append(
            SubmissionResult(index, candidate_id, directory, outcome, result, detail)
        )

    return SessionResult(
        session_id=session_id,
        submissions=tuple(outcomes),
        attempts_started=budget.attempts_started,
        max_attempts=budget.max_attempts,
        retries_started=budget.retries_started,
        stop_reason=budget.stop_reason,
    )
