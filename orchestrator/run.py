"""Coordinate candidate validation, trusted evaluation, budget, and journal."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evaluator.schemas import CandidateIdentity, EvaluationReport, ExperimentContract
from orchestrator.candidate_manager import (
    DEFAULT_CANDIDATES_ROOT,
    Candidate,
    CandidateValidationError,
    FrozenCandidate,
    freeze_candidate,
    validate_candidate,
    verify_frozen_candidate,
)
from orchestrator.experiment_budget import Attempt, BudgetExhausted, ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.state_machine import (
    EventType,
    ExperimentEvent,
    ExperimentState,
    RunIdentity,
    new_experiment,
    transition,
)


class Evaluator(Protocol):
    """Interface implemented by ``TrustedPassThroughEvaluator`` and test fakes."""

    def __call__(
        self,
        run_identity: RunIdentity,
        frozen_candidate: FrozenCandidate,
        contract: ExperimentContract,
        *,
        on_diagnostic_passed: Callable[[EvaluationReport], None] | None = None,
    ) -> EvaluationReport: ...


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """The final controller state and untouched evaluator response, if any."""

    state: ExperimentState | None
    candidate: Candidate | None = None
    frozen_candidate: FrozenCandidate | None = None
    attempt: Attempt | None = None
    report: EvaluationReport | None = None
    error: str | None = None


def _advance(
    state: ExperimentState,
    event: ExperimentEvent,
    journal: ExperimentJournal,
    *,
    observed_report: EvaluationReport | None = None,
) -> ExperimentState:
    updated = transition(state, event)
    journal.record(updated, event, observed_report=observed_report)
    return updated


def run_candidate(
    candidate_dir: str | Path,
    *,
    evaluator: Evaluator,
    contract: ExperimentContract,
    budget: ExperimentBudget,
    journal: ExperimentJournal,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
    experiment_id: str | None = None,
) -> CandidateRunResult:
    """Validate, freeze, reserve, evaluate, and journal one candidate.

    The evaluator receives only the controller-created run identity, frozen
    submission metadata/files, and validated contract. The runner never imports
    or executes candidate code. Stop submission writers before freezing and
    keep ``frozen_root`` inaccessible to them; process/filesystem isolation is
    the trusted evaluator/runtime's responsibility.
    """

    if not isinstance(contract, ExperimentContract):
        raise TypeError("contract must be a validated ExperimentContract")
    if not isinstance(budget, ExperimentBudget):
        raise TypeError("budget must be an ExperimentBudget")
    if not isinstance(journal, ExperimentJournal):
        raise TypeError("journal must be an ExperimentJournal")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")

    candidate_id = Path(candidate_dir).name
    run_group = experiment_id or f"{contract.experiment_id}-{uuid.uuid4().hex}"
    try:
        state = new_experiment(run_group, candidate_id)
    except (TypeError, ValueError) as exc:
        journal.record_rejection(run_group, candidate_dir, str(exc))
        return CandidateRunResult(state=None, error=str(exc))

    submitted = ExperimentEvent(EventType.SUBMITTED)
    state = _advance(state, submitted, journal)
    try:
        candidate = validate_candidate(candidate_dir, candidates_root)
    except CandidateValidationError as exc:
        rejected = ExperimentEvent(EventType.CANDIDATE_REJECTED, str(exc))
        state = _advance(state, rejected, journal)
        return CandidateRunResult(state, error=str(exc))

    state = _advance(state, ExperimentEvent(EventType.CANDIDATE_VALIDATED), journal)
    if budget.must_stop:
        reason = budget.stop_reason or "maximum total attempts reached"
        failed = ExperimentEvent(EventType.CONTROLLER_FAILED, reason)
        state = _advance(state, failed, journal)
        return CandidateRunResult(state, candidate=candidate, error=reason)

    frozen: FrozenCandidate | None = None
    try:
        frozen = freeze_candidate(
            candidate.directory,
            frozen_root=frozen_root,
            candidates_root=candidates_root,
        )
        verify_frozen_candidate(frozen)
        identity = RunIdentity(
            run_id=str(uuid.uuid4()),
            candidate=CandidateIdentity(
                frozen.candidate_id, frozen.implementation_type, frozen.sha256
            ),
            contract_sha256=contract.sha256,
        )
        attempt = budget.start_attempt(identity)
    except (
        CandidateValidationError,
        BudgetExhausted,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failed = ExperimentEvent(EventType.CONTROLLER_FAILED, reason)
        state = _advance(state, failed, journal)
        return CandidateRunResult(
            state, candidate=candidate, frozen_candidate=frozen, error=reason
        )

    state = _advance(
        state,
        ExperimentEvent(EventType.CANDIDATE_FROZEN, run=attempt.run),
        journal,
    )

    def on_diagnostic_passed(report: EvaluationReport) -> None:
        """Journal diagnostics without advancing candidate correctness phases."""

        attempt.run.validate_report(report)
        journal.record_signal(state, "on_diagnostic_passed", report)

    report: EvaluationReport | None = None
    evaluator_error: Exception | None = None
    evaluator_interrupted: KeyboardInterrupt | None = None
    try:
        report = evaluator(
            attempt.run,
            frozen,
            contract,
            on_diagnostic_passed=on_diagnostic_passed,
        )
    except KeyboardInterrupt as exc:
        evaluator_interrupted = exc
    except Exception as exc:  # noqa: BLE001 -- Retain any injected evaluator failure.
        evaluator_error = exc

    try:
        verify_frozen_candidate(frozen)
    except CandidateValidationError as exc:
        reason = f"snapshot integrity failure: {exc}"
        budget.stop(reason)
        failed = ExperimentEvent(EventType.CONTROLLER_FAILED, reason)
        state = _advance(
            state,
            failed,
            journal,
            observed_report=(report if isinstance(report, EvaluationReport) else None),
        )
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            report=report,
            error=reason,
        )

    if evaluator_interrupted is not None:
        reason = f"evaluator interrupted: {evaluator_interrupted}".rstrip()
        budget.stop(reason)
        state = _advance(state, ExperimentEvent(EventType.INTERRUPTED, reason), journal)
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            error=reason,
        )
    if evaluator_error is not None:
        reason = (
            f"evaluator failed: {type(evaluator_error).__name__}: {evaluator_error}"
        )
        state = _advance(
            state, ExperimentEvent(EventType.CONTROLLER_FAILED, reason), journal
        )
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            error=reason,
        )

    try:
        attempt.run.validate_report(report)
        budget.observe_result(report)
    except CandidateValidationError as exc:
        reason = f"snapshot integrity failure: {exc}"
        budget.stop(reason)
        state = _advance(
            state,
            ExperimentEvent(EventType.CONTROLLER_FAILED, reason),
            journal,
            observed_report=(report if isinstance(report, EvaluationReport) else None),
        )
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            report=report,
            error=reason,
        )
    except (TypeError, ValueError) as exc:
        reason = f"invalid evaluator report: {exc}"
        state = _advance(
            state,
            ExperimentEvent(EventType.CONTROLLER_FAILED, reason),
            journal,
            observed_report=(report if isinstance(report, EvaluationReport) else None),
        )
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            report=report,
            error=reason,
        )

    state = _advance(
        state, ExperimentEvent(EventType.EVALUATOR_RESULT, report=report), journal
    )
    return CandidateRunResult(
        state,
        candidate=candidate,
        frozen_candidate=frozen,
        attempt=attempt,
        report=report,
        error=state.failure_reason,
    )
