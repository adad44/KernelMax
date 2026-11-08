"""Coordinate candidate validation, trusted evaluation, budget, and journal."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from orchestrator.journal import ExperimentJournal, JournalError
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
class StageDuration:
    """Controller wall time, never an inference benchmark sample."""

    stage: str
    elapsed_seconds: float


@contextmanager
def _measure(stage, durations, clock):
    started = clock()
    try:
        yield
    finally:
        durations.append(StageDuration(stage, clock() - started))


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """The final controller state and untouched evaluator response, if any."""

    state: ExperimentState | None
    candidate: Candidate | None = None
    frozen_candidate: FrozenCandidate | None = None
    attempt: Attempt | None = None
    report: EvaluationReport | None = None
    error: str | None = None
    durations: tuple[StageDuration, ...] = ()


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
    retry_of: CandidateRunResult | None = None,
    retryable: bool = False,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CandidateRunResult:
    """Run one attempt; optional retries reuse the original frozen snapshot.

    Only the controller supplies ``retryable``. ExperimentBudget still decides
    whether an observed failed report permits a retry. Deadline checks prevent
    new evaluation work; arbitrary Python evaluators cannot be safely interrupted.
    Search durations include journal/callback overhead and are not GPU timings.
    """
    durations = []
    with _measure("runner_total", durations, clock):
        result = _run_candidate(
            candidate_dir,
            evaluator=evaluator,
            contract=contract,
            budget=budget,
            journal=journal,
            frozen_root=frozen_root,
            candidates_root=candidates_root,
            experiment_id=experiment_id,
            retry_of=retry_of,
            retryable=retryable,
            deadline=deadline,
            clock=clock,
            durations=durations,
        )
    return replace(result, durations=tuple(durations))


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


def _run_candidate(
    candidate_dir: str | Path,
    *,
    evaluator: Evaluator,
    contract: ExperimentContract,
    budget: ExperimentBudget,
    journal: ExperimentJournal,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
    experiment_id: str | None = None,
    retry_of: CandidateRunResult | None = None,
    retryable: bool = False,
    deadline: float | None = None,
    clock=time.monotonic,
    durations=None,
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
        with _measure("validation", durations, clock):
            if retry_of is None:
                candidate = validate_candidate(candidate_dir, candidates_root)
            else:
                if (
                    retry_of.state is None
                    or retry_of.state.evaluator_report is not retry_of.report
                    or retry_of.frozen_candidate is None
                    or retry_of.candidate is None
                    or retry_of.candidate.candidate_id != candidate_id
                    or retry_of.report is None
                    or retry_of.report.contract_sha256 != contract.sha256
                ):
                    raise CandidateValidationError(
                        "retry requires a retained evaluator result"
                    )
                frozen_identity = CandidateIdentity(
                    retry_of.frozen_candidate.candidate_id,
                    retry_of.frozen_candidate.implementation_type,
                    retry_of.frozen_candidate.sha256,
                )
                if frozen_identity != retry_of.report.candidate:
                    raise CandidateValidationError(
                        "retry snapshot differs from the observed candidate identity"
                    )
                decision = budget.can_retry(retry_of.report, retryable=retryable)
                if not decision.allowed:
                    raise CandidateValidationError(decision.reason)
                candidate = retry_of.candidate
    except CandidateValidationError as exc:
        rejected = ExperimentEvent(EventType.CANDIDATE_REJECTED, str(exc))
        state = _advance(state, rejected, journal)
        return CandidateRunResult(state, error=str(exc))

    state = _advance(state, ExperimentEvent(EventType.CANDIDATE_VALIDATED), journal)
    if deadline is not None and clock() >= deadline:
        budget.stop("session deadline reached")
    if budget.must_stop:
        reason = budget.stop_reason or "maximum total attempts reached"
        failed = ExperimentEvent(EventType.CONTROLLER_FAILED, reason)
        state = _advance(state, failed, journal)
        return CandidateRunResult(state, candidate=candidate, error=reason)

    frozen: FrozenCandidate | None = None
    try:
        with _measure("freezing", durations, clock):
            frozen = (
                freeze_candidate(
                    candidate.directory,
                    frozen_root=frozen_root,
                    candidates_root=candidates_root,
                )
                if retry_of is None
                else retry_of.frozen_candidate
            )
            verify_frozen_candidate(frozen)
        identity = RunIdentity(
            run_id=str(uuid.uuid4()),
            candidate=CandidateIdentity(
                frozen.candidate_id, frozen.implementation_type, frozen.sha256
            ),
            contract_sha256=contract.sha256,
        )
        if deadline is not None and clock() >= deadline:
            budget.stop("session deadline reached")
        attempt = (
            budget.start_attempt(identity)
            if retry_of is None
            else budget.start_retry(
                retry_of.report, run_id=identity.run_id, retryable=retryable
            )
        )
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

    diagnostic_error: Exception | None = None

    def on_diagnostic_passed(report: EvaluationReport) -> None:
        """Journal diagnostics without advancing candidate correctness phases."""

        nonlocal diagnostic_error
        try:
            attempt.run.validate_report(report)
        except (TypeError, ValueError) as exc:
            diagnostic_error = exc
            budget.stop(f"invalid evaluator diagnostic: {exc}")
            raise
        try:
            journal.record_signal(state, "on_diagnostic_passed", report)
        except JournalError as exc:
            diagnostic_error = exc
            budget.stop(f"diagnostic journal failed: {exc}")
            raise

    def on_candidate_correctness_passed(report: EvaluationReport) -> None:
        """Record real operator correctness without entering benchmarking."""
        attempt.run.validate_report(report)
        if (
            report.status.value != "completed"
            or report.errors
            or report.verdict is not None
            or report.performance
        ):
            raise ValueError(
                "candidate correctness signal requires completed untimed evidence"
            )
        if len(report.correctness) != len(contract.correctness.required_cases) or {
            r.case_name for r in report.correctness
        } != set(contract.correctness.required_cases):
            raise ValueError(
                "candidate correctness signal requires all declared cases exactly once"
            )
        for result in report.correctness:
            detail = json.loads(result.detail)
            if (
                not result.passed
                or detail.get("evaluation_scope") != "reviewed_candidate_execution"
                or detail.get("candidate_execution_started") is not True
            ):
                raise ValueError("diagnostics are not candidate correctness evidence")
        journal.record_signal(state, "on_candidate_correctness_passed", report)

    report: EvaluationReport | None = None
    evaluator_error: Exception | None = None
    evaluator_interrupted: KeyboardInterrupt | None = None
    try:
        if deadline is not None and clock() >= deadline:
            budget.stop("session deadline reached")
            raise TimeoutError("session deadline reached before evaluator call")
        callbacks = {"on_diagnostic_passed": on_diagnostic_passed}
        # Legacy diagnostics/fakes retain their existing call signature. Session
        # orchestration need not know about this opt-in evaluator capability.
        if getattr(evaluator, "supports_candidate_correctness", False) is True:
            callbacks["on_candidate_correctness_passed"] = (
                on_candidate_correctness_passed
            )
        # Reuse the existing worker process timeout, without mutating its frozen
        # configuration. Setup/cleanup and arbitrary Python calls are not bounded.
        bounded_evaluator = evaluator
        if (
            deadline is not None
            and type(evaluator).__module__ == "evaluator.candidate_execution"
        ):
            from evaluator.candidate_execution import ReviewedCandidateEvaluator

            if isinstance(evaluator, ReviewedCandidateEvaluator):
                remaining = deadline - clock()
                if remaining <= 0:
                    budget.stop("session deadline reached")
                    raise TimeoutError("session deadline reached before worker setup")
                bounded_evaluator = replace(
                    evaluator,
                    timeout_seconds=min(evaluator.timeout_seconds, remaining / 2),
                )
        with _measure("evaluation_call", durations, clock):
            report = bounded_evaluator(attempt.run, frozen, contract, **callbacks)
    except KeyboardInterrupt as exc:
        evaluator_interrupted = exc
    except Exception as exc:  # noqa: BLE001 -- Retain any injected evaluator failure.
        evaluator_error = exc

    # An evaluator may catch callback exceptions. They remain controller failures.
    if diagnostic_error is not None:
        evaluator_error = diagnostic_error

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
        if isinstance(evaluator_error, JournalError):
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

    try:
        state = _advance(
            state, ExperimentEvent(EventType.EVALUATOR_RESULT, report=report), journal
        )
    except JournalError as exc:
        reason = f"result journal failed: {exc}"
        budget.stop(reason)
        # Keep available evidence in memory; this state was not durably recorded.
        state = transition(state, ExperimentEvent(EventType.CONTROLLER_FAILED, reason))
        return CandidateRunResult(
            state,
            candidate=candidate,
            frozen_candidate=frozen,
            attempt=attempt,
            report=report,
            error=reason,
        )
    return CandidateRunResult(
        state,
        candidate=candidate,
        frozen_candidate=frozen,
        attempt=attempt,
        report=report,
        error=state.failure_reason,
    )
