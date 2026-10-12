"""Pure state transitions for one candidate experiment.

The state machine records controller progress and evaluator output. It never
executes candidate code or changes an evaluator-owned verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from evaluator.schemas import EvaluationReport, RunStatus, Verdict


class ExperimentPhase(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    FREEZING = "freezing"
    CORRECTNESS = "correctness"
    BENCHMARK = "benchmark"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class EventType(str, Enum):
    SUBMITTED = "submitted"
    CANDIDATE_VALIDATED = "candidate_validated"
    CANDIDATE_REJECTED = "candidate_rejected"
    CANDIDATE_FROZEN = "candidate_frozen"
    CORRECTNESS_PASSED = "correctness_passed"
    EVALUATOR_RESULT = "evaluator_result"
    CONTROLLER_FAILED = "controller_failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ExperimentEvent:
    """An event supplied by the controller or protected evaluator."""

    type: EventType
    detail: str = ""
    report: EvaluationReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, EventType):
            raise ValueError("event type must be an EventType")
        if self.type is EventType.EVALUATOR_RESULT and self.report is None:
            raise ValueError("evaluator_result events require an EvaluationReport")
        if self.report is not None and not isinstance(self.report, EvaluationReport):
            raise ValueError("event report must be an EvaluationReport")
        if self.type is not EventType.EVALUATOR_RESULT and self.report is not None:
            raise ValueError("only evaluator_result events may include a report")
        if not isinstance(self.detail, str):
            raise ValueError("event detail must be a string")


@dataclass(frozen=True, slots=True)
class ExperimentState:
    """Current phase and append-only in-memory event history for one candidate."""

    experiment_id: str
    candidate_id: str
    phase: ExperimentPhase = ExperimentPhase.CREATED
    history: tuple[ExperimentEvent, ...] = ()
    evaluator_report: EvaluationReport | None = None
    failure_reason: str | None = None

    @property
    def verdict(self) -> Verdict | None:
        """Return the evaluator's verdict unchanged, if one was issued."""

        if self.evaluator_report is None or self.evaluator_report.verdict is None:
            return None
        return self.evaluator_report.verdict.verdict

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            ExperimentPhase.COMPLETED,
            ExperimentPhase.FAILED,
            ExperimentPhase.INTERRUPTED,
        }


def new_experiment(experiment_id: str, candidate_id: str) -> ExperimentState:
    """Create an initial state for a validated-format candidate ID."""

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")
    if not isinstance(candidate_id, str) or not re.fullmatch(
        r"cand_[0-9a-f]{12}", candidate_id
    ):
        raise ValueError("candidate_id must have the form cand_<12 lowercase hex>")
    return ExperimentState(experiment_id=experiment_id, candidate_id=candidate_id)


_NEXT_PHASE = {
    (ExperimentPhase.CREATED, EventType.SUBMITTED): ExperimentPhase.VALIDATING,
    (ExperimentPhase.VALIDATING, EventType.CANDIDATE_VALIDATED): ExperimentPhase.FREEZING,
    (ExperimentPhase.FREEZING, EventType.CANDIDATE_FROZEN): ExperimentPhase.CORRECTNESS,
    (ExperimentPhase.CORRECTNESS, EventType.CORRECTNESS_PASSED): ExperimentPhase.BENCHMARK,
}
_ACTIVE_PHASES = frozenset(
    {
        ExperimentPhase.CREATED,
        ExperimentPhase.VALIDATING,
        ExperimentPhase.FREEZING,
        ExperimentPhase.CORRECTNESS,
        ExperimentPhase.BENCHMARK,
    }
)


def transition(state: ExperimentState, event: ExperimentEvent) -> ExperimentState:
    """Apply one legal event and return a new state without mutating ``state``."""

    if not isinstance(state, ExperimentState) or not isinstance(event, ExperimentEvent):
        raise TypeError("transition requires an ExperimentState and ExperimentEvent")
    if state.is_terminal:
        raise ValueError(f"cannot transition a terminal experiment ({state.phase.value})")

    next_phase = _NEXT_PHASE.get((state.phase, event.type))
    if next_phase is not None:
        return ExperimentState(
            experiment_id=state.experiment_id,
            candidate_id=state.candidate_id,
            phase=next_phase,
            history=(*state.history, event),
            evaluator_report=state.evaluator_report,
            failure_reason=state.failure_reason,
        )

    if event.type is EventType.CANDIDATE_REJECTED:
        if state.phase is not ExperimentPhase.VALIDATING:
            raise ValueError("candidate_rejected is only valid during validation")
        return _terminal(
            state,
            event,
            ExperimentPhase.FAILED,
            event.detail.strip() or "candidate rejected",
        )

    if event.type is EventType.EVALUATOR_RESULT:
        if state.phase not in {ExperimentPhase.CORRECTNESS, ExperimentPhase.BENCHMARK}:
            raise ValueError("evaluator results are only valid during evaluation")
        report = event.report
        if report is None:  # Also guard against unusual object construction.
            raise ValueError("evaluator_result events require an EvaluationReport")
        if report.candidate.candidate_id != state.candidate_id:
            raise ValueError("evaluator report belongs to a different candidate")
        if report.status is RunStatus.FAILED:
            return _terminal(
                state,
                event,
                ExperimentPhase.FAILED,
                "; ".join(report.errors) or "evaluator run failed",
                report,
            )
        if report.status is RunStatus.INTERRUPTED:
            return _terminal(
                state,
                event,
                ExperimentPhase.INTERRUPTED,
                "; ".join(report.errors) or "evaluator run interrupted",
                report,
            )
        return _terminal(state, event, ExperimentPhase.COMPLETED, None, report)

    if event.type is EventType.INTERRUPTED:
        if state.phase not in _ACTIVE_PHASES:
            raise ValueError(f"cannot interrupt from {state.phase.value}")
        return _terminal(
            state,
            event,
            ExperimentPhase.INTERRUPTED,
            event.detail.strip() or "interrupted",
        )

    if event.type is EventType.CONTROLLER_FAILED:
        if state.phase not in _ACTIVE_PHASES:
            raise ValueError(f"cannot fail from {state.phase.value}")
        return _terminal(
            state,
            event,
            ExperimentPhase.FAILED,
            event.detail.strip() or "controller failed",
        )

    raise ValueError(f"event {event.type.value} is not valid during {state.phase.value}")


def _terminal(
    state: ExperimentState,
    event: ExperimentEvent,
    phase: ExperimentPhase,
    reason: str | None,
    report: EvaluationReport | None = None,
) -> ExperimentState:
    return ExperimentState(
        experiment_id=state.experiment_id,
        candidate_id=state.candidate_id,
        phase=phase,
        history=(*state.history, event),
        evaluator_report=report,
        failure_reason=reason,
    )
