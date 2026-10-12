"""Attempt and retry accounting for the optimization controller.

Every retry must be explicitly classified as transient by the controller and
must correspond to a failed evaluator report. Completed evaluator verdicts and
interrupted runs are never retried automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluator.schemas import EvaluationReport, RunStatus
from orchestrator.state_machine import RunIdentity


class BudgetExhausted(RuntimeError):
    """Raised when an attempt would exceed the configured experiment budget."""


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str
    attempts_remaining: int
    retries_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class Attempt:
    number: int
    run: RunIdentity
    retry_number: int

    @property
    def candidate_id(self) -> str:
        return self.run.candidate.candidate_id

    @property
    def run_id(self) -> str:
        return self.run.run_id


@dataclass
class ExperimentBudget:
    """Mutable controller-side budget for a bounded series of evaluator runs.

    ``max_attempts`` counts all evaluator runs, including retries. The retry
    limit is per candidate; a retry is permitted only for a failed report that
    the controller has independently classified as transient/retryable.

    Start with a RunIdentity built from a controller-assigned run ID, frozen
    candidate metadata/hash, and the locked contract hash. Observe the evaluator
    report before retrying; start_retry requires a fresh run ID and retains the
    original candidate and contract identities. Persistence is not implemented.
    """

    max_attempts: int
    max_retries_per_candidate: int = 0
    attempts_started: int = 0
    retries_by_candidate: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    _attempted_candidates: set[str] = field(default_factory=set, repr=False)
    _retried_run_ids: set[str] = field(default_factory=set, repr=False)
    _runs: dict[str, RunIdentity] = field(default_factory=dict, init=False, repr=False)
    _reports: dict[str, EvaluationReport] = field(
        default_factory=dict, init=False, repr=False
    )
    _latest_run_ids: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _finalized_candidates: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if (
            type(self.max_retries_per_candidate) is not int
            or self.max_retries_per_candidate < 0
        ):
            raise ValueError("max_retries_per_candidate must be a non-negative integer")
        if (
            type(self.attempts_started) is not int
            or not 0 <= self.attempts_started <= self.max_attempts
        ):
            raise ValueError("attempts_started is outside the configured budget")
        if any(
            type(count) is not int
            or count < 0
            or count > self.max_retries_per_candidate
            for count in self.retries_by_candidate.values()
        ):
            raise ValueError("retry counts are outside the configured budget")

    @property
    def retries_started(self) -> int:
        return sum(self.retries_by_candidate.values())

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts_started)

    @property
    def must_stop(self) -> bool:
        """Whether the controller must stop starting attempts now."""

        return self.stop_reason is not None or self.attempts_remaining == 0

    def stop(self, reason: str) -> None:
        """Record an explicit controller stop, such as user cancellation."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason must be a non-empty string")
        self.stop_reason = reason.strip()

    def observe_result(self, report: EvaluationReport) -> None:
        """Update stop state from a result; interruption stops the run budget.

        A failed result does not automatically stop the full search: the
        controller may classify a transient failure and retry within budget, or
        move on to a different candidate. Completed reports are final for their
        candidate but do not consume the budget for other candidates.
        """

        if not isinstance(report, EvaluationReport):
            raise TypeError("observe_result requires an EvaluationReport")
        run = self._runs.get(report.run_id)
        if run is None:
            raise ValueError(
                "evaluator result belongs to a run not started in this budget"
            )
        run.validate_report(report)
        previous = self._reports.get(report.run_id)
        if previous is not None:
            if previous != report:
                raise ValueError("conflicting evaluator results for the same run")
            return  # Duplicate notifications must not change finality or stop state.
        candidate_id = report.candidate.candidate_id
        if self._latest_run_ids[candidate_id] != report.run_id:
            raise ValueError("evaluator result is not for the latest candidate run")
        self._reports[report.run_id] = report
        if report.status is RunStatus.COMPLETED or report.verdict is not None:
            self._finalized_candidates.add(candidate_id)
        if report.status is RunStatus.INTERRUPTED:
            details = "; ".join(report.errors) or "evaluator run interrupted"
            self.stop(details)

    def can_start_attempt(self, run: RunIdentity) -> BudgetDecision:
        if self.stop_reason is not None:
            return self._decision(False, f"budget stopped: {self.stop_reason}")
        if self.attempts_remaining == 0:
            return self._decision(False, "maximum total attempts reached")
        if not isinstance(run, RunIdentity):
            return self._decision(False, "attempt requires a RunIdentity")
        if run.run_id in self._runs:
            return self._decision(False, "run_id has already been started")
        candidate_id = run.candidate.candidate_id
        if candidate_id in self._attempted_candidates:
            return self._decision(False, "candidate already attempted; use start_retry")
        return self._decision(True, "new attempt is within budget")

    def start_attempt(self, run: RunIdentity) -> Attempt:
        """Reserve a unique run bound to the frozen candidate and contract."""

        decision = self.can_start_attempt(run)
        if not decision.allowed:
            raise BudgetExhausted(decision.reason)
        return self._reserve(run, retry_number=0)

    def can_retry(self, report: EvaluationReport, *, retryable: bool) -> BudgetDecision:
        """Explain whether an observed failed run may consume a retry slot.

        Call ``observe_result`` first. A retry must refer to that exact report,
        for the latest reserved run, and can never reopen a finalized candidate.
        """

        if type(retryable) is not bool:
            raise ValueError(
                "retryable must be a bool classification from the controller"
            )
        if not isinstance(report, EvaluationReport):
            return self._decision(False, "retry requires an evaluator EvaluationReport")
        candidate_id = report.candidate.candidate_id
        retries_used = self.retries_by_candidate.get(candidate_id, 0)
        retries_remaining = max(0, self.max_retries_per_candidate - retries_used)
        if self.stop_reason is not None:
            return self._decision(
                False, f"budget stopped: {self.stop_reason}", retries_remaining
            )
        if self.attempts_remaining == 0:
            return self._decision(
                False, "maximum total attempts reached", retries_remaining
            )
        run = self._runs.get(report.run_id)
        if run is None:
            return self._decision(
                False, "run was not started in this budget", retries_remaining
            )
        try:
            run.validate_report(report)
        except ValueError as exc:
            return self._decision(False, str(exc), retries_remaining)
        if candidate_id in self._finalized_candidates:
            return self._decision(
                False,
                "candidate already has a final evaluator result",
                retries_remaining,
            )
        if report.run_id not in self._reports:
            return self._decision(
                False, "run result has not been observed", retries_remaining
            )
        if self._reports[report.run_id] != report:
            return self._decision(
                False,
                "report differs from the observed evaluator result",
                retries_remaining,
            )
        if report.run_id in self._retried_run_ids:
            return self._decision(
                False, "this evaluator run has already been retried", retries_remaining
            )
        if self._latest_run_ids[candidate_id] != report.run_id:
            return self._decision(
                False, "report is not for the latest candidate run", retries_remaining
            )
        if report.status is RunStatus.INTERRUPTED:
            return self._decision(
                False,
                "interrupted runs are not retried automatically",
                retries_remaining,
            )
        if report.status is not RunStatus.FAILED:
            return self._decision(
                False, "only failed evaluator runs may be retried", retries_remaining
            )
        if report.verdict is not None:
            return self._decision(
                False,
                "evaluator-issued verdicts are final for this run",
                retries_remaining,
            )
        if not retryable:
            return self._decision(
                False,
                "failure was not classified as transient/retryable",
                retries_remaining,
            )
        if retries_remaining == 0:
            return self._decision(
                False, "maximum retries for this candidate reached", 0
            )
        return self._decision(
            True, "transient failed run may be retried", retries_remaining
        )

    def start_retry(
        self, report: EvaluationReport, *, run_id: str, retryable: bool
    ) -> Attempt:
        """Reserve a retry after validating its evaluator result and budget."""

        decision = self.can_retry(report, retryable=retryable)
        if not decision.allowed:
            raise BudgetExhausted(decision.reason)
        candidate_id = report.candidate.candidate_id
        previous = self._runs[report.run_id]
        run = RunIdentity(run_id, previous.candidate, previous.contract_sha256)
        if run_id in self._runs:
            raise BudgetExhausted("run_id has already been started")
        retry_number = self.retries_by_candidate.get(candidate_id, 0) + 1
        self.retries_by_candidate[candidate_id] = retry_number
        self._retried_run_ids.add(report.run_id)
        return self._reserve(run, retry_number)

    def _reserve(self, run: RunIdentity, retry_number: int) -> Attempt:
        self.attempts_started += 1
        candidate_id = run.candidate.candidate_id
        self._attempted_candidates.add(candidate_id)
        self._runs[run.run_id] = run
        self._latest_run_ids[candidate_id] = run.run_id
        return Attempt(self.attempts_started, run, retry_number)

    def _decision(
        self, allowed: bool, reason: str, retries_remaining: int | None = None
    ) -> BudgetDecision:
        return BudgetDecision(
            allowed=allowed,
            reason=reason,
            attempts_remaining=self.attempts_remaining,
            retries_remaining=retries_remaining,
        )
