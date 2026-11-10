"""Paired, partitioned comparisons for search-policy revisions.

This module is controller-owned protocol code. It accepts only immutable
search evidence from a trusted executor; it does not execute candidate code or
reinterpret evaluator reports. Production promotion remains gated on the
executor supplying authoritative, eligible end-to-end search scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from orchestrator.journal import ExperimentJournal
from orchestrator.search import (
    CallbackFailure,
    CallbackTimeout,
    SearchPolicy,
    _invoke_with_timeout,
    callback_failure_category,
)


class TaskPartition(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True, slots=True)
class SearchTask:
    task_id: str
    partition: TaskPartition
    task_sha256: str
    workload_sha256: str
    starting_state_sha256: str
    environment_sha256: str
    contract_sha256: str
    generator_version: str
    model_configuration_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.partition, TaskPartition):
            raise TypeError("partition must be a TaskPartition")
        for name in (
            "task_sha256", "workload_sha256", "starting_state_sha256", "environment_sha256",
            "contract_sha256", "model_configuration_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version must be a non-empty identity")


@dataclass(frozen=True, slots=True)
class SearchLimits:
    """Per-search caps provided to the trusted executor."""

    max_elapsed_seconds: float
    max_attempts: int
    max_generation_units: int

    def __post_init__(self) -> None:
        if type(self.max_elapsed_seconds) not in (int, float) or not math.isfinite(
            self.max_elapsed_seconds
        ) or self.max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be finite and positive")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if type(self.max_generation_units) is not int or self.max_generation_units < 0:
            raise ValueError("max_generation_units must be non-negative")


class SearchGenerationQuota:
    """Attempt/generation meters passed into the isolated trusted search callback."""

    def __init__(self, maximum: int, max_attempts: int = 2**31 - 1):
        if type(maximum) is not int or maximum < 0:
            raise ValueError("maximum generation units must be non-negative")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.maximum = maximum
        self.max_attempts = max_attempts
        self.used = 0
        self.attempts_used = 0

    def consume_attempt(self) -> None:
        if self.attempts_used >= self.max_attempts:
            raise RuntimeError("evaluator attempt budget exhausted")
        self.attempts_used += 1

    def consume(self, units: int = 1) -> None:
        if type(units) is not int or units < 1:
            raise ValueError("generation units charged must be a positive integer")
        if self.used + units > self.maximum:
            raise RuntimeError("generation resource budget exhausted")
        self.used += units


@dataclass(frozen=True, slots=True)
class ComparisonBudget:
    """Whole-comparison caps; confirmation allocation is reserved up front."""

    max_elapsed_seconds: float
    max_attempts: int
    max_generation_units: int
    per_search: SearchLimits
    proposal_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if type(self.max_elapsed_seconds) not in (int, float) or not math.isfinite(
            self.max_elapsed_seconds
        ) or self.max_elapsed_seconds <= 0:
            raise ValueError("comparison max_elapsed_seconds must be finite and positive")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("comparison max_attempts must be positive")
        if type(self.max_generation_units) is not int or self.max_generation_units < 0:
            raise ValueError("comparison max_generation_units must be non-negative")
        if not isinstance(self.per_search, SearchLimits):
            raise TypeError("per_search must be SearchLimits")
        if type(self.proposal_timeout_seconds) not in (int, float) or not math.isfinite(
            self.proposal_timeout_seconds
        ) or self.proposal_timeout_seconds <= 0:
            raise ValueError("proposal_timeout_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    """Score and cost returned by a trusted full-search executor.

    `speedup` is the best eligible candidate's normalized speedup over the
    frozen reference. A completed valid search with no eligible candidate must
    use 1.0. Diagnostic-only output is never eligible.
    """

    completed: bool
    environment_valid: bool
    eligible: bool
    speedup: float | None
    attempts_used: int
    generation_units_used: int
    best_run_id: str | None = None
    best_candidate_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    report_ids: tuple[str, ...] = ()
    selected_hypothesis: str | None = None
    stage_durations: tuple[tuple[str, float], ...] = ()
    stop_reason: str = ""
    failure_category: str | None = None

    def __post_init__(self) -> None:
        if type(self.completed) is not bool or type(self.environment_valid) is not bool:
            raise TypeError("completed and environment_valid must be booleans")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a boolean")
        if self.speedup is not None and (
            type(self.speedup) not in (int, float)
            or not math.isfinite(self.speedup)
            or self.speedup <= 0
        ):
            raise ValueError("speedup must be finite and positive")
        if self.eligible and (not self.completed or not self.environment_valid or self.speedup is None):
            raise ValueError("eligible evidence requires a valid completed search and speedup")
        if self.eligible and (
            not self.best_run_id or not self.best_candidate_id or not self.report_ids
        ):
            raise ValueError("eligible evidence requires best candidate/run IDs and unchanged report IDs")
        if type(self.attempts_used) is not int or self.attempts_used < 0:
            raise ValueError("attempts_used must be non-negative")
        if type(self.generation_units_used) is not int or self.generation_units_used < 0:
            raise ValueError("generation_units_used must be non-negative")
        if self.completed and self.environment_valid and not self.eligible and self.speedup not in (None, 1.0):
            raise ValueError("a valid search without an eligible candidate scores 1.0")
        if any(not isinstance(item, str) or not item for item in self.report_ids):
            raise ValueError("report_ids must contain non-empty immutable report identities")
        for field_name in ("candidate_ids", "run_ids"):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"{field_name} must be an immutable tuple")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field_name} must contain non-empty IDs")
        if type(self.report_ids) is not tuple or type(self.stage_durations) is not tuple:
            raise TypeError("report_ids and stage_durations must be immutable tuples")
        if self.best_candidate_id is not None and self.best_candidate_id not in self.candidate_ids:
            raise ValueError("best_candidate_id must appear in candidate_ids")
        if self.best_run_id is not None and self.best_run_id not in self.run_ids:
            raise ValueError("best_run_id must appear in run_ids")
        if self.selected_hypothesis is not None and not isinstance(self.selected_hypothesis, str):
            raise TypeError("selected_hypothesis must be a string or None")
        if any(
            not name or type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
            for name, duration in self.stage_durations
        ):
            raise ValueError("stage_durations must contain named finite non-negative durations")
        if not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string")
        if self.failure_category is not None and (
            not isinstance(self.failure_category, str) or not self.failure_category.strip()
        ):
            raise ValueError("failure_category must be non-empty or None")

    @property
    def score(self) -> float | None:
        if not self.completed or not self.environment_valid:
            return None
        return float(self.speedup if self.eligible else 1.0)


@dataclass(frozen=True, slots=True)
class SearchObservation:
    policy_id: str
    policy_revision: int
    policy_sha256: str
    task: SearchTask
    seed: int
    evidence: SearchEvidence
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "policy_sha256": self.policy_sha256,
            "task_id": self.task.task_id,
            "partition": self.task.partition.value,
            "task_sha256": self.task.task_sha256,
            "workload_sha256": self.task.workload_sha256,
            "starting_state_sha256": self.task.starting_state_sha256,
            "environment_sha256": self.task.environment_sha256,
            "contract_sha256": self.task.contract_sha256,
            "generator_version": self.task.generator_version,
            "model_configuration_sha256": self.task.model_configuration_sha256,
            "seed": self.seed,
            "score": self.evidence.score,
            "completed": self.evidence.completed,
            "environment_valid": self.evidence.environment_valid,
            "eligible": self.evidence.eligible,
            "attempts_used": self.evidence.attempts_used,
            "generation_units_used": self.evidence.generation_units_used,
            "best_run_id": self.evidence.best_run_id,
            "best_candidate_id": self.evidence.best_candidate_id,
            "candidate_ids": list(self.evidence.candidate_ids),
            "run_ids": list(self.evidence.run_ids),
            "report_ids": list(self.evidence.report_ids),
            "selected_hypothesis": self.evidence.selected_hypothesis,
            "stage_durations": [
                {"stage": name, "elapsed_seconds": duration}
                for name, duration in self.evidence.stage_durations
            ],
            "stop_reason": self.evidence.stop_reason,
            "failure_category": self.evidence.failure_category,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class ComparisonRules:
    minimum_paired_seeds: int = 2
    minimum_gain_fraction: float = 0.01
    maximum_task_regression_fraction: float = 0.02
    confidence_z: float = 1.96
    task_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.task_weights) is not tuple:
            raise TypeError("task_weights must be an immutable tuple")
        if type(self.minimum_paired_seeds) is not int or self.minimum_paired_seeds < 2:
            raise ValueError("minimum_paired_seeds must be at least 2")
        for name in ("minimum_gain_fraction", "maximum_task_regression_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if type(self.confidence_z) not in (int, float) or not math.isfinite(self.confidence_z) or self.confidence_z <= 0:
            raise ValueError("confidence_z must be finite and positive")
        names = [name for name, _ in self.task_weights]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("task_weights must have unique non-empty task IDs")
        if any(type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0 for _, weight in self.task_weights):
            raise ValueError("task weights must be finite and positive")


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    comparison_id: str
    incumbent: SearchPolicy
    challenger: SearchPolicy | None
    training: tuple[SearchObservation, ...]
    validation: tuple[SearchObservation, ...]
    confirmation: tuple[SearchObservation, ...]
    validation_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    validation_decision: str
    confirmation_decision: str
    final_decision: str
    validation_mean_log_gain: float | None
    validation_lower_log_gain: float | None
    validation_task_results: tuple["TaskComparison", ...]
    confirmation_mean_log_gain: float | None
    confirmation_lower_log_gain: float | None
    confirmation_task_results: tuple["TaskComparison", ...]
    elapsed_seconds: float
    attempts_used: int
    generation_units_used: int
    stop_reason: str | None = None

    @property
    def promoted_policy(self) -> SearchPolicy | None:
        return self.challenger if self.final_decision == "promote" else None


SearchExecutor = Callable[
    [SearchPolicy, SearchTask, int, SearchLimits, SearchGenerationQuota], SearchEvidence
]
TrainingProposer = Callable[
    [SearchPolicy, tuple[SearchObservation, ...]], "SearchPolicyProposal"
]


@dataclass(frozen=True, slots=True)
class SearchPolicyProposal:
    policy: SearchPolicy
    rationale: str
    expected_benefit: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SearchPolicy):
            raise TypeError("proposal policy must be a SearchPolicy")
        for name in ("rationale", "expected_benefit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class TaskComparison:
    task_id: str
    mean_log_gain: float
    standard_error: float
    lower_confidence_log_gain: float
    regression_limit_log_gain: float
    decision: str

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mean_log_gain": self.mean_log_gain,
            "standard_error": self.standard_error,
            "lower_confidence_log_gain": self.lower_confidence_log_gain,
            "regression_limit_log_gain": self.regression_limit_log_gain,
            "decision": self.decision,
        }


def _partitioned_tasks(
    tasks: Sequence[SearchTask], partition: TaskPartition,
    all_ids: set[str], all_hashes: set[str],
) -> tuple[SearchTask, ...]:
    result = tuple(tasks)
    if not result:
        raise ValueError(f"at least one {partition.value} task is required")
    for task in result:
        if not isinstance(task, SearchTask) or task.partition is not partition:
            raise ValueError(f"tasks must belong to the {partition.value} partition")
        if task.task_id in all_ids:
            raise ValueError("training, validation, and confirmation task IDs must be disjoint")
        if task.task_sha256 in all_hashes:
            raise ValueError("training, validation, and confirmation task contents must be disjoint")
        all_ids.add(task.task_id)
        all_hashes.add(task.task_sha256)
    return result


def _paired_summary(
    incumbent: Sequence[SearchObservation], challenger: Sequence[SearchObservation],
    task_ids: Sequence[str], seeds: Sequence[int], rules: ComparisonRules,
) -> tuple[str, float | None, float | None, tuple[TaskComparison, ...]]:
    task_means: list[tuple[str, float, float, int]] = []
    for task_id in task_ids:
        deltas = []
        for seed in seeds:
            left = next((x for x in incumbent if x.task.task_id == task_id and x.seed == seed), None)
            right = next((x for x in challenger if x.task.task_id == task_id and x.seed == seed), None)
            if left is None or right is None or left.evidence.score is None or right.evidence.score is None:
                return "inconclusive", None, None, ()
            deltas.append(math.log(right.evidence.score / left.evidence.score))
        if len(deltas) < rules.minimum_paired_seeds:
            return "inconclusive", None, None, ()
        mean = sum(deltas) / len(deltas)
        variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        task_means.append((task_id, mean, math.sqrt(variance / len(deltas)), len(deltas)))

    weights = dict(rules.task_weights)
    if weights and not set(task_ids).issubset(weights):
        raise ValueError("task_weights must specify every compared task")
    raw_weights = [weights.get(task_id, 1.0) for task_id in task_ids]
    total_weight = sum(raw_weights)
    normalized = [value / total_weight for value in raw_weights]
    mean = sum(weight * row[1] for weight, row in zip(normalized, task_means))
    standard_error = math.sqrt(
        sum((weight * row[2]) ** 2 for weight, row in zip(normalized, task_means))
    )
    lower = mean - rules.confidence_z * standard_error
    regression_floor = math.log1p(-rules.maximum_task_regression_fraction)
    task_results = tuple(
        TaskComparison(
            task_id=row[0],
            mean_log_gain=row[1],
            standard_error=row[2],
            lower_confidence_log_gain=row[1] - rules.confidence_z * row[2],
            regression_limit_log_gain=regression_floor,
            decision=(
                "reject_regression"
                if row[1] - rules.confidence_z * row[2] < regression_floor
                else "pass"
            ),
        )
        for row in task_means
    )
    if any(row.decision == "reject_regression" for row in task_results):
        return "reject", mean, lower, task_results
    gain_floor = math.log1p(rules.minimum_gain_fraction)
    if lower > gain_floor:
        return "pass", mean, lower, task_results
    return "reject", mean, lower, task_results


def _record(journal: ExperimentJournal, comparison_id: str, event: str, payload: dict) -> None:
    journal.record_session(comparison_id, event, {"comparison_id": comparison_id, **payload})


def run_policy_comparison(
    *,
    incumbent: SearchPolicy,
    training_tasks: Sequence[SearchTask],
    validation_tasks: Sequence[SearchTask],
    confirmation_tasks: Sequence[SearchTask],
    training_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    confirmation_seeds: Sequence[int],
    proposer: TrainingProposer,
    executor: SearchExecutor,
    budget: ComparisonBudget,
    rules: ComparisonRules,
    journal: ExperimentJournal,
    comparison_id: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ComparisonResult:
    """Train a proposal, compare it on paired validation, then confirm on test.

    Test/confirmation task objects are not sent to the proposer or validation
    executor. Each callback is a killable process with a hard wall timeout.
    """

    if not isinstance(incumbent, SearchPolicy) or not isinstance(budget, ComparisonBudget):
        raise TypeError("incumbent and budget must be SearchPolicy and ComparisonBudget")
    if not isinstance(rules, ComparisonRules) or not isinstance(journal, ExperimentJournal):
        raise TypeError("rules and journal have invalid types")
    if not callable(proposer) or not callable(executor) or not callable(clock):
        raise TypeError("proposer, executor, and clock must be callable")
    for label, seeds in (
        ("training_seeds", training_seeds),
        ("validation_seeds", validation_seeds),
        ("confirmation_seeds", confirmation_seeds),
    ):
        if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{label} must be a non-empty sequence of unique integers")
    if (
        set(training_seeds) & set(validation_seeds)
        or set(training_seeds) & set(confirmation_seeds)
        or set(validation_seeds) & set(confirmation_seeds)
    ):
        raise ValueError("training, validation, and confirmation seeds must be disjoint")

    all_ids: set[str] = set()
    all_task_hashes: set[str] = set()
    train = _partitioned_tasks(training_tasks, TaskPartition.TRAINING, all_ids, all_task_hashes)
    validation = _partitioned_tasks(validation_tasks, TaskPartition.VALIDATION, all_ids, all_task_hashes)
    confirmation = _partitioned_tasks(confirmation_tasks, TaskPartition.CONFIRMATION, all_ids, all_task_hashes)
    all_tasks = (*train, *validation, *confirmation)
    for field_name in (
        "environment_sha256", "contract_sha256", "generator_version",
        "model_configuration_sha256",
    ):
        if len({getattr(task, field_name) for task in all_tasks}) != 1:
            raise ValueError(f"all paired tasks must share {field_name}")
    if min(len(validation_seeds), len(confirmation_seeds)) < rules.minimum_paired_seeds:
        raise ValueError("validation and confirmation seeds do not meet minimum_paired_seeds")
    if set(dict(rules.task_weights)) and set(dict(rules.task_weights)) != {
        task.task_id for task in (*validation, *confirmation)
    }:
        raise ValueError("weights must cover the paired validation and confirmation tasks")

    train_slots = len(train) * len(training_seeds)
    validation_slots = 2 * len(validation) * len(validation_seeds)
    confirmation_slots = 2 * len(confirmation) * len(confirmation_seeds)
    total_slots = train_slots + validation_slots + confirmation_slots
    reserved_confirmation_attempts = confirmation_slots * budget.per_search.max_attempts
    reserved_confirmation_generation = confirmation_slots * budget.per_search.max_generation_units
    reserved_confirmation_time = confirmation_slots * budget.per_search.max_elapsed_seconds
    required_attempts = total_slots * budget.per_search.max_attempts
    required_generation = total_slots * budget.per_search.max_generation_units
    required_time = (
        total_slots * budget.per_search.max_elapsed_seconds
        + budget.proposal_timeout_seconds
    )
    if budget.max_attempts < required_attempts:
        raise ValueError("total attempt budget cannot fund equal paired searches and reserved confirmation")
    if budget.max_generation_units < required_generation:
        raise ValueError("generation budget cannot fund equal paired searches and reserved confirmation")
    if budget.max_elapsed_seconds < required_time:
        raise ValueError("elapsed budget cannot fund equal paired searches and reserved confirmation")

    comparison_id = comparison_id or f"comparison-{uuid.uuid4().hex}"
    if not isinstance(comparison_id, str) or not comparison_id.strip():
        raise ValueError("comparison_id must be a non-empty string")
    started = clock()
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError("clock must return finite monotonic seconds")
    last_tick = started

    def now() -> float:
        nonlocal last_tick
        value = clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < last_tick:
            raise ValueError("clock must return finite monotonic seconds")
        last_tick = value
        return value
    train_observations: list[SearchObservation] = []
    validation_observations: list[SearchObservation] = []
    confirmation_observations: list[SearchObservation] = []
    attempts_used = generation_used = 0
    elapsed_used = 0.0
    challenger: SearchPolicy | None = None
    validation_decision = confirmation_decision = "not_run"
    final_decision = "inconclusive"
    validation_mean = validation_lower = confirmation_mean = confirmation_lower = None
    validation_task_results: tuple[TaskComparison, ...] = ()
    confirmation_task_results: tuple[TaskComparison, ...] = ()
    stop_reason = None

    def record(event: str, payload: dict) -> None:
        nonlocal elapsed_used
        before = now()
        _record(journal, comparison_id, event, payload)
        elapsed_used += now() - before
        if elapsed_used > budget.max_elapsed_seconds:
            raise RuntimeError("journal overhead exhausted the whole-comparison time budget")

    record("comparison_started", {
        "incumbent_policy_id": incumbent.policy_id,
        "incumbent_revision": incumbent.revision,
        "incumbent_sha256": incumbent.sha256,
        "incumbent_specification": incumbent.specification,
        "task_partitions": {
            "training": [t.task_id for t in train],
            "validation": [t.task_id for t in validation],
            "confirmation": [t.task_id for t in confirmation],
        },
        "training_seeds": list(training_seeds),
        "validation_seeds": list(validation_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "per_search_limits": {
            "elapsed_seconds": budget.per_search.max_elapsed_seconds,
            "attempts": budget.per_search.max_attempts,
            "generation_units": budget.per_search.max_generation_units,
        },
        "reserved_confirmation": {
            "elapsed_seconds": reserved_confirmation_time,
            "attempts": reserved_confirmation_attempts,
            "generation_units": reserved_confirmation_generation,
        },
        "rules": {
            "minimum_paired_seeds": rules.minimum_paired_seeds,
            "minimum_gain_fraction": rules.minimum_gain_fraction,
            "maximum_task_regression_fraction": rules.maximum_task_regression_fraction,
            "confidence_z": rules.confidence_z,
            "task_weights": dict(rules.task_weights),
        },
    })

    def execute(policy: SearchPolicy, task: SearchTask, seed: int, phase: str) -> SearchObservation:
        nonlocal attempts_used, generation_used, elapsed_used
        remaining_time = budget.max_elapsed_seconds - elapsed_used
        if remaining_time < budget.per_search.max_elapsed_seconds:
            raise RuntimeError("comparison elapsed budget exhausted before complete equal-budget schedule")
        begin = now()
        quota = SearchGenerationQuota(
            budget.per_search.max_generation_units,
            budget.per_search.max_attempts,
        )
        try:
            evidence, generation_charged, attempts_charged = _invoke_with_timeout(
                executor,
                (policy, task, seed, budget.per_search),
                min(budget.per_search.max_elapsed_seconds, remaining_time),
                resource_quota=quota,
            )
            for _ in range(generation_charged):
                quota.consume()
            for _ in range(attempts_charged):
                quota.consume_attempt()
            measured = now() - begin
            if not isinstance(evidence, SearchEvidence):
                raise TypeError("search executor must return SearchEvidence")
            if evidence.attempts_used > budget.per_search.max_attempts:
                raise RuntimeError("search executor exceeded its attempt budget")
            if evidence.attempts_used != attempts_charged:
                raise RuntimeError("search evidence attempt usage does not match metered usage")
            if evidence.generation_units_used != generation_charged:
                raise RuntimeError("search evidence generation usage does not match metered usage")
            if measured > budget.per_search.max_elapsed_seconds:
                raise RuntimeError("search completed after its allotted time")
        except Exception as exc:
            measured = now() - begin
            elapsed_used += measured
            if isinstance(exc, CallbackFailure):
                for _ in range(exc.used):
                    quota.consume()
                for _ in range(exc.attempts):
                    quota.consume_attempt()
            elif isinstance(exc, CallbackTimeout):
                for _ in range(budget.per_search.max_generation_units - quota.used):
                    quota.consume()
                for _ in range(budget.per_search.max_attempts - quota.attempts_used):
                    quota.consume_attempt()
            attempts_used += quota.attempts_used
            generation_used += quota.used
            record("search_failure", {
                "phase": phase,
                "policy_id": policy.policy_id,
                "policy_revision": policy.revision,
                "policy_sha256": policy.sha256,
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "partition": task.partition.value,
                "seed": seed,
                "failure_category": callback_failure_category(exc),
                "failure": str(exc),
                "attempts_used": quota.attempts_used,
                "generation_units_used": quota.used,
                "elapsed_seconds": measured,
                "remaining_elapsed_seconds": max(0.0, budget.max_elapsed_seconds - elapsed_used),
                "remaining_attempts": budget.max_attempts - attempts_used,
                "remaining_generation_units": budget.max_generation_units - generation_used,
            })
            raise
        attempts_used += quota.attempts_used
        generation_used += quota.used
        elapsed_used += measured
        if attempts_used > budget.max_attempts or generation_used > budget.max_generation_units:
            raise RuntimeError("whole-comparison resource budget exceeded")
        if elapsed_used > budget.max_elapsed_seconds:
            raise RuntimeError("whole-comparison elapsed budget exceeded")
        item = SearchObservation(
            policy.policy_id, policy.revision, policy.sha256, task, seed, evidence, measured
        )
        record("search_observation", {
            "phase": phase,
            **item.to_dict(),
            "input_summary_sha256": hashlib.sha256(
                json.dumps({
                    "task": task.task_id,
                    "task_hash": task.task_sha256,
                    "partition": task.partition.value,
                    "seed": seed,
                    "policy": policy.sha256,
                    "workload": task.workload_sha256,
                    "starting_state": task.starting_state_sha256,
                    "environment": task.environment_sha256,
                    "contract": task.contract_sha256,
                    "generator_version": task.generator_version,
                    "model_configuration": task.model_configuration_sha256,
                }, sort_keys=True).encode()
            ).hexdigest(),
            "remaining_elapsed_seconds": budget.max_elapsed_seconds - elapsed_used,
            "remaining_attempts": budget.max_attempts - attempts_used,
            "remaining_generation_units": budget.max_generation_units - generation_used,
        })
        return item

    try:
        # Training feedback goes only to the proposer, never validation/test data.
        for task in train:
            for seed in training_seeds:
                item = execute(incumbent, task, seed, "training")
                train_observations.append(item)
        train_hash = hashlib.sha256(
            json.dumps([item.to_dict() for item in train_observations], sort_keys=True).encode()
        ).hexdigest()
        proposal_started = now()
        proposal_time_charged = False
        try:
            proposal = _invoke_with_timeout(
                proposer,
                (incumbent, tuple(train_observations)),
                min(budget.proposal_timeout_seconds, budget.max_elapsed_seconds - elapsed_used),
            )
            elapsed_used += now() - proposal_started
            proposal_time_charged = True
            if elapsed_used > budget.max_elapsed_seconds:
                raise RuntimeError("proposal exceeded the whole-comparison elapsed budget")
            if not isinstance(proposal, SearchPolicyProposal):
                raise TypeError("proposer must return SearchPolicyProposal")
        except Exception as exc:
            if not proposal_time_charged:
                elapsed_used += now() - proposal_started
            record("policy_proposal_failed", {
                "incumbent_policy_id": incumbent.policy_id,
                "incumbent_revision": incumbent.revision,
                "incumbent_sha256": incumbent.sha256,
                "training_summary_sha256": train_hash,
                "failure_category": callback_failure_category(exc),
                "failure": str(exc),
            })
            raise
        challenger = proposal.policy
        if challenger.revision <= incumbent.revision:
            raise ValueError("proposer must return a higher-revision SearchPolicy")
        if challenger.sha256 == incumbent.sha256:
            raise ValueError("challenger policy must change the declarative policy contents")
        record("policy_proposed", {
            "challenger_policy_id": challenger.policy_id,
            "challenger_revision": challenger.revision,
            "challenger_sha256": challenger.sha256,
            "challenger_specification": challenger.specification,
            "rationale": proposal.rationale,
            "expected_benefit": proposal.expected_benefit,
            "proposal_input_summary_sha256": train_hash,
        })
        confirmation_reserve = confirmation_slots * budget.per_search.max_elapsed_seconds
        validation_reserve = validation_slots * budget.per_search.max_elapsed_seconds
        if budget.max_elapsed_seconds - elapsed_used < confirmation_reserve + validation_reserve:
            raise RuntimeError("training/proposal costs encroached on validation or reserved confirmation budget")

        def paired_phase(
            tasks: Sequence[SearchTask], seeds: Sequence[int], phase: str
        ) -> list[SearchObservation]:
            result = []
            pair_number = 0
            for task in tasks:
                for seed in seeds:
                    order = (incumbent, challenger) if pair_number % 2 == 0 else (challenger, incumbent)
                    for policy in order:
                        result.append(execute(policy, task, seed, phase))
                    pair_number += 1
            return result

        validation_observations.extend(
            paired_phase(validation, validation_seeds, "validation")
        )
        validation_left = [x for x in validation_observations if x.policy_sha256 == incumbent.sha256]
        validation_right = [x for x in validation_observations if x.policy_sha256 == challenger.sha256]
        validation_decision, validation_mean, validation_lower, validation_task_results = _paired_summary(
            validation_left, validation_right, [t.task_id for t in validation], validation_seeds, rules
        )
        record("validation_decision", {
            "decision": validation_decision,
            "mean_log_gain": validation_mean,
            "lower_confidence_log_gain": validation_lower,
            "task_results": [item.to_dict() for item in validation_task_results],
        })
        if validation_decision != "pass":
            final_decision = (
                "inconclusive" if validation_decision == "inconclusive" else "retain_incumbent"
            )
            stop_reason = "validation did not satisfy the predeclared gain and regression rules"
        else:
            # Test task identities are first exposed to the executor here.
            if budget.max_elapsed_seconds - elapsed_used < confirmation_reserve:
                raise RuntimeError("confirmation budget reservation is no longer available")
            confirmation_observations.extend(
                paired_phase(confirmation, confirmation_seeds, "confirmation")
            )
            confirmation_left = [x for x in confirmation_observations if x.policy_sha256 == incumbent.sha256]
            confirmation_right = [x for x in confirmation_observations if x.policy_sha256 == challenger.sha256]
            confirmation_decision, confirmation_mean, confirmation_lower, confirmation_task_results = _paired_summary(
                confirmation_left,
                confirmation_right,
                [t.task_id for t in confirmation],
                confirmation_seeds,
                rules,
            )
            record("confirmation_decision", {
                "decision": confirmation_decision,
                "mean_log_gain": confirmation_mean,
                "lower_confidence_log_gain": confirmation_lower,
                "task_results": [item.to_dict() for item in confirmation_task_results],
            })
            if confirmation_decision == "pass":
                final_decision = "promote"
                stop_reason = "independent confirmation passed; activation remains controller-owned"
            else:
                final_decision = (
                    "inconclusive" if confirmation_decision == "inconclusive" else "retain_incumbent"
                )
                stop_reason = "confirmation failed or was inconclusive; incumbent retained"
    except Exception as exc:  # noqa: BLE001 -- Any missing comparison evidence blocks promotion.
        stop_reason = f"comparison stopped: {type(exc).__name__}: {exc}"
        final_decision = "inconclusive"

    elapsed = now() - started
    try:
        if final_decision == "promote" and challenger is not None:
            record("policy_promoted", {
                "policy_id": challenger.policy_id,
                "revision": challenger.revision,
                "sha256": challenger.sha256,
                "reason": "validation and independent confirmation passed predeclared rules",
            })
        elif final_decision in {"retain_incumbent", "inconclusive"}:
            record("policy_retained", {
                "incumbent_policy_id": incumbent.policy_id,
                "revision": incumbent.revision,
                "sha256": incumbent.sha256,
                "reason": stop_reason,
            })
    except Exception as exc:  # noqa: BLE001 -- Missing final decision persistence is inconclusive.
        final_decision = "inconclusive"
        stop_reason = f"promotion decision journal failed: {exc}"

    result = ComparisonResult(
        comparison_id,
        incumbent,
        challenger,
        tuple(train_observations),
        tuple(validation_observations),
        tuple(confirmation_observations),
        tuple(validation_seeds),
        tuple(confirmation_seeds),
        validation_decision,
        confirmation_decision,
        final_decision,
        validation_mean,
        validation_lower,
        validation_task_results,
        confirmation_mean,
        confirmation_lower,
        confirmation_task_results,
        elapsed,
        attempts_used,
        generation_used,
        stop_reason,
    )
    try:
        record("comparison_completed", {
            "final_decision": result.final_decision,
            "promoted_policy_sha256": result.promoted_policy.sha256 if result.promoted_policy else None,
            "elapsed_seconds": result.elapsed_seconds,
            "attempts_used": result.attempts_used,
            "generation_units_used": result.generation_units_used,
            "stop_reason": result.stop_reason,
        })
    except Exception:
        # Missing durable provenance makes even a statistically passing result
        # inconclusive. Withdraw a promotion record that has no completion entry.
        promoted = result.final_decision == "promote"
        result = ComparisonResult(
            result.comparison_id, result.incumbent, result.challenger, result.training,
            result.validation, result.confirmation, result.validation_seeds,
            result.confirmation_seeds, result.validation_decision,
            result.confirmation_decision, "inconclusive", result.validation_mean_log_gain,
            result.validation_lower_log_gain, result.validation_task_results,
            result.confirmation_mean_log_gain, result.confirmation_lower_log_gain,
            result.confirmation_task_results, result.elapsed_seconds,
            result.attempts_used, result.generation_units_used,
            "comparison journal failed; provenance incomplete",
        )
        if promoted:
            try:
                record("policy_retained", {
                    "incumbent_policy_id": incumbent.policy_id,
                    "revision": incumbent.revision,
                    "sha256": incumbent.sha256,
                    "reason": result.stop_reason,
                    "withdrawn": True,
                })
            except Exception:
                pass
    return result
