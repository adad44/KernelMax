"""Bounded outer loop for proposing search-policy revisions.

The outer loop runs immutable policies over the existing ``run_session`` inner
loop. It shares one attempt budget across rounds. An outer deadline stops new
rounds and controller callbacks; the inner session still stops on that shared
attempt budget. Before each inner session, trusted controller code applies the
current policy's ranking,
cross-round duplicate avoidance, new-hypothesis allocation, and unchanged-round
stop. A proposed policy is recorded as an experiment input, never as a
promotion: the current session result has no authoritative search score or
paired confirmation gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import signal
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evaluator.schemas import ExperimentContract
from orchestrator.candidate_manager import DEFAULT_CANDIDATES_ROOT
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal, JournalError
from orchestrator.run import Evaluator
from orchestrator.session import SessionResult, run_session


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """Canonical, versioned declarative policy; it cannot execute code."""

    policy_id: str
    revision: int
    specification: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("policy revision must be a positive integer")
        if not isinstance(self.specification, str):
            raise TypeError("policy specification must be canonical JSON text")
        try:
            decoded = json.loads(self.specification)
            canonical = json.dumps(
                decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("policy specification must be finite JSON") from exc
        if not isinstance(decoded, dict) or canonical != self.specification:
            raise ValueError("policy specification must be a canonical JSON object")

    @classmethod
    def create(
        cls, policy_id: str, revision: int, specification: Mapping[str, object]
    ) -> SearchPolicy:
        if not isinstance(specification, Mapping):
            raise TypeError("policy specification must be a mapping")
        encoded = json.dumps(
            dict(specification), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(policy_id, revision, encoded)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.specification.encode("utf-8")).hexdigest()

    @property
    def controls(self) -> dict[str, object]:
        """Return a fresh value so callers cannot mutate the frozen policy."""

        return json.loads(self.specification)


@dataclass(frozen=True, slots=True)
class SearchRound:
    index: int
    policy: SearchPolicy
    session: SessionResult | None
    started_seconds: float
    finished_seconds: float
    candidate_count: int
    error: str | None = None
    skipped_duplicates: tuple[str, ...] = ()
    deferred_candidates: tuple[str, ...] = ()
    ranking: str = "input_order"
    attempts_started: int = 0


@dataclass(frozen=True, slots=True)
class CandidateFeedback:
    candidate_id: str
    outcome: str


@dataclass(frozen=True, slots=True)
class SearchRoundFeedback:
    """Policy-visible summary without raw evaluator or hidden-case details."""

    index: int
    policy_id: str
    policy_revision: int
    policy_sha256: str
    candidates: tuple[CandidateFeedback, ...]
    attempts_started: int
    elapsed_seconds: float
    stop_reason: str
    skipped_duplicates: tuple[str, ...] = ()
    deferred_candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchLoopResult:
    search_id: str
    rounds: tuple[SearchRound, ...]
    attempts_started: int
    attempts_remaining: int
    stop_reason: str
    elapsed_seconds: float
    journal_error: str | None = None

    @property
    def promoted_policy(self) -> None:
        """No automatic promotion exists until authoritative scoring is added."""

        return None

    def summary(self) -> str:
        lines = [
            f"Search {self.search_id}",
            (
                f"Rounds: {len(self.rounds)} | attempts: {self.attempts_started} | "
                f"remaining: {self.attempts_remaining}"
            ),
        ]
        for item in self.rounds:
            state = item.session.stop_reason if item.session else item.error or "not run"
            schedule = ""
            if item.skipped_duplicates or item.deferred_candidates:
                schedule = (
                    f" | skipped {len(item.skipped_duplicates)}"
                    f" | deferred {len(item.deferred_candidates)}"
                )
            lines.append(
                f"{item.index}. {item.policy.policy_id}@{item.policy.revision} "
                f"({item.policy.sha256[:12]}): {state}{schedule}"
            )
        lines.append(f"Search stop reason: {self.stop_reason}")
        lines.append("Policy promotion: none; search scoring/confirmation is unavailable")
        if self.journal_error:
            lines.append(f"Journal error: {self.journal_error}")
        return "\n".join(lines)


CandidateFactory = Callable[
    [SearchPolicy, tuple[SearchRoundFeedback, ...]], Iterable[str | Path]
]
PolicyProposer = Callable[
    [SearchPolicy, tuple[SearchRoundFeedback, ...]], SearchPolicy | None
]


class CallbackTimeout(TimeoutError):
    """A controller callback exceeded its hard subprocess deadline."""


class CallbackFailure(RuntimeError):
    """A callback failed and reports quota usage charged before it failed."""

    def __init__(self, name: str, message: str, used: int = 0, attempts: int = 0):
        super().__init__(f"{name}: {message}")
        self.cause_name = name
        self.used = used
        self.attempts = attempts


class SearchQuota(Protocol):
    used: int
    attempts_used: int
    maximum: int
    max_attempts: int

    def consume(self, units: int = 1) -> None: ...

    def consume_attempt(self) -> None: ...


def _callback_child(connection, callback, args, materialize_iterable: bool, resource_quota) -> None:
    # Give this callback and any nested workers a process group that can be
    # terminated together if its wall-clock allowance expires.
    if multiprocessing.parent_process() is None:
        try:
            os.setsid()
        except OSError:
            pass
    try:
        value = callback(*args, resource_quota) if resource_quota is not None else callback(*args)
        if materialize_iterable:
            value = tuple(value)
        if resource_quota is not None:
            value = (
                value,
                resource_quota.used,
                resource_quota.attempts_used,
            )
        connection.send((True, value))
    except BaseException as exc:  # relay callback failures to the parent
        connection.send((False, (
            type(exc).__name__,
            str(exc),
            resource_quota.used if resource_quota is not None else 0,
            resource_quota.attempts_used if resource_quota is not None else 0,
        )))
    finally:
        connection.close()


def _invoke_with_timeout(
    callback, args, timeout_seconds: float, *, materialize_iterable=False, resource_quota=None
):
    """Run trusted control callbacks in a killable worker process.

    The spawn context is deliberate: a timeout must terminate the callback,
    rather than leave a background thread mutating controller state.
    Callbacks must therefore be importable/pickleable module-level callables.
    """

    context = multiprocessing.get_context("spawn")
    initial_used = resource_quota.used if resource_quota is not None else 0
    initial_attempts = resource_quota.attempts_used if resource_quota is not None else 0
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_callback_child,
        args=(child, callback, args, materialize_iterable, resource_quota),
        daemon=False,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.join()
            raise CallbackTimeout(
                f"controller callback exceeded {timeout_seconds:.3f}s timeout"
            )
        ok, value = parent.recv()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join()
        if not ok:
            name, message, used, attempts = value
            raise CallbackFailure(
                name,
                message,
                max(0, used - initial_used),
                max(0, attempts - initial_attempts),
            )
        if resource_quota is not None:
            result, used, attempts = value
            return result, used - initial_used, attempts - initial_attempts
        return value
    finally:
        parent.close()
        if process.exitcode is not None:
            process.close()


def _feedback(rounds: Sequence[SearchRound]) -> tuple[SearchRoundFeedback, ...]:
    summary = []
    for item in rounds:
        session = item.session
        candidates = (
            tuple(
                CandidateFeedback(
                    submission.candidate_id,
                    submission.outcome.value,
                )
                for submission in session.submissions
            )
            if session is not None
            else ()
        )
        summary.append(
            SearchRoundFeedback(
                index=item.index,
                policy_id=item.policy.policy_id,
                policy_revision=item.policy.revision,
                policy_sha256=item.policy.sha256,
                candidates=candidates,
                attempts_started=item.attempts_started,
                elapsed_seconds=item.finished_seconds - item.started_seconds,
                stop_reason=(session.stop_reason if session else item.error or "failed"),
                skipped_duplicates=item.skipped_duplicates,
                deferred_candidates=item.deferred_candidates,
            )
        )
    return tuple(summary)


_RANKINGS = frozenset({"input_order", "unseen_first"})


@dataclass(frozen=True, slots=True)
class PolicyControls:
    """Scheduling keys interpreted by the controller. Other keys stay factory data."""

    ranking: str = "input_order"
    avoid_duplicates: bool = False
    new_hypothesis_fraction: float | None = None
    stop_after_unchanged_rounds: int | None = None


def scheduling_controls(policy: SearchPolicy) -> PolicyControls:
    """Validate declarative scheduling keys without executing policy text."""

    if not isinstance(policy, SearchPolicy):
        raise TypeError("policy must be a SearchPolicy")
    controls = policy.controls
    ranking = controls.get("ranking", "input_order")
    if ranking not in _RANKINGS:
        raise ValueError("ranking must be input_order or unseen_first")
    avoid = controls.get("avoid_duplicates", False)
    if type(avoid) is not bool:
        raise ValueError("avoid_duplicates must be a bool")
    fraction = controls.get("new_hypothesis_fraction", None)
    if fraction is not None and (
        type(fraction) not in (int, float)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("new_hypothesis_fraction must be finite and in [0, 1]")
    stop_after = controls.get("stop_after_unchanged_rounds", None)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after_unchanged_rounds must be a positive integer")
    return PolicyControls(
        ranking=str(ranking),
        avoid_duplicates=avoid,
        new_hypothesis_fraction=None if fraction is None else float(fraction),
        stop_after_unchanged_rounds=stop_after,
    )


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    selected: tuple[str | Path, ...]
    skipped_duplicates: tuple[str, ...]
    deferred_candidates: tuple[str, ...]
    ranking: str

    def journal_fields(self) -> dict[str, object]:
        return {
            "ranking": self.ranking,
            "scheduled_candidate_count": len(self.selected),
            "skipped_duplicates": list(self.skipped_duplicates),
            "deferred_candidates": list(self.deferred_candidates),
        }


def _attempted_candidate_ids(rounds: Sequence[SearchRound]) -> set[str]:
    seen: set[str] = set()
    for item in rounds:
        session = item.session
        if session is None:
            continue
        for submission in session.submissions:
            if submission.result is not None:
                seen.add(submission.candidate_id)
    return seen


def _has_new_attempt(session: SessionResult | None, seen_before: set[str]) -> bool:
    if session is None:
        return False
    return any(
        submission.result is not None and submission.candidate_id not in seen_before
        for submission in session.submissions
    )


def schedule_candidates(
    candidates: Sequence[str | Path],
    *,
    seen_ids: set[str],
    attempts_remaining: int,
    controls: PolicyControls,
) -> ScheduleDecision:
    """Choose which factory paths may enter the inner session.

    Admission uses controller-computed candidate IDs, the submission directory
    names ``run_session`` already uses. ``new_hypothesis_fraction`` limits how
    many new IDs are admitted relative to the attempts still remaining. It does
    not raise the shared attempt cap. Within-round repeats of an admitted ID
    stay in the list so the inner session can record them as duplicates.
    """

    if not isinstance(controls, PolicyControls):
        raise TypeError("controls must be PolicyControls")
    if type(attempts_remaining) is not int or attempts_remaining < 0:
        raise ValueError("attempts_remaining must be a non-negative integer")
    ordered: list[tuple[str, str | Path, bool]] = []
    for path in candidates:
        if not isinstance(path, (str, Path)):
            raise TypeError("candidate_factory must return only paths")
        candidate_id = Path(path).name
        ordered.append((candidate_id, path, candidate_id not in seen_ids))
    if controls.ranking == "unseen_first":
        ordered = [item for item in ordered if item[2]] + [
            item for item in ordered if not item[2]
        ]

    new_limit = None
    refinement_limit = None
    if controls.new_hypothesis_fraction is not None:
        new_limit = int(math.floor(controls.new_hypothesis_fraction * attempts_remaining))
        refinement_limit = attempts_remaining - new_limit

    selected: list[str | Path] = []
    skipped: list[str] = []
    deferred: list[str] = []
    admitted: set[str] = set()
    skipped_ids: set[str] = set()
    deferred_ids: set[str] = set()
    new_taken = 0
    refinement_taken = 0
    for candidate_id, path, is_new in ordered:
        if candidate_id in admitted:
            selected.append(path)
            continue
        if candidate_id in skipped_ids:
            skipped.append(candidate_id)
            continue
        if candidate_id in deferred_ids:
            deferred.append(candidate_id)
            continue
        if not is_new and controls.avoid_duplicates:
            skipped.append(candidate_id)
            skipped_ids.add(candidate_id)
            continue
        if is_new and new_limit is not None and new_taken >= new_limit:
            deferred.append(candidate_id)
            deferred_ids.add(candidate_id)
            continue
        if (
            not is_new
            and refinement_limit is not None
            and refinement_taken >= refinement_limit
        ):
            deferred.append(candidate_id)
            deferred_ids.add(candidate_id)
            continue
        selected.append(path)
        admitted.add(candidate_id)
        if is_new:
            new_taken += 1
        else:
            refinement_taken += 1
    return ScheduleDecision(
        tuple(selected),
        tuple(skipped),
        tuple(deferred),
        controls.ranking,
    )


def callback_failure_category(exc: BaseException) -> str:
    """Prefer the callback's own exception name over the controller wrapper."""

    cause = getattr(exc, "cause_name", None)
    if isinstance(cause, str) and cause.strip():
        return cause
    return type(exc).__name__


def _run_search_loop_impl(
    *,
    initial_policy: SearchPolicy,
    candidate_factory: CandidateFactory,
    policy_proposer: PolicyProposer,
    max_rounds: int,
    evaluator: Evaluator,
    contract: ExperimentContract,
    budget: ExperimentBudget,
    journal: ExperimentJournal,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
    search_id: str | None = None,
    max_elapsed_seconds: float | None = None,
    callback_timeout_seconds: float = 300.0,
    generation_quota: SearchQuota | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SearchLoopResult:
    """Run an outer policy-proposal loop around the existing inner session.

    ``candidate_factory`` and ``policy_proposer`` are trusted controller
    integrations. The proposer sees sanitized round summaries and may return a
    new declarative policy for the next exploratory round. Scheduling keys on
    that policy are applied by this controller before the inner session. The
    proposer cannot change the contract, evaluator, attempt cap, or deadline.
    Results are retained; this function never selects a winner or promotes a
    policy.
    """

    if not isinstance(initial_policy, SearchPolicy):
        raise TypeError("initial_policy must be a SearchPolicy")
    scheduling_controls(initial_policy)
    if not callable(candidate_factory) or not callable(policy_proposer):
        raise TypeError("candidate_factory and policy_proposer must be callable")
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    if not isinstance(contract, ExperimentContract):
        raise TypeError("contract must be a validated ExperimentContract")
    if not isinstance(budget, ExperimentBudget):
        raise TypeError("budget must be an ExperimentBudget")
    if not isinstance(journal, ExperimentJournal):
        raise TypeError("journal must be an ExperimentJournal")
    if not callable(clock):
        raise TypeError("clock must be callable")
    started = clock()
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError("clock must return finite monotonic seconds")
    if max_elapsed_seconds is not None and (
        type(max_elapsed_seconds) not in (int, float)
        or not math.isfinite(max_elapsed_seconds)
        or max_elapsed_seconds <= 0
    ):
        raise ValueError("max_elapsed_seconds must be finite and positive")
    if (
        type(callback_timeout_seconds) not in (int, float)
        or not math.isfinite(callback_timeout_seconds)
        or callback_timeout_seconds <= 0
    ):
        raise ValueError("callback_timeout_seconds must be finite and positive")
    if search_id is None:
        search_id = f"search-{uuid.uuid4().hex}"
    if not isinstance(search_id, str) or not search_id.strip():
        raise ValueError("search_id must be a non-empty string")
    if generation_quota is not None and (
        not callable(getattr(generation_quota, "consume", None))
        or not callable(getattr(generation_quota, "consume_attempt", None))
        or budget.max_attempts > generation_quota.max_attempts - generation_quota.attempts_used
    ):
        raise ValueError("generation_quota must provide sufficient attempt and generation limits")

    deadline = started + max_elapsed_seconds if max_elapsed_seconds else None
    last = started

    def now() -> float:
        nonlocal last
        value = clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < last:
            raise ValueError("clock must return finite monotonic seconds")
        last = value
        return value

    rounds: list[SearchRound] = []
    journal_error = None
    stop_reason = "round limit reached"

    def record(event: str, payload: dict) -> bool:
        nonlocal journal_error
        try:
            journal.record_session(search_id, event, {"search_id": search_id, **payload})
            return True
        except JournalError as exc:
            journal_error = f"search journal failed: {exc}"
            budget.stop(journal_error)
            return False

    record(
        "started",
        {
            "loop_type": "policy_proposal",
            "max_rounds": max_rounds,
            "max_attempts": budget.max_attempts,
            "attempts_started": budget.attempts_started,
            "max_elapsed_seconds": max_elapsed_seconds,
            "contract_sha256": contract.sha256,
            "initial_policy_id": initial_policy.policy_id,
            "initial_policy_revision": initial_policy.revision,
            "initial_policy_sha256": initial_policy.sha256,
            "initial_policy_specification": initial_policy.specification,
        },
    )

    policy = initial_policy
    unchanged_streak = 0
    for index in range(1, max_rounds + 1):
        round_started = now()
        if budget.must_stop:
            stop_reason = budget.stop_reason or "maximum total attempts reached"
            break
        if deadline is not None and round_started >= deadline:
            budget.stop("outer search deadline reached")
            stop_reason = budget.stop_reason
            break

        try:
            history = _feedback(rounds)
            callback_timeout = callback_timeout_seconds
            if deadline is not None:
                callback_timeout = min(callback_timeout, max(0.0, deadline - now()))
            if callback_timeout <= 0:
                raise CallbackTimeout("search deadline reached before candidate generation")
            if generation_quota is None:
                candidates = _invoke_with_timeout(
                    candidate_factory,
                    (policy, history),
                    callback_timeout,
                    materialize_iterable=True,
                )
            else:
                candidates, generated_units, factory_attempts = _invoke_with_timeout(
                    candidate_factory,
                    (policy, history),
                    callback_timeout,
                    materialize_iterable=True,
                    resource_quota=generation_quota,
                )
                for _ in range(generated_units):
                    generation_quota.consume()
                for _ in range(factory_attempts):
                    generation_quota.consume_attempt()
            if any(not isinstance(path, (str, Path)) for path in candidates):
                raise TypeError("candidate_factory must return only paths")
        except Exception as exc:  # noqa: BLE001 -- Preserve policy-loop failures.
            if generation_quota is not None:
                if isinstance(exc, CallbackFailure):
                    for _ in range(exc.used):
                        generation_quota.consume()
                    for _ in range(exc.attempts):
                        generation_quota.consume_attempt()
                elif isinstance(exc, CallbackTimeout):
                    for _ in range(generation_quota.maximum - generation_quota.used):
                        generation_quota.consume()
                    for _ in range(generation_quota.max_attempts - generation_quota.attempts_used):
                        generation_quota.consume_attempt()
            reason = f"candidate generation failed: {type(exc).__name__}: {exc}"
            budget.stop(reason)
            finished = now()
            item = SearchRound(index, policy, None, round_started, finished, 0, reason)
            rounds.append(item)
            record(
                "round",
                {
                    "round": index,
                    "policy_id": policy.policy_id,
                    "policy_revision": policy.revision,
                    "policy_sha256": policy.sha256,
                    "candidate_count": 0,
                    "error": reason,
                },
            )
            stop_reason = reason
            break

        seen_before = _attempted_candidate_ids(rounds)
        try:
            controls = scheduling_controls(policy)
            decision = schedule_candidates(
                candidates,
                seen_ids=seen_before,
                attempts_remaining=budget.attempts_remaining,
                controls=controls,
            )
        except (TypeError, ValueError) as exc:
            reason = f"search policy controls rejected: {type(exc).__name__}: {exc}"
            budget.stop(reason)
            finished = now()
            item = SearchRound(index, policy, None, round_started, finished, 0, reason)
            rounds.append(item)
            record(
                "round",
                {
                    "round": index,
                    "policy_id": policy.policy_id,
                    "policy_revision": policy.revision,
                    "policy_sha256": policy.sha256,
                    "candidate_count": 0,
                    "error": reason,
                },
            )
            stop_reason = reason
            break
        scheduled = decision.selected
        schedule_record = decision.journal_fields()

        remaining = None if deadline is None else max(0.0, deadline - now())
        if remaining == 0:
            reason = "outer search deadline reached during candidate generation"
            budget.stop(reason)
            finished = now()
            item = SearchRound(
                index,
                policy,
                None,
                round_started,
                finished,
                len(scheduled),
                reason,
                skipped_duplicates=decision.skipped_duplicates,
                deferred_candidates=decision.deferred_candidates,
                ranking=decision.ranking,
            )
            rounds.append(item)
            record(
                "round",
                {
                    "round": index,
                    "policy_id": policy.policy_id,
                    "policy_revision": policy.revision,
                    "policy_sha256": policy.sha256,
                    "candidate_count": len(scheduled),
                    "error": reason,
                    **schedule_record,
                },
            )
            stop_reason = reason
            break
        attempts_before = budget.attempts_started
        try:
            session = run_session(
                scheduled,
                evaluator=evaluator,
                contract=contract,
                budget=budget,
                journal=journal,
                frozen_root=Path(frozen_root) / search_id,
                candidates_root=candidates_root,
                session_id=f"{search_id}-round-{index}",
            )
            if generation_quota is not None:
                for _ in range(budget.attempts_started - attempts_before):
                    generation_quota.consume_attempt()
            finished = now()
            item = SearchRound(
                index,
                policy,
                session,
                round_started,
                finished,
                len(scheduled),
                skipped_duplicates=decision.skipped_duplicates,
                deferred_candidates=decision.deferred_candidates,
                ranking=decision.ranking,
                attempts_started=budget.attempts_started - attempts_before,
            )
            rounds.append(item)
        except Exception as exc:  # noqa: BLE001 -- Stop on controller failures.
            reason = f"inner search failed: {type(exc).__name__}: {exc}"
            budget.stop(reason)
            finished = now()
            item = SearchRound(
                index,
                policy,
                None,
                round_started,
                finished,
                len(scheduled),
                reason,
                skipped_duplicates=decision.skipped_duplicates,
                deferred_candidates=decision.deferred_candidates,
                ranking=decision.ranking,
            )
            rounds.append(item)
            stop_reason = reason
            record(
                "round",
                {
                    "round": index,
                    "policy_id": policy.policy_id,
                    "policy_revision": policy.revision,
                    "policy_sha256": policy.sha256,
                    "candidate_count": len(scheduled),
                    "error": reason,
                    **schedule_record,
                },
            )
            break

        if not record(
            "round",
            {
                "round": index,
                "policy_id": policy.policy_id,
                "policy_revision": policy.revision,
                "policy_sha256": policy.sha256,
                "candidate_count": len(scheduled),
                "session_id": session.session_id,
                "session_stop_reason": session.stop_reason,
                "attempts_started": item.attempts_started,
                "attempts_remaining": budget.attempts_remaining,
                "elapsed_seconds": finished - round_started,
                **schedule_record,
            },
        ):
            stop_reason = journal_error or "search journal failed"
            break
        if _has_new_attempt(session, seen_before):
            unchanged_streak = 0
        else:
            unchanged_streak += 1
        if budget.must_stop:
            stop_reason = budget.stop_reason or "maximum total attempts reached"
            break
        if deadline is not None and now() >= deadline:
            budget.stop("outer search deadline reached")
            stop_reason = budget.stop_reason
            break
        if index == max_rounds:
            stop_reason = "round limit reached"
            break
        if (
            controls.stop_after_unchanged_rounds is not None
            and unchanged_streak >= controls.stop_after_unchanged_rounds
        ):
            stop_reason = "policy stop: no new hypotheses"
            break

        try:
            callback_timeout = callback_timeout_seconds
            if deadline is not None:
                callback_timeout = min(callback_timeout, max(0.0, deadline - now()))
            if callback_timeout <= 0:
                raise CallbackTimeout("search deadline reached before policy proposal")
            proposal = _invoke_with_timeout(
                policy_proposer, (policy, _feedback(rounds)), callback_timeout
            )
            if proposal is not None:
                if not isinstance(proposal, SearchPolicy):
                    raise TypeError("policy_proposer must return SearchPolicy or None")
                if proposal.revision <= policy.revision:
                    raise ValueError("a revised policy must advance its revision")
                scheduling_controls(proposal)
                if not record(
                    "policy_proposed",
                    {
                        "after_round": index,
                        "incumbent_policy_id": policy.policy_id,
                        "incumbent_revision": policy.revision,
                        "incumbent_sha256": policy.sha256,
                        "proposal_policy_id": proposal.policy_id,
                        "proposal_revision": proposal.revision,
                        "proposal_sha256": proposal.sha256,
                        "proposal_specification": proposal.specification,
                        "activation": "exploratory next round only; not promoted",
                    },
                ):
                    stop_reason = journal_error or "search journal failed"
                    break
                policy = proposal
            else:
                stop_reason = "policy proposer stopped"
                break
        except Exception as exc:  # noqa: BLE001 -- Preserve failed proposals.
            reason = f"policy proposal failed: {type(exc).__name__}: {exc}"
            budget.stop(reason)
            stop_reason = reason
            record(
                "policy_proposed",
                {"after_round": index, "error": reason, "proposal_sha256": None},
            )
            break

    elapsed = now() - started
    result = SearchLoopResult(
        search_id=search_id,
        rounds=tuple(rounds),
        attempts_started=budget.attempts_started,
        attempts_remaining=budget.attempts_remaining,
        stop_reason=budget.stop_reason or stop_reason,
        elapsed_seconds=elapsed,
        journal_error=journal_error,
    )
    record(
        "completed",
        {
            "loop_type": "policy_proposal",
            "rounds_completed": len(rounds),
            "attempts_started": result.attempts_started,
            "attempts_remaining": result.attempts_remaining,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
            "policy_promotion": None,
        },
    )
    return SearchLoopResult(
        search_id=result.search_id,
        rounds=result.rounds,
        attempts_started=result.attempts_started,
        attempts_remaining=result.attempts_remaining,
        stop_reason=budget.stop_reason or result.stop_reason,
        elapsed_seconds=now() - started,
        journal_error=journal_error,
    )


def run_search_loop(
    *,
    initial_policy: SearchPolicy,
    candidate_factory: CandidateFactory,
    policy_proposer: PolicyProposer,
    max_rounds: int,
    evaluator: Evaluator,
    contract: ExperimentContract,
    budget: ExperimentBudget,
    journal: ExperimentJournal,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
    search_id: str | None = None,
    max_elapsed_seconds: float | None = None,
    callback_timeout_seconds: float = 300.0,
    generation_quota: SearchQuota | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SearchLoopResult:
    """Run policy rounds over the existing shared-budget inner session."""

    if not isinstance(budget, ExperimentBudget):
        raise TypeError("budget must be an ExperimentBudget")
    return _run_search_loop_impl(
        initial_policy=initial_policy,
        candidate_factory=candidate_factory,
        policy_proposer=policy_proposer,
        max_rounds=max_rounds,
        evaluator=evaluator,
        contract=contract,
        budget=budget,
        journal=journal,
        frozen_root=frozen_root,
        candidates_root=candidates_root,
        search_id=search_id,
        max_elapsed_seconds=max_elapsed_seconds,
        callback_timeout_seconds=callback_timeout_seconds,
        generation_quota=generation_quota,
        clock=clock,
    )
