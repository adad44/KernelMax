"""Bounded search accounting through the real runner, without models or sleeps."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from evaluator.candidate_execution import ReviewedCandidateEvaluator
from evaluator.schemas import RunStatus, Verdict, VerdictResult
from orchestrator.candidate_manager import freeze_candidate
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal, JournalError
from orchestrator.run import run_candidate
from orchestrator.session import STRATEGY_ID, SubmissionOutcome, run_session
from tests.test_orchestrator_session import CONTRACT, make_submission, report_for


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class BoundedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()
        self.journal = ExperimentJournal(self.root / "journal")
        self.clock = FakeClock()
        self.session_count = 0
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for path in self.root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def submission(self, **kwargs):
        return make_submission(self.candidates, **kwargs)

    def run_search(self, directories, evaluator=None, budget=None, **kwargs):
        self.session_count += 1
        return run_session(
            directories,
            evaluator=evaluator or (lambda run, *a, **k: report_for(run)),
            contract=CONTRACT,
            budget=budget or ExperimentBudget(max_attempts=5),
            journal=self.journal,
            frozen_root=self.root / "frozen",
            candidates_root=self.candidates,
            max_elapsed_seconds=kwargs.pop("max_elapsed_seconds", 10),
            clock=self.clock,
            **kwargs,
        )

    def test_existing_runner_order_duplicates_and_fixed_provenance(self):
        first, second = self.submission(), self.submission()
        calls = []

        def evaluator(run, *args, **kwargs):
            calls.append(run.candidate.candidate_id)
            self.clock.advance(2)
            return report_for(run)

        identities = {"generator": "fixture-v1", "controller_revision": "test-revision"}
        with patch("orchestrator.session.run_candidate", wraps=run_candidate) as runner:
            result = self.run_search(
                [first, first, second], evaluator, configuration_ids=identities
            )
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(calls, [first.name, second.name])
        self.assertEqual(result.submissions[1].outcome, SubmissionOutcome.DUPLICATE)
        self.assertEqual(result.elapsed_seconds, 4)
        self.assertEqual(result.stop_reason, "submissions exhausted")
        self.assertEqual(result.provenance.strategy_id, STRATEGY_ID)
        self.assertEqual(result.provenance.contract_sha256, CONTRACT.sha256)
        self.assertEqual(result.provenance.workload_seed, CONTRACT.workload.seed)
        identities["generator"] = "mutated"
        self.assertEqual(
            dict(result.provenance.configuration_ids)["generator"], "fixture-v1"
        )
        with self.assertRaises(FrozenInstanceError):
            result.provenance.strategy_id = "changed"

    def test_deadline_retains_late_report_and_prevents_next_submission(self):
        first, second = self.submission(), self.submission()
        reports = []

        def evaluator(run, *args, **kwargs):
            self.clock.advance(11)
            report = replace(
                report_for(run),
                verdict=VerdictResult(
                    run.candidate,
                    run.contract_sha256,
                    Verdict.ACCEPTED,
                    "synthetic claim",
                ),
            )
            reports.append(report)
            return report

        result = self.run_search([first, second], evaluator)
        self.assertIs(result.submissions[0].report, reports[0])
        self.assertIs(result.submissions[0].result.state.evaluator_report, reports[0])
        self.assertFalse(result.submissions[0].attempts[0].within_deadline)
        self.assertEqual(
            result.submissions[1].outcome, SubmissionOutcome.BUDGET_EXHAUSTED
        )
        self.assertIsNone(result.score)
        self.assertIsNone(result.best_run_id)
        self.assertEqual(result.stop_reason, "session deadline reached")
        self.assertEqual(result.to_dict()["remaining_seconds"], 0)
        self.assertEqual(result.attempts_started, 1)

    def test_exact_deadline_is_a_scheduling_boundary(self):
        first, second = self.submission(), self.submission()

        def evaluator(run, *args, **kwargs):
            self.clock.advance(10)
            return report_for(run)

        result = self.run_search([first, second], evaluator)
        self.assertTrue(result.submissions[0].attempts[0].within_deadline)
        self.assertEqual(result.attempts_started, 1)
        self.assertEqual(result.stop_reason, "session deadline reached")

    def test_collection_cost_counts_and_can_exhaust_time_before_evaluation(self):
        candidate = self.submission()

        def collection():
            self.clock.advance(12)
            yield candidate

        with patch("orchestrator.session.run_candidate", wraps=run_candidate) as runner:
            result = self.run_search(collection())
        runner.assert_not_called()
        self.assertEqual(result.elapsed_seconds, 12)
        self.assertEqual(
            next(
                d for d in result.durations if d.stage == "submission_collection"
            ).elapsed_seconds,
            12,
        )
        self.assertEqual(
            result.submissions[0].outcome, SubmissionOutcome.BUDGET_EXHAUSTED
        )

    def test_freezing_past_deadline_does_not_start_an_evaluator_attempt(self):
        candidate = self.submission()
        calls = []

        def slow_freeze(*args, **kwargs):
            frozen = freeze_candidate(*args, **kwargs)
            self.clock.advance(11)
            return frozen

        with patch("orchestrator.run.freeze_candidate", side_effect=slow_freeze):
            result = self.run_search([candidate], lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])
        self.assertEqual(result.attempts_started, 0)
        self.assertEqual(result.stop_reason, "session deadline reached")
        self.assertIn(
            "freezing", [d.stage for d in result.submissions[0].result.durations]
        )

    def test_retries_reuse_frozen_identity_and_share_total_budget(self):
        first, second, third = self.submission(), self.submission(), self.submission()
        reports, snapshots, runs, classified = [], [], [], []

        def evaluator(run, frozen, *args, **kwargs):
            self.clock.advance(1)
            snapshots.append(frozen)
            runs.append(run)
            report = report_for(run)
            if len(runs) == 1:
                report = replace(
                    report, status=RunStatus.FAILED, errors=("temporary fixture",)
                )
            reports.append(report)
            return report

        def classifier(report):
            classified.append(report)
            return True

        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        result = self.run_search(
            [first, second, third], evaluator, budget, retry_classifier=classifier
        )
        self.assertEqual(
            [r.candidate.candidate_id for r in runs],
            [first.name, first.name, second.name],
        )
        self.assertEqual((result.attempts_started, result.retries_started), (3, 1))
        self.assertIs(snapshots[0], snapshots[1])
        self.assertEqual(runs[0].candidate, runs[1].candidate)
        self.assertNotEqual(runs[0].run_id, runs[1].run_id)
        self.assertEqual(classified, reports[:1])
        first_attempts = result.submissions[0].attempts
        self.assertEqual(
            [a.result.attempt.retry_number for a in first_attempts], [0, 1]
        )
        self.assertIs(first_attempts[0].result.report, reports[0])
        self.assertTrue(first_attempts[0].retry_classification)
        self.assertEqual(
            result.submissions[2].outcome, SubmissionOutcome.BUDGET_EXHAUSTED
        )
        self.assertEqual(result.elapsed_seconds, 3)

    def test_per_candidate_retry_cap_survives_next_candidate(self):
        first, second = self.submission(), self.submission()

        def failed(run, *args, **kwargs):
            return replace(
                report_for(run), status=RunStatus.FAILED, errors=("transient",)
            )

        budget = ExperimentBudget(max_attempts=5, max_retries_per_candidate=1)
        result = self.run_search(
            [first, second], failed, budget, retry_classifier=lambda r: True
        )
        self.assertEqual([len(s.attempts) for s in result.submissions], [2, 2])
        self.assertEqual(budget.retries_by_candidate, {first.name: 1, second.name: 1})
        self.assertEqual(result.attempts_started, 4)

    def test_default_and_negative_classification_never_retry(self):
        def failed(run, *args, **kwargs):
            return replace(
                report_for(run), status=RunStatus.FAILED, errors=("failure",)
            )

        for classifier in (None, lambda report: False):
            result = self.run_search(
                [self.submission()],
                failed,
                ExperimentBudget(max_attempts=3, max_retries_per_candidate=2),
                retry_classifier=classifier,
            )
            self.assertEqual((result.attempts_started, result.retries_started), (1, 0))
            self.assertEqual(result.submissions[0].outcome, SubmissionOutcome.FAILED)

    def test_final_verdicts_and_interruptions_do_not_reach_classifier(self):
        for status, verdict in (
            (RunStatus.COMPLETED, Verdict.REJECTED),
            (RunStatus.COMPLETED, Verdict.INCONCLUSIVE),
            (RunStatus.FAILED, Verdict.REJECTED),
            (RunStatus.INTERRUPTED, None),
        ):
            with self.subTest(status=status, verdict=verdict):

                def evaluator(run, *args, status=status, verdict=verdict, **kwargs):
                    return replace(
                        report_for(run),
                        status=status,
                        errors=("fixture failure",)
                        if status is not RunStatus.COMPLETED
                        else (),
                        verdict=VerdictResult(
                            run.candidate, run.contract_sha256, verdict, "fixture final"
                        )
                        if verdict
                        else None,
                    )

                classified = []
                result = self.run_search(
                    [self.submission()],
                    evaluator,
                    ExperimentBudget(max_attempts=4, max_retries_per_candidate=3),
                    retry_classifier=lambda r, classified=classified: (
                        classified.append(r) or True
                    ),
                )
                self.assertEqual(classified, [])
                self.assertEqual(result.retries_started, 0)
                self.assertEqual(
                    result.submissions[0].report.verdict.verdict if verdict else None,
                    verdict,
                )

    def test_deadline_prevents_retry_even_for_transient_failure(self):
        def failed(run, *args, **kwargs):
            self.clock.advance(11)
            return replace(
                report_for(run), status=RunStatus.FAILED, errors=("transient",)
            )

        result = self.run_search(
            [self.submission()],
            failed,
            ExperimentBudget(max_attempts=3, max_retries_per_candidate=2),
            retry_classifier=lambda report: True,
        )
        self.assertEqual(result.attempts_started, 1)
        self.assertEqual(result.retries_started, 0)

    def test_timeout_and_exception_failures_are_retained_without_fabricated_report(
        self,
    ):
        first, second = self.submission(), self.submission()

        def evaluator(run, *args, **kwargs):
            self.clock.advance(2)
            if run.candidate.candidate_id == first.name:
                raise TimeoutError("worker deadline")
            return report_for(run)

        result = self.run_search(
            [first, second], evaluator, retry_classifier=lambda r: True
        )
        self.assertEqual(result.submissions[0].outcome, SubmissionOutcome.FAILED)
        self.assertIn("TimeoutError", result.submissions[0].detail)
        self.assertIsNone(result.submissions[0].report)
        self.assertEqual(result.retries_started, 0)
        self.assertEqual(result.elapsed_seconds, 4)

    def test_reviewed_worker_timeout_is_capped_without_mutating_evaluator(self):
        evaluator = ReviewedCandidateEvaluator(
            frozenset(), self.root / "artifacts", timeout_seconds=600
        )
        seen = []

        def fake_call(instance, run, *args, **kwargs):
            seen.append(instance.timeout_seconds)
            self.clock.advance(1)
            return report_for(run)

        with patch.object(ReviewedCandidateEvaluator, "__call__", new=fake_call):
            result = self.run_search([self.submission(), self.submission()], evaluator)
        self.assertEqual(seen, [5, 4.5])
        self.assertEqual(evaluator.timeout_seconds, 600)
        self.assertEqual(result.attempts_started, 2)

    def test_sessions_share_no_mutable_search_state(self):
        candidate = self.submission()
        first = self.run_search([candidate], configuration_ids={"generator": "first"})
        second = self.run_search([candidate])
        self.assertNotEqual(first.search_id, second.search_id)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual((first.attempts_started, second.attempts_started), (1, 1))
        self.assertNotEqual(
            first.submissions[0].result.frozen_candidate.directory,
            second.submissions[0].result.frozen_candidate.directory,
        )
        self.assertNotEqual(
            first.submissions[0].result.attempt.run_id,
            second.submissions[0].result.attempt.run_id,
        )
        self.assertEqual(dict(first.provenance.configuration_ids)["generator"], "first")
        self.assertIsNone(dict(second.provenance.configuration_ids)["generator"])

    def test_summary_and_journal_link_reports_and_record_costs_without_zero_filling(
        self,
    ):
        def evaluator(run, *args, **kwargs):
            self.clock.advance(3)
            return report_for(run)

        result = self.run_search([self.submission()], evaluator)
        records = [
            r for r in self.journal.read_history() if r["kind"] == "search_session"
        ]
        self.assertEqual(
            [r["session_event"] for r in records],
            ["started", "submission", "completed"],
        )
        self.assertEqual(
            {r["payload"]["search_id"] for r in records}, {result.search_id}
        )
        saved = records[-1]["payload"]
        self.assertEqual(saved["elapsed_seconds"], 3)
        self.assertEqual(saved["attempts_remaining"], 4)
        self.assertEqual(saved["stop_reason"], result.stop_reason)
        observation = saved["submissions"][0]["attempts"][0]
        report = result.submissions[0].report
        self.assertEqual(observation["report_run_id"], report.run_id)
        self.assertEqual(observation["candidate"], report.candidate.to_dict())
        durations = {d["stage"]: d["elapsed_seconds"] for d in observation["durations"]}
        self.assertEqual(durations["evaluation_call"], 3)
        self.assertEqual(durations["runner_total"], 3)
        self.assertIsNone(
            saved["accounting"]["stage_breakdown_unavailable"]["compilation"]
        )
        self.assertIsNone(saved["provenance"]["configuration_ids"]["environment"])
        self.assertIn("partial accounting", result.summary())
        self.assertIn("unscored", result.summary())
        terminal = next(
            r
            for r in self.journal.read_history()
            if r["kind"] == "state_transition"
            and r["event"]["type"] == "evaluator_result"
        )
        self.assertEqual(terminal["event"]["report"], report.to_dict())
        json.dumps(result.to_dict(), allow_nan=False)

    def test_disabled_gates_cannot_turn_claimed_acceptance_into_a_search_score(self):
        def accepted(run, *args, **kwargs):
            return replace(
                report_for(run),
                verdict=VerdictResult(
                    run.candidate,
                    run.contract_sha256,
                    Verdict.ACCEPTED,
                    "fake unsupported claim",
                ),
            )

        result = self.run_search([self.submission()], accepted)
        self.assertEqual(result.submissions[0].report.verdict.verdict, Verdict.ACCEPTED)
        self.assertIsNone(result.score)
        self.assertIsNone(result.best_run_id)
        self.assertIn("task-quality", result.scoring_reason)

    def test_session_journal_failure_stops_and_retains_completed_results(self):
        original = self.journal.record_session

        def fail(event_session, event, payload):
            if event == "submission":
                raise JournalError("session disk full")
            original(event_session, event, payload)

        with patch.object(self.journal, "record_session", side_effect=fail):
            result = self.run_search([self.submission(), self.submission()])
        self.assertEqual(result.attempts_started, 1)
        self.assertIsNotNone(result.submissions[0].report)
        self.assertEqual(
            result.submissions[1].outcome, SubmissionOutcome.BUDGET_EXHAUSTED
        )
        self.assertIn("session disk full", result.journal_error)

    def test_classifier_failure_stops_without_losing_failed_attempt(self):
        def failed(run, *args, **kwargs):
            return replace(
                report_for(run), status=RunStatus.FAILED, errors=("failure",)
            )

        result = self.run_search(
            [self.submission(), self.submission()],
            failed,
            ExperimentBudget(max_attempts=4, max_retries_per_candidate=1),
            retry_classifier=lambda report: "not-a-boolean",
        )
        self.assertIsNotNone(result.submissions[0].report)
        self.assertIn("must return a bool", result.stop_reason)
        self.assertEqual(result.attempts_started, 1)

    def test_previously_consumed_attempt_budget_is_not_reset(self):
        budget = ExperimentBudget(max_attempts=3, attempts_started=2)
        result = self.run_search([self.submission(), self.submission()], budget=budget)
        self.assertEqual(result.attempts_started, 3)
        self.assertEqual(result.submissions[0].result.attempt.number, 3)
        self.assertEqual(
            result.submissions[1].outcome, SubmissionOutcome.BUDGET_EXHAUSTED
        )

    def test_retry_never_refreezes_changed_live_submission(self):
        candidate = self.submission()
        snapshots = []

        def evaluator(run, frozen, *args, **kwargs):
            snapshots.append(frozen)
            report = report_for(run)
            if len(snapshots) == 1:
                (candidate / "implementation.py").write_text(
                    "# later unrelated source version\n"
                )
                return replace(report, status=RunStatus.FAILED, errors=("transient",))
            self.assertIn(
                "submission code was imported",
                (frozen.directory / "implementation.py").read_text(),
            )
            return report

        result = self.run_search(
            [candidate],
            evaluator,
            ExperimentBudget(max_attempts=2, max_retries_per_candidate=1),
            retry_classifier=lambda r: True,
        )
        self.assertIs(snapshots[0], snapshots[1])
        self.assertEqual(result.retries_started, 1)

    def test_final_journal_cost_is_in_returned_total_and_flush_failure_is_retained(
        self,
    ):
        original = self.journal.record_session

        def flush(session_id, event, payload):
            if event == "completed":
                self.clock.advance(2)
            original(session_id, event, payload)

        with patch.object(self.journal, "record_session", side_effect=flush):
            result = self.run_search([self.submission()])
        self.assertEqual(result.elapsed_seconds, 2)
        saved = self.journal.read_history()[-1]["payload"]
        self.assertEqual(saved["elapsed_seconds"], 0)
        self.assertIn("Final journal flush", saved["accounting"]["note"])

        def fail(session_id, event, payload):
            if event == "completed":
                raise JournalError("final session flush")
            original(session_id, event, payload)

        with patch.object(self.journal, "record_session", side_effect=fail):
            failed = self.run_search([self.submission()])
        self.assertIn("final session flush", failed.journal_error)
        self.assertIsNotNone(failed.submissions[0].report)

    def test_configuration_and_clock_errors_are_setup_errors(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(max_elapsed_seconds=value), self.assertRaises(ValueError):
                self.run_search([], max_elapsed_seconds=value)
        with self.assertRaisesRegex(ValueError, "configuration_ids"):
            self.run_search([], configuration_ids={"generator": []})

    def test_real_pass_through_diagnostics_remain_unscored(self):
        from evaluator.pass_through import TrustedPassThroughEvaluator
        from target.adapter import TrustedExpertAdapter
        from tests.test_orchestrator_session import fake_expert

        evaluator = TrustedPassThroughEvaluator(
            lambda layer, x, ids: fake_expert(x, ids),
            TrustedExpertAdapter(
                {i: fake_expert for i in (0, 12, 23)}, hidden_size=8, num_experts=32
            ),
        )
        result = self.run_search([self.submission()], evaluator)
        self.assertTrue(all(r.passed for r in result.submissions[0].report.correctness))
        self.assertEqual(
            result.submissions[0].outcome, SubmissionOutcome.COMPLETED_NO_VERDICT
        )
        self.assertIsNone(result.score)
        self.assertIsNone(result.best_run_id)

    def test_journal_rejects_invalid_session_events_and_preserves_run_state(self):
        with self.assertRaises(ValueError):
            self.journal.record_session("s", "invented", {})
        with self.assertRaises(ValueError):
            self.journal.record_session("", "started", {})
        with self.assertRaises(TypeError):
            self.journal.record_session("s", "started", [])
        result = self.run_search([self.submission()])
        state = result.submissions[0].result.state
        saved = self.journal.read_state(state.experiment_id)
        self.assertEqual(saved["evaluator_report"], state.evaluator_report.to_dict())

    def test_retry_cannot_substitute_a_different_frozen_candidate_hash(self):
        candidate = self.submission()
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)

        def evaluator(run, *args, **kwargs):
            return replace(
                report_for(run), status=RunStatus.FAILED, errors=("transient",)
            )

        first = run_candidate(
            candidate,
            evaluator=evaluator,
            contract=CONTRACT,
            budget=budget,
            journal=self.journal,
            frozen_root=self.root / "direct",
            candidates_root=self.candidates,
        )
        forged = replace(
            first, frozen_candidate=replace(first.frozen_candidate, sha256="0" * 64)
        )
        retry = run_candidate(
            candidate,
            evaluator=evaluator,
            contract=CONTRACT,
            budget=budget,
            journal=self.journal,
            frozen_root=self.root / "direct",
            candidates_root=self.candidates,
            retry_of=forged,
            retryable=True,
        )
        self.assertIn("snapshot differs", retry.error)
        self.assertEqual(budget.attempts_started, 1)
        self.assertEqual(budget.retries_started, 0)


if __name__ == "__main__":
    unittest.main()
