"""Experiment-budget tests use synthetic evaluator results."""

import unittest
from dataclasses import replace

from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    Verdict,
    VerdictResult,
)
from orchestrator.experiment_budget import BudgetExhausted, ExperimentBudget
from orchestrator.state_machine import RunIdentity

CONTRACT_HASH = "b" * 64


def fake_run(candidate_id, run_id="run-1"):
    return RunIdentity(
        run_id, CandidateIdentity(candidate_id, "mlx", "a" * 64), CONTRACT_HASH
    )


def fake_report(candidate_id, run_id, status=RunStatus.FAILED, verdict=None):
    candidate = CandidateIdentity(candidate_id, "mlx", "a" * 64)
    evaluator_verdict = (
        VerdictResult(candidate, CONTRACT_HASH, verdict, "Synthetic evaluator result")
        if verdict is not None
        else None
    )
    errors = ("Synthetic run failure",) if status is not RunStatus.COMPLETED else ()
    return EvaluationReport(
        run_id=run_id,
        candidate=candidate,
        contract_sha256=CONTRACT_HASH,
        environment=None,
        correctness=(),
        performance=(),
        status=status,
        errors=errors,
        verdict=evaluator_verdict,
    )


class ExperimentBudgetTests(unittest.TestCase):
    def test_counts_attempts_and_stops_at_total_limit(self):
        budget = ExperimentBudget(max_attempts=2, max_retries_per_candidate=1)
        first = budget.start_attempt(fake_run("cand_012345abcdef"))
        second = budget.start_attempt(fake_run("cand_fedcba987654", "run-2"))
        self.assertEqual((first.number, first.retry_number), (1, 0))
        self.assertEqual((second.number, second.retry_number), (2, 0))
        self.assertEqual(budget.attempts_started, 2)
        self.assertEqual(budget.attempts_remaining, 0)
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "maximum total attempts"):
            budget.start_attempt(fake_run("cand_aaaaaaaaaaaa", "run-3"))

    def test_existing_candidate_cannot_bypass_retry_accounting(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id))
        with self.assertRaisesRegex(BudgetExhausted, "use start_retry"):
            budget.start_attempt(fake_run(candidate_id, "another-run"))

    def test_retries_only_explicitly_retryable_failed_runs(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id))
        report = fake_report(candidate_id, "run-1")
        budget.observe_result(report)

        self.assertFalse(budget.can_retry(report, retryable=False).allowed)
        retry = budget.start_retry(report, run_id="run-2", retryable=True)
        self.assertEqual((retry.number, retry.retry_number), (2, 1))
        self.assertEqual(budget.retries_started, 1)
        self.assertEqual(budget.retries_by_candidate[candidate_id], 1)

    def test_retry_limit_and_repeated_report_are_enforced(self):
        budget = ExperimentBudget(max_attempts=4, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id))
        first_report = fake_report(candidate_id, "run-1")
        budget.observe_result(first_report)
        budget.start_retry(first_report, run_id="run-2", retryable=True)
        with self.assertRaisesRegex(BudgetExhausted, "already been retried"):
            budget.start_retry(first_report, run_id="run-3", retryable=True)

        second_report = fake_report(candidate_id, "run-2")
        budget.observe_result(second_report)
        with self.assertRaisesRegex(BudgetExhausted, "maximum retries"):
            budget.start_retry(second_report, run_id="run-3", retryable=True)

    def test_completed_verdict_is_never_retried(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=2)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id, "run-complete"))
        report = fake_report(
            candidate_id, "run-complete", RunStatus.COMPLETED, Verdict.REJECTED
        )
        budget.observe_result(report)
        decision = budget.can_retry(report, retryable=True)
        self.assertFalse(decision.allowed)
        self.assertIn("final evaluator result", decision.reason)

    def test_interrupted_run_is_not_retried_automatically(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=2)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id, "run-interrupted"))
        report = fake_report(candidate_id, "run-interrupted", RunStatus.INTERRUPTED)
        budget.observe_result(report)
        decision = budget.can_retry(report, retryable=True)
        self.assertFalse(decision.allowed)
        self.assertIn("budget stopped", decision.reason)

    def test_observing_interruption_stops_the_whole_budget(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(fake_run(candidate_id, "run-interrupted"))
        budget.observe_result(
            fake_report(candidate_id, "run-interrupted", RunStatus.INTERRUPTED)
        )
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "Synthetic run failure"):
            budget.start_attempt(fake_run("cand_fedcba987654", "run-2"))

    def test_explicit_stop_prevents_new_attempts(self):
        budget = ExperimentBudget(max_attempts=3)
        budget.start_attempt(fake_run("cand_012345abcdef"))
        budget.stop("user requested stop")
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "user requested stop"):
            budget.start_attempt(fake_run("cand_fedcba987654", "run-2"))

    def test_completed_candidate_cannot_retry_an_earlier_failure(self):
        for verdict in (Verdict.ACCEPTED, Verdict.REJECTED, Verdict.INCONCLUSIVE, None):
            with self.subTest(verdict=verdict):
                candidate_id = "cand_012345abcdef"
                budget = ExperimentBudget(max_attempts=4, max_retries_per_candidate=2)
                budget.start_attempt(fake_run(candidate_id))
                failed = fake_report(candidate_id, "run-1")
                budget.observe_result(failed)
                budget.start_retry(failed, run_id="run-2", retryable=True)
                completed = fake_report(
                    candidate_id, "run-2", RunStatus.COMPLETED, verdict
                )
                budget.observe_result(completed)
                with self.assertRaisesRegex(BudgetExhausted, "final evaluator result"):
                    budget.start_retry(failed, run_id="run-3", retryable=True)
                with self.assertRaisesRegex(BudgetExhausted, "final evaluator result"):
                    budget.start_retry(completed, run_id="run-3", retryable=True)
                budget.observe_result(failed)  # Duplicate old notification is harmless.
                self.assertEqual(budget.attempts_started, 2)
                self.assertFalse(budget.must_stop)
                budget.start_attempt(fake_run("cand_fedcba987654", "run-3"))

    def test_unreserved_and_mismatched_reports_do_not_change_budget(self):
        candidate_id = "cand_012345abcdef"
        budget = ExperimentBudget(max_attempts=4, max_retries_per_candidate=2)
        budget.start_attempt(fake_run(candidate_id))
        report = fake_report(candidate_id, "run-1")
        wrong_reports = (
            replace(report, run_id="older-run"),
            replace(
                report,
                candidate=replace(report.candidate, candidate_id="cand_fedcba987654"),
            ),
            replace(report, candidate=replace(report.candidate, sha256="c" * 64)),
            replace(
                report, candidate=replace(report.candidate, implementation_type="metal")
            ),
            replace(report, contract_sha256="d" * 64),
        )
        for wrong_report in wrong_reports:
            with self.subTest(report=wrong_report):
                with self.assertRaises(ValueError):
                    budget.observe_result(wrong_report)
                self.assertFalse(budget.can_retry(wrong_report, retryable=True).allowed)
                self.assertEqual(budget.attempts_started, 1)
                self.assertEqual(budget.retries_started, 0)
                self.assertIsNone(budget.stop_reason)
        budget.observe_result(report)
        self.assertTrue(budget.can_retry(report, retryable=True).allowed)

    def test_retry_requires_the_observed_result(self):
        candidate_id = "cand_012345abcdef"
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        budget.start_attempt(fake_run(candidate_id))
        report = fake_report(candidate_id, "run-1")
        with self.assertRaisesRegex(BudgetExhausted, "not been observed"):
            budget.start_retry(report, run_id="run-2", retryable=True)
        budget.observe_result(report)
        changed = replace(report, errors=("Different failure details",))
        with self.assertRaisesRegex(ValueError, "conflicting evaluator results"):
            budget.observe_result(changed)
        with self.assertRaisesRegex(BudgetExhausted, "differs from the observed"):
            budget.start_retry(changed, run_id="run-2", retryable=True)
        self.assertEqual(budget.attempts_started, 1)

    def test_run_ids_are_unique_and_retries_preserve_provenance(self):
        candidate_id = "cand_012345abcdef"
        run = fake_run(candidate_id)
        budget = ExperimentBudget(max_attempts=4, max_retries_per_candidate=2)
        budget.start_attempt(run)
        with self.assertRaisesRegex(BudgetExhausted, "run_id has already"):
            budget.start_attempt(fake_run("cand_fedcba987654"))
        report = fake_report(candidate_id, "run-1")
        budget.observe_result(report)
        with self.assertRaisesRegex(BudgetExhausted, "run_id has already"):
            budget.start_retry(report, run_id="run-1", retryable=True)
        with self.assertRaisesRegex(ValueError, "run_id"):
            budget.start_retry(report, run_id=" ", retryable=True)
        self.assertEqual(budget.attempts_started, 1)
        self.assertEqual(budget.retries_started, 0)
        retry = budget.start_retry(report, run_id="run-2", retryable=True)
        self.assertEqual(retry.run_id, "run-2")
        self.assertEqual(retry.run.candidate, run.candidate)
        self.assertEqual(retry.run.contract_sha256, run.contract_sha256)

    def test_duplicate_results_are_idempotent_and_conflicts_rejected(self):
        candidate_id = "cand_012345abcdef"
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        budget.start_attempt(fake_run(candidate_id))
        report = fake_report(candidate_id, "run-1")
        budget.observe_result(report)
        budget.observe_result(report)
        self.assertEqual(budget.attempts_started, 1)
        self.assertEqual(budget.retries_started, 0)
        with self.assertRaisesRegex(ValueError, "conflicting evaluator results"):
            budget.observe_result(replace(report, status=RunStatus.INTERRUPTED))
        self.assertFalse(budget.must_stop)
        budget.start_retry(report, run_id="run-2", retryable=True)
        budget.observe_result(report)
        self.assertEqual(budget.attempts_started, 2)
        self.assertFalse(budget.must_stop)

    def test_foreign_interruption_cannot_stop_budget(self):
        candidate_id = "cand_012345abcdef"
        budget = ExperimentBudget(max_attempts=3)
        budget.start_attempt(fake_run(candidate_id))
        report = fake_report(candidate_id, "foreign-run", RunStatus.INTERRUPTED)
        with self.assertRaisesRegex(ValueError, "not started"):
            budget.observe_result(report)
        self.assertFalse(budget.must_stop)

    def test_limits_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ExperimentBudget(max_attempts=0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ExperimentBudget(max_attempts=2, max_retries_per_candidate=-1)


if __name__ == "__main__":
    unittest.main()
