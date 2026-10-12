"""Experiment-budget tests use synthetic evaluator results."""

import unittest

from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    Verdict,
    VerdictResult,
)
from orchestrator.experiment_budget import BudgetExhausted, ExperimentBudget


CONTRACT_HASH = "b" * 64


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
        first = budget.start_attempt("cand_012345abcdef")
        second = budget.start_attempt("cand_fedcba987654")
        self.assertEqual((first.number, first.retry_number), (1, 0))
        self.assertEqual((second.number, second.retry_number), (2, 0))
        self.assertEqual(budget.attempts_started, 2)
        self.assertEqual(budget.attempts_remaining, 0)
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "maximum total attempts"):
            budget.start_attempt("cand_aaaaaaaaaaaa")

    def test_existing_candidate_cannot_bypass_retry_accounting(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        with self.assertRaisesRegex(BudgetExhausted, "use start_retry"):
            budget.start_attempt(candidate_id)

    def test_retries_only_explicitly_retryable_failed_runs(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        report = fake_report(candidate_id, "run-1")

        self.assertFalse(budget.can_retry(report, retryable=False).allowed)
        retry = budget.start_retry(report, retryable=True)
        self.assertEqual((retry.number, retry.retry_number), (2, 1))
        self.assertEqual(budget.retries_started, 1)
        self.assertEqual(budget.retries_by_candidate[candidate_id], 1)

    def test_retry_limit_and_repeated_report_are_enforced(self):
        budget = ExperimentBudget(max_attempts=4, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        first_report = fake_report(candidate_id, "run-1")
        budget.start_retry(first_report, retryable=True)
        with self.assertRaisesRegex(BudgetExhausted, "already been retried"):
            budget.start_retry(first_report, retryable=True)

        second_report = fake_report(candidate_id, "run-2")
        with self.assertRaisesRegex(BudgetExhausted, "maximum retries"):
            budget.start_retry(second_report, retryable=True)

    def test_completed_verdict_is_never_retried(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=2)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        report = fake_report(
            candidate_id, "run-complete", RunStatus.COMPLETED, Verdict.REJECTED
        )
        decision = budget.can_retry(report, retryable=True)
        self.assertFalse(decision.allowed)
        self.assertIn("only failed", decision.reason)

    def test_interrupted_run_is_not_retried_automatically(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=2)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        report = fake_report(candidate_id, "run-interrupted", RunStatus.INTERRUPTED)
        decision = budget.can_retry(report, retryable=True)
        self.assertFalse(decision.allowed)
        self.assertIn("interrupted", decision.reason)

    def test_observing_interruption_stops_the_whole_budget(self):
        budget = ExperimentBudget(max_attempts=3, max_retries_per_candidate=1)
        candidate_id = "cand_012345abcdef"
        budget.start_attempt(candidate_id)
        budget.observe_result(
            fake_report(candidate_id, "run-interrupted", RunStatus.INTERRUPTED)
        )
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "Synthetic run failure"):
            budget.start_attempt("cand_fedcba987654")

    def test_explicit_stop_prevents_new_attempts(self):
        budget = ExperimentBudget(max_attempts=3)
        budget.start_attempt("cand_012345abcdef")
        budget.stop("user requested stop")
        self.assertTrue(budget.must_stop)
        with self.assertRaisesRegex(BudgetExhausted, "user requested stop"):
            budget.start_attempt("cand_fedcba987654")

    def test_limits_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ExperimentBudget(max_attempts=0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ExperimentBudget(max_attempts=2, max_retries_per_candidate=-1)


if __name__ == "__main__":
    unittest.main()
