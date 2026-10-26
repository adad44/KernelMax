"""Shared-budget multi-submission session tests with offline fake experts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluator.pass_through import TrustedPassThroughEvaluator
from evaluator.schemas import (
    EvaluationReport,
    RunStatus,
    Verdict,
    VerdictResult,
    load_contract,
)
from orchestrator.candidate_manager import create_candidate_id
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.session import SubmissionOutcome, run_session
from orchestrator.state_machine import RunIdentity
from target.adapter import TrustedExpertAdapter


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "kernelmaxxing.yaml")


def fake_expert(activations, selected_expert_ids):
    return activations[:, :, None, :] + selected_expert_ids[:, :, :, None].astype(
        activations.dtype
    )


def make_submission(root: Path, *, valid: bool = True) -> Path:
    candidate_id = create_candidate_id()
    directory = root / candidate_id
    directory.mkdir(parents=True)
    if valid:
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "name": "Session test candidate",
                    "implementation_type": "mlx",
                    "description": "Test fixture; candidate code must not execute",
                }
            ),
            encoding="utf-8",
        )
        for name in ("wrapper.py", "implementation.py"):
            (directory / name).write_text(
                'raise AssertionError("submission code was imported")\n',
                encoding="utf-8",
            )
    return directory


def report_for(run: RunIdentity) -> EvaluationReport:
    return EvaluationReport(
        run_id=run.run_id,
        candidate=run.candidate,
        contract_sha256=run.contract_sha256,
        environment=None,
        correctness=(),
        performance=(),
        status=RunStatus.COMPLETED,
        errors=(),
        verdict=None,
    )


class OrchestratorSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()
        self.frozen = self.root / "controller" / "frozen"
        self.journal = ExperimentJournal(self.root / "controller" / "journal")

    def run_batch(self, submissions, evaluator, budget):
        return run_session(
            submissions,
            evaluator=evaluator,
            contract=CONTRACT,
            budget=budget,
            journal=self.journal,
            frozen_root=self.frozen,
            candidates_root=self.candidates,
            session_id="session-test",
        )

    def test_multiple_submissions_share_budget_and_duplicate_is_skipped(self):
        first = make_submission(self.candidates)
        second = make_submission(self.candidates)
        third = make_submission(self.candidates)
        calls = []

        def evaluator(run, frozen, contract, *, on_diagnostic_passed=None):
            calls.append(run.candidate.candidate_id)
            return report_for(run)

        budget = ExperimentBudget(max_attempts=2)
        result = self.run_batch([first, first, second, third], evaluator, budget)

        self.assertEqual(
            [item.outcome for item in result.submissions],
            [
                SubmissionOutcome.COMPLETED_NO_VERDICT,
                SubmissionOutcome.DUPLICATE,
                SubmissionOutcome.COMPLETED_NO_VERDICT,
                SubmissionOutcome.BUDGET_EXHAUSTED,
            ],
        )
        self.assertEqual(calls, [first.name, second.name])
        self.assertEqual(budget.attempts_started, 2)
        self.assertEqual(result.attempts_started, 2)
        self.assertIn("maximum total attempts reached", result.summary())
        self.assertIn("no candidate approval", result.summary())

    def test_evaluator_failure_is_local_and_later_submission_runs(self):
        first = make_submission(self.candidates)
        second = make_submission(self.candidates)
        calls = []

        def evaluator(run, frozen, contract, *, on_diagnostic_passed=None):
            calls.append(run.candidate.candidate_id)
            if run.candidate.candidate_id == first.name:
                raise RuntimeError("temporary synthetic failure")
            return report_for(run)

        budget = ExperimentBudget(max_attempts=2)
        result = self.run_batch([first, second], evaluator, budget)

        self.assertEqual(
            [item.outcome for item in result.submissions],
            [SubmissionOutcome.FAILED, SubmissionOutcome.COMPLETED_NO_VERDICT],
        )
        self.assertIn("temporary synthetic failure", result.submissions[0].detail)
        self.assertEqual(calls, [first.name, second.name])
        self.assertEqual(budget.attempts_started, 2)

    def test_failed_evaluator_report_is_local_and_later_submission_runs(self):
        first = make_submission(self.candidates)
        second = make_submission(self.candidates)
        calls = []
        failed_reports = []

        def evaluator(run, frozen, contract, *, on_diagnostic_passed=None):
            calls.append(run.candidate.candidate_id)
            report = report_for(run)
            if run.candidate.candidate_id == first.name:
                report = replace(
                    report,
                    status=RunStatus.FAILED,
                    errors=("synthetic failed evaluator report",),
                )
                failed_reports.append(report)
            return report

        budget = ExperimentBudget(max_attempts=2)
        result = self.run_batch([first, second], evaluator, budget)

        self.assertEqual(
            [item.outcome for item in result.submissions],
            [SubmissionOutcome.FAILED, SubmissionOutcome.COMPLETED_NO_VERDICT],
        )
        self.assertIs(result.submissions[0].report, failed_reports[0])
        self.assertEqual(
            result.submissions[0].result.state.evaluator_report, failed_reports[0]
        )
        self.assertEqual(calls, [first.name, second.name])
        self.assertEqual(budget.attempts_started, 2)
        self.assertIsNone(budget.stop_reason)

    def test_invalid_submission_does_not_consume_budget_or_block_next(self):
        invalid = make_submission(self.candidates, valid=False)
        valid = make_submission(self.candidates)

        def evaluator(run, frozen, contract, *, on_diagnostic_passed=None):
            return report_for(run)

        budget = ExperimentBudget(max_attempts=1)
        result = self.run_batch([invalid, valid], evaluator, budget)

        self.assertEqual(
            [item.outcome for item in result.submissions],
            [SubmissionOutcome.INVALID, SubmissionOutcome.COMPLETED_NO_VERDICT],
        )
        self.assertEqual(budget.attempts_started, 1)

    def test_trusted_diagnostic_success_is_not_candidate_approval(self):
        candidate = make_submission(self.candidates)

        def reference(layer, activations, selected_expert_ids):
            return fake_expert(activations, selected_expert_ids)

        evaluator = TrustedPassThroughEvaluator(
            reference,
            TrustedExpertAdapter(
                {layer: fake_expert for layer in (0, 12, 23)},
                hidden_size=8,
                num_experts=32,
            ),
        )
        result = self.run_batch(
            [candidate], evaluator, ExperimentBudget(max_attempts=1)
        )
        submission = result.submissions[0]

        self.assertEqual(submission.outcome, SubmissionOutcome.COMPLETED_NO_VERDICT)
        self.assertFalse(submission.candidate_approved)
        self.assertIsNotNone(submission.report)
        self.assertIsNone(submission.report.verdict)
        self.assertEqual(submission.report.performance, ())
        self.assertIn("diagnostics recorded; no candidate approval", result.summary())

    def test_only_a_valid_retained_accepted_verdict_counts_as_approval(self):
        candidate = make_submission(self.candidates)

        def accepted_evaluator(run, frozen, contract, *, on_diagnostic_passed=None):
            report = report_for(run)
            verdict = VerdictResult(
                candidate=run.candidate,
                contract_sha256=run.contract_sha256,
                verdict=Verdict.ACCEPTED,
                reason="synthetic evaluator approval",
            )
            return replace(report, verdict=verdict)

        approved = self.run_batch(
            [candidate], accepted_evaluator, ExperimentBudget(max_attempts=1)
        ).submissions[0]
        self.assertEqual(approved.outcome, SubmissionOutcome.APPROVED)
        self.assertTrue(approved.candidate_approved)

        candidate2 = make_submission(self.candidates)

        def mismatched_accepted_evaluator(
            run, frozen, contract, *, on_diagnostic_passed=None
        ):
            report = accepted_evaluator(run, frozen, contract)
            return replace(report, run_id="foreign-run")

        rejected = self.run_batch(
            [candidate2],
            mismatched_accepted_evaluator,
            ExperimentBudget(max_attempts=1),
        ).submissions[0]
        self.assertEqual(rejected.outcome, SubmissionOutcome.FAILED)
        self.assertFalse(rejected.candidate_approved)
        self.assertIsNotNone(rejected.report)
        self.assertIsNone(rejected.result.state.evaluator_report)

    def test_pre_stopped_budget_marks_every_unique_submission_exhausted(self):
        first = make_submission(self.candidates)
        second = make_submission(self.candidates)
        budget = ExperimentBudget(max_attempts=2)
        budget.stop("operator requested stop")
        calls = []

        result = self.run_batch(
            [first, second], lambda *args, **kwargs: calls.append(args), budget
        )

        self.assertEqual(
            [item.outcome for item in result.submissions],
            [SubmissionOutcome.BUDGET_EXHAUSTED, SubmissionOutcome.BUDGET_EXHAUSTED],
        )
        self.assertEqual(calls, [])
        self.assertIn("operator requested stop", result.summary())


if __name__ == "__main__":
    unittest.main()
