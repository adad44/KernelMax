"""Offline runner integration tests with fake evaluators and fake experts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluator.pass_through import TrustedPassThroughEvaluator
from evaluator.schemas import (
    EvaluationReport,
    RunStatus,
    load_contract,
)
from orchestrator.candidate_manager import create_candidate_id
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.run import run_candidate
from orchestrator.state_machine import EventType, ExperimentPhase, RunIdentity
from target.adapter import TrustedExpertAdapter


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "kernelmaxxing.yaml")


def fake_expert(x, selected_expert_ids):
    return x[:, :, None, :] + selected_expert_ids[:, :, :, None].astype(x.dtype)


def make_report(run: RunIdentity, status: RunStatus = RunStatus.COMPLETED):
    return EvaluationReport(
        run_id=run.run_id,
        candidate=run.candidate,
        contract_sha256=run.contract_sha256,
        environment=None,
        correctness=(),
        performance=(),
        status=status,
        errors=("synthetic evaluator failure",) if status is RunStatus.FAILED else (),
        verdict=None,
    )


class CandidateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidates_root = self.root / "candidates"
        self.candidates_root.mkdir()
        self.frozen_root = self.root / "controller" / "frozen"
        self.journal = ExperimentJournal(self.root / "journal")
        self.candidate_id = create_candidate_id()
        self.candidate_dir = self.candidates_root / self.candidate_id
        self.candidate_dir.mkdir()
        (self.candidate_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": self.candidate_id,
                    "name": "Offline candidate fixture",
                    "implementation_type": "mlx",
                    "description": "Provenance only; evaluator must not execute it",
                }
            ),
            encoding="utf-8",
        )
        # If the runner or evaluator imports this file, the integration test fails.
        for filename in ("wrapper.py", "implementation.py"):
            (self.candidate_dir / filename).write_text(
                'raise AssertionError("candidate code must not be imported")\n',
                encoding="utf-8",
            )
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.frozen_root.exists():
            for path in self.frozen_root.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            for path in self.frozen_root.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)
            self.temp.cleanup()
        else:
            self.temp.cleanup()

    def run_one(self, evaluator, budget=None, *, experiment_id="runner-test"):
        return run_candidate(
            self.candidate_dir,
            evaluator=evaluator,
            contract=CONTRACT,
            budget=budget or ExperimentBudget(max_attempts=3),
            journal=self.journal,
            frozen_root=self.frozen_root,
            candidates_root=self.candidates_root,
            experiment_id=experiment_id,
        )

    def test_valid_fake_evaluator_receives_inputs_and_report_is_retained(self):
        seen = {}

        def evaluator(run_identity, frozen_candidate, contract, *, on_diagnostic_passed=None):
            seen.update(
                run=run_identity,
                frozen=frozen_candidate,
                contract=contract,
                diagnostic_callback=on_diagnostic_passed,
            )
            return make_report(run_identity)

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.COMPLETED)
        self.assertIs(result.state.evaluator_report, result.report)
        self.assertIs(result.state.run, seen["run"])
        self.assertIs(result.frozen_candidate, seen["frozen"])
        self.assertIs(seen["contract"], CONTRACT)
        self.assertTrue(callable(seen["diagnostic_callback"]))
        self.assertEqual(result.attempt.run, result.state.run)
        self.assertEqual(result.report.performance, ())
        self.assertEqual(result.report.verdict, None)
        self.assertEqual(
            self.journal.read_history()[-1]["event"]["type"], "evaluator_result"
        )

    def test_exhausted_budget_is_journaled_without_freezing_or_evaluating(self):
        budget = ExperimentBudget(max_attempts=1, attempts_started=1)
        calls = []
        result = self.run_one(lambda *args, **kwargs: calls.append(args), budget)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIn("maximum total attempts", result.error)
        self.assertEqual(calls, [])
        self.assertEqual(budget.attempts_started, 1)
        self.assertFalse(self.frozen_root.exists())
        history = self.journal.read_history()
        self.assertEqual(
            [item["event"]["type"] for item in history],
            ["submitted", "candidate_validated", "controller_failed"],
        )

    def test_mismatched_report_is_retained_but_not_accepted_as_evaluator_result(self):
        reports = []

        def evaluator(run_identity, frozen_candidate, contract, *, on_diagnostic_passed=None):
            report = replace(make_report(run_identity), run_id="another-run")
            reports.append(report)
            return report

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIs(result.report, reports[0])
        self.assertIsNone(result.state.evaluator_report)
        self.assertIn("different run", result.error)
        failure = self.journal.read_history()[-1]
        self.assertEqual(failure["event"]["type"], "controller_failed")
        self.assertEqual(failure["observed_report"], reports[0].to_dict())

    def test_candidate_identity_mismatch_is_rejected_and_retained(self):
        reports = []

        def evaluator(
            run_identity,
            frozen_candidate,
            contract,
            *,
            on_diagnostic_passed=None,
        ):
            report = replace(
                make_report(run_identity),
                candidate=replace(run_identity.candidate, sha256="f" * 64),
            )
            reports.append(report)
            return report

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIs(result.report, reports[0])
        self.assertIsNone(result.state.evaluator_report)
        self.assertIn("different candidate identity", result.error)
        self.assertEqual(
            self.journal.read_history()[-1]["observed_report"],
            reports[0].to_dict(),
        )

    def test_contract_hash_mismatch_is_rejected_and_retained(self):
        reports = []

        def evaluator(
            run_identity,
            frozen_candidate,
            contract,
            *,
            on_diagnostic_passed=None,
        ):
            report = replace(make_report(run_identity), contract_sha256="f" * 64)
            reports.append(report)
            return report

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIs(result.report, reports[0])
        self.assertIsNone(result.state.evaluator_report)
        self.assertIn("different contract hash", result.error)
        self.assertEqual(
            self.journal.read_history()[-1]["observed_report"],
            reports[0].to_dict(),
        )

    def test_snapshot_change_after_evaluation_is_journaled_and_stops_budget(self):
        def evaluator(
            run_identity,
            frozen_candidate,
            contract,
            *,
            on_diagnostic_passed=None,
        ):
            source = frozen_candidate.directory / "implementation.py"
            source.chmod(0o600)
            source.write_text("# modified after freezing\n", encoding="utf-8")
            source.chmod(0o400)
            return make_report(run_identity)

        budget = ExperimentBudget(max_attempts=3)
        result = self.run_one(evaluator, budget)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIsNotNone(result.report)
        self.assertIsNone(result.state.evaluator_report)
        self.assertTrue(budget.must_stop)
        self.assertIn("contents changed", result.error)
        entry = self.journal.read_history()[-1]
        self.assertEqual(entry["event"]["type"], "controller_failed")
        self.assertEqual(entry["observed_report"], result.report.to_dict())

    def test_evaluator_exception_is_a_terminal_journaled_failure(self):
        def evaluator(*args, **kwargs):
            raise RuntimeError("synthetic evaluator crash")

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIn("synthetic evaluator crash", result.error)
        self.assertIsNone(result.report)
        self.assertEqual(
            self.journal.read_history()[-1]["event"]["type"], "controller_failed"
        )

    def test_failed_evaluator_report_is_retained_as_its_terminal_result(self):
        results = []

        def evaluator(
            run_identity,
            frozen_candidate,
            contract,
            *,
            on_diagnostic_passed=None,
        ):
            report = make_report(run_identity, RunStatus.FAILED)
            results.append(report)
            return report

        result = self.run_one(evaluator)

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIs(result.report, results[0])
        self.assertIs(result.state.evaluator_report, results[0])
        self.assertEqual(
            self.journal.read_history()[-1]["event"]["report"],
            results[0].to_dict(),
        )

    def test_invalid_submission_is_rejected_and_retained_in_journal(self):
        (self.candidate_dir / "implementation.py").unlink()
        calls = []
        result = self.run_one(lambda *args, **kwargs: calls.append(args))

        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIn("implementation.py", result.error)
        self.assertEqual(calls, [])
        self.assertEqual(
            self.journal.read_history()[-1]["event"]["type"], "candidate_rejected"
        )

    def test_trusted_pass_through_diagnostic_signal_is_not_candidate_approval(self):
        reference_calls = []

        def reference(layer_index, x, selected_expert_ids):
            reference_calls.append(layer_index)
            return fake_expert(x, selected_expert_ids)

        evaluator = TrustedPassThroughEvaluator(
            reference,
            TrustedExpertAdapter(
                {layer: fake_expert for layer in (0, 12, 23)},
                hidden_size=8,
                num_experts=32,
            ),
        )
        result = self.run_one(evaluator)

        history = self.journal.read_history()
        signals = [item for item in history if item["kind"] == "evaluator_signal"]
        self.assertEqual(result.state.phase, ExperimentPhase.COMPLETED)
        self.assertEqual(len(reference_calls), 21)
        self.assertEqual(len(result.report.correctness), 7)
        self.assertEqual(result.report.performance, ())
        self.assertIsNone(result.report.verdict)
        self.assertTrue(all(item.passed for item in result.report.correctness))
        self.assertNotIn(
            EventType.CORRECTNESS_PASSED,
            [event.type for event in result.state.history],
        )
        self.assertEqual(result.state.phase, ExperimentPhase.COMPLETED)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal"], "on_diagnostic_passed")
        self.assertEqual(signals[0]["phase"], "correctness")
        self.assertEqual(signals[0]["report"], result.report.to_dict())
        for case in result.report.correctness:
            detail = json.loads(case.detail)
            self.assertFalse(detail["candidate_code_executed"])
            self.assertFalse(detail["candidate_correctness_established"])
            self.assertFalse(detail["performance_measured"])


if __name__ == "__main__":
    unittest.main()
