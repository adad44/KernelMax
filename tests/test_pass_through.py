"""Offline evaluator handoff tests; frozen submission code is never executed."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluator.pass_through import TrustedPassThroughEvaluator
from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    load_contract,
    validate_report_against_contract,
)
from orchestrator.candidate_manager import create_candidate_id, freeze_candidate
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.state_machine import (
    EventType,
    ExperimentEvent,
    ExperimentPhase,
    RunIdentity,
    new_experiment,
    transition,
)
from target.adapter import TrustedExpertAdapter


CONTRACT = load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


def expert(x, ids):
    return x[:, :, None, :] + ids[:, :, :, None].astype(x.dtype)


class PassThroughTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.candidates = root / "candidates"
        self.candidate_id = create_candidate_id()
        source = self.candidates / self.candidate_id
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": self.candidate_id,
                    "name": "Non-executable diagnostic submission",
                    "implementation_type": "mlx",
                    "description": "Provenance fixture, never executed",
                }
            )
        )
        for filename in ("wrapper.py", "implementation.py"):
            (source / filename).write_text(
                'raise AssertionError("submission code was executed")\n'
            )
        self.frozen = freeze_candidate(
            source, candidates_root=self.candidates, frozen_root=root / "frozen"
        )
        self.run = RunIdentity(
            "diagnostic-001",
            CandidateIdentity(self.candidate_id, "mlx", self.frozen.sha256),
            CONTRACT.sha256,
        )
        self.reference_calls = []

        def reference(layer, x, ids):
            self.reference_calls.append(layer)
            return expert(x, ids)

        self.reference = reference

    def evaluator(self, bindings=None, reference=None):
        return TrustedPassThroughEvaluator(
            self.reference if reference is None else reference,
            TrustedExpertAdapter(
                {layer: expert for layer in (0, 12, 23)}
                if bindings is None
                else bindings,
                hidden_size=8,
                num_experts=32,
            ),
        )

    def tamper(self):
        wrapper = self.frozen.directory / "wrapper.py"
        wrapper.chmod(0o644)
        wrapper.write_text("# changed\n")

    def test_handoff_preserves_identity_and_all_21_checks_without_execution(self):
        signals = []
        report = self.evaluator()(
            self.run, self.frozen, CONTRACT, on_diagnostic_passed=signals.append
        )
        self.assertEqual(report.status, RunStatus.COMPLETED)
        self.assertEqual(report.run_id, self.run.run_id)
        self.assertEqual(report.candidate, self.run.candidate)
        self.assertEqual(report.contract_sha256, CONTRACT.sha256)
        self.assertIsNone(report.environment)
        self.assertIsNone(report.verdict)
        self.assertEqual(report.performance, ())
        self.assertEqual(report.errors, ())
        self.assertEqual(len(self.reference_calls), 21)
        self.assertEqual(len(report.correctness), 7)
        self.assertEqual(len(signals), 1)
        self.assertIs(signals[0], report)
        for result in report.correctness:
            detail = json.loads(result.detail)
            self.assertTrue(result.passed)
            self.assertEqual(len(detail["checks"]), 3)
            self.assertEqual(detail["evaluation_scope"], "trusted_pass_through")
            self.assertEqual(detail["input_mode"], "numpy_fixture")
            self.assertFalse(detail["candidate_code_executed"])
            self.assertFalse(detail["candidate_correctness_established"])
            self.assertFalse(detail["performance_measured"])
        self.assertEqual(
            EvaluationReport.from_mapping(json.loads(json.dumps(report.to_dict()))),
            report,
        )
        self.assertTrue(validate_report_against_contract(report, CONTRACT))

    def test_bad_run_identity_rejected_before_any_reference_execution(self):
        wrong_candidates = (
            replace(self.run.candidate, candidate_id="cand_000000000000"),
            replace(self.run.candidate, sha256="f" * 64),
            replace(self.run.candidate, implementation_type="metal"),
        )
        bad_runs = [replace(self.run, candidate=item) for item in wrong_candidates]
        bad_runs.append(replace(self.run, contract_sha256="f" * 64))
        for run in bad_runs:
            with self.subTest(run=run), self.assertRaises(ValueError):
                self.evaluator()(run, self.frozen, CONTRACT)
        self.assertEqual(self.reference_calls, [])

    def test_invalid_arguments_and_bindings_are_rejected(self):
        with self.assertRaises(TypeError):
            TrustedPassThroughEvaluator(None, self.evaluator().adapter)
        with self.assertRaises(TypeError):
            TrustedPassThroughEvaluator(self.reference, object())
        with self.assertRaises(TypeError):
            TrustedPassThroughEvaluator(
                self.reference, self.evaluator().adapter, object()
            )
        for args in (
            (object(), self.frozen, CONTRACT),
            (self.run, object(), CONTRACT),
            (self.run, self.frozen, object()),
        ):
            with self.subTest(args=args), self.assertRaises(TypeError):
                self.evaluator()(*args)
        with self.assertRaises(TypeError):
            self.evaluator()(
                self.run, self.frozen, CONTRACT, on_diagnostic_passed="not callable"
            )
        self.assertEqual(self.reference_calls, [])

    def test_snapshot_tampering_blocks_diagnostics(self):
        self.tamper()
        signals = []
        report = self.evaluator()(
            self.run, self.frozen, CONTRACT, on_diagnostic_passed=signals.append
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertIn("snapshot before", report.errors[0])
        self.assertEqual(report.correctness, ())
        self.assertEqual(self.reference_calls, [])
        self.assertEqual(signals, [])

    def test_tampering_during_suite_retains_checks_and_suppresses_signal(self):
        def reference(layer, x, ids):
            self.tamper()
            return expert(x, ids)

        signals = []
        report = self.evaluator(reference=reference)(
            self.run, self.frozen, CONTRACT, on_diagnostic_passed=signals.append
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertIn("snapshot after", report.errors[-1])
        self.assertEqual(len(report.correctness), 7)
        self.assertEqual(signals, [])

    def test_numerical_failure_retains_complete_diagnostics_without_signal(self):
        def wrong(x, ids):
            return expert(x, ids) + np.float32(100)

        signals = []
        report = self.evaluator(bindings={layer: wrong for layer in (0, 12, 23)})(
            self.run, self.frozen, CONTRACT, on_diagnostic_passed=signals.append
        )
        self.assertEqual(report.status, RunStatus.COMPLETED)
        self.assertEqual(report.errors, ())
        self.assertTrue(all(not result.passed for result in report.correctness))
        self.assertIsNone(report.verdict)
        self.assertEqual(signals, [])

    def test_missing_layer_retains_all_checks_and_errors(self):
        signals = []
        report = self.evaluator(bindings={0: expert, 23: expert})(
            self.run, self.frozen, CONTRACT, on_diagnostic_passed=signals.append
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertEqual(len(report.errors), 7)
        self.assertEqual(len(self.reference_calls), 21)
        self.assertEqual(len(report.correctness), 7)
        for result in report.correctness:
            self.assertEqual(len(json.loads(result.detail)["checks"]), 3)
        self.assertEqual(signals, [])

    def test_backend_errors_remain_execution_failures(self):
        def broken(x, ids):
            raise RuntimeError("backend failed")

        report = self.evaluator(bindings={0: expert, 12: broken, 23: expert})(
            self.run, self.frozen, CONTRACT
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertEqual(len(report.errors), 7)
        self.assertTrue(all("backend failed" in error for error in report.errors))
        self.assertIsNone(report.verdict)

    def test_callback_failure_or_interruption_retains_checks(self):
        for error, expected in (
            (RuntimeError("recording failed"), RunStatus.FAILED),
            (KeyboardInterrupt("cancelled"), RunStatus.INTERRUPTED),
        ):

            def callback(report):
                raise error

            with self.subTest(error=error):
                report = self.evaluator()(
                    self.run, self.frozen, CONTRACT, on_diagnostic_passed=callback
                )
                self.assertEqual(report.status, expected)
                self.assertEqual(len(report.correctness), 7)
                self.assertIn(str(error), report.errors[-1])

    def test_callback_tampering_is_detected(self):
        report = self.evaluator()(
            self.run,
            self.frozen,
            CONTRACT,
            on_diagnostic_passed=lambda report: self.tamper(),
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertIn("snapshot after diagnostic callback", report.errors[-1])
        self.assertEqual(len(report.correctness), 7)

    def test_suite_interruption_propagates_to_controller(self):
        def interrupted(layer, x, ids):
            raise KeyboardInterrupt("cancelled during suite")

        with self.assertRaises(KeyboardInterrupt):
            self.evaluator(reference=interrupted)(self.run, self.frozen, CONTRACT)

    def test_report_connects_to_state_and_budget_without_benchmark_transition(self):
        budget = ExperimentBudget(max_attempts=1)
        budget.start_attempt(self.run)
        state = new_experiment(CONTRACT.experiment_id, self.candidate_id)
        for event in (
            ExperimentEvent(EventType.SUBMITTED),
            ExperimentEvent(EventType.CANDIDATE_VALIDATED),
            ExperimentEvent(EventType.CANDIDATE_FROZEN, run=self.run),
        ):
            state = transition(state, event)
        report = self.evaluator()(self.run, self.frozen, CONTRACT)
        budget.observe_result(report)
        state = transition(
            state, ExperimentEvent(EventType.EVALUATOR_RESULT, report=report)
        )
        self.assertEqual(state.phase, ExperimentPhase.COMPLETED)
        self.assertIs(state.evaluator_report, report)
        self.assertIsNone(state.verdict)
        self.assertNotIn(
            EventType.CORRECTNESS_PASSED, [event.type for event in state.history]
        )
        self.assertEqual(budget.attempts_started, 1)
        self.assertTrue(budget.must_stop)


if __name__ == "__main__":
    unittest.main()
