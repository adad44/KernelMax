"""State-machine tests use synthetic controller events and evaluator reports."""

import unittest
from dataclasses import replace

from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    Verdict,
    VerdictResult,
)
from orchestrator.state_machine import (
    EventType,
    ExperimentEvent,
    ExperimentPhase,
    RunIdentity,
    new_experiment,
    transition,
)

CANDIDATE_ID = "cand_012345abcdef"
CONTRACT_HASH = "b" * 64
RUN = RunIdentity(
    "run-001", CandidateIdentity(CANDIDATE_ID, "mlx", "a" * 64), CONTRACT_HASH
)


def fake_report(status=RunStatus.COMPLETED, verdict=None, *, candidate_id=CANDIDATE_ID):
    candidate = CandidateIdentity(candidate_id, "mlx", "a" * 64)
    evaluator_verdict = (
        VerdictResult(candidate, CONTRACT_HASH, verdict, "Synthetic evaluator result")
        if verdict is not None
        else None
    )
    errors = ("Synthetic run failure",) if status is not RunStatus.COMPLETED else ()
    return EvaluationReport(
        run_id="run-001",
        candidate=candidate,
        contract_sha256=CONTRACT_HASH,
        environment=None,
        correctness=(),
        performance=(),
        status=status,
        errors=errors,
        verdict=evaluator_verdict,
    )


class ExperimentStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.initial = new_experiment("exp-001", CANDIDATE_ID)

    def test_happy_path_records_events_and_preserves_evaluator_verdict(self):
        state = transition(self.initial, ExperimentEvent(EventType.SUBMITTED))
        self.assertEqual(state.phase, ExperimentPhase.VALIDATING)
        state = transition(state, ExperimentEvent(EventType.CANDIDATE_VALIDATED))
        self.assertEqual(state.phase, ExperimentPhase.FREEZING)
        state = transition(state, ExperimentEvent(EventType.CANDIDATE_FROZEN, run=RUN))
        self.assertEqual(state.phase, ExperimentPhase.CORRECTNESS)
        state = transition(state, ExperimentEvent(EventType.CORRECTNESS_PASSED))
        self.assertEqual(state.phase, ExperimentPhase.BENCHMARK)

        report = fake_report(verdict=Verdict.REJECTED)
        completed = transition(
            state, ExperimentEvent(EventType.EVALUATOR_RESULT, report=report)
        )
        self.assertEqual(completed.phase, ExperimentPhase.COMPLETED)
        self.assertEqual(completed.verdict, Verdict.REJECTED)
        self.assertIs(completed.evaluator_report, report)
        self.assertEqual(completed.run, RUN)
        self.assertEqual(len(completed.history), 5)
        self.assertEqual(
            state.phase, ExperimentPhase.BENCHMARK
        )  # Prior state is immutable.

    def test_candidate_validation_rejection_is_terminal(self):
        validating = transition(self.initial, ExperimentEvent(EventType.SUBMITTED))
        rejected = transition(
            validating,
            ExperimentEvent(EventType.CANDIDATE_REJECTED, detail="invalid manifest"),
        )
        self.assertEqual(rejected.phase, ExperimentPhase.FAILED)
        self.assertEqual(rejected.failure_reason, "invalid manifest")
        self.assertTrue(rejected.is_terminal)

    def test_invalid_event_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not valid"):
            transition(
                self.initial, ExperimentEvent(EventType.CANDIDATE_FROZEN, run=RUN)
            )

    def test_evaluator_report_must_match_candidate(self):
        state = self.initial
        for event in (
            EventType.SUBMITTED,
            EventType.CANDIDATE_VALIDATED,
            EventType.CANDIDATE_FROZEN,
        ):
            state = transition(
                state,
                ExperimentEvent(
                    event, run=RUN if event is EventType.CANDIDATE_FROZEN else None
                ),
            )
        with self.assertRaisesRegex(ValueError, "different candidate"):
            transition(
                state,
                ExperimentEvent(
                    EventType.EVALUATOR_RESULT,
                    report=fake_report(candidate_id="cand_fedcba987654"),
                ),
            )

    def test_failed_evaluator_result_fails_experiment(self):
        evaluating = self._evaluating_state()
        failed = transition(
            evaluating,
            ExperimentEvent(
                EventType.EVALUATOR_RESULT,
                report=fake_report(status=RunStatus.FAILED),
            ),
        )
        self.assertEqual(failed.phase, ExperimentPhase.FAILED)
        self.assertEqual(failed.failure_reason, "Synthetic run failure")

    def test_interrupted_evaluator_result_marks_experiment_interrupted(self):
        interrupted = transition(
            self._evaluating_state(),
            ExperimentEvent(
                EventType.EVALUATOR_RESULT,
                report=fake_report(status=RunStatus.INTERRUPTED),
            ),
        )
        self.assertEqual(interrupted.phase, ExperimentPhase.INTERRUPTED)

    def test_terminal_state_cannot_be_advanced(self):
        terminal = transition(
            self._evaluating_state(),
            ExperimentEvent(
                EventType.EVALUATOR_RESULT,
                report=fake_report(verdict=Verdict.ACCEPTED),
            ),
        )
        with self.assertRaisesRegex(ValueError, "terminal"):
            transition(terminal, ExperimentEvent(EventType.INTERRUPTED))

    def test_evaluator_result_event_requires_report(self):
        with self.assertRaisesRegex(ValueError, "require an EvaluationReport"):
            ExperimentEvent(EventType.EVALUATOR_RESULT)

    def test_report_must_match_run_candidate_hash_type_and_contract(self):
        report = fake_report()
        wrong_reports = (
            replace(report, run_id="another-run"),
            replace(report, candidate=replace(report.candidate, sha256="c" * 64)),
            replace(
                report, candidate=replace(report.candidate, implementation_type="metal")
            ),
            replace(report, contract_sha256="d" * 64),
        )
        state = self._evaluating_state()
        for wrong_report in wrong_reports:
            with self.subTest(report=wrong_report), self.assertRaises(ValueError):
                transition(
                    state,
                    ExperimentEvent(EventType.EVALUATOR_RESULT, report=wrong_report),
                )
        self.assertFalse(state.is_terminal)
        self.assertIsNone(state.evaluator_report)
        self.assertEqual(len(state.history), 3)

    def test_replayed_report_cannot_complete_a_different_run(self):
        state = new_experiment("exp-002", CANDIDATE_ID)
        different_run = replace(RUN, run_id="run-002")
        for event in (EventType.SUBMITTED, EventType.CANDIDATE_VALIDATED):
            state = transition(state, ExperimentEvent(event))
        state = transition(
            state, ExperimentEvent(EventType.CANDIDATE_FROZEN, run=different_run)
        )
        with self.assertRaisesRegex(ValueError, "different run"):
            transition(
                state, ExperimentEvent(EventType.EVALUATOR_RESULT, report=fake_report())
            )

    def test_freezing_requires_the_expected_candidate_identity(self):
        with self.assertRaisesRegex(ValueError, "require a RunIdentity"):
            ExperimentEvent(EventType.CANDIDATE_FROZEN)
        state = transition(self.initial, ExperimentEvent(EventType.SUBMITTED))
        state = transition(state, ExperimentEvent(EventType.CANDIDATE_VALIDATED))
        other = replace(
            RUN, candidate=CandidateIdentity("cand_fedcba987654", "mlx", "a" * 64)
        )
        with self.assertRaisesRegex(ValueError, "different candidate"):
            transition(state, ExperimentEvent(EventType.CANDIDATE_FROZEN, run=other))
        with self.assertRaisesRegex(ValueError, "only candidate_frozen"):
            ExperimentEvent(EventType.SUBMITTED, run=RUN)

    def test_invalid_run_provenance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_id"):
            replace(RUN, run_id=" ")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace(RUN, contract_sha256="main")
        with self.assertRaises(TypeError):
            replace(RUN, candidate=CANDIDATE_ID)

    def _evaluating_state(self):
        state = self.initial
        for event in (
            EventType.SUBMITTED,
            EventType.CANDIDATE_VALIDATED,
            EventType.CANDIDATE_FROZEN,
        ):
            state = transition(
                state,
                ExperimentEvent(
                    event, run=RUN if event is EventType.CANDIDATE_FROZEN else None
                ),
            )
        return state


if __name__ == "__main__":
    unittest.main()
