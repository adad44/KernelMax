"""State-machine tests use synthetic controller events and evaluator reports."""

import unittest

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
    new_experiment,
    transition,
)


CANDIDATE_ID = "cand_012345abcdef"
CONTRACT_HASH = "b" * 64


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
        state = transition(state, ExperimentEvent(EventType.CANDIDATE_FROZEN))
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
        self.assertEqual(len(completed.history), 5)
        self.assertEqual(state.phase, ExperimentPhase.BENCHMARK)  # Prior state is immutable.

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
            transition(self.initial, ExperimentEvent(EventType.CANDIDATE_FROZEN))

    def test_evaluator_report_must_match_candidate(self):
        state = self.initial
        for event in (
            EventType.SUBMITTED,
            EventType.CANDIDATE_VALIDATED,
            EventType.CANDIDATE_FROZEN,
        ):
            state = transition(state, ExperimentEvent(event))
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

    def _evaluating_state(self):
        state = self.initial
        for event in (
            EventType.SUBMITTED,
            EventType.CANDIDATE_VALIDATED,
            EventType.CANDIDATE_FROZEN,
        ):
            state = transition(state, ExperimentEvent(event))
        return state


if __name__ == "__main__":
    unittest.main()
