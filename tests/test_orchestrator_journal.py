"""Persistence and history tests for the orchestrator journal."""

import tempfile
import unittest
from pathlib import Path

from evaluator.schemas import CandidateIdentity, EvaluationReport, RunStatus
from orchestrator.journal import ExperimentJournal
from orchestrator.state_machine import (
    EventType,
    ExperimentEvent,
    ExperimentPhase,
    RunIdentity,
    new_experiment,
    transition,
)


CONTRACT_HASH = "b" * 64
CANDIDATE_ID = "cand_012345abcdef"


def make_report(run):
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


class ExperimentJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = ExperimentJournal(self.root / "journal")

    def test_appends_each_step_and_saves_latest_state_and_report(self):
        report = None
        state = new_experiment("journal-run", CANDIDATE_ID)
        for event in (ExperimentEvent(EventType.SUBMITTED),):
            state = transition(state, event)
            self.journal.record(state, event)

        event = ExperimentEvent(EventType.CANDIDATE_VALIDATED)
        state = transition(state, event)
        self.journal.record(state, event)

        identity = RunIdentity(
            "journal-run-id",
            CandidateIdentity(CANDIDATE_ID, "mlx", "a" * 64),
            CONTRACT_HASH,
        )
        event = ExperimentEvent(EventType.CANDIDATE_FROZEN, run=identity)
        state = transition(state, event)
        self.journal.record(state, event)

        report = make_report(identity)
        event = ExperimentEvent(EventType.EVALUATOR_RESULT, report=report)
        state = transition(state, event)
        self.journal.record(state, event)

        history = self.journal.read_history()
        self.assertEqual(len(history), 4)
        self.assertEqual(
            [entry["event"]["type"] for entry in history],
            ["submitted", "candidate_validated", "candidate_frozen", "evaluator_result"],
        )
        self.assertEqual(state.phase, ExperimentPhase.COMPLETED)
        saved = self.journal.read_state("journal-run")
        self.assertEqual(saved["phase"], "completed")
        self.assertEqual(saved["evaluator_report"], report.to_dict())
        self.assertEqual(history[-1]["event"]["report"], report.to_dict())

    def test_failure_reason_and_failed_phase_are_retained(self):
        state = transition(
            new_experiment("failed-run", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        event = ExperimentEvent(EventType.CANDIDATE_REJECTED, "manifest is malformed")
        state = transition(state, event)
        self.journal.record(state, event)

        self.assertEqual(state.phase, ExperimentPhase.FAILED)
        self.assertEqual(
            self.journal.read_history()[-1]["event"]["detail"],
            "manifest is malformed",
        )
        saved = self.journal.read_state("failed-run")
        self.assertEqual(saved["phase"], "failed")
        self.assertEqual(saved["failure_reason"], "manifest is malformed")

    def test_diagnostic_signal_is_logged_without_changing_current_phase(self):
        state = new_experiment("diagnostic-run", CANDIDATE_ID)
        event = ExperimentEvent(EventType.SUBMITTED)
        state = transition(state, event)
        self.journal.record(state, event)
        event = ExperimentEvent(EventType.CANDIDATE_VALIDATED)
        state = transition(state, event)
        self.journal.record(state, event)
        identity = RunIdentity(
            "diagnostic-run-id",
            CandidateIdentity(CANDIDATE_ID, "mlx", "a" * 64),
            CONTRACT_HASH,
        )
        event = ExperimentEvent(EventType.CANDIDATE_FROZEN, run=identity)
        state = transition(state, event)
        self.journal.record(state, event)
        before = state

        report = make_report(identity)
        self.journal.record_signal(state, "on_diagnostic_passed", report)

        self.assertIs(state, before)
        self.assertEqual(state.phase, ExperimentPhase.CORRECTNESS)
        entry = self.journal.read_history()[-1]
        self.assertEqual(entry["kind"], "evaluator_signal")
        self.assertEqual(entry["signal"], "on_diagnostic_passed")
        self.assertEqual(entry["report"], report.to_dict())
        self.assertEqual(self.journal.read_state("diagnostic-run")["phase"], "correctness")

    def test_state_file_is_atomically_replaced_without_following_symlink(self):
        state = transition(
            new_experiment("symlink-safe", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        event = state.history[-1]
        state_path = self.journal._state_path(state.experiment_id)
        outside = self.root / "outside.json"
        outside.write_text("keep me", encoding="utf-8")
        state_path.symlink_to(outside)

        self.journal.record(state, event)

        self.assertFalse(state_path.is_symlink())
        self.assertEqual(outside.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(
            self.journal.read_state(state.experiment_id)["phase"], "validating"
        )

    def test_record_requires_the_latest_transition(self):
        state = transition(
            new_experiment("ordered-run", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        with self.assertRaisesRegex(ValueError, "latest event"):
            self.journal.record(
                state, ExperimentEvent(EventType.CANDIDATE_VALIDATED)
            )
        self.assertEqual(self.journal.read_history(), [])


if __name__ == "__main__":
    unittest.main()
