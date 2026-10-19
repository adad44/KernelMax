"""Persistence and history tests for the orchestrator journal."""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.schemas import CandidateIdentity, EvaluationReport, RunStatus
from orchestrator.journal import ExperimentJournal, JournalError
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
            [
                "submitted",
                "candidate_validated",
                "candidate_frozen",
                "evaluator_result",
            ],
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
        self.assertEqual(
            self.journal.read_state("diagnostic-run")["phase"], "correctness"
        )

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
            self.journal.record(state, ExperimentEvent(EventType.CANDIDATE_VALIDATED))
        self.assertEqual(self.journal.read_history(), [])

    def test_partial_write_is_quarantined_after_restart_without_losing_history(self):
        first = transition(
            new_experiment("first", CANDIDATE_ID), ExperimentEvent(EventType.SUBMITTED)
        )
        self.journal.record(first, first.history[-1])
        committed = self.journal.history_path.read_bytes()
        second = transition(
            new_experiment("interrupted-write", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        real_write = os.write
        writes = 0

        def partial_write(fd, data):
            nonlocal writes
            writes += 1
            if writes == 1:
                return real_write(fd, data[:20])
            raise OSError(errno.ENOSPC, "simulated disk full")

        with (
            patch("orchestrator.journal.os.write", side_effect=partial_write),
            self.assertRaises(JournalError),
        ):
            self.journal.record(second, second.history[-1])
        fragment = self.journal.history_path.read_bytes()[len(committed) :]
        self.assertEqual(len(fragment), 20)
        self.assertIsNone(self.journal.read_state("interrupted-write"))

        restarted = ExperimentJournal(self.journal.root)
        self.assertEqual(len(restarted.read_history()), 1)
        self.assertEqual(restarted.history_path.read_bytes(), committed)
        quarantined = list(restarted.root.glob("history-incomplete-*.bin"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), fragment)
        self.assertEqual(quarantined[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(restarted.read_state("first")["phase"], "validating")

        third = transition(
            new_experiment("after-restart", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        restarted.record(third, third.history[-1])
        self.assertEqual(
            [entry["experiment_id"] for entry in restarted.read_history()],
            ["first", "after-restart"],
        )
        self.assertTrue(restarted.history_path.read_bytes().startswith(committed))
        self.assertEqual(restarted.read_state("after-restart")["phase"], "validating")

    def test_append_recovers_binary_and_large_unterminated_tails(self):
        for index, fragment in enumerate(
            (b'{"unfinished":', b'{"valid":true}', b"\xf0\x9f", b"x" * 20000)
        ):
            with self.subTest(fragment_index=index):
                journal = ExperimentJournal(self.root / f"tail-{index}")
                first = transition(
                    new_experiment("first", CANDIDATE_ID),
                    ExperimentEvent(EventType.SUBMITTED),
                )
                journal.record(first, first.history[-1])
                committed = journal.history_path.read_bytes()
                with journal.history_path.open("ab") as stream:
                    stream.write(fragment)
                restarted = ExperimentJournal(journal.root)
                second = transition(
                    new_experiment("second", CANDIDATE_ID),
                    ExperimentEvent(EventType.SUBMITTED),
                )
                restarted.record(second, second.history[-1])
                self.assertEqual(len(restarted.read_history()), 2)
                self.assertTrue(
                    restarted.history_path.read_bytes().startswith(committed)
                )
                quarantined = list(journal.root.glob("history-incomplete-*.bin"))
                self.assertEqual(len(quarantined), 1)
                self.assertEqual(quarantined[0].read_bytes(), fragment)

    def test_first_partial_record_can_be_recovered_without_a_committed_prefix(self):
        fragment = b'{"candidate_id":"cand_'
        self.journal.history_path.write_bytes(fragment)
        self.assertEqual(self.journal.read_history(), [])
        self.assertEqual(self.journal.history_path.read_bytes(), b"")
        quarantined = list(self.journal.root.glob("history-incomplete-*.bin"))
        self.assertEqual(quarantined[0].read_bytes(), fragment)

    def test_failed_quarantine_never_truncates_or_appends_history(self):
        raw = b'{"incomplete":'
        self.journal.history_path.write_bytes(raw)
        state = transition(
            new_experiment("blocked", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        with patch.object(
            self.journal, "_preserve_tail", side_effect=OSError("disk full")
        ):
            with self.assertRaises(JournalError):
                self.journal.record(state, state.history[-1])
            with self.assertRaises(JournalError):
                self.journal.read_history()
        self.assertEqual(self.journal.history_path.read_bytes(), raw)
        self.assertIsNone(self.journal.read_state("blocked"))

    def test_corrupt_complete_records_are_not_silently_removed(self):
        raw = b'{"complete":broken}\n'
        self.journal.history_path.write_bytes(raw)
        with self.assertRaises(JournalError):
            self.journal.read_history()
        self.assertEqual(self.journal.history_path.read_bytes(), raw)
        self.assertEqual(list(self.journal.root.glob("history-incomplete-*.bin")), [])

    def test_healthy_history_is_unchanged_when_read(self):
        state = transition(
            new_experiment("healthy", CANDIDATE_ID),
            ExperimentEvent(EventType.SUBMITTED),
        )
        self.journal.record(state, state.history[-1])
        committed = self.journal.history_path.read_bytes()
        self.assertEqual(len(self.journal.read_history()), 1)
        self.assertEqual(len(self.journal.read_history()), 1)
        self.assertEqual(self.journal.history_path.read_bytes(), committed)
        self.assertEqual(list(self.journal.root.glob("history-incomplete-*.bin")), [])


if __name__ == "__main__":
    unittest.main()
