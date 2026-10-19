"""Append-only experiment history and atomic current-state snapshots.

The journal is controller-owned runtime state. It stores evaluator outcomes
for audit and recovery, but it does not publish or replace official reports.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evaluator.schemas import EvaluationReport
from orchestrator.state_machine import ExperimentEvent, ExperimentState, RunIdentity


class JournalError(OSError):
    """Raised when the controller cannot durably record experiment state."""


def _run_mapping(run: RunIdentity | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": run.run_id,
        "candidate": run.candidate.to_dict(),
        "contract_sha256": run.contract_sha256,
    }


def _event_mapping(event: ExperimentEvent) -> dict[str, Any]:
    return {
        "type": event.type.value,
        "detail": event.detail,
        "run": _run_mapping(event.run),
        "report": event.report.to_dict() if event.report is not None else None,
    }


def state_mapping(state: ExperimentState) -> dict[str, Any]:
    """Return a JSON-safe copy of the state without altering report objects."""

    return {
        "experiment_id": state.experiment_id,
        "candidate_id": state.candidate_id,
        "phase": state.phase.value,
        "history": [_event_mapping(event) for event in state.history],
        "evaluator_report": (
            state.evaluator_report.to_dict()
            if state.evaluator_report is not None
            else None
        ),
        "failure_reason": state.failure_reason,
        "run": _run_mapping(state.run),
    }


class ExperimentJournal:
    """Persist event history and the latest state for each experiment.

    History is JSON Lines and append-only. State snapshots use a same-directory
    temporary file, file and directory fsync, and atomic replacement. State
    filenames are derived from a hash of experiment IDs to avoid path traversal.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"
        self.lock_path = self.root / ".journal.lock"
        self.states_path = self.root / "states"
        self.states_path.mkdir(parents=True, exist_ok=True)

    def _state_path(self, experiment_id: str) -> Path:
        key = hashlib.sha256(experiment_id.encode("utf-8")).hexdigest()
        return self.states_path / f"{key}.json"

    def record(
        self,
        state: ExperimentState,
        event: ExperimentEvent,
        *,
        observed_report: EvaluationReport | None = None,
    ) -> None:
        """Append one transition and atomically persist its resulting state."""

        if not isinstance(state, ExperimentState) or not isinstance(
            event, ExperimentEvent
        ):
            raise TypeError("record requires an ExperimentState and ExperimentEvent")
        if not state.history or state.history[-1] != event:
            raise ValueError("event must be the latest event in the supplied state")
        if observed_report is not None and not isinstance(
            observed_report, EvaluationReport
        ):
            raise TypeError("observed_report must be an EvaluationReport or None")
        entry = {
            "kind": "state_transition",
            "experiment_id": state.experiment_id,
            "candidate_id": state.candidate_id,
            "phase": state.phase.value,
            "event": _event_mapping(event),
            "observed_report": (
                observed_report.to_dict() if observed_report is not None else None
            ),
        }
        self._append_and_save(state, entry)

    def record_signal(
        self, state: ExperimentState, signal: str, report: EvaluationReport
    ) -> None:
        """Retain an evaluator diagnostic signal without changing experiment state."""

        if not isinstance(state, ExperimentState):
            raise TypeError("record_signal requires an ExperimentState")
        if not isinstance(signal, str) or not signal.strip():
            raise ValueError("signal must be a non-empty string")
        if not isinstance(report, EvaluationReport):
            raise TypeError("signal report must be an EvaluationReport")
        if state.run is None:
            raise ValueError("diagnostic signals require a bound run identity")
        state.run.validate_report(report)
        self._append_and_save(
            state,
            {
                "kind": "evaluator_signal",
                "experiment_id": state.experiment_id,
                "candidate_id": state.candidate_id,
                "phase": state.phase.value,
                "signal": signal.strip(),
                "report": report.to_dict(),
            },
        )

    def record_rejection(
        self, experiment_id: str, candidate_dir: str | Path, reason: str
    ) -> None:
        """Retain a rejected submission even when its ID cannot form a state."""

        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rejection reason must be a non-empty string")
        self._append_and_save(
            None,
            {
                "kind": "submission_rejected",
                "experiment_id": experiment_id,
                "candidate_id": Path(candidate_dir).name,
                "candidate_path": str(candidate_dir),
                "phase": "failed",
                "reason": reason,
            },
        )

    def _append_and_save(
        self, state: ExperimentState | None, entry: dict[str, Any]
    ) -> None:
        encoded = (
            json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")

        lock_fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._recover_tail()
            self._append(encoded)
            self._sync_directory(self.root)
            if state is not None:
                self._save_state(
                    self._state_path(state.experiment_id), state_mapping(state)
                )
        except (OSError, TypeError, ValueError) as exc:
            raise JournalError(f"cannot record experiment journal: {exc}") from exc
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _recover_tail(self) -> None:
        """Quarantine an unfinished last record while holding the journal lock.

        A newline is the commit boundary for a history record. Never truncate
        until the exact unfinished bytes and their directory entry are durable.
        Complete records, including malformed newline-terminated ones, stay put.
        """

        try:
            fd = os.open(self.history_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return
        try:
            end = os.lseek(fd, 0, os.SEEK_END)
            if end == 0 or os.pread(fd, 1, end - 1) == b"\n":
                return
            position = end
            boundary = 0
            while position:
                start = max(0, position - 8192)
                block = os.pread(fd, position - start, start)
                newline = block.rfind(b"\n")
                if newline >= 0:
                    boundary = start + newline + 1
                    break
                position = start
            # Read through a duplicate descriptor without loading the older log.
            with os.fdopen(os.dup(fd), "rb") as stream:
                stream.seek(boundary)
                fragment = stream.read()
            self._preserve_tail(fragment)
            os.ftruncate(fd, boundary)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _preserve_tail(self, fragment: bytes) -> None:
        fd, name = tempfile.mkstemp(
            prefix="history-incomplete-", suffix=".bin", dir=self.root
        )
        path = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(fragment)
                stream.flush()
                os.fsync(stream.fileno())
            self._sync_directory(self.root)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _sync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append(self, encoded: bytes) -> None:
        fd = os.open(
            self.history_path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("short write while appending history")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _save_state(path: Path, state: dict[str, Any]) -> None:
        data = (
            json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise

    def read_history(self) -> list[dict[str, Any]]:
        """Read history, recovering an unfinished final record if necessary."""

        lock_fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._recover_tail()
            try:
                fd = os.open(
                    self.history_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
            except FileNotFoundError:
                return []
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                return [json.loads(line) for line in stream]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalError(f"cannot read experiment history: {exc}") from exc
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def read_state(self, experiment_id: str) -> dict[str, Any] | None:
        """Read an experiment's latest saved snapshot, if one exists."""

        path = self._state_path(experiment_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalError(f"cannot read saved experiment state: {exc}") from exc
