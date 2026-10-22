"""Parent-owned grading of explicitly reviewed MLX snapshots.

Subprocesses, clean environments and hashes are reliability controls, NOT a
hostile-code sandbox. Never approve an unreviewed/agent-self-approved snapshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from evaluator.candidate_protocol import (
    VERSION,
    MAX_RESPONSE_BYTES,
    file_digest,
    pack_tensor,
    unpack_tensor,
    write_json,
)
from evaluator.correctness import InvalidReferenceError, compare_outputs
from evaluator.correctness_cases import case_plan, make_inputs
from evaluator.correctness_suite import LayerCheck, summarize_checks
from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    ExperimentContract,
    RunStatus,
)
from orchestrator.candidate_manager import FrozenCandidate, verify_frozen_candidate
from orchestrator.state_machine import RunIdentity


ROOT = Path(__file__).resolve().parents[1]
_WORKER = Path(__file__).with_name("candidate_worker.py")


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    """Controller-owned runtime config. Native hashes must come from verified pins."""

    mode: str = "numpy_fixture"
    python: Path = field(default_factory=lambda: Path(sys.executable))
    hidden_size: int = 8
    num_experts: int = 32
    model_path: Path | None = None
    model_revision: str | None = None
    model_files: dict[str, str] = field(default_factory=dict)
    expert_factory: str | None = None

    def __post_init__(self):
        if self.mode not in ("numpy_fixture", "mlx_native"):
            raise ValueError("unknown worker mode")
        if (
            type(self.hidden_size) is not int
            or self.hidden_size <= 0
            or type(self.num_experts) is not int
            or self.num_experts < 5
        ):
            raise ValueError("invalid worker dimensions")
        if not Path(self.python).is_file():
            raise ValueError("worker interpreter is unavailable")
        object.__setattr__(self, "python", Path(self.python).absolute())
        files = dict(self.model_files)
        if any(
            type(k) is not str
            or Path(k).is_absolute()
            or ".." in Path(k).parts
            or not re.fullmatch(r"[0-9a-f]{64}", v)
            for k, v in files.items()
        ):
            raise ValueError("invalid checkpoint file manifest")
        object.__setattr__(self, "model_files", MappingProxyType(files))
        if self.mode == "mlx_native":
            if (
                self.model_path is None
                or not Path(self.model_path).is_dir()
                or not files
                or self.model_revision is None
            ):
                raise ValueError(
                    "native execution requires available local weights, revision and verified hashes"
                )
            object.__setattr__(self, "model_path", Path(self.model_path).resolve())
        elif (
            self.model_path is not None
            or files
            or self.expert_factory is not None
            or self.model_revision is not None
        ):
            raise ValueError("fixture mode cannot claim model/factory provenance")
        if self.expert_factory is not None and not re.fullmatch(
            r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", self.expert_factory
        ):
            raise ValueError("expert_factory must be a trusted module:symbol")

    def payload(self):
        return {
            "mode": self.mode,
            "python": str(self.python),
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_revision": self.model_revision,
            "model_files": dict(self.model_files),
            "expert_factory": self.expert_factory,
        }


@dataclass(frozen=True, slots=True)
class _Session:
    rows: dict
    errors: tuple[str, ...]
    interrupted: bool = False


def _compare_transported(reference, candidate, *, case_name, policy):
    result = compare_outputs(
        unpack_tensor(reference),
        unpack_tensor(candidate),
        case_name=case_name,
        policy=policy,
    )
    if (reference["backend"], reference["dtype"]) != (
        candidate["backend"],
        candidate["dtype"],
    ):
        result = replace(
            result,
            passed=False,
            dtype_match=False,
            tolerance_passed=False,
            detail=result.detail
            + "; original backend/dtype mismatch before transport conversion",
        )
    return result


def _protected_files(extra):
    paths = {ROOT / "AGENTS.md", ROOT / "kernelmaxxing.yaml"}
    for directory in ("evaluator", "target", "orchestrator"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update(Path(path).absolute() for path in extra)
    return {str(path): file_digest(path) for path in sorted(paths)}


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _read_session(path, request, plan, limit):
    rows, errors = {}, []
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("response must be a regular unlinked file")
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError("worker response exceeds byte limit")
        records = data.splitlines()
        if not records:
            raise ValueError("worker returned no session header")
        header = json.loads(records[0])
        if (
            header.get("type"),
            header.get("version"),
            header.get("phase"),
            header.get("identity"),
            header.get("mode"),
        ) != (
            "session",
            VERSION,
            request["phase"],
            request["identity"],
            request["runtime"]["mode"],
        ):
            raise ValueError("worker session identity/mode mismatch")
        expected = {(case.case_name, case.layer_index): case for case in plan}
        complete = False
        for line in records[1:]:
            item = json.loads(line)
            if complete:
                raise ValueError("unexpected records after worker completion")
            if item.get("type") == "error":
                errors.append(str(item.get("error", "worker error"))[:4096])
                continue
            if item.get("type") == "complete":
                if item.get("rows") != len(plan) or len(rows) != len(plan):
                    raise ValueError("worker completed with missing results")
                complete = True
                continue
            key = (item.get("case_name"), item.get("layer_index"))
            if (
                item.get("type") != "result"
                or type(item.get("layer_index")) is not int
                or type(item.get("seed")) is not int
                or key not in expected
                or key in rows
                or item.get("seed") != expected[key].seed
            ):
                raise ValueError("duplicate, unexpected or mismatched worker result")
            if (item.get("output") is None) == (item.get("error") is None):
                raise ValueError("worker result requires exactly one output or error")
            if item["error"] is not None and (
                type(item["error"]) is not str
                or not item["error"].strip()
                or len(item["error"]) > 4096
            ):
                raise ValueError("malformed worker error")
            if item["output"] is not None:
                unpack_tensor(item["output"], limit)
                if not re.fullmatch(r"[0-9a-f]{64}", item.get("input_sha256", "")):
                    raise ValueError("missing input fingerprint")
            rows[key] = item
        if not complete:
            errors.append("worker response is incomplete")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"invalid worker response: {type(exc).__name__}: {exc}")
    return rows, errors


def _session(request, directory, runtime, plan, timeout, log_limit, response_limit):
    directory.mkdir()
    (directory / "home").mkdir()
    request_path, response_path = (
        directory / "request.json",
        directory / "response.jsonl",
    )
    write_json(request_path, request)
    env = {
        "PATH": str(runtime.python.parent),
        "HOME": str(directory / "home"),
        "TMPDIR": str(directory),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    errors, interrupted = [], False
    process = subprocess.Popen(
        [
            str(runtime.python),
            "-I",
            "-B",
            str(_WORKER),
            str(request_path),
            str(response_path),
        ],
        cwd=directory,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    write_json(
        directory / "process.json", {"pid": process.pid, "phase": request["phase"]}
    )
    selector = selectors.DefaultSelector()
    for stream in (process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    deadline, total = time.monotonic() + timeout, 0
    try:
        with (directory / "worker.log").open("wb") as log:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker execution deadline exceeded")
                if (
                    response_path.exists()
                    and response_path.lstat().st_size > response_limit
                ):
                    raise ValueError("worker response exceeds byte limit")
                for key, _ in selector.select(
                    timeout=min(0.05, max(0, deadline - time.monotonic()))
                ):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        log.write(chunk[: max(0, log_limit - total)])
                        total += len(chunk)
                        if total > log_limit:
                            raise ValueError("worker log exceeds byte limit")
            # A short-lived worker can exit before poll/select first observes
            # it. Still inspect buffered output before closing its pipes.
            for stream in (process.stdout, process.stderr):
                while True:
                    try:
                        chunk = os.read(stream.fileno(), 4096)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    log.write(chunk[: max(0, log_limit - total)])
                    total += len(chunk)
                    if total > log_limit:
                        raise ValueError("worker log exceeds byte limit")
    except KeyboardInterrupt:
        interrupted = True
        errors.append("worker interrupted")
    except (TimeoutError, ValueError, OSError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        _kill_group(
            process
        )  # Also kill inherited-process-group descendants after normal exit.
        selector.close()
        process.stdout.close()
        process.stderr.close()
    if process.returncode != 0:
        errors.append(f"worker exited with status {process.returncode}")
    rows, response_errors = _read_session(response_path, request, plan, response_limit)
    return _Session(rows, tuple(errors + response_errors), interrupted)


@dataclass(frozen=True, slots=True)
class ReviewedCandidateEvaluator:
    """Grade approved frozen MLX code without importing it into this process.

    `approved_sha256s` is HUMAN/CONTROLLER owned. It is never read from the
    candidate manifest. All artifacts remain under artifact_root on every exit.
    """

    approved_sha256s: frozenset[str]
    artifact_root: Path
    runtime: WorkerRuntime = field(default_factory=WorkerRuntime)
    protected_paths: tuple[Path, ...] = ()
    timeout_seconds: float = 600
    max_log_bytes: int = 65536
    max_response_bytes: int = MAX_RESPONSE_BYTES
    supports_candidate_correctness = True

    def __post_init__(self):
        approved = frozenset(self.approved_sha256s)
        if any(
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in approved
        ):
            raise ValueError("approvals must be exact snapshot SHA-256 values")
        object.__setattr__(self, "approved_sha256s", approved)
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).absolute())
        object.__setattr__(
            self,
            "protected_paths",
            tuple(Path(path).absolute() for path in self.protected_paths),
        )
        if not isinstance(self.runtime, WorkerRuntime):
            raise TypeError("runtime must be a WorkerRuntime")
        if (
            not np.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or any(
                type(n) is not int or n <= 0
                for n in (self.max_log_bytes, self.max_response_bytes)
            )
        ):
            raise ValueError("worker limits must be finite positive bounds")

    def __call__(
        self,
        run_identity,
        frozen_candidate,
        contract,
        *,
        on_diagnostic_passed=None,
        on_candidate_correctness_passed=None,
    ):
        if not isinstance(contract, ExperimentContract):
            raise TypeError("contract must be a validated ExperimentContract")
        if not isinstance(run_identity, RunIdentity) or not isinstance(
            frozen_candidate, FrozenCandidate
        ):
            raise TypeError("expected controller RunIdentity and FrozenCandidate")
        candidate = CandidateIdentity(
            frozen_candidate.candidate_id,
            frozen_candidate.implementation_type,
            frozen_candidate.sha256,
        )
        run_identity.validate_report(
            EvaluationReport(
                run_identity.run_id,
                candidate,
                contract.sha256,
                None,
                (),
                (),
                RunStatus.COMPLETED,
                (),
                None,
            )
        )
        for callback in (on_diagnostic_passed, on_candidate_correctness_passed):
            if callback is not None and not callable(callback):
                raise TypeError("callbacks must be callable")
        report = EvaluationReport(
            run_identity.run_id,
            candidate,
            contract.sha256,
            None,
            (),
            (),
            RunStatus.COMPLETED,
            (),
            None,
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        artifacts = Path(tempfile.mkdtemp(prefix="execution-", dir=self.artifact_root))
        protected, checks, errors, interrupted, started = None, [], [], False, False
        plan = case_plan(contract.correctness)
        protected_paths = (
            tuple(self.protected_paths)
            + tuple(self.runtime.model_path / name for name in self.runtime.model_files)
            if self.runtime.model_path
            else tuple(self.protected_paths)
        )
        try:
            if candidate.sha256 not in self.approved_sha256s:
                raise ValueError(
                    "frozen snapshot has no human/controller review approval"
                )
            if candidate.implementation_type != "mlx":
                raise ValueError(
                    "reviewed execution supports MLX Python candidates only"
                )
            verify_frozen_candidate(frozen_candidate)
            protected = _protected_files(protected_paths)
            write_json(artifacts / "protected_sources.json", protected)
            if (
                self.runtime.mode == "mlx_native"
                and self.runtime.model_revision != contract.model.revision
            ):
                raise ValueError("native checkpoint revision disagrees with contract")
            identity = {
                "run_id": run_identity.run_id,
                "candidate": candidate.to_dict(),
                "contract_sha256": contract.sha256,
            }
            inputs = []
            for case in plan:
                x, ids = make_inputs(
                    case,
                    hidden_size=self.runtime.hidden_size,
                    num_experts=self.runtime.num_experts,
                )
                inputs.append(
                    {
                        "case_name": case.case_name,
                        "layer_index": case.layer_index,
                        "seed": case.seed,
                        "x": pack_tensor(x),
                        "ids": pack_tensor(ids),
                    }
                )
            request = {
                "version": VERSION,
                "identity": identity,
                "approved_sha256": candidate.sha256,
                "snapshot": {
                    "directory": str(frozen_candidate.directory),
                    "sha256": candidate.sha256,
                    "file_sha256": dict(frozen_candidate.file_sha256),
                },
                "runtime": self.runtime.payload(),
                "inputs": inputs,
                "max_response_bytes": self.max_response_bytes,
            }
            reference = _session(
                dict(request, phase="reference"),
                artifacts / "reference",
                self.runtime,
                plan,
                self.timeout_seconds,
                self.max_log_bytes,
                self.max_response_bytes,
            )
            errors.extend(f"reference: {error}" for error in reference.errors)
            interrupted = reference.interrupted
            valid = {}
            for case in plan:
                key = (case.case_name, case.layer_index)
                row = reference.rows.get(key)
                try:
                    if row is None or row["error"] is not None:
                        raise InvalidReferenceError(
                            row["error"] if row else "missing reference output"
                        )
                    tensor = row["output"]
                    array = unpack_tensor(tensor)
                    backend, dtype = (
                        ("numpy", "<f4")
                        if self.runtime.mode == "numpy_fixture"
                        else ("mlx", "mlx.core.bfloat16")
                    )
                    if (
                        tensor["backend"] != backend
                        or tensor["dtype"] != dtype
                        or array.shape != (1, case.tokens, 4, self.runtime.hidden_size)
                    ):
                        raise InvalidReferenceError(
                            "reference has wrong shape/backend/original dtype"
                        )
                    compare_outputs(
                        array,
                        array,
                        case_name=case.case_name,
                        policy=contract.correctness,
                    )
                    if not np.isfinite(array).all():
                        raise InvalidReferenceError("reference is nonfinite")
                    valid[key] = array
                except (ValueError, TypeError) as exc:
                    checks.append(
                        LayerCheck(
                            case, error=f"reference: {type(exc).__name__}: {exc}"
                        )
                    )
            if errors or checks:
                raise RuntimeError(
                    "reference session failed; candidate was not executed"
                )
            verify_frozen_candidate(frozen_candidate)
            if protected != _protected_files(protected_paths):
                raise ValueError("protected sources changed during reference execution")
            # Reference process has exited/reaped BEFORE candidate/model starts.
            started = True
            actual = _session(
                dict(request, phase="candidate"),
                artifacts / "candidate",
                self.runtime,
                plan,
                self.timeout_seconds,
                self.max_log_bytes,
                self.max_response_bytes,
            )
            errors.extend(f"candidate: {error}" for error in actual.errors)
            interrupted = actual.interrupted
            for case in plan:
                key = (case.case_name, case.layer_index)
                row = actual.rows.get(key)
                try:
                    if row is None or row["error"] is not None:
                        raise ValueError(
                            row["error"] if row else "missing candidate output"
                        )
                    if row["input_sha256"] != reference.rows[key]["input_sha256"]:
                        raise ValueError(
                            "candidate/reference input fingerprints disagree"
                        )
                    golden, observed = reference.rows[key]["output"], row["output"]
                    result = _compare_transported(
                        golden,
                        observed,
                        case_name=case.case_name,
                        policy=contract.correctness,
                    )
                    checks.append(LayerCheck(case, result=result))
                except (ValueError, TypeError) as exc:
                    checks.append(
                        LayerCheck(
                            case, error=f"candidate: {type(exc).__name__}: {exc}"
                        )
                    )
        except KeyboardInterrupt:
            interrupted = True
            errors.append("parent evaluation interrupted")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                verify_frozen_candidate(frozen_candidate)
                if protected is not None and protected != _protected_files(
                    protected_paths
                ):
                    raise ValueError("protected sources changed during execution")
            except Exception as exc:
                errors.append(f"integrity: {type(exc).__name__}: {exc}")
        suite = summarize_checks(checks, policy=contract.correctness)
        errors.extend(suite.errors)
        results = []
        for result in suite.correctness:
            detail = json.loads(result.detail)
            detail.update(
                evaluation_scope="reviewed_candidate_execution",
                input_mode=self.runtime.mode,
                candidate_execution_started=started,
                performance_measured=False,
                artifacts=str(artifacts),
                runtime=self.runtime.payload(),
                approved_snapshot_sha256=candidate.sha256
                if candidate.sha256 in self.approved_sha256s
                else None,
            )
            results.append(replace(result, detail=json.dumps(detail, sort_keys=True)))
        report = replace(
            report,
            correctness=tuple(results),
            errors=tuple(errors),
            status=RunStatus.INTERRUPTED
            if interrupted
            else RunStatus.FAILED
            if errors
            else RunStatus.COMPLETED,
        )
        if not errors and suite.passed and on_candidate_correctness_passed is not None:
            try:
                on_candidate_correctness_passed(report)
                verify_frozen_candidate(frozen_candidate)
                if protected != _protected_files(protected_paths):
                    raise ValueError(
                        "protected sources changed during correctness callback"
                    )
            except KeyboardInterrupt:
                report = replace(
                    report,
                    status=RunStatus.INTERRUPTED,
                    errors=("candidate correctness callback interrupted",),
                )
            except Exception as exc:
                report = replace(
                    report,
                    status=RunStatus.FAILED,
                    errors=(
                        f"candidate correctness callback: {type(exc).__name__}: {exc}",
                    ),
                )
        run_identity.validate_report(report)
        write_json(artifacts / "report.json", report.to_dict())
        return report
