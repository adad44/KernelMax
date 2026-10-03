"""Creation and validation helpers for optimizer candidate submissions.

Candidate code is untrusted input. This module checks the on-disk submission
contract; it does not import or execute candidate code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


DEFAULT_CANDIDATES_ROOT = Path(__file__).resolve().parent.parent / "candidates"
MANIFEST_NAME = "manifest.json"
COMMON_REQUIRED_FILES = (MANIFEST_NAME, "wrapper.py")
IMPLEMENTATION_FILES = {
    "mlx": "implementation.py",
    "metal": "implementation.metal",
}
REQUIRED_MANIFEST_FIELDS = frozenset(
    {"candidate_id", "name", "implementation_type", "description"}
)
ALLOWED_MANIFEST_FIELDS = REQUIRED_MANIFEST_FIELDS
_CANDIDATE_ID_RE = re.compile(r"cand_[0-9a-f]{12}\Z")


class CandidateValidationError(ValueError):
    """Raised when a candidate does not satisfy the submission contract."""


@dataclass(frozen=True)
class Candidate:
    """Validated candidate metadata and its confined on-disk directory."""

    candidate_id: str
    name: str
    implementation_type: str
    description: str
    directory: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenCandidate(Candidate):
    """Controller-owned snapshot with hashes of its paths and file contents."""

    sha256: str
    file_sha256: Mapping[str, str]


def create_candidate_id() -> str:
    """Return a collision-resistant ID suitable for a candidate directory."""

    return f"cand_{uuid.uuid4().hex[:12]}"


def _confined_candidate_dir(candidate_dir: str | Path, candidates_root: str | Path) -> Path:
    root = Path(candidates_root).resolve()
    candidate = Path(candidate_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise CandidateValidationError(
            "candidate must be a real direct child directory, not a symlink"
        )
    candidate = candidate.resolve()

    if candidate.parent != root:
        raise CandidateValidationError(
            f"candidate must be a direct child of the allowed directory: {root}"
        )
    if not candidate.is_dir():
        raise CandidateValidationError(f"candidate directory does not exist: {candidate}")
    for path in candidate.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(candidate):
            raise CandidateValidationError(
                f"candidate symlink points outside its allowed directory: {path.relative_to(candidate)}"
            )
    return candidate


def validate_candidate(
    candidate_dir: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
) -> Candidate:
    """Validate a candidate's directory, manifest, implementation type, and files.

    ``candidate_dir`` may be an absolute path or a child directory name relative
    to ``candidates_root``. The manifest ID must equal the directory name.
    """

    try:
        directory = _confined_candidate_dir(candidate_dir, candidates_root)
    except (OSError, RuntimeError) as exc:
        raise CandidateValidationError(f"cannot resolve candidate directory: {exc}") from exc
    if not _CANDIDATE_ID_RE.fullmatch(directory.name):
        raise CandidateValidationError(
            "candidate directory name must be a generated ID of the form cand_<12 lowercase hex>"
        )

    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CandidateValidationError(f"missing required file: {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(f"cannot read valid JSON from {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CandidateValidationError("manifest must contain a JSON object")

    missing_fields = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing_fields:
        raise CandidateValidationError(f"manifest is missing required fields: {', '.join(missing_fields)}")
    unknown_fields = sorted(manifest.keys() - ALLOWED_MANIFEST_FIELDS)
    if unknown_fields:
        raise CandidateValidationError(f"manifest contains unsupported fields: {', '.join(unknown_fields)}")

    candidate_id = manifest["candidate_id"]
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise CandidateValidationError("candidate_id must have the form cand_<12 lowercase hex>")
    if candidate_id != directory.name:
        raise CandidateValidationError(
            f"candidate_id {candidate_id!r} must match directory name {directory.name!r}"
        )

    for field in ("name", "description"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise CandidateValidationError(f"{field} must be a non-empty string")

    implementation_type = manifest["implementation_type"]
    if not isinstance(implementation_type, str) or implementation_type not in IMPLEMENTATION_FILES:
        allowed = ", ".join(sorted(IMPLEMENTATION_FILES))
        raise CandidateValidationError(
            f"implementation_type must be one of: {allowed}"
        )

    required_files = (*COMMON_REQUIRED_FILES, IMPLEMENTATION_FILES[implementation_type])
    for filename in required_files:
        file_path = directory / filename
        if not file_path.is_file():
            raise CandidateValidationError(f"missing required file for {implementation_type}: {filename}")
    return Candidate(
        candidate_id=candidate_id,
        name=manifest["name"].strip(),
        implementation_type=implementation_type,
        description=manifest["description"].strip(),
        directory=directory,
        manifest=MappingProxyType(manifest),
    )


def _read_regular_file(path: Path) -> bytes:
    """Read a source file without following a final-component symlink."""

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CandidateValidationError(
                f"snapshot files must be regular files without hard links: {path.name}"
            )
        return stream.read()


def _submission_paths(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise CandidateValidationError("snapshot must be a real directory")
    paths = sorted(directory.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise CandidateValidationError("frozen candidates must not contain symlinks")
    return paths


def _snapshot_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for path in _submission_paths(directory):
        if not path.is_dir():
            hashes[path.relative_to(directory).as_posix()] = hashlib.sha256(
                _read_regular_file(path)
            ).hexdigest()
    return hashes


def _bundle_hash(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remove_snapshot(directory: Path) -> None:
    # Restore owner permissions so cleanup also works after partial chmod failure.
    directory.chmod(0o700)
    for path in directory.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
    shutil.rmtree(directory)


def freeze_candidate(
    candidate_dir: str | Path,
    *,
    frozen_root: str | Path,
    candidates_root: str | Path = DEFAULT_CANDIDATES_ROOT,
) -> FrozenCandidate:
    """Copy a submission into a new read-only, hashed controller snapshot.

    The controller must stop the candidate writer before calling this function
    and keep ``frozen_root`` inaccessible to optimization agents. This function
    does not sandbox processes or freeze the evaluator's protected inputs.
    Existing snapshot IDs are never overwritten. All files are copied; symlinks,
    hard links, and special files are rejected. Evaluate only the returned path.
    """

    candidate = validate_candidate(candidate_dir, candidates_root)
    root = Path(frozen_root).resolve()
    if root.is_relative_to(Path(candidates_root).resolve()):
        raise CandidateValidationError("frozen_root must be outside candidates_root")
    destination = root / candidate.candidate_id
    try:
        root.mkdir(parents=True, exist_ok=True)
        destination.mkdir()  # Reserve the ID without replacing an existing snapshot.
    except OSError as exc:
        raise CandidateValidationError(f"cannot create new candidate snapshot: {exc}") from exc

    try:
        for path in _submission_paths(candidate.directory):
            copied = destination / path.relative_to(candidate.directory)
            if path.is_dir():
                copied.mkdir()
            else:
                copied.write_bytes(_read_regular_file(path))
        snapshot = validate_candidate(destination, root)
        hashes = _snapshot_hashes(destination)
        frozen = FrozenCandidate(
            candidate_id=snapshot.candidate_id,
            name=snapshot.name,
            implementation_type=snapshot.implementation_type,
            description=snapshot.description,
            directory=destination,
            manifest=snapshot.manifest,
            sha256=_bundle_hash(hashes),
            file_sha256=MappingProxyType(hashes),
        )
        for path in sorted(destination.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
        return frozen
    except (OSError, CandidateValidationError) as exc:
        _remove_snapshot(destination)
        if isinstance(exc, CandidateValidationError):
            raise
        raise CandidateValidationError(f"cannot freeze candidate: {exc}") from exc


def verify_frozen_candidate(candidate: FrozenCandidate) -> None:
    """Reject a snapshot whose file names or contents changed since freezing."""

    try:
        hashes = _snapshot_hashes(candidate.directory)
    except OSError as exc:
        raise CandidateValidationError(f"cannot verify frozen candidate: {exc}") from exc
    if hashes != candidate.file_sha256 or _bundle_hash(hashes) != candidate.sha256:
        raise CandidateValidationError("frozen candidate contents changed")
