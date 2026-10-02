"""Creation and validation helpers for optimizer candidate submissions.

Candidate code is untrusted input. This module checks the on-disk submission
contract; it does not import or execute candidate code.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES_ROOT = Path(__file__).resolve().parent.parent / "candidates"
MANIFEST_NAME = "manifest.json"
COMMON_REQUIRED_FILES = (MANIFEST_NAME, "wrapper.py")
IMPLEMENTATION_FILES = {
    "python": "implementation.py",
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
    manifest: dict[str, Any]


def create_candidate_id() -> str:
    """Return a collision-resistant ID suitable for a candidate directory."""

    return f"cand_{uuid.uuid4().hex[:12]}"


def _confined_candidate_dir(candidate_dir: str | Path, candidates_root: str | Path) -> Path:
    root = Path(candidates_root).resolve()
    candidate = Path(candidate_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    if candidate.parent != root:
        raise CandidateValidationError(
            f"candidate must be a direct child of the allowed directory: {root}"
        )
    if not candidate.is_dir():
        raise CandidateValidationError(f"candidate directory does not exist: {candidate}")
    for path in candidate.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(candidate):
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

    directory = _confined_candidate_dir(candidate_dir, candidates_root)
    if not _CANDIDATE_ID_RE.fullmatch(directory.name):
        raise CandidateValidationError(
            "candidate directory name must be a generated ID of the form cand_<12 lowercase hex>"
        )

    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CandidateValidationError(f"missing required file: {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
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
        manifest=manifest,
    )
