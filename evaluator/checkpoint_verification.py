"""Controller-owned checkpoint verification reuse for authorized local runs.

This is a reviewed-code reliability control, not a hostile-code sandbox. Receipts
must stay controller-owned. A hit reuses a real byte verification only while the
path, revision, complete digest manifest and OS file identities still match.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import time

from evaluator.candidate_protocol import file_digest, read_json


def checkpoint_identity(root, manifest):
    root = Path(root).resolve()
    result = {}
    for name in sorted(manifest):
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid checkpoint path")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"checkpoint path escapes or is a symlink: {name}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"checkpoint must be a regular file: {name}")
        result[name] = [info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns]
    return result


def validate_receipt(receipt, root, revision, manifest):
    expected = {
        "schema": 1,
        "verifier_sha256": file_digest(Path(__file__)),
        "model_path": str(Path(root).resolve()),
        "model_revision": revision,
        "model_files": dict(manifest),
        "file_identity": checkpoint_identity(root, manifest),
    }
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint verification receipt is stale or mismatched")
    if receipt.get("byte_verification") != "passed":
        raise ValueError("checkpoint receipt has no successful byte verification")


def verify_checkpoint(root, revision, manifest, cache_path):
    """Hash on a miss; never refresh a mismatching digest into an approval.

    Return (receipt, reused). A stale/corrupt cache triggers a fresh byte check;
    missing or changed checkpoint files fail closed. Writes are atomic and occur
    only after all digests and before/after identities agree.
    """
    root, cache_path = Path(root).resolve(), Path(cache_path).absolute()
    if not manifest:
        raise ValueError("empty checkpoint manifest")
    if cache_path.resolve().is_relative_to(root):
        raise ValueError("verification cache must be outside the checkpoint")
    try:
        receipt = read_json(cache_path)
        validate_receipt(receipt, root, revision, manifest)
        return receipt, True
    except (OSError, ValueError, TypeError):
        pass
    before = checkpoint_identity(root, manifest)
    for name, expected in sorted(manifest.items()):
        if file_digest(root / name) != expected:
            raise ValueError(f"checkpoint integrity failure: {name}")
    if checkpoint_identity(root, manifest) != before:
        raise ValueError("checkpoint changed during byte verification")
    receipt = {
        "schema": 1,
        "verifier_sha256": file_digest(Path(__file__)),
        "model_path": str(root),
        "model_revision": revision,
        "model_files": dict(manifest),
        "file_identity": before,
        "byte_verification": "passed",
        "verified_unix": time.time(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=cache_path.parent)
    try:
        with os.fdopen(fd, "w") as target:
            json.dump(receipt, target, sort_keys=True, allow_nan=False)
            target.write("\n")
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt, False
