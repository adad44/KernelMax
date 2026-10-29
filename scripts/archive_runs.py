"""Catalog run evidence, keeping bulky artifacts outside the Git repository.

Source bytes are never changed. The manifest records SHA256, size and the
workspace-relative location of every file, including external artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path

CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}
BINARY = {".npz", ".npy", ".safetensors", ".dylib", ".gz"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(f"Path escapes root: {relative}")
    return path


def external_reason(path, size):
    if path.suffix in BINARY:
        return "numerical array, generated weight layout, compiled binary or compressed trace"
    if path.name in {"request.json", "response.jsonl"} and size >= 1024 * 1024:
        return (
            "bulk worker tensor input/output; summary and failure records remain in Git"
        )
    return None


def capture(source, includes, output):
    source = source.resolve()
    if output.exists():
        raise ValueError("Catalog output must be a new directory")
    files, excluded = set(), set()
    for relative in includes:
        directory = inside(source, relative)
        if not directory.is_dir():
            raise ValueError(f"Missing input directory: {relative}")
        for path in directory.rglob("*"):
            relative = str(path.relative_to(source))
            if any(part in CACHES for part in path.relative_to(source).parts):
                if path.is_file():
                    excluded.add(relative)
                continue
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink: {relative}")
            if path.is_file():
                files.add(path)
    output.mkdir(parents=True)
    entries = []
    for path in sorted(files):
        before = path.stat()
        relative = str(path.relative_to(source))
        entry = {
            "path": relative,
            "size": before.st_size,
            "sha256": digest(path),
            "mode": stat.S_IMODE(before.st_mode),
        }
        reason = external_reason(path, before.st_size)
        if reason:
            entry.update(storage="external", artifact_path=relative, reason=reason)
        else:
            stored = "files/" + relative
            destination = output / stored
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            entry.update(storage="file", stored_path=stored)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Input changed during capture: {relative}")
        entries.append(entry)
    manifest = {
        "schema": 2,
        "description": "Historical run catalog; no new evaluation or acceptance",
        "artifact_root": str(source),
        "external_availability": "Local only; external artifacts are not hosted by GitHub",
        "includes": includes,
        "excluded_cache_files": sorted(excluded),
        "files": entries,
        "original_bytes": sum(e["size"] for e in entries),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    retained = [e for e in entries if e["storage"] == "file"]
    print(
        f"Cataloged {len(entries)} files; retained {len(retained)} files / "
        f"{sum(e['size'] for e in retained)} bytes in Git"
    )


def verify(archive, artifact_root=None):
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest["schema"] != 2:
        raise ValueError("Unsupported catalog schema")
    checked = skipped = 0
    for entry in manifest["files"]:
        if entry["storage"] == "file":
            path = inside(archive, entry["stored_path"])
        elif entry["storage"] == "external":
            if artifact_root is None:
                skipped += 1
                continue
            path = inside(artifact_root, entry["artifact_path"])
        else:
            raise ValueError(f"Unknown storage for {entry['path']}")
        if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {entry['path']}")
        checked += 1
    print(f"Verified {checked} files; {skipped} external artifacts not checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("capture")
    build.add_argument("source", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--include", action="append", required=True)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    check.add_argument(
        "--artifact-root",
        type=Path,
        help="Also verify external artifacts under this workspace root",
    )
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.source, args.include, args.output)
    else:
        verify(args.archive, args.artifact_root)


if __name__ == "__main__":
    main()
