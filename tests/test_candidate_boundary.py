"""Candidate submission contract tests use isolated temporary directories."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from orchestrator.candidate_manager import (
    CandidateValidationError,
    create_candidate_id,
    freeze_candidate,
    validate_candidate,
    verify_frozen_candidate,
)


class CandidateFixture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "candidates"
        self.root.mkdir()
        self.candidate_id = create_candidate_id()
        self.directory = self.root / self.candidate_id
        self.directory.mkdir()
        self.write_manifest()
        (self.directory / "wrapper.py").write_text("# fake wrapper\n", encoding="utf-8")
        (self.directory / "implementation.py").write_text("# fake implementation\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(self, **changes):
        manifest = {
            "candidate_id": self.candidate_id,
            "name": "Test candidate",
            "implementation_type": "mlx",
            "description": "Temporary fake candidate for validation tests.",
        }
        manifest.update(changes)
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )


class CandidateManagerTests(CandidateFixture):
    def test_generated_candidate_id_has_expected_format(self):
        self.assertRegex(create_candidate_id(), r"^cand_[0-9a-f]{12}$")

    def test_valid_mlx_candidate_returns_manifest(self):
        candidate = validate_candidate(self.directory, self.root)
        self.assertEqual(candidate.candidate_id, self.candidate_id)
        self.assertEqual(candidate.implementation_type, "mlx")
        self.assertEqual(candidate.directory, self.directory.resolve())

    def test_relative_candidate_directory_is_supported(self):
        candidate = validate_candidate(self.candidate_id, self.root)
        self.assertEqual(candidate.candidate_id, self.candidate_id)

    def test_missing_required_manifest_field_is_rejected(self):
        manifest = json.loads((self.directory / "manifest.json").read_text())
        del manifest["description"]
        (self.directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(CandidateValidationError, "description"):
            validate_candidate(self.directory, self.root)

    def test_empty_name_is_rejected(self):
        self.write_manifest(name=" ")
        with self.assertRaisesRegex(CandidateValidationError, "name"):
            validate_candidate(self.directory, self.root)

    def test_unknown_manifest_field_is_rejected(self):
        self.write_manifest(extra="not in the contract")
        with self.assertRaisesRegex(CandidateValidationError, "unsupported fields"):
            validate_candidate(self.directory, self.root)

    def test_directory_and_manifest_ids_must_match(self):
        self.write_manifest(candidate_id="cand_000000000000")
        with self.assertRaisesRegex(CandidateValidationError, "match directory"):
            validate_candidate(self.directory, self.root)

    def test_unsupported_implementation_type_is_rejected(self):
        for implementation_type in ("cuda", "python"):
            with self.subTest(implementation_type=implementation_type):
                self.write_manifest(implementation_type=implementation_type)
                with self.assertRaisesRegex(CandidateValidationError, "implementation_type"):
                    validate_candidate(self.directory, self.root)

    def test_invalid_utf8_manifest_is_a_validation_error(self):
        (self.directory / "manifest.json").write_bytes(b"\xff")
        with self.assertRaisesRegex(CandidateValidationError, "cannot read valid JSON"):
            validate_candidate(self.directory, self.root)

    def test_validated_manifest_is_immutable(self):
        candidate = validate_candidate(self.directory, self.root)
        with self.assertRaises(TypeError):
            candidate.manifest["candidate_id"] = "cand_000000000000"

    def test_metal_candidate_requires_metal_source(self):
        self.write_manifest(implementation_type="metal")
        with self.assertRaisesRegex(CandidateValidationError, "implementation.metal"):
            validate_candidate(self.directory, self.root)
        (self.directory / "implementation.metal").write_text("// fake kernel\n")
        candidate = validate_candidate(self.directory, self.root)
        self.assertEqual(candidate.implementation_type, "metal")

    def test_missing_wrapper_is_rejected(self):
        (self.directory / "wrapper.py").unlink()
        with self.assertRaisesRegex(CandidateValidationError, "wrapper.py"):
            validate_candidate(self.directory, self.root)

    def test_candidate_must_be_direct_child_of_allowed_directory(self):
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(CandidateValidationError, "direct child"):
            validate_candidate(outside, self.root)

    def test_symlinked_candidate_cannot_escape_allowed_directory(self):
        linked_id = create_candidate_id()
        outside = Path(self.temp_dir.name) / linked_id
        outside.mkdir()
        link = self.root / linked_id
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(CandidateValidationError, "direct child"):
            validate_candidate(link, self.root)

    def test_candidate_file_symlink_cannot_escape_candidate(self):
        outside = Path(self.temp_dir.name) / "outside.py"
        outside.write_text("# outside file\n", encoding="utf-8")
        (self.directory / "helper.py").symlink_to(outside)
        with self.assertRaisesRegex(CandidateValidationError, "symlink points outside"):
            validate_candidate(self.directory, self.root)

    def test_symlink_cannot_submit_another_candidate_under_a_new_id(self):
        alias = self.root / create_candidate_id()
        alias.symlink_to(self.directory, target_is_directory=True)
        for submitted in (alias, alias.name):
            with self.subTest(submitted=submitted):
                with self.assertRaisesRegex(CandidateValidationError, "not a symlink"):
                    validate_candidate(submitted, self.root)

    def test_symlink_loop_is_a_validation_error(self):
        loop = self.directory / "loop.py"
        loop.symlink_to(loop)
        with self.assertRaises(CandidateValidationError):
            validate_candidate(self.directory, self.root)


class CandidateSnapshotTests(CandidateFixture):
    def setUp(self):
        super().setUp()
        self.frozen_root = Path(self.temp_dir.name) / "frozen"

    def freeze(self, frozen_root=None):
        return freeze_candidate(
            self.directory,
            frozen_root=self.frozen_root if frozen_root is None else frozen_root,
            candidates_root=self.root,
        )

    def test_snapshot_copies_nested_helpers_and_preserves_source(self):
        helpers = self.directory / "helpers"
        helpers.mkdir()
        (helpers / "ops.py").write_text("# helper\n")
        frozen = self.freeze()
        self.assertEqual(frozen.directory, (self.frozen_root / self.candidate_id).resolve())
        self.assertEqual(set(frozen.file_sha256), {
            "manifest.json", "wrapper.py", "implementation.py", "helpers/ops.py"
        })
        for relative_path, digest in frozen.file_sha256.items():
            original = self.directory / relative_path
            copied = frozen.directory / relative_path
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
            self.assertEqual(digest, hashlib.sha256(copied.read_bytes()).hexdigest())
            self.assertEqual(copied.stat().st_mode & 0o222, 0)
        self.assertEqual(frozen.directory.stat().st_mode & 0o222, 0)
        self.assertNotEqual((self.directory / "wrapper.py").stat().st_mode & 0o222, 0)
        with self.assertRaises(TypeError):
            frozen.file_sha256["wrapper.py"] = "0" * 64
        verify_frozen_candidate(frozen)

    def test_source_changes_do_not_change_snapshot(self):
        frozen = self.freeze()
        wrapper = frozen.directory / "wrapper.py"
        original = wrapper.read_bytes()
        (self.directory / "wrapper.py").write_text("# updated submission\n")
        self.assertEqual(wrapper.read_bytes(), original)
        verify_frozen_candidate(frozen)

    def test_hash_is_stable_and_covers_contents_and_file_names(self):
        first = self.freeze()
        second = self.freeze(Path(self.temp_dir.name) / "second")
        self.assertEqual(first.sha256, second.sha256)
        helper = self.directory / "helper.py"
        helper.write_text("# helper\n")
        added = self.freeze(Path(self.temp_dir.name) / "added")
        self.assertNotEqual(first.sha256, added.sha256)
        helper.rename(self.directory / "renamed.py")
        renamed = self.freeze(Path(self.temp_dir.name) / "renamed")
        self.assertNotEqual(added.sha256, renamed.sha256)
        (self.directory / "wrapper.py").write_text("# changed\n")
        changed = self.freeze(Path(self.temp_dir.name) / "changed")
        self.assertNotEqual(renamed.sha256, changed.sha256)

    def test_snapshot_id_cannot_be_overwritten(self):
        first = self.freeze()
        (self.directory / "wrapper.py").write_text("# replacement\n")
        with self.assertRaisesRegex(CandidateValidationError, "new candidate snapshot"):
            self.freeze()
        verify_frozen_candidate(first)

    def test_snapshot_root_cannot_be_agent_writable(self):
        for root in (self.root, self.directory / "frozen"):
            with self.subTest(root=root):
                with self.assertRaisesRegex(CandidateValidationError, "outside candidates_root"):
                    self.freeze(root)

    def test_snapshot_rejects_even_internal_symlinks(self):
        (self.directory / "alias.py").symlink_to(self.directory / "implementation.py")
        with self.assertRaisesRegex(CandidateValidationError, "must not contain symlinks"):
            self.freeze()
        self.assertFalse((self.frozen_root / self.candidate_id).exists())

    def test_snapshot_rejects_hard_links(self):
        (self.directory / "alias.py").hardlink_to(self.directory / "implementation.py")
        with self.assertRaisesRegex(CandidateValidationError, "without hard links"):
            self.freeze()
        self.assertFalse((self.frozen_root / self.candidate_id).exists())

    def test_snapshot_rejects_special_files_without_blocking(self):
        os.mkfifo(self.directory / "pipe")
        with self.assertRaisesRegex(CandidateValidationError, "regular files"):
            self.freeze()
        self.assertFalse((self.frozen_root / self.candidate_id).exists())

    def test_integrity_check_rejects_modified_deleted_and_added_files(self):
        for change in ("modified", "deleted", "added"):
            with self.subTest(change=change):
                frozen = self.freeze(Path(self.temp_dir.name) / change)
                frozen.directory.chmod(0o755)
                wrapper = frozen.directory / "wrapper.py"
                if change == "modified":
                    wrapper.chmod(0o644)
                    wrapper.write_text("# tampered\n")
                elif change == "deleted":
                    wrapper.unlink()
                else:
                    (frozen.directory / "extra.py").write_text("# extra\n")
                with self.assertRaisesRegex(CandidateValidationError, "contents changed"):
                    verify_frozen_candidate(frozen)

    def test_integrity_check_rejects_replacement_symlinks(self):
        frozen = self.freeze()
        frozen.directory.chmod(0o755)
        wrapper = frozen.directory / "wrapper.py"
        wrapper.unlink()
        wrapper.symlink_to(self.directory / "wrapper.py")
        with self.assertRaisesRegex(CandidateValidationError, "symlinks"):
            verify_frozen_candidate(frozen)

    def test_integrity_check_rejects_a_symlinked_snapshot_directory(self):
        frozen = self.freeze()
        frozen.directory.rename(self.frozen_root / "original")
        frozen.directory.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(CandidateValidationError, "real directory"):
            verify_frozen_candidate(frozen)


if __name__ == "__main__":
    unittest.main()
