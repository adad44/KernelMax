"""Candidate submission contract tests use isolated temporary directories."""

import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.candidate_manager import (
    CandidateValidationError,
    create_candidate_id,
    validate_candidate,
)


class CandidateManagerTests(unittest.TestCase):
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
        self.write_manifest(implementation_type="cuda")
        with self.assertRaisesRegex(CandidateValidationError, "implementation_type"):
            validate_candidate(self.directory, self.root)

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


if __name__ == "__main__":
    unittest.main()
