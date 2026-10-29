"""Authorization boundaries and fail-closed checkpoint verification reuse."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluator.candidate_protocol import file_digest
from evaluator.checkpoint_verification import validate_receipt, verify_checkpoint
from evaluator.schemas import load_contract
from scripts.loop_control import ROOT, controller_sources, load_authorization
from scripts.run_reviewed_candidate import main


class CheckpointReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / "model"
        self.model.mkdir()
        self.weight = self.model / "weight.bin"
        self.weight.write_bytes(b"native-weight")
        self.manifest = {"weight.bin": file_digest(self.weight)}
        self.cache = self.base / "verification.json"

    def verify(self, **kwargs):
        return verify_checkpoint(self.model, kwargs.get("revision", "pinned"),
                                 kwargs.get("manifest", self.manifest), self.cache)

    def test_cold_hash_then_warm_reuse_without_reading_weight_bytes(self):
        receipt, reused = self.verify()
        self.assertFalse(reused)
        real_digest = file_digest
        with patch("evaluator.checkpoint_verification.file_digest", wraps=real_digest) as digest:
            second, reused = self.verify()
        self.assertTrue(reused)
        self.assertEqual(second, receipt)
        self.assertNotIn(self.weight.resolve(), [call.args[0].resolve() for call in digest.call_args_list])
        validate_receipt(receipt, self.model, "pinned", self.manifest)

    def test_runtime_transports_receipt_and_rejects_wrong_pin(self):
        from evaluator.candidate_execution import WorkerRuntime

        receipt, _ = self.verify()
        options = dict(mode="mlx_native", model_path=self.model,
                       model_revision="pinned", model_files=self.manifest,
                       checkpoint_verification=receipt)
        runtime = WorkerRuntime(**options)
        self.assertEqual(runtime.payload()["checkpoint_verification"], receipt)
        options["model_revision"] = "other-pin"
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            WorkerRuntime(**options)

    def test_same_size_corruption_and_restored_mtime_cannot_reuse(self):
        receipt, _ = self.verify()
        before = self.weight.stat()
        self.weight.write_bytes(b"broken-weight")
        os.utime(self.weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            validate_receipt(receipt, self.model, "pinned", self.manifest)
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            self.verify()
        self.assertEqual(json.loads(self.cache.read_text()), receipt)

    def test_replacement_missing_file_revision_and_manifest_invalidate_cache(self):
        receipt, _ = self.verify()
        replacement = self.model / "replacement"
        replacement.write_bytes(self.weight.read_bytes())
        replacement.replace(self.weight)
        with self.assertRaises(ValueError):
            validate_receipt(receipt, self.model, "pinned", self.manifest)
        _, reused = self.verify()
        self.assertFalse(reused)
        _, reused = self.verify(revision="another-pin")
        self.assertFalse(reused)
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            self.verify(manifest={"weight.bin": "0" * 64})
        self.weight.unlink()
        with self.assertRaises(OSError):
            self.verify()

    def test_corrupt_cache_rehashes_and_does_not_claim_reuse(self):
        self.cache.write_text("{")
        receipt, reused = self.verify()
        self.assertFalse(reused)
        self.assertEqual(receipt["byte_verification"], "passed")

    def test_change_during_hash_does_not_create_receipt(self):
        def mutate(path):
            value = file_digest(path)
            if path.resolve() == self.weight.resolve():
                self.weight.write_bytes(b"other-content")
            return value
        with patch("evaluator.checkpoint_verification.file_digest", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.verify()
        self.assertFalse(self.cache.exists())

    def test_symlink_and_escaping_paths_are_rejected(self):
        self.weight.unlink()
        outside = self.base / "outside"
        outside.write_bytes(b"native-weight")
        self.weight.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.verify()
        with self.assertRaises(ValueError):
            self.verify(manifest={"../outside": file_digest(outside)})


class StandingAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup)
        self.base = Path(self.temp.name)
        self.contract = load_contract(ROOT / "kernelmaxxing.yaml")
        self.authorization = self.base / "session.json"
        self.data = {
            "schema": 1, "authorized": True, "repository": str(ROOT),
            "contract_sha256": self.contract.sha256,
            "model_revision": self.contract.model.revision,
            "allowed_operations": ["candidate_correctness"],
            "acceptance_enabled": False,
            "authorization_basis": "Explicit test-owned standing authorization",
            "protected_sources": controller_sources(),
        }
        self.save()

    def cleanup(self):
        for path in self.base.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def save(self):
        self.authorization.write_text(json.dumps(self.data))

    def test_changed_contract_sources_or_revoked_scope_block(self):
        load_authorization(self.authorization, self.contract)
        for key, bad in (("contract_sha256", "0" * 64),
                         ("protected_sources", {}), ("authorized", False),
                         ("allowed_operations", ["publish"]),
                         ("acceptance_enabled", True)):
            with self.subTest(key=key):
                good = self.data[key]
                self.data[key] = bad
                self.save()
                with self.assertRaises(ValueError):
                    load_authorization(self.authorization, self.contract)
                self.data[key] = good
        self.save()

    def test_candidate_local_authorization_is_not_trusted(self):
        with self.assertRaisesRegex(ValueError, "controller-owned"):
            load_authorization(ROOT / "candidates/cand_000000000000/session.json", self.contract)

    def test_loop_entry_point_passes_session_cache_and_refuses_existing_attempt(self):
        from scripts.run_candidate_loop import main as run_loop

        self.data.update(model_path="/test/native-model",
                         checkpoint_cache=str(self.base / "checkpoint.json"))
        self.save()
        attempt = self.base / "new-attempt"
        args = ["--session", str(self.authorization), "--candidate",
                str(ROOT / "candidates/cand_0123456789ab"), "--work-dir", str(attempt)]
        with patch("scripts.run_candidate_loop.run_correctness", return_value=0) as run:
            self.assertEqual(run_loop(args), 0)
            command = run.call_args.args[0]
            self.assertIn("--checkpoint-cache", command)
            self.assertIn(str(self.authorization), command)
            self.assertNotIn("--approved-sha256", command)
            run.reset_mock()
            with self.assertRaises(FileExistsError):
                run_loop(args)
            run.assert_not_called()

    def test_single_freeze_fixture_run_without_per_snapshot_approval(self):
        source = self.base / "candidates/cand_0123456789ab"
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({
            "candidate_id": source.name, "name": "Identity fixture",
            "implementation_type": "mlx", "description": "Test only",
        }))
        for path in (ROOT / "tests/fixtures/reviewed_identity").glob("*.py"):
            (source / path.name).write_bytes(path.read_bytes())
        output = self.base / "attempt"
        from orchestrator.candidate_manager import freeze_candidate
        with patch("orchestrator.run.freeze_candidate", wraps=freeze_candidate) as freeze:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = main(["--candidate", str(source), "--work-dir", str(output),
                             "--fixture", "--session-authorization", str(self.authorization)])
        self.assertEqual(code, 0, stream.getvalue())
        self.assertEqual(freeze.call_count, 1)
        summary = json.loads(stream.getvalue())
        self.assertTrue(all(summary["correctness"].values()))
        self.assertIsNone(summary["verdict"])
        self.assertFalse(summary["performance_measured"])
        record = json.loads(next((output / "artifacts").glob("*-authorization.json")).read_text())
        self.assertFalse(record["human_reviewed_exact_snapshot"])
        self.assertEqual(record["snapshot_sha256"], summary["snapshot_sha256"])
        self.assertEqual(len(list((output / "runs").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
