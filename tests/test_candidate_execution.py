"""Real child-process execution of reviewed temporary fixture snapshots."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from evaluator.candidate_execution import (
    ReviewedCandidateEvaluator,
    WorkerRuntime,
    _compare_transported,
    _read_session,
)
from evaluator.candidate_protocol import VERSION, pack_tensor, unpack_tensor
from evaluator.correctness_cases import case_plan
from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    load_contract,
)
from orchestrator.candidate_manager import create_candidate_id, freeze_candidate
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.run import run_candidate
from orchestrator.state_machine import EventType, ExperimentPhase, RunIdentity


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "kernelmaxxing.yaml")


class CandidateExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.submissions = self.root / "submissions"
        self.submissions.mkdir()
        self.source = self.submissions / create_candidate_id()
        self.source.mkdir()
        (self.source / "manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": self.source.name,
                    "name": "Reviewed test fixture",
                    "description": "Explicit test-owned approval; not a production approval",
                    "implementation_type": "mlx",
                }
            )
        )
        for path in (ROOT / "tests/fixtures/reviewed_identity").glob("*.py"):
            (self.source / path.name).write_bytes(path.read_bytes())
        self.artifacts = self.root / "artifacts"
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for path in self.root.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
            elif not path.is_symlink():
                path.chmod(0o600)
        self.temp.cleanup()

    def freeze(self, implementation=None, wrapper=None):
        if implementation is not None:
            (self.source / "implementation.py").write_text(implementation)
        if wrapper is not None:
            (self.source / "wrapper.py").write_text(wrapper)
        return freeze_candidate(
            self.source,
            frozen_root=self.root / "frozen",
            candidates_root=self.submissions,
        )

    def evaluate(self, frozen, **kwargs):
        identity = RunIdentity(
            "reviewed-test",
            CandidateIdentity(frozen.candidate_id, "mlx", frozen.sha256),
            CONTRACT.sha256,
        )
        options = {
            "approved_sha256s": frozenset({frozen.sha256}),
            "artifact_root": self.artifacts,
            **kwargs,
        }
        evaluator = ReviewedCandidateEvaluator(**options)
        return evaluator, identity

    def run_frozen(self, frozen, **kwargs):
        evaluator, identity = self.evaluate(frozen, **kwargs)
        return evaluator(identity, frozen, CONTRACT)

    def assert_no_verdict_or_timings(self, report):
        self.assertIsNone(report.verdict)
        self.assertEqual(report.performance, ())
        self.assertFalse(CONTRACT.acceptance.enabled)

    def test_identity_executes_frozen_files_in_child_and_parent_grades_all_checks(self):
        frozen = self.freeze()
        # Poison mutable files AFTER approval. Neither helper nor wrapper may be used.
        (self.source / "implementation.py").write_text(
            'raise AssertionError("mutable source executed")'
        )
        (self.source / "wrapper.py").write_text(
            'raise AssertionError("mutable wrapper executed")'
        )
        report = self.run_frozen(frozen)
        self.assertEqual(report.status, RunStatus.COMPLETED, report.errors)
        self.assertEqual(report.errors, ())
        self.assertEqual(len(report.correctness), 7)
        self.assertTrue(all(result.passed for result in report.correctness))
        self.assert_no_verdict_or_timings(report)
        folder = next(self.artifacts.iterdir())
        pids = [
            json.loads((folder / phase / "process.json").read_text())["pid"]
            for phase in ("reference", "candidate")
        ]
        self.assertNotEqual(*pids)
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        for result in report.correctness:
            detail = json.loads(result.detail)
            self.assertEqual(len(detail["checks"]), 3)
            self.assertTrue(detail["candidate_execution_started"])
        restored = EvaluationReport.from_mapping(
            json.loads((folder / "report.json").read_text())
        )
        self.assertEqual(restored, report)
        self.assertTrue((folder / "protected_sources.json").is_file())

    def test_incorrect_candidate_is_completed_with_failed_numerical_checks(self):
        bad = (ROOT / "tests/fixtures/reviewed_incorrect/implementation.py").read_text()
        report = self.run_frozen(self.freeze(bad))
        self.assertEqual(report.status, RunStatus.COMPLETED, report.errors)
        self.assertEqual(report.errors, ())
        self.assertTrue(all(not result.passed for result in report.correctness))
        self.assert_no_verdict_or_timings(report)

    def test_unapproved_hash_is_blocked_before_any_process(self):
        frozen = self.freeze('raise AssertionError("unapproved file imported")')
        report = self.run_frozen(frozen, approved_sha256s=frozenset())
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(
            any(
                "no human/controller review approval" in error
                for error in report.errors
            )
        )
        self.assertFalse(list(self.artifacts.rglob("process.json")))

    def test_parent_never_imports_candidate_and_import_failure_is_retained(self):
        frozen = self.freeze('raise RuntimeError("candidate import sentinel")')
        report = self.run_frozen(frozen)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(
            any("candidate import sentinel" in error for error in report.errors)
        )
        self.assert_no_verdict_or_timings(report)

    def test_crash_preserves_completed_layer_results(self):
        frozen = self.freeze(
            "import os\ndef run(experts,x,ids):\n    if x.shape[1] == 19: os._exit(7)\n    return experts(x,ids)\n"
        )
        report = self.run_frozen(frozen)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(any("status 7" in error for error in report.errors))
        details = [
            check
            for result in report.correctness
            for check in json.loads(result.detail)["checks"]
        ]
        self.assertTrue(
            any(check["result"] and check["result"]["passed"] for check in details)
        )
        self.assertTrue(any(check["error"] for check in details))

    def test_timeout_kills_worker_and_descendant_process_group(self):
        code = 'import subprocess,sys,time\ndef run(experts,x,ids):\n    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n    print("descendant="+str(child.pid),flush=True)\n    time.sleep(60)\n'
        report = self.run_frozen(self.freeze(code), timeout_seconds=2)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(any("deadline" in error for error in report.errors))
        candidate = next(self.artifacts.glob("*/candidate"))
        pid = json.loads((candidate / "process.json").read_text())["pid"]
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        log = (candidate / "worker.log").read_text()
        self.assertIn("descendant=", log)
        child_pid = int(log.split("descendant=")[1].splitlines()[0])
        # Orphans can briefly be zombies until the OS reaps them; neither may run.
        import subprocess

        for _ in range(20):
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.05)
        self.assertTrue(not state or state.startswith("Z"), state)

    def test_clean_environment_and_isolated_scratch_directory(self):
        code = 'import os\ndef run(experts,x,ids):\n    assert "KERNELMAX_TEST_SECRET" not in os.environ\n    assert "candidate" in os.getcwd()\n    assert os.environ["HF_HUB_OFFLINE"] == "1"\n    return experts(x,ids)\n'
        with patch.dict(os.environ, {"KERNELMAX_TEST_SECRET": "must-not-be-inherited"}):
            report = self.run_frozen(self.freeze(code))
        self.assertEqual(report.status, RunStatus.COMPLETED, report.errors)
        self.assertTrue(all(result.passed for result in report.correctness))

    def test_wrong_shape_dtype_and_nonfinite_are_not_hidden_by_transport(self):
        variants = [
            "return experts(x,ids)[:, :, :1, :]",
            "return experts(x,ids).astype('float64')",
            "output=experts(x,ids); output[0,0,0,0]=float('nan'); return output",
            "output=experts(x,ids); output[0,0,0,0]=float('inf'); return output",
        ]
        # Each run has its own fresh candidate/frozen directory.
        for variant in variants:
            with self.subTest(variant=variant):
                source = self.root / "frozen"
                if source.exists():
                    for path in source.rglob("*"):
                        path.chmod(0o700 if path.is_dir() else 0o600)
                    self.source.rename(self.submissions / create_candidate_id())
                    self.source = self.submissions / create_candidate_id()
                    self.source.mkdir()
                    (self.source / "manifest.json").write_text(
                        json.dumps(
                            {
                                "candidate_id": self.source.name,
                                "name": "bad tensor",
                                "description": "reviewed test",
                                "implementation_type": "mlx",
                            }
                        )
                    )
                    (self.source / "wrapper.py").write_text(
                        "from .implementation import run\n"
                    )
                frozen = self.freeze("def run(experts,x,ids):\n    " + variant + "\n")
                report = self.run_frozen(frozen)
                self.assertEqual(report.status, RunStatus.COMPLETED, report.errors)
                self.assertTrue(all(not result.passed for result in report.correctness))

    def test_input_mutation_and_backend_exceptions_are_execution_errors(self):
        frozen = self.freeze(
            "def run(experts,x,ids):\n    output=experts(x,ids)\n    ids[:]=0\n    return output\n"
        )
        report = self.run_frozen(frozen)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(
            any("mutated evaluator-owned inputs" in error for error in report.errors)
        )

    def test_snapshot_tampering_before_launch_is_blocked(self):
        frozen = self.freeze()
        path = frozen.directory / "implementation.py"
        path.chmod(0o600)
        path.write_text('raise AssertionError("tampered snapshot must not execute")')
        report = self.run_frozen(frozen)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertFalse(list(self.artifacts.rglob("process.json")))

    def test_persistent_protected_file_change_is_detected(self):
        protected = self.root / "protected.txt"
        protected.write_text("original")
        code = f'from pathlib import Path\ndef run(experts,x,ids):\n    Path({str(protected)!r}).write_text("changed")\n    return experts(x,ids)\n'
        report = self.run_frozen(self.freeze(code), protected_paths=(protected,))
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(
            any("protected sources changed" in error for error in report.errors)
        )

    def test_log_flood_is_bounded_and_stops_execution(self):
        report = self.run_frozen(
            self.freeze(
                'print("x"*200000,flush=True)\ndef run(experts,x,ids): return experts(x,ids)\n'
            ),
            max_log_bytes=1024,
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(any("log exceeds" in error for error in report.errors))
        self.assertLessEqual(
            next(self.artifacts.glob("*/candidate/worker.log")).stat().st_size, 1024
        )

    def test_failure_after_process_spawn_still_reaps_worker(self):
        import evaluator.candidate_execution as execution

        original_popen = execution.subprocess.Popen
        original_write = execution.write_json
        processes = []

        def spawn(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        def write(path, value):
            if path.name == "process.json":
                raise OSError("process bookkeeping failed")
            return original_write(path, value)

        with (
            patch.object(execution.subprocess, "Popen", side_effect=spawn),
            patch.object(execution, "write_json", side_effect=write),
        ):
            report = self.run_frozen(self.freeze())
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(any("bookkeeping failed" in error for error in report.errors))
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        with self.assertRaises(ProcessLookupError):
            os.kill(processes[0].pid, 0)

    def test_response_size_limit_fails_closed(self):
        report = self.run_frozen(self.freeze(), max_response_bytes=1024)
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertFalse(any(result.passed for result in report.correctness))

    def test_distinct_candidate_callback_and_runner_journal_never_start_benchmark(self):
        frozen = self.freeze()
        evaluator, _ = self.evaluate(frozen)
        journal = ExperimentJournal(self.root / "journal")
        result = run_candidate(
            self.source,
            evaluator=evaluator,
            contract=CONTRACT,
            budget=ExperimentBudget(max_attempts=1),
            journal=journal,
            frozen_root=self.root / "runner-frozen",
            candidates_root=self.submissions,
        )
        self.assertEqual(result.state.phase, ExperimentPhase.COMPLETED, result.error)
        self.assertTrue(all(row.passed for row in result.report.correctness))
        self.assertNotIn(
            EventType.CORRECTNESS_PASSED, [event.type for event in result.state.history]
        )
        signals = [
            row["signal"]
            for row in journal.read_history()
            if row["kind"] == "evaluator_signal"
        ]
        self.assertEqual(signals, ["on_candidate_correctness_passed"])
        self.assert_no_verdict_or_timings(result.report)

    def test_wrong_run_identity_is_rejected_and_callback_failure_retains_results(self):
        frozen = self.freeze()
        evaluator, identity = self.evaluate(frozen)
        wrong = replace(identity, contract_sha256="f" * 64)
        with self.assertRaises(ValueError):
            evaluator(wrong, frozen, CONTRACT)

        def callback(report):
            raise RuntimeError("callback failed")

        report = evaluator(
            identity, frozen, CONTRACT, on_candidate_correctness_passed=callback
        )
        self.assertEqual(report.status, RunStatus.FAILED)
        self.assertTrue(all(result.passed for result in report.correctness))
        self.assertIn("callback failed", report.errors[0])

    def test_numerical_failure_emits_neither_success_callback(self):
        frozen = self.freeze("def run(experts,x,ids): return experts(x,ids)+100\n")
        evaluator, identity = self.evaluate(frozen)
        signals = []
        report = evaluator(
            identity,
            frozen,
            CONTRACT,
            on_candidate_correctness_passed=lambda report: signals.append("candidate"),
            on_diagnostic_passed=lambda report: signals.append("diagnostic"),
        )
        self.assertEqual(report.status, RunStatus.COMPLETED)
        self.assertFalse(any(result.passed for result in report.correctness))
        self.assertEqual(signals, [])

    def test_callback_interruption_retains_completed_comparisons(self):
        frozen = self.freeze()
        evaluator, identity = self.evaluate(frozen)

        def interrupt(report):
            raise KeyboardInterrupt()

        report = evaluator(
            identity, frozen, CONTRACT, on_candidate_correctness_passed=interrupt
        )
        self.assertEqual(report.status, RunStatus.INTERRUPTED)
        self.assertTrue(all(result.passed for result in report.correctness))
        self.assertIn("callback interrupted", report.errors[0])

    def test_runner_rejects_diagnostic_evidence_on_candidate_callback(self):
        class Impostor:
            supports_candidate_correctness = True

            def __call__(
                self,
                run,
                frozen,
                contract,
                *,
                on_diagnostic_passed=None,
                on_candidate_correctness_passed=None,
            ):
                from evaluator.pass_through import TrustedPassThroughEvaluator
                from target.adapter import TrustedExpertAdapter

                def expert(x, ids):
                    return x[:, :, None, :] + ids[:, :, :, None].astype(x.dtype)

                adapter = TrustedExpertAdapter(
                    {i: expert for i in (0, 12, 23)}, hidden_size=8, num_experts=32
                )
                report = TrustedPassThroughEvaluator(
                    lambda layer, x, ids: expert(x, ids), adapter
                )(run, frozen, contract)
                on_candidate_correctness_passed(report)
                return report

        result = run_candidate(
            self.source,
            evaluator=Impostor(),
            contract=CONTRACT,
            budget=ExperimentBudget(max_attempts=1),
            journal=ExperimentJournal(self.root / "journal"),
            frozen_root=self.root / "runner-frozen",
            candidates_root=self.submissions,
        )
        self.assertEqual(result.state.phase, ExperimentPhase.FAILED)
        self.assertIn("diagnostics are not candidate correctness", result.error)

    def test_unavailable_native_model_and_invalid_options_fail_before_execution(self):
        for kwargs in (
            {"mode": "invalid"},
            {"hidden_size": 0},
            {"mode": "mlx_native"},
            {"mode": "mlx_native", "model_path": self.root / "missing"},
            {"model_files": {"../escape": "a" * 64}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WorkerRuntime(**kwargs)
        with self.assertRaises(ValueError):
            ReviewedCandidateEvaluator(frozenset({"not-a-hash"}), self.artifacts)
        with self.assertRaises(ValueError):
            ReviewedCandidateEvaluator(frozenset(), self.artifacts, timeout_seconds=0)


class CandidateTransportTests(unittest.TestCase):
    def test_original_native_dtype_cannot_be_masked_by_float32_storage(self):
        value = pack_tensor(np.ones((1, 1, 4, 8), dtype=np.float32))
        reference = value | {"backend": "mlx", "dtype": "mlx.core.bfloat16"}
        candidate = value | {"backend": "mlx", "dtype": "mlx.core.float32"}
        result = _compare_transported(
            reference, candidate, case_name="decode", policy=CONTRACT.correctness
        )
        self.assertFalse(result.passed or result.dtype_match or result.tolerance_passed)
        self.assertIn("before transport conversion", result.detail)

    def test_safe_tensor_roundtrip_and_metadata_bounds(self):
        array = np.arange(24, dtype=np.float32).reshape(1, 3, 4, 2)
        packed = pack_tensor(array)
        np.testing.assert_array_equal(unpack_tensor(packed), array)
        for changed in (
            {"storage_dtype": "|O"},
            {"shape": [1000000000]},
            {"data": "not-base64"},
            {"backend": "pickle"},
            {"dtype": "<f8"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                unpack_tensor(packed | changed)
        with self.assertRaises(ValueError):
            unpack_tensor(packed, limit=4)

    def test_missing_mismatched_duplicate_and_malformed_responses_fail_closed(self):
        plan = case_plan(CONTRACT.correctness)
        identity = {
            "run_id": "expected",
            "candidate": {},
            "contract_sha256": CONTRACT.sha256,
        }
        request = {
            "identity": identity,
            "phase": "candidate",
            "runtime": {"mode": "numpy_fixture"},
        }
        header = {
            "type": "session",
            "version": VERSION,
            "phase": "candidate",
            "identity": identity,
            "mode": "numpy_fixture",
        }
        output = pack_tensor(np.ones((1, 1, 4, 8), dtype=np.float32))
        row = {
            "type": "result",
            "case_name": "decode",
            "layer_index": 0,
            "seed": 4200,
            "output": output,
            "error": None,
            "input_sha256": "a" * 64,
        }
        variants = (
            [],
            [header | {"identity": {"run_id": "wrong"}}],
            [header, row, row],
            [header, row | {"layer_index": 12}],
            [header, row | {"seed": 0}],
            [header, {"type": "complete", "rows": 21}],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.jsonl"
            for records in variants:
                path.write_text("\n".join(json.dumps(record) for record in records))
                _, errors = _read_session(path, request, plan, 100000)
                self.assertTrue(errors)
            path.write_text(json.dumps(header) + "\n{invalid")
            _, errors = _read_session(path, request, plan, 100000)
            self.assertTrue(errors)


@unittest.skipUnless(
    os.environ.get("KERNELMAX_TEST_MLX") == "1",
    "opt in to tiny MLX transport/loader tests",
)
class MLXCandidateTransportTests(unittest.TestCase):
    def test_original_bf16_and_float32_metadata_with_identical_transport_values(self):
        import mlx.core as mx

        reference = pack_tensor(mx.ones((1, 1, 4, 8), dtype=mx.bfloat16), mx)
        candidate = pack_tensor(mx.ones((1, 1, 4, 8), dtype=mx.float32), mx)
        np.testing.assert_array_equal(
            unpack_tensor(reference), unpack_tensor(candidate)
        )
        result = _compare_transported(
            reference, candidate, case_name="decode", policy=CONTRACT.correctness
        )
        self.assertFalse(result.passed or result.dtype_match)
        self.assertTrue(
            _compare_transported(
                reference, reference, case_name="decode", policy=CONTRACT.correctness
            ).passed
        )

    def test_native_loader_is_called_once_for_the_worker_binding(self):
        # Exercises native setup with a MOCK loader and tiny tensors, not weights.
        import mlx.core as mx
        from types import SimpleNamespace
        from evaluator.candidate_protocol import file_digest
        from evaluator.candidate_worker import _binding

        class Expert:
            gate_proj = up_proj = down_proj = SimpleNamespace(
                mode="mxfp4", bits=4, group_size=32
            )

            def __call__(self, x, ids):
                return x[:, :, None, :] + ids[:, :, :, None].astype(mx.bfloat16)

            def parameters(self):
                return {"tiny": mx.zeros((1,), dtype=mx.bfloat16)}

        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(mlp=SimpleNamespace(experts=Expert()))
                    for _ in range(24)
                ]
            ),
            parameters=lambda: {"tiny": mx.zeros((1,), dtype=mx.bfloat16)},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gpt_oss",
                        "quantization_config": {"quant_method": "mxfp4"},
                        "hidden_size": 8,
                        "num_local_experts": 32,
                        "num_experts_per_tok": 4,
                    }
                )
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"tiny": "fixture.safetensors"}})
            )
            (root / "fixture.safetensors").write_bytes(
                b"not real weights; loader is mocked"
            )
            runtime = {
                "mode": "mlx_native",
                "model_path": str(root),
                "model_files": {
                    name: file_digest(root / name)
                    for name in (
                        "config.json",
                        "model.safetensors.index.json",
                        "fixture.safetensors",
                    )
                },
                "hidden_size": 8,
                "num_experts": 32,
                "expert_factory": None,
            }
            with patch("mlx_lm.load", return_value=(model, None)) as loader:
                module, reference, experts = _binding({"runtime": runtime})
                loader.assert_called_once_with(str(root), lazy=True)
                self.assertIs(module, mx)
                self.assertEqual(set(experts), {0, 12, 23})
                x, ids = (
                    mx.ones((1, 1, 8), dtype=mx.bfloat16),
                    mx.zeros((1, 1, 4), dtype=mx.int32),
                )
                self.assertEqual(reference(12, x, ids).shape, (1, 1, 4, 8))


if __name__ == "__main__":
    unittest.main()
