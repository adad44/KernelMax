"""Offline checks for the human-owned experiment contract and result formats."""

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from evaluator.schemas import (
    CalibrationStatus,
    CandidateIdentity,
    ContractError,
    CorrectnessResult,
    EnvironmentRecord,
    EvaluationReport,
    ExperimentContract,
    Metric,
    PerformanceResult,
    RunStatus,
    Verdict,
    VerdictResult,
    load_contract,
    validate_report_against_contract,
)


ROOT = Path(__file__).resolve().parents[1]


class ExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(ROOT / "kernelmaxxing.yaml")

    def changed(self, section, field, value):
        data = self.contract.to_dict()
        data[section][field] = value
        return data

    def test_checked_in_contract_pins_the_agreed_target(self):
        self.assertEqual(self.contract.model.repository, "openai/gpt-oss-20b")
        self.assertEqual(
            self.contract.model.revision, "6cee5e81ee83917806bbde320786a8fb61efebee"
        )
        self.assertEqual(self.contract.model.expert_weight_format, "native_mxfp4")
        self.assertEqual(self.contract.hardware.chip, "Apple M4")
        self.assertEqual(self.contract.hardware.memory_gib, 24)
        self.assertEqual(self.contract.workload.prompt_lengths, (512, 2048, 4096))
        self.assertEqual(self.contract.workload.seed, 42)
        self.assertEqual(self.contract.workload.generation_tokens, 128)
        self.assertFalse(self.contract.acceptance.enabled)

    def test_json_round_trip_and_key_order_preserve_hash(self):
        data = json.loads(json.dumps(self.contract.to_dict()))
        reordered = dict(reversed(list(data.items())))
        parsed = ExperimentContract.from_mapping(reordered)
        self.assertEqual(parsed, self.contract)
        self.assertEqual(parsed.sha256, self.contract.sha256)

    def test_modified_contract_has_a_different_hash(self):
        parsed = ExperimentContract.from_mapping(self.changed("workload", "seed", 43))
        self.assertNotEqual(parsed.sha256, self.contract.sha256)

    def test_contract_and_nested_sequences_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.contract.workload.seed = 7
        data = self.contract.to_dict()
        data["workload"]["prompt_lengths"].clear()
        self.assertEqual(self.contract.workload.prompt_lengths, (512, 2048, 4096))

    def test_missing_unknown_and_non_string_fields_are_rejected(self):
        cases = []
        missing = self.contract.to_dict()
        del missing["model"]["revision"]
        cases.append(missing)
        unknown = self.contract.to_dict()
        unknown["benchmark"]["syncronize"] = True
        cases.append(unknown)
        bad_key = self.contract.to_dict()
        bad_key[42] = "unexpected"
        cases.extend([bad_key, None, [], "contract"])
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ContractError):
                ExperimentContract.from_mapping(data)

    def test_invalid_configuration_cannot_silently_change_the_experiment(self):
        cases = [
            ("model", "repository", ""),
            ("model", "repository", "another/model"),
            ("model", "revision", "main"),
            ("model", "revision", "abc123"),
            ("model", "expert_weight_format", "affine_4bit"),
            ("model", "requantize_weights", True),
            ("hardware", "chip", "Apple M5"),
            ("hardware", "memory_gib", 16),
            ("hardware", "model_identifier", " "),
            ("hardware", "backend", "cuda"),
            ("workload", "batch_size", True),
            ("workload", "batch_size", "1"),
            ("workload", "batch_size", 2),
            ("workload", "prompt_lengths", []),
            ("workload", "prompt_lengths", [512, 2048]),
            ("workload", "generation_tokens", 0),
            ("workload", "seed", -1),
            ("workload", "prefill_step_size", 0),
            ("workload", "eos_stopping", True),
            ("workload", "reuse_prompt_cache", True),
            ("correctness", "atol", -0.01),
            ("correctness", "rtol", float("nan")),
            ("correctness", "atol", True),
            ("correctness", "tolerance_status", "measured-ish"),
            ("correctness", "require_finite", False),
            ("correctness", "required_cases", ["decode", "prefill"]),
            ("benchmark", "warmups", 0),
            ("benchmark", "trials_per_implementation", 0),
            ("benchmark", "order", "all_a_then_all_b"),
            ("benchmark", "synchronize_before_and_after", False),
            ("benchmark", "include_model_loading", True),
            ("benchmark", "include_agent_activity", True),
            ("benchmark", "invalidate_on_swap_growth", False),
            ("benchmark", "required_metrics", ["decode_tokens_per_second"]),
            ("benchmark", "statistics", ["mean"]),
            ("acceptance", "minimum_speedup", 1.0),
            ("acceptance", "minimum_speedup", float("inf")),
            ("acceptance", "maximum_peak_memory_ratio", 0.5),
            ("acceptance", "require_improvement_above_noise", False),
            ("agent", "candidate_root", "."),
            ("agent", "optimization_boundary", "whole_model"),
            ("agent", "allowed_implementation_types", ["python"]),
        ]
        for section, field, value in cases:
            with self.subTest(section=section, field=field, value=value):
                with self.assertRaises(ContractError):
                    ExperimentContract.from_mapping(self.changed(section, field, value))

    def test_acceptance_requires_calibrated_thresholds_and_tolerances(self):
        data = self.changed("acceptance", "enabled", True)
        with self.assertRaisesRegex(ContractError, "provisional thresholds"):
            ExperimentContract.from_mapping(data)
        data["acceptance"]["status"] = "calibrated"
        with self.assertRaisesRegex(ContractError, "provisional numerical tolerances"):
            ExperimentContract.from_mapping(data)
        data["correctness"]["tolerance_status"] = "calibrated"
        parsed = ExperimentContract.from_mapping(data)
        self.assertTrue(parsed.acceptance.enabled)
        self.assertEqual(parsed.acceptance.status, CalibrationStatus.CALIBRATED)
        # Schema validation is not evidence that calibration actually occurred.

    def test_schema_version_is_explicit(self):
        data = self.contract.to_dict()
        data["schema_version"] = 2
        with self.assertRaises(ContractError):
            ExperimentContract.from_mapping(data)

    def test_agent_can_only_write_inside_its_assigned_candidate(self):
        agent = self.contract.agent
        candidate_id = "cand_0123456789ab"
        self.assertTrue(
            agent.allows_write(
                f"candidates/{candidate_id}/implementation.py", candidate_id
            )
        )
        self.assertTrue(
            agent.allows_write(
                f"candidates/{candidate_id}/src/kernel.metal", candidate_id
            )
        )
        forbidden = [
            "evaluator/benchmark.py",
            "AGENTS.md",
            "kernelmaxxing.yaml",
            "target/reference_moe.py",
            "reports/verdict.json",
            ".kernelmaxxing/state.json",
            "candidates/cand_aaaaaaaaaaaa/wrapper.py",
            "candidates/candidate-template/wrapper.py",
            f"candidates/{candidate_id}/../../evaluator/verdict.py",
            f"/candidates/{candidate_id}/wrapper.py",
            f"candidates\\{candidate_id}\\wrapper.py",
            "C:/evaluator/verdict.py",
            "",
        ]
        for path in forbidden:
            with self.subTest(path=path):
                self.assertFalse(agent.allows_write(path, candidate_id))
        self.assertFalse(agent.allows_write("candidates/x/wrapper.py", "x"))

    def test_protected_inputs_cannot_be_removed_or_escape_the_repository(self):
        protected = list(self.contract.agent.protected_paths)
        invalid = [
            protected[1:],
            [*protected, "../outside"],
            [*protected, "/tmp"],
            [*protected, "candidates"],
            [*protected, "evaluator"],
            [],
        ]
        for paths in invalid:
            with self.subTest(paths=paths), self.assertRaises(ContractError):
                ExperimentContract.from_mapping(
                    self.changed("agent", "protected_paths", paths)
                )

    def test_yaml_rejects_duplicate_keys_invalid_syntax_and_object_tags(self):
        original = (ROOT / "kernelmaxxing.yaml").read_text()
        invalid = [
            original + "\nschema_version: 1\n",
            original.replace("  batch_size: 1", "  batch_size: 1\n  batch_size: 2"),
            "schema_version: [unterminated",
            "!!python/object/apply:builtins.eval ['1+1']",
            "42: value",
            "",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.yaml"
            for text in invalid:
                path.write_text(text)
                with self.subTest(text=text[:80]), self.assertRaises(ContractError):
                    load_contract(path)


class ResultSchemaTests(unittest.TestCase):
    def test_verdict_round_trip_and_invalid_values(self):
        candidate = CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64)
        for verdict in Verdict:
            result = VerdictResult(
                candidate, "b" * 64, verdict, "Evaluator explanation"
            )
            parsed = VerdictResult.from_mapping(
                json.loads(json.dumps(result.to_dict()))
            )
            self.assertEqual(parsed, result)
        with self.assertRaises(ContractError):
            VerdictResult(candidate, "b" * 64, "FAST_ENOUGH", "Invalid value")
        with self.assertRaises(ContractError):
            VerdictResult(candidate, "short", Verdict.ACCEPTED, "Missing provenance")

    def test_candidate_identity_matches_the_team_format(self):
        for candidate_id, implementation, digest in [
            ("exp-0001", "mlx", "a" * 64),
            ("cand_0123456789ab", "python", "a" * 64),
            ("cand_0123456789ab", "metal", "short"),
        ]:
            with (
                self.subTest(candidate_id=candidate_id),
                self.assertRaises(ContractError),
            ):
                CandidateIdentity(candidate_id, implementation, digest)

    def test_correctness_record_cannot_hide_a_failed_check(self):
        result = CorrectnessResult(
            "decode", True, True, True, True, True, "All checks passed"
        )
        for field in ("shape_match", "dtype_match", "finite", "tolerance_passed"):
            data = result.to_dict()
            data[field] = False
            with self.subTest(field=field), self.assertRaises(ContractError):
                CorrectnessResult.from_mapping(data)
            data["passed"] = False
            self.assertFalse(CorrectnessResult.from_mapping(data).passed)

    def test_timing_records_preserve_samples_and_reject_invalid_pairs(self):
        result = PerformanceResult(
            512,
            Metric.DECODE_THROUGHPUT,
            (20, 21),
            (22, 23),
            True,
            "Synthetic schema fixture",
        )
        self.assertEqual(result.baseline_samples, (20.0, 21.0))
        for field, value in [
            ("candidate_samples", [22]),
            ("baseline_samples", []),
            ("candidate_samples", [0, 23]),
            ("candidate_samples", [float("nan"), 23]),
            ("metric", "fake_speed"),
            ("prompt_length", 3),
        ]:
            data = result.to_dict()
            data[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ContractError),
            ):
                PerformanceResult.from_mapping(data)

    def test_invalid_timing_retains_partial_or_empty_measurements(self):
        for baseline, candidate in [([], []), ([20], []), ([20, 21, 22], [23, 24])]:
            with self.subTest(baseline=baseline, candidate=candidate):
                result = PerformanceResult(
                    512,
                    Metric.DECODE_THROUGHPUT,
                    baseline,
                    candidate,
                    False,
                    "Interrupted during candidate trial",
                )
                self.assertEqual(result.baseline_samples, tuple(baseline))
                self.assertEqual(result.candidate_samples, tuple(candidate))
                self.assertFalse(result.valid)
                self.assertEqual(
                    PerformanceResult.from_mapping(
                        json.loads(json.dumps(result.to_dict()))
                    ),
                    result,
                )
        for samples in ([0], [-1], [float("nan")], [float("inf")], [True]):
            with self.subTest(samples=samples), self.assertRaises(ContractError):
                PerformanceResult(
                    512,
                    Metric.DECODE_THROUGHPUT,
                    samples,
                    (),
                    False,
                    "Failed measurement is an error, not a sample",
                )
        with self.assertRaises(ContractError):
            PerformanceResult(512, Metric.DECODE_THROUGHPUT, (), (), False, " ")

    def test_environment_requires_versions_and_model_hash_provenance(self):
        data = dict(
            chip="Apple M4",
            memory_gib=24,
            model_identifier="Mac16,12",
            macos_version="example",
            python_version="example",
            mlx_version="example",
            mlx_lm_version="example",
            model_files_sha256="c" * 64,
        )
        EnvironmentRecord.from_mapping(data)
        for field, value in [
            ("mlx_version", ""),
            ("model_files_sha256", "unpinned"),
            ("memory_gib", 0),
            ("model_identifier", ""),
        ]:
            invalid = copy.deepcopy(data)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                EnvironmentRecord.from_mapping(invalid)


class EvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(ROOT / "kernelmaxxing.yaml")
        cls.report = EvaluationReport(
            run_id="run-synthetic-fixture",
            candidate=CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64),
            contract_sha256=cls.contract.sha256,
            environment=EnvironmentRecord(
                chip="Apple M4",
                memory_gib=24,
                model_identifier="Mac16,12",
                macos_version="synthetic",
                python_version="synthetic",
                mlx_version="synthetic",
                mlx_lm_version="synthetic",
                model_files_sha256="b" * 64,
            ),
            correctness=tuple(
                CorrectnessResult(case, True, True, True, True, True, "Synthetic pass")
                for case in cls.contract.correctness.required_cases
            ),
            performance=tuple(
                PerformanceResult(
                    length,
                    metric,
                    (20.0,) * 20,
                    (21.0,) * 20,
                    True,
                    "Synthetic timing fixture, not measured evidence",
                )
                for length in cls.contract.workload.prompt_lengths
                for metric in cls.contract.benchmark.required_metrics
            ),
            status=RunStatus.COMPLETED,
            errors=(),
            verdict=None,
        )

    def issues(self, report, contract=None):
        return validate_report_against_contract(report, contract or self.contract)

    def test_complete_report_round_trip_and_completeness_are_not_acceptance(self):
        parsed = EvaluationReport.from_mapping(
            json.loads(json.dumps(self.report.to_dict()))
        )
        self.assertEqual(parsed, self.report)
        self.assertEqual(self.issues(parsed), ())
        self.assertIsNone(parsed.verdict)
        self.assertFalse(self.contract.acceptance.enabled)
        with self.assertRaises(FrozenInstanceError):
            parsed.errors = ("Changed",)

    def test_early_failure_can_be_saved_without_environment_or_verdict(self):
        report = replace(
            self.report,
            environment=None,
            correctness=(),
            performance=(),
            status=RunStatus.FAILED,
            errors=("Build failed before environment capture",),
        )
        parsed = EvaluationReport.from_mapping(json.loads(json.dumps(report.to_dict())))
        self.assertEqual(parsed, report)
        self.assertIn("environment was not captured", self.issues(parsed))
        self.assertIn("run is failed, not completed", self.issues(parsed))

    def test_interrupted_report_preserves_partial_evidence(self):
        timing = replace(
            self.report.performance[0],
            baseline_samples=(20, 21, 22),
            candidate_samples=(23, 24),
            valid=False,
            detail="Interrupted third pair",
        )
        report = replace(
            self.report,
            performance=(timing,),
            status=RunStatus.INTERRUPTED,
            errors=("Candidate process timed out",),
        )
        before = report.to_dict()
        issues = self.issues(report)
        self.assertIn("run is interrupted, not completed", issues)
        self.assertTrue(
            any(issue.startswith("invalid performance") for issue in issues)
        )
        self.assertTrue(
            any(issue.startswith("missing performance") for issue in issues)
        )
        self.assertEqual(report.to_dict(), before)
        self.assertEqual(
            EvaluationReport.from_mapping(json.loads(json.dumps(before))),
            report,
        )

    def test_report_rejects_malformed_fields_but_not_incomplete_evidence(self):
        for field, value in [
            ("run_id", ""),
            ("contract_sha256", "short"),
            ("status", "unknown"),
            ("status", RunStatus.FAILED),
            ("environment", False),
            ("verdict", False),
            ("errors", [""]),
            ("correctness", [None]),
        ]:
            data = self.report.to_dict()
            data[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ContractError),
            ):
                EvaluationReport.from_mapping(data)

    def test_verdict_identity_must_match_its_report(self):
        verdict = VerdictResult(
            self.report.candidate,
            self.report.contract_sha256,
            Verdict.INCONCLUSIVE,
            "Synthetic verdict fixture",
        )
        report = replace(self.report, verdict=verdict)
        self.assertEqual(
            EvaluationReport.from_mapping(json.loads(json.dumps(report.to_dict()))),
            report,
        )
        for candidate in [
            replace(verdict.candidate, candidate_id="cand_aaaaaaaaaaaa"),
            replace(verdict.candidate, sha256="c" * 64),
            replace(verdict.candidate, implementation_type="metal"),
        ]:
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                replace(self.report, verdict=replace(verdict, candidate=candidate))
        with self.assertRaises(ContractError):
            replace(self.report, verdict=replace(verdict, contract_sha256="d" * 64))

    def test_contract_and_canonical_hardware_must_match(self):
        self.assertIn(
            "report contract hash does not match the supplied contract",
            self.issues(replace(self.report, contract_sha256="c" * 64)),
        )
        for field, value in [
            ("chip", "Apple M5"),
            ("memory_gib", 16),
            ("model_identifier", "Mac16,1"),
        ]:
            with self.subTest(field=field):
                environment = replace(self.report.environment, **{field: value})
                self.assertIn(
                    "observed hardware does not match the canonical host",
                    self.issues(replace(self.report, environment=environment)),
                )

    def test_missing_duplicate_and_failed_correctness_are_detected(self):
        first = self.report.correctness[0]
        variants = [
            (
                self.report.correctness[1:],
                f"missing correctness case: {first.case_name}",
            ),
            (
                (*self.report.correctness, first),
                f"duplicate correctness case: {first.case_name}",
            ),
            (
                (
                    replace(first, passed=False, tolerance_passed=False),
                    *self.report.correctness[1:],
                ),
                f"correctness failed: {first.case_name}",
            ),
        ]
        for correctness, expected in variants:
            with self.subTest(expected=expected):
                self.assertIn(
                    expected, self.issues(replace(self.report, correctness=correctness))
                )

    def test_missing_and_duplicate_workload_metrics_are_detected(self):
        first = self.report.performance[0]
        label = f"{first.prompt_length}/{first.metric.value}"
        self.assertIn(
            f"missing performance result: {label}",
            self.issues(replace(self.report, performance=self.report.performance[1:])),
        )
        self.assertIn(
            f"duplicate performance result: {label}",
            self.issues(
                replace(self.report, performance=(*self.report.performance, first))
            ),
        )

    def test_trial_counts_come_from_contract_not_record_minimum(self):
        for count in (2, 19, 21):
            performance = tuple(
                replace(
                    result,
                    baseline_samples=(20,) * count,
                    candidate_samples=(21,) * count,
                )
                for result in self.report.performance
            )
            with self.subTest(count=count):
                issues = self.issues(replace(self.report, performance=performance))
                self.assertEqual(len(issues), len(performance))
                self.assertTrue(
                    all("exactly 20 paired trials" in issue for issue in issues)
                )
        data = self.contract.to_dict()
        data["benchmark"]["trials_per_implementation"] = 2
        contract = ExperimentContract.from_mapping(data)
        performance = tuple(
            replace(result, baseline_samples=(20, 20), candidate_samples=(21, 21))
            for result in self.report.performance
        )
        report = replace(
            self.report, contract_sha256=contract.sha256, performance=performance
        )
        self.assertEqual(self.issues(report, contract), ())

    def test_accepted_claim_is_invalid_while_acceptance_is_disabled(self):
        verdict = VerdictResult(
            self.report.candidate,
            self.report.contract_sha256,
            Verdict.ACCEPTED,
            "Untrusted acceptance claim",
        )
        report = replace(self.report, verdict=verdict)
        self.assertIn(
            "ACCEPTED verdict is forbidden while acceptance is disabled",
            self.issues(report),
        )
        # It remains serializable as an audit artifact, not an official acceptance.
        self.assertEqual(EvaluationReport.from_mapping(report.to_dict()), report)

    def test_failed_invalid_or_incomplete_evidence_blocks_accepted_eligibility(self):
        data = self.contract.to_dict()
        data["acceptance"].update(enabled=True, status="calibrated")
        data["correctness"]["tolerance_status"] = "calibrated"
        contract = ExperimentContract.from_mapping(data)
        report = replace(self.report, contract_sha256=contract.sha256)
        report = replace(
            report,
            verdict=VerdictResult(
                report.candidate,
                contract.sha256,
                Verdict.ACCEPTED,
                "Synthetic claim",
            ),
        )
        first = report.correctness[0]
        for changes in [
            {"status": RunStatus.INTERRUPTED, "errors": ("Timeout",)},
            {"correctness": ()},
            {
                "correctness": (
                    replace(first, passed=False, finite=False),
                    *report.correctness[1:],
                )
            },
            {"performance": ()},
            {
                "performance": (
                    replace(report.performance[0], valid=False),
                    *report.performance[1:],
                )
            },
            {"errors": ("Thermal state changed",)},
        ]:
            with self.subTest(changes=changes):
                self.assertTrue(self.issues(replace(report, **changes), contract))


if __name__ == "__main__":
    unittest.main()
