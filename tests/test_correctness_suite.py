"""Trusted fake callables and opt-in tiny MLX tests, not model/candidate proof."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluator.correctness_cases import ExpertCase
from evaluator.correctness_suite import (
    LayerCheck,
    run_correctness_suite,
    summarize_checks,
)
from evaluator.schemas import (
    CandidateIdentity,
    ContractError,
    CorrectnessResult,
    EvaluationReport,
    RunStatus,
    load_contract,
    validate_report_against_contract,
)


CONTRACT = load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


def fake_experts(layer, x, indices):
    # Shape-correct fixture sensitive to activations, IDs and layer; no MoE maths.
    return x[:, :, None, :] + indices[:, :, :, None].astype(x.dtype) + np.float32(layer)


class CorrectnessSuiteTests(unittest.TestCase):
    def run_suite(self, test=fake_experts, reference=fake_experts, **kwargs):
        return run_correctness_suite(
            reference,
            test,
            policy=CONTRACT.correctness,
            hidden_size=8,
            num_experts=64,
            **kwargs,
        )

    def passing_checks(self):
        return self.run_suite().checks

    def test_all_cases_pass_with_exact_inputs_and_reference_then_test_order(self):
        calls = []

        def reference(layer, x, ids):
            calls.append(("reference", layer, x.copy(), ids.copy(), x, ids))
            return fake_experts(layer, x, ids)

        def test(layer, x, ids):
            previous = calls[-1]
            self.assertEqual(previous[0:2], ("reference", layer))
            np.testing.assert_array_equal(x, previous[2])
            np.testing.assert_array_equal(ids, previous[3])
            self.assertFalse(np.shares_memory(x, previous[4]))
            self.assertFalse(np.shares_memory(ids, previous[5]))
            calls.append(("test", layer))
            return fake_experts(layer, x, ids)

        suite = self.run_suite(test, reference)
        self.assertTrue(suite.complete and suite.passed)
        self.assertEqual(len(calls), 42)
        self.assertEqual(len(suite.checks), 21)
        self.assertEqual(
            tuple(result.case_name for result in suite.correctness),
            CONTRACT.correctness.required_cases,
        )
        self.assertEqual(suite.errors, ())
        for result in suite.correctness:
            detail = json.loads(result.detail)
            self.assertEqual(detail["expected_layers"], [0, 12, 23])
            self.assertEqual([item["layer"] for item in detail["checks"]], [0, 12, 23])

    def test_one_bad_element_at_middle_layer_is_never_overwritten(self):
        def bad(layer, x, ids):
            output = fake_experts(layer, x, ids)
            if layer == 12 and x.shape[1] == 19:
                output[0, -1, -1, -1] += 100
            return output

        suite = self.run_suite(bad)
        self.assertTrue(suite.complete)
        self.assertFalse(suite.passed)
        result = next(
            r for r in suite.correctness if r.case_name == "non_divisible_token_counts"
        )
        self.assertTrue(result.shape_match and result.dtype_match and result.finite)
        self.assertFalse(result.tolerance_passed)
        self.assertEqual(
            [c["result"]["passed"] for c in json.loads(result.detail)["checks"]],
            [True, False, True],
        )
        self.assertTrue(all(r.passed for r in suite.correctness if r is not result))

    def test_wrong_shape_dtype_nonfinite_and_malformed_outputs_fail_numerically(self):
        transforms = (
            lambda value: value[:, :, :1, :],
            lambda value: value.astype(np.float64),
            lambda value: np.full_like(value, np.nan),
            lambda value: np.full_like(value, np.inf),
            lambda value: None,
        )
        for transform in transforms:
            with self.subTest(transform=transform):
                suite = self.run_suite(
                    lambda layer, x, ids: transform(fake_experts(layer, x, ids))
                )
                self.assertTrue(suite.complete)
                self.assertFalse(suite.passed)
                self.assertEqual(suite.errors, ())
                self.assertTrue(all(not r.passed for r in suite.correctness))

    def test_invalid_reference_never_calls_test_or_blames_candidate(self):
        for reference in (
            lambda layer, x, ids: None,
            lambda layer, x, ids: np.ones((1,), dtype=np.float32),
            lambda layer, x, ids: fake_experts(layer, x, ids).astype(np.float16),
            lambda layer, x, ids: np.full(
                (1, x.shape[1], 4, 8), np.nan, dtype=np.float32
            ),
        ):
            with self.subTest(reference=reference):

                def never(*args):
                    self.fail("test called with an unusable reference")

                suite = self.run_suite(never, reference)
                self.assertFalse(suite.complete or suite.passed)
                self.assertEqual(len(suite.errors), 21)
                self.assertTrue(
                    all(
                        "reference: InvalidReferenceError" in error
                        for error in suite.errors
                    )
                )
                self.assertTrue(all(check.result is None for check in suite.checks))

    def test_exceptions_are_stage_labelled_and_remaining_checks_are_retained(self):
        for stage in ("reference", "test", "comparison"):
            with self.subTest(stage=stage):

                def failing(layer, x, ids):
                    if layer == 12 and x.shape[1] == 19:
                        raise RuntimeError("fixture backend failed")
                    return fake_experts(layer, x, ids)

                if stage == "comparison":
                    from evaluator.correctness import compare_outputs

                    calls = 0

                    def compare(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 24:  # Test comparison for layer12/non-divisible.
                            raise RuntimeError("fixture backend failed")
                        return compare_outputs(*args, **kwargs)

                    with patch(
                        "evaluator.correctness_suite.compare_outputs",
                        side_effect=compare,
                    ):
                        suite = self.run_suite()
                else:
                    suite = (
                        self.run_suite(reference=failing)
                        if stage == "reference"
                        else self.run_suite(failing)
                    )
                self.assertFalse(suite.complete or suite.passed)
                self.assertEqual(len(suite.checks), 21)
                self.assertEqual(len(suite.errors), 1)
                self.assertIn(
                    f"{stage}: RuntimeError: fixture backend failed", suite.errors[0]
                )
                self.assertTrue(suite.checks[-1].result.passed)

    def test_mutations_are_errors_and_do_not_contaminate_later_inputs(self):
        for stage in ("reference", "test"):
            for target in ("x", "ids"):
                with self.subTest(stage=stage, target=target):

                    def mutate(layer, x, ids):
                        output = fake_experts(layer, x, ids)
                        (x if target == "x" else ids)[:] = 0
                        return output

                    suite = (
                        self.run_suite(reference=mutate)
                        if stage == "reference"
                        else self.run_suite(mutate)
                    )
                    self.assertFalse(suite.passed or suite.complete)
                    self.assertTrue(
                        all(
                            "mutated evaluator-owned inputs" in error
                            for error in suite.errors
                        )
                    )
        self.assertTrue(self.run_suite().passed)

    def test_reference_output_is_snapshotted_before_test_reuses_buffer(self):
        shared = None

        def reference(layer, x, ids):
            nonlocal shared
            shared = fake_experts(layer, x, ids)
            return shared

        def test(layer, x, ids):
            shared[:] += 100
            return shared

        suite = self.run_suite(test, reference)
        self.assertTrue(suite.complete)
        self.assertFalse(suite.passed)
        self.assertTrue(all(not result.passed for result in suite.correctness))

    def test_missing_duplicate_and_unexpected_checks_fail_closed(self):
        checks = self.passing_checks()
        variants = (
            (checks[1:], "missing", 20),
            ((*checks, checks[0]), "duplicate", 22),
            (
                (*checks, replace(checks[0], case=ExpertCase("decode", 1))),
                "unexpected",
                22,
            ),
            ((), "missing", 0),
        )
        for observed, message, retained in variants:
            with self.subTest(message=message):
                suite = summarize_checks(observed, policy=CONTRACT.correctness)
                self.assertFalse(suite.complete or suite.passed)
                self.assertEqual(len(suite.correctness), 7)
                self.assertEqual(len(suite.checks), retained)
                self.assertTrue(any(message in error for error in suite.errors))
                self.assertFalse(suite.correctness[0].passed)
        duplicate = summarize_checks((*checks, checks[0]), policy=CONTRACT.correctness)
        self.assertEqual(len(json.loads(duplicate.correctness[0].detail)["checks"]), 4)

    def test_undeclared_case_cannot_be_ignored(self):
        policy = replace(
            CONTRACT.correctness,
            required_cases=("decode", "prefill", "held_out_inputs"),
        )
        suite = summarize_checks(self.passing_checks(), policy=policy)
        self.assertFalse(suite.passed)
        self.assertTrue(any("unexpected" in error for error in suite.errors))
        unexpected = json.loads(suite.correctness[0].detail)["unexpected_checks"]
        self.assertEqual(len(unexpected), 12)
        self.assertEqual(
            {item["case_name"] for item in unexpected},
            {
                "uneven_expert_load",
                "unused_experts",
                "non_divisible_token_counts",
                "large_finite_activations",
            },
        )
        self.assertTrue(all(item["result"]["passed"] for item in unexpected))

    def test_report_roundtrip_retains_layer_failures_and_errors_without_fake_timings(
        self,
    ):
        original = self.run_suite()
        checks = list(original.checks)
        checks[0] = LayerCheck(
            checks[0].case,
            result=CorrectnessResult(
                "decode", False, True, True, True, False, "injected numerical failure"
            ),
        )
        checks[1] = LayerCheck(
            checks[1].case, error="reference: RuntimeError: injected failure"
        )
        suite = summarize_checks(checks, policy=CONTRACT.correctness)
        report = EvaluationReport(
            "trusted-suite-fixture",
            CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64),
            CONTRACT.sha256,
            None,
            suite.correctness,
            (),
            RunStatus.FAILED,
            suite.errors,
            None,
        )
        restored = EvaluationReport.from_mapping(
            json.loads(json.dumps(report.to_dict(), allow_nan=False))
        )
        self.assertEqual(restored, report)
        self.assertIn("injected numerical failure", restored.correctness[0].detail)
        self.assertIn("injected failure", restored.correctness[1].detail)
        issues = validate_report_against_contract(restored, CONTRACT)
        self.assertFalse(
            any(
                "missing correctness case" in issue
                or "duplicate correctness case" in issue
                for issue in issues
            )
        )
        self.assertIn("correctness failed: decode", issues)
        self.assertIn("run contains execution or environment errors", issues)
        self.assertIsNone(restored.verdict)
        self.assertEqual(restored.performance, ())
        self.assertFalse(CONTRACT.acceptance.enabled)

    def test_bad_api_arguments_are_rejected_before_execution(self):
        with self.assertRaises(TypeError):
            self.run_suite(test=None)
        for args in (
            {"hidden_size": 0},
            {"num_experts": 4},
            {"layers": ()},
            {"layers": (0, 0)},
        ):
            with self.subTest(args=args), self.assertRaises(ContractError):
                run_correctness_suite(
                    fake_experts,
                    fake_experts,
                    policy=CONTRACT.correctness,
                    **({"hidden_size": 8, "num_experts": 64} | args),
                )
        case = ExpertCase("decode", 0)
        for kwargs in (
            {},
            {"error": ""},
            {"result": self.passing_checks()[0].result, "error": "error"},
            {"result": self.passing_checks()[1].result},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                LayerCheck(case, **kwargs)
        with self.assertRaises(TypeError):
            summarize_checks([None], policy=CONTRACT.correctness)

    def test_keyboard_interrupt_is_not_swallowed(self):
        def interrupt(*args):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run_suite(interrupt)


@unittest.skipUnless(
    os.environ.get("KERNELMAX_TEST_MLX") == "1", "opt in to tiny MLX smoke tests"
)
class MLXSuiteTests(unittest.TestCase):
    def test_native_bf16_inputs_and_all_case_aggregation(self):
        import mlx.core as mx

        def reference(layer, x, ids):
            self.assertEqual(x.dtype, mx.bfloat16)
            self.assertEqual(ids.dtype, mx.int32)
            return x[:, :, None, :] + ids[:, :, :, None].astype(mx.bfloat16)

        kwargs = dict(policy=CONTRACT.correctness, hidden_size=8, num_experts=64, mx=mx)
        self.assertTrue(run_correctness_suite(reference, reference, **kwargs).passed)

        def bad(layer, x, ids):
            result = reference(layer, x, ids)
            return result + mx.array(1, dtype=mx.bfloat16) if layer == 12 else result

        suite = run_correctness_suite(reference, bad, **kwargs)
        self.assertTrue(suite.complete)
        self.assertFalse(suite.passed)

        def wrong_dtype(layer, x, ids):
            return reference(layer, x, ids).astype(mx.float32)

        self.assertFalse(run_correctness_suite(reference, wrong_dtype, **kwargs).passed)

    def test_mlx_mutation_and_shared_output_snapshot(self):
        import mlx.core as mx

        shared = None

        def reference(layer, x, ids):
            nonlocal shared
            shared = x[:, :, None, :] + ids[:, :, :, None].astype(mx.bfloat16)
            return shared

        kwargs = dict(policy=CONTRACT.correctness, hidden_size=8, num_experts=64, mx=mx)

        def shared_test(layer, x, ids):
            shared[:] = shared + mx.array(100, dtype=mx.bfloat16)
            return shared

        suite = run_correctness_suite(reference, shared_test, **kwargs)
        self.assertTrue(suite.complete)
        self.assertFalse(suite.passed)

        def mutate(layer, x, ids):
            output = reference(layer, x, ids)
            x[:] = 0
            return output

        suite = run_correctness_suite(reference, mutate, **kwargs)
        self.assertFalse(suite.complete or suite.passed)
        self.assertTrue(
            all("mutated evaluator-owned inputs" in error for error in suite.errors)
        )


if __name__ == "__main__":
    unittest.main()
