"""Trusted tensor fixtures: no model weights, candidate imports, or timing."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluator.correctness import InvalidReferenceError, compare_outputs
from evaluator.schemas import (
    CandidateIdentity,
    ContractError,
    EvaluationReport,
    RunStatus,
    load_contract,
    validate_report_against_contract,
)


CONTRACT = load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


class CorrectnessTests(unittest.TestCase):
    def compare(self, reference, candidate, **tolerances):
        return compare_outputs(
            reference,
            candidate,
            case_name="decode",
            policy=replace(CONTRACT.correctness, **tolerances),
        )

    def test_identical_outputs_and_inputs_unchanged(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                reference = np.arange(24, dtype=dtype).reshape(1, 3, 4, 2)
                candidate = reference.copy()
                original = reference.copy()
                result = self.compare(reference, candidate)
                self.assertTrue(result.passed)
                self.assertIn("violations=0/24", result.detail)
                np.testing.assert_array_equal(reference, original)
                np.testing.assert_array_equal(candidate, original)

    def test_one_bad_element_fails_the_entire_case(self):
        reference = np.ones((1, 17, 4, 8), dtype=np.float32)
        candidate = reference.copy()
        candidate[0, -1, -1, -1] += 1
        result = self.compare(reference, candidate)
        self.assertFalse(result.passed)
        self.assertTrue(result.shape_match and result.dtype_match and result.finite)
        self.assertFalse(result.tolerance_passed)
        self.assertIn("violations=1/544", result.detail)

    def test_shape_mismatch_never_broadcasts_or_reshapes(self):
        reference = np.ones((1, 2, 4, 3), dtype=np.float32)
        for shape in ((1, 1, 4, 3), (2, 4, 3), (1, 2, 3, 4), (0,)):
            with self.subTest(shape=shape):
                result = self.compare(reference, np.ones(shape, dtype=np.float32))
                self.assertFalse(
                    result.passed or result.shape_match or result.tolerance_passed
                )
                self.assertIn("shape mismatch", result.detail)

    def test_original_dtype_cannot_be_hidden_by_conversion(self):
        reference = np.ones((2,), dtype=np.float32)
        for dtype in (np.float16, np.float64, np.int32, np.bool_, np.complex64, object):
            with self.subTest(dtype=dtype):
                result = self.compare(reference, np.ones((2,), dtype=dtype))
                self.assertTrue(result.shape_match)
                self.assertFalse(
                    result.passed or result.dtype_match or result.tolerance_passed
                )
                self.assertIn("dtype/backend mismatch", result.detail)

    def test_nonfinite_candidates_fail_even_if_metadata_matches(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = self.compare(
                    np.ones((2,), dtype=np.float32),
                    np.array([1, value], dtype=np.float32),
                )
                self.assertTrue(result.shape_match and result.dtype_match)
                self.assertFalse(
                    result.passed or result.finite or result.tolerance_passed
                )

    def test_invalid_reference_is_a_run_error_not_a_candidate_failure(self):
        references = [
            None,
            [1.0],
            np.array([], dtype=np.float32),
            np.array(1.0),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=object),
        ]
        references += [np.array([value]) for value in (np.nan, np.inf, -np.inf)]
        for reference in references:
            with (
                self.subTest(reference=reference),
                self.assertRaises(InvalidReferenceError),
            ):
                self.compare(reference, np.ones((1,), dtype=np.float32))

    def test_malformed_candidate_returns_a_failed_record(self):
        for candidate in (None, [1.0], "1", 1.0):
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    self.compare(np.ones((1,), dtype=np.float32), candidate).passed
                )

    def test_backend_failure_propagates_instead_of_becoming_candidate_rejection(self):
        for exception in (RuntimeError, TypeError, ValueError):
            for which in ("reference", "candidate"):
                snapshot = ((1,), "numpy:<f4", np.ones((1,)))
                outcomes = (
                    [exception("backend failed")]
                    if which == "reference"
                    else [snapshot, exception("backend failed")]
                )
                with (
                    self.subTest(exception=exception, which=which),
                    patch(
                        "evaluator.correctness._snapshot",
                        side_effect=outcomes,
                    ),
                    self.assertRaisesRegex(exception, "backend failed"),
                ):
                    self.compare(None, None)

    def test_absolute_tolerance_inclusive_boundary_at_zero(self):
        reference = np.zeros((1,), dtype=np.float64)
        for error, passes in [(0.125, True), (np.nextafter(0.125, np.inf), False)]:
            with self.subTest(error=error):
                result = self.compare(
                    reference, np.array([error]), atol=0.125, rtol=0.0
                )
                self.assertEqual(result.passed, passes)

    def test_relative_tolerance_uses_reference_not_candidate(self):
        reference = np.array([8.0, -8.0])
        self.assertTrue(
            self.compare(reference, np.array([9.0, -9.0]), atol=0, rtol=0.125).passed
        )
        self.assertFalse(
            self.compare(
                reference, np.array([9.125, -9.125]), atol=0, rtol=0.125
            ).passed
        )
        self.assertFalse(
            self.compare(np.zeros((1,)), np.array([0.001]), atol=0, rtol=0.125).passed
        )

    def test_absolute_and_relative_tolerances_are_added(self):
        reference = np.array([8.0])
        self.assertTrue(
            self.compare(reference, np.array([9.125]), atol=0.125, rtol=0.125).passed
        )
        self.assertFalse(
            self.compare(reference, np.array([9.25]), atol=0.125, rtol=0.125).passed
        )

    def test_zero_tolerance_is_exact_equality(self):
        reference = np.array([1.0, 0.0])
        self.assertTrue(
            self.compare(reference, reference.copy(), atol=0, rtol=0).passed
        )
        candidate = reference.copy()
        candidate[0] = np.nextafter(1.0, np.inf)
        self.assertFalse(self.compare(reference, candidate, atol=0, rtol=0).passed)

    def test_large_float32_values_do_not_overflow_in_subtraction(self):
        reference = np.array([np.finfo(np.float32).max], dtype=np.float32)
        result = self.compare(reference, -reference)
        self.assertFalse(result.passed)
        self.assertNotIn("overflowed", result.detail)

    def test_comparison_overflow_never_passes_inf_less_equal_inf(self):
        result = self.compare(np.array([1e308]), np.array([-1.7e308]), atol=0, rtol=2)
        self.assertTrue(result.finite)
        self.assertFalse(result.passed)
        self.assertIn("arithmetic overflowed", result.detail)

    def test_noncontiguous_arrays_work(self):
        reference = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        self.assertFalse(reference.flags.c_contiguous)
        self.assertTrue(self.compare(reference, reference.copy()).passed)

    def test_bad_api_arguments_are_rejected(self):
        for name in ("", " ", None, True):
            with self.subTest(name=name), self.assertRaises(ContractError):
                compare_outputs(None, None, case_name=name, policy=CONTRACT.correctness)
        with self.assertRaises(TypeError):
            compare_outputs(None, None, case_name="decode", policy={})

    def test_pass_and_failure_survive_report_serialization(self):
        reference = np.ones((2,), dtype=np.float32)
        for candidate, passed in ((reference.copy(), True), (reference + 1, False)):
            result = self.compare(reference, candidate)
            report = EvaluationReport(
                "trusted-fixture",
                CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64),
                CONTRACT.sha256,
                None,
                (result,),
                (),
                RunStatus.COMPLETED,
                (),
                None,
            )
            restored = EvaluationReport.from_mapping(
                json.loads(json.dumps(report.to_dict(), allow_nan=False))
            )
            self.assertEqual(restored, report)
            self.assertEqual(restored.correctness[0].passed, passed)
            self.assertEqual(restored.correctness[0].detail, result.detail)
            if not passed:
                self.assertIn(
                    "correctness failed: decode",
                    validate_report_against_contract(restored, CONTRACT),
                )
            self.assertIsNone(restored.verdict)
        self.assertFalse(CONTRACT.acceptance.enabled)

    def test_invalid_reference_error_survives_failed_report_serialization(self):
        try:
            self.compare(np.array([np.nan]), np.ones((1,)))
        except InvalidReferenceError as exc:
            report = EvaluationReport(
                "invalid-oracle-fixture",
                CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64),
                CONTRACT.sha256,
                None,
                (),
                (),
                RunStatus.FAILED,
                (str(exc),),
                None,
            )
        else:
            self.fail("invalid reference was treated as a candidate result")
        restored = EvaluationReport.from_mapping(
            json.loads(json.dumps(report.to_dict(), allow_nan=False))
        )
        self.assertEqual(restored.status, RunStatus.FAILED)
        self.assertEqual(restored.errors, ("reference contains NaN or infinity",))
        self.assertEqual(restored.correctness, ())


@unittest.skipUnless(
    os.environ.get("KERNELMAX_TEST_MLX") == "1", "opt in to tiny MLX smoke tests"
)
class MLXCorrectnessTests(unittest.TestCase):
    def test_bf16_pass_wrong_values_and_original_dtype(self):
        import mlx.core as mx

        reference = mx.array([1.0, -2.0], dtype=mx.bfloat16).reshape(1, 1, 1, 2)
        for candidate, passes in [
            (reference + mx.array(0, dtype=mx.bfloat16), True),
            (reference + mx.array(1, dtype=mx.bfloat16), False),
            (reference.astype(mx.float32), False),
        ]:
            result = compare_outputs(
                reference, candidate, case_name="decode", policy=CONTRACT.correctness
            )
            self.assertEqual(result.passed, passes)
        self.assertEqual(reference.dtype, mx.bfloat16)

    def test_mlx_nonfinite_reference_and_candidate(self):
        import mlx.core as mx

        reference = mx.array([1.0], dtype=mx.bfloat16)
        for value in (float("nan"), float("inf"), -float("inf")):
            candidate = mx.array([value], dtype=mx.bfloat16)
            self.assertFalse(
                compare_outputs(
                    reference,
                    candidate,
                    case_name="decode",
                    policy=CONTRACT.correctness,
                ).finite
            )
            with self.assertRaises(InvalidReferenceError):
                compare_outputs(
                    candidate,
                    reference,
                    case_name="decode",
                    policy=CONTRACT.correctness,
                )


if __name__ == "__main__":
    unittest.main()
