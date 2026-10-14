"""Trusted fake experts and opt-in tiny MLX tests; no candidate imports."""

import json
import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from evaluator.correctness_suite import run_correctness_suite
from evaluator.schemas import (
    CandidateIdentity,
    EvaluationReport,
    RunStatus,
    load_contract,
)
from target.adapter import AdapterError, TrustedExpertAdapter


CONTRACT = load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


def fake_expert(x, ids):
    return x[:, :, None, :] + ids[:, :, :, None].astype(x.dtype)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.x = np.ones((1, 3, 8), dtype=np.float32)
        self.ids = np.tile(np.arange(4, dtype=np.int32), (1, 3, 1))

    def adapter(self, expert=fake_expert):
        return TrustedExpertAdapter({0: expert}, hidden_size=8, num_experts=64)

    def test_exact_arguments_and_output_identity_and_layer_selection(self):
        output = fake_expert(self.x, self.ids)
        calls = []

        def expert(x, ids):
            self.assertIs(x, self.x)
            self.assertIs(ids, self.ids)
            calls.append("chosen")
            return output

        def other(*args):
            self.fail("wrong layer selected")

        adapter = TrustedExpertAdapter(
            {0: other, 12: expert}, hidden_size=8, num_experts=64
        )
        self.assertIs(adapter(12, self.x, self.ids), output)
        self.assertEqual(calls, ["chosen"])
        np.testing.assert_array_equal(self.x, np.ones_like(self.x))
        np.testing.assert_array_equal(self.ids, np.tile(np.arange(4), (1, 3, 1)))

    def test_binding_mapping_is_copied_readonly_without_copying_experts(self):
        bindings = {0: fake_expert}
        adapter = TrustedExpertAdapter(bindings, hidden_size=8, num_experts=64)
        bindings.clear()
        self.assertIs(adapter.experts[0], fake_expert)
        self.assertEqual(adapter(0, self.x, self.ids).shape, (1, 3, 4, 8))
        with self.assertRaises(TypeError):
            adapter.experts[0] = None
        with self.assertRaises(FrozenInstanceError):
            adapter.hidden_size = 4

    def test_invalid_layers_do_not_execute_or_fallback(self):
        def never(*args):
            self.fail("unsupported layer executed")

        for layer in (1, -1, True, np.int32(0), "0"):
            with self.subTest(layer=layer), self.assertRaises(AdapterError):
                self.adapter(never)(layer, self.x, self.ids)

    def test_invalid_input_shapes_dtypes_values_and_array_subclasses(self):
        class Subclass(np.ndarray):
            pass

        cases = [
            (self.x.astype(np.float16), self.ids),
            (self.x.astype(np.float64), self.ids),
            (self.x, self.ids.astype(np.int64)),
            (self.x, self.ids.astype(np.float32)),
            (self.x[0], self.ids),
            (self.x[:, :0], self.ids[:, :0]),
            (np.ones((2, 3, 8), dtype=np.float32), self.ids),
            (self.x[:, :, :4], self.ids),
            (self.x, self.ids[:, :1]),
            (self.x, self.ids[:, :, :3]),
            (self.x, self.ids.reshape(3, 4)),
            (self.x.tolist(), self.ids),
            (self.x, None),
            (self.x.view(Subclass), self.ids),
            (self.x, self.ids.view(Subclass)),
        ]
        for value in (np.nan, np.inf, -np.inf):
            x = self.x.copy()
            x[0, 0, 0] = value
            cases.append((x, self.ids))
        for value in (-1, 64):
            ids = self.ids.copy()
            ids[0, 0, 0] = value
            cases.append((self.x, ids))

        def never(*args):
            self.fail("invalid input reached expert")

        for x, ids in cases:
            with (
                self.subTest(x_type=type(x), ids_type=type(ids)),
                self.assertRaises(AdapterError),
            ):
                self.adapter(never)(0, x, ids)

    def test_noncontiguous_inputs_are_not_reformatted_or_copied(self):
        x = np.ones((1, 6, 8), dtype=np.float32)[:, ::2]
        ids = np.tile(np.arange(4, dtype=np.int32), (1, 6, 1))[:, ::2]

        def expert(actual_x, actual_ids):
            self.assertIs(actual_x, x)
            self.assertIs(actual_ids, ids)
            return fake_expert(actual_x, actual_ids)

        self.assertEqual(self.adapter(expert)(0, x, ids).shape, (1, 3, 4, 8))

    def test_invalid_output_interfaces_are_rejected_without_coercion(self):
        output = fake_expert(self.x, self.ids)
        for value in (
            None,
            output.tolist(),
            output.astype(np.float16),
            output[:, :, :1],
            output[0],
        ):
            with self.subTest(value_type=type(value)), self.assertRaises(AdapterError):
                self.adapter(lambda x, ids: value)(0, self.x, self.ids)

    def test_nonfinite_output_is_left_for_the_correctness_comparator(self):
        output = np.full((1, 3, 4, 8), np.nan, dtype=np.float32)
        self.assertIs(self.adapter(lambda x, ids: output)(0, self.x, self.ids), output)

    def test_backend_errors_propagate_unchanged(self):
        error = RuntimeError("fake backend failed")

        def failing(*args):
            raise error

        with self.assertRaises(RuntimeError) as raised:
            self.adapter(failing)(0, self.x, self.ids)
        self.assertIs(raised.exception, error)

    def test_bad_setup_and_path_bindings_are_rejected(self):
        for bindings in (
            {},
            {True: fake_expert},
            {-1: fake_expert},
            {0: Path("candidate.py")},
            {0: "candidate.py"},
            None,
        ):
            with self.subTest(bindings=bindings), self.assertRaises(AdapterError):
                TrustedExpertAdapter(bindings, hidden_size=8, num_experts=64)
        for hidden, experts in ((0, 64), (True, 64), (8, True), (8, 3), (8, 2**31)):
            with (
                self.subTest(hidden=hidden, experts=experts),
                self.assertRaises(AdapterError),
            ):
                TrustedExpertAdapter(
                    {0: fake_expert}, hidden_size=hidden, num_experts=experts
                )
        with self.assertRaises(AdapterError):
            TrustedExpertAdapter(
                {0: fake_expert}, hidden_size=8, num_experts=64, mx="mlx"
            )

    def test_suite_detects_mutation_inside_the_bound_expert(self):
        def mutate(x, ids):
            output = fake_expert(x, ids)
            ids[:] = 0
            return output

        adapter = TrustedExpertAdapter(
            {layer: mutate for layer in (0, 12, 23)}, hidden_size=8, num_experts=64
        )
        suite = run_correctness_suite(
            lambda layer, x, ids: fake_expert(x, ids),
            adapter,
            policy=CONTRACT.correctness,
            hidden_size=8,
            num_experts=64,
        )
        self.assertFalse(suite.complete or suite.passed)
        self.assertEqual(len(suite.errors), 21)
        self.assertTrue(
            all("mutated evaluator-owned inputs" in error for error in suite.errors)
        )

    def test_adapter_connects_all_21_suite_checks_and_report_serialization(self):
        adapter = TrustedExpertAdapter(
            {layer: fake_expert for layer in (0, 12, 23)}, hidden_size=8, num_experts=64
        )
        suite = run_correctness_suite(
            lambda layer, x, ids: fake_expert(x, ids),
            adapter,
            policy=CONTRACT.correctness,
            hidden_size=8,
            num_experts=64,
        )
        self.assertTrue(suite.passed)
        self.assertEqual(len(suite.checks), 21)
        report = EvaluationReport(
            "trusted-adapter-fixture",
            CandidateIdentity("cand_0123456789ab", "mlx", "a" * 64),
            CONTRACT.sha256,
            None,
            suite.correctness,
            (),
            RunStatus.COMPLETED,
            suite.errors,
            None,
        )
        self.assertEqual(
            EvaluationReport.from_mapping(json.loads(json.dumps(report.to_dict()))),
            report,
        )
        self.assertFalse(CONTRACT.acceptance.enabled)

    def test_suite_retains_wrong_values_interface_failures_and_missing_layer(self):
        variants = (
            {
                layer: lambda x, ids: fake_expert(x, ids) + np.float32(100)
                for layer in (0, 12, 23)
            },
            {
                layer: lambda x, ids: fake_expert(x, ids).astype(np.float64)
                for layer in (0, 12, 23)
            },
            {0: fake_expert, 23: fake_expert},
        )
        for index, bindings in enumerate(variants):
            with self.subTest(index=index):
                adapter = TrustedExpertAdapter(bindings, hidden_size=8, num_experts=64)
                suite = run_correctness_suite(
                    lambda layer, x, ids: fake_expert(x, ids),
                    adapter,
                    policy=CONTRACT.correctness,
                    hidden_size=8,
                    num_experts=64,
                )
                self.assertFalse(suite.passed)
                self.assertEqual(suite.complete, index == 0)
                if index == 2:
                    self.assertEqual(len(suite.errors), 7)
                    self.assertTrue(all("layer12" in error for error in suite.errors))
                    self.assertTrue(suite.checks[-1].result.passed)


@unittest.skipUnless(
    os.environ.get("KERNELMAX_TEST_MLX") == "1", "opt in to tiny MLX tests"
)
class MLXAdapterTests(unittest.TestCase):
    def test_bf16_adapter_and_suite_with_exact_tensor_identity(self):
        import mlx.core as mx

        def native_fixture(x, ids):
            return x[:, :, None, :] + ids[:, :, :, None].astype(mx.bfloat16)

        adapter = TrustedExpertAdapter(
            {layer: native_fixture for layer in (0, 12, 23)},
            hidden_size=8,
            num_experts=64,
            mx=mx,
        )
        suite = run_correctness_suite(
            lambda layer, x, ids: native_fixture(x, ids),
            adapter,
            policy=CONTRACT.correctness,
            hidden_size=8,
            num_experts=64,
            mx=mx,
        )
        self.assertTrue(suite.passed)
        x = mx.ones((1, 3, 8), dtype=mx.bfloat16)
        ids = mx.zeros((1, 3, 4), dtype=mx.int32)
        output = native_fixture(x, ids)

        def exact(actual_x, actual_ids):
            self.assertIs(actual_x, x)
            self.assertIs(actual_ids, ids)
            return output

        exact_adapter = TrustedExpertAdapter(
            {0: exact}, hidden_size=8, num_experts=64, mx=mx
        )
        self.assertIs(exact_adapter(0, x, ids), output)

    def test_native_mode_rejects_wrong_backend_dtype_and_id_values(self):
        import mlx.core as mx

        def never(*args):
            self.fail("invalid input reached native fixture")

        adapter = TrustedExpertAdapter({0: never}, hidden_size=8, num_experts=64, mx=mx)
        x = mx.ones((1, 3, 8), dtype=mx.bfloat16)
        ids = mx.zeros((1, 3, 4), dtype=mx.int32)
        cases = (
            (np.ones((1, 3, 8), dtype=np.float32), ids),
            (x.astype(mx.float32), ids),
            (x, ids.astype(mx.int64)),
            (x, mx.full((1, 3, 4), -1, dtype=mx.int32)),
            (x, mx.full((1, 3, 4), 64, dtype=mx.int32)),
            (mx.full((1, 3, 8), float("nan"), dtype=mx.bfloat16), ids),
        )
        for actual_x, actual_ids in cases:
            with self.subTest(dtype=actual_x.dtype), self.assertRaises(AdapterError):
                adapter(0, actual_x, actual_ids)


if __name__ == "__main__":
    unittest.main()
