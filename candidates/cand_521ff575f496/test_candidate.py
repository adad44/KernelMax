"""CPU development tests for permutation wiring, using no model or MLX runtime.

These tests exercise candidate helpers with NumPy and fake expert projections.
They do not approve a frozen submission or establish native model correctness.
"""

from __future__ import annotations

import builtins
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.correctness import compare_outputs
from evaluator.correctness_cases import case_plan, make_inputs
from evaluator.schemas import load_contract


PACKAGE = "candidates." + Path(__file__).resolve().parent.name
candidate = importlib.import_module(PACKAGE + ".implementation")
wrapper = importlib.import_module(PACKAGE + ".wrapper")
CONTRACT = load_contract(ROOT / "kernelmaxxing.yaml")
TOKEN_COUNTS = (16, 17, 19, 23, 512, 2048, 4096)


class NumpyBackend:
    """The small array API used by the permutation helper, with sort counting."""

    def __init__(self):
        self.sort_inputs = []

    def argsort(self, values):
        self.sort_inputs.append(values.copy())
        return np.argsort(values)

    zeros_like = staticmethod(np.zeros_like)
    arange = staticmethod(np.arange)


class FakeProjection:
    """Elementwise fake projection sensitive to values, expert IDs, and columns."""

    def __init__(self, output_size, multiplier):
        self.output_size = output_size
        self.multiplier = np.float32(multiplier)
        self.calls = []

    def values(self, x, ids):
        columns = np.arange(self.output_size) % x.shape[-1]
        return (
            np.take(x, columns, axis=-1) * self.multiplier
            + ids[:, None, None].astype(x.dtype) * np.float32(0.125)
            + np.arange(self.output_size, dtype=x.dtype) * np.float32(0.03125)
        )

    def __call__(self, x, ids, *, sorted_indices):
        output = self.values(x, ids)
        self.calls.append((x, ids, sorted_indices, output))
        return output


class FakeActivation:
    def __init__(self):
        self.calls = []

    @staticmethod
    def values(up, gate):
        return up * (gate + np.float32(0.25))

    def __call__(self, up, gate):
        output = self.values(up, gate)
        self.calls.append((up, gate, output))
        return output


class FakeExperts:
    def __init__(self, hidden_size=8):
        self.up_proj = FakeProjection(12, 0.5)
        self.gate_proj = FakeProjection(12, -0.25)
        self.down_proj = FakeProjection(hidden_size, 0.75)
        self.activation = FakeActivation()


def two_sort_baseline(experts, x, ids):
    """Independent reproduction of the pinned implementation's two-sort path."""
    flat_ids = ids.flatten()
    order = np.argsort(flat_ids)
    inverse = np.argsort(order)
    sorted_x = x.reshape(-1, 1, x.shape[-1])[order // 4]
    sorted_ids = flat_ids[order]
    up = experts.up_proj(sorted_x, sorted_ids, sorted_indices=True)
    gate = experts.gate_proj(sorted_x, sorted_ids, sorted_indices=True)
    output = experts.down_proj(
        experts.activation(up, gate), sorted_ids, sorted_indices=True
    )
    return output[inverse].reshape(1, x.shape[1], 4, x.shape[-1])


def original_order_baseline(experts, x, ids):
    """Evaluate each token/slot independently; no sorting or inverse permutation."""
    output = np.empty((1, x.shape[1], 4, x.shape[-1]), dtype=x.dtype)
    for token in range(x.shape[1]):
        row = x[:, token : token + 1, :]
        for slot in range(4):
            expert_id = ids[:, token, slot]
            up = experts.up_proj.values(row, expert_id)
            gate = experts.gate_proj.values(row, expert_id)
            value = experts.down_proj.values(
                experts.activation.values(up, gate), expert_id
            )
            output[0, token, slot] = value[0, 0]
    return output


def inputs(tokens, pattern="varied"):
    rng = np.random.default_rng(42 + tokens)
    x = rng.normal(size=(1, tokens, 8)).astype(np.float32)
    if pattern == "equal":
        ids = np.full((1, tokens, 4), 7, dtype=np.int32)
    elif pattern == "unused":
        ids = np.broadcast_to(np.array([0, 1, 2, 3], np.int32), (1, tokens, 4)).copy()
    else:
        ids = rng.integers(0, 32, size=(1, tokens, 4), dtype=np.int32)
    return x, ids


class InversePermutationTests(unittest.TestCase):
    def test_inverse_matches_second_sort_exactly_for_every_workload_size(self):
        for tokens in TOKEN_COUNTS:
            for dtype in (np.int32, np.uint32):
                with self.subTest(tokens=tokens, dtype=dtype):
                    order = (
                        np.random.default_rng(tokens)
                        .permutation(tokens * 4)
                        .astype(dtype)
                    )
                    before = order.copy()
                    backend = NumpyBackend()
                    inverse = candidate._inverse_permutation(order, backend)
                    np.testing.assert_array_equal(inverse, np.argsort(order))
                    np.testing.assert_array_equal(inverse[order], np.arange(order.size))
                    np.testing.assert_array_equal(order, before)
                    self.assertEqual(inverse.dtype, order.dtype)
                    self.assertEqual(backend.sort_inputs, [])


class SortedPathTests(unittest.TestCase):
    def assert_exact(self, expected, actual):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_all_sizes_and_expert_patterns_match_two_sort_baseline_exactly(self):
        for tokens in TOKEN_COUNTS:
            for pattern in ("varied", "unused", "equal"):
                with self.subTest(tokens=tokens, pattern=pattern):
                    x, ids = inputs(tokens, pattern)
                    backend = NumpyBackend()
                    output = candidate._run_sorted(FakeExperts(), x, ids, backend)
                    expected = two_sort_baseline(FakeExperts(), x, ids)
                    self.assert_exact(expected, output)
                    self.assertEqual(len(backend.sort_inputs), 1)
                    np.testing.assert_array_equal(backend.sort_inputs[0], ids.flatten())

    def test_original_token_and_expert_slot_order_is_preserved(self):
        for tokens in (16, 17, 19, 23):
            with self.subTest(tokens=tokens):
                x, ids = inputs(tokens)
                output = candidate._run_sorted(FakeExperts(), x, ids, NumpyBackend())
                self.assert_exact(original_order_baseline(FakeExperts(), x, ids), output)

    def test_existing_case_inputs_and_comparison_policy_cover_all_21_checks(self):
        checks = 0
        for case in case_plan(CONTRACT.correctness):
            with self.subTest(case=case.case_name, layer=case.layer_index):
                x, ids = make_inputs(case, hidden_size=8, num_experts=32)
                expected = two_sort_baseline(FakeExperts(), x, ids)
                output = candidate._run_sorted(FakeExperts(), x, ids, NumpyBackend())
                self.assert_exact(expected, output)
                result = compare_outputs(
                    expected,
                    output,
                    case_name=case.case_name,
                    policy=CONTRACT.correctness,
                )
                self.assertTrue(result.passed, result.detail)
                checks += 1
        self.assertEqual(checks, 21)

    def test_native_call_order_sorted_flag_and_objects_are_preserved(self):
        x, ids = inputs(19)
        experts = FakeExperts()
        candidate._run_sorted(experts, x, ids, NumpyBackend())
        (up_call,) = experts.up_proj.calls
        (gate_call,) = experts.gate_proj.calls
        (down_call,) = experts.down_proj.calls
        (activation_call,) = experts.activation.calls
        order = np.argsort(ids.flatten())
        np.testing.assert_array_equal(up_call[0], x.reshape(-1, 1, 8)[order // 4])
        np.testing.assert_array_equal(up_call[1], ids.flatten()[order])
        self.assertIs(up_call[0], gate_call[0])
        self.assertIs(up_call[1], gate_call[1])
        self.assertIs(up_call[1], down_call[1])
        self.assertIs(activation_call[0], up_call[3])
        self.assertIs(activation_call[1], gate_call[3])
        self.assertIs(down_call[0], activation_call[2])
        self.assertTrue(all(call[2] is True for call in (up_call, gate_call, down_call)))

    def test_noncontiguous_inputs_and_ids_are_supported_without_mutation(self):
        x_storage, _ = inputs(38)
        _, ids_storage = inputs(19)
        ids_storage = np.repeat(ids_storage, 2, axis=-1)
        x = x_storage[:, ::2, ::-1]
        ids = ids_storage[..., ::2]
        self.assertFalse(x.flags.c_contiguous or ids.flags.c_contiguous)
        x_before, ids_before = x_storage.copy(), ids_storage.copy()
        expected = two_sort_baseline(FakeExperts(), x, ids)
        output = candidate._run_sorted(FakeExperts(), x, ids, NumpyBackend())
        self.assert_exact(expected, output)
        np.testing.assert_array_equal(x_storage, x_before)
        np.testing.assert_array_equal(ids_storage, ids_before)

    def test_projection_and_activation_errors_propagate_unchanged(self):
        x, ids = inputs(16)
        for name in ("up_proj", "gate_proj", "activation", "down_proj"):
            with self.subTest(component=name):
                experts = FakeExperts()
                error = RuntimeError("distinctive fake component failure")

                def failing(*args, **kwargs):
                    raise error

                setattr(experts, name, failing)
                with self.assertRaises(RuntimeError) as raised:
                    candidate._run_sorted(experts, x, ids, NumpyBackend())
                self.assertIs(raised.exception, error)


class FallbackTests(unittest.TestCase):
    def test_wrapper_exports_the_existing_entry_point(self):
        self.assertIs(wrapper.run, candidate.run)

    def test_fake_inputs_delegate_exact_objects_and_return_output_unchanged(self):
        for tokens in (1, 15, 16, 4096):
            with self.subTest(tokens=tokens):
                x, ids = inputs(tokens)
                output = object()
                calls = []

                def experts(actual_x, actual_ids):
                    calls.append((actual_x, actual_ids))
                    return output

                self.assertIs(wrapper.run(experts, x, ids), output)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][0], x)
                self.assertIs(calls[0][1], ids)

    def test_fallback_does_not_need_or_import_mlx(self):
        original_import = builtins.__import__

        def no_mlx(name, *args, **kwargs):
            if name == "mlx" or name.startswith("mlx."):
                raise AssertionError("CPU fallback must not import MLX")
            return original_import(name, *args, **kwargs)

        x, ids = inputs(16)
        output = object()
        with patch("builtins.__import__", side_effect=no_mlx):
            self.assertIs(wrapper.run(lambda a, b: output, x, ids), output)

    def test_fallback_expert_error_is_not_hidden(self):
        x, ids = inputs(16)
        error = ValueError("distinctive fallback failure")

        def experts(actual_x, actual_ids):
            self.assertIs(actual_x, x)
            self.assertIs(actual_ids, ids)
            raise error

        with self.assertRaises(ValueError) as raised:
            wrapper.run(experts, x, ids)
        self.assertIs(raised.exception, error)


@unittest.skipUnless(
    os.environ.get("KERNELMAX_TEST_MLX") == "1",
    "opt in with KERNELMAX_TEST_MLX=1 in an existing MLX/MLX-LM environment",
)
class TinyNativeTests(unittest.TestCase):
    """Optional synthetic native operators; constructors create tiny fake weights."""

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import SwiGLU
        from mlx_lm.models.switch_layers import SwitchGLU

        cls.mx = mx
        mx.random.seed(42)
        cls.experts = SwitchGLU(64, 96, num_experts=8, activation=SwiGLU(), bias=True)
        for name in ("up_proj", "gate_proj", "down_proj"):
            projection = getattr(cls.experts, name)
            setattr(
                cls.experts,
                name,
                projection.to_quantized(group_size=32, bits=4, mode="mxfp4"),
            )
        cls.experts.set_dtype(mx.bfloat16)
        cls.experts.eval()
        mx.eval(cls.experts.parameters())

    def test_public_native_path_matches_original_exactly_at_all_workload_sizes(self):
        mx = self.mx
        for tokens in TOKEN_COUNTS:
            for id_dtype in (mx.int32, mx.uint32):
                with self.subTest(tokens=tokens, id_dtype=id_dtype):
                    rng = np.random.default_rng(42 + tokens)
                    x = mx.array(
                        rng.normal(size=(1, tokens, 64)).astype(np.float32),
                        dtype=mx.bfloat16,
                    )
                    ids = mx.array(
                        rng.integers(0, 8, size=(1, tokens, 4), dtype=np.int32),
                        dtype=id_dtype,
                    )
                    x_before = np.array(x.astype(mx.float32))
                    ids_before = np.array(ids)
                    expected = self.experts(x, ids)
                    mx.eval(expected)
                    with patch.object(
                        candidate, "_run_sorted", wraps=candidate._run_sorted
                    ) as optimized:
                        output = wrapper.run(self.experts, x, ids)
                    optimized.assert_called_once()
                    mx.eval(output)
                    self.assertIsInstance(output, mx.array)
                    self.assertEqual(output.dtype, mx.bfloat16)
                    self.assertEqual(output.shape, (1, tokens, 4, 64))
                    # Widening BF16 to FP32 is exact and preserves signed zeros.
                    expected_bits = np.array(expected.astype(mx.float32)).view(
                        np.uint32
                    )
                    output_bits = np.array(output.astype(mx.float32)).view(np.uint32)
                    np.testing.assert_array_equal(output_bits, expected_bits)
                    np.testing.assert_array_equal(
                        np.array(x.astype(mx.float32)), x_before
                    )
                    np.testing.assert_array_equal(np.array(ids), ids_before)

    def test_decode_uses_the_supplied_native_experts_without_sorted_replacement(self):
        mx = self.mx
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        ids = mx.array([[[0, 2, 4, 6]]], dtype=mx.int32)
        expected = self.experts(x, ids)
        with patch.object(
            candidate, "_run_sorted", wraps=candidate._run_sorted
        ) as optimized:
            output = wrapper.run(self.experts, x, ids)
        optimized.assert_not_called()
        mx.eval(expected, output)
        np.testing.assert_array_equal(
            np.array(output.astype(mx.float32)).view(np.uint32),
            np.array(expected.astype(mx.float32)).view(np.uint32),
        )


if __name__ == "__main__":
    unittest.main()
