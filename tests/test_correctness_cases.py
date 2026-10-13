"""CPU case generation; no checkpoint, MLX import or expert mathematics."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluator.correctness_cases import (
    DEFAULT_LAYERS,
    ExpertCase,
    case_plan,
    make_inputs,
)
from evaluator.schemas import ContractError, load_contract


CONTRACT = load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


class CorrectnessCaseTests(unittest.TestCase):
    def test_plan_covers_every_case_and_layer_once(self):
        plan = case_plan(CONTRACT.correctness)
        self.assertEqual(len(plan), 21)
        self.assertEqual(len(set(plan)), 21)
        for layer in DEFAULT_LAYERS:
            self.assertEqual(
                tuple(case.case_name for case in plan if case.layer_index == layer),
                CONTRACT.correctness.required_cases,
            )

    def test_inputs_match_original_baseline_recipe_for_all_layers(self):
        recipes = (
            ("decode", 1, "uneven", 1),
            ("prefill", 16, "varied", 1),
            ("uneven_expert_load", 17, "uneven", 1),
            ("unused_experts", 17, "unused", 1),
            ("non_divisible_token_counts", 19, "varied", 1),
            ("large_finite_activations", 17, "varied", 64),
            ("held_out_inputs", 23, "varied", 1),
        )
        for layer in DEFAULT_LAYERS:
            for index, (name, count, pattern, scale) in enumerate(recipes):
                with self.subTest(layer=layer, name=name):
                    case = ExpertCase(name, layer)
                    x, ids = make_inputs(case, hidden_size=8, num_experts=64)
                    seed = 4200 + layer * 31 + index
                    expected = (
                        np.random.default_rng(seed)
                        .normal(size=(1, count, 8))
                        .astype(np.float32)
                        * scale
                    )
                    expected_ids = [
                        [0, 1, 2, 3]
                        if pattern == "unused"
                        else [0, 1, 2, 3 if i % 5 else 4]
                        if pattern == "uneven"
                        else [(i + j * 7) % 64 for j in range(4)]
                        for i in range(count)
                    ]
                    self.assertEqual(case.seed, seed)
                    self.assertEqual(case.tokens, count)
                    np.testing.assert_array_equal(x, expected)
                    np.testing.assert_array_equal(
                        ids, np.array([expected_ids], dtype=np.int32)
                    )
                    self.assertEqual(x.dtype, np.float32)
                    self.assertEqual(ids.dtype, np.int32)
                    self.assertTrue(np.isfinite(x).all())
                    self.assertTrue(((ids >= 0) & (ids < 64)).all())

    def test_reordered_policy_does_not_change_seeds(self):
        reordered = replace(
            CONTRACT.correctness,
            required_cases=CONTRACT.correctness.required_cases[::-1],
        )
        original = {case: case.seed for case in case_plan(CONTRACT.correctness)}
        self.assertEqual({case: case.seed for case in case_plan(reordered)}, original)

    def test_calls_return_independent_arrays_without_global_rng_changes(self):
        case = ExpertCase("held_out_inputs", 12)
        before = np.random.get_state()
        x, ids = make_inputs(case, hidden_size=8, num_experts=64)
        original_x, original_ids = x.copy(), ids.copy()
        x[:] = 0
        ids[:] = 0
        again_x, again_ids = make_inputs(case, hidden_size=8, num_experts=64)
        np.testing.assert_array_equal(again_x, original_x)
        np.testing.assert_array_equal(again_ids, original_ids)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_bad_plan_and_case_configuration_is_rejected(self):
        for layers in ((), (0, 0), (-1,), (True,), (1.0,)):
            with self.subTest(layers=layers), self.assertRaises(ContractError):
                case_plan(CONTRACT.correctness, layers)
        with self.assertRaises(TypeError):
            case_plan({})
        unknown = replace(
            CONTRACT.correctness,
            required_cases=(*CONTRACT.correctness.required_cases, "unknown"),
        )
        with self.assertRaises(ContractError):
            case_plan(unknown)
        for name, layer in (("", 0), (None, 0), ("decode", True), ("decode", -1)):
            with self.subTest(name=name, layer=layer), self.assertRaises(ContractError):
                ExpertCase(name, layer)

    def test_bad_dimensions_are_rejected(self):
        case = ExpertCase("decode", 0)
        for hidden, experts in ((0, 64), (True, 64), (8, 4), (8, True), (8, 2**31)):
            with (
                self.subTest(hidden=hidden, experts=experts),
                self.assertRaises(ContractError),
            ):
                make_inputs(case, hidden_size=hidden, num_experts=experts)
        with self.assertRaises(TypeError):
            make_inputs(None, hidden_size=8, num_experts=64)


if __name__ == "__main__":
    unittest.main()
