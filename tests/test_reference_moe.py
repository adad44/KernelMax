"""Pass-through tests for the native selected-expert reference wrapper."""

import unittest
from types import SimpleNamespace

from target.reference_moe import ReferenceMoE


def fake_model(*experts):
    """Build only the attribute path used by the real model's expert modules."""
    layers = [
        SimpleNamespace(mlp=SimpleNamespace(experts=expert)) for expert in experts
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


class RecordingExpert:
    """Record the exact call and return a caller-provided object unchanged."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


class ReferenceMoETests(unittest.TestCase):
    def test_positional_inputs_reach_selected_layer_in_exact_order(self):
        x = object()
        selected_expert_ids = object()
        output = object()
        selected = RecordingExpert(output)
        other = RecordingExpert(object())
        reference = ReferenceMoE(fake_model(other, selected))

        actual = reference(1, x, selected_expert_ids)

        self.assertEqual(len(selected.calls), 1)
        args, kwargs = selected.calls[0]
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], x)
        self.assertIs(args[1], selected_expert_ids)
        self.assertEqual(kwargs, {})
        self.assertEqual(other.calls, [])
        self.assertIs(actual, output)

    def test_evaluator_keyword_arguments_preserve_input_identity(self):
        x = object()
        selected_expert_ids = object()
        output = object()
        selected = RecordingExpert(output)
        reference = ReferenceMoE(fake_model(selected))

        actual = reference(
            layer_index=0,
            x=x,
            selected_expert_ids=selected_expert_ids,
        )

        args, kwargs = selected.calls[0]
        self.assertIs(args[0], x)
        self.assertIs(args[1], selected_expert_ids)
        self.assertEqual(kwargs, {})
        self.assertIs(actual, output)

    def test_selected_expert_result_is_not_altered_or_reinterpreted(self):
        class OpaqueResult:
            def __bool__(self):
                raise AssertionError("the wrapper must not inspect the result")

            def __iter__(self):
                raise AssertionError("the wrapper must not iterate over the result")

        output = OpaqueResult()
        selected = RecordingExpert(output)
        reference = ReferenceMoE(fake_model(selected))

        actual = reference(0, object(), object())

        self.assertIs(actual, output)


if __name__ == "__main__":
    unittest.main()
