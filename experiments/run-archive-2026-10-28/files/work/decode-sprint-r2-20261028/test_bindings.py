"""CPU checks for trusted full-model placement/restoration; no GPU imports."""

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "sprint_controller_for_test", Path(__file__).with_name("run_round.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Linear:
    def __call__(self, x):
        return ("native", id(self), x)


class Expert:
    def __call__(self, x, ids):
        return x


class Block:
    def __init__(self):
        self.mlp = SimpleNamespace(experts=Expert())

    def __call__(self, x, mask, cache=None):
        return ("native-block", x, mask, cache)


class TestBindings(unittest.TestCase):
    def model(self):
        return SimpleNamespace(
            lm_head=Linear(), model=SimpleNamespace(layers=[Block()])
        )

    def test_vocabulary_placement_preserves_attention_and_restores_on_exception(self):
        model = self.model()
        attention = Linear()
        x = SimpleNamespace(ndim=3, shape=(1, 1, 2880))
        native = Linear.__call__

        class Adapter:
            kind = "vocabulary"

            def call_projection(self, original, layer, value):
                return ("candidate", value)

        binding = module.Binding(model, Adapter(), None)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with binding.lane(True):
                self.assertEqual(model.lm_head(x), ("candidate", x))
                self.assertEqual(attention(x), native(attention, x))
                self.assertEqual(binding.counts["projection_specialized"], 1)
                raise RuntimeError("sentinel")
        self.assertIs(Linear.__call__, native)
        self.assertEqual(model.lm_head(x), native(model.lm_head, x))

    def test_moe_prefill_fallback_and_decode_callback(self):
        model = self.model()
        layer = model.model.layers[0]
        native = Block.__call__
        expert_native = Expert.__call__
        audit = SimpleNamespace(current=[])

        class Adapter:
            kind = "moe"

            def call_block(self, original, block, x, mask, cache, trace):
                trace("actual-selected-ids")
                return ("candidate-block", x)

        binding = module.Binding(model, Adapter(), None)
        prefill = SimpleNamespace(ndim=3, shape=(1, 511, 2880))
        decode = SimpleNamespace(ndim=3, shape=(1, 1, 2880))
        with binding.lane(True, audit):
            self.assertEqual(
                layer(prefill, "mask", "cache"), native(layer, prefill, "mask", "cache")
            )
            self.assertEqual(layer(decode, None, "cache"), ("candidate-block", decode))
            self.assertEqual(audit.current, [(0, "actual-selected-ids")])
            self.assertEqual(binding.counts["block_specialized"], 1)
            self.assertEqual(binding.counts["block_fallback"], 1)
        self.assertIs(Block.__call__, native)
        self.assertIs(Expert.__call__, expert_native)
        with binding.lane(False):
            self.assertEqual(
                layer(decode, None, "cache"), native(layer, decode, None, "cache")
            )


if __name__ == "__main__":
    unittest.main()
