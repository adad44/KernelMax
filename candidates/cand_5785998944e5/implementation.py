"""Integrate the exact lossless reader only at the vocabulary projection."""

import importlib.util
from pathlib import Path
import sys

_name = __name__ + "_reader"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("reader.py")
)
_reader = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _reader
_spec.loader.exec_module(_reader)


class Adapter:
    kind = "vocabulary"

    def __init__(self, model):
        self.projection = _reader.prepare_projection(model.lm_head.weight)
        self.statistics = self.projection.statistics

    def call_projection(self, original, layer, x):
        if x.ndim == 3 and x.shape[:2] == (1, 1):
            return self.projection(x)
        return original(layer, x)


def prepare(model):
    return Adapter(model)
