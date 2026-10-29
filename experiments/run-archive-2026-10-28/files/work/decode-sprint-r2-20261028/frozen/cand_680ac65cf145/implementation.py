"""Exact vocabulary layout with controller-owned single-representation residency.

The controller stores the native/packed backing files and activates only one
representation before a request. This module never reads or writes those files.
"""
import importlib.util
from pathlib import Path
import sys

_name = __name__ + '_reader'
_spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name('reader.py'))
_reader = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _reader
_spec.loader.exec_module(_reader)


class Adapter:
    kind = 'vocabulary'
    requires_weight_residency = True

    def __init__(self, model):
        self.projection = _reader.prepare_projection(model.lm_head.weight)
        self.statistics = dict(self.projection.statistics)
        self.shape = (self.projection.n, self.projection.k)
        self.statistics.update(persistent_native_and_packed_overlap=False,
            native_prefill_fallback='exact lazy reconstruction then original linear',
            preparation_overlap='native plus packed allowed only during untimed conversion/validation')

    def export_layout(self):
        if self.projection is None:
            raise RuntimeError('No active packed representation')
        return {name: getattr(self.projection, name)
                for name in ('payload', 'headers', 'escapes', 'empty_bias')}

    def load_layout(self, arrays):
        if self.projection is not None:
            raise RuntimeError('Unload previous representation before activation')
        self.projection = _reader.PreparedProjection.from_storage(self.shape, arrays, self.statistics)

    def unload_layout(self):
        self.projection = None

    def packed_bytes(self):
        return 0 if self.projection is None else sum(a.nbytes for a in self.export_layout().values())

    def reconstructed_weight(self):
        if self.projection is None:
            raise RuntimeError('Packed representation is not active')
        return self.projection.reconstruct()

    def call_projection(self, original, layer, x):
        if self.projection is None:
            raise RuntimeError('Controller must activate packed representation before candidate execution')
        if x.ndim == 3 and x.shape[:2] == (1, 1):
            return self.projection(x)
        # Native arithmetic for other shapes. The dense weight is a temporary,
        # lazy graph result, never retained by the model or adapter. In canonical
        # cache-only prefill, this unused projection is not evaluated by MLX.
        placeholder = layer.weight
        try:
            layer.weight = self.reconstructed_weight()
            return original(layer, x)
        finally:
            layer.weight = placeholder


def prepare(model):
    return Adapter(model)
