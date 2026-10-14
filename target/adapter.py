"""Trusted selected-expert pass-through, not a candidate loader or sandbox.

Bind already-loaded unchanged expert callables. No routing, weighting, tensor
conversion, fallback, model loading, or weight edits occur here. Validation is
untimed: native value checks may synchronize GPU work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


class AdapterError(ValueError):
    """Unsupported layer, input contract, or expert output interface."""


@dataclass(frozen=True, slots=True)
class TrustedExpertAdapter:
    """Suite callable: adapter(layer_index, x, selected_expert_ids).

    `experts` maps layer indices to trusted functions with signature (x, ids).
    Native mode uses the supplied mlx.core module and BF16; mx=None is explicitly
    FP32 NumPy fixture mode, not a claim of native checkpoint execution.
    Checkpoint identity, precision/weight provenance and isolation belong to the
    controller. Callable objects/weights are referenced, not copied or protected.
    """

    experts: Mapping[int, Callable]
    hidden_size: int
    num_experts: int
    mx: object | None = None

    def __post_init__(self) -> None:
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise AdapterError("hidden_size must be a positive integer")
        if (
            type(self.num_experts) is not int
            or not 4 <= self.num_experts <= np.iinfo(np.int32).max
        ):
            raise AdapterError("num_experts must fit int32 and be at least four")
        if not isinstance(self.experts, Mapping) or not self.experts:
            raise AdapterError("experts must be a nonempty trusted callable mapping")
        bound = dict(self.experts)
        if any(
            type(layer) is not int or layer < 0 or not callable(expert)
            for layer, expert in bound.items()
        ):
            raise AdapterError(
                "expert bindings require nonnegative integer layers and callables"
            )
        if self.mx is not None and getattr(self.mx, "__name__", None) != "mlx.core":
            raise AdapterError(
                "mx must be the existing mlx.core module or None for CPU fixtures"
            )
        object.__setattr__(self, "experts", MappingProxyType(bound))

    def _matches(self, value, dtype) -> bool:
        return (
            type(value) is np.ndarray and value.dtype == dtype
            if self.mx is None
            else isinstance(value, self.mx.array) and value.dtype == dtype
        )

    def __call__(self, layer_index: int, x, selected_expert_ids):
        if type(layer_index) is not int or layer_index not in self.experts:
            raise AdapterError("layer_index must identify an explicitly bound layer")
        float_dtype = np.float32 if self.mx is None else self.mx.bfloat16
        index_dtype = np.int32 if self.mx is None else self.mx.int32
        if not self._matches(x, float_dtype) or not self._matches(
            selected_expert_ids, index_dtype
        ):
            raise AdapterError(
                "activations/IDs must use the configured backend and BF16/int32 (FP32/int32 for CPU fixtures)"
            )
        if (
            x.ndim != 3
            or x.shape[0] != 1
            or x.shape[1] <= 0
            or x.shape[2] != self.hidden_size
        ):
            raise AdapterError(
                "activations must have shape [1,T,H] with T>0 and the bound hidden_size"
            )
        if tuple(selected_expert_ids.shape) != (1, x.shape[1], 4):
            raise AdapterError("selected_expert_ids must have shape [1,T,4]")
        if self.mx is None:
            finite = bool(np.all(np.isfinite(x)))
            ids_valid = bool(
                np.all(
                    (selected_expert_ids >= 0)
                    & (selected_expert_ids < self.num_experts)
                )
            )
        else:
            finite = bool(self.mx.all(self.mx.isfinite(x)).item())
            ids_valid = bool(
                self.mx.all(
                    (selected_expert_ids >= 0)
                    & (selected_expert_ids < self.num_experts)
                ).item()
            )
        if not finite or not ids_valid:
            raise AdapterError(
                "activations must be finite and expert IDs must be in range"
            )
        # Exact objects go to the selected binding; output is returned unchanged.
        output = self.experts[layer_index](x, selected_expert_ids)
        if not self._matches(output, float_dtype) or tuple(output.shape) != (
            1,
            x.shape[1],
            4,
            self.hidden_size,
        ):
            raise AdapterError(
                "expert output must be [1,T,4,H] in the original backend/dtype"
            )
        return output
