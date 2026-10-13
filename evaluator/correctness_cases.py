"""Deterministic selected-expert fixtures, not full-model timing workloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluator.schemas import ContractError, CorrectnessPolicy


DEFAULT_LAYERS = (0, 12, 23)
# Same counts, patterns, magnitudes and seeds as baseline_validation.expert_checks.
_DEFINITIONS = {
    "decode": (1, "uneven", 1),
    "prefill": (16, "varied", 1),
    "uneven_expert_load": (17, "uneven", 1),
    "unused_experts": (17, "unused", 1),
    "non_divisible_token_counts": (19, "varied", 1),
    "large_finite_activations": (17, "varied", 64),
    "held_out_inputs": (23, "varied", 1),
}


@dataclass(frozen=True, slots=True)
class ExpertCase:
    case_name: str
    layer_index: int

    def __post_init__(self) -> None:
        if type(self.case_name) is not str or self.case_name not in _DEFINITIONS:
            raise ContractError("unknown selected-expert correctness case")
        if type(self.layer_index) is not int or self.layer_index < 0:
            raise ContractError("layer_index must be a nonnegative integer")

    @property
    def seed(self) -> int:
        return 4200 + self.layer_index * 31 + tuple(_DEFINITIONS).index(self.case_name)

    @property
    def tokens(self) -> int:
        return _DEFINITIONS[self.case_name][0]


def case_plan(
    policy: CorrectnessPolicy, layers: tuple[int, ...] = DEFAULT_LAYERS
) -> tuple[ExpertCase, ...]:
    """Declare all required checks before execution; never infer coverage from output."""
    if not isinstance(policy, CorrectnessPolicy):
        raise TypeError("policy must be a validated CorrectnessPolicy")
    layers = tuple(layers)
    if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
        raise ContractError("layers must be nonempty nonnegative integers")
    if len(set(layers)) != len(layers):
        raise ContractError("layers must be unique")
    return tuple(
        ExpertCase(name, layer) for layer in layers for name in policy.required_cases
    )


def make_inputs(
    case: ExpertCase, *, hidden_size: int, num_experts: int, mx=None
) -> tuple[object, object]:
    """Fresh FP32/int32 CPU fixtures or BF16/int32 MLX inputs [1,T,H]/[1,T,4].

    CPU fixtures deliberately do not claim native BF16/model execution. `mx` is
    the caller's existing mlx.core module; this function never loads a model.
    """
    if not isinstance(case, ExpertCase):
        raise TypeError("case must be an ExpertCase")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ContractError("hidden_size must be a positive integer")
    if type(num_experts) is not int or not 5 <= num_experts <= np.iinfo(np.int32).max:
        raise ContractError("num_experts must fit int32 and be at least five")
    count, pattern, magnitude = _DEFINITIONS[case.case_name]
    rng = np.random.default_rng(case.seed)
    x = rng.normal(size=(1, count, hidden_size)).astype(np.float32) * magnitude
    indices = np.empty((1, count, 4), dtype=np.int32)
    for i in range(count):
        indices[0, i] = (
            [0, 1, 2, 3]
            if pattern == "unused"
            else [0, 1, 2, 3 if i % 5 else 4]
            if pattern == "uneven"
            else [(i + j * 7) % num_experts for j in range(4)]
        )
    if mx is None:
        return x, indices
    return mx.array(x, dtype=mx.bfloat16), mx.array(indices, dtype=mx.int32)
