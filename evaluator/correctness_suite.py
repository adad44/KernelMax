"""Untimed correctness for injected trusted callables; no candidate imports/sandbox.

The controller still owns frozen identity, isolation, timeouts and model binding.
Numerical failure is distinct from incomplete coverage or an execution error.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from evaluator.correctness import InvalidReferenceError, compare_outputs
from evaluator.correctness_cases import (
    DEFAULT_LAYERS,
    ExpertCase,
    case_plan,
    make_inputs,
)
from evaluator.schemas import ContractError, CorrectnessPolicy, CorrectnessResult


@dataclass(frozen=True, slots=True)
class LayerCheck:
    case: ExpertCase
    result: CorrectnessResult | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case, ExpertCase):
            raise TypeError("case must be an ExpertCase")
        if (self.result is None) == (self.error is None):
            raise ContractError("a layer check requires exactly one result or error")
        if self.result is not None:
            if not isinstance(self.result, CorrectnessResult):
                raise TypeError("result must be a CorrectnessResult")
            if self.result.case_name != self.case.case_name:
                raise ContractError("result case does not match its layer check")
        if self.error is not None and (
            type(self.error) is not str or not self.error.strip()
        ):
            raise ContractError("error must be a nonempty string")


@dataclass(frozen=True, slots=True)
class SuiteResult:
    correctness: tuple[CorrectnessResult, ...]
    checks: tuple[LayerCheck, ...]
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """All expected comparisons ran exactly once without execution errors."""
        return bool(self.correctness) and not self.errors

    @property
    def passed(self) -> bool:
        return self.complete and all(result.passed for result in self.correctness)


def _check_detail(check: LayerCheck) -> dict:
    return {
        "case_name": check.case.case_name,
        "layer": check.case.layer_index,
        "seed": check.case.seed,
        "tokens": check.case.tokens,
        "result": check.result.to_dict() if check.result is not None else None,
        "error": check.error,
    }


def summarize_checks(
    checks: Iterable[LayerCheck],
    *,
    policy: CorrectnessPolicy,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> SuiteResult:
    """Retain every check; duplicates, extras and omissions cannot supply a pass.

    Layer details are embedded as JSON in each result.detail so they survive
    ordinary EvaluationReport serialization without a schema change.
    """
    plan = case_plan(policy, layers)
    checks = tuple(checks)
    if any(not isinstance(check, LayerCheck) for check in checks):
        raise TypeError("checks must contain LayerCheck records")
    expected, counts = set(plan), Counter(check.case for check in checks)
    errors = []
    for case in plan:
        label = f"{case.case_name}/layer{case.layer_index}"
        if counts[case] == 0:
            errors.append(f"missing correctness check: {label}")
        elif counts[case] > 1:
            errors.append(f"duplicate correctness check: {label}")
    for check in checks:
        label = f"{check.case.case_name}/layer{check.case.layer_index}"
        if check.case not in expected:
            errors.append(f"unexpected correctness check: {label}")
        if check.error is not None:
            errors.append(f"{label}: {check.error}")

    results = []
    for name in policy.required_cases:
        required = tuple(case for case in plan if case.case_name == name)
        observed = tuple(check for check in checks if check.case.case_name == name)
        complete = all(counts[case] == 1 for case in required) and all(
            check.case in expected and check.result is not None for check in observed
        )
        flags = [
            complete and all(getattr(check.result, flag) for check in observed)
            for flag in ("shape_match", "dtype_match", "finite", "tolerance_passed")
        ]
        diagnostics = {
            "complete": complete,
            "expected_layers": [case.layer_index for case in required],
            "checks": [_check_detail(check) for check in observed],
        }
        # Preserve undeclared case records in the first report result as well as
        # in SuiteResult.checks/errors; never silently lose them on serialization.
        if name == policy.required_cases[0]:
            diagnostics["unexpected_checks"] = [
                _check_detail(check)
                for check in checks
                if check.case.case_name not in policy.required_cases
            ]
        detail = json.dumps(diagnostics, allow_nan=False, sort_keys=True)
        results.append(CorrectnessResult(name, all(flags), *flags, detail))
    return SuiteResult(tuple(results), checks, tuple(errors))


def _input_fingerprint(x, indices, mx) -> tuple:
    """Observe input metadata/values, not a sandbox against change-and-restore."""
    views = (
        (x, indices)
        if mx is None
        else (
            np.array(x.astype(mx.float32)),
            np.array(indices),
        )
    )
    return tuple(
        (
            tuple(original.shape),
            str(original.dtype),
            hashlib.sha256(view.tobytes()).hexdigest(),
        )
        for original, view in zip((x, indices), views)
    )


def _invoke(function, case, hidden_size, num_experts, mx):
    x, indices = make_inputs(
        case, hidden_size=hidden_size, num_experts=num_experts, mx=mx
    )
    before = _input_fingerprint(x, indices, mx)
    output = function(case.layer_index, x, indices)
    # Snapshot/materialize before the other callable runs, preserving dtype/backend.
    if type(output) is np.ndarray:
        output = output.copy()
    elif mx is not None and isinstance(output, mx.array):
        output = mx.array(output)
        mx.eval(output)
    if before != _input_fingerprint(x, indices, mx):
        raise ContractError("callable mutated evaluator-owned inputs")
    return output


def run_correctness_suite(
    reference: Callable,
    test: Callable,
    *,
    policy: CorrectnessPolicy,
    hidden_size: int,
    num_experts: int,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    mx=None,
) -> SuiteResult:
    """Call reference/test(layer_index, x, indices) using fresh identical inputs.

    Only inject trusted fixtures or a separately isolated execution adapter.
    Per-check Exceptions are retained with their stage, then coverage continues.
    BaseExceptions (including KeyboardInterrupt) propagate to the controller.
    No final verdict, timing, report identity or checkpoint verification occurs.
    """
    layers = tuple(layers)
    plan = case_plan(policy, layers)
    if not callable(reference) or not callable(test):
        raise TypeError("reference and test must be callable")
    # Reject bad setup before invoking either callable, not as 21 run failures.
    make_inputs(plan[0], hidden_size=hidden_size, num_experts=num_experts, mx=mx)
    checks = []
    for case in plan:
        stage = "reference"
        try:
            expected = _invoke(reference, case, hidden_size, num_experts, mx)
            backend_matches = (
                type(expected) is np.ndarray and expected.dtype == np.float32
                if mx is None
                else isinstance(expected, mx.array) and expected.dtype == mx.bfloat16
            )
            if not backend_matches or tuple(expected.shape) != (
                1,
                case.tokens,
                4,
                hidden_size,
            ):
                raise InvalidReferenceError(
                    "reference must return [1,T,4,H] in the input backend/dtype"
                )
            compare_outputs(expected, expected, case_name=case.case_name, policy=policy)
            stage = "test"
            actual = _invoke(test, case, hidden_size, num_experts, mx)
            stage = "comparison"
            result = compare_outputs(
                expected, actual, case_name=case.case_name, policy=policy
            )
            checks.append(LayerCheck(case, result=result))
        except Exception as exc:
            checks.append(
                LayerCheck(case, error=f"{stage}: {type(exc).__name__}: {exc}")
            )
    return summarize_checks(checks, policy=policy, layers=layers)
