"""Untimed candidate-output comparison, not candidate execution or acceptance.

The caller owns the unchanged reference, case coverage, and frozen candidate.
Only plain NumPy arrays and MLX arrays are accepted; no candidate code is loaded.
"""

from __future__ import annotations

import numpy as np

from evaluator.schemas import ContractError, CorrectnessPolicy, CorrectnessResult


class InvalidReferenceError(ValueError):
    """The oracle is unusable; record a run error, not a candidate rejection."""


class _UnsupportedOutputError(TypeError):
    """Not an array; distinct from an error while evaluating a real array."""


def _snapshot(output: object) -> tuple[tuple[int, ...], str, np.ndarray | None]:
    """Keep original metadata and copy floating values for untimed CPU checking."""
    if type(output) is np.ndarray:
        shape, dtype = output.shape, f"numpy:{output.dtype.str}"
        values = (
            output.astype(np.float64, copy=True)
            if output.dtype.kind == "f" and output.dtype.itemsize <= 8
            else None
        )
        return shape, dtype, values
    if type(output).__module__ == "mlx.core":
        import mlx.core as mx  # Optional: CPU tests never need MLX installed.

        if isinstance(output, mx.array):
            shape, dtype = tuple(output.shape), f"mlx:{output.dtype}"
            values = None
            if output.dtype in (mx.bfloat16, mx.float16, mx.float32):
                host = output.astype(mx.float32)
                mx.eval(host)
                values = np.array(host, dtype=np.float64, copy=True)
            return shape, dtype, values
    raise _UnsupportedOutputError("output must be a plain NumPy ndarray or MLX array")


def compare_outputs(
    reference: object,
    candidate: object,
    *,
    case_name: str,
    policy: CorrectnessPolicy,
) -> CorrectnessResult:
    """Compare tensors using abs(candidate-reference) <= atol+rtol*abs(reference).

    Shape/backend/original dtype must match; no broadcasting or coercion can
    hide a mismatch. BF16 is widened only after its dtype has been recorded.
    Every element must pass. Empty/nonfloating/nonfinite reference tensors raise
    InvalidReferenceError; malformed candidate outputs produce failed records.
    MLX evaluation failures propagate as execution errors. This is untimed work
    and must never run inside performance measurement intervals.
    """
    if not isinstance(policy, CorrectnessPolicy):
        raise TypeError("policy must be a validated CorrectnessPolicy")
    if type(case_name) is not str or not case_name.strip():
        raise ContractError("case_name must be a nonempty string")
    try:
        shape, dtype, expected = _snapshot(reference)
    except _UnsupportedOutputError as exc:
        raise InvalidReferenceError(str(exc)) from exc
    if expected is None or expected.ndim == 0 or expected.size == 0:
        raise InvalidReferenceError(
            "reference must be a nonempty floating-point tensor"
        )
    if not bool(np.all(np.isfinite(expected))):
        raise InvalidReferenceError("reference contains NaN or infinity")

    try:
        actual_shape, actual_dtype, actual = _snapshot(candidate)
    except _UnsupportedOutputError as exc:
        return CorrectnessResult(case_name, False, False, False, False, False, str(exc))
    shape_match, dtype_match = actual_shape == shape, actual_dtype == dtype
    finite = actual is not None and bool(np.all(np.isfinite(actual)))
    reasons = []
    if not shape_match:
        reasons.append(f"shape mismatch: expected {shape}, got {actual_shape}")
    if not dtype_match:
        reasons.append(f"dtype/backend mismatch: expected {dtype}, got {actual_dtype}")
    if not finite:
        reasons.append("candidate has unsupported dtype or contains NaN/infinity")
    if reasons:
        return CorrectnessResult(
            case_name,
            False,
            shape_match,
            dtype_match,
            finite,
            False,
            "; ".join(reasons),
        )

    # Float64 prevents FP16/BF16 subtraction overflow; still fail closed if
    # float64 fixtures or extreme tolerances overflow the comparison itself.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        difference = np.abs(actual - expected)
        allowed = policy.atol + policy.rtol * np.abs(expected)
    comparison_finite = np.isfinite(difference) & np.isfinite(allowed)
    within_tolerance = comparison_finite & (difference <= allowed)
    violations = int(np.count_nonzero(~within_tolerance))
    passed = violations == 0
    detail = (
        f"max_abs_error={float(np.max(difference)):.9g}; "
        f"violations={violations}/{expected.size}; atol={policy.atol}; rtol={policy.rtol}; "
        f"tolerance_status={policy.tolerance_status.value}"
    )
    if not bool(np.all(comparison_finite)):
        detail += "; comparison arithmetic overflowed; not eligible"
    return CorrectnessResult(case_name, passed, True, True, True, passed, detail)
