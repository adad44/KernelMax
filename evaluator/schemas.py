"""Immutable experiment contracts and evaluator result records.

This module validates data; it never imports MLX, loads a model, executes a
candidate, times GPU work, or decides a verdict. PyYAML is only needed when
reading a YAML contract. All record construction and JSON handling use stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Self, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """Configuration or report data violates the experiment contract."""


class Verdict(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CalibrationStatus(str, Enum):
    PROVISIONAL = "provisional"
    CALIBRATED = "calibrated"


class Metric(str, Enum):
    DECODE_THROUGHPUT = "decode_tokens_per_second"
    TTFT = "time_to_first_token_ms"
    PREFILL_THROUGHPUT = "prefill_tokens_per_second"
    INTER_TOKEN_LATENCY = "inter_token_latency_ms"
    PEAK_MEMORY = "peak_memory_bytes"
    WALL_TIME = "full_model_wall_time_ms"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _decode(value: Any, annotation: Any, name: str) -> Any:
    """Decode only the simple field types used by the records below."""
    if get_origin(annotation) is tuple:
        _require(isinstance(value, (list, tuple)), f"{name} must be a sequence")
        item_type, _ = get_args(annotation)
        return tuple(
            _decode(item, item_type, f"{name}[{i}]") for i, item in enumerate(value)
        )
    if is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        return annotation.from_mapping(value)
    if issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"{name} must be one of {[item.value for item in annotation]}"
            ) from exc
    if annotation is float:
        _require(type(value) in (int, float), f"{name} must be a number")
        _require(math.isfinite(value), f"{name} must be finite")
        return float(value)
    _require(type(value) is annotation, f"{name} must be {annotation.__name__}")
    if annotation is str:
        _require(bool(value.strip()), f"{name} must not be empty")
    return value


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


class _Record:
    __slots__ = ()

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for field in fields(self):
            value = _decode(getattr(self, field.name), hints[field.name], field.name)
            object.__setattr__(self, field.name, value)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        _require(isinstance(data, Mapping), f"{cls.__name__} must be an object")
        _require(
            all(isinstance(key, str) for key in data), "field names must be strings"
        )
        expected = {field.name for field in fields(cls)}
        missing, unknown = expected - data.keys(), data.keys() - expected
        _require(not missing, f"{cls.__name__}: missing fields {sorted(missing)}")
        _require(not unknown, f"{cls.__name__}: unknown fields {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class ModelSpec(_Record):
    repository: str
    revision: str
    architecture: str
    expert_weight_format: str
    activation_dtype: str
    kv_cache_dtype: str
    requantize_weights: bool

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            self.repository == "openai/gpt-oss-20b",
            "Week 1 requires openai/gpt-oss-20b",
        )
        _require(
            bool(re.fullmatch(r"[0-9a-f]{40}", self.revision)),
            "revision must be a full commit SHA",
        )
        _require(self.architecture == "gpt_oss", "architecture must be gpt_oss")
        _require(
            self.expert_weight_format == "native_mxfp4",
            "preserve native MXFP4 expert weights",
        )
        _require(
            self.activation_dtype == self.kv_cache_dtype == "bfloat16",
            "Week 1 requires BF16 activations and KV cache",
        )
        _require(not self.requantize_weights, "checkpoint requantization is forbidden")


@dataclass(frozen=True, slots=True)
class HardwareSpec(_Record):
    chip: str
    memory_gib: int
    model_identifier: str
    backend: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            self.chip == "Apple M4" and self.memory_gib == 24,
            "canonical host must be a 24 GiB Apple M4",
        )
        _require(self.backend == "mlx_metal", "backend must be mlx_metal")


@dataclass(frozen=True, slots=True)
class WorkloadSpec(_Record):
    batch_size: int
    prompt_lengths: tuple[int, ...]
    generation_tokens: int
    seed: int
    input_source: str
    sampling: str
    prefill_step_size: int
    eos_stopping: bool
    reuse_prompt_cache: bool

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(self.batch_size == 1, "Week 1 batch_size must be 1")
        _require(
            self.prompt_lengths == (512, 2048, 4096),
            "prompt_lengths must be [512, 2048, 4096]",
        )
        _require(self.generation_tokens == 128, "generation_tokens must be 128")
        _require(
            self.seed >= 0 and self.prefill_step_size > 0,
            "seed must be nonnegative and prefill_step_size positive",
        )
        _require(
            self.input_source == "fixed_token_ids" and self.sampling == "greedy",
            "use fixed token IDs and greedy sampling",
        )
        _require(
            not self.eos_stopping and not self.reuse_prompt_cache,
            "timing requires fixed output length and fresh request caches",
        )


@dataclass(frozen=True, slots=True)
class CorrectnessPolicy(_Record):
    reference: str
    tolerance_status: CalibrationStatus
    atol: float
    rtol: float
    require_finite: bool
    require_shape_match: bool
    require_dtype_match: bool
    required_cases: tuple[str, ...]

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            self.reference == "unmodified_mlx_lm_same_checkpoint",
            "reference must use unchanged MLX-LM and the same checkpoint",
        )
        _require(self.atol >= 0 and self.rtol >= 0, "tolerances must be nonnegative")
        _require(
            self.require_finite
            and self.require_shape_match
            and self.require_dtype_match,
            "finite, shape, and dtype checks are mandatory",
        )
        _require(
            bool(self.required_cases)
            and len(set(self.required_cases)) == len(self.required_cases),
            "required_cases must be nonempty and unique",
        )
        _require(
            {"decode", "prefill", "held_out_inputs"} <= set(self.required_cases),
            "correctness must cover decode, prefill, and held-out inputs",
        )


@dataclass(frozen=True, slots=True)
class BenchmarkPolicy(_Record):
    warmups: int
    trials_per_implementation: int
    order: str
    synchronize_before_and_after: bool
    primary_metric: Metric
    required_metrics: tuple[Metric, ...]
    statistics: tuple[str, ...]
    resident_model: bool
    include_model_loading: bool
    include_agent_activity: bool
    invalidate_on_swap_growth: bool
    invalidate_on_thermal_change: bool

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            self.warmups > 0 and self.trials_per_implementation >= 2,
            "use positive warmups and at least two trials",
        )
        _require(self.order == "alternating_ab_ba", "use alternating A/B and B/A pairs")
        _require(
            self.synchronize_before_and_after,
            "GPU timing must synchronize before and after each interval",
        )
        _require(
            self.primary_metric == "decode_tokens_per_second",
            "primary metric must be decode throughput",
        )
        _require(
            set(Metric) <= set(self.required_metrics),
            "missing required benchmark metrics",
        )
        _require(
            len(set(self.required_metrics)) == len(self.required_metrics),
            "required_metrics must be unique",
        )
        _require(
            set(self.statistics) == {"median", "p90", "mad"}
            and len(self.statistics) == 3,
            "statistics must contain median, p90, and mad once",
        )
        _require(
            self.resident_model
            and not self.include_model_loading
            and not self.include_agent_activity,
            "time resident-model inference with agent activity excluded",
        )
        _require(
            self.invalidate_on_swap_growth and self.invalidate_on_thermal_change,
            "swap growth and thermal changes must invalidate timing",
        )


@dataclass(frozen=True, slots=True)
class AcceptancePolicy(_Record):
    status: CalibrationStatus
    enabled: bool
    minimum_speedup: float
    maximum_p90_latency_ratio: float
    maximum_peak_memory_ratio: float
    require_all_correctness_cases: bool
    require_improvement_above_noise: bool

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(self.minimum_speedup > 1, "minimum_speedup must be greater than 1")
        _require(
            self.maximum_p90_latency_ratio >= 1 and self.maximum_peak_memory_ratio >= 1,
            "regression caps must be ratios of at least 1",
        )
        _require(
            self.require_all_correctness_cases and self.require_improvement_above_noise,
            "acceptance requires all correctness cases and improvement above noise",
        )
        _require(
            not self.enabled or self.status is CalibrationStatus.CALIBRATED,
            "cannot enable acceptance with provisional thresholds",
        )


def _relative_path(value: str) -> PurePosixPath:
    _require(
        "\\" not in value and ":" not in value,
        "paths must use repository-relative POSIX syntax",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute() and ".." not in path.parts and bool(path.parts),
        "paths must stay inside the repository",
    )
    return path


@dataclass(frozen=True, slots=True)
class AgentPolicy(_Record):
    candidate_root: str
    optimization_boundary: str
    allowed_implementation_types: tuple[str, ...]
    protected_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            self.candidate_root == "candidates", "candidate_root must be candidates"
        )
        _require(
            self.optimization_boundary == "moe_expert_execution",
            "only MoE expert execution is open to optimization",
        )
        _require(
            set(self.allowed_implementation_types) == {"mlx", "metal"}
            and len(self.allowed_implementation_types) == 2,
            "allow mlx and metal once each",
        )
        required = {
            "AGENTS.md",
            "kernelmaxxing.yaml",
            "pyproject.toml",
            "uv.lock",
            "orchestrator",
            "target",
            "evaluator",
            "tests",
            "reports",
            ".kernelmaxxing",
            "candidates/README.md",
            "candidates/candidate-template",
        }
        paths = tuple(_relative_path(path).as_posix() for path in self.protected_paths)
        _require(len(set(paths)) == len(paths), "protected_paths must be unique")
        _require(
            required <= set(paths),
            "protected_paths omits a protected system input or output",
        )
        _require(
            not any(PurePosixPath("candidates").is_relative_to(path) for path in paths),
            "protected paths must not block all candidate submissions",
        )

    def allows_write(self, relative_path: str, candidate_id: str) -> bool:
        """Lexical policy check only; filesystem/process isolation is separate."""
        if not isinstance(candidate_id, str) or not re.fullmatch(
            r"cand_[0-9a-f]{12}", candidate_id
        ):
            return False
        if not isinstance(relative_path, str):
            return False
        try:
            path = _relative_path(relative_path)
        except ContractError:
            return False
        assigned = PurePosixPath(self.candidate_root) / candidate_id
        return path.is_relative_to(assigned) and not any(
            path.is_relative_to(protected) for protected in self.protected_paths
        )


@dataclass(frozen=True, slots=True)
class ExperimentContract(_Record):
    schema_version: int
    experiment_id: str
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    correctness: CorrectnessPolicy
    benchmark: BenchmarkPolicy
    acceptance: AcceptancePolicy
    agent: AgentPolicy

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(self.schema_version == 1, "unsupported schema_version")
        _require(
            not self.acceptance.enabled
            or self.correctness.tolerance_status is CalibrationStatus.CALIBRATED,
            "cannot enable acceptance with provisional numerical tolerances",
        )

    @property
    def sha256(self) -> str:
        """Hash normalized contract semantics, independent of YAML formatting."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateIdentity(_Record):
    candidate_id: str
    implementation_type: str
    sha256: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            bool(re.fullmatch(r"cand_[0-9a-f]{12}", self.candidate_id)),
            "invalid candidate_id",
        )
        _require(
            self.implementation_type in {"mlx", "metal"},
            "unsupported optimization implementation",
        )
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.sha256)),
            "candidate hash must be SHA-256",
        )


@dataclass(frozen=True, slots=True)
class CorrectnessResult(_Record):
    case_name: str
    passed: bool
    shape_match: bool
    dtype_match: bool
    finite: bool
    tolerance_passed: bool
    detail: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        checks = (
            self.shape_match
            and self.dtype_match
            and self.finite
            and self.tolerance_passed
        )
        _require(
            self.passed == checks, "correctness passed flag contradicts its checks"
        )


@dataclass(frozen=True, slots=True)
class PerformanceResult(_Record):
    prompt_length: int
    metric: Metric
    baseline_samples: tuple[float, ...]
    candidate_samples: tuple[float, ...]
    valid: bool
    detail: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(self.prompt_length in (512, 2048, 4096), "unknown timing workload")
        _require(
            len(self.baseline_samples) == len(self.candidate_samples) >= 2,
            "timing requires equal paired sample counts of at least two",
        )
        _require(
            all(
                sample > 0
                for sample in (*self.baseline_samples, *self.candidate_samples)
            ),
            "timing samples must be positive",
        )


@dataclass(frozen=True, slots=True)
class EnvironmentRecord(_Record):
    chip: str
    memory_gib: int
    macos_version: str
    python_version: str
    mlx_version: str
    mlx_lm_version: str
    model_files_sha256: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(self.memory_gib > 0, "observed memory must be positive")
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.model_files_sha256)),
            "model file manifest hash must be SHA-256",
        )


@dataclass(frozen=True, slots=True)
class VerdictResult(_Record):
    candidate: CandidateIdentity
    contract_sha256: str
    verdict: Verdict
    reason: str

    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.contract_sha256)),
            "verdict must identify its contract hash",
        )


def load_contract(path: str | Path) -> ExperimentContract:
    """Load safe YAML, rejecting duplicate keys instead of silently overriding."""
    import yaml  # Optional parser dependency; no ML dependency.

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            _require(isinstance(key, str), "YAML field names must be strings")
            _require(key not in result, f"duplicate YAML field: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ContractError(f"invalid YAML: {exc}") from exc
    return ExperimentContract.from_mapping(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Week 1 experiment contract"
    )
    parser.add_argument("path", nargs="?", default="kernelmaxxing.yaml")
    args = parser.parse_args()
    try:
        contract = load_contract(args.path)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        parser.exit(
            1,
            "PyYAML is required to read YAML; use the uv command in evaluator/README.md.\n",
        )
    except (ContractError, OSError) as exc:
        parser.exit(1, f"Invalid contract: {exc}\n")
    print(f"Valid contract: {contract.experiment_id}")
    print(f"Contract SHA-256: {contract.sha256}")
    print(f"Acceptance enabled: {contract.acceptance.enabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
