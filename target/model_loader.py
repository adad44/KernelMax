"""Model loading boundary with an explicitly supplied backend.

This module has no default model backend and imports no ML libraries. Tests
can supply a loader returning fake objects; a future local backend can use the
same interface. Contract-backed loading checks supplied checkpoint configuration
before calling the backend. The caller controls any work performed by it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from evaluator.schemas import ExperimentContract


ModelLoader = Callable[[], Tuple[object, object]]


@dataclass(frozen=True)
class LoadedModel:
    """Keep a model and its matching tokenizer together without modifying them."""

    model: object
    tokenizer: object


def load_model(
    *,
    loader: ModelLoader,
    contract: ExperimentContract | None = None,
    model_config: Mapping[str, object] | None = None,
) -> LoadedModel:
    """Call the supplied loader once and return its model/tokenizer pair.

    ``loader`` must be a zero-argument callable returning ``(model, tokenizer)``.
    It is required: omitting it cannot trigger a real model load or download.
    For offline use, supply a fake loader or an existing local-only backend.
    Backend exceptions propagate so callers can diagnose loading failures.

    For contract-backed loading, pass an ``ExperimentContract`` from the existing
    ``evaluator.schemas.load_contract("kernelmaxxing.yaml")`` and the checkpoint's
    parsed ``config.json`` as ``model_config``. Both are required together. Check
    architecture, native MXFP4 configuration, vocabulary size, and any declared
    dtype before invoking the same zero-argument loader. Contract errors use the
    evaluator's existing ``ContractError``. This mode uses the evaluator's Python
    environment; legacy fake loading remains independent of it.

    Configuration checks do not verify checkpoint identity/hashes, weight tensor
    formats, runtime/cache precision, or that the supplied loader uses this
    configuration. Those remain caller/evaluator responsibilities.
    """

    if contract is not None or model_config is not None:
        # Keep evaluator/PyYAML requirements out of the original offline path.
        from evaluator.schemas import ContractError, ExperimentContract

        if not isinstance(contract, ExperimentContract):
            raise ContractError("contract must be a validated ExperimentContract")
        if not isinstance(model_config, Mapping):
            raise ContractError("model_config must be a checkpoint configuration mapping")
        if model_config.get("model_type") != contract.model.architecture:
            raise ContractError("model_config.model_type must match contract.model.architecture")
        quantization = model_config.get("quantization_config")
        if not isinstance(quantization, Mapping) or quantization.get("quant_method") != "mxfp4":
            raise ContractError("model_config must preserve native MXFP4 expert weights")
        vocab_size = model_config.get("vocab_size")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ContractError("model_config.vocab_size must be a positive integer")
        for field in ("torch_dtype", "dtype"):
            if field in model_config and model_config[field] != contract.model.activation_dtype:
                raise ContractError(f"model_config.{field} must match contract.model.activation_dtype")

    model, tokenizer = loader()
    return LoadedModel(model=model, tokenizer=tokenizer)
