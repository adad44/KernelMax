"""Reproducible synthetic inputs for the three fixed batch-size-one cases.

Token IDs are integers standing in for text tokens. These inputs describe a
workload; they do not run inference, generate output tokens, or measure speed.
No model, tokenizer, tensor library, or network access is needed. Contract-backed
inputs use the existing validated contract and require the model's vocabulary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from evaluator.schemas import ExperimentContract


PROMPT_LENGTHS = (512, 2048, 4096)
MAX_NEW_TOKENS = 128
DEFAULT_SEED = 42
DEFAULT_VOCAB_SIZE = 32000
_UNSET = object()


@dataclass(frozen=True)
class WorkloadCase:
    """One immutable batch of input token IDs and its output-token limit."""

    input_ids: Tuple[Tuple[int, ...], ...]
    max_new_tokens: int = MAX_NEW_TOKENS

    @property
    def batch_size(self) -> int:
        """Return the number of prompts (rows) in this batch."""

        return len(self.input_ids)

    @property
    def prompt_length(self) -> int:
        """Return the number of input tokens in the generated batch's row."""

        return len(self.input_ids[0])

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Describe input dimensions as (number of prompts, tokens per prompt)."""

        return self.batch_size, self.prompt_length


def generate_workload(
    *,
    seed: int | object = _UNSET,
    vocab_size: int | object = _UNSET,
    contract: ExperimentContract | None = None,
) -> Tuple[WorkloadCase, ...]:
    """Create exactly the 512-, 2048-, and 4096-token cases, in that order.

    Each case contains one prompt and allows at most 128 new output tokens.
    A private random generator keeps results repeatable for the same integer
    seed and vocabulary size without changing Python's global random state.
    IDs lie in ``[0, vocab_size)``; pass a fake tokenizer's vocabulary size when
    needed. The default vocabulary size is synthetic, not tied to a real model.

    With no contract, the existing defaults remain seed 42 and vocabulary 32000.
    For real inputs, pass ``contract=load_contract("kernelmaxxing.yaml")`` using
    the evaluator's existing loader, and explicitly supply ``vocab_size`` from
    the model's checkpoint configuration. Prompt lengths, batch size, output
    limit, and seed then come from the contract; a separate seed is rejected.
    The validated Week 1 contract requires batch size one. Contract errors use
    ``ContractError``; legacy argument errors retain their original types.

    One random stream is consumed across prompt lengths in contract order,
    exactly as in ``evaluator.native_baseline.fixed_inputs``. This helper only
    prepares inputs; sampling, cache, and timing policies remain runner-owned.
    """

    lengths, max_new_tokens = PROMPT_LENGTHS, MAX_NEW_TOKENS
    if contract is not None:
        # Keep evaluator/PyYAML requirements out of the original offline path.
        from evaluator.schemas import ContractError, ExperimentContract

        if not isinstance(contract, ExperimentContract):
            raise ContractError("contract must be a validated ExperimentContract")
        if seed is not _UNSET:
            raise ContractError("seed must come from contract.workload; omit the seed argument")
        if vocab_size is _UNSET:
            raise ContractError("vocab_size from the model configuration is required for contract inputs")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ContractError("vocab_size must be a positive integer for contract inputs")
        seed = contract.workload.seed
        lengths = contract.workload.prompt_lengths
        max_new_tokens = contract.workload.generation_tokens
    else:
        if seed is _UNSET:
            seed = DEFAULT_SEED
        if vocab_size is _UNSET:
            vocab_size = DEFAULT_VOCAB_SIZE

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    rng = random.Random(seed)
    return tuple(
        WorkloadCase(
            input_ids=(tuple(rng.randrange(vocab_size) for _ in range(length)),),
            max_new_tokens=max_new_tokens,
        )
        for length in lengths
    )
