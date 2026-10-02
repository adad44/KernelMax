"""Reproducible synthetic inputs for the three fixed batch-size-one cases.

Token IDs are integers standing in for text tokens. These inputs describe a
workload; they do not run inference, generate output tokens, or measure speed.
No model, tokenizer, tensor library, or network access is needed.
"""

import random
from dataclasses import dataclass
from typing import Tuple


PROMPT_LENGTHS = (512, 2048, 4096)
MAX_NEW_TOKENS = 128
DEFAULT_SEED = 42
DEFAULT_VOCAB_SIZE = 32000


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
    *, seed: int = DEFAULT_SEED, vocab_size: int = DEFAULT_VOCAB_SIZE
) -> Tuple[WorkloadCase, ...]:
    """Create exactly the 512-, 2048-, and 4096-token cases, in that order.

    Each case contains one prompt and allows at most 128 new output tokens.
    A private random generator keeps results repeatable for the same integer
    seed and vocabulary size without changing Python's global random state.
    IDs lie in ``[0, vocab_size)``; pass a fake tokenizer's vocabulary size when
    needed. The default vocabulary size is synthetic, not tied to a real model.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    rng = random.Random(seed)
    return tuple(
        WorkloadCase(
            input_ids=(tuple(rng.randrange(vocab_size) for _ in range(length)),)
        )
        for length in PROMPT_LENGTHS
    )
