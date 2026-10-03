"""Verify the fixed workload entirely with synthetic integer data."""

import random

import pytest

from target.workload import generate_workload


def test_exact_fixed_cases_and_input_shapes():
    """All three cases have one row, the required length, and a 128-token limit."""

    workload = generate_workload()

    assert [case.prompt_length for case in workload] == [512, 2048, 4096]
    for case, length in zip(workload, (512, 2048, 4096)):
        assert case.batch_size == 1
        assert case.max_new_tokens == 128
        assert case.input_shape == (1, length)
        assert len(case.input_ids) == 1
        assert len(case.input_ids[0]) == length


def test_same_seed_produces_same_workload():
    """Compare complete workloads, including every input token ID."""

    assert generate_workload(seed=7) == generate_workload(seed=7)
    assert generate_workload() == generate_workload(seed=42)
    assert generate_workload(seed=7) != generate_workload(seed=8)


def test_generation_preserves_global_random_state():
    """Preparing benchmark inputs must not change other code's randomness."""

    state = random.getstate()

    generate_workload()

    assert random.getstate() == state


def test_synthetic_tokens_fit_the_vocabulary():
    """Every token is a plain integer in the requested synthetic vocabulary."""

    workload = generate_workload(vocab_size=16)

    assert all(
        type(token) is int and 0 <= token < 16
        for case in workload
        for row in case.input_ids
        for token in row
    )


@pytest.mark.parametrize("seed", [None, True, 1.5, "42"])
def test_seed_must_be_an_integer(seed):
    """Reject values such as None that could make generation nondeterministic."""

    with pytest.raises(TypeError, match="seed must be an integer"):
        generate_workload(seed=seed)


@pytest.mark.parametrize("vocab_size", [None, True, 1.5, "16"])
def test_vocabulary_size_must_be_an_integer(vocab_size):
    """Only integer vocabulary sizes define valid integer token ranges."""

    with pytest.raises(TypeError, match="vocab_size must be an integer"):
        generate_workload(vocab_size=vocab_size)


@pytest.mark.parametrize("vocab_size", [0, -1])
def test_vocabulary_size_must_be_positive(vocab_size):
    """An empty or negative vocabulary cannot supply any token IDs."""

    with pytest.raises(ValueError, match="vocab_size must be positive"):
        generate_workload(vocab_size=vocab_size)
