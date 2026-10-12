"""Verify the fixed workload entirely with synthetic integer data."""

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from evaluator.native_baseline import fixed_inputs
from evaluator.schemas import ContractError, load_contract
from target.workload import generate_workload


@pytest.fixture(scope="module")
def contract():
    """Read the actual experiment contract through the existing schema loader."""

    return load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


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


@pytest.mark.parametrize("seed", [0, 42, 43, 1042])
@pytest.mark.parametrize("vocab_size", [16, 201088])
def test_contract_tokens_match_baseline_exactly(contract, seed, vocab_size):
    """Compare every token, case order, and size against the unchanged baseline."""

    configured = replace(contract, workload=replace(contract.workload, seed=seed))
    expected = fixed_inputs(
        configured.workload.prompt_lengths, vocab_size, configured.workload.seed
    )

    cases = generate_workload(contract=configured, vocab_size=vocab_size)

    actual = [(str(case.prompt_length), list(case.input_ids[0])) for case in cases]
    assert actual == list(expected.items())
    assert len(cases) == len(configured.workload.prompt_lengths)
    for case, length in zip(cases, configured.workload.prompt_lengths):
        assert case.batch_size == configured.workload.batch_size
        assert case.input_shape == (configured.workload.batch_size, length)
        assert case.max_new_tokens == configured.workload.generation_tokens
        assert len(case.input_ids[0]) == length
    assert cases == generate_workload(contract=configured, vocab_size=vocab_size)


def test_workload_uses_settings_loaded_from_yaml(contract, tmp_path):
    """A valid revised seed in a YAML contract changes the actual token stream."""

    settings = contract.to_dict()
    settings["workload"]["seed"] = 7
    path = tmp_path / "kernelmaxxing.yaml"
    # JSON is valid YAML and uses the existing parser without a second serializer.
    path.write_text(json.dumps(settings), encoding="utf-8")
    configured = load_contract(path)

    cases = generate_workload(contract=configured, vocab_size=201088)

    assert cases == generate_workload(seed=7, vocab_size=201088)
    assert cases != generate_workload(contract=contract, vocab_size=201088)
    assert [(str(case.prompt_length), list(case.input_ids[0])) for case in cases] == list(
        fixed_inputs(configured.workload.prompt_lengths, 201088, configured.workload.seed).items()
    )


def test_contract_generation_preserves_global_random_state(contract):
    """The contract-backed path must also leave unrelated randomness untouched."""

    state = random.getstate()

    generate_workload(contract=contract, vocab_size=201088)

    assert random.getstate() == state


def test_contract_inputs_require_explicit_model_vocabulary(contract):
    """The synthetic default vocabulary must never leak into real inputs."""

    with pytest.raises(ContractError, match="vocab_size from the model configuration is required"):
        generate_workload(contract=contract)


@pytest.mark.parametrize("vocab_size", [None, True, 0, -1, 1.5, "201088"])
def test_contract_vocabulary_must_be_a_positive_integer(contract, vocab_size):
    """Invalid model metadata cannot create a fallback workload."""

    with pytest.raises(ContractError, match="positive integer"):
        generate_workload(contract=contract, vocab_size=vocab_size)


@pytest.mark.parametrize("seed", [None, 0, 42, 43])
def test_contract_seed_cannot_be_overridden(contract, seed):
    """The YAML contract alone controls the seed, even for matching overrides."""

    with pytest.raises(ContractError, match="seed must come from contract.workload"):
        generate_workload(contract=contract, seed=seed, vocab_size=201088)


def test_workload_requires_a_validated_contract(contract):
    """Raw configuration must pass the shared schema before input generation."""

    with pytest.raises(ContractError, match="validated ExperimentContract"):
        generate_workload(contract=contract.to_dict(), vocab_size=201088)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 2),
        ("prompt_lengths", [4096, 2048, 512]),
        ("generation_tokens", 127),
        ("seed", -1),
    ],
)
def test_invalid_yaml_workload_is_rejected_by_existing_schema(contract, tmp_path, field, value):
    """Unsupported settings fail validation instead of using hardcoded defaults."""

    settings = contract.to_dict()
    settings["workload"][field] = value
    path = tmp_path / "kernelmaxxing.yaml"
    path.write_text(json.dumps(settings), encoding="utf-8")

    with pytest.raises(ContractError):
        generate_workload(contract=load_contract(path), vocab_size=201088)


def test_legacy_defaults_and_seed_behavior_are_unchanged():
    """The old fake-data API keeps its defaults and accepts negative integer seeds."""

    assert generate_workload() == generate_workload(seed=42, vocab_size=32000)
    for seed in (7, -1):
        actual = generate_workload(seed=seed, vocab_size=16)
        assert [(str(case.prompt_length), list(case.input_ids[0])) for case in actual] == list(
            fixed_inputs((512, 2048, 4096), 16, seed).items()
        )
