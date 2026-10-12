"""Check the loading interface with tiny in-memory fakes."""

import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from evaluator.schemas import ContractError, load_contract
from target.model_loader import LoadedModel, load_model
from target.workload import generate_workload


class FakeModel:
    """Stand in for a model without weights, files, or inference dependencies."""


class FakeTokenizer:
    """Expose only the vocabulary size needed to create synthetic token IDs."""

    vocab_size = 64


@pytest.fixture(scope="module")
def contract():
    """Use the repository's validated YAML contract, without loading a model."""

    return load_contract(Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml")


@pytest.fixture
def model_config():
    """Stand in for checkpoint config.json; extra checkpoint fields are allowed."""

    return {
        "model_type": "gpt_oss",
        "quantization_config": {"quant_method": "mxfp4"},
        "vocab_size": 201088,
        "torch_dtype": "bfloat16",
        "architectures": ["GptOssForCausalLM"],
    }


def test_loader_returns_fake_objects_without_a_real_model():
    """A fake backend is called once and its exact objects are preserved."""

    model = FakeModel()
    tokenizer = FakeTokenizer()
    calls = []

    def fake_loader():
        """Return existing in-memory objects instead of loading model files."""

        calls.append(True)
        return model, tokenizer

    loaded = load_model(loader=fake_loader)

    assert isinstance(loaded, LoadedModel)
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert calls == [True]

    workload = generate_workload(vocab_size=tokenizer.vocab_size)
    assert [case.input_shape for case in workload] == [
        (1, 512),
        (1, 2048),
        (1, 4096),
    ]
    assert all(
        0 <= token < tokenizer.vocab_size
        for case in workload
        for row in case.input_ids
        for token in row
    )


def test_loader_requires_an_explicit_backend():
    """There is no automatic model backend when the caller omits the loader."""

    with pytest.raises(TypeError):
        load_model()


def test_contract_config_is_checked_before_fake_loading(contract, model_config):
    """Valid configuration reaches the unchanged loader interface once."""

    model, tokenizer = FakeModel(), FakeTokenizer()
    loader = Mock(return_value=(model, tokenizer))
    original = copy.deepcopy(model_config)

    loaded = load_model(loader=loader, contract=contract, model_config=model_config)

    loader.assert_called_once_with()
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert model_config == original
    workload = generate_workload(contract=contract, vocab_size=model_config["vocab_size"])
    assert [case.input_shape for case in workload] == [
        (contract.workload.batch_size, length)
        for length in contract.workload.prompt_lengths
    ]
    assert all(case.max_new_tokens == contract.workload.generation_tokens for case in workload)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model_type", "llama", "model_type"),
        ("quantization_config", None, "native MXFP4"),
        ("quantization_config", {}, "native MXFP4"),
        ("quantization_config", {"quant_method": "affine"}, "native MXFP4"),
        ("vocab_size", None, "vocab_size"),
        ("vocab_size", True, "vocab_size"),
        ("vocab_size", 0, "vocab_size"),
        ("vocab_size", -1, "vocab_size"),
        ("vocab_size", 1.5, "vocab_size"),
        ("vocab_size", "201088", "vocab_size"),
        ("torch_dtype", "float16", "torch_dtype"),
        ("dtype", "float32", "dtype"),
    ],
)
def test_invalid_model_config_never_calls_loader(contract, model_config, field, value, message):
    """Reject incompatible metadata before a backend could load any weights."""

    model_config[field] = value
    loader = Mock()

    with pytest.raises(ContractError, match=message):
        load_model(loader=loader, contract=contract, model_config=model_config)

    loader.assert_not_called()


@pytest.mark.parametrize("field", ["model_type", "quantization_config", "vocab_size"])
def test_missing_model_config_fields_never_call_loader(contract, model_config, field):
    """Required configuration fields cannot silently fall back to fake defaults."""

    del model_config[field]
    loader = Mock()

    with pytest.raises(ContractError):
        load_model(loader=loader, contract=contract, model_config=model_config)

    loader.assert_not_called()


@pytest.mark.parametrize("model_config", [[], "config.json", 42])
def test_model_config_must_be_a_mapping(contract, model_config):
    """Use parsed checkpoint metadata rather than inventing another config format."""

    loader = Mock()

    with pytest.raises(ContractError, match="configuration mapping"):
        load_model(loader=loader, contract=contract, model_config=model_config)

    loader.assert_not_called()


@pytest.mark.parametrize("missing", ["contract", "model_config"])
def test_contract_and_model_config_are_required_together(contract, model_config, missing):
    """An incomplete contract-backed request cannot take the unchecked legacy path."""

    arguments = {"contract": contract, "model_config": model_config}
    del arguments[missing]
    loader = Mock()

    with pytest.raises(ContractError):
        load_model(loader=loader, **arguments)

    loader.assert_not_called()


def test_unvalidated_contract_never_calls_loader(contract, model_config):
    """Callers must use the existing contract parser and its strict validation."""

    loader = Mock()

    with pytest.raises(ContractError, match="validated ExperimentContract"):
        load_model(loader=loader, contract=contract.to_dict(), model_config=model_config)

    loader.assert_not_called()


@pytest.mark.parametrize("dtype_field", [None, "torch_dtype", "dtype"])
def test_optional_dtype_metadata_uses_contract_policy(contract, model_config, dtype_field):
    """Native config may omit dtype or declare it with either known field name."""

    del model_config["torch_dtype"]
    if dtype_field is not None:
        model_config[dtype_field] = contract.model.activation_dtype
    loader = Mock(return_value=(FakeModel(), FakeTokenizer()))

    load_model(loader=loader, contract=contract, model_config=model_config)

    loader.assert_called_once_with()


def test_backend_errors_still_propagate_without_retry(contract, model_config):
    """The integration validates inputs but does not change backend error behavior."""

    error = RuntimeError("fake backend failed")
    loader = Mock(side_effect=error)

    with pytest.raises(RuntimeError) as raised:
        load_model(loader=loader, contract=contract, model_config=model_config)

    assert raised.value is error
    loader.assert_called_once_with()
