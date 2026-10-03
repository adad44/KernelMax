"""Check the loading interface with tiny in-memory fakes."""

import pytest

from target.model_loader import LoadedModel, load_model
from target.workload import generate_workload


class FakeModel:
    """Stand in for a model without weights, files, or inference dependencies."""


class FakeTokenizer:
    """Expose only the vocabulary size needed to create synthetic token IDs."""

    vocab_size = 64


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
