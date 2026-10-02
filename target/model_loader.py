"""Model loading boundary with an explicitly supplied backend.

This module has no default model backend and imports no ML libraries. Tests
can supply a loader returning fake objects; a future local backend can use the
same interface. The caller controls any work performed by that backend.
"""

from dataclasses import dataclass
from typing import Callable, Tuple


ModelLoader = Callable[[], Tuple[object, object]]


@dataclass(frozen=True)
class LoadedModel:
    """Keep a model and its matching tokenizer together without modifying them."""

    model: object
    tokenizer: object


def load_model(*, loader: ModelLoader) -> LoadedModel:
    """Call the supplied loader once and return its model/tokenizer pair.

    ``loader`` must be a zero-argument callable returning ``(model, tokenizer)``.
    It is required: omitting it cannot trigger a real model load or download.
    For offline use, supply a fake loader or an existing local-only backend.
    Backend exceptions propagate so callers can diagnose loading failures.
    """

    model, tokenizer = loader()
    return LoadedModel(model=model, tokenizer=tokenizer)
