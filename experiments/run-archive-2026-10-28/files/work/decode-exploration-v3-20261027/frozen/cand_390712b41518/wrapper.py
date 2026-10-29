"""Dense projection interface for the trusted complete-decode controller."""

import importlib.util
from pathlib import Path
import sys

_name = __name__ + "_dense_impl"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("implementation.py")
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _module
_spec.loader.exec_module(_module)
prepare_projection = _module.prepare_projection
