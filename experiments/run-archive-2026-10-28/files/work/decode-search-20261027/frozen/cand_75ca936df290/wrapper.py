"""Broader decode entry point; intentionally no expert-only run function."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    __name__ + '_decode_impl', Path(__file__).with_name('implementation.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
decode_request = _module.decode_request
