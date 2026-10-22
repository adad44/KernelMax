"""Bounded, non-executable tensor transport for reviewed-candidate workers."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VERSION = 1
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_STORAGE_DTYPES = {"<f2", "<f4", "<f8", "<i4", "<i8", "|b1"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, limit: int = MAX_RESPONSE_BYTES):
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON payload exceeds byte limit")
    return json.loads(data)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8"
    )


def pack_tensor(value, mx=None) -> dict:
    if type(value) is np.ndarray:
        backend, dtype, host = "numpy", value.dtype.str, value
    elif mx is not None and isinstance(value, mx.array):
        if value.dtype not in (
            mx.bfloat16,
            mx.float16,
            mx.float32,
            mx.int32,
            mx.int64,
            mx.bool_,
        ):
            raise ValueError("unsupported MLX output dtype")
        backend, dtype = "mlx", str(value.dtype)
        # Original metadata is captured BEFORE BF16 host conversion.
        host = (
            np.array(value.astype(mx.float32))
            if value.dtype == mx.bfloat16
            else np.array(value)
        )
    else:
        raise ValueError("output must be a plain NumPy or MLX array")
    if host.dtype.str not in _STORAGE_DTYPES or host.nbytes > MAX_RESPONSE_BYTES:
        raise ValueError("unsupported or oversized tensor storage")
    return {
        "backend": backend,
        "dtype": dtype,
        "shape": list(host.shape),
        "storage_dtype": host.dtype.str,
        "data": base64.b64encode(host.tobytes()).decode("ascii"),
    }


def unpack_tensor(value: dict, limit: int = MAX_RESPONSE_BYTES) -> np.ndarray:
    if type(value) is not dict or set(value) != {
        "backend",
        "dtype",
        "shape",
        "storage_dtype",
        "data",
    }:
        raise ValueError("malformed tensor record")
    if value["backend"] not in ("numpy", "mlx") or type(value["dtype"]) is not str:
        raise ValueError("malformed tensor metadata")
    if value["storage_dtype"] not in _STORAGE_DTYPES:
        raise ValueError("forbidden tensor storage dtype")
    shape = value["shape"]
    if (
        type(shape) is not list
        or len(shape) > 8
        or any(type(n) is not int or n < 0 for n in shape)
    ):
        raise ValueError("malformed tensor shape")
    dtype = np.dtype(value["storage_dtype"])
    size = math.prod(shape) * dtype.itemsize
    if (
        size > limit
        or type(value["data"]) is not str
        or len(value["data"]) != 4 * ((size + 2) // 3)
    ):
        raise ValueError("tensor payload exceeds bounds or disagrees with shape")
    data = base64.b64decode(value["data"], validate=True)
    if len(data) != size:
        raise ValueError("tensor byte count disagrees with shape")
    if value["backend"] == "numpy" and value["dtype"] != value["storage_dtype"]:
        raise ValueError("NumPy original dtype disagrees with storage")
    if value["backend"] == "mlx":
        storage = {
            "mlx.core.bfloat16": "<f4",
            "mlx.core.float16": "<f2",
            "mlx.core.float32": "<f4",
            "mlx.core.int32": "<i4",
            "mlx.core.int64": "<i8",
            "mlx.core.bool_": "|b1",
        }
        if storage.get(value["dtype"]) != value["storage_dtype"]:
            raise ValueError("MLX original dtype disagrees with storage")
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)
