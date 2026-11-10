"""An exact representation of BF16 words; no floating-point conversion."""

import numpy as np


def pack_bf16_bits(words):
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2 or words.shape[1] % 64:
        raise ValueError("Expected uint16 [N,K], K divisible by64")
    rows, columns = words.shape
    blocks = np.ascontiguousarray(words).reshape(-1, 64)
    upper = blocks >> 8
    exponent_high = upper & 127
    base = exponent_high.min(axis=1)
    fits = exponent_high.max(axis=1) - base <= 7
    codes = ((exponent_high - base[:, None]) & 7) | ((upper >> 7) << 3)
    payload = np.empty((len(blocks), 96), dtype=np.uint8)
    payload[:, :64] = (blocks & 255).astype(np.uint8)
    payload[:, 64:] = (codes[:, ::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    headers = base.astype(np.uint32)
    escaped = np.flatnonzero(~fits)
    if len(escaped) >= 2**24:
        raise ValueError("Escape table exceeds its declared24bit index")
    headers[escaped] = (np.arange(len(escaped), dtype=np.uint32) + 1) << 8
    raw = blocks[escaped].copy()
    return {
        "shape": (rows, columns),
        "payload": payload,
        "headers": headers,
        "escapes": raw,
    }


def pack_chunks(chunks, shape):
    rows, columns = shape
    if columns <= 0 or columns % 64 or rows <= 0:
        raise ValueError("Positive [N,K], K divisible by64 required")
    count = rows * (columns // 64)
    payload = np.empty((count, 96), dtype=np.uint8)
    headers = np.empty(count, dtype=np.uint32)
    escapes = []
    written = 0
    escape_offset = 0
    for chunk in chunks:
        packed = pack_bf16_bits(chunk)
        if packed["shape"][1] != columns:
            raise ValueError("Chunk width changed")
        n = len(packed["headers"])
        if written + n > count:
            raise ValueError("Too many weight rows")
        h = packed["headers"].copy()
        indices = h >> 8
        mask = indices != 0
        if escape_offset + len(packed["escapes"]) >= 2**24:
            raise ValueError("Escape index overflow")
        h[mask] = (indices[mask] + escape_offset) << 8
        payload[written : written + n] = packed["payload"]
        headers[written : written + n] = h
        if len(packed["escapes"]):
            escapes.append(packed["escapes"])
        escape_offset += len(packed["escapes"])
        written += n
    if written != count:
        raise ValueError("Incomplete weight rows")
    raw = (
        np.concatenate(escapes, axis=0)
        if escapes
        else np.empty((0, 64), dtype=np.uint16)
    )
    return {
        "shape": tuple(shape),
        "payload": payload,
        "headers": headers,
        "escapes": raw,
    }


def unpack_bf16_bits(packed):
    payload = packed["payload"]
    headers = packed["headers"]
    codes = np.empty((len(headers), 64), dtype=np.uint16)
    codes[:, ::2] = payload[:, 64:] & 15
    codes[:, 1::2] = payload[:, 64:] >> 4
    upper = ((headers & 127)[:, None] + (codes & 7)) | ((codes & 8) << 4)
    words = payload[:, :64].astype(np.uint16) | (upper << 8).astype(np.uint16)
    indices = headers >> 8
    mask = indices != 0
    words[mask] = packed["escapes"][indices[mask] - 1]
    return words.reshape(packed["shape"])


def statistics(packed):
    native = int(np.prod(packed["shape"])) * 2
    encoded = sum(packed[name].nbytes for name in ("payload", "headers", "escapes"))
    return {
        "native_bytes": native,
        "encoded_bytes": encoded,
        "encoded_ratio": encoded / native,
        "blocks": len(packed["headers"]),
        "escaped_blocks": len(packed["escapes"]),
    }
