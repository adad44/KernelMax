"""Exact-layout reader for supported BF16 dense one-token projections."""

import importlib.util
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

_name = __name__ + "_codec"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("codec.py")
)
_codec = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _codec
_spec.loader.exec_module(_codec)

_HEADER = r"""
#include <metal_simdgroup>
inline float4 km_load_bf16_4(
    const device uchar* payload, const device uint* headers,
    const device ushort* escapes, uint block, uint offset) {
    const uint header = headers[block];
    const uint escape = header >> 8;
    uint4 words;
    if (escape != 0) {
        ushort4 raw = *reinterpret_cast<const device ushort4*>(
            escapes + (escape - 1) * 64 + offset);
        words = uint4(raw);
    } else {
        const device uchar* data = payload + ulong(block) * 96;
        const uint low = *reinterpret_cast<const device uint*>(data + offset);
        const ushort high = *reinterpret_cast<const device ushort*>(data + 64 + offset / 2);
        #pragma unroll
        for (uint q = 0; q < 4; ++q) {
            const uint code = (high >> (q * 4)) & 15u;
            const uint upper = ((header & 127u) + (code & 7u)) | ((code & 8u) << 4);
            words[q] = ((low >> (q * 8)) & 255u) | (upper << 8);
        }
    }
    return float4(as_type<float>(words[0] << 16), as_type<float>(words[1] << 16),
                  as_type<float>(words[2] << 16), as_type<float>(words[3] << 16));
}
"""

_KERNEL = mx.fast.metal_kernel(
    name="kernelmax_lossless_dense_cand_390712b41518",
    input_names=["x", "payload", "headers", "escapes", "bias"],
    output_names=["output"],
    source=Path(__file__).with_name("implementation.metal").read_text(),
    header=_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)


class PreparedProjection:
    def __init__(self, weight, *, chunk_rows=4096):
        if weight.ndim != 2 or weight.dtype != mx.bfloat16:
            raise ValueError("BF16 [N,K] native dense weights required")
        self.n, self.k = weight.shape
        if (
            self.n < 4
            or self.n % 4
            or self.k <= 64
            or self.k % 64
            or self.k >= 16 * self.n
        ):
            raise ValueError(
                "Unsupported native GEMV reduction/layout; use declared native fallback"
            )

        def chunks():
            with mx.stream(mx.cpu):
                for start in range(0, self.n, chunk_rows):
                    yield np.array(weight[start : start + chunk_rows].view(mx.uint16))

        packed = _codec.pack_chunks(chunks(), (self.n, self.k))
        self.statistics = _codec.statistics(packed)
        # CPU preparation is outside inference timing. Keep prototype sidecar
        # cost explicit; native weights remain owned by the reference model.
        with mx.stream(mx.cpu):
            self.payload = mx.array(packed["payload"], dtype=mx.uint8)
            self.headers = mx.array(packed["headers"], dtype=mx.uint32)
            raw = packed["escapes"].reshape(-1)
            self.escapes = mx.array(
                raw if len(raw) else np.zeros(64, np.uint16), dtype=mx.uint16
            )
            self.empty_bias = mx.zeros((1,), dtype=mx.bfloat16)
            mx.eval(self.payload, self.headers, self.escapes, self.empty_bias)
        self.statistics["allocated_sidecar_bytes"] = (
            self.payload.nbytes
            + self.headers.nbytes
            + self.escapes.nbytes
            + self.empty_bias.nbytes
        )

    def __call__(self, x, bias=None):
        if x.dtype != mx.bfloat16 or x.shape[-1] != self.k or x.size != self.k:
            raise ValueError(
                "Prepared projection specializes one BF16 vector; use native prefill fallback"
            )
        if bias is not None and (bias.dtype != mx.bfloat16 or bias.shape != (self.n,)):
            raise ValueError("Unsupported native dense bias")
        simds = 8 if self.n >= 4096 else 4
        groups = (self.n + 4 * simds - 1) // (4 * simds)
        result = _KERNEL(
            inputs=[
                x,
                self.payload,
                self.headers,
                self.escapes,
                self.empty_bias if bias is None else bias,
            ],
            template=[
                ("T", mx.bfloat16),
                ("K", self.k),
                ("N", self.n),
                ("BM", simds),
                ("HAS_BIAS", bias is not None),
            ],
            grid=(groups * simds * 32, 1, 1),
            threadgroup=(simds * 32, 1, 1),
            output_shapes=[(*x.shape[:-1], self.n)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return result


def prepare_projection(weight):
    return PreparedProjection(weight)
