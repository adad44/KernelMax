"""Trusted, untimed native/packed vocabulary storage exchange.

Candidate code receives arrays only. Backing files, provenance, stat receipts
and final byte checks are controller owned; old snapshots are never modified.
"""
import gc
import hashlib
import json
from pathlib import Path
import time


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def word_digest(array, mx):
    import numpy as np
    digest = hashlib.sha256()
    with mx.stream(mx.cpu):
        for start in range(0, array.shape[0], 4096):
            bits = np.array(array[start:start + 4096].view(mx.uint16))
            digest.update(bits.tobytes())
    return digest.hexdigest()


class WeightResidency:
    def __init__(self, model, adapter, mx, directory):
        self.model, self.adapter, self.mx = model, adapter, mx
        self.directory = Path(directory)
        self.directory.mkdir()
        self.native_path = self.directory / 'native-head.safetensors'
        self.packed_path = self.directory / 'packed-head.safetensors'
        self.shape = tuple(model.lm_head.weight.shape)
        self.native_bytes = model.lm_head.weight.nbytes
        self.packed_size = adapter.packed_bytes()
        self.placeholder = mx.zeros((0,), dtype=mx.bfloat16)
        mx.eval(self.placeholder)
        self.mode = None
        self.events = []
        self.original_word_sha256 = word_digest(model.lm_head.weight, mx)
        # Complete actual-matrix bit proof for the new unpack/fallback path.
        reconstructed = adapter.reconstructed_weight()
        equal = mx.all(reconstructed.view(mx.uint16) == model.lm_head.weight.view(mx.uint16))
        mx.eval(equal)
        self.actual_matrix_unpack_bit_exact = bool(equal.item())
        del reconstructed, equal
        if not self.actual_matrix_unpack_bit_exact:
            raise ValueError('Actual vocabulary matrix failed exact unpack proof')
        mx.synchronize()
        with mx.stream(mx.cpu):
            mx.save_safetensors(str(self.native_path), {'weight': model.lm_head.weight})
            mx.save_safetensors(str(self.packed_path), adapter.export_layout())
        self.receipt = {
            str(path): {'sha256': digest_file(path), 'identity': file_identity(path)}
            for path in (self.native_path, self.packed_path)
        }
        # Drop both representations before restoring the first baseline. No
        # native backup array survives in the controller, model or adapter.
        model.lm_head.weight = self.placeholder
        adapter.unload_layout()
        gc.collect()
        mx.clear_cache()
        self.switch(False)
        self.native_roundtrip_bit_exact = word_digest(model.lm_head.weight, mx) == self.original_word_sha256
        if not self.native_roundtrip_bit_exact:
            raise ValueError('Native baseline backing-store roundtrip changed BF16 words')
        self.check_files()
        (self.directory / 'receipt.json').write_text(json.dumps(self.description(), indent=2, sort_keys=True) + '\n')

    def check_files(self):
        for path, expected in self.receipt.items():
            if file_identity(path) != expected['identity']:
                raise RuntimeError('Vocabulary backing store identity changed: ' + path)

    def switch(self, enabled):
        """Complete exchange before cooldown, observation and request timing."""
        self.check_files()
        if self.mode is bool(enabled):
            self.assert_lane(enabled)
            return
        start = time.perf_counter()
        self.mx.synchronize()
        self.mode = None
        self.model.lm_head.weight = self.placeholder
        self.adapter.unload_layout()
        gc.collect()
        self.mx.clear_cache()
        if self.adapter.packed_bytes() or self.model.lm_head.weight.nbytes:
            raise RuntimeError('Previous vocabulary storage retained before exchange')
        with self.mx.stream(self.mx.cpu):
            if enabled:
                arrays = self.mx.load(str(self.packed_path))
                self.mx.eval(*arrays.values())
                self.adapter.load_layout(arrays)
                del arrays
            else:
                arrays = self.mx.load(str(self.native_path))
                if set(arrays) != {'weight'}:
                    raise ValueError('Native backing store tensor names changed')
                weight = arrays['weight']
                if tuple(weight.shape) != self.shape or weight.dtype != self.mx.bfloat16:
                    raise ValueError('Native baseline shape or dtype changed')
                self.mx.eval(weight)
                self.model.lm_head.weight = weight
                del arrays, weight
        self.mx.synchronize()
        self.mode = bool(enabled)
        self.assert_lane(enabled)
        self.events.append({
            'candidate': bool(enabled), 'outside_request_timing': True,
            'wall_ms': (time.perf_counter() - start) * 1000,
            **self.snapshot(),
        })

    def assert_lane(self, enabled):
        native, packed = self.model.lm_head.weight.nbytes, self.adapter.packed_bytes()
        if self.mode is not bool(enabled):
            raise RuntimeError('Vocabulary lane not activated before timing')
        if (native, packed) != ((0, self.packed_size) if enabled else (self.native_bytes, 0)):
            raise RuntimeError('Native and packed vocabulary residency overlap/mismatch')

    def snapshot(self):
        return {'native_vocabulary_bytes': self.model.lm_head.weight.nbytes,
                'packed_vocabulary_bytes': self.adapter.packed_bytes(),
                'mlx_active_bytes': self.mx.get_active_memory(),
                'candidate': self.mode}

    def verify_final(self):
        self.check_files()
        for path, expected in self.receipt.items():
            if digest_file(path) != expected['sha256']:
                raise RuntimeError('Vocabulary backing-store bytes changed: ' + path)
        return True

    def description(self):
        return {'native_bytes': self.native_bytes, 'packed_bytes': self.packed_size,
                'original_word_sha256': self.original_word_sha256,
                'actual_matrix_unpack_bit_exact': self.actual_matrix_unpack_bit_exact,
                'native_roundtrip_bit_exact': self.native_roundtrip_bit_exact,
                'backing_stores': self.receipt,
                'switches': self.events,
                'current': self.snapshot()}
