"""CPU lifetime tests for the trusted storage switch; no MLX imports."""
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from weight_residency import WeightResidency


class Array:
    def __init__(self, size, shape=(), dtype='bf16'):
        self.nbytes, self.shape, self.dtype = size, shape, dtype


class LifetimeTests(unittest.TestCase):
    def fixture(self):
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=Array(160,(10,8))))
        class Adapter:
            size = 0
            def unload_layout(self): self.size=0
            def packed_bytes(self): return self.size
            def load_layout(self, arrays): self.size=sum(a.nbytes for a in arrays.values())
        adapter = Adapter()
        loads = []
        fail = {'packed':False}
        def load(path):
            # The former representation must be gone before allocating the new.
            self.assertEqual(model.lm_head.weight.nbytes,0)
            self.assertEqual(adapter.packed_bytes(),0)
            loads.append(path)
            if path=='packed':
                if fail['packed']: raise OSError('injected load failure')
                return {'payload':Array(120)}
            return {'weight':Array(160,(10,8))}
        mx = SimpleNamespace(cpu='cpu',stream=lambda _:contextlib.nullcontext(),
            bfloat16='bf16',load=load,eval=lambda *args:None,synchronize=lambda:None,
            clear_cache=lambda:None,get_active_memory=lambda:1000+model.lm_head.weight.nbytes+adapter.size)
        storage = WeightResidency.__new__(WeightResidency)
        storage.model,storage.adapter,storage.mx=model,adapter,mx
        storage.shape=(10,8)
        storage.native_bytes,storage.packed_size=160,120
        storage.placeholder=Array(0)
        storage.mode=False
        storage.events=[]
        storage.native_path,storage.packed_path=Path('native'),Path('packed')
        storage.check_files=lambda:None
        return storage, loads, fail

    def test_each_lane_retains_only_its_own_representation(self):
        storage,loads,fail=self.fixture()
        storage.switch(True)
        self.assertEqual(storage.snapshot()['mlx_active_bytes'],1120)
        storage.assert_lane(True)
        storage.switch(False)
        self.assertEqual(storage.snapshot()['mlx_active_bytes'],1160)
        storage.assert_lane(False)
        self.assertEqual(loads,['packed','native'])
        self.assertTrue(all(e['outside_request_timing'] for e in storage.events))

    def test_same_lane_reuses_loaded_weights_without_io(self):
        storage,loads,fail=self.fixture()
        storage.switch(True)
        storage.switch(True)
        self.assertEqual(loads,['packed'])
        self.assertEqual(len(storage.events),1)

    def test_failed_candidate_activation_can_restore_native(self):
        storage,loads,fail=self.fixture()
        fail['packed']=True
        with self.assertRaises(OSError): storage.switch(True)
        self.assertIsNone(storage.mode)
        storage.switch(False)
        storage.assert_lane(False)

    def test_unprepared_or_overlapping_lane_rejected(self):
        storage,loads,fail=self.fixture()
        with self.assertRaises(RuntimeError): storage.assert_lane(True)
        storage.adapter.size=120
        with self.assertRaises(RuntimeError): storage.assert_lane(False)

    def test_backing_identity_failure_precedes_mutation(self):
        storage,loads,fail=self.fixture()
        def invalid(): raise RuntimeError('changed file')
        storage.check_files=invalid
        original=storage.model.lm_head.weight
        with self.assertRaisesRegex(RuntimeError,'changed file'): storage.switch(True)
        self.assertIs(storage.model.lm_head.weight,original)
        self.assertEqual(loads,[])


if __name__=='__main__': unittest.main()
