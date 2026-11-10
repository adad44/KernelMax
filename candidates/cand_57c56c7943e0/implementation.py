"""Exact C2 arithmetic composed with one-step asynchronous decode scheduling."""
import importlib.util
from pathlib import Path
import sys


def load_child(kind):
    name=__name__+'_'+kind
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parent/kind/'implementation.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


_vocabulary=load_child('vocabulary')
_moe=load_child('moe')
_schedule_name=__name__+'_scheduling'
_schedule_spec=importlib.util.spec_from_file_location(_schedule_name,Path(__file__).with_name('scheduling.py'))
_schedule=importlib.util.module_from_spec(_schedule_spec)
sys.modules[_schedule_name]=_schedule
_schedule_spec.loader.exec_module(_schedule)


class Adapter:
    kind='combined'
    requires_weight_residency=True

    def __init__(self,model):
        self.vocabulary=_vocabulary.prepare(model)
        self.moe=_moe.prepare(model)
        self.shape=self.vocabulary.shape
        self.statistics={**self.vocabulary.statistics,'moe':dict(self.moe.statistics),
                         'composition':'exact C2 vocabulary and BN8 fused MoE plus true-step async scheduling',
                         'decode_scheduling':{'outputs':128,'async_submissions':127,
                            'maximum_queued_successor_graphs':1,'first_token_and_final_drain_accounted':True}}

    def export_layout(self): return self.vocabulary.export_layout()
    def load_layout(self,arrays): self.vocabulary.load_layout(arrays)
    def unload_layout(self): self.vocabulary.unload_layout()
    def packed_bytes(self): return self.vocabulary.packed_bytes()
    def reconstructed_weight(self): return self.vocabulary.reconstructed_weight()
    def call_projection(self,original,layer,x):
        return self.vocabulary.call_projection(original,layer,x)
    def call_block(self,original,layer,x,mask,cache,trace):
        return self.moe.call_block(original,layer,x,mask,cache,trace)
    def decode_request(self,*args): return _schedule.decode_request(*args)
    def trace_request(self,*args): return _schedule.trace_request(*args)


def prepare(model): return Adapter(model)
