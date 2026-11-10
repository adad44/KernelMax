"""Single-representation vocabulary layout plus source-aligned fused MoE."""
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


class Adapter:
    kind='combined'
    requires_weight_residency=True

    def __init__(self,model):
        self.vocabulary=_vocabulary.prepare(model)
        self.moe=_moe.prepare(model)
        self.shape=self.vocabulary.shape
        self.statistics={**self.vocabulary.statistics,'moe':dict(self.moe.statistics),
                         'composition':'single-representation exact vocabulary plus fused MoE with native BN8 router reduction'}

    def export_layout(self): return self.vocabulary.export_layout()
    def load_layout(self,arrays): self.vocabulary.load_layout(arrays)
    def unload_layout(self): self.vocabulary.unload_layout()
    def packed_bytes(self): return self.vocabulary.packed_bytes()
    def reconstructed_weight(self): return self.vocabulary.reconstructed_weight()
    def call_projection(self,original,layer,x):
        return self.vocabulary.call_projection(original,layer,x)
    def call_block(self,original,layer,x,mask,cache,trace):
        return self.moe.call_block(original,layer,x,mask,cache,trace)


def prepare(model): return Adapter(model)
