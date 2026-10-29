"""Compose the two independently frozen placements without changing arithmetic.

Source subdirectories contain byte-identical vendored files from the selected
vocabulary and MoE snapshots. The controller records their original hashes.
"""
import importlib.util
from pathlib import Path
import sys


def _load(relative, suffix):
    name=__name__+'_'+suffix
    path=Path(__file__).parent/relative/'implementation.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


_vocabulary=_load('vocabulary','vocabulary')
_moe=_load('moe','moe')


class Adapter:
    kind='combined'

    def __init__(self,model):
        self.vocabulary=_vocabulary.prepare(model)
        self.moe=_moe.prepare(model)
        self.statistics={'vocabulary':self.vocabulary.statistics,'moe':self.moe.statistics,
                         'changes':'composition of frozen placements; no new numerical operations'}

    def call_projection(self,original,layer,x):
        return self.vocabulary.call_projection(original,layer,x)

    def call_block(self,original,layer,x,mask,cache,trace):
        return self.moe.call_block(original,layer,x,mask,cache,trace)


def prepare(model):
    return Adapter(model)
