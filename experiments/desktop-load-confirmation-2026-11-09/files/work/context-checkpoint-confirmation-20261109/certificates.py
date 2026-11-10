"""Verify a sealed prefix independently before retaining a context checkpoint."""
import json
from pathlib import Path
import sys
from evidence import EvidenceError,GUARDS,Store,sha_file,state_of
from verification import ContextVerifier

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
sys.path.insert(0,str(ROOT))
from evaluator.checkpoint_verification import validate_receipt


class PrefixStore:
    def __init__(self,store,events,head):
        self.original=store;self.root=store.root;self.protocol=store.protocol
        self.events=events;self.head=head
    def read(self): return self.events,self.head
    def verify_artifacts(self,events=None):
        return Store.verify_artifacts(self,self.events if events is None else events)


def verify_identities(identities,protocol):
    for path,expected in identities['source_sha256'].items():
        if sha_file(path)!=expected: raise EvidenceError('Frozen executable/input changed: '+path)
    if (sha_file(identities['original_reference_path'])!=protocol['original_reference_sha256']
            or identities['candidate_sha256']!=protocol['candidate_sha256']):
        raise EvidenceError('Original reference or candidate identity changed')
    frozen=json.loads(Path(identities['candidate_metadata_path']).read_text())
    if sha_file(identities['candidate_metadata_path'])!=identities['candidate_metadata_sha256'] or frozen['sha256']!=protocol['candidate_sha256']:
        raise EvidenceError('Frozen candidate receipt changed')
    for name,expected in frozen['file_sha256'].items():
        if sha_file(Path(frozen['directory'])/name)!=expected: raise EvidenceError('Candidate content changed')
    original_path=Path(identities['original_checkpoint_receipt_path'])
    if sha_file(original_path)!='a40793bc665c8fbb108028d391abf2b4d3bf0b5836d712e6eedf910545d5e2df':
        raise EvidenceError('Original checkpoint receipt changed')
    original=json.loads(original_path.read_text())
    local_path=Path(identities['checkpoint_receipt_path'])
    if sha_file(local_path)!=identities['checkpoint_receipt_sha256']: raise EvidenceError('Local checkpoint receipt changed')
    local=json.loads(local_path.read_text())
    if local['model_files']!=original['model_files'] or local['model_revision']!=original['model_revision']:
        raise EvidenceError('Checkpoint byte manifest or revision changed')
    validate_receipt(local,local['model_path'],original['model_revision'],original['model_files'])


def verify_context(store,context,*,verifier_factory=ContextVerifier,identity_validator=verify_identities):
    events,_=store.read();store.verify_artifacts()
    matches=[(index,event) for index,event in enumerate(events)
        if event['kind']=='seal_context' and event['context']==context]
    if len(matches)!=1: raise EvidenceError('Exactly one immutable context seal required')
    index,event=matches[0]
    raw_records=store.path.read_bytes().splitlines()
    head=json.loads(raw_records[index])['previous_sha256']
    prefix=PrefixStore(store,events[:index],head)
    receipt=event['receipt'];value=json.loads((store.root/receipt['path']).read_text())
    state=state_of(prefix.read()[0],store.protocol)
    identities=state['freeze']['identities']
    identity_validator(identities,store.protocol)
    if value['frozen_identities']!=identities or value['protocol_sha256']!=state['freeze']['protocol_sha256']:
        raise EvidenceError('Seal was produced under different frozen identities')
    guards=value['guard_receipt']
    if not all(guards.get(key) is True for key in GUARDS): raise EvidenceError('Context boundary guard failed')
    for key in ('protected_source_sha256','numerical_source_sha256'):
        for path,expected in guards[key].items():
            if sha_file(path)!=expected: raise EvidenceError('Sealed provenance changed')
    actual=verifier_factory(prefix).analyze(context)
    for key,expected in actual.items():
        if value.get(key)!=expected: raise EvidenceError('Sealed conclusion differs from complete raw evidence: '+key)
    if value['evidence_prefix_sha256']!=head: raise EvidenceError('Seal is not bound to its actual journal prefix')
    return value
