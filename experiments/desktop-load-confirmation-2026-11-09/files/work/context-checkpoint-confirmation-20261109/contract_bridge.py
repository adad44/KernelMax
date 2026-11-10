"""Verify policy-only contract changes without replacing the original reference."""
import hashlib
import json
from pathlib import Path
from evaluator.schemas import ConfirmationSwapPolicy,load_contract


ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def differences(old,new,prefix=''):
    if isinstance(old,dict) and isinstance(new,dict) and old.keys()==new.keys():
        return set().union(*(differences(old[key],new[key],prefix+('.' if prefix else '')+key)
            for key in old))
    return set() if old==new else {prefix}


def validate_bridge(bridge_path,contract_path,reference_path,*,candidate_sha256=None,root=ROOT):
    bridge_path,contract_path,reference_path=map(Path,(bridge_path,contract_path,reference_path))
    bridge=json.loads(bridge_path.read_text())
    reference=json.loads(reference_path.read_text())
    if digest(reference_path)!=bridge['original_reference_sha256']:
        raise RuntimeError('Original reference manifest changed')
    original_path=Path(bridge['original_contract_path'])
    if (str(original_path) not in reference['archived_source_sha256']
            or digest(original_path)!=reference['archived_source_sha256'][str(original_path)]):
        raise RuntimeError('Original archived contract changed')
    old=load_contract(original_path)
    new=load_contract(contract_path)
    changed=differences(old.to_dict(),new.to_dict())
    required={'experiment_id','measurement.confirmation.swap_policy','benchmark.invalidate_on_swap_growth'}
    if (changed!=required or old.sha256!=reference['contract_sha256']
            or old.sha256!=bridge['original_contract_sha256'] or new.sha256!=bridge['desktop_contract_sha256']
            or old.measurement.confirmation.swap_policy is not ConfirmationSwapPolicy.STRICT
            or new.measurement.confirmation.swap_policy is not ConfirmationSwapPolicy.DESKTOP_LOAD):
        raise RuntimeError('Contract changes exceed the approved desktop-load policy')
    if candidate_sha256 is not None and bridge['candidate_sha256']!=candidate_sha256:
        raise RuntimeError('Frozen mathematics identity changed')
    allowed={str(root/'evaluator/schemas.py'),str(root/'evaluator/measurement.py'),str(root/'AGENTS.md')}
    sources={}
    changes=bridge['policy_source_changes']
    seen=set()
    for path,previous in reference['live_source_sha256'].items():
        current=digest(path)
        if current!=previous:
            entry=changes.get(path)
            if path not in allowed or entry is None:
                raise RuntimeError('Non-policy original reference source changed: '+path)
            archive=entry['original_archive_path']
            if (entry['original_sha256']!=previous or entry['desktop_sha256']!=current
                    or reference['archived_source_sha256'].get(archive)!=previous
                    or digest(archive)!=previous):
                raise RuntimeError('Policy source bridge changed: '+path)
            seen.add(path)
        sources[path]=current
    if set(changes)!=seen:
        raise RuntimeError('Policy source allowlist is not exact')
    for path,expected in reference['archived_source_sha256'].items():
        if digest(path)!=expected: raise RuntimeError('Original source archive changed: '+path)
        sources[path]=expected
    if digest(root/'evaluator/native_baseline.py')!=reference['reference_driver_sha256']:
        raise RuntimeError('Original native reference driver changed')
    sources.update({str(bridge_path):digest(bridge_path),str(contract_path):digest(contract_path),
        str(reference_path):digest(reference_path)})
    return {'original_contract_sha256':old.sha256,'desktop_contract_sha256':new.sha256,
        'original_reference_sha256':digest(reference_path),'bridge_sha256':digest(bridge_path),
        'environment_policy':'desktop_load','sources':sources}
