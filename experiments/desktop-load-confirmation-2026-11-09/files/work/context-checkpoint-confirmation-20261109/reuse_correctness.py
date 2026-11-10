"""Reuse completed numerical evidence only after exact identity restoration.

The prior failed confirmation stays failed. No old performance sample is reused.
The driver separately revalidates the checkpoint receipt before every operation.
"""
import hashlib
import json
from pathlib import Path
from contract_bridge import validate_bridge
from model_relocation import validate_relocation


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def completed_correctness(parent_path, *, candidate_id, snapshot_sha256,
                          contract_sha256, receipt_path, residency,
                          bridge_path=None,contract_path=None,reference_path=None,relocation_path=None):
    import numpy as np
    parent_path=Path(parent_path)
    parent=json.loads(parent_path.read_text())
    prior=parent['prior_failed_run']
    for label in ('resident_state','command002_state'):
        if digest(prior[label+'_path'])!=prior[label+'_sha256']:
            raise RuntimeError('Historical evidence artifact changed: '+label)
    if digest(prior['terminal_audit_path'])!=prior['terminal_audit_sha256']:
        raise RuntimeError('Historical terminal audit changed')
    bridged=validate_bridge(bridge_path,contract_path,reference_path,candidate_sha256=snapshot_sha256) if bridge_path else None
    old_sources=json.loads(Path(reference_path).read_text())['live_source_sha256'] if bridged else {}
    numerical_contract=bridged['original_contract_sha256'] if bridged else contract_sha256
    relocation=validate_relocation(relocation_path,receipt_path=receipt_path) if relocation_path else None
    numerical_receipt_sha=relocation['original_receipt_sha256'] if relocation else digest(receipt_path)
    if bridged and bridged['desktop_contract_sha256']!=contract_sha256:
        raise RuntimeError('Desktop timing contract identity changed')
    current_sources={}
    for path,expected in parent['prior_worker_protected_before'].items():
        if path in old_sources and old_sources[path]!=expected:
            raise RuntimeError('Original numerical source identity changed: '+path)
        current=bridged['sources'].get(path,expected) if bridged else expected
        if digest(path)!=current:
            raise RuntimeError('Numerical source identity changed: '+path)
        current_sources[path]=current
    old=json.loads(Path(prior['command002_state_path']).read_text())
    if (old['candidate_id']!=candidate_id or old['snapshot_sha256']!=snapshot_sha256
            or old['contract_sha256']!=numerical_contract
            or parent['old_candidate']['bundle_sha256']!=snapshot_sha256
            or numerical_receipt_sha!=parent['checkpoint_receipt']['sha256']):
        raise RuntimeError('Candidate/contract/checkpoint receipt identity changed')
    # Historical final-checkpoint unavailability is acknowledged explicitly;
    # it is not relabeled as an old pass. Restoration is independently verified
    # by the current driver's exact original receipt/file-identity validation.
    if (old['status']!='failed_integrity' or old.get('checkpoint_and_snapshot_unchanged') is not False
            or not any('FileNotFoundError' in error and 'chat_template.jinja' in error for error in old['errors'])
            or not all(old.get(key) is True for key in
                ('protected_unchanged','vocabulary_storage_unchanged','original_methods_restored'))
            or len(old['requests'])!=39 or len(old['resident_primers'])!=40
            or old['calibrations'] or old['summaries']
            or any(row['phase'] not in ('aa_warmup','aa') or row['lane'] not in ('A','B')
                   or row['invalid_reasons'] or not row['tokens_match_frozen_reference']
                   or row['actual_forward_calls']!=129 for row in old['requests'])):
        raise RuntimeError('Prior failure differs from the declared external interruption')
    failed=old['resident_primers'][-1]
    if (failed['phase']!='aa' or failed['lane']!='B' or failed['trial']!=15
            or failed['prompt_tokens']!=512
            or failed['environment_assessment']['stop_reasons']!=['AC power unavailable']
            or "'AC Power'" not in failed['environment_before']['power']
            or "'Battery Power'" not in failed['environment_after']['power']
            or failed['environment_before']['thermal_state']!=0
            or failed['environment_after']['thermal_state']!=0):
        raise RuntimeError('Prior stop was not the declared AC and drive interruption')
    old_storage=old['vocabulary_residency']
    if (residency is None or not residency.actual_matrix_unpack_bit_exact
            or not residency.native_roundtrip_bit_exact
            or residency.original_word_sha256!=old_storage['original_word_sha256']):
        raise RuntimeError('Restored actual vocabulary matrix identity mismatch')
    rows={row['length']:row for row in old['correctness']}
    if set(rows)!={512,2048,4096} or len(old['correctness'])!=3:
        raise RuntimeError('Missing completed full-length numerical coverage')
    result=[]
    tokens={}
    arrays={}
    for length,row in sorted(rows.items()):
        if (row['outputs']!=128 or not row['passed'] or not row['numerical_passed']
                or not row['all_logits_bit_exact'] or row['token_differences']
                or row['route_set_differences'] or row['route_slot_differences']
                or not row['cache_within_tolerance'] or not row['cache_metadata_equal']
                or any(error!=0 for error in row['cache_max_abs_errors'])
                or row['reference_cache_dtypes']!=['mlx.core.bfloat16']
                or row['candidate_cache_dtypes']!=['mlx.core.bfloat16']):
            raise RuntimeError('Completed numerical result is not the preserved exact pass')
        if (row['candidate_trace_driver']!='actual_async_greedy'
                or row['candidate_forced_reference_tokens'] is not False
                or row['candidate_forward_calls']!=129 or row['candidate_async_submissions']!=127):
            raise RuntimeError('Preserved result did not qualify the actual scheduled driver')
        if digest(row['arrays'])!=row['arrays_sha256']:
            raise RuntimeError('Numerical arrays changed')
        with np.load(row['arrays'],allow_pickle=False) as data:
            reference=data['reference_tokens'].tolist()
            if len(reference)!=128 or reference!=data['candidate_tokens'].tolist():
                raise RuntimeError('Preserved deterministic token evidence inconsistent')
        tokens[length]=reference
        arrays[row['arrays']]=row['arrays_sha256']
        result.append({**row,'reused_completed_evidence':True,
                       'source_state':prior['command002_state_path']})
    sources={**(bridged['sources'] if bridged else {}),**current_sources,
             prior['resident_state_path']:prior['resident_state_sha256'],
             prior['command002_state_path']:prior['command002_state_sha256'],
             prior['terminal_audit_path']:prior['terminal_audit_sha256'],
             str(receipt_path):digest(receipt_path),**(relocation['sources'] if relocation else {})}
    evidence={'parent_manifest_sha256':digest(parent_path),'source_state_sha256':prior['command002_state_sha256'],
              'prior_final_checkpoint_guard':False,'current_original_receipt_restored':relocation is None,
              'equivalent_checkpoint_relocation_verified':relocation is not None,
              'old_failure_preserved':True,'old_performance_samples_reused':False,
              'actual_matrix_word_sha256':residency.original_word_sha256,'array_sha256':arrays,
              'source_artifact_sha256':sources,'old_calibration_rows_reused':False,
              'numerical_source_contract_sha256':numerical_contract,
              'fresh_timing_contract_sha256':contract_sha256,
              'policy_bridge_sha256':bridged['bridge_sha256'] if bridged else None,
              'basis':'Same frozen mathematics, runtime/source/contract, original checkpoint identities and exact materialized vocabulary words; fresh matched timing after restored power and mounted drive.'}
    return result,tokens,evidence
