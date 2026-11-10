"""Recompute context evidence from immutable raw artifacts, without MLX."""
import json
import math
from pathlib import Path
import random
import statistics
import sys
from evidence import EvidenceError,GUARDS,pair_rows,publish,sha_file,state_of

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
sys.path.insert(0,str(ROOT))
from evaluator.measurement import assess_environment


def interval(values,seed):
    rng=random.Random(seed)
    values=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [values[249],values[9749]]


def verify_sample(raw,*,context,lane,reference_tokens,primer=False):
    if (raw['prompt_tokens']!=context or raw['lane']!=lane
            or raw['actual_forward_calls']!=129 or raw['output_token_ids']!=reference_tokens
            or len(raw['output_token_ids'])!=128 or raw['tokens_match_frozen_reference'] is not True):
        raise EvidenceError('Sample changed workload, tokens or forward count')
    if primer and raw.get('untimed') is not True: raise EvidenceError('Primer role missing')
    mode='exploration' if primer else 'confirmation'
    flags=assess_environment(raw['environment_before'],raw['environment_after'],mode=mode,
        thermal_states=raw['thermal_states'],confirmation_policy='retain_and_flag_desktop_load')
    if not flags['continue_pass'] or (not primer and not flags['confirmation_sample_eligible']):
        raise EvidenceError('Hard environment failure in raw evidence')
    if raw['environment_assessment']!=flags:
        raise EvidenceError('Stored environment assessment does not match raw observations')
    metrics=raw['metrics']
    required=('decode_tokens_per_second','time_to_first_token_ms','prefill_tokens_per_second',
        'inter_token_latency_ms','peak_memory_bytes','full_model_wall_time_ms')
    if any(not isinstance(metrics[key],(int,float)) or not math.isfinite(metrics[key]) or metrics[key]<=0 for key in required):
        raise EvidenceError('Nonfinite or nonpositive metrics')
    decode_seconds=(metrics['full_model_wall_time_ms']-metrics['time_to_first_token_ms'])/1000
    if decode_seconds<=0 or not math.isclose(metrics['decode_tokens_per_second'],127/decode_seconds,rel_tol=1e-7):
        raise EvidenceError('Decode throughput inconsistent with complete request timing')
    if lane=='candidate':
        if (raw['timing_observations']['async_submissions']!=127
                or raw['placement_counts']['projection_specialized']!=128
                or raw['placement_counts']['block_specialized']!=3072
                or raw['placement_counts']['projection_fallback']!=1
                or raw['placement_counts']['block_fallback']!=24):
            raise EvidenceError('Candidate placement or scheduled-work count changed')
        observed=raw['timing_observations'];tokens=observed['tokens_at']
        if (len(tokens)!=128 or any(not math.isfinite(value) for value in tokens)
                or any(b<a for a,b in zip(tokens,tokens[1:]))
                or not observed['start']<=observed['prefill_end']<=tokens[0]
                or tokens[-1]!=observed['gpu_drained_at']
                or observed['gpu_drained_at']<observed['last_token_observed_at']
                or not math.isclose(metrics['full_model_wall_time_ms'],1000*(tokens[-1]-observed['start']),rel_tol=1e-7)):
            raise EvidenceError('First token or final GPU drain timing changed')
    return metrics['decode_tokens_per_second']


def verify_numeric(proof):
    """Verify hash-bound numerical receipts and the logits/routes/token arrays.

    Cache receipt fields are produced by the frozen native correctness suite;
    legacy NPZs contain logits/routes/tokens, not cache arrays. Source receipts
    and their original failure statuses are retained rather than relabeled.
    """
    import numpy as np
    full={row['length']:row for row in proof['full']}
    if set(full)!={512,2048,4096} or len(proof['full'])!=3:
        raise EvidenceError('Full numerical coverage missing')
    reuse=proof['reuse']
    if reuse.get('old_performance_samples_reused') is not False or reuse.get('old_calibration_rows_reused') is not False:
        raise EvidenceError('Historical performance cannot qualify a new pass')
    for path,expected in reuse['source_artifact_sha256'].items():
        if sha_file(path)!=expected: raise EvidenceError('Numerical source receipt changed')
    reference_tokens={}
    for row in list(full.values())+[proof['handoff']]:
        steps=row['outputs']
        if steps not in (8,128) or row['length'] not in full:
            raise EvidenceError('Unexpected numerical workload')
        if (not row['passed'] or not row['numerical_passed'] or row['token_differences']
                or row['route_set_differences'] or not row['cache_within_tolerance']
                or not row['cache_metadata_equal'] or row['reference_cache_dtypes']!=['mlx.core.bfloat16']
                or row['candidate_cache_dtypes']!=['mlx.core.bfloat16']
                or row['candidate_trace_driver']!='actual_async_greedy'
                or row['candidate_forced_reference_tokens'] is not False
                or row['candidate_forward_calls']!=steps+1 or row['candidate_async_submissions']!=steps-1
                or not row['cache_max_abs_errors']
                or any(value is None or not math.isfinite(value) or value<0 for value in row['cache_max_abs_errors'])
                or sha_file(row['arrays'])!=row['arrays_sha256']):
            raise EvidenceError('Invalid full-model numerical receipt')
        with np.load(row['arrays'],allow_pickle=False) as arrays:
            a,b=arrays['reference_logits'],arrays['candidate_logits']
            if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.abs(a-b)<=0.02+0.02*np.abs(a)):
                raise EvidenceError('Numerical logits changed')
            if not np.array_equal(arrays['reference_tokens'],arrays['candidate_tokens']) or len(arrays['reference_tokens'])!=steps:
                raise EvidenceError('Generation trajectory changed')
            if not np.array_equal(np.sort(arrays['reference_routes'],axis=-1),np.sort(arrays['candidate_routes'],axis=-1)):
                raise EvidenceError('Routing trajectory changed')
            if steps==128: reference_tokens[row['length']]=arrays['reference_tokens'].tolist()
            elif arrays['reference_tokens'].tolist()!=reference_tokens[512][:8]:
                raise EvidenceError('Fresh runtime handoff differs from the frozen reference trajectory')
    if proof['handoff']['outputs']!=8 or proof['handoff']['length']!=512:
        raise EvidenceError('Fresh runtime handoff missing')
    return reference_tokens


class ContextVerifier:
    def __init__(self,store,*,sample_validator=verify_sample,numeric_validator=verify_numeric):
        self.store=store
        self.sample_validator=sample_validator;self.numeric_validator=numeric_validator

    def _references(self,state,epoch):
        qualifications=[collection for collection in state['collections'].values()
            if collection['epoch']==epoch and collection['role']=='qualification']
        if len(qualifications)!=1 or not qualifications[0]['committed'] or len(qualifications[0]['calls'])!=2:
            raise EvidenceError('Each epoch needs its own charged fresh native/candidate handoff')
        proofs=[]
        for artifact in qualifications[0]['artifacts']:
            row=json.loads((self.store.root/artifact['path']).read_text())
            if not row['executed'] or not row['eligible']: raise EvidenceError('Runtime handoff failed or missing')
            proofs.append(row['raw'])
        if proofs[0]!=proofs[1]: raise EvidenceError('Qualified trace pair has inconsistent evidence')
        return self.numeric_validator(proofs[0])

    def _rows(self,collection,reference_tokens):
        if not collection['committed']: raise EvidenceError('Uncommitted collection')
        if len(collection['calls'])!=2 or [call['role'] for call in collection['calls']]!=['primer','measured']:
            raise EvidenceError('Primer/measurement reservation missing')
        values=[]
        for primer,artifact in zip((True,False),collection['artifacts']):
            wrapper=json.loads((self.store.root/artifact['path']).read_text())
            if not wrapper['executed'] or not wrapper['eligible']: raise EvidenceError('Invalid or missing raw row')
            raw=wrapper['raw'];c=collection['coordinates']
            expected_phase={'comparison':f"confirm-{c.get('block')}",
                'warmup':f"warmup-{c.get('block')}", 'aa':'aa','aa_warmup':'aa_warmup',
                'diagnostic':'phase_memory'}.get(c.get('phase'))
            if raw.get('phase')!=expected_phase or raw.get('trial')!=c['pair']:
                raise EvidenceError('Raw row does not match its predeclared schedule coordinates')
            values.append(self.sample_validator(raw,context=c['context'],lane=c['lane'],
                reference_tokens=reference_tokens[c['context']],primer=primer))
        return values[-1]

    def _warmups(self,state,epoch,context,phase,block,before,references):
        lanes=('A','B') if phase=='aa_warmup' else ('baseline','candidate')
        groups={lane:{} for lane in lanes}
        for collection in state['collections'].values():
            if collection['collection_id']==before: break
            c=collection['coordinates']
            if collection['epoch']!=epoch or c.get('context')!=context or c.get('phase')!=phase or (phase=='warmup' and c.get('block')!=block): continue
            if c['lane'] not in groups or c['pair'] in groups[c['lane']]: raise EvidenceError('Duplicate warmup')
            groups[c['lane']][c['pair']]=self._rows(collection,references)
        if any(sorted(rows)!=list(range(1,6)) for rows in groups.values()):
            raise EvidenceError('Each lane needs its five predeclared epoch-local warmups')

    def control(self,epoch,context):
        self.store.verify_artifacts();state=state_of(self.store.read()[0],self.store.protocol)
        references=self._references(state,epoch)
        groups={lane:{} for lane in ('A','B')}
        order={};first=None
        for collection in state['collections'].values():
            c=collection['coordinates']
            if collection['epoch']!=epoch or c.get('phase')!='aa' or c.get('context')!=context: continue
            if first is None: first=collection['collection_id']
            if c['pair'] not in order and c['pair']!=len(order)+1: raise EvidenceError('A/A pair order changed')
            seen=order.setdefault(c['pair'],[])
            expected=('A','B') if c['pair']%2 else ('B','A')
            if len(seen)>=2 or c['lane']!=expected[len(seen)]: raise EvidenceError('A/A alternating order changed')
            seen.append(c['lane'])
            if c['lane'] not in groups or c['pair'] in groups[c['lane']]: raise EvidenceError('Duplicate A/A row')
            groups[c['lane']][c['pair']]=self._rows(collection,references)
        if any(sorted(rows)!=list(range(1,21)) for rows in groups.values()):
            raise EvidenceError('A/A requires every one of20same-epoch pairs')
        self._warmups(state,epoch,context,'aa_warmup',None,first,references)
        values=[groups['B'][trial]/groups['A'][trial] for trial in range(1,21)]
        ci=interval(values,42+context)
        return {'epoch':epoch,'context':context,'pairs':20,'paired_ratios':values,
            'median_ratio':statistics.median(values),'bootstrap_95pct_ci':ci,
            'passed':ci[0]<=1<=ci[1] and ci[1]-ci[0]<=0.04}

    def record_control(self,controller,context):
        value=self.control(controller.epoch,context)
        path=f'controls/{controller.epoch}-{context}.json'
        receipt=publish(self.store.root/path,value)
        controller.append({'kind':'control','epoch':controller.epoch,'context':context,'passed':value['passed'],
            'receipt':{'path':path,'sha256':receipt['sha256']}})
        return value

    def analyze(self,context):
        self.store.verify_artifacts();events,head=self.store.read();state=state_of(events,self.store.protocol)
        controls=[]
        for event in state['controls'].values():
            if event['context']!=context: continue
            actual=self.control(event['epoch'],context)
            stored=json.loads((self.store.root/event['receipt']['path']).read_text())
            if actual!=stored or actual['passed'] is not event['passed']:
                raise EvidenceError('A/A summary does not match all retained raw rows')
            controls.append(actual)
        if not controls: raise EvidenceError('No complete fresh A/A group')
        noise=max(abs(bound-1) for control in controls for bound in control['bootstrap_95pct_ci'])
        all_controls_passed=all(control['passed'] for control in controls)
        blocks={};epochs=set();unmatched_all=[];references_by_epoch={}
        for block in range(1,self.store.protocol['workload']['comparison_blocks'][str(context)]+1):
            pairs,unmatched=pair_rows(self.store,context,block)
            for item in unmatched:
                collection=item['collection'];wrapper=item['row']
                unmatched_all.append({'collection_id':collection['collection_id'],'epoch':collection['epoch'],
                    'lane':collection['coordinates']['lane'],'executed':wrapper['executed'] if wrapper else None,
                    'eligible':wrapper['eligible'] if wrapper else False,
                    'metrics':wrapper['raw'].get('metrics') if wrapper else None})
            if len(pairs)!=20: raise EvidenceError('A block needs exactly20complete pairs; cannot choose a subset')
            values=[];saves=[]
            warmed=set()
            for pair in pairs:
                epoch=pair['epoch'];epochs.add(epoch)
                control=state['controls'].get(f'{epoch}:{context}')
                if control is None: raise EvidenceError('Each contributing epoch needs its own complete control')
                if epoch not in references_by_epoch: references_by_epoch[epoch]=self._references(state,epoch)
                references=references_by_epoch[epoch]
                if epoch not in warmed:
                    earliest=next(row['collection_id'] for row in state['collections'].values()
                        if row['epoch']==epoch and row['coordinates'].get('phase')=='comparison'
                        and row['coordinates'].get('context')==context and row['coordinates'].get('block')==block)
                    self._warmups(state,epoch,context,'warmup',block,earliest,references)
                    warmed.add(epoch)
                a=self._rows(pair['lanes']['baseline']['collection'],references)
                b=self._rows(pair['lanes']['candidate']['collection'],references)
                values.append(b/a);saves.append(1000/a-1000/b)
            ci=interval(values,context);median=statistics.median(values)
            blocks[str(block)]={'pairs':20,'paired_ratios':values,'median_ratio':median,
                'bootstrap_95pct_ratio':ci,'median_saved_ms_per_token':statistics.median(saves),
                'repeatable_gain_verified':all_controls_passed and median>1 and ci[0]>1+noise,
                'five_percent_verified':all_controls_passed and median>=1.05 and ci[0]>max(1.05,1+noise)}
        return {'context':context,'blocks':blocks,'controls':controls,'aa_noise':noise,
            'all_controls_passed':all_controls_passed,'contributing_epochs':sorted(epochs),
            'structurally_unmatched_collections':unmatched_all,'evidence_prefix_sha256':head,
            'repeatable_gain_verified':all(block['repeatable_gain_verified'] for block in blocks.values()),
            'five_percent_verified':all(block['five_percent_verified'] for block in blocks.values()),
            'claim_scope':self.store.protocol['environment']['claim_scope'],'acceptance_enabled':False}

    def seal(self,controller,context,guard_receipt):
        controller.check()
        if not all(guard_receipt.get(key) is True for key in GUARDS): raise EvidenceError('Context final guard failed')
        state=controller.state()
        diagnostics=[row for row in state['collections'].values()
            if row['coordinates'].get('phase')=='diagnostic' and row['committed'] and row['artifacts'][-1]['executed']]
        if (len(diagnostics)!=2 or {row['coordinates']['lane'] for row in diagnostics}!={'baseline','candidate'}
                or any(row['coordinates']['context']!=512 for row in diagnostics)):
            raise EvidenceError('The two predeclared initial phase diagnostics are missing or duplicated')
        for diagnostic in diagnostics:
            references=self._references(state,diagnostic['epoch'])
            self._rows(diagnostic,references)
            measured=json.loads((self.store.root/diagnostic['artifacts'][-1]['path']).read_text())['raw']
            if measured.get('instrumented') is not True or measured.get('phase_memory') is None:
                raise EvidenceError('Phase diagnostic instrumentation missing')
        for epoch in state['epochs'].values():
            if epoch['epoch']==controller.epoch: continue
            if not epoch['closed']: raise EvidenceError('Contributing epoch remains live')
            previous=json.loads((self.store.root/epoch['receipt']['path']).read_text())
            if not all(previous.get(key) is True for key in GUARDS): raise EvidenceError('Previous epoch guard failed')
        value=self.analyze(context)
        value['guard_receipt']=guard_receipt
        value['protocol_sha256']=state['freeze']['protocol_sha256']
        value['frozen_identities']=state['freeze']['identities']
        path=f'contexts/{context}.json';receipt=publish(self.store.root/path,value)
        controller.append({'kind':'seal_context','context':context,
            'receipt':{'path':path,'sha256':receipt['sha256']}})
        return value
