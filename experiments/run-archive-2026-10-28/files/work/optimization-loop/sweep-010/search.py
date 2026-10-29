"""Bounded deterministic kernel search; immediate transitions without a schedule.

Only numeric launch/reduction variants of a reviewed kernel family are generated.
Every candidate is frozen once and passes native/tiny correctness before timing.
All outcomes are diagnostic; a shortlist requires independent full-model evidence.
"""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent
ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
from evaluator.schemas import load_contract
from scripts.loop_control import load_authorization

SEARCH=CONTROL/'search-20261027.json'
DEADLINE=json.loads(SEARCH.read_text())['deadline_unix']
TEMPLATE=CONTROL/'candidate-009'
BASE=ROOT/'candidates/cand_5dc343a37cb3'
HALF=ROOT/'candidates/cand_430b7852e4b5'
state={'status':'starting','pid':os.getpid(),'started_unix':time.time(),
       'deadline_unix':DEADLINE,'trials':[],'shortlist':[]}
STATE=HERE/'state.json'
if STATE.exists():
    raise SystemExit('Existing search retained; inspect before resuming')


def save():
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');tmp.replace(STATE)


def update(**changes):
    s=json.loads(SEARCH.read_text());s.update(changes)
    tmp=SEARCH.with_suffix('.sweep-tmp')
    tmp.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');tmp.replace(SEARCH)


def tree():
    rows={}
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
        f=line.split(None,7)
        if len(f)==8:
            rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
    return rows


def track(root_pid,known):
    rows=tree();included={root_pid}
    changed=True
    while changed:
        changed=False
        for pid,row in rows.items():
            if row['ppid'] in included and pid not in included:
                included.add(pid);changed=True
    for pid in included:
        if pid in rows:known[pid]=rows[pid]


def cleanup(known):
    rows=tree()
    for pid,old in reversed(list(known.items())):
        if pid in rows and rows[pid]['start']==old['start'] and rows[pid]['command']==old['command']:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass


def metal(lanes,groups,values):
    # Reuse the correctness-passed BF16 epilogue; only numeric reduction mapping varies.
    original=(BASE/'octet_up_gate.metal').read_text()
    epilogue=original[original.index('if (k_lane == 0u) {\n    // Match BF16'):]
    rows=groups*(32//lanes)
    source=f'''// Generated numeric variant: lanes={lanes}, simdgroups={groups}, values/lane={values}.
const uint simd_lane=thread_index_in_simdgroup;
const uint k_lane=simd_lane & {lanes-1}u;
const uint row=threadgroup_position_in_grid.x*{rows}u
    +simdgroup_index_in_threadgroup*{32//lanes}u+simd_lane/{lanes}u;
const uint slot=threadgroup_position_in_grid.y;
const uint expert=uint(ids[slot]);
if(expert>=E){{if(k_lane==0u)activated[slot*N+row]=T(as_type<float>(0x7fc00000u));return;}}
const uint output_row=expert*N+row;
float sum_up=0.0f,sum_gate=0.0f;
for(uint base=0u;base<K;base+={lanes*values}u){{
    const uint column_base=base+k_lane*{values}u;
    #pragma unroll
    for(uint pack=0u;pack<{values//8}u;++pack){{
        const uint column=column_base+pack*8u;
        if(column<K){{
            float x_values[8];
            #pragma unroll
            for(uint j=0u;j<8u;++j)x_values[j]=float(x[column+j]);
            const uint w_offset=output_row*(K/8u)+column/8u;
            const uint s_offset=output_row*(K/32u)+column/32u;
            sum_up+=km_scale(up_s[s_offset])*km_dot8(up_w[w_offset],x_values);
            sum_gate+=km_scale(gate_s[s_offset])*km_dot8(gate_w[w_offset],x_values);
        }}
    }}
}}
'''
    delta=lanes//2
    while delta:
        source+=f'sum_up+=simd_shuffle_down(sum_up,{delta});\nsum_gate+=simd_shuffle_down(sum_gate,{delta});\n'
        delta//=2
    return source+epilogue


def generate(config,index):
    lanes,groups,values,decoder=config
    rows=groups*(32//lanes)
    cid='cand_'+secrets.token_hex(6)
    candidate=ROOT/'candidates'/cid;candidate.mkdir()
    source=((BASE if decoder=='lut' else HALF)/'implementation.py').read_text()
    source=source.replace('octet_up_gate.metal','projection.metal')
    source=source.replace('kernelmax_octet_up_gate_decode',f'kernelmax_tune_{cid}')
    source=source.replace('kernelmax_half_fp4_up_gate_decode',f'kernelmax_tune_{cid}')
    source=source.replace('grid=((n // 8) * 64, 4, 1)',f'grid=((n // {rows}) * {groups*32}, 4, 1)')
    source=source.replace('threadgroup=(64, 1, 1)',f'threadgroup=({groups*32}, 1, 1)')
    # Unsupported small fixture output sizes use the declared reference fallback.
    needle='    if k <= 0 or k % 32 or n <= 0 or n % 8 or e <= 0:'
    source=source.replace(needle,f'    if n % {rows}:\n        return experts(x, selected_expert_ids)\n'+needle)
    ast.parse(source)
    body=metal(lanes,groups,values)
    for k in (64,96,256,2880):
        columns=[base+lane*values+pack*8+j for base in range(0,k,lanes*values)
                 for lane in range(lanes) for pack in range(values//8)
                 if base+lane*values+pack*8<k for j in range(8)]
        if sorted(columns)!=list(range(k)):raise ValueError('K coverage failure')
    if 2880%rows:raise ValueError('Native output row coverage unsupported')
    for n in (64,96,2880):
        if n%rows:continue
        covered=[block*rows+sg*(32//lanes)+lane//lanes for block in range(n//rows)
                 for sg in range(groups) for lane in range(0,32,lanes)]
        if covered!=list(range(n)):raise ValueError('Output row coverage failure')
    (candidate/'implementation.py').write_text(source)
    (candidate/'projection.metal').write_text(body)
    (candidate/'wrapper.py').write_bytes((BASE/'wrapper.py').read_bytes())
    (candidate/'manifest.json').write_text(json.dumps({
        'candidate_id':cid,'name':f'Joint up/gate L{lanes} G{groups} V{values} {decoder}',
        'implementation_type':'mlx',
        'description':f'Measured-region numeric tuning: {lanes} lanes per row, {groups} SIMD groups per threadgroup, {values} K values per lane, exact {decoder} FP4 decoder. Targets the audited 0.379 ms up/gate isolated saving, not a full-model share. Native MXFP4 weights/scales/routes, float accumulations and BF16 epilogue unchanged; no copied weights, requantization or tensor cache. Native down and prefill fallback remain; unsupported small output row tiles use reference fallback. No performance claim.'},indent=2)+'\n')
    work=HERE/f'trial-{index:03d}-{cid}';work.mkdir()
    (work/'plan.json').write_text(json.dumps({
        'candidate_id':cid,'candidate_source':str(candidate),'config':list(config),
        'measured_target':str(CONTROL/'probe-004/audited-result.json'),
        'hypothesis':'Numeric reduction/coalescing/parallelism variants within the measured up/gate region. Emitted-instruction, occupancy and bandwidth causes are unmeasured.',
        'model_contribution':'Unverified. A hypothetical 20% region gain gives 0.0759 ms per sampled expert or 1.82 ms/token only if all24 layers are representative and additive.',
        'calibration':'New20pair paired-ratio bootstrap A/A; CI includes1, width<=0.04, center/order bias<=0.02. All raw samples/MAD/range retained. This replaces the exploratory four-wall-time MAD screen; no old results relabeled.',
        'futility':'After20 candidate pairs, park unless CI lower>1 and median gain exceeds A/A order bias. Independent full-model confirmation required after the exploratory multi-candidate search.',
        'acceptance_enabled':False},indent=2,sort_keys=True)+'\n')
    (work/'tiny_native_checks.py').write_bytes((TEMPLATE/'tiny_native_checks.py').read_bytes())
    (work/'paired_noise.py').write_bytes((HERE/'paired_noise.py').read_bytes())
    (work/'operator_diagnostic.py').write_bytes((HERE/'operator-template.py').read_bytes())
    pipeline=(TEMPLATE/'pipeline.py').read_text()
    pipeline,count=re.subn(r"^CID='cand_[0-9a-f]{12}'$",f"CID='{cid}'",pipeline,flags=re.MULTILINE)
    if count!=1:raise ValueError('Pipeline candidate identity replacement failed')
    ast.parse(pipeline);(work/'pipeline.py').write_text(pipeline)
    (work/'prepared-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in work.glob('*.py')},indent=2,sort_keys=True)+'\n')
    return cid,work


def audit(work,expected_cid):
    reports=list((work/'native-correctness/artifacts').glob('*/report.json'))
    if len(reports)==1:
        native=json.loads(reports[0].read_text())
        if native['candidate']['candidate_id']!=expected_cid:
            raise ValueError('Native report executed a different candidate')
        if any(any(word in error.lower() for word in ('integrity','protected sources','authorization','checkpoint'))
               for error in native.get('errors',[])):
            raise ValueError('Native integrity/authorization failure; search stopped')
    path=work/'operator-diagnostic.json'
    if not path.exists():return {'status':'no_operator_result','shortlisted':False}
    data=json.loads(path.read_text())
    if data['candidate_id']!=expected_cid:
        raise ValueError('Timing report executed a different candidate')
    lookup={(r['layer'],r['phase'],r['pair']):r['blocks'] for r in data.get('microblocks',[])}
    for r in data['samples']:
        matching=[b for b in lookup[r['layer'],r['phase'],r['pair']] if b['lane']==r['lane']]
        if sum(b['calls'] for b in matching)!=r['calls'] or r['calls']!=1024:
            raise ValueError('Raw sample call-count audit failure')
        if abs(sum(b['elapsed_ms'] for b in matching)-r['elapsed_ms'])>1e-7:
            raise ValueError('Raw sample timing aggregation audit failure')
    guards=all(data.get(k) is True for k in ('protected_sources_unchanged','snapshot_unchanged','checkpoint_identity_unchanged'))
    if not guards:
        raise ValueError('Timing integrity failure; search stopped')
    shortlist=(data['status']=='completed_diagnostic' and set(data['summaries'])=={'0','12','23'} and guards
               and not data.get('environment_issues') and all(v['passed'] for v in data.get('aa_screens',{}).values()))
    return {'status':data['status'],'summaries':data['summaries'],'guards_passed':guards,
            'all_actual_counts_verified':True,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'shortlisted':shortlist,'full_model_speedup_established':False}


def main():
    excludes={(8,2,32,'lut'),(8,2,32,'half'),(16,2,16,'half')}
    pool=[v for v in itertools.product((16,32,4,8),(2,4,1,8),(8,16,32),('lut','half')) if v not in excludes]
    state['predeclared_pool']=[list(v) for v in pool];state['status']='running';save()
    assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    child=None;known={};failures=0;calibration_failures=0
    try:
        for index,config in enumerate(pool,1):
            if time.time()+60>=DEADLINE or (HERE/'STOP').exists():break
            load_authorization(CONTROL/'session.json',load_contract(ROOT/'kernelmaxxing.yaml'))
            cid,work=generate(config,index)
            trial_deadline=min(time.time()+1800,DEADLINE)
            process={'deadline_unix':trial_deadline,'supervisor_pid':os.getpid(),
                     'status':'running','candidate_id':cid,'started_unix':time.time()}
            (work/'process.json').write_text(json.dumps(process,indent=2)+'\n')
            state['current_trial']={'candidate_id':cid,'directory':str(work),'config':list(config)};save()
            update(status='continuous_numeric_search',active_job={'label':'com.kernelmax.search.sweep010.20261027',
                   'directory':str(HERE),'supervisor_pid':os.getpid(),'trial_directory':str(work)},
                   current_candidate={'candidate_id':cid,'source':str(ROOT/'candidates'/cid),'work':str(work),'stage':'pipeline_starting'})
            known={}
            with (work/'pipeline-stdout.log').open('w') as out,(work/'pipeline-stderr.log').open('w') as err:
                child=subprocess.Popen([sys.executable,'-B','-u',str(work/'pipeline.py')],cwd=ROOT,
                                       stdout=out,stderr=err,start_new_session=True)
                while child.poll() is None:
                    track(child.pid,known)
                    if time.time()>=trial_deadline or (HERE/'STOP').exists():break
                    time.sleep(.5)
                track(child.pid,known);cleanup(known);child.wait()
            process.update(status='completed' if child.returncode==0 else 'failed',returncode=child.returncode,ended_unix=time.time())
            (work/'process.json').write_text(json.dumps(process,indent=2)+'\n')
            outcome=audit(work,cid);outcome.update(candidate_id=cid,directory=str(work),config=list(config),process=process)
            (work/'result.json').write_text(json.dumps(outcome,indent=2,sort_keys=True)+'\n')
            state['trials'].append(outcome);save()
            print(json.dumps({'trial':index,'candidate_id':cid,'status':outcome['status'],'shortlisted':outcome['shortlisted']}),flush=True)
            if outcome['shortlisted']:
                state['shortlist'].append(outcome);state['status']='shortlist_needs_independent_full_model_confirmation';save();break
            failures=failures+1 if outcome['status']=='no_operator_result' else 0
            calibration_failures=calibration_failures+1 if outcome['status']=='stopped_aa_screen' else 0
            if failures>=2:
                state['status']='stopped_repeated_pipeline_failure';save();break
            if calibration_failures>=3:
                state['status']='stopped_repeated_calibration_failure';save();break
            if time.time()>=DEADLINE or (HERE/'STOP').exists():break
            # The next distinct variant starts immediately; no heartbeat gate.
        else:
            state['status']='numeric_pool_exhausted'
        if state['status']=='running':state['status']='deadline_or_stop_reached'
    finally:
        if child is not None:
            track(child.pid,known);cleanup(known)
        assertion.terminate();assertion.wait()
        state['ended_unix']=time.time();save()
        update(status=state['status'],active_job=None,continuous_search=str(STATE),
               next_action='Immediately inspect sweep-010 state/trial results. Shortlists need independent full-model evidence. If pool exhausted or failed, identify the next justified action rather than rerolling old snapshots. Preserve all samples; no commits. Stop at1PM Pacific.')


if __name__=='__main__':
    main()
