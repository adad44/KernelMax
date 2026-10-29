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
TEMPLATE=CONTROL/'candidate-019'
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


def generate(config,index):
    factor=config[0]
    if factor not in (2,4):raise ValueError('Only predeclared loop factors')
    source_work=CONTROL/'native-order-resume-02/trial-001-cand_02e5f860f143'
    approval=json.loads((source_work/'approval.json').read_text());base=Path(approval['snapshot'])
    cid='cand_'+secrets.token_hex(6);candidate=ROOT/'candidates'/cid;candidate.mkdir()
    implementation=(base/'implementation.py').read_text().replace('cand_02e5f860f143',cid)
    shader=(base/'fused_up_gate.metal').read_text()
    marker='for (uint base = 0; base < K; base += 256) {'
    if shader.count(marker)!=1:raise ValueError('Native loop source anchor changed')
    start=shader.index(marker);opening=shader.index('{',start);depth=1;closing=opening+1
    while depth:
        if shader[closing]=='{':depth+=1
        elif shader[closing]=='}':depth-=1
        closing+=1
    inner=shader[opening+1:closing-1]
    old='const uint column = base + lane * 8;'
    if inner.count(old)!=1:raise ValueError('Native ordered-column anchor changed')
    inner=inner.replace(old,'const uint column = base + km_kblock * 256 + lane * 8;')
    replacement=f'for (uint base = 0; base < K; base += {256*factor}) {{\n    #pragma unroll\n    for (uint km_kblock = 0; km_kblock < {factor}; ++km_kblock) {{'+inner+'\n    }\n}'
    shader=shader[:start]+replacement+shader[closing:]
    if shader.count('{')!=shader.count('}'):raise ValueError('Shader brace mismatch')
    ast.parse(implementation);(candidate/'implementation.py').write_text(implementation)
    (candidate/'fused_up_gate.metal').write_text(shader);(candidate/'wrapper.py').write_bytes((base/'wrapper.py').read_bytes())
    (candidate/'manifest.json').write_text(json.dumps({'candidate_id':cid,'name':f'Native-order joint up/gate K-loop unroll{factor}',
        'implementation_type':'mlx','description':f'Preserves the exact native32-lane ordered dot8 accumulations and SIMD sum while unrolling the outer K loop by{factor}. Uses the earlier native-order LUT up/gate kernel, unchanged native down, BF16 projections/bias/SwiGLU and native prefill fallback. No weight/layout/precision/routing change or cache. Tests loop-control/load scheduling on a measured up/gate region; hardware cause and performance unproven.'},indent=2)+'\n')
    work=HERE/f'trial-{index:03d}-{cid}';work.mkdir()
    hypothesis=json.loads((CONTROL/'next-native-loop-hypothesis.json').read_text())
    hypothesis.update(candidate_id=cid,loop_factor=factor,source_frozen_candidate=approval['snapshot_sha256'],exact_repair_required=True,status='prepared_candidate')
    (work/'plan.json').write_text(json.dumps(hypothesis,indent=2,sort_keys=True)+'\n')
    for name in ['tiny_native_checks.py','operator_diagnostic.py','paired_noise.py']:(work/name).write_bytes((TEMPLATE/name).read_bytes())
    repair=(TEMPLATE/'repair_checks.py').read_text().replace('CONTROL=WORK.parent','CONTROL=WORK.parent.parent');(work/'repair_checks.py').write_text(repair)
    pipeline=(TEMPLATE/'pipeline.py').read_text().replace('CONTROL=WORK.parent','CONTROL=WORK.parent.parent')
    pipeline,count=re.subn(r"^CID='cand_[0-9a-f]{12}'$",f"CID='{cid}'",pipeline,flags=re.MULTILINE)
    if count!=1:raise ValueError('Pipeline identity mismatch')
    ast.parse(pipeline);(work/'pipeline.py').write_text(pipeline)
    (work/'prepared-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob('*.py')},indent=2,sort_keys=True)+'\n')
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
    pool=[(2,), (4,)]
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
            update(status='continuous_numeric_search',active_job={'label':'com.kernelmax.search.kunroll032.20261027',
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
            if outcome['status']=='no_operator_result':
                state['status']='stopped_pipeline_failure';save();break
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
               next_action='Immediately inspect native-k-unroll-032 state/trial results. Shortlists need independent full-model evidence. If pool exhausted or failed, identify the next justified action rather than rerolling old snapshots. Preserve all samples; no commits. Stop at1PM Pacific.')


if __name__=='__main__':
    main()
