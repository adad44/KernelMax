"""Immediate native-order candidate transitions and conditional full-model checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent
ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
SEARCH=CONTROL/'search-20261027.json'
DEADLINE=json.loads(SEARCH.read_text())['deadline_unix']
STATE=HERE/'state.json'
if STATE.exists():raise SystemExit('Retain existing coordinator')
state={'status':'waiting_current_operator','pid':os.getpid(),'started_unix':time.time(),
       'deadline_unix':DEADLINE,'outcomes':[],'full_model_runs':[]}


def save():
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');tmp.replace(STATE)


def update(**changes):
    s=json.loads(SEARCH.read_text());s.update(changes)
    tmp=SEARCH.with_suffix('.followup-tmp');tmp.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');tmp.replace(SEARCH)


def tree():
    rows={}
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
        f=line.split(None,7)
        if len(f)==8:rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
    return rows


def run(command,directory):
    known={};child=None
    try:
        with (directory/'coordinator-stdout.log').open('w') as out,(directory/'coordinator-stderr.log').open('w') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
            while child.poll() is None:
                rows=tree();included={child.pid};changed=True
                while changed:
                    changed=False
                    for pid,row in rows.items():
                        if row['ppid'] in included and pid not in included:included.add(pid);changed=True
                for pid in included:
                    if pid in rows:known[pid]=rows[pid]
                if time.time()>=DEADLINE or (HERE/'STOP').exists():break
                time.sleep(.5)
    finally:
        rows=tree()
        for pid,old in reversed(list(known.items())):
            if pid in rows and rows[pid]['start']==old['start'] and rows[pid]['command']==old['command']:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if child is not None:child.wait()
    return child.returncode


def audit(work,cid):
    p=work/'operator-diagnostic.json'
    if not p.exists():return {'status':'no_operator_result','candidate_id':cid,'directory':str(work),'shortlisted':False}
    d=json.loads(p.read_text())
    if d['candidate_id']!=cid:raise ValueError('Candidate identity mismatch')
    blocks={(r['layer'],r['phase'],r['pair']):r['blocks'] for r in d.get('microblocks',[])}
    for r in d['samples']:
        matching=[b for b in blocks[r['layer'],r['phase'],r['pair']] if b['lane']==r['lane']]
        if sum(b['calls'] for b in matching)!=r['calls'] or r['calls']!=1024:raise ValueError('Call-count mismatch')
        if abs(sum(b['elapsed_ms'] for b in matching)-r['elapsed_ms'])>1e-7:raise ValueError('Elapsed aggregation mismatch')
    guards=all(d.get(k) is True for k in ('snapshot_unchanged','checkpoint_identity_unchanged','protected_sources_unchanged'))
    if not guards:raise ValueError('Integrity failure; successor work blocked')
    repair=json.loads((work/'repair-checks.json').read_text())
    positive=(d['status']=='completed_diagnostic' and set(d['summaries'])=={'0','12','23'}
              and not d.get('environment_issues') and all(r['passed'] for r in d['aa_screens'].values())
              and repair['status']=='passed_exact_repair_checks'
              and all(r['descriptive_bootstrap_95pct'][0]>1 for r in d['summaries'].values()))
    result={'candidate_id':cid,'directory':str(work),'status':d['status'],'shortlisted':positive,
            'summaries':d['summaries'],'all_actual_counts_verified':True,'guards_passed':guards,
            'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'full_model_speedup_established':False}
    (work/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def full_model(selected,index):
    work=CONTROL/f'full-model-022-{index:02d}';work.mkdir()
    base=CONTROL/'full-model-012'
    for name in ('run.py','plan.json'):(work/name).write_bytes((base/name).read_bytes())
    (work/'selected.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
    code=(base/'supervise.py').read_text().replace('cand_b64853d31601',selected['candidate_id'])
    code=code.replace("str(WORK/'workflow.py')","str(WORK/'run.py')")
    code=code.replace('com.kernelmax.search.fullmodel012.20261027','com.kernelmax.search.nativeorder021.20261027')
    code=code.replace('full-model-012',work.name)
    ast.parse(code);(work/'supervise.py').write_text(code)
    state['status']='running_independent_full_model';state['active_full_model']=str(work);save()
    update(status='running_independent_full_model',current_candidate={'candidate_id':selected['candidate_id'],'work':str(work),'stage':'full_model_validation'},
           active_job={'label':'com.kernelmax.search.nativeorder021.20261027','directory':str(HERE),'supervisor_pid':os.getpid(),'full_model_directory':str(work)})
    rc=run([sys.executable,'-B','-u',str(work/'supervise.py')],work)
    d=json.loads((work/'state.json').read_text())
    result={'candidate_id':selected['candidate_id'],'directory':str(work),'returncode':rc,'status':d['status'],
            'full_model_speedup_established':d.get('full_model_speedup_established',False)}
    state['full_model_runs'].append(result);save()
    if time.time()>=DEADLINE or (HERE/'STOP').exists():
        return result
    if not all(d.get(k) is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):
        raise ValueError('Full-model integrity failure')
    if d['full_model_correctness'] and not all(r['passed'] for r in d['full_model_correctness']):
        raise ValueError('Native-order family full-model numerical mismatch; diagnose before more variants')
    return result


def main():
    save();assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        initial=CONTROL/'candidate-019'
        while time.time()<DEADLINE and not (HERE/'STOP').exists():
            process=json.loads((initial/'process.json').read_text())
            if process['status']!='running':
                row=tree().get(process['supervisor_pid'])
                if row is None or str(initial/'supervise.py') not in row['command']:break
            time.sleep(.5)
        if time.time()>=DEADLINE or (HERE/'STOP').exists():return
        outcome=audit(initial,'cand_d5987f75f173');state['outcomes'].append(outcome);save()
        index=1
        if outcome['shortlisted']:full_model(outcome,index);index+=1
        remaining=[(4,'half'),(1,'half'),(8,'half'),(4,'lut'),(1,'lut'),(8,'lut')]
        while remaining and time.time()+60<DEADLINE and not (HERE/'STOP').exists():
            queue=CONTROL/'native-order-020' if index==1 else CONTROL/f'native-order-resume-{index:02d}'
            if queue!=CONTROL/'native-order-020':
                queue.mkdir()
                for name in ('paired_noise.py','operator-template.py','search.py'):
                    (queue/name).write_bytes((CONTROL/'native-order-020'/name).read_bytes())
            code=(queue/'search.py').read_text()
            start=code.index('    pool=');end=code.index("    state['predeclared_pool']",start)
            code=code[:start]+f'    pool={remaining!r}\n'+code[end:]
            ast.parse(code);(queue/'search.py').write_text(code)
            state['status']='running_untried_native_order_geometry';state['active_queue']=str(queue);save()
            update(continuous_controller={'state':str(queue/'state.json'),'stop_file':str(HERE/'STOP'),'label':'com.kernelmax.search.nativeorder021.20261027'},
                   active_job={'label':'com.kernelmax.search.nativeorder021.20261027','directory':str(HERE),'supervisor_pid':os.getpid(),'queue_directory':str(queue)})
            run([sys.executable,'-B','-u',str(queue/'search.py')],queue)
            data=json.loads((queue/'state.json').read_text())
            attempted={tuple(r['config']) for r in data['trials']}
            remaining=[config for config in remaining if config not in attempted]
            state['outcomes'].extend(data['trials']);save()
            if data['shortlist']:
                full_model(data['shortlist'][0],index);index+=1
            elif data['status']!='numeric_pool_exhausted':
                raise ValueError('Geometry queue stopped: '+data['status'])
        state['status']='deadline_or_stop_reached' if time.time()>=DEADLINE or (HERE/'STOP').exists() else 'native_order_pool_exhausted'
    except BaseException as error:
        state['status']='deadline_or_stop_reached' if time.time()>=DEADLINE or (HERE/'STOP').exists() else 'needs_investigation'
        state['error']=f'{type(error).__name__}: {error}'
        if state['status']=='needs_investigation':raise
    finally:
        assertion.terminate();assertion.wait();state['ended_unix']=time.time();save()
        update(active_job=None,status=state['status'],followup_controller=str(STATE),
               next_action='Immediately inspect followup021/state.json and its active/result directories. Do not wait for heartbeat after actionable results. Preserve numerical and timing gates, no commits, stop at1PM.')


if __name__=='__main__':main()
