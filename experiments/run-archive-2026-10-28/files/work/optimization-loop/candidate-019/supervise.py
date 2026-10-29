"""Bounded correctness-only run, with tracked descendant cleanup on every exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
SEARCH=CONTROL/'search-20261027.json'
LABEL='com.kernelmax.search.candidate019.20261027'
state=json.loads(SEARCH.read_text())
deadline=min(time.time()+1800,state['deadline_unix'])
if deadline-time.time()<30:
    raise SystemExit('Insufficient time before deadline')
process_path=WORK/'process.json'
if process_path.exists():
    raise SystemExit('Retain existing attempt')
record={'status':'starting','kind':'sequential_candidate_pipeline','candidate_id':'cand_d5987f75f173',
        'supervisor_pid':os.getpid(),'started_unix':time.time(),'deadline_unix':deadline,
        'command':[sys.executable,'-B','-u',str(WORK/'pipeline.py')]}
def save():
    process_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
def processes():
    lines=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines()
    rows={}
    for line in lines:
        fields=line.split(None,7)
        if len(fields)==8:
            rows[int(fields[0])]={'ppid':int(fields[1]),'start':' '.join(fields[2:7]),'command':fields[7]}
    return rows
def observe_tree(root_pid):
    rows=processes();included={root_pid}
    changed=True
    while changed:
        changed=False
        for pid,row in rows.items():
            if row['ppid'] in included and pid not in included:
                included.add(pid);changed=True
    for pid in included:
        if pid in rows:
            observed[pid]=rows[pid]
save();observed={};child=None
assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
try:
    with (WORK/'pipeline-stdout.log').open('w') as out,(WORK/'pipeline-stderr.log').open('w') as err:
        child=subprocess.Popen(record['command'],stdout=out,stderr=err,start_new_session=True,cwd=ROOT)
        record.update(status='running',worker_pid=child.pid);save()
        state['active_job']={'label':LABEL,'directory':str(WORK),'supervisor_pid':os.getpid(),'worker_pid':child.pid,'deadline_unix':deadline}
        state['status']='running_candidate_correctness';state['current_candidate']['stage']='native_correctness_running'
        SEARCH.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
        while child.poll() is None:
            observe_tree(child.pid)
            if time.time()>=deadline:
                record['status']='timeout';break
            time.sleep(.5)
        if child.poll() is not None:
            record['returncode']=child.returncode
            record['status']='completed' if child.returncode==0 else 'failed'
finally:
    if child is not None:
        observe_tree(child.pid)
        current=processes()
        for pid,old in reversed(list(observed.items())):
            if pid in current and current[pid]['start']==old['start'] and current[pid]['command']==old['command']:
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
        child.wait()
    assertion.terminate();assertion.wait()
    record['tracked_processes']={str(pid):row for pid,row in observed.items()}
    record['ended_unix']=time.time();save()
    state=json.loads(SEARCH.read_text());state['active_job']=None
    state['attempts'].append({'directory':str(WORK),**record})
    state['status']='candidate_pipeline_ready_for_analysis'
    state['current_candidate']['stage']=record['status']
    state['next_action']='Immediately analyze candidate019 native/tiny/exact-boundary/timing stages. Native32-lane order is the numerical hypothesis; no custom-reduction sweep until exactness is demonstrated. Park nonwins, retain failures, and only full-model-validate a valid operator winner. No schedule wait after an actionable result.'
    SEARCH.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
