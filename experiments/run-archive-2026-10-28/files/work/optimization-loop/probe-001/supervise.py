"""Single bounded Interactive diagnostic job; no unrelated process control."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
search_path=CONTROL/'search-20261027.json'
state=json.loads(search_path.read_text())
deadline=state['deadline_unix']
remaining=min(1200,deadline-time.time())
if remaining<=0:
    raise SystemExit('Search deadline already passed')
record={'supervisor_pid':os.getpid(),'status':'running','started_unix':time.time(),
        'timeout_seconds':remaining,'deadline_unix':deadline}
path=WORK/'process.json'
if path.exists():
    raise SystemExit('Do not repeat completed or active attempt')
def save():
    path.write_text(json.dumps(record,indent=2)+'\n')
save()
child=None
assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
try:
    child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'probe.py')],start_new_session=True)
    record['worker_pid']=child.pid;save()
    state['active_job']={'label':'com.kernelmax.search.probe001.20261027','supervisor_pid':os.getpid(),
                         'worker_pid':child.pid,'directory':str(WORK),'hard_timeout_seconds':remaining}
    state['status']='running_native_bottleneck_probe'
    search_path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
    record['returncode']=child.wait(timeout=remaining)
    record['status']='completed' if child.returncode==0 else 'failed'
except subprocess.TimeoutExpired:
    record['status']='timeout'
finally:
    if child is not None:
        try:
            os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
    assertion.terminate();assertion.wait()
    record['ended_unix']=time.time();save()
    state=json.loads(search_path.read_text())
    state['active_job']=None
    state['attempts'].append({'kind':'native_operator_deletion_probe','directory':str(WORK),**record})
    state['status']='probe_ready_for_analysis'
    state['next_action']='Inspect probe-001/result.json and its A/A screen. If valid, rank distinct candidate hypotheses using isolated savings and explicit model-share uncertainty. If A/A failed, investigate a specific measurement intervention rather than rerolling unchanged.'
    search_path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
