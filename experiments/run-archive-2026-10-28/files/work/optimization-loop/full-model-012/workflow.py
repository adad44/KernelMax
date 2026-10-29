"""Full-model result followed immediately by untried numeric candidates."""
import json
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
deadline=json.loads((CONTROL/'search-20261027.json').read_text())['deadline_unix']
record={'status':'running_full_model','started_unix':time.time()}
path=WORK/'workflow.json'
if path.exists():raise SystemExit('Retain existing workflow')
def save():path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
save()
try:
    child=subprocess.run([sys.executable,'-B','-u',str(WORK/'run.py')],
                         timeout=max(1,deadline-time.time()-5))
    record['full_model_returncode']=child.returncode;save()
    result=json.loads((WORK/'state.json').read_text())
    guards=all(result.get(k) is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged'))
    if not guards:raise ValueError('Integrity failure requires investigation; numeric search blocked')
    if (WORK/'STOP').exists() or time.time()+60>=deadline:
        record['status']='deadline_or_stop'
    else:
        # Reap the full-model process before the next native load: never overlap.
        record['status']='resuming_untried_numeric_candidates';save()
        state_path=CONTROL/'search-20261027.json';s=json.loads(state_path.read_text())
        s['full_model_diagnostic']={'candidate_id':result['candidate_id'],'state':str(WORK/'state.json'),
            'status':result['status'],'full_model_speedup_established':result.get('full_model_speedup_established',False)}
        s['active_job']=None
        s['continuous_controller'].update(state=str(CONTROL/'sweep-013/state.json'),stop_file=str(CONTROL/'sweep-013/STOP'),label='com.kernelmax.search.fullmodel012.20261027')
        state_path.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
        child=subprocess.run([sys.executable,'-B','-u',str(CONTROL/'sweep-013/search.py')],
                             timeout=max(1,deadline-time.time()-5))
        record['resumed_search_returncode']=child.returncode
        record['status']='resumed_search_completed'
except BaseException as error:
    record['status']='failed';record['error']=f'{type(error).__name__}: {error}';raise
finally:
    record['ended_unix']=time.time();save()
