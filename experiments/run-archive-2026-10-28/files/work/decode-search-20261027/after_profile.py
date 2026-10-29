"""One-off sequential supervisor; never overlaps model benchmark workers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORK=Path(__file__).resolve().parent
HERE=WORK/'timeline'
PYTHON=str(WORK.parent.parent/'runtime/gpt-oss/.venv/bin/python')
OUTPUT=WORK/'after-profile-state.json'


def write(value):
    temp=OUTPUT.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(OUTPUT)


def main():
    if OUTPUT.exists():raise ValueError('Fresh supervisor only')
    sources=json.loads((HERE/'prepared-sha256.json').read_text())
    state={'pid':os.getpid(),'phase':'waiting_profiler','started_unix':time.time(),
        'profiler_pid':71655,'profiler_birth':'Tue Oct 27 18:10:46 2026',
        'source_snapshot':sources,'children':[],'errors':[]}
    def check_sources():
        for p,h in sources.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Prepared follow-up changed: '+p)
    write(state)
    try:
        check_sources();deadline=time.monotonic()+14400
        while True:
            result=subprocess.run(['ps','-p','71655','-o','lstart=,command='],capture_output=True,text=True)
            still_profiler=(result.returncode==0 and 'Tue Oct 27 18:10:46 2026' in result.stdout
                            and 'cooled-profile/profile.py' in result.stdout)
            if not still_profiler:break
            if time.monotonic()>deadline:raise TimeoutError('Four-hour profiler observation cap; no duplicate run')
            time.sleep(5)
        observed=json.loads((WORK/'cooled-profile/profile-state.json').read_text())
        state['profiler_terminal_status']=observed['status'];write(state)
        if (observed['status']!='completed_profile' or not observed.get('protected_unchanged')
            or not observed.get('checkpoint_unchanged') or not observed.get('aa',{}).get('passed')):
            raise RuntimeError('Profiler did not complete validly; no follow-up GPU process started')
        check_sources()
        for phase,script,stdout,stderr in (
            ('observer_validation',HERE/'validate.py',HERE/'validation-stdout.log',HERE/'validation-stderr.log'),
            ('timeline_and_candidate',HERE/'run.py',HERE/'stdout.log',HERE/'stderr.log')):
            state['phase']=phase;write(state);check_sources()
            with stdout.open('w') as out,stderr.open('w') as err:
                process=subprocess.Popen([PYTHON,'-B','-u',str(script)],cwd=str(WORK.parent.parent),stdout=out,stderr=err)
                record={'phase':phase,'pid':process.pid,'script':str(script),'started_unix':time.time()}
                state['children'].append(record);write(state)
                code=process.wait();record.update(returncode=code,finished_unix=time.time());write(state)
                if code:raise RuntimeError(phase+' exited '+str(code))
        state['phase']='completed'
    except BaseException:
        state['phase']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        state['finished_unix']=time.time();write(state)


if __name__=='__main__':main()
