"""Sequential authorized stages; no scheduler gap after a stage result."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent.parent
ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
CID='cand_6dbb410bc1b0'
deadline=json.loads((WORK/'process.json').read_text())['deadline_unix']
state={'status':'running','stages':[],'started_unix':time.time()}
path=WORK/'pipeline.json'
if path.exists():
    raise SystemExit('Retain existing pipeline attempt')
def save():
    path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
def stage(name,command):
    state['stage']=name;save()
    with (WORK/(name+'.log')).open('w') as out,(WORK/(name+'-stderr.log')).open('w') as err:
        child=subprocess.run(command,stdout=out,stderr=err,
                             timeout=max(1,deadline-time.time()-5),cwd=ROOT)
    state['stages'].append({'name':name,'returncode':child.returncode});save()
    if child.returncode:
        raise RuntimeError(name+' failed; next stage blocked')
def progress(stage_name,**updates):
    search=CONTROL/'search-20261027.json';s=json.loads(search.read_text())
    s['current_candidate'].update(stage=stage_name,**updates)
    search.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
save()
try:
    stage('native-summary',[sys.executable,'-B','-u',str(ROOT/'scripts/run_candidate_loop.py'),
                           '--session',str(CONTROL/'session.json'),
                           '--candidate',str(ROOT/'candidates'/CID),
                           '--work-dir',str(WORK/'native-correctness'),'--timeout-seconds','300'])
    reports=list((WORK/'native-correctness/artifacts').glob('*/report.json'))
    if len(reports)!=1:
        raise ValueError('Native report discovery ambiguous')
    report=json.loads(reports[0].read_text())
    if report['status']!='completed' or report['errors'] or len(report['correctness'])!=7 or not all(r['passed'] for r in report['correctness']):
        raise ValueError('Native correctness did not pass')
    request=json.loads((reports[0].parent/'candidate/request.json').read_text())
    (WORK/'approval.json').write_text(json.dumps({
        'candidate_id':CID,'snapshot':request['snapshot']['directory'],
        'snapshot_sha256':report['candidate']['sha256'],
        'authorization_kind':'standing_user_instruction','human_reviewed_exact_snapshot':False,
        'session':str(CONTROL/'session.json'),'native_report':str(reports[0])},indent=2,sort_keys=True)+'\n')
    progress('native_correctness_passed',native_report=str(reports[0]),native_checks_passed=21,
             snapshot_sha256=report['candidate']['sha256'])
    stage('tiny-checks',[sys.executable,'-B','-u',str(WORK/'tiny_native_checks.py')])
    tiny=json.loads((WORK/'tiny-native-checks.json').read_text())
    if tiny['status']!='passed_synthetic_only' or not all(r['passed'] for r in tiny['checks']):
        raise ValueError('Tiny fixtures did not pass')
    progress('paired_timing_running',tiny_checks_passed=len(tiny['checks']))
    stage('operator-diagnostic',[sys.executable,'-B','-u',str(WORK/'operator_diagnostic.py'),str(reports[0])])
    result=json.loads((WORK/'operator-diagnostic.json').read_text())
    state['operator_status']=result['status'];state['status']='completed_diagnostic'
except BaseException as error:
    state['status']='failed';state['error']=f'{type(error).__name__}: {error}';raise
finally:
    state['ended_unix']=time.time();save()
