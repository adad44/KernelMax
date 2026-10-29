"""One bounded native correctness + real-input operator candidate run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORK=Path(__file__).resolve().parent
ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')


def main():
    path=WORK/'pipeline.json'
    if path.exists():
        raise ValueError('Pipeline already ran; retain attempt')
    state={'status':'running','pid':os.getpid(),'started_unix':time.time(),
           'execution_class':'temporary launchd Interactive','stages':[],'errors':[]}
    save=lambda:path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
    save()
    try:
        a=json.loads((WORK/'approval.json').read_text())
        tiny=json.loads((WORK/'tiny-native-checks.json').read_text())
        if tiny['status']!='passed_synthetic_only' or not all(r['passed'] for r in tiny['checks']):
            raise ValueError('Tiny checks have not passed')
        expected=json.loads((WORK/'prepared-sha256.json').read_text())
        for name,sha in expected.items():
            if hashlib.sha256((WORK/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Prepared controller changed: '+name)
        stages=[('native-correctness', [sys.executable,'-B','-u',str(ROOT/'scripts/run_reviewed_candidate.py'),
                                      '--candidate',str(ROOT/'candidates'/a['candidate_id']),
                                      '--work-dir',str(WORK/'native-correctness'),
                                      '--native-model','/Volumes/BankOfSouls/models/gpt-oss-20b',
                                      '--approved-sha256',a['snapshot_sha256'],'--timeout-seconds','1200'])]
        for name,command in stages:
            print('Starting '+name,flush=True)
            with (WORK/(name+'-summary.json')).open('w') as out,(WORK/(name+'-stderr.log')).open('w') as err:
                child=subprocess.run(command,stdout=out,stderr=err,timeout=2600)
            state['stages'].append({'stage':name,'returncode':child.returncode});save()
            if child.returncode:
                raise RuntimeError(name+' failed; timing blocked')
        reports=list((WORK/'native-correctness/artifacts').glob('*/report.json'))
        if len(reports)!=1:
            raise ValueError('Native report discovery ambiguous')
        state['native_report']=str(reports[0]);save()
        print('Starting real-input paired timing',flush=True)
        with (WORK/'operator-diagnostic.log').open('w') as out,(WORK/'operator-diagnostic-stderr.log').open('w') as err:
            child=subprocess.run([sys.executable,'-B','-u',str(WORK/'operator_diagnostic.py'),str(reports[0])],
                                 stdout=out,stderr=err,timeout=900)
        state['stages'].append({'stage':'operator-diagnostic','returncode':child.returncode});save()
        if child.returncode:
            raise RuntimeError('Operator diagnostic failed; retain all evidence')
        state['status']='completed_diagnostic'
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        state['ended_unix']=time.time();save()


if __name__=='__main__':
    main()
