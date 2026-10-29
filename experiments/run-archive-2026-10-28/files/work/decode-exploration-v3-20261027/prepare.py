"""Prepare a fresh controller-owned source/session snapshot; no model run."""
import ast
import hashlib
import json
from pathlib import Path
import plistlib
import sys
from datetime import datetime,timezone

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
from evaluator.schemas import load_contract
from scripts.loop_control import controller_sources,load_authorization

if (WORK/'session.json').exists():raise ValueError('Fresh session preparation only')
contract=load_contract(ROOT/'kernelmaxxing.yaml')
old=json.loads((WORK.parent/'decode-search-20261027/session.json').read_text())
session={**old,'contract_sha256':contract.sha256,'protected_sources':controller_sources(),
    'authorization_basis':'User explicitly authorizes schema3 bounded exploration retaining and flagging swap-affected samples, preserving completed work, phase-specific memory/swap-rate observations; confirmation requires fresh strict matched measurements. Complete-decode scheduling/fusion/exact memory scope remains authorized; no acceptance, merge or publication.',
    'prior_authorization':str(WORK.parent/'decode-search-20261027/session.json'),
    'recorded_utc':datetime.now(timezone.utc).isoformat()}
(WORK/'session.json').write_text(json.dumps(session,indent=2,sort_keys=True)+'\n')
load_authorization(WORK/'session.json',contract)
(WORK/'contract-snapshot.yaml').write_bytes((ROOT/'kernelmaxxing.yaml').read_bytes())
for p in (WORK/'profile.py',WORK/'smoke.py'):ast.parse(p.read_text())
job={'Label':'com.kernelmax.exploration.v3.20261027','ProgramArguments':[str(WORK.parent.parent/'runtime/gpt-oss/.venv/bin/python'),'-B','-u',str(WORK/'profile.py')],'WorkingDirectory':str(WORK.parent.parent),'ProcessType':'Interactive','RunAtLoad':False,'KeepAlive':False,'StandardOutPath':str(WORK/'stdout.log'),'StandardErrorPath':str(WORK/'stderr.log'),'EnvironmentVariables':{'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}}
(WORK/'job.plist').write_bytes(plistlib.dumps(job))
files=[WORK/'profile.py',WORK/'plan.json',WORK/'session.json',ROOT/'evaluator/measurement.py']
(WORK/'prepared-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2,sort_keys=True)+'\n')
print(json.dumps({'contract_sha256':contract.sha256,'schema':contract.schema_version,'mode':'exploration','prepared_not_started':True}))
