"""Prepare executable inputs and a disabled activation example; never run a model."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE=Path('/Users/alandiaz/Downloads/KernelMaxxxing')
ROOT=BASE/'kernelmax-first-loop'
OLD=BASE/'work/decode-r2-resumed-20261103'
SPEC=BASE/'work/resumable-confirmation-contract-20261104'
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(SPEC),str(ROOT)]
from contract_spec import validate
from load_confirmation_protocol import load_protocol
from evidence import encode,sha_bytes
from contract_bridge import validate_bridge
from model_relocation import validate_relocation
from evaluator.schemas import load_contract
from scripts.loop_control import controller_sources,load_authorization
from scripts.run_reviewed_candidate import native_runtime
from evaluator.checkpoint_verification import validate_receipt


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write(path,value):
    if path.exists(): raise RuntimeError('Refuse overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


if (WORK/'runtime-build.json').exists() or (WORK/'session.json').exists():
    raise RuntimeError('Fresh executable preparation only')
validate(json.loads((SPEC/'protocol.json').read_text()))
protocol=load_protocol(SPEC/'protocol.json',WORK)
assessment={'protocol_sha256':sha_bytes(encode(protocol))}
prepared=json.loads((SPEC/'contract-bundle.json').read_text())
for path,expected in prepared['source_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Reviewed contract source changed: '+path)
old_manifest=json.loads((OLD/'prelaunch-manifest.json').read_text())
for path,expected in old_manifest['immutable_input_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Historical frozen input changed: '+path)
for name,source in (('contract.yaml',SPEC/'base-contract.yaml'),('reference-bridge.json',SPEC/'reference-bridge.json'),
        ('recovery-parent.json',OLD/'recovery-parent.json'),('model-relocation.json',OLD/'model-relocation.json')):
    if (WORK/name).exists(): raise RuntimeError('Fresh input copy only')
    shutil.copy2(source,WORK/name)
contract=load_contract(WORK/'contract.yaml')
reference=BASE/'work/checkpoint-ladder-20261030/original-reference.json'
bridge=validate_bridge(WORK/'reference-bridge.json',WORK/'contract.yaml',reference,
    candidate_sha256=protocol['candidate_sha256'])
relocation=validate_relocation(WORK/'model-relocation.json')
old_frozen=json.loads((OLD/'slot3-frozen.json').read_text())
destination=WORK/'frozen'/protocol['candidate_id']
shutil.copytree(old_frozen['directory'],destination)
frozen={**old_frozen,'directory':str(destination)}
for name,expected in frozen['file_sha256'].items():
    if sha(destination/name)!=expected: raise RuntimeError('Candidate math changed')
write(WORK/'slot3-frozen.json',frozen)
old_plan=json.loads((OLD/'plan.json').read_text())
plan={key:old_plan[key] for key in ('round_id','candidate_cap','existing_candidate_ids','previous_round_candidates',
    'quiet_preparation','qualification_preparation','desktop_workload','screen')}
plan.update(selected_candidate_id=protocol['candidate_id'],selected_candidate_sha256=protocol['candidate_sha256'],
    allowed_modes=['confirm'],contract_sha256=contract.sha256,
    confirmation={'before_each_request_idle_seconds':60,'max_request_or_trace_starts':240,'max_wall_seconds':21600},
    resident_worker_max_wall_seconds=21600,resumable_protocol=str(SPEC/'protocol.json'),
    resumable_protocol_sha256=assessment['protocol_sha256'],acceptance_enabled=False,
    timing_authorized=False,new_candidate_ids_allowed_in_this_worker=False,
    overall_goal='Produce a 4096 receipt for the separately approved per-context audit.',
    original_reference={'path':str(reference),'sha256':sha(reference)})
write(WORK/'plan.json',plan)
session=json.loads((OLD/'session.json').read_text())
session.update(contract_sha256=contract.sha256,protected_sources=controller_sources(),
    authorization_basis='Standing user goal authorizes controller integration and correctness preparation. RuntimeSession separately requires specific timing activation; this session file alone does not authorize a model run.',
    timing_authorized=False,confirmation_attempt_cap=1,allowed_modes=['confirm'])
write(WORK/'session.json',session);load_authorization(WORK/'session.json',contract)
write(WORK/'round-authorization.json',{'schema':1,'scope':'controller_preparation_only',
    'timing_authorized':False,'contract_sha256':contract.sha256,'protocol_sha256':assessment['protocol_sha256'],
    'new_candidate_allocation':False,'historical_timing_imported':False})
write(WORK/'commands/001.json',{'mode':'confirm','frozen_info':str(WORK/'slot3-frozen.json')})
runtime=native_runtime(Path(session['model_path']),contract)
receipt=json.loads(Path(session['checkpoint_cache']).read_text())
validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
if dict(runtime.model_files)!=relocation['model_files']: raise RuntimeError('Runtime manifest differs from original pinned bytes')
inputs=[path for path in WORK.iterdir() if path.is_file() and path.suffix in ('.py','.json','.yaml')]
inputs+=[WORK/'commands/001.json',reference,SPEC/'contract-bundle.json',SPEC/'protocol.json']
source_map={str(path):sha(path) for path in inputs}
source_map.update(controller_sources());source_map.update(bridge['sources']);source_map.update(relocation['sources'])
source_map.update({str(destination/name):expected for name,expected in frozen['file_sha256'].items()})
metadata_path=BASE/'work/model-internal-20261102/runtime-metadata-proof.json'
metadata=json.loads(metadata_path.read_text());source_map[str(metadata_path)]=sha(metadata_path)
for path,entry in metadata['metadata_files'].items():
    if sha(path)!=entry['sha256']: raise RuntimeError('Pinned download metadata changed')
    source_map[path]=entry['sha256']
previous_state=json.loads((OLD/'resident-state.json').read_text())
for path,expected in previous_state['protected_before'].items():
    if sha(path)!=expected: raise RuntimeError('Pinned reference runtime/source changed: '+path)
    source_map[path]=expected
identities={'source_sha256':source_map,'original_reference_path':str(reference),
    'original_reference_sha256':sha(reference),'candidate_sha256':protocol['candidate_sha256'],
    'candidate_metadata_path':str(WORK/'slot3-frozen.json'),
    'candidate_metadata_sha256':sha(WORK/'slot3-frozen.json'),
    'checkpoint_receipt_path':session['checkpoint_cache'],'checkpoint_receipt_sha256':sha(session['checkpoint_cache']),
    'original_checkpoint_receipt_path':str(BASE/'work/optimization-loop/checkpoint-verification.json')}
write(WORK/'runtime-build.json',{'schema':1,'status':'prepared_inactive','timing_authorized':False,
    'protocol_sha256':assessment['protocol_sha256'],'base_contract_sha256':contract.sha256,
    'source_sha256':source_map,'frozen_identities':identities,
    'native_generation_proof_sha256':sha(WORK/'native-generation.json'),
    'native_runtime_preflight':'passed_without_model_import_or_inference',
    'candidate_file_count':len(frozen['file_sha256'])})
write(WORK/'timing-authorization.example.json',{'schema':1,'authorized':False,
    'protocol_sha256':assessment['protocol_sha256'],'prepared_bundle_sha256':sha(SPEC/'contract-bundle.json'),
    'runtime_build_sha256':sha(WORK/'runtime-build.json'),'frozen_identities':identities,
    'user_timing_authorization':'','operator':'','measurement_basis':'','statistical_failure_handling':'',
    'unchanged_statistical_reroll':False,
    'note':'Disabled example only. No timing-authorization.json is created. Do not turn an unchanged failed-control comparison into an automatic reroll.'})
print(json.dumps({'status':'prepared_inactive','runtime_build_sha256':sha(WORK/'runtime-build.json'),
    'frozen_source_count':len(source_map),'candidate_files':len(frozen['file_sha256']),
    'timing_authorized':False,'model_or_gpu_run_started':False},indent=2))
