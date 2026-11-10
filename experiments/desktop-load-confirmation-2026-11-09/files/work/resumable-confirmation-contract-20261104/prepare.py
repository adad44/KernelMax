"""Freeze a reviewable specification only; no model import, execution or activation."""
import hashlib
import json
from pathlib import Path
import sys

BASE=Path('/Users/alandiaz/Downloads/KernelMaxxxing')
ROOT=BASE/'kernelmax-first-loop'
PREVIOUS=BASE/'work/decode-r2-resumed-20261103'
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(ROOT),str(PREVIOUS)]
import yaml
from contract_spec import validate
from contract_bridge import validate_bridge
from evaluator.schemas import load_contract


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write(path,value):
    if path.exists(): raise RuntimeError('Refuse overwrite: '+str(path))
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


if (WORK/'contract-bundle.json').exists() or (WORK/'base-contract.yaml').exists():
    raise RuntimeError('New prepared contract only; never overwrite a frozen bundle')
protocol=json.loads((WORK/'protocol.json').read_text())
assessment=validate(protocol)
prior=json.loads((PREVIOUS/'prelaunch-manifest.json').read_text())
for path,expected in prior['immutable_input_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Prior immutable input changed: '+path)
reference=BASE/'work/checkpoint-ladder-20261030/original-reference.json'
if sha(reference)!=protocol['original_reference_sha256']:
    raise RuntimeError('Original reference changed')
reference_data=json.loads(reference.read_text())
for path,expected in reference_data['archived_source_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Original source archive changed')
catalog=BASE/'work/checkpoint-ladder-20261030/catalog.json'
original_catalog_sha=sha(catalog)
data=yaml.safe_load((PREVIOUS/'contract.yaml').read_text())
data['experiment_id']='gpt-oss-20b-m4-resumable-prepared-20261104'
(WORK/'base-contract.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
contract=load_contract(WORK/'base-contract.yaml')
bridge=json.loads((PREVIOUS/'reference-bridge.json').read_text())
bridge['desktop_contract_sha256']=contract.sha256
write(WORK/'reference-bridge.json',bridge)
bridge_result=validate_bridge(WORK/'reference-bridge.json',WORK/'base-contract.yaml',reference,
    candidate_sha256=protocol['candidate_sha256'])
frozen=json.loads((PREVIOUS/'slot3-frozen.json').read_text())
if frozen['sha256']!=protocol['candidate_sha256']:
    raise RuntimeError('Candidate bundle changed')
for name,expected in frozen['file_sha256'].items():
    if sha(Path(frozen['directory'])/name)!=expected: raise RuntimeError('Candidate source changed')

review='''# Prepared resumable confirmation contract

Status: prepared for review, inactive. The user authorized contract preparation.
No timing allocation, worker launch, new candidate, or historical promotion is
authorized by this bundle. The existing journal is an offline prototype.

The frozen numerical base is `base-contract.yaml`; `protocol.json` is the
controller-owned execution/evidence overlay. Changing either identity requires
a new contract. `reference-bridge.json` proves that the numerical base differs
from the archived original only in experiment identity and the approved desktop
paging policy. The original reference, source archive and catalog are untouched.

## Bounds and resume accounting

The proposal permits at most1,000 model-call starts,3worker epochs (initial plus
2explicit user resumes),20cumulative active hours,20hours per worker and7calendar
days including pauses. Baseline expected starts remain706. Each resume reserves
100starts for a complete fresh A/A group with warmups/primers,20for symmetric
candidate-block warmups/primers,2for an8outputnative/candidate handoff, and4for an
abandoned pair. Total planned envelope is958starts, leaving42within the cap.
The wall cap is finite; this arithmetic is not a guarantee of completion.

Compared with the previous single-worker contract, the global start cap changes
from800to1,000 and active wall from16to20hours. The resident20hour cap stays.
All worker time is charged, including loading, recovery waits, serialization and
cleanup. Only a fully closed, verified user-pause interval is excluded from
active time; the calendar cap still includes it. Budgets never reset on resume.

## Evidence and retention

Twenty complete same-epoch pairs are required per block, with the independent
second512block. A resumed context needs a fresh full control group. Use every
eligible complete pair, preserve all unmatched and failed rows, never pair
across epochs, and never retry a completed failed control through a new epoch.
The maximum A/A uncertainty across all complete controls for the context is used
for its gain threshold. The existing median/bootstrap95% and5% rules stay.

A context checkpoint can finalize after all3full numerical contexts, its required
blocks, valid controls/rows and final contributing-epoch provenance guards pass.
An ongoing worker must explicitly seal the context at a GPU-drained boundary,
verify raw evidence and all identity/storage guards, and commit an immutable
journal-prefix/guard receipt before certification. Mutable live state is not
enough. Closed earlier epochs also require their own final guard receipts.
This allows useful verified gains below5% to survive unrelated later failures.
The full goal still requires all3contexts/all4blocks under this contract. Any
failed context control makes that context inconclusive. Hard stops are terminal
for the pass. Historical evidence is retained under its original policy, without
importing old timing or retroactively changing a failed/incomplete status.

The original catalog remains unchanged. A new resumable-policy catalog would be
created only at activation, linked to the same immutable original reference.
Acceptance remains disabled and held-out task quality remains planned.

## Implementation review before any timing

The production controller must reserve both primer/measured-call slots atomically
before the primer, keeping one GPU call at a time. No journal fsync or artifact
serialization may interpose between primer and measurement. Unused reserved
slots stay charged and actual calls are reported separately. It must implement
atomic raw-artifact publication before journaling hashes, orphan/start accounting, cumulative wall/watchdog limits,
GPU-drained pause handling, source/model validation, fresh controls and a verifier
for context certificates. Pending buffers and partial pairs cannot be invented
or selected by speed. A torn journal or unresolved crash fails closed; recovery
never silently truncates evidence. The tested journal alone is insufficient.

Activation also needs a specific timing basis/operator authorization. A new
resumable storage format alone does not justify rerunning the latest unchanged
512statisticalfailure. This prepared bundle neither launches a model nor removes
that requirement. The current task is contract preparation and review only.
'''
(WORK/'REVIEW.md').write_text(review)
inputs=[WORK/name for name in ('protocol.json','contract_spec.py','test_contract_spec.py',
    'prepare.py','base-contract.yaml','reference-bridge.json','REVIEW.md')]
prototype=BASE/'work/resumable-confirmation-draft-20261104'
inputs.extend(prototype/name for name in ('journal.py','test_journal.py','validation/result.json'))
bundle={'schema':1,'status':'prepared_inactive','timing_authorized':False,
    'user_authorization':'Prepare resumable confirmation contract',
    'base_contract_sha256':contract.sha256,**assessment,
    'original_reference_path':str(reference),'original_reference_sha256':sha(reference),
    'original_catalog_path':str(catalog),'original_catalog_sha256':original_catalog_sha,
    'reference_bridge_sha256':bridge_result['bridge_sha256'],
    'candidate_id':frozen['candidate_id'],'candidate_sha256':frozen['sha256'],
    'frozen_candidate_metadata_path':str(PREVIOUS/'slot3-frozen.json'),
    'frozen_candidate_metadata_sha256':sha(PREVIOUS/'slot3-frozen.json'),
    'original_policy_preserved':True,'historical_samples_imported':False,
    'integration_status':'specification_prepared_production_controller_not_implemented',
    'required_before_activation':protocol['activation_requirements'],
    'source_sha256':{str(path):sha(path) for path in inputs}}
if sha(catalog)!=original_catalog_sha: raise RuntimeError('Catalog changed during preparation')
write(WORK/'contract-bundle.json',bundle)
print(json.dumps({'status':bundle['status'],'base_contract_sha256':contract.sha256,
    'protocol_sha256':assessment['protocol_sha256'],'bundle_sha256':sha(WORK/'contract-bundle.json'),
    'expected_starts_with_two_resumes':assessment['expected_starts_with_two_resumes'],
    'timing_authorized':False},indent=2))
