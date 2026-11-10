"""Freeze the approved per-context design, leaving timing activation absent."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
BASE = WORK.parent.parent
OLD = BASE / 'work/resumable-boot-uuid-20261107'
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(name, before, after):
    path = WORK / name
    source = path.read_text()
    if source.count(before) != 1:
        raise RuntimeError('Unexpected input for scoped integration: ' + name)
    path.write_text(source.replace(before, after))


def write(name, value):
    with (WORK / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if (WORK / 'runtime-build.json').exists() or (WORK / 'evidence').exists():
    raise RuntimeError('Fresh preparation only')
if digest(OLD / 'runtime-build.json') != '30c4cbd2976bf59ea57b082a400f93488f70d7a1b0fd9b46c54f5a6e2c3eaadf':
    raise RuntimeError('Prior build identity changed')
source_map = json.loads((OLD / 'runtime-build.json').read_text())['source_sha256']
for path, expected in source_map.items():
    if digest(Path(path)) != expected:
        raise RuntimeError('Prior frozen source changed: ' + path)
for path in sorted(OLD.glob('*.py')):
    destination = WORK / path.name
    if destination.exists():
        raise RuntimeError('Preparation input already exists: ' + path.name)
    shutil.copy2(path, destination)
    if digest(destination) != source_map[str(path)]:
        raise RuntimeError('Executable copy changed')
shutil.copy2(OLD / 'native-generation.json', WORK / 'native-generation.json')

replace_once('runtime_integration.py', 'from certificates import verify_context',
             'from certificates import verify_context\nfrom load_confirmation_protocol import load_protocol')
replace_once('runtime_integration.py', "self.work=Path(work);self.protocol=json.loads((PREPARED/'protocol.json').read_text())",
             "self.work=Path(work);self.protocol=load_protocol(PREPARED/'protocol.json',self.work)")
replace_once('runtime_integration.py', 'for context in (512,2048,4096):',
             "for context in self.protocol['workload']['contexts']:")
replace_once('verify_exit.py', 'from runtime_integration import boot_identity,process_identity,PREPARED',
             'from runtime_integration import boot_identity,process_identity,PREPARED\nfrom load_confirmation_protocol import load_protocol')
replace_once('verify_exit.py', "protocol=json.loads((PREPARED/'protocol.json').read_text())",
             "protocol=load_protocol(PREPARED/'protocol.json',work)")
replace_once('verify_exit.py', "    complete=(last['reason']=='complete' and all_exits",
             "    scope={str(context) for context in protocol['workload']['contexts']}\n"
             "    scoped_complete=(last['reason']=='complete' and all_exits")
replace_once('verify_exit.py', "        and set(contexts)=={'512','2048','4096'})",
             "        and set(contexts)==scope)\n"
             "    complete=scoped_complete and scope=={'512','2048','4096'}")
replace_once('verify_exit.py', "        'complete_locked_workload':complete,'five_percent_improvement_verified':goal,",
             "        'completed_confirmation_scope':sorted(map(int,scope)) if scoped_complete else [],\n"
             "        'complete_locked_workload':complete,'five_percent_improvement_verified':goal,")
replace_once('prepare_runtime.py', 'from contract_spec import validate',
             'from contract_spec import validate\nfrom load_confirmation_protocol import load_protocol\n'
             'from evidence import encode,sha_bytes')
replace_once('prepare_runtime.py', "protocol=json.loads((SPEC/'protocol.json').read_text());assessment=validate(protocol)",
             "validate(json.loads((SPEC/'protocol.json').read_text()))\n"
             "protocol=load_protocol(SPEC/'protocol.json',WORK)\n"
             "assessment={'protocol_sha256':sha_bytes(encode(protocol))}")
replace_once('prepare_runtime.py', "confirmation={'before_each_request_idle_seconds':60,'max_request_or_trace_starts':1000,'max_wall_seconds':72000},",
             "confirmation={'before_each_request_idle_seconds':60,'max_request_or_trace_starts':240,'max_wall_seconds':21600},")
replace_once('prepare_runtime.py', 'resident_worker_max_wall_seconds=72000,',
             'resident_worker_max_wall_seconds=21600,')
replace_once('prepare_runtime.py', "overall_goal='All3contexts/all4blocks, unchanged paired confidence and conservative>=5% rule.',",
             "overall_goal='Produce a 4096 receipt for the separately approved per-context audit.',")
replace_once('test_schedule.py', 'def __init__(self,root,*,failed_512=False,pause_primer=False):',
             'def __init__(self,root,*,failed_512=False,pause_primer=False,protocol=None):')
replace_once('test_schedule.py', 'self.root=Path(root);self.protocol=json.loads(SPEC.read_text())',
             'self.root=Path(root);self.protocol=protocol if protocol is not None else json.loads(SPEC.read_text())')
write('confirmation-scope.json', {'contexts': [4096], 'max_starts': 240, 'max_active_seconds': 21600})
write('proposed-run.json', {
    'authorized': False,
    'operator': 'GPT-6 Luna high',
    'measurement_basis': 'One fresh 4096 confirmation under the approved per-context evidence rule, with pinned verified 512 and 2048 receipts.',
    'statistical_failure_handling': 'Preserve every row and failed control. Complete the declared bounded schedule unless a hard stop occurs. No automatic rerun.',
    'attempt_limit': 1,
    'max_sessions': 1,
    'expected_starts': 206,
    'review_sha256': digest(WORK / 'REVIEW.md'),
})
(WORK / 'OPERATOR.md').write_text(
    '# Scoped 4096 confirmation\n\n'
    'The prepared launch uses one e001 worker, a PID-bound sleep inhibitor and an exact-birth exit monitor. '
    'The cumulative controller limits are 240 starts and six active hours. '
    'The workload starts with the existing fresh 512 numerical handoff and two phase-memory diagnostics, '
    'then runs only the 4096 A/A and candidate block. The native timing runner is byte-identical. '
    'The post-exit report records the completed 4096 scope. '
    'The per-context audit reads and verifies the separately pinned receipts.\n'
)
subprocess.run([str(PYTHON), '-B', str(WORK / 'prepare_runtime.py')], check=True)
(WORK / 'validation').mkdir()
subprocess.run([str(PYTHON), '-B', str(WORK / 'refresh_runtime_build.py')], check=True)
