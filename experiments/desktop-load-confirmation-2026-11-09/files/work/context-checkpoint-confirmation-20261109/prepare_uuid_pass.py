"""Prepare an inactive full confirmation pass with the tested boot identity fix."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
BASE = WORK.parent.parent
OLD = BASE / 'work/resumable-planned-pauses-20261107'
FIX = BASE / 'work/boot-session-identity-fix-20261107'
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


if {path.name for path in WORK.iterdir()} != {'prepare_uuid_pass.py', 'test_boot_session_resume.py'}:
    raise RuntimeError('Preparation requires an isolated fresh workspace')
expected = {
    OLD / 'runtime-build.json': '2bff506b2b469f56fedf2ae6de7310bd88ba993d6bc74f7b64d3dfb17891b56f',
    OLD / 'evidence/journal.jsonl': '640dfcebb3a0ecc1e67eb4384d87098b376e4bb4d6a66676b9fbaa7b229eee31',
    OLD / 'evidence/contexts/512.json': 'd3786fe1fca809dcb047b7ed83e9a698b40734c1aab1fa93cd445294fa4f1afd',
    OLD / 'epochs/e001/epoch-conclusion.json': '231ccaeb7ca559c4b33cdb02e40ac6a1b2b6a1a1a8b288cc7981bf3049ed25c2',
    FIX / 'runtime-integration.patch': 'cdd2faf727faf81c357b6ab24050b26d00c1c6650fdf6a0bc9c7696a126a79e0',
    FIX / 'scratch-001/runtime_integration.py': 'ac9b5a495d85be40614a25f275c672a25b30546ab7d3f7d3a97ae3121b00103b',
    FIX / 'scratch-001/test_boot_identity_fix.py': '1d0691ed92a4530ffdb6193d9bd17d9682873adf5889b9bc57050d574c1432a2',
}
for path, digest in expected.items():
    if sha(path) != digest:
        raise RuntimeError('Reviewed input changed: ' + str(path))
old_build = json.loads((OLD / 'runtime-build.json').read_text())
for path, digest in old_build['source_sha256'].items():
    if sha(Path(path)) != digest:
        raise RuntimeError('Frozen source changed: ' + path)

for source in sorted(OLD.glob('*.py')):
    destination = WORK / source.name
    shutil.copy2(source, destination)
    if sha(destination) != old_build['source_sha256'][str(source)]:
        raise RuntimeError('Executable copy changed: ' + str(destination))
for name in ('runtime_integration.py', 'test_boot_identity_fix.py'):
    shutil.copy2(FIX / 'scratch-001' / name, WORK / name)
    if sha(WORK / name) != expected[FIX / 'scratch-001' / name]:
        raise RuntimeError('Tested fix copy changed: ' + name)
shutil.copy2(OLD / 'native-generation.json', WORK / 'native-generation.json')
operator = (OLD / 'OPERATOR.md').read_text().replace(str(OLD), str(WORK))
(WORK / 'OPERATOR.md').write_text(operator + '\n## Boot identity\n\n'
    'Record the boot-session UUID at launch and recheck it for exit verification and resume. '
    'The controller requires the same UUID across epochs. Each epoch uses the existing process, '
    'provenance, power, thermal, numerical and cumulative-budget checks.\n')
write(WORK / 'preparation-provenance.json', {
    'reviewed_inputs_sha256': {str(path): digest for path, digest in expected.items()},
    'historical_timing_imported': False,
    'new_candidate_allocation': False,
})
proposal = json.loads((OLD / 'proposed-run.json').read_text())
proposal.update(
    status='uuid_controller_prepared_separate_activation_required',
    authorized=False,
    known_previous_failure='The preceding pass completed the 512 seal and closed e001 as user_pause. Its frozen exit verifier rejected a changed kern.boottime string. The journal, failed verifier and context receipt remain historical evidence.',
    measurement_basis='One new full confirmation using the same candidate, immutable baseline, measurement contract, planned pauses after the new 512 and 2048 seals, and explicit resumes. The newly frozen controller binds monotonic continuity to the boot-session UUID. All performance rows and controls will be collected within the new pass.',
    boot_identity_patch_sha256=expected[FIX / 'runtime-integration.patch'],
)
write(WORK / 'proposed-run.json', proposal)
subprocess.run([str(PYTHON), '-B', str(WORK / 'prepare_runtime.py')], check=True)
(WORK / 'validation').mkdir()
subprocess.run([str(PYTHON), '-B', str(WORK / 'refresh_runtime_build.py')], check=True)
