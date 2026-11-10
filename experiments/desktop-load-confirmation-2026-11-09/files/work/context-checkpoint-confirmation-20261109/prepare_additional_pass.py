"""Prepare the explicitly authorized independent pass; never load/run a model."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path(__file__).resolve().parent
BASE = WORK.parent.parent
OLD = BASE / 'work/resumable-controller-20261104'
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


if list(WORK.iterdir()) != [Path(__file__).resolve()]:
    raise RuntimeError('Preparation requires a fresh directory containing only this script')
if sha(OLD / 'runtime-build.json') != '712111a7aa0e6f37fd3d45f49f3c7e76baad4ad87d3bdaa132a492ba31f8de46':
    raise RuntimeError('Prior reviewed controller build changed')
build = json.loads((OLD / 'runtime-build.json').read_text())
for path, expected in build['source_sha256'].items():
    if sha(path) != expected:
        raise RuntimeError('Prior frozen executable/input changed: ' + path)
post = json.loads((OLD / 'post-exit/e001.json').read_text())
if not post['worker_process_verified_gone'] or post['complete_locked_workload']:
    raise RuntimeError('Expected independently verified terminal/incomplete prior pass')
retained = {
    'post-exit/e001.json': '57ad1230c779e982d38d72b0cc2ed3156aa6f492dce29957587aefa608367fb8',
    'evidence/contexts/512.json': '86feda16aee859c9ff87adaeef59f5487209781c7afc032ba316600914de001c',
    'catalog-revisions/001.json': '868cab230950efa1d0863663cc9ba809584c127d5087482c20f1dc4b85ae00f7',
    'bookkeeping/next-pass-review.md': '90ab9b80a0d33ccb67d943937c983fc0820f286ea90a62a10203ae2e250ad689',
}
for name, expected in retained.items():
    if sha(OLD / name) != expected:
        raise RuntimeError('Prior retained evidence/review changed: ' + name)
copied = {}
for source in sorted(OLD.glob('*.py')):
    destination = WORK / source.name
    shutil.copy2(source, destination)
    copied[str(source)] = sha(source)
    if sha(destination) != copied[str(source)]:
        raise RuntimeError('Executable copy changed')
shutil.copy2(OLD / 'native-generation.json', WORK / 'native-generation.json')
operator = (OLD / 'OPERATOR.md').read_text().replace(str(OLD), str(WORK))
(WORK / 'OPERATOR.md').write_text(operator)
write(WORK / 'preparation-provenance.json', {
    'schema': 1,
    'status': 'independent_additional_pass_preparation',
    'prior_runtime_build_sha256': sha(OLD / 'runtime-build.json'),
    'copied_executable_sources': copied,
    'prior_evidence_sha256': {str(OLD / name): digest for name, digest in retained.items()},
    'historical_timing_imported': False,
    'prior_pass_reopened': False,
    'new_candidate_allocation': False,
    'candidate_code_changed': False,
    'timing_authorized_by_this_preparation_file': False,
})
old_proposal = json.loads((OLD / 'proposed-run.json').read_text())
proposal = {**old_proposal,
    'status': 'additional_pass_reviewed_separate_activation_required',
    'authorized': False,
    'measurement_basis': 'One explicitly authorized additional independent full-workload confirmation after the previous pass terminated on AC loss. Preserve its verified 512 context checkpoint separately. Collect all three fresh A/A groups and all four fresh candidate blocks; no prior timing or control rows imported, no prior pass reopened, unchanged candidate and statistics. This is one new bounded attempt with no automatic replacement.',
    'known_previous_failure': 'Previous pass hard-stopped on AC loss before 2048 A/A pair 14. Its 512 A/A and both 512 blocks passed; that scoped checkpoint remains retained. Its overall full-workload flag remains false. This pass is not a statistical reroll of a failed control.',
    'review_path': str(OLD / 'bookkeeping/next-pass-review.md'),
    'review_sha256': retained['bookkeeping/next-pass-review.md'],
}
write(WORK / 'proposed-run.json', proposal)
subprocess.run([str(PYTHON), '-B', str(WORK / 'prepare_runtime.py')], check=True)
subprocess.run([str(PYTHON), '-B', str(WORK / 'refresh_runtime_build.py')], check=True)
print(json.dumps({'status': 'prepared_no_activation_or_worker', 'work': str(WORK),
    'runtime_build_sha256': sha(WORK / 'runtime-build.json'),
    'copied_python_files': len(copied), 'model_or_gpu_run_started': False}))
