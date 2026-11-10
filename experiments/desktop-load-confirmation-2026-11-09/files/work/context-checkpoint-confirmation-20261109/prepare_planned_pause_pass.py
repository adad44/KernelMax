"""Freeze a fresh pass using the existing controller and reviewed pause schedule."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
BASE = WORK.parent.parent
OLD = BASE / 'work/resumable-confirmation-20261105'
REVIEW = BASE / 'work/context-pause-proposal-20261106/REVIEW.md'
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


if list(WORK.iterdir()) != [Path(__file__).resolve()]:
    raise RuntimeError('Preparation requires a fresh directory containing only this script')
retained = {
    OLD / 'runtime-build.json': 'e4f152a14bbab7c7d6936953981d0112b90c7dcebece63023e8c62f55459c0d6',
    OLD / 'post-exit/e001.json': '1ae63bf76f661c21c59467a16e81d9ee0775595e5669791533abe4c51590f677',
    OLD / 'evidence/contexts/512.json': '648f97633c8fced585eed86efee7ab3eabff06a5e42e0ccd37d22351d393a572',
    OLD / 'evidence/contexts/2048.json': '1e59c0cf426261f94df9d083bfee47b2b496b0ee5270374bab33c5a33b2d3cb7',
    OLD / 'catalog-revisions/002.json': '63d7c67b5b6c88a32ea14971b121e58f2a5c0a2a2d9da892af4773cfd8a782af',
    REVIEW: 'c614fb67d896a099c85ebbaef337f05898f1a8044e2c082067e461f4b5771453',
}
for path, expected in retained.items():
    if sha(path) != expected:
        raise RuntimeError('Reviewed prior evidence changed: ' + str(path))
build = json.loads((OLD / 'runtime-build.json').read_text())
for path, expected in build['source_sha256'].items():
    if sha(path) != expected:
        raise RuntimeError('Frozen executable or input changed: ' + path)

copied = {}
for source in sorted(OLD.glob('*.py')):
    digest = build['source_sha256'][str(source)]
    destination = WORK / source.name
    shutil.copy2(source, destination)
    if sha(destination) != digest:
        raise RuntimeError('Executable copy changed: ' + str(destination))
    copied[str(source)] = digest
shutil.copy2(OLD / 'native-generation.json', WORK / 'native-generation.json')
operator = (OLD / 'OPERATOR.md').read_text().replace(str(OLD), str(WORK))
(WORK / 'OPERATOR.md').write_text(operator + '\n' + REVIEW.read_text())
write(WORK / 'preparation-provenance.json', {
    'copied_executable_sources': copied,
    'prior_evidence_and_review_sha256': {str(path): digest for path, digest in retained.items()},
    'historical_timing_imported': False,
    'new_candidate_allocation': False,
})
proposal = json.loads((OLD / 'proposed-run.json').read_text())
proposal.update(
    status='planned_pause_pass_separate_activation_required',
    authorized=False,
    review_path=str(REVIEW),
    review_sha256=retained[REVIEW],
    measurement_basis='One independent confirmation under the unchanged evidence contract. Fresh observations for all three contexts and four blocks. The operator requests a clean pause after the new 512 seal in e001 and the new 2048 seal in e002. Each subsequent epoch requires explicit user resume. Caps apply cumulatively to the single pass.',
    known_previous_failure='The preceding pass stopped on AC loss during 4096 A/A. Both completed context receipts remain retained separately.',
    planned_pause_after_context={'e001': 512, 'e002': 2048},
    automatic_resume=False,
)
write(WORK / 'proposed-run.json', proposal)
subprocess.run([str(PYTHON), '-B', str(WORK / 'prepare_runtime.py')], check=True)
(WORK / 'validation').mkdir()
subprocess.run([str(PYTHON), '-B', str(WORK / 'refresh_runtime_build.py')], check=True)
