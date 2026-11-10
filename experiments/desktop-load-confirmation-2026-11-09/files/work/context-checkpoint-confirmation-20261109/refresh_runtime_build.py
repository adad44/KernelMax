"""Extend the inactive build with final CPU coverage and bounded exit monitor."""
import hashlib
import json
from pathlib import Path
import shutil

WORK=Path(__file__).resolve().parent
archive=WORK/'validation/runtime-build-001'
if archive.exists() or (WORK/'timing-authorization.json').exists() or (WORK/'evidence').exists():
    raise RuntimeError('Inactive prelaunch revision only; never overwrite a running/frozen pass')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


build=json.loads((WORK/'runtime-build.json').read_text())
for path,expected in build['source_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Previously frozen executable changed: '+path)
archive.mkdir()
for name in ('runtime-build.json','timing-authorization.example.json'):
    shutil.copy2(WORK/name,archive/name)
sources={**build['source_sha256']}
for path in list(WORK.glob('*.py'))+[WORK/'OPERATOR.md']:
    sources[str(path)]=sha(path)
build['source_sha256']=sources
build['frozen_identities']['source_sha256']=sources
build['revision']=2
build['previous_build_path']=str(archive/'runtime-build.json')
build['previous_build_sha256']=sha(archive/'runtime-build.json')
(WORK/'runtime-build.json').write_text(json.dumps(build,indent=2,sort_keys=True)+'\n')
example=json.loads((WORK/'timing-authorization.example.json').read_text())
example['runtime_build_sha256']=sha(WORK/'runtime-build.json')
example['frozen_identities']=build['frozen_identities']
(WORK/'timing-authorization.example.json').write_text(json.dumps(example,indent=2,sort_keys=True)+'\n')
print(json.dumps({'revision':2,'runtime_build_sha256':sha(WORK/'runtime-build.json'),
    'frozen_source_count':len(sources),'timing_authorized':False},indent=2))
