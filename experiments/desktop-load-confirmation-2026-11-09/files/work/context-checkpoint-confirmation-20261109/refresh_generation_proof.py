"""Preserve the first CPU build proof; record cleanup-accounting integration."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

WORK=Path(__file__).resolve().parent
old_path=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-r2-resumed-20261103/run_round.py')
archive=WORK/'validation/native-generation-001/original-generation-proof.json'
if archive.exists(): raise RuntimeError('Generation proof revision already preserved')
shutil.copy2(WORK/'native-generation.json',archive)
old=ast.parse(old_path.read_text());new=ast.parse((WORK/'native_runner.py').read_text())
def code(tree,name):
    return ast.dump(next(node for node in ast.walk(tree) if isinstance(node,(ast.ClassDef,ast.FunctionDef))
        and node.name==name),include_attributes=False)
unchanged={name:code(old,name)==code(new,name) for name in ('Binding','AuditModel','interval','trace','correctness')}
if not all(unchanged.values()): raise RuntimeError('Native numerical AST changed')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
record={'source':str(old_path),'source_sha256':sha(old_path),'generated_sha256':sha(WORK/'native_runner.py'),
    'unchanged_AST':unchanged,'revision':2,'previous_proof_path':str(archive),'previous_proof_sha256':sha(archive),
    'reason':'Stage conclusions until outer native cleanup finishes, then require external exit accounting before a full-goal claim.',
    'model_or_gpu_run_started':False,'timing_authorized':False}
(WORK/'native-generation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
