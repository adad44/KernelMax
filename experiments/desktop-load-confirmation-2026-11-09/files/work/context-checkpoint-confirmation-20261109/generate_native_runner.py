"""Generate reviewed integration hooks without changing native math or old files."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path('/Users/alandiaz/Downloads/KernelMaxxxing')
OLD=BASE/'work/decode-r2-resumed-20261103'
WORK=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change(source,old,new,count=1):
    if source.count(old)!=count: raise RuntimeError('Unexpected patch site: '+old[:100])
    return source.replace(old,new)


output=WORK/'native_runner.py'
if output.exists(): raise RuntimeError('Never overwrite generated integration; preserve previous builds')
manifest=json.loads((OLD/'prelaunch-manifest.json').read_text())
for path,expected in manifest['immutable_input_sha256'].items():
    if sha(path)!=expected: raise RuntimeError('Historical integration source changed: '+path)
for name in ('weight_residency.py','conditioning.py','quiet_preparation.py','reuse_correctness.py',
        'contract_bridge.py','model_relocation.py','desktop_telemetry.py'):
    if (WORK/name).exists(): raise RuntimeError('Fresh helper copy only')
    shutil.copy2(OLD/name,WORK/name)
original=(OLD/'run_round.py').read_text();source=original
source=change(source,'import contextlib','import argparse\nimport contextlib')
source=change(source,'from checkpoint_policy import assess_block,assess_workload',
    'from runtime_integration import RuntimeSession,UserPause')
source=change(source,'from checkpoint_catalog import retain_checkpoint','')
source=change(source,'    output = WORK / "resident-state.json"',
    '''    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    integration=RuntimeSession(WORK,args.epoch)
    EPOCH_WORK=integration.epoch_work
    output = EPOCH_WORK / "resident-state.json"''')
source=change(source,'        WORK / "session.json",',
    '''        WORK / "session.json",
        WORK / "runtime_integration.py",WORK / "evidence.py",WORK / "verification.py",
        WORK / "runtime-build.json",WORK / "timing-authorization.json",''')
for line in ('        WORK / "checkpoint_policy.py",\n','        WORK / "checkpoint_catalog.py",\n'):
    source=change(source,line,'')
source=change(source,'    def check(frozen=None):\n',
    '    def check(frozen=None):\n        integration.check(honor_pause=False)\n')
source=change(source,'WORK / (\'storage-\' + selected.candidate_id)',
    'EPOCH_WORK / (\'storage-\' + selected.candidate_id)')
source=change(source,'directory = WORK / f"command-{number:03d}-{mode}-{current_id}"',
    'directory = EPOCH_WORK / f"command-{number:03d}-{mode}-{current_id}"')
source=change(source,'max_requests = 128 if mode == "screen" else 800',
    'max_requests = 128 if mode == "screen" else 1000')
source=change(source,'            def operation_check():\n',
    '            def operation_check():\n                integration.check()\n')
source=change(source,"                    7200 if mode=='screen' else 57600",
    "                    7200 if mode=='screen' else 72000")
source=change(source,"                    # Persist both rows together after measurement.",
    "                    if integration.pause_requested:\n                        store()\n                        raise UserPause('User pause after drained primer; measurement did not start')\n                    # Persist both rows together after measurement.")

# Replace only the old confirmation schedule, leaving trace/correctness and
# numerical model calls intact. The screen branch is unreachable as before.
tree=ast.parse(source)
branch=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
    and isinstance(node.test,ast.Compare) and ast.unparse(node.test)=="mode == 'screen'")
start=branch.orelse[0].lineno-1;end=branch.orelse[-1].end_lineno
lines=source.splitlines(keepends=True)
replacement='''                    reused,reference_tokens,reuse_info=completed_correctness(
                        WORK/'recovery-parent.json',candidate_id=current_id,snapshot_sha256=frozen.sha256,
                        contract_sha256=contract.sha256,receipt_path=session['checkpoint_cache'],residency=residency,
                        bridge_path=WORK/'reference-bridge.json',contract_path=WORK/'contract.yaml',reference_path=ORIGINAL_REFERENCE,relocation_path=WORK/'model-relocation.json')
                    cs['correctness']=reused
                    cs['correctness_reuse']=reuse_info
                    store()

                    def context_guards():
                        mx.synchronize()
                        check(frozen)
                        if residency is not None: residency.switch(False)
                        storage_ok=residency.verify_final() if residency is not None else True
                        sources={**reuse_info['source_artifact_sha256'],
                            **{row['arrays']:row['arrays_sha256'] for row in cs['correctness']}}
                        sources_ok=all(digest(path)==expected for path,expected in sources.items())
                        methods_ok=(binding.linear_cls.__call__ is binding.linear
                            and binding.block_cls.__call__ is binding.block
                            and binding.expert_cls.__call__ is binding.expert)
                        return {'protected_unchanged':protected==_protected_files(extra),
                            'checkpoint_and_snapshot_unchanged':True,'vocabulary_storage_unchanged':storage_ok,
                            'original_methods_restored':methods_ok,'reused_correctness_sources_unchanged':sources_ok,
                            'protected_source_sha256':_protected_files(extra),
                            'numerical_source_sha256':sources,'worker_process':integration.process}

                    integration.execute(collect,correctness,cs,context_guards,lambda:request_count)
'''
lines[start:end]=[replacement];source=''.join(lines)
source=change(source,'            except ConfirmationStop as error:',
    '''            except UserPause as error:
                cs['status']='paused_by_user'
                cs['pause_reason']=str(error)
            except ConfirmationStop as error:''')
start=source.index("            if cs['repeatable_improvement_verified']:")
end=source.index('            if cs["five_percent_improvement_verified"]:',start)
source=source[:start]+"            integration.stage_finalize(cs,directory/'state.json')\n            store()\n"+source[end:]
source=change(source,'                run_command(command, number)\n                number += 1',
    '                run_command(command, number)\n                return  # Exactly one epoch per worker; no autonomous restart.')
source=change(source,'        state["finished_unix"] = time.time()\n        save()',
    '        state["finished_unix"] = time.time()\n        integration.finish_native(state)\n        save()')
ast.parse(source)
old_tree=ast.parse(original);new_tree=ast.parse(source)
def code(tree,name):
    return ast.dump(next(node for node in ast.walk(tree)
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name),include_attributes=False)
unchanged={name:code(old_tree,name)==code(new_tree,name)
    for name in ('Binding','AuditModel','interval','trace','correctness')}
if not all(unchanged.values()): raise RuntimeError('Native numerical/timing helper changed')
output.write_text(source)
(WORK/'native-generation.json').write_text(json.dumps({'source':str(OLD/'run_round.py'),
    'source_sha256':sha(OLD/'run_round.py'),'generated_sha256':sha(output),
    'unchanged_AST':unchanged,'model_or_gpu_run_started':False,'timing_authorized':False},indent=2)+'\n')
print(json.dumps({'generated':str(output),'sha256':sha(output),'unchanged_AST':unchanged},indent=2))
