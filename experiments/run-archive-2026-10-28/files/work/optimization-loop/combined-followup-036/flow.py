"""Finish the fresh combined candidate, then one predeclared distinct loop factor."""
import ast,hashlib,json,os,re,secrets,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent;ROOT=CONTROL.parent.parent/'kernelmax-first-loop';SEARCH=CONTROL/'search-20261027.json';STATE=HERE/'state.json';DEADLINE=json.loads(SEARCH.read_text())['deadline_unix'];LABEL='com.kernelmax.search.combined036.20261027'
if STATE.exists():raise SystemExit('Retain existing combined workflow')
state={'status':'waiting_current_combined_candidate','pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'outcomes':[],'full_model_runs':[],'records':[]}
def save():
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');t.replace(STATE)
def update(**changes):
 d=json.loads(SEARCH.read_text());d.update(changes);t=SEARCH.with_suffix('.combo-flow-tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(SEARCH)
def tree():
 rows={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return rows

def run(name,path):
 if time.time()+30>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 known={};child=None;rec={'name':name,'command':[sys.executable,'-B','-u',str(path)],'started_unix':time.time(),'deadline_unix':DEADLINE};state['records'].append(rec);state['status']=name;save()
 try:
  with (path.parent/'flow-stdout.log').open('w')as out,(path.parent/'flow-stderr.log').open('w')as err:
   child=subprocess.Popen(rec['command'],cwd=ROOT,stdout=out,stderr=err,start_new_session=True);rec['worker_pid']=child.pid;save()
   update(active_job={'label':LABEL,'supervisor_pid':os.getpid(),'worker_pid':child.pid,'directory':str(path.parent),'deadline_unix':DEADLINE},outer_controller_label=LABEL)
   while child.poll()is None:
    rows=tree();included={child.pid};changed=True
    while changed:
     changed=False
     for pid,row in rows.items():
      if row['ppid']in included and pid not in included:included.add(pid);changed=True
    for pid in included:
     if pid in rows:known[pid]=rows[pid]
    if time.time()>=DEADLINE or(HERE/'STOP').exists():break
    time.sleep(min(.25,max(.001,DEADLINE-time.time())))
 finally:
  rows=tree()
  for pid,old in reversed(list(known.items())):
   if pid in rows and rows[pid]['start']==old['start']and rows[pid]['command']==old['command']:
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
  if child is not None:child.wait();rec['returncode']=child.returncode
  rec.update(ended_unix=time.time(),tracked_processes=known);save()
 if time.time()>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 if rec['returncode']!=0:raise RuntimeError(name+' failed; inspect retained evidence')

# Reuse the already reviewed count/identity audit and resident-primer model stage.
def import_function(file,name):
 tree_node=ast.parse(file.read_text());node=next(n for n in tree_node.body if isinstance(n,ast.FunctionDef)and n.name==name)
 code=ast.unparse(node).replace('full-model-034-','full-model-038-')
 exec(compile(code,str(file),'exec'),globals())
import_function(CONTROL/'native-k-unroll-032/search.py','audit')
import_function(CONTROL/'native-k-unroll-032/coordinator.py','full_model')

def finish(work,cid,index):
 result=audit(work,cid);result.update(candidate_id=cid,directory=str(work));(work/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');state['outcomes'].append(result);save()
 if result['status']=='no_operator_result':raise ValueError('Candidate pipeline failure; inspect before successor')
 if result['shortlisted']:full_model(result,index)
 return result

def generate_factor4():
 work=CONTROL/'combined-037';work.mkdir();cid='cand_'+secrets.token_hex(6);candidate=ROOT/'candidates'/cid;candidate.mkdir()
 ua=json.loads((CONTROL/'native-k-unroll-032/trial-002-cand_f36e3678fbf1/approval.json').read_text());da=json.loads((CONTROL/'down-candidate-025-cand_38d341fb17de/approval.json').read_text());up=Path(ua['snapshot']);down=Path(da['snapshot']);us=(up/'implementation.py').read_text();ds=(down/'implementation.py').read_text()
 source=us[:us.index('def run(')]+ds[ds.index('_DOWN='):ds.index('def run(')]+us[us.index('def run('):]
 source=source.replace('kernelmax_native_group_cand_f36e3678fbf1','kernelmax_combo_up_'+cid).replace('kernelmax_down_cand_38d341fb17de','kernelmax_combo_down_'+cid)
 old='    return down(activated, selected_expert_ids, sorted_indices=False).squeeze(-2)'
 new='''    if (down.weight.dtype != mx.uint32 or down.scales.dtype != mx.uint8
        or down.weight.shape != (e,k,n//8) or down.scales.shape != (e,k,n//32)
        or down.get("biases") is not None or "bias" not in down
        or down.bias.dtype != mx.bfloat16 or down.bias.shape != (e,k)):
        raise ValueError("Unsupported native packed down parameters")
    return _DOWN(inputs=[activated,selected_expert_ids,down.weight,down.scales,down.bias],
                 template=[("T",mx.bfloat16),("K",n),("N",k),("E",e)],
                 grid=((k//8)*64,4,1),threadgroup=(64,1,1),
                 output_shapes=[(1,1,4,k)],output_dtypes=[mx.bfloat16])[0]'''
 if source.count(old)!=1:raise ValueError('Combined-source anchor changed')
 source=source.replace(old,new);ast.parse(source)
 (candidate/'implementation.py').write_text(source);(candidate/'fused_up_gate.metal').write_bytes((up/'fused_up_gate.metal').read_bytes());(candidate/'down_bias.metal').write_bytes((down/'down_bias.metal').read_bytes());(candidate/'wrapper.py').write_bytes((up/'wrapper.py').read_bytes())
 (candidate/'manifest.json').write_text(json.dumps({'candidate_id':cid,'name':'Native-order joint up/gate and down pair unroll4','implementation_type':'mlx','description':'Predeclared distinct unroll4 two-kernel composition. Native MXFP4 weights/scales, BF16 roundings, native per-lane accumulation order, selected IDs and reference prefill preserved. No cache or parameter conversion. Tests interaction of measured component changes; no speedup or additive-gain claim.'},indent=2)+'\n')
 plan=json.loads((CONTROL/'combined-035/plan.json').read_text());plan.update(candidate_id=cid,loop_factor=4,predeclared_successor=True,up_component_sha256=ua['snapshot_sha256'],down_component_sha256=da['snapshot_sha256']);(work/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 for name in ['pipeline.py','tiny_native_checks.py','repair_checks.py','operator_diagnostic.py','paired_noise.py']:
  code=(CONTROL/'candidate-019'/name).read_text().replace('cand_d5987f75f173',cid);ast.parse(code);(work/name).write_text(code)
 code=(CONTROL/'candidate-019/supervise.py').read_text().replace('cand_d5987f75f173',cid).replace('com.kernelmax.search.candidate019.20261027',LABEL);ast.parse(code);(work/'supervise.py').write_text(code)
 update(current_candidate={'candidate_id':cid,'source':str(candidate),'work':str(work),'stage':'native_correctness_starting'})
 return cid,work

def main():
 save();claimed=False;assertion=None
 try:
  initial=CONTROL/'combined-035'
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   p=initial/'process.json'
   if p.exists():
    proc=json.loads(p.read_text());row=tree().get(proc['supervisor_pid'])
    if proc['status']not in ('starting','running')and(row is None or str(initial/'supervise.py')not in row['command']):break
   time.sleep(.5)
  if time.time()+60>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
  claimed=True;assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);cid=json.loads((initial/'plan.json').read_text())['candidate_id'];finish(initial,cid,1)
  if time.time()+60<DEADLINE and not(HERE/'STOP').exists():
   cid,work=generate_factor4();run('running_distinct_combined_factor4',work/'supervise.py');finish(work,cid,2)
  state['status']='combined_pool_finished'
 except TimeoutError as e:state['status']='deadline_or_stop_reached';state['error']=str(e)
 except BaseException as e:state['status']='needs_investigation';state['error']=f'{type(e).__name__}: {e}';raise
 finally:
  if assertion is not None:assertion.terminate();assertion.wait()
  state['ended_unix']=time.time();save()
  if claimed:update(active_job=None,status=state['status'],combined_controller=str(STATE),next_action='Inspect combined036 outcomes/fullmodel038 immediately; no unchanged rerolls, acceptance disabled, no commits. Stop task-owned work20:00UTC and retain partial evidence.')
if __name__=='__main__':main()
