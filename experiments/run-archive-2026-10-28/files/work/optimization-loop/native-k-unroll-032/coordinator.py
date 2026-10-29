"""Immediately run predeclared native-order K-loop candidates after down search."""
import ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent;ROOT=CONTROL.parent.parent/'kernelmax-first-loop';SEARCH=CONTROL/'search-20261027.json';STATE=HERE/'controller-state.json';DEADLINE=json.loads(SEARCH.read_text())['deadline_unix'];LABEL='com.kernelmax.search.kunroll032.20261027'
if STATE.exists():raise SystemExit('Retain existing controller')
state={'status':'waiting_current_down_controller','pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'queues':[],'full_model_runs':[],'records':[]}
def save():
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');t.replace(STATE)
def update(**changes):
 d=json.loads(SEARCH.read_text());d.update(changes);t=SEARCH.with_suffix('.kunroll-tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(SEARCH)
def tree():
 rows={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return rows

def run(name,path):
 if time.time()+30>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 child=None;known={};rec={'name':name,'command':[sys.executable,'-B','-u',str(path)],'started_unix':time.time(),'deadline_unix':DEADLINE};state['records'].append(rec);state['status']=name;save()
 try:
  with (path.parent/'controller-stdout.log').open('w')as out,(path.parent/'controller-stderr.log').open('w')as err:
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

def full_model(selected,index):
 full=CONTROL/f'full-model-034-{index:02d}';full.mkdir();base=CONTROL/'memory-residency-027';cid=selected['candidate_id']
 code=(base/'run.py').read_text();marker="                state['summaries'][str(length)]=summary;save()"
 if code.count(marker)!=1:raise ValueError('Full-model decision anchor changed')
 code=code.replace(marker,marker+"\n                if not summary['improvement_above_measured_noise']:\n                    state['status']='stopped_context_futility';save();return")
 ast.parse(code);(full/'run.py').write_text(code)
 plan=json.loads((base/'plan.json').read_text());plan.update(status='execution_plan',selection='Fresh native-order loop candidate '+cid+' after valid operator shortlist, no timing reroll.',candidate_id=cid,order='Sequential after current queue; no overlapping benchmark.',completed_context_futility='Stop after20valid paired samples fail the existing improvement_above_measured_noise gate.',operator_evidence=str(Path(selected['directory'])/'result.json'))
 (full/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');(full/'selected.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
 code=(base/'supervise.py').read_text().replace('cand_02e5f860f143',cid).replace('com.kernelmax.search.residency027.20261027',LABEL).replace('memory-residency-027',full.name);ast.parse(code);(full/'supervise.py').write_text(code)
 update(current_candidate={'candidate_id':cid,'stage':'full_model_validation','work':str(full)})
 run('running_independent_native_loop_full_model',full/'supervise.py')
 d=json.loads((full/'state.json').read_text());state['full_model_runs'].append({'candidate_id':cid,'directory':str(full),'status':d['status'],'full_model_speedup_established':d['full_model_speedup_established']});save()
 if not all(d.get(k)is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):raise ValueError('Full-model integrity failure')
 if d['full_model_correctness']and not all(r['passed']for r in d['full_model_correctness']):raise ValueError('Native loop family numerical failure; diagnose before more variants')
 if d['status']=='failed'and not any('Environment invalid'in e or'Resident primer power/thermal invalid'in e or'Primer preflight invalid'in e for e in d['errors']):raise ValueError('Unexpected full-model failure')


def main():
 save();claimed=False;assertion=None
 try:
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   d=json.loads((CONTROL/'model-futility-030/state.json').read_text());row=tree().get(d['pid'])
   if 'ended_unix'in d and(row is None or str(CONTROL/'model-futility-030/watch.py')not in row['command']):break
   time.sleep(.5)
  if time.time()+60>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
  if d['status']!='futility_and_distinct_target_finished':raise ValueError('Prior controller needs investigation: '+d['status'])
  prior=json.loads((CONTROL/'down-followup-031/state.json').read_text())
  if prior['status']not in ('parked_down_candidate_no_operator_gain','down_full_model_finished','down_full_model_numerical_failure','down_budget_inconclusive','down_budget_insufficient'):raise ValueError('Prior down outcome requires investigation: '+prior['status'])
  claimed=True;assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);remaining=[(2,),(4,)];index=1
  while remaining and time.time()+60<DEADLINE and not(HERE/'STOP').exists():
   queue=HERE if index==1 else CONTROL/f'native-k-unroll-resume-{index:02d}'
   if queue!=HERE:
    queue.mkdir()
    for name in ['search.py','paired_noise.py','operator-template.py']:(queue/name).write_bytes((HERE/name).read_bytes())
   code=(queue/'search.py').read_text();start=code.index('    pool=');end=code.index("    state['predeclared_pool']",start);code=code[:start]+f'    pool={remaining!r}\n'+code[end:];ast.parse(code);(queue/'search.py').write_text(code)
   update(next_controller=str(STATE),status='testing_predeclared_native_k_loop',continuous_search=str(queue/'state.json'),next_action='Predeclared ordered K-loop factors2then4, one fresh candidate at a time. Native/tiny/exact checks, fresh paired operator calibration, then only reliable winners to resident-primer model test. No rerolls or commits; stop20:00UTC.')
   state['queues'].append(str(queue));save();run('running_native_k_loop_queue',queue/'search.py')
   d=json.loads((queue/'state.json').read_text());attempted={tuple(r['config'])for r in d['trials']};remaining=[r for r in remaining if r not in attempted]
   if d['shortlist']:full_model(d['shortlist'][0],index);index+=1
   elif d['status']!='numeric_pool_exhausted':raise ValueError('Native K-loop queue stopped: '+d['status'])
  state['status']='deadline_or_stop_reached'if time.time()>=DEADLINE or(HERE/'STOP').exists()else'predeclared_native_loop_pool_exhausted'
 except TimeoutError as e:state['status']='deadline_or_stop_reached';state['error']=str(e)
 except BaseException as e:state['status']='needs_investigation';state['error']=f'{type(e).__name__}: {e}';raise
 finally:
  if assertion is not None:assertion.terminate();assertion.wait()
  state['ended_unix']=time.time();save()
  if claimed:update(active_job=None,status=state['status'],native_k_loop_controller=str(STATE),next_action='Immediately inspect native-k-unroll032 controller/queue/fullmodel034 results. Preserve every sample; no unchanged rerolls. Continue justified work until20:00UTC harddeadline; no commits.')
if __name__=='__main__':main()
