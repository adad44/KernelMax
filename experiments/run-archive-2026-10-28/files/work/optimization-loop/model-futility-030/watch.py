"""Enforce completed-context futility without modifying a frozen experiment."""
import json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent;ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
STATE=HERE/'state.json';SEARCH=CONTROL/'search-20261027.json';DEADLINE=json.loads(SEARCH.read_text())['deadline_unix'];LABEL='com.kernelmax.search.futility030.20261027'
if STATE.exists():raise SystemExit('Retain existing monitor')
state={'status':'watching_completed_contexts','pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'records':[]}
def save():
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');t.replace(STATE)
def update(**changes):
 d=json.loads(SEARCH.read_text());d.update(changes);t=SEARCH.with_suffix('.futility-tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(SEARCH)
def tree():
 out={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:out[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return out

def run_followup(path):
 known={};child=None
 rec={'command':[sys.executable,'-B','-u',str(path)],'started_unix':time.time(),'deadline_unix':DEADLINE};state['records'].append(rec);save()
 try:
  with (path.parent/'futility-parent-stdout.log').open('w')as out,(path.parent/'futility-parent-stderr.log').open('w')as err:
   child=subprocess.Popen(rec['command'],cwd=ROOT,stdout=out,stderr=err,start_new_session=True);rec['worker_pid']=child.pid;save()
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
 if rec['returncode']!=0:raise RuntimeError('Down followup failed; inspect evidence')

def main():
 save();claimed=False
 try:
  work=CONTROL/'memory-residency-027';failure=None
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   d=json.loads((work/'state.json').read_text())
   if d['status']in ('stopped_aa_screen','failed','failed_integrity','completed_full_model_diagnostic'):
    state['status']='existing_controller_handles_result';return
   for length,summary in d['summaries'].items():
    if not summary['improvement_above_measured_noise']:
     rows=[r for r in d['requests']if r['length']==int(length)and r['phase']=='candidate']
     for lane in ('baseline','candidate'):
      selected=[r for r in rows if r['lane']==lane]
      if len(selected)!=20 or{r['pair']for r in selected}!=set(range(1,21))or any(r['invalid_reasons']for r in selected):raise ValueError('Completed-context futility requires all20valid pairs')
     failure={'length':int(length),'summary':summary,'complete_valid_candidate_pairs':20,'decision':'park; gain not established above fresh A/A noise; no later contexts or reroll','decision_unix':time.time()};break
   if failure:break
   time.sleep(.25)
  if not failure:state['status']='deadline_or_stop_reached';return
  if (work/'STOP').exists():raise ValueError('Another stop reason exists; inspect before successor')
  (work/'STOP').write_text('Completed-context statistical futility under existing improvement_above_measured_noise gate. All20paired candidate samples retained. Stop later contexts; proceed to distinct measured down target.\n')
  state['status']='waiting_clean_model_exit';state['futility']=failure;save()
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   parent=json.loads((CONTROL/'residency-followup-028/state.json').read_text());rows=tree();row=rows.get(parent['pid'])
   if 'ended_unix'in parent and(row is None or str(CONTROL/'residency-followup-028/coordinator.py')not in row['command']):break
   time.sleep(.25)
  if time.time()+60>=DEADLINE or(HERE/'STOP').exists():state['status']='deadline_or_stop_reached';return
  d=json.loads((work/'state.json').read_text())
  if not all(d.get(k)is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):raise ValueError('Model integrity failure')
  if not all(r['passed']for r in d['full_model_correctness']):raise ValueError('Model numerical failure')
  if d['status']!='failed'or not any('Deadline or explicit stop reached'in e for e in d['errors']):raise ValueError('Unexpected model stop result')
  if parent['status']!='needs_investigation'or'Unexpected resident experiment failure'not in parent.get('error',''):raise ValueError('Unexpected parent stop result')
  raw=work/'state.json';failure.update(candidate_id=d['candidate_id'],state_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),all_integrity_guards_passed=True,full_model_speedup_established=False)
  (HERE/'futility-result.json').write_text(json.dumps(failure,indent=2,sort_keys=True)+'\n')
  next_work=CONTROL/'down-followup-031';next_work.mkdir()
  for name in ('coordinator.py','build_candidate.py'):
   code=(CONTROL/'down-followup-029'/name).read_text().replace('com.kernelmax.search.residency028.20261027',LABEL).replace('down-followup029','down-followup031');(next_work/name).write_text(code)
  (next_work/'prepared-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in next_work.iterdir()},indent=2,sort_keys=True)+'\n')
  claimed=True;state['status']='continuing_distinct_down_target';save()
  update(status='parked_model_context_futility',active_job=None,model_futility=str(HERE/'futility-result.json'),next_controller=str(next_work/'state.json'),outer_controller_label=LABEL,next_action='All20valid candidate pairs failed existing model-noise improvement gate; no later context/unchanged reroll. Immediately continue distinct down budget/candidate via031; no overlap; stop20:00UTC.')
  run_followup(next_work/'coordinator.py');state['status']='futility_and_distinct_target_finished'
 except BaseException as e:
  state['status']='deadline_or_stop_reached'if time.time()>=DEADLINE or(HERE/'STOP').exists()else'needs_investigation';state['error']=f'{type(e).__name__}: {e}';raise
 finally:
  state['ended_unix']=time.time();save()
  if claimed:update(active_job=None,status=state['status'],model_futility_controller=str(STATE),next_action='Inspect model-futility030 and down-followup031 stage evidence immediately. Preserve all samples and guards; no rerolls or commits; stop20:00UTC.')
if __name__=='__main__':main()
