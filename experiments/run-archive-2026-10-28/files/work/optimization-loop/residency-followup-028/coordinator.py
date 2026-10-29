"""Sequential measured residency intervention, then the held down-region search."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent;ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
STATE=HERE/'state.json';SEARCH=CONTROL/'search-20261027.json';DEADLINE=json.loads(SEARCH.read_text())['deadline_unix'];LABEL='com.kernelmax.search.residency028.20261027'
if STATE.exists():raise SystemExit('Retain existing run')
state={'status':'waiting_current_geometry','pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'stages':[]}
def save():
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');p.replace(STATE)
def update(**changes):
 d=json.loads(SEARCH.read_text());d.update(changes);p=SEARCH.with_suffix('.residency-tmp');p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');p.replace(SEARCH)
def tree():
 rows={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return rows

def run(name,path):
 if time.time()+30>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 known={};child=None;record={'stage':name,'command':[sys.executable,'-B','-u',str(path)],'deadline_unix':DEADLINE,'started_unix':time.time()};state['stages'].append(record);state['status']=name;save()
 try:
  with (path.parent/'parent-stdout.log').open('w')as out,(path.parent/'parent-stderr.log').open('w')as err:
   child=subprocess.Popen(record['command'],cwd=ROOT,stdout=out,stderr=err,start_new_session=True);record['worker_pid']=child.pid;save()
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
  if child is not None:child.wait();record['returncode']=child.returncode
  record.update(ended_unix=time.time(),tracked_processes=known);save()
 if time.time()>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 if record['returncode']!=0:raise RuntimeError(name+' failed; inspect')

def main():
 save();assertion=None;claimed=False
 try:
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   old=CONTROL/'full-model-022-03'
   if old.exists()and not(old/'STOP').exists():
    d=json.loads((old/'state.json').read_text())if(old/'state.json').exists()else{}
    if not d.get('requests'):
     (old/'STOP').write_text('Controller hold before timing: two previous old-cadence full-model runs failed paging. Follow with predeclared resident-primer intervention027, not an unchanged timing retry.\n')
     state['unchanged_cadence_hold']={'directory':str(old),'utc_unix':time.time(),'timed_requests_started':False};save()
   d=json.loads((CONTROL/'followup-021/state.json').read_text());row=tree().get(d['pid'])
   if 'ended_unix'in d and(row is None or str(CONTROL/'followup-021/coordinator.py')not in row['command']):break
   time.sleep(.25)
  if time.time()+60>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
  if d['status']!='native_order_pool_exhausted':raise ValueError('Predecessor needs investigation: '+d['status'])
  claimed=True;assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
  work=CONTROL/'memory-residency-027'
  update(status='testing_full_model_residency_intervention',current_candidate={'candidate_id':'cand_02e5f860f143','stage':'changed_resident_primer_protocol','work':str(work)},active_job={'label':LABEL,'directory':str(work),'supervisor_pid':os.getpid(),'deadline_unix':DEADLINE},residency_intervention=str(work/'plan.json'))
  run('testing_resident_primer_protocol',work/'supervise.py')
  d=json.loads((work/'state.json').read_text());state['residency_outcome']={'status':d['status'],'full_model_speedup_established':d['full_model_speedup_established'],'timed_requests':len(d['requests']),'resident_primers':len(d['resident_primers'])};save()
  if not all(d.get(k)is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):raise ValueError('Residency integrity failure')
  if d['full_model_correctness']and not all(r['passed']for r in d['full_model_correctness']):raise ValueError('Numerical failure; investigate')
  if d['status']=='failed'and not any('Environment invalid'in e or'Resident primer power/thermal invalid'in e or'Primer preflight invalid'in e for e in d['errors']):raise ValueError('Unexpected resident experiment failure')
  run('continuing_measured_down_target',CONTROL/'down-followup-029/coordinator.py')
  state['status']='residency_and_down_work_finished'
 except TimeoutError as e:state['status']='deadline_or_stop_reached';state['error']=str(e)
 except BaseException as e:state['status']='needs_investigation';state['error']=f'{type(e).__name__}: {e}';raise
 finally:
  if assertion is not None:assertion.terminate();assertion.wait()
  state['ended_unix']=time.time();save()
  if claimed:update(active_job=None,status=state['status'],residency_followup=str(STATE),next_action='Immediately inspect residency028, memory-residency027 and down-followup029 state. Preserve all failed controls; no unchanged statistical rerolls. No commits; stop20:00UTC.')
if __name__=='__main__':main()
