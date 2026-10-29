"""Immediately follow the finite native-order queue with a measured down target."""
import ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from build_candidate import generate
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent
ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
SEARCH=CONTROL/'search-20261027.json';STATE=HERE/'state.json'
DEADLINE=json.loads(SEARCH.read_text())['deadline_unix']
LABEL='com.kernelmax.search.residency028.20261027'
if STATE.exists():raise SystemExit('Existing controller retained')
state={'status':'waiting_native_order_controller','pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'stages':[]}

def save():
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');p.replace(STATE)
def update(**changes):
 s=json.loads(SEARCH.read_text());s.update(changes);p=SEARCH.with_suffix('.down-tmp');p.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');p.replace(SEARCH)
def tree():
 rows={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:rows[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return rows

def run(name,command,directory,max_seconds):
 if time.time()+30>=DEADLINE or (HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
 bound=min(DEADLINE,time.time()+max_seconds);known={};child=None
 rec={'name':name,'command':command,'directory':str(directory),'started_unix':time.time(),'deadline_unix':bound,'status':'running'}
 state['stages'].append(rec);state['status']=name;save()
 try:
  with (directory/'controller-stdout.log').open('w')as out,(directory/'controller-stderr.log').open('w')as err:
   child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
   rec['worker_pid']=child.pid;save()
   update(status=name,active_job={'label':LABEL,'supervisor_pid':os.getpid(),'worker_pid':child.pid,'directory':str(directory),'deadline_unix':bound})
   while child.poll()is None:
    rows=tree();included={child.pid};changed=True
    while changed:
     changed=False
     for pid,row in rows.items():
      if row['ppid']in included and pid not in included:included.add(pid);changed=True
    for pid in included:
     if pid in rows:known[pid]=rows[pid]
    if time.time()>=bound or (HERE/'STOP').exists():break
    time.sleep(min(.25,max(.001,bound-time.time())))
 finally:
  rows=tree()
  for pid,old in reversed(list(known.items())):
   if pid in rows and rows[pid]['start']==old['start']and rows[pid]['command']==old['command']:
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
  if child is not None:child.wait();rec['returncode']=child.returncode
  rec.update(status='completed'if child is not None and child.returncode==0 else 'failed',ended_unix=time.time(),tracked_processes=known);save()
 if time.time()>=bound or (HERE/'STOP').exists():raise TimeoutError('Bounded experiment ended')
 if rec['returncode']!=0:raise RuntimeError(name+' failed; inspect retained evidence')

def audit_probe():
 p=CONTROL/'down-budget-023/result.json';d=json.loads(p.read_text())
 if not d['sources_unchanged']or not d['checkpoint_unchanged']:raise ValueError('Probe integrity failure')
 blocks={(r['layer'],r['group'],r['pair']):r['blocks']for r in d['microblocks']}
 for r in d['samples']:
  b=[v for v in blocks[r['layer'],r['group'],r['pair']]if v['mode']==r['mode']]
  if sum(v['calls']for v in b)!=r['calls']or r['calls']!=1024:raise ValueError('Probe count mismatch')
  if abs(sum(v['elapsed_ms']for v in b)-r['elapsed_ms'])>1e-7:raise ValueError('Probe elapsed mismatch')
 state['probe_audit']={'all_actual_counts_verified':True,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':d['status']};save()
 return d

def audit_candidate(work,cid):
 d=json.loads((work/'operator-diagnostic.json').read_text())
 if d['candidate_id']!=cid:raise ValueError('Candidate identity mismatch')
 if not all(d.get(k)is True for k in ('snapshot_unchanged','protected_sources_unchanged','checkpoint_identity_unchanged')):raise ValueError('Candidate integrity failure')
 b={(r['layer'],r['phase'],r['pair']):r['blocks']for r in d['microblocks']}
 for r in d['samples']:
  selected=[v for v in b[r['layer'],r['phase'],r['pair']]if v['lane']==r['lane']]
  if r['calls']!=1024 or sum(v['calls']for v in selected)!=r['calls']or abs(sum(v['elapsed_ms']for v in selected)-r['elapsed_ms'])>1e-7:raise ValueError('Candidate count/elapsed mismatch')
 positive=d['status']=='completed_diagnostic'and set(d['summaries'])=={'0','12','23'}and all(v['passed']for v in d['aa_screens'].values())and all(v['descriptive_bootstrap_95pct'][0]>1 for v in d['summaries'].values())
 out={'candidate_id':cid,'directory':str(work),'status':d['status'],'shortlisted':positive,'summaries':d['summaries'],'all_actual_counts_verified':True,'guards_passed':True,'raw_sha256':hashlib.sha256((work/'operator-diagnostic.json').read_bytes()).hexdigest(),'full_model_speedup_established':False}
 (work/'result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');state['candidate_outcome']=out;save();return out

def main():
 save();assertion=None;claimed=False
 try:
  prior=CONTROL/'followup-021/state.json'
  while time.time()<DEADLINE and not(HERE/'STOP').exists():
   d=json.loads(prior.read_text());row=tree().get(d['pid'])
   if 'ended_unix'in d and (row is None or str(CONTROL/'followup-021/coordinator.py')not in row['command']):break
   time.sleep(.5)
  if time.time()+60>=DEADLINE or(HERE/'STOP').exists():raise TimeoutError('Deadline/stop')
  if d['status']!='native_order_pool_exhausted':raise ValueError('Predecessor requires investigation: '+d['status'])
  claimed=True;assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
  update(next_controller=str(STATE),next_action='Immediate measured down-budget probe, then one distinct down+bias candidate only if supported; no scheduler gap.')
  probe=CONTROL/'down-budget-023'
  run('measuring_down_budget',[sys.executable,'-B','-u',str(probe/'probe.py')],probe,1800)
  d=audit_probe()
  if d['status']!='completed_operator_diagnostic':state['status']='down_budget_inconclusive';return
  rows=[d['summaries'][f'{layer}:without_down']for layer in (0,12,23)]
  if not all(v['environment_valid']and v['descriptive_95pct_saved_ms'][0]>.05 for v in rows):state['status']='down_budget_insufficient';return
  cid,work=generate(d);state['candidate_id']=cid;state['candidate_directory']=str(work);save()
  (work/'process.json').write_text(json.dumps({'status':'running','supervisor_pid':os.getpid(),'deadline_unix':min(DEADLINE,time.time()+1800),'candidate_id':cid},indent=2)+'\n')
  update(current_candidate={'candidate_id':cid,'work':str(work),'source':str(ROOT/'candidates'/cid),'stage':'pipeline_starting'})
  run('running_down_candidate',[sys.executable,'-B','-u',str(work/'pipeline.py')],work,1800)
  selected=audit_candidate(work,cid)
  if not selected['shortlisted']:state['status']='parked_down_candidate_no_operator_gain';return
  full=CONTROL/'full-model-026';full.mkdir();base=CONTROL/'memory-residency-027'
  code=(base/'run.py').read_text()
  marker="                state['summaries'][str(length)]=summary;save()"
  if code.count(marker)!=1:raise ValueError('Full-model decision anchor changed')
  code=code.replace(marker,marker+"\n                if not summary['improvement_above_measured_noise']:\n                    state['status']='stopped_context_futility';save();return")
  ast.parse(code);(full/'run.py').write_text(code)
  plan=json.loads((base/'plan.json').read_text())
  plan.update(status='execution_plan',selection='Fresh down candidate '+cid+' after complete valid operator shortlist; new candidate, no timing reroll.',candidate_id=cid,operator_evidence=str(work/'result.json'),order='Sequential after measured down budget and native/tiny/exact checks; one benchmark at a time.',completed_context_futility='Stop after20valid paired samples if existing improvement_above_measured_noise false; retain complete context and every failed sample.',residency_calibration_evidence=str(base/'state.json'))
  (full/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
  (full/'selected.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
  source=(base/'supervise.py').read_text().replace('cand_02e5f860f143',cid).replace('com.kernelmax.search.residency027.20261027',LABEL).replace('memory-residency-027',full.name)
  ast.parse(source);(full/'supervise.py').write_text(source)
  state['full_model_directory']=str(full);save()
  run('running_down_full_model',[sys.executable,'-B','-u',str(full/'supervise.py')],full,DEADLINE-time.time())
  d=json.loads((full/'state.json').read_text());state['full_model_outcome']={'status':d['status'],'full_model_speedup_established':d['full_model_speedup_established']}
  if not all(d.get(k)is True for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):raise ValueError('Full-model integrity failure')
  if d['full_model_correctness']and not all(r['passed']for r in d['full_model_correctness']):state['status']='down_full_model_numerical_failure';return
  state['status']='down_full_model_finished'
 except TimeoutError as e:state['status']='deadline_or_stop_reached';state['error']=str(e)
 except BaseException as e:state['status']='needs_investigation';state['error']=f'{type(e).__name__}: {e}';raise
 finally:
  if assertion is not None:assertion.terminate();assertion.wait()
  state['ended_unix']=time.time();save()
  if claimed:update(active_job=None,status=state['status'],down_followup=str(STATE),next_action='Immediately inspect down-followup029 state and retained stage evidence; continue justified analysis without unchanged rerolls. No commits; stop1PM.')

if __name__=='__main__':main()
