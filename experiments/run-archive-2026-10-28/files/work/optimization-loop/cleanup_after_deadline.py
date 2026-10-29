"""After the fixed deadline, unload this search's jobs and stop only owned processes."""
import datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent
DEADLINE=json.loads((W/'owned-jobs.json').read_text())['deadline_unix']
if time.time()<DEADLINE:raise SystemExit('Before hard deadline; no cleanup performed')
PYTHON='/Users/alandiaz/Downloads/KernelMaxxxing/runtime/gpt-oss/.venv/bin/python'
def processes():
 out={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
  f=line.split(None,7)
  if len(f)==8:out[int(f[0])]={'ppid':int(f[1]),'start':' '.join(f[2:7]),'command':f[7]}
 return out

def owned(rows):
 ids=set()
 for pid,r in rows.items():
  cmd=r['command']
  if pid==os.getpid()or'build_evidence_index.py'in cmd or'cleanup_after_deadline.py'in cmd:continue
  if cmd.startswith(PYTHON+' ')and(str(W)+'/'in cmd):ids.add(pid)
 changed=True
 while changed:
  changed=False
  for pid,r in rows.items():
   if r['ppid']in ids and pid not in ids and pid!=os.getpid():ids.add(pid);changed=True
 for pid,r in rows.items():
  tokens=r['command'].split()
  if tokens and tokens[0]=='/usr/bin/caffeinate'and'-w'in tokens:
   i=tokens.index('-w')+1
   if i<len(tokens)and tokens[i].isdigit()and int(tokens[i])in ids:ids.add(pid)
 return ids

before=processes();captured={pid:before[pid]for pid in owned(before)}
result={'deadline_unix':DEADLINE,'cleanup_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'captured_task_processes':captured,'job_unloads':[],'terminated_processes':[],'unrelated_services_touched':False}
for job in json.loads((W/'owned-jobs.json').read_text())['jobs']:
 if not job['label'].startswith('com.kernelmax.search.')or not any(str(W)+'/'in arg for arg in job['program']):raise ValueError('Unowned job manifest entry')
 r=subprocess.run(['/bin/launchctl','bootout','gui/501/'+job['label']],capture_output=True,text=True,timeout=10)
 result['job_unloads'].append({'label':job['label'],'returncode':r.returncode,'output':(r.stdout+r.stderr).strip()})
# Catch any task-owned orphan that outlived its bounded parent; verify identity.
for attempt in range(3):
 current=processes()
 for pid in owned(current):captured.setdefault(pid,current[pid])
 for pid,old in list(captured.items()):
  row=current.get(pid)
  if row and row['start']==old['start']and row['command']==old['command']:
   try:os.kill(pid,signal.SIGKILL);result['terminated_processes'].append({'pid':pid,**old})
   except ProcessLookupError:pass
 if not owned(processes()):break
 time.sleep(.2)
latest=processes();remaining_ids=owned(latest)
result['remaining_owned_processes']={pid:latest[pid]for pid in remaining_ids}
result['cleanup_finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(W/'deadline-cleanup.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
p=W/'search-20261027.json'
try:s=json.loads(p.read_text())
except json.JSONDecodeError:
 (W/'search-state-interrupted.json').write_bytes(p.read_bytes());s=json.loads((W/'search-last-good-snapshot.json').read_text());s['root_state_recovered_from_last_good_snapshot']=True
s.update(status='deadline_reached',active_job=None,deadline_cleanup=str(W/'deadline-cleanup.json'),search_ended_utc=result['cleanup_finished_utc'],next_action='No new experiments: the20:00UTC hard deadline has passed. Retain partial evidence, rebuild evidence-index.json, pause this heartbeat, and report measured outcomes without acceptance or model-speedup claims from incomplete data.')
s['current_candidate']['execution_status']='stopped_or_finished_by_deadline';t=p.with_suffix('.deadline-tmp');t.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');t.replace(p)
print(json.dumps({'jobs_checked':len(result['job_unloads']),'jobs_unloaded':sum(r['returncode']==0 for r in result['job_unloads']),'remaining_task_processes':len(result['remaining_owned_processes']),'cleanup_report':str(W/'deadline-cleanup.json')}))
if result['remaining_owned_processes']:raise SystemExit('Task-owned processes remain; inspect cleanup report')
