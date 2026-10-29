"""Read retained reports into a compact index; never grade partial timing as a win."""
import datetime,json,subprocess
from pathlib import Path
W=Path(__file__).resolve().parent
native={}
for p in W.glob('**/native-correctness/artifacts/*/report.json'):
 d=json.loads(p.read_text());cid=d['candidate']['candidate_id'];passed=d['status']=='completed'and not d['errors']and len(d['correctness'])==7 and all(r['passed']for r in d['correctness'])
 native.setdefault(cid,[]).append({'report':str(p),'snapshot_sha256':d['candidate']['sha256'],'status':d['status'],'protected_native_suite_passed':passed,'errors':d['errors']})
models=[];exact={}
for p in W.glob('**/state.json'):
 d=json.loads(p.read_text())
 if not all(k in d for k in ['candidate_id','full_model_correctness','requests','aa_screens']):continue
 cid=d['candidate_id'];contexts=[r['length']for r in d['full_model_correctness']if r['passed']and'max_abs_error=0;'in r['detail']and r['all_routes_equal']and r['all_greedy_tokens_equal']]
 exact[cid]=sorted(set(exact.get(cid,[])+contexts))
 models.append({'candidate_id':cid,'state':str(p),'status':d['status'],'bit_exact_contexts':contexts,'timed_requests':len(d['requests']),'invalid_requests':sum(bool(r.get('invalid_reasons'))for r in d['requests']),'complete_context_summaries':{k:{x:v[x]for x in ['median_ratio','descriptive_bootstrap_95pct','improvement_above_measured_noise']}for k,v in d['summaries'].items()},'aa_contexts':{k:v['passed']for k,v in d['aa_screens'].items()},'full_model_speedup_established':d.get('full_model_speedup_established',False),'errors':d['errors']})
operators=[]
for p in W.glob('**/operator-diagnostic.json'):
 d=json.loads(p.read_text());operators.append({'candidate_id':d['candidate_id'],'raw':str(p),'status':d['status'],'samples':len(d['samples']),'guards_passed':all(d.get(k)is True for k in ['snapshot_unchanged','protected_sources_unchanged','checkpoint_identity_unchanged']),'layer_results':{k:{x:v[x]for x in ['median_ratio','descriptive_bootstrap_95pct','environment_valid']}for k,v in d['summaries'].items()},'errors':d['errors']})
s=json.loads((W/'search-20261027.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W.parent.parent/'kernelmax-first-loop',text=True).strip();out={'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'This scheduled search under work/optimization-loop; earlier three parked reference candidates are outside this count.','unique_candidates_with_protected_native_reports':len(native),'candidates_passing_protected_native_suite':sum(any(r['protected_native_suite_passed']for r in v)for v in native.values()),'native_reports':native,'operator_attempts':operators,'full_model_attempts':models,'candidates_bit_exact_on_all3full_model_contexts':[cid for cid,lengths in exact.items()if lengths==[512,2048,4096]],'full_model_speedup_established':any(r['full_model_speedup_established']for r in models),'git_head':head,'expected_unchanged_git_head':'fc1dc28a4f18a12340668af2cee5cea9a0d6cbe7','git_head_unchanged':head=='fc1dc28a4f18a12340668af2cee5cea9a0d6cbe7','acceptance_enabled':False,'execution_gap':s.get('execution_gap'),'deadline_unix':s['deadline_unix'],'notes':['Operator improvements are isolated and require independent model validation.','A stopped/partial model pass is not a measured model speedup.','Native fixtures and numerical equality do not satisfy the planned held-out task-quality gate.','Historical failures, samples and source snapshots are retained.']}
(W/'evidence-index.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:out[k]for k in ['as_of_utc','unique_candidates_with_protected_native_reports','candidates_passing_protected_native_suite','candidates_bit_exact_on_all3full_model_contexts','full_model_speedup_established','git_head_unchanged']}))
