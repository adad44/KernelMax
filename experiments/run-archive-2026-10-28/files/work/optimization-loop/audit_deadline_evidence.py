"""Post-deadline integrity/count audit; no inference or timing statistics."""
import collections,datetime,hashlib,json,time
from pathlib import Path
W=Path(__file__).resolve().parent
root=json.loads((W/'search-20261027.json').read_text())
if time.time()<root['deadline_unix']:raise SystemExit('Audit only after experiment deadline')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
records=[]
for p in W.glob('full-model-038-*/state.json'):
 s=json.loads(p.read_text());selection=json.loads(Path(s['operator_selection']).read_text());a=json.loads((Path(selection['directory'])/'approval.json').read_text());request=json.loads((Path(a['native_report']).parent/'candidate/request.json').read_text());snap=request['snapshot'];sd=Path(snap['directory'])
 source_mismatches=[name for name,sha in s['protected_sources_before'].items()if not Path(name).is_file()or digest(Path(name))!=sha]
 snapshot_mismatches=[name for name,sha in snap['file_sha256'].items()if not(sd/name).is_file()or digest(sd/name)!=sha]
 receipt=s['checkpoint_verification'];mp=Path(receipt['model_path']);identity_mismatches=[]
 for name,expected in receipt['file_identity'].items():
  try:
   st=(mp/name).stat();actual=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  except OSError:actual=None
  if actual!=expected:identity_mismatches.append(name)
 arrays_mismatches=[r['arrays']for r in s['full_model_correctness']if digest(Path(r['arrays']))!=r['arrays_sha256']]
 counts=collections.Counter(r['phase']for r in s['requests']);lanes=collections.Counter((r['length'],r['phase'],r['pair'])for r in s['requests']if r['phase']=='candidate')
 reference={}
 for r in s['requests']:
  if r['phase']=='aa':reference.setdefault(r['length'],r['output_token_ids'])
 structural_issues=[]
 for i,r in enumerate(s['requests']):
  if len(r['output_token_ids'])!=128 or len(r['inter_token_intervals_ms'])!=127 or not r['last_logits_finite']:structural_issues.append({'request':i,'issue':'incomplete128output127intervals'})
  if r['length']in reference and r['output_token_ids']!=reference[r['length']]:structural_issues.append({'request':i,'issue':'greedy_tokens_differ_from_reference'})
 records.append({'state':str(p),'candidate_id':s['candidate_id'],'snapshot_sha256':s['snapshot_sha256'],'raw_status':s['status'],'protected_source_mismatches':source_mismatches,'snapshot_mismatches':snapshot_mismatches,'checkpoint_identity_mismatches':identity_mismatches,'correctness_array_mismatches':arrays_mismatches,'request_structure_or_token_issues':structural_issues,'request_counts_by_phase':dict(counts),'complete_candidate_pairs_by_context':{str(length):sum(n==2 for(length2,phase,pair),n in lanes.items()if length2==length)for length in [512,2048,4096]},'invalid_requests':sum(bool(r.get('invalid_reasons'))for r in s['requests']),'resident_primers':len(s['resident_primers']),'aa_screens':s['aa_screens'],'complete_context_summaries':s['summaries'],'raw_final_guard_presence':{k:k in s for k in ['expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged']},'raw_errors':s['errors'],'full_model_speedup_established':s.get('full_model_speedup_established',False)})
out={'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only post-deadline audit. Preserved partial requests are not a complete comparison; no partial statistical result or acceptance is inferred. In-memory dispatch restoration cannot be audited after process exit.','model_runs':records}
(W/'deadline-evidence-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out))
