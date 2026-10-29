"""Recompute probe diagnostics from raw counts; preserve the original artifact."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics as st

WORK=Path(__file__).resolve().parent
path=WORK/'result.json'
data=json.loads(path.read_text())
blocks={(r['layer'],r['group'],r['pair']):r['blocks'] for r in data['microblocks']}
groups=defaultdict(list)
for row in data['samples']:
    matching=[b for b in blocks[row['layer'],row['group'],row['pair']] if b['mode']==row['mode']]
    assert sum(b['calls'] for b in matching)==row['calls']==4096
    assert abs(sum(b['elapsed_ms'] for b in matching)-row['elapsed_ms'])<1e-7
    groups[row['layer'],row['group']].append(row)
result={'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':data['status'],
        'summary_correction':'Original summary divided by 256 despite 4096 actual calls. Recomputed below from each raw sample calls. Original artifact preserved.',
        'all_recorded_samples_call_counts_verified':True,'raw_samples':len(data['samples']),
        'original_integrity':{k:data.get(k) for k in ('sources_unchanged','checkpoint_unchanged')},
        'regions':{}}
for (layer,group),rows in groups.items():
    if group=='aa':
        continue
    a={r['pair']:r for r in rows if r['mode']=='native'}
    b={r['pair']:r for r in rows if r['mode']==group}
    pairs=sorted(a.keys()&b.keys())
    savings=[a[p]['elapsed_ms']/a[p]['calls']-b[p]['elapsed_ms']/b[p]['calls'] for p in pairs]
    fractions=[1-(b[p]['elapsed_ms']/b[p]['calls'])/(a[p]['elapsed_ms']/a[p]['calls']) for p in pairs]
    rng=random.Random(42+layer)
    bootstrap=sorted(st.median(rng.choices(savings,k=len(savings))) for _ in range(10000))
    issues=sorted({i for r in rows for i in r['environment_issues']})
    integrity=all(data.get(k) is True for k in ('sources_unchanged','checkpoint_unchanged'))
    usable=(len(pairs)==20 and len(rows)==40 and not issues
            and data['screens'].get(str(layer),{}).get('passed') is True and integrity)
    result['regions'][f'{layer}:{group}']={
        'complete_pairs':len(pairs),'calls_per_lane_sample':4096,
        'native_ms_per_call':st.median(a[p]['elapsed_ms']/a[p]['calls'] for p in pairs),
        'deleted_ms_per_call':st.median(b[p]['elapsed_ms']/b[p]['calls'] for p in pairs),
        'median_saved_ms_per_call':st.median(savings),'descriptive_95pct_saved_ms':[bootstrap[249],bootstrap[9749]],
        'median_isolated_operator_fraction':st.median(fractions),'environment_issues':issues,
        'usable_as_sampled_operator_deletion_evidence':usable,'full_model_contribution_established':False,
        'paired_savings_ms':savings}
(WORK/'audited-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='regions'},indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='paired_savings_ms'} for k,v in result['regions'].items()},indent=2))
