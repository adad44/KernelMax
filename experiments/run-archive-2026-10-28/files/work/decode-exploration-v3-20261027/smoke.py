"""Native passive-telemetry API smoke check; not an inference benchmark."""
import json
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
import mlx.core as mx
from evaluator.measurement import PhaseMemoryObserver

observer=PhaseMemoryObserver(513,mx,interval_seconds=3600)
observer.start()
observer.before_model_call(512)
observer.before_model_call(1)
observer.before_model_call(1)
result=observer.finish()
assert set(result['phase_deltas'])=={'prefill','decode'}
assert not result['errors']
assert all(not row['errors'] for row in result['samples'])
assert all(row['vm']['page_size_bytes']>0 for row in result['samples'])
(WORK/'telemetry-smoke.json').write_text(json.dumps({'passed':True,'scope':'read-only native telemetry API only; no candidate correctness/performance claim','result':result},indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':True,'phases':list(result['phase_deltas']),'samples':len(result['samples']),'pressure_codes':[s['memory_pressure_level_raw'] for s in result['samples']]}))
