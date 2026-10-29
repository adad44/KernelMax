"""Tiny native observer validation; run only after the profiler has exited."""
import json
import hashlib
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ps=subprocess.run(['ps','-p','71655','-o','lstart=,command='],text=True,capture_output=True)
if ps.returncode==0 and 'cooled-profile/profile.py' in ps.stdout:
    raise RuntimeError('Profiler still live; GPU fixture deferred to avoid overlap')

import mlx.core as mx
from probe import Timeline,summarize

x=mx.arange(65536,dtype=mx.float32)/65536
expected=mx.sin(x)+mx.cos(x);mx.eval(expected);mx.synchronize()
timeline=Timeline(HERE/'command_timeline.dylib')
try:
    timeline.begin();start=time.perf_counter()
    actual=mx.sin(x)+mx.cos(x);mx.eval(actual);mx.synchronize()
    end=time.perf_counter();rows=timeline.finish()
    summary=summarize(rows,start,end)
    passed=bool(mx.all(actual==expected).item())
    if not passed or any(r['status']!=4 for r in rows):raise ValueError('Observer output/status mismatch')
    timeline.restore()
    if timeline.lib.kms_begin()!=-1:raise ValueError('Observer remained installed')
    result={'passed':True,'command_buffer_class':timeline.cls.decode(),'rows':rows,
        'summary':summary,'original_method_restored':True,'output_exact':passed,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),HERE/'command_timeline.m',HERE/'command_timeline.dylib',HERE/'probe.py')}}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'class':timeline.cls.decode(),'buffers':len(rows),'summary':summary}))
except BaseException:
    try:timeline.finish();timeline.restore()
    except Exception:pass
    raise
