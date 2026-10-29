"""Read whole-decode command-buffer timestamps without per-stage GPU sync."""
import ctypes
import json
from pathlib import Path
import time


class Row(ctypes.Structure):
    _fields_=[('sequence',ctypes.c_uint64)]+[(name,ctypes.c_double) for name in
        ('cpu_commit','gpu_start','gpu_end','cpu_observed_complete','driver_start','driver_end')]+[('status',ctypes.c_uint64)]


class Timeline:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(library))
        self.lib.kms_install.restype=ctypes.c_char_p
        self.lib.kms_begin.restype=ctypes.c_int
        self.lib.kms_end.argtypes=[ctypes.c_int]
        self.lib.kms_end.restype=ctypes.c_int
        self.lib.kms_rows.argtypes=[ctypes.POINTER(Row),ctypes.c_int]
        self.lib.kms_rows.restype=ctypes.c_int
        self.cls=self.lib.kms_install()
        if not self.cls:raise RuntimeError('No command-buffer class')
    def begin(self):
        if self.lib.kms_begin()!=0:raise RuntimeError('Outstanding timeline activity')
    def finish(self):
        count=self.lib.kms_end(5000)
        if count<0:raise RuntimeError('Completion callbacks did not drain')
        rows=(Row*count)()
        if self.lib.kms_rows(rows,count)!=count:raise RuntimeError('Timeline count changed')
        return [{name:getattr(row,name) for name,_ in Row._fields_} for row in rows]
    def restore(self):
        if self.lib.kms_uninstall()!=0:raise RuntimeError('Cannot restore command-buffer method')


def summarize(rows,start,end):
    selected=[r for r in rows if r['gpu_start']>=start and r['gpu_end']<=end and r['gpu_end']>r['gpu_start']]
    if not selected:raise ValueError('No GPU timestamps align with CPU interval')
    spans=sorted((r['gpu_start'],r['gpu_end']) for r in selected)
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    covered=sum(b-a for a,b in merged)
    gaps=[(merged[i][0]-merged[i-1][1])*1000 for i in range(1,len(merged))]
    return {'command_buffers':len(selected),'gpu_command_interval_union_ms':covered*1000,
            'decode_wall_ms':(end-start)*1000,'wall_outside_gpu_command_intervals_ms':((end-start)-covered)*1000,
            'between_command_interval_gaps_ms':gaps,
            'warning':'Command-buffer intervals include all commands and stalls inside buffers; they are not kernel-active time. Outside intervals includes CPU work, readback, submission and observation overhead. A gap alone does not establish recoverable latency.'}


def trace_decode(model,ids,mx,make_prompt_cache,timeline):
    cache=make_prompt_cache(model)
    prompt=mx.array([ids],dtype=mx.int32);mx.eval(prompt)
    mx.clear_cache();mx.reset_peak_memory();mx.synchronize()
    model(prompt[:,:-1],cache=cache);mx.eval([c.state for c in cache]);mx.clear_cache()
    logits=model(prompt[:,-1:],cache=cache)[:,-1,:];mx.eval(logits);mx.synchronize()
    token=mx.argmax(logits,axis=-1);mx.eval(token)
    tokens=[token.item()];first_emitted=time.perf_counter();steps=[];rows=[];sampled=[]
    selected=(1,19,37,55,73,91,109,127)
    for step in range(1,128):
        if step in selected:timeline.begin()
        c0=time.process_time();t0=time.perf_counter()
        logits=model(token[:,None],cache=cache)[:,-1,:];token=mx.argmax(logits,axis=-1)
        c1=time.process_time();t1=time.perf_counter()
        mx.eval(token)
        c2=time.process_time();t2=time.perf_counter()
        tokens.append(token.item());t3=time.perf_counter()
        steps.append({'step':step,'start':t0,'built':t1,'evaluated':t2,'emitted':t3,
            'build_cpu_ms':(c1-c0)*1000,'eval_cpu_ms':(c2-c1)*1000,
            'build_ms':(t1-t0)*1000,'eval_wait_ms':(t2-t1)*1000,'readback_ms':(t3-t2)*1000})
        if step in selected:
            # Read completed-buffer metadata after the existing token wait.
            # Retain at most one step of buffers; add no GPU synchronization.
            captured=timeline.finish()
            for row in captured:row['sampled_step']=step
            rows.extend(captured)
            sampled.append({'step':step,**summarize(captured,t0,t3)})
    mx.synchronize()
    covered=sum(v['gpu_command_interval_union_ms'] for v in sampled)
    wall=sum(v['decode_wall_ms'] for v in sampled)
    summary={'sampled_steps':list(selected),'command_buffers':sum(v['command_buffers'] for v in sampled),
        'sampled_decode_wall_ms':wall,'sampled_gpu_command_interval_union_ms':covered,
        'sampled_wall_outside_gpu_command_intervals_ms':wall-covered,
        'mean_sampled_wall_ms_per_token':wall/len(sampled),
        'mean_sampled_gpu_command_interval_ms_per_token':covered/len(sampled),
        'mean_sampled_outside_gpu_command_intervals_ms_per_token':(wall-covered)/len(sampled),
        'complete127interval_ms_per_token':(steps[-1]['emitted']-first_emitted)*1000/127,
        'warning':'Eight fixed complete decode steps are sampled across the full128token request. Buffer intervals include stalls, not kernel-active time. Outside time is not automatically recoverable. Metadata observation between sampled steps is excluded from their step timers but included in complete-request cadence.'}
    return {'tokens':tokens,'steps':steps,'command_buffers':rows,'sampled_step_summaries':sampled,'summary':summary}
