"""CPU scheduling fixture: check emitted sequence, exact call count and drain."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
from evaluator.native_baseline import request

spec=importlib.util.spec_from_file_location('new_decode_candidate',ROOT/'candidates/cand_75ca936df290/implementation.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


class Tensor:
    def __init__(self,data,backend,dtype=None):self.data=np.asarray(data);self.backend=backend;self.dtype=dtype
    @property
    def shape(self):return self.data.shape
    def __getitem__(self,index):return Tensor(self.data[index],self.backend,self.dtype)
    def item(self):
        value=self.data.item();self.backend.events.append(('read',value));return value


class Backend:
    int32='int32'
    def __init__(self):self.events=[]
    def array(self,x,dtype=None):return Tensor(np.array(x,dtype=np.int32),self,dtype)
    def eval(self,*values):self.events.append(('eval',))
    def async_eval(self,*values):self.events.append(('async',))
    def synchronize(self):self.events.append(('drain',))
    def clear_cache(self):pass
    def reset_peak_memory(self):pass
    def get_peak_memory(self):return 1024
    def argmax(self,x,axis):return Tensor(np.argmax(x.data,axis=axis),self,'int32')
    def isfinite(self,x):return Tensor(np.isfinite(x.data),self)
    def all(self,x):return Tensor(np.all(x.data),self)


class Cache:
    def __init__(self,mx):self.offset=0;self.total=0;self.state=[Tensor([0],mx,'mlx.core.bfloat16')]


class Model:
    def __init__(self,mx):self.mx=mx;self.calls=[];self.last_cache=None
    def __call__(self,x,cache):
        c=cache[0];self.last_cache=c;self.calls.append(list(x.shape))
        c.offset+=x.shape[1];c.total+=int(np.sum(x.data))
        value=(c.total+c.offset)%17
        out=np.zeros((1,x.shape[1],17),dtype=np.float32);out[...,value]=1
        return Tensor(out,self.mx,'mlx.core.bfloat16')


records=[]
for length,step_size in ((1,4096),(512,4096),(2048,4096),(4096,4096),(19,3)):
    ids=[i%17 for i in range(length)]
    lanes={}
    for lane,fn in (('reference',request),('candidate',candidate.decode_request)):
        mx=Backend();model=Model(mx)
        result=fn(model,ids,step_size,mx,lambda _: [Cache(mx)])
        lanes[lane]=(result,model,mx)
    ref,a,_=lanes['reference'];actual,b,mx=lanes['candidate']
    if actual['output_token_ids']!=ref['output_token_ids']:raise ValueError('Token sequence changed')
    if a.calls!=b.calls or len(actual['output_token_ids'])!=128:raise ValueError('Missing or extra model work')
    if (a.last_cache.offset,a.last_cache.total)!=(b.last_cache.offset,b.last_cache.total):raise ValueError('Cache behavior changed')
    asynchronous=[i for i,e in enumerate(mx.events) if e[0]=='async']
    reads=[i for i,e in enumerate(mx.events) if e[0]=='read']
    if len(asynchronous)!=127 or not reads[0]<asynchronous[0]:raise ValueError('First token delayed or wrong submission count')
    # Last emitted token is read before the final drain, then validation reads.
    if not any(e[0]=='drain' for e in mx.events[reads[127]+1:reads[128]]):raise ValueError('Final GPU drain omitted')
    obs=actual['timing_observations']
    if obs['tokens_at'][-1]!=obs['gpu_drained_at'] or obs['gpu_drained_at']<obs['last_token_observed_at']:
        raise ValueError('Final drain excluded from reported timing')
    records.append({'prompt':length,'step_size':step_size,'tokens':128,'async_submissions':127,
                    'model_calls':len(b.calls),'same_reference_forward_shapes':True,'same_fake_cache':True,
                    'same_fake_greedy_tokens':True,'first_token_before_async':True,'final_drain_included':True})
(WORK/'candidate-fixture.json').write_text(json.dumps({'scope':'CPU scheduling fixture only; no native/GPU correctness or performance claim','records':records,'passed':True},indent=2)+'\n')
print(json.dumps({'passed':True,'cases':len(records),'native_gpu_validation':'pending'}))
