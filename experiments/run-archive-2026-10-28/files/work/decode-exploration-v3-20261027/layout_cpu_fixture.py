"""CPU proof for exact BF16 word storage; never imports MLX or runs inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parent
CANDIDATE=WORK.parent.parent/'kernelmax-first-loop/candidates/cand_390712b41518'
spec=importlib.util.spec_from_file_location('lossless_layout_codec',CANDIDATE/'codec.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
rng=np.random.default_rng(42)
all_words=np.arange(65536,dtype=np.uint16)
cases=[('every_word_ordered',all_words.reshape(1024,64)),
       ('every_word_shuffled',rng.permutation(all_words).reshape(1024,64))]
hi=rng.integers(58,65,size=(27,192),dtype=np.uint16)
sign=rng.integers(0,2,size=hi.shape,dtype=np.uint16)
low=rng.integers(0,256,size=hi.shape,dtype=np.uint16)
mixed=low|((hi|(sign<<7))<<8)
mixed.reshape(-1,64)[::7,0]=0
mixed.reshape(-1,64)[1::11,0]=0x8000
cases.append(('mixed_normal_and_raw_blocks',mixed))
records=[]
for label,words in cases:
    packed=codec.pack_bf16_bits(words)
    decoded=codec.unpack_bf16_bits(packed)
    if not np.array_equal(words,decoded):raise ValueError(label+' lost BF16 bits')
    records.append({'case':label,'words':words.size,'all_words_bit_exact':True,**codec.statistics(packed)})
if records[0]['escaped_blocks']!=0 or records[1]['escaped_blocks']==0:
    raise ValueError('Both compressed and escape branches must be exercised')
if not 0<records[2]['escaped_blocks']<records[2]['blocks']:
    raise ValueError('Mixed branch fixture did not contain both layouts')
single=codec.pack_bf16_bits(mixed)
chunked=codec.pack_chunks((mixed[:1],mixed[1:6],mixed[6:13],mixed[13:]),mixed.shape)
for key in ('payload','headers','escapes'):
    if not np.array_equal(single[key],chunked[key]):raise ValueError('Chunk escape indexing differs: '+key)
if not np.array_equal(codec.unpack_bf16_bits(chunked),mixed):raise ValueError('Chunk roundtrip failed')
invalid=[np.zeros((4,96),np.uint16),np.zeros((4,64),np.float32)]
for value in invalid:
    try:codec.pack_bf16_bits(value)
    except ValueError:pass
    else:raise ValueError('Unsupported codec input silently accepted')
for chunks in ((mixed[:1],),(mixed,mixed)):
    try:codec.pack_chunks(iter(chunks),mixed.shape)
    except ValueError:pass
    else:raise ValueError('Incomplete/oversized chunks silently accepted')
result={'passed':True,'scope':'CPU integer-word representation only; native GPU/projection/full-model correctness and performance pending',
    'records':records,'chunk_escape_indices_match':True,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CANDIDATE.iterdir() if p.is_file()}}
(WORK/'layout-cpu-fixture.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':True,'cases':len(cases),'words_checked':sum(r['words'] for r in records),'chunked':True,'native_gpu_pending':True}))
