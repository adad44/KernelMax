"""Lossless native up/gate row packing; unchanged native matrix arithmetic."""
import weakref

import mlx.core as mx
from mlx_lm.models.gpt_oss import SwiGLU


# Model-parameter representation only. No activations, IDs or outputs are cached.
_PACKED={}


def _fields(experts):
    return tuple(getattr(getattr(experts,name),field)
                 for name in ('up_proj','gate_proj')
                 for field in ('weight','scales','bias'))


def _signature(experts):
    return tuple(id(value) for value in _fields(experts))


def _validate(experts):
    up,gate,down=experts.up_proj,experts.gate_proj,experts.down_proj
    if type(experts.activation) is not SwiGLU:
        raise ValueError('Native GPT-OSS SwiGLU required')
    k,n,e=up.input_dims,up.output_dims,up.num_experts
    if (gate.input_dims,gate.output_dims,gate.num_experts)!=(k,n,e):
        raise ValueError('Up/gate dimensions disagree')
    if (down.input_dims,down.output_dims,down.num_experts)!=(n,k,e):
        raise ValueError('Down dimensions disagree')
    for p in (up,gate,down):
        if (p.mode,p.bits,p.group_size)!=('mxfp4',4,32):
            raise ValueError('Native MXFP4 required')
    for p in (up,gate):
        if (p.weight.dtype!=mx.uint32 or p.scales.dtype!=mx.uint8
            or p.bias.dtype!=mx.bfloat16 or p.get('biases') is not None
            or p.weight.shape!=(e,n,k//8) or p.scales.shape!=(e,n,k//32)
            or p.bias.shape!=(e,n)):
            raise ValueError('Unsupported native parameter layout')
    return k,n,e


def prepare_candidate(experts):
    """Pack once, then replace original fields with value-identical views.

    The packed buffers own the original logical storage volume. Old contiguous
    parameters are released rather than retained alongside a model-size copy.
    Callers must account for this setup and its transient allocation separately.
    """
    key=id(experts)
    record=_PACKED.get(key)
    if record is not None and record['owner']() is experts and record['signature']==_signature(experts):
        return record
    k,n,e=_validate(experts)
    up,gate=experts.up_proj,experts.gate_proj
    with mx.stream(mx.cpu):
        weight=mx.concatenate([up.weight,gate.weight],axis=1)
        scales=mx.concatenate([up.scales,gate.scales],axis=1)
        bias=mx.concatenate([up.bias,gate.bias],axis=1)
        mx.eval(weight,scales,bias)
        up.weight,gate.weight=weight[:,:n,:],weight[:,n:,:]
        up.scales,gate.scales=scales[:,:n,:],scales[:,n:,:]
        up.bias,gate.bias=bias[:,:n],bias[:,n:]
        mx.eval(_fields(experts))
    record={'owner':weakref.ref(experts,lambda ref,key=key:_PACKED.pop(key,None)),
            'weight':weight,'scales':scales,'bias':bias,'n':n,
            'signature':_signature(experts)}
    _PACKED[key]=record
    return record


def prepare_native(experts):
    """Restore contiguous native fields for a fair original-layout baseline."""
    key=id(experts)
    if key not in _PACKED:
        return
    # Always preserve CURRENT values, including parameter updates after packing.
    with mx.stream(mx.cpu):
        restored=tuple(mx.contiguous(value) for value in _fields(experts))
        mx.eval(restored)
        index=0
        for name in ('up_proj','gate_proj'):
            projection=getattr(experts,name)
            for field in ('weight','scales','bias'):
                setattr(projection,field,restored[index]);index+=1
    _PACKED.pop(key,None)


def run(experts,x,selected_expert_ids):
    if x.ndim!=3 or x.shape[:2]!=(1,1) or selected_expert_ids.shape!=(1,1,4):
        return experts(x,selected_expert_ids)
    if x.dtype!=mx.bfloat16 or selected_expert_ids.dtype not in (mx.int32,mx.uint32):
        raise ValueError('BF16 decode and int32/uint32 routing IDs required')
    packed=prepare_candidate(experts)
    if x.shape[-1]!=experts.up_proj.input_dims:
        raise ValueError('Input dimensions disagree')
    expanded=mx.expand_dims(x,(-2,-3))
    projected=mx.gather_qmm(expanded,packed['weight'],packed['scales'],
        rhs_indices=selected_expert_ids,transpose=True,group_size=32,bits=4,
        mode='mxfp4',sorted_indices=False)
    projected=projected+mx.expand_dims(packed['bias'][selected_expert_ids],-2)
    n=packed['n']
    activated=experts.activation(projected[...,:n],projected[...,n:])
    return experts.down_proj(activated,selected_expert_ids,sorted_indices=False).squeeze(-2)
