"""Small native-MXFP4 model checks for the controller, not model evidence."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gpt_oss import Model, ModelArgs
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.models.cache import make_prompt_cache
from profile_decode import Hooks, request, write

mx.random.seed(42)
args = ModelArgs(num_hidden_layers=2, num_local_experts=8, num_experts_per_tok=4,
                 vocab_size=256, hidden_size=64, intermediate_size=64,
                 head_dim=32, num_attention_heads=2, num_key_value_heads=1)
model = Model(args)
model.apply(lambda a: a.astype(mx.bfloat16))
nn.quantize(model, group_size=32, bits=4, mode='mxfp4',
            class_predicate=lambda path, module: isinstance(module, SwitchLinear))
mx.eval(model.parameters())
ids = list(range(128))*4
reference, _ = request(model, ids, mx, make_prompt_cache)
hooks = Hooks(model, mx)
results = []
try:
    hooks.mode = 'capture'
    captured, logits = request(model, ids, mx, make_prompt_cache, hooks)
    assert captured['tokens'] == reference['tokens']
    assert len(hooks.saved) == 127*2*7
    for mode in ('normal', 'oracle_pointwise', 'oracle_up_gate', 'oracle_down', 'oracle_experts', 'serial'):
        hooks.mode = mode
        out, _ = request(model, ids, mx, make_prompt_cache, hooks, logits)
        assert out['tokens'] == reference['tokens'], mode
        assert out['max_abs_logit_error'] == 0, (mode, out['max_abs_logit_error'])
        results.append({'mode': mode, 'tokens_equal': True, 'max_abs_logit_error': 0})
    assert len(hooks.serial) == 16*(2*8+3), len(hooks.serial)
finally:
    hooks.restore()
write(Path(__file__).with_suffix('.json'), {'scope': 'tiny random model; controller validation only',
                                           'saved_entries': len(hooks.saved), 'checks': results,
                                           'serial_events': len(hooks.serial)})
print(json.dumps(results))
