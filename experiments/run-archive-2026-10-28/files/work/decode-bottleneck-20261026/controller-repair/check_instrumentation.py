"""Small native-MXFP4 model checks for the controller, not model evidence."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gpt_oss import Model, ModelArgs
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.models.cache import make_prompt_cache
from profile_decode import BASE, Hooks, request, write, candidate_entry, up_gate_region, digest
from orchestrator.candidate_manager import FrozenCandidate, validate_candidate, verify_frozen_candidate

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
reference, reference_logits = request(model, ids, mx, make_prompt_cache, retain_logits=True)
hooks = Hooks(model, mx)
results = []
try:
    hooks.mode = 'capture'
    captured, logits = request(model, ids, mx, make_prompt_cache, hooks, reference_logits)
    assert captured['tokens'] == reference['tokens']
    assert captured['max_abs_logit_error'] == 0
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
entries = []
for dirname in ('first-optimization-20261026', 'second-optimization-20261026'):
    approval = json.loads((BASE/'work'/dirname/'approval.json').read_text())
    c = validate_candidate(Path(approval['snapshot']), Path(approval['snapshot']).parent)
    hashes = {p.relative_to(c.directory).as_posix(): digest(p) for p in c.directory.rglob('*') if p.is_file()}
    frozen = FrozenCandidate(**vars(c), sha256=approval['snapshot_sha256'], file_sha256=hashes)
    verify_frozen_candidate(frozen)
    entry = candidate_entry(frozen)
    region, source = up_gate_region(entry)
    entries.append(entry)
    for step in (1, 55, 127):
        x = hooks.saved[(step, 0, 'x')]
        ids = hooks.saved[(step, 0, 'ids')]
        if len(entries) == 1:
            ids = ids.astype(mx.int32)
        full = entry(model.model.layers[0].mlp.experts, x, ids)
        activated = region(model.model.layers[0].mlp.experts, x, ids)
        expected_full = hooks.saved[(step, 0, 'experts')]
        expected_act = hooks.saved[(step, 0, 'activation')]
        for actual, expected in ((full, expected_full), (activated, expected_act)):
            assert mx.all(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32)) <= .02+.02*mx.abs(expected.astype(mx.float32))).item()
    results.append({'candidate_id': frozen.candidate_id, 'module': entry.__module__,
                    'source_file': entry.__code__.co_filename, 'full_and_region_checks': 6})
assert entries[0] is not entries[1]
assert entries[0].__module__ != entries[1].__module__
write(Path(__file__).with_suffix('.json'), {'scope': 'tiny random model; controller validation only',
                                           'saved_entries': len(hooks.saved), 'checks': results,
                                           'serial_events': len(hooks.serial)})
print(json.dumps(results))
