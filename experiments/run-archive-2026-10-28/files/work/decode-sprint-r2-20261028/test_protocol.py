"""CPU-only behavioral checks of the actual nested request collector."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class NumericalFailure(RuntimeError):
    pass


class EnvironmentFailure(RuntimeError):
    pass


class ProtocolTests(unittest.TestCase):
    def harness(self, *, primer_swap=False, measured_swap=False, primer_hard=False,
                primer_bad_tokens=False, limit=800):
        source = ast.parse(Path(__file__).with_name('run_round.py').read_text())
        collect = next(node for node in ast.walk(source)
                       if isinstance(node, ast.FunctionDef) and node.name == 'collect')
        wrapper = ast.parse('def build():\n request_count=0\n def count(): return request_count\n return collect,count').body[0]
        wrapper.body.insert(1, collect)
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        events = []
        cs = {'resident_primers': [], 'requests': [], 'phase_memory_requests': []}
        flags_calls = []
        environment_number = 0
        def environment():
            nonlocal environment_number
            environment_number += 1
            events.append('environment')
            return {'thermal_state': 'nominal', 'number': environment_number}
        def assess(before, after, *, mode, **kwargs):
            flags_calls.append(mode)
            number = len(flags_calls)
            swap = (number == 2 and primer_swap) or (number == 4 and measured_swap)
            hard = number == 2 and primer_hard
            return {'continue_pass': not hard and not (swap and mode == 'confirmation'),
                    'swap_affected': swap, 'flags': ['swap'] if swap else ['thermal'] if hard else []}
        class Audit:
            def __init__(self, model): self.calls = []
        class Binding:
            counts = {}
            @contextlib.contextmanager
            def lane(self, enabled):
                self.counts = {'projection_specialized': 128 if enabled else 0}
                yield
        binding = Binding()
        def request(audit, *args):
            events.append('request')
            audit.calls = [[1, 511]] + [[1, 1]] * 128
            tokens = [99] * 128 if primer_bad_tokens and events.count('request') == 1 else [42] * 128
            return {'output_token_ids': tokens, 'metrics': {'decode_tokens_per_second': 25.0}}
        def store(): events.append('store')
        namespace = dict(operation_check=lambda: events.append('check'), max_requests=limit,
                         plan={'confirmation': {'before_each_request_idle_seconds': 27}},
                         time=SimpleNamespace(sleep=lambda seconds: events.append(('sleep', seconds))),
                         mode='confirm', environment=environment, assess_environment=assess,
                         threading=SimpleNamespace(Event=Mock, Thread=Mock), thermal_state=lambda: 'nominal',
                         AuditModel=Audit, model=object(), binding=binding, request=request,
                         inputs={'512': list(range(512))}, mx=None, make_prompt_cache=None,
                         swap_observation=lambda *args: {}, reference_tokens={512: [42]*128},
                         cs=cs, EnvironmentFailure=EnvironmentFailure, NumericalFailure=NumericalFailure,
                         ObservedModel=None, PhaseMemoryObserver=None, adapter=SimpleNamespace(kind='vocabulary'), residency=None,
                         store=store, print=lambda *args, **kwargs: None)
        exec(compile(module, '<actual collector>', 'exec'), namespace)
        collect_fn, count = namespace['build']()
        return collect_fn, count, cs, events, flags_calls

    def test_unconditional_same_lane_primer_then_fresh_measured_request(self):
        collect, count, cs, events, modes = self.harness(primer_swap=True)
        row = collect(512, 'candidate', 'confirm-1', 1)
        self.assertEqual(count(), 2)
        self.assertEqual(modes, ['exploration', 'exploration', 'confirmation', 'confirmation'])
        self.assertEqual(events.count(('sleep', 27)), 1)
        self.assertEqual(events.count('request'), 2)
        # Persistence cannot interpose between the declared hydration and measurement.
        self.assertGreater(events.index('store'), max(i for i, x in enumerate(events) if x == 'request'))
        self.assertTrue(cs['resident_primers'][0]['environment_assessment']['swap_affected'])
        self.assertFalse(row['environment_assessment']['swap_affected'])
        self.assertEqual(row['primer_index'], 0)
        self.assertTrue(row['tokens_match_frozen_reference'])

    def test_measured_swap_stops_and_retains_both_rows(self):
        collect, count, cs, events, modes = self.harness(measured_swap=True)
        with self.assertRaises(EnvironmentFailure): collect(512, 'baseline', 'confirm-1', 1)
        self.assertEqual(count(), 2)
        self.assertEqual(len(cs['resident_primers']), 1)
        self.assertEqual(len(cs['requests']), 1)
        self.assertTrue(cs['requests'][0]['environment_assessment']['swap_affected'])

    def test_primer_hard_failure_stops_before_measured_request(self):
        collect, count, cs, events, modes = self.harness(primer_hard=True)
        with self.assertRaises(EnvironmentFailure): collect(512, 'baseline', 'confirm-1', 1)
        self.assertEqual(count(), 1)
        self.assertEqual(len(cs['resident_primers']), 1)
        self.assertFalse(cs['requests'])

    def test_primer_token_mismatch_stops_before_measurement(self):
        collect, count, cs, events, modes = self.harness(primer_bad_tokens=True)
        with self.assertRaises(NumericalFailure): collect(512, 'candidate', 'confirm-1', 1)
        self.assertEqual(count(), 1)
        self.assertFalse(cs['requests'])

    def test_bound_does_not_start_partial_pair(self):
        collect, count, cs, events, modes = self.harness(limit=1)
        with self.assertRaises(TimeoutError): collect(512, 'baseline', 'confirm-1', 1)
        self.assertEqual(count(), 0)
        self.assertFalse(cs['resident_primers'])


if __name__ == '__main__':
    unittest.main()
