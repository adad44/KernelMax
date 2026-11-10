"""Run the real orchestration with fake CPU-only model collectors."""
import json
from pathlib import Path
import tempfile
import unittest
from evidence import Controller,Store,GUARDS
from runtime_integration import RuntimeSession,UserPause
from verification import ContextVerifier
from test_verification import mock_sample

SPEC=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104/protocol.json')
GOOD={**{key:True for key in GUARDS},'protected_source_sha256':{},'numerical_source_sha256':{}}


class FakeNative:
    def __init__(self,root,*,failed_512=False,pause_primer=False,protocol=None):
        self.root=Path(root);self.protocol=protocol if protocol is not None else json.loads(SPEC.read_text())
        self.store=Store(self.root/'evidence',self.protocol)
        self.controller=Controller(self.store);self.controller.freeze({'synthetic_cpu_test':True})
        self.controller.open_epoch('one',identities={'synthetic_cpu_test':True})
        self.calls=0;self.failed_512=failed_512;self.pause_primer=pause_primer
        self.retained=[];self.attach('one')

    def attach(self,epoch):
        runtime=RuntimeSession.__new__(RuntimeSession)
        runtime.controller=self.controller;runtime.store=self.store;runtime.protocol=self.protocol;runtime.epoch=epoch
        runtime.verifier=ContextVerifier(self.store,sample_validator=mock_sample,
            numeric_validator=lambda proof:{context:[42]*128 for context in (512,2048,4096)})
        runtime._retain=lambda value:self.retained.append(value) if value['repeatable_gain_verified'] else None
        runtime.reason=None
        self.runtime=runtime
        self.cs={'resident_primers':[],'requests':[],'phase_memory_requests':[],
            'correctness':[{'length':context,'outputs':128} for context in (512,2048,4096)],
            'correctness_reuse':{},'calibrations':{},'summaries':{}}

    def correctness(self,*_):
        self.calls+=2
        path=self.root/f'{self.runtime.epoch}-synthetic-arrays-placeholder'
        path.write_bytes(b'CPU fixture only; not a numerical certificate')
        value={'length':512,'outputs':8,'arrays':str(path)}
        self.cs['correctness'].append(value)
        return value

    def collect(self,context,lane,phase,pair,instrument):
        writes=list(self.store.audit_writes)
        ratio=1.10 if lane=='candidate' else (0.99 if self.failed_512 and context==512 and lane=='B' else 1)
        raw={'phase':phase,'trial':pair,'valid':True,'tps':20*ratio,
            'instrumented':instrument,'phase_memory':{} if instrument else None}
        self.calls+=1
        self.cs['resident_primers'].append({**raw,'environment_assessment':{'continue_pass':True}})
        self.assert_no_write(writes)
        if self.pause_primer and phase=='confirm-1':
            self.pause_primer=False;self.controller.request_pause()
            raise UserPause('CPU simulated pause after primer')
        self.calls+=1
        target=self.cs['phase_memory_requests'] if instrument else self.cs['requests']
        target.append({**raw,'environment_assessment':{'confirmation_sample_eligible':True}})
        self.assert_no_write(writes)

    def assert_no_write(self,old):
        if old!=self.store.audit_writes: raise AssertionError('Evidence I/O interposed between fake native calls')

    def execute(self):
        self.runtime.execute(self.collect,self.correctness,self.cs,lambda:GOOD,lambda:self.calls)

    def resume(self):
        self.controller.close('user_pause',GOOD)
        self.controller=Controller(self.store)
        self.controller.open_epoch('two',identities={'synthetic_cpu_test':True},previous_process_verified_gone=True)
        self.attach('two')

    def cleanup(self):
        if self.controller.lease is not None: self.controller.lease.close();self.controller.lease=None


class RealScheduleTests(unittest.TestCase):
    def exercise(self,**kwargs):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        backend=FakeNative(tmp.name,**kwargs);self.addCleanup(backend.cleanup)
        return backend

    def test_exact_uninterrupted_plan_is706_calls_with350_primary_rows(self):
        backend=self.exercise();backend.execute()
        self.assertEqual(backend.calls,706)
        self.assertEqual(backend.controller.state()['starts_reserved'],706)
        self.assertEqual(len(backend.cs['requests']),350)
        self.assertEqual(len(backend.cs['resident_primers']),352)
        self.assertEqual(len(backend.cs['phase_memory_requests']),2)
        self.assertEqual([value['context'] for value in backend.retained],[512,2048,4096])
        self.assertEqual(backend.runtime.reason,'complete')

    def test_failed_control_keeps_all350_rows_and_supports_no_failed_scope(self):
        backend=self.exercise(failed_512=True);backend.execute()
        self.assertEqual(len(backend.cs['requests']),350)
        self.assertEqual(backend.calls,706)
        self.assertFalse(backend.cs['calibrations']['512']['passed'])
        self.assertFalse(backend.cs['summaries']['512-block-1']['five_percent_verified'])
        self.assertFalse(backend.cs['summaries']['512-block-2']['five_percent_verified'])
        self.assertEqual([value['context'] for value in backend.retained],[2048,4096])

    def test_pause_then_resume_preserves_abandoned_lane_and_charges_extra_controls(self):
        backend=self.exercise(pause_primer=True)
        with self.assertRaises(UserPause): backend.execute()
        old_reserved=backend.controller.state()['starts_reserved']
        old_executed=backend.calls
        backend.resume();backend.execute()
        self.assertEqual(old_reserved,128);self.assertEqual(old_executed,127)
        self.assertEqual(backend.controller.state()['starts_reserved'],830)
        self.assertEqual(backend.calls,829)
        self.assertEqual(len(backend.retained),3)
        seal=json.loads((backend.store.root/'contexts/512.json').read_text())
        self.assertEqual(len(seal['structurally_unmatched_collections']),1)
        self.assertEqual(seal['blocks']['1']['pairs'],20)


if __name__=='__main__': unittest.main()
