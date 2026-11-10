import copy
import json
from pathlib import Path
import tempfile
import unittest
from evidence import Controller,Store,EvidenceError,GUARDS
from verification import ContextVerifier,verify_sample,verify_numeric

SPEC=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104/protocol.json')
HISTORY=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-r2-resumed-20261103/command-001-confirm-cand_57c56c7943e0/state.json')
GOOD={key:True for key in GUARDS}


def mock_sample(raw,**kwargs):
    if raw.get('valid') is not True: raise EvidenceError('Synthetic invalid row')
    return raw['tps']


class SyntheticPass:
    """CPU-only backend; default native numerical validators are not bypassed in production."""
    def __init__(self,root):
        self.store=Store(root,json.loads(SPEC.read_text()));self.serial=0
        self.controller=Controller(self.store);self.controller.freeze({'synthetic_cpu_test':True})
        self.controller.open_epoch('one',identities={'synthetic_cpu_test':True})
        self.verifier=ContextVerifier(self.store,sample_validator=mock_sample,
            numeric_validator=lambda proof:{int(context):tokens for context,tokens in proof['tokens'].items()})
        self.qualify()
        for lane in ('baseline','candidate'): self.collect('diagnostic',512,lane,0)

    def qualify(self):
        name='qualification-'+self.controller.epoch
        self.controller.reserve(name,'qualification',{},['reference_trace','candidate_trace'])
        proof={'tokens':{context:[42]*128 for context in (512,2048,4096)}}
        rows=[{'executed':True,'eligible':True,'raw':proof} for _ in range(2)]
        self.controller.commit(name,rows)

    def collect(self,phase,context,lane,pair,block=None,tps=20,valid=True):
        self.serial+=1;name=f'collection-{self.serial}'
        coords={'phase':phase,'context':context,'lane':lane,'pair':pair,'block':block}
        raw_phase={'comparison':f'confirm-{block}','warmup':f'warmup-{block}',
            'aa':'aa','aa_warmup':'aa_warmup','diagnostic':'phase_memory'}[phase]
        raw={'tps':tps,'phase':raw_phase,'trial':pair,'valid':valid,
            'instrumented':phase=='diagnostic','phase_memory':{} if phase=='diagnostic' else None}
        self.controller.collect(name,coords,lambda:[{'executed':True,'eligible':True,'raw':raw} for _ in range(2)])

    def control(self,context,ratio=1):
        for pair in range(1,6):
            for lane in ('A','B'): self.collect('aa_warmup',context,lane,pair)
        for pair in range(1,21):
            for lane in ('A','B') if pair%2 else ('B','A'):
                self.collect('aa',context,lane,pair,tps=20*(ratio if lane=='B' else 1))
        return self.verifier.record_control(self.controller,context)

    def warm(self,context,block):
        for pair in range(1,6):
            for lane in ('baseline','candidate'): self.collect('warmup',context,lane,pair,block)

    def block(self,context,block,ratio=1.005,start=1,stop=20,warm=True):
        if warm: self.warm(context,block)
        for pair in range(start,stop+1):
            for lane in ('baseline','candidate') if pair%2 else ('candidate','baseline'):
                self.collect('comparison',context,lane,pair,block,tps=20*(ratio if lane=='candidate' else 1))

    def resume(self):
        self.controller.close('user_pause',GOOD)
        self.controller=Controller(self.store)
        self.controller.open_epoch('two',identities={'synthetic_cpu_test':True},previous_process_verified_gone=True)
        self.qualify()

    def cleanup(self):
        if self.controller.lease is not None: self.controller.lease.close();self.controller.lease=None


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=SyntheticPass(self.tmp.name);self.addCleanup(self.run.cleanup)

    def test_half_percent_context_can_seal_before_other_contexts(self):
        self.assertTrue(self.run.control(2048)['passed'])
        self.run.block(2048,1)
        value=self.run.verifier.seal(self.run.controller,2048,GOOD)
        self.assertTrue(value['repeatable_gain_verified'])
        self.assertFalse(value['five_percent_verified'])
        self.assertEqual(set(self.run.controller.state()['seals']),{'2048'})
        self.assertFalse(value['acceptance_enabled'])

    def test_failed_aa_leaves_positive_comparison_inconclusive(self):
        self.assertFalse(self.run.control(2048,ratio=0.99)['passed'])
        self.run.block(2048,1,ratio=1.15)
        value=self.run.verifier.seal(self.run.controller,2048,GOOD)
        self.assertFalse(value['repeatable_gain_verified']);self.assertFalse(value['five_percent_verified'])

    def test_missing_independent_512_repeat_cannot_certify(self):
        self.run.control(512);self.run.block(512,1,ratio=1.15)
        with self.assertRaises(EvidenceError): self.run.verifier.seal(self.run.controller,512,GOOD)

    def test_resumed_block_uses_all_complete_same_epoch_pairs(self):
        self.run.control(2048);self.run.block(2048,1,start=1,stop=10)
        self.run.resume();self.run.control(2048);self.run.block(2048,1,start=11,stop=20)
        value=self.run.verifier.seal(self.run.controller,2048,GOOD)
        self.assertEqual(value['contributing_epochs'],['one','two'])
        self.assertEqual(value['blocks']['1']['pairs'],20)

    def test_resume_without_new_control_cannot_certify(self):
        self.run.control(2048);self.run.block(2048,1,start=1,stop=10)
        self.run.resume();self.run.block(2048,1,start=11,stop=20)
        with self.assertRaises(EvidenceError): self.run.verifier.seal(self.run.controller,2048,GOOD)

    def test_no_performance_based_pair_subset(self):
        self.run.control(2048);self.run.block(2048,1,start=1,stop=21)
        with self.assertRaises(EvidenceError): self.run.verifier.seal(self.run.controller,2048,GOOD)

    def test_incomplete_pair_is_retained_and_never_matched_across_sessions(self):
        self.run.control(2048);self.run.warm(2048,1)
        self.run.collect('comparison',2048,'baseline',1,1)
        self.run.resume();self.run.control(2048);self.run.block(2048,1,start=2,stop=21)
        value=self.run.verifier.seal(self.run.controller,2048,GOOD)
        self.assertEqual(len(value['structurally_unmatched_collections']),1)

    def test_invalid_raw_row_cannot_be_filtered_out(self):
        self.run.control(2048);self.run.warm(2048,1)
        self.run.collect('comparison',2048,'baseline',1,1,valid=False)
        self.run.collect('comparison',2048,'candidate',1,1,tps=23)
        self.run.block(2048,1,start=2,stop=20,warm=False)
        with self.assertRaises(EvidenceError): self.run.verifier.seal(self.run.controller,2048,GOOD)

    def test_final_context_guard_is_mandatory(self):
        with self.assertRaises(EvidenceError):
            self.run.verifier.seal(self.run.controller,2048,{**GOOD,'checkpoint_and_snapshot_unchanged':False})


class NativeSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.history=json.loads(HISTORY.read_text())

    def test_real_saved_baseline_candidate_and_primer_rows_revalidate(self):
        for primer,key in ((False,'requests'),(True,'resident_primers')):
            for lane in ('baseline','candidate'):
                raw=next(row for row in self.history[key] if row['lane']==lane and row['prompt_tokens']==512)
                verify_sample(raw,context=512,lane=lane,reference_tokens=raw['output_token_ids'],primer=primer)

    def test_actual_hash_bound_full_numerical_receipts_and_handoff_revalidate(self):
        proof={'full':[row for row in self.history['correctness'] if row['outputs']==128],
            'handoff':next(row for row in self.history['correctness'] if row['outputs']==8),
            'reuse':self.history['correctness_reuse']}
        references=verify_numeric(proof)
        self.assertEqual(set(references),{512,2048,4096})
        self.assertTrue(all(len(tokens)==128 for tokens in references.values()))

    def test_counter_flags_cannot_be_erased_from_stored_assessment(self):
        raw=copy.deepcopy(next(row for row in self.history['requests'] if row['lane']=='candidate' and row['environment_assessment']['flags']))
        raw['environment_assessment']['flags']=[]
        with self.assertRaises(EvidenceError): verify_sample(raw,context=raw['prompt_tokens'],lane='candidate',reference_tokens=raw['output_token_ids'])

    def test_invented_tps_or_missing_gpu_drain_is_rejected(self):
        raw=copy.deepcopy(next(row for row in self.history['requests'] if row['lane']=='candidate'))
        raw['metrics']['decode_tokens_per_second']*=1.1
        with self.assertRaises(EvidenceError): verify_sample(raw,context=raw['prompt_tokens'],lane='candidate',reference_tokens=raw['output_token_ids'])
        raw=copy.deepcopy(next(row for row in self.history['requests'] if row['lane']=='candidate'))
        raw['timing_observations']['gpu_drained_at']-=0.01
        with self.assertRaises(EvidenceError): verify_sample(raw,context=raw['prompt_tokens'],lane='candidate',reference_tokens=raw['output_token_ids'])


if __name__=='__main__': unittest.main()
