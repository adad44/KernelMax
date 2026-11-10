import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from evidence import Controller,Store,EvidenceError,BoundExceeded,GUARDS,pair_rows,publish

SPEC=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104/protocol.json')
GOOD={key:True for key in GUARDS}


class Clock:
    def __init__(self): self.elapsed=0
    def monotonic(self): return self.elapsed
    def utc(self): return 2000000000+self.elapsed


def row(value=1,eligible=True,executed=True):
    return {'executed':executed,'eligible':eligible,'raw':{'value':value}}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.protocol=json.loads(SPEC.read_text());self.clock=Clock()
        self.store=Store(self.temp.name,self.protocol)
        self.controller=Controller(self.store,monotonic=self.clock.monotonic,utc=self.clock.utc)
        self.addCleanup(self.release)
        self.controller.freeze({'original':'original','candidate':'candidate'})
        self.controller.open_epoch('one',identities={'original':'original','candidate':'candidate'})

    def release(self):
        if self.controller.lease is not None: self.controller.lease.close();self.controller.lease=None

    def coordinates(self,lane,pair=1):
        return {'phase':'comparison','context':512,'block':1,'pair':pair,'lane':lane}

    def collect(self,name,lane,pair=1):
        return self.controller.collect(name,self.coordinates(lane,pair),lambda:[row(),row()])

    def resume(self):
        self.controller=Controller(self.store,monotonic=self.clock.monotonic,utc=self.clock.utc)
        self.controller.open_epoch('two',identities={'original':'original','candidate':'candidate'},previous_process_verified_gone=True)

    def test_atomic_collection_reserves_both_slots_before_gpu_work(self):
        before=[]
        def backend():
            before.append(self.controller.state()['starts_reserved'])
            writes=list(self.store.audit_writes)
            primer=row(1)
            self.assertEqual(self.store.audit_writes,writes)
            measured=row(2)
            self.assertEqual(self.store.audit_writes,writes)
            return [primer,measured]
        self.controller.collect('sample',self.coordinates('baseline'),backend)
        self.assertEqual(before,[2])
        self.assertTrue(self.store.verify_artifacts())

    def test_second_live_controller_cannot_acquire_worker_lease(self):
        other=Controller(self.store)
        with self.assertRaises(EvidenceError):
            other.open_epoch('two',identities={'original':'original','candidate':'candidate'},previous_process_verified_gone=True)

    def test_os_worker_lease_rejects_a_competing_owner(self):
        self.controller.close('user_pause',GOOD)
        with (Path(self.temp.name)/'worker.lock').open('a+b') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.resume()

    def test_pause_after_primer_keeps_unused_measured_slot_charged(self):
        self.controller.collect('sample',self.coordinates('baseline'),lambda:[row(),row(executed=False,eligible=False)])
        self.controller.close('user_pause',GOOD);self.resume()
        self.assertEqual(self.controller.state()['starts_reserved'],2)
        self.assertFalse(self.controller.state()['collections']['sample']['artifacts'][1]['executed'])

    def test_unresolved_crash_orphans_are_terminal_even_if_caller_says_pause(self):
        self.controller.reserve('sample','collection',self.coordinates('baseline'),['primer','measured'])
        self.controller.close('user_pause',GOOD)
        self.assertTrue(self.controller.state()['terminal'])
        with self.assertRaises(EvidenceError): self.resume()

    def test_pairing_never_crosses_epochs_and_unmatched_rows_remain(self):
        self.collect('old','baseline',1)
        self.controller.close('user_pause',GOOD);self.resume()
        self.collect('new','candidate',2)
        pairs,unmatched=pair_rows(self.store,512,1)
        self.assertEqual(pairs,[]);self.assertEqual(len(unmatched),2)

    def test_abandoned_pair_cannot_be_reused_in_another_epoch(self):
        self.collect('old','baseline',1)
        self.controller.close('user_pause',GOOD);self.resume()
        self.collect('new','candidate',1)
        with self.assertRaises(EvidenceError): pair_rows(self.store,512,1)

    def test_all_complete_pairs_are_returned_including_ineligible_ones(self):
        self.collect('a','baseline')
        self.controller.collect('b',self.coordinates('candidate'),lambda:[row(),row(eligible=False)])
        pairs,unmatched=pair_rows(self.store,512,1)
        self.assertEqual(len(pairs),1);self.assertFalse(pairs[0]['lanes']['candidate']['row']['eligible'])
        self.assertEqual(unmatched,[])

    def test_nonfinite_failed_values_remain_explicit_in_raw_artifacts(self):
        self.controller.collect('bad',self.coordinates('baseline'),lambda:[row(),row(float('nan'),eligible=False)])
        raw=json.loads((Path(self.temp.name)/'rows/bad-1.json').read_text())
        self.assertEqual(raw['raw']['value'],{'__nonfinite_float__':'nan'})
        self.assertFalse(raw['eligible']);self.assertTrue(self.store.verify_artifacts())

    def test_failed_control_stays_failed_after_resume(self):
        receipt=publish(Path(self.temp.name)/'aa.json',{'passed':False})
        self.controller.append({'kind':'control','epoch':'one','context':512,'passed':False,
            'receipt':{'path':'aa.json','sha256':receipt['sha256']}})
        self.controller.close('user_pause',GOOD);self.resume()
        with self.assertRaises(EvidenceError):
            self.controller.append({'kind':'control','epoch':'two','context':512,'passed':True,
                'receipt':{'path':'aa.json','sha256':receipt['sha256']}})

    def test_hard_stop_cannot_reopen(self):
        self.controller.close('hard_stop',GOOD)
        with self.assertRaises(EvidenceError): self.resume()

    def test_bad_final_guard_cannot_resume(self):
        self.controller.close('user_pause',{**GOOD,'protected_unchanged':False})
        self.assertTrue(self.controller.state()['terminal'])
        with self.assertRaises(EvidenceError): self.resume()

    def test_global_active_wall_accumulates_and_pause_interval_is_excluded(self):
        self.clock.elapsed=36000;self.controller.close('user_pause',GOOD)
        self.clock.elapsed+=86400;self.resume()
        self.assertEqual(self.controller.check(),0)
        self.clock.elapsed+=36000
        with self.assertRaises(BoundExceeded): self.controller.check()

    def test_calendar_limit_includes_user_pause(self):
        self.controller.close('user_pause',GOOD)
        self.clock.elapsed=604801
        with self.assertRaises(BoundExceeded): self.resume()

    def test_post_request_bound_preserves_results_before_raising(self):
        def backend():
            self.clock.elapsed=72000
            return [row(),row()]
        with self.assertRaises(BoundExceeded):
            self.controller.collect('late',self.coordinates('baseline'),backend)
        self.assertTrue(self.controller.state()['collections']['late']['committed'])
        self.assertTrue(self.store.verify_artifacts())

    def test_conservative_global_start_cap_is_not_reset(self):
        self.controller.close('user_pause',GOOD)
        # A small synthetic protocol exercises the same generic reservation cap.
        other=copy.deepcopy(self.protocol);other['bounds']['max_starts_global']=2
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,other);controller=Controller(store)
            controller.freeze({});controller.open_epoch('one',identities={})
            controller.collect('first',{},lambda:[row(),row()]);controller.close('user_pause',GOOD)
            controller=Controller(store);controller.open_epoch('two',identities={},previous_process_verified_gone=True)
            with self.assertRaises(BoundExceeded): controller.collect('extra',{},lambda:[row(),row()])
            self.assertEqual(controller.state()['starts_reserved'],2)

    def test_artifact_cannot_be_overwritten_or_tampered(self):
        self.collect('a','baseline')
        path=Path(self.temp.name)/'rows/a-0.json';original=path.read_bytes()
        with self.assertRaises(FileExistsError): publish(path,{'fake':True})
        self.assertEqual(path.read_bytes(),original)
        path.write_text('{}')
        with self.assertRaises(EvidenceError): self.store.verify_artifacts()

    def test_torn_journal_is_not_silently_repaired(self):
        with self.store.path.open('ab') as stream: stream.write(b'{"torn":')
        original=self.store.path.read_bytes()
        with self.assertRaises(EvidenceError): self.store.read()
        self.assertEqual(self.store.path.read_bytes(),original)

    def test_stale_controller_append_is_rejected(self):
        with self.assertRaises(EvidenceError):
            self.store.append({'kind':'close_epoch','epoch':'one','reason':'user_pause'},0)

    def test_resume_requires_verified_old_process_exit_and_same_identities(self):
        self.controller.close('user_pause',GOOD)
        controller=Controller(self.store)
        with self.assertRaises(EvidenceError): controller.open_epoch('two',identities={'changed':True},previous_process_verified_gone=True)
        with self.assertRaises(EvidenceError): controller.open_epoch('two',identities={'original':'original','candidate':'candidate'})

    def test_native_process_exit_observation_charges_final_cleanup(self):
        self.clock.elapsed=10
        self.controller.close('user_pause',{**GOOD,'worker_process':{'pid':123,'birth':'synthetic'}})
        with self.assertRaises(EvidenceError): self.resume()
        self.clock.elapsed=13
        observer=Controller(self.store,monotonic=self.clock.monotonic,utc=self.clock.utc)
        observer.verify_exit('one',clock_domain=None,process_verified_gone=True,observation={'synthetic_cpu_test':True})
        self.assertEqual(observer.state()['active_seconds_closed'],13)
        self.clock.elapsed+=86400;self.resume()
        self.assertEqual(self.controller.state()['active_seconds_closed'],13)

    def test_changed_boot_identity_cannot_shrink_or_reset_active_wall(self):
        self.controller.close('user_pause',{**GOOD,'worker_process':{'pid':123,'birth':'synthetic'}})
        observer=Controller(self.store,monotonic=self.clock.monotonic,utc=self.clock.utc)
        with self.assertRaises(EvidenceError): observer.verify_exit('one',clock_domain='different-boot',
            process_verified_gone=True,observation={})

    def test_watchdog_sets_violation_without_serializing_evidence(self):
        class OneTick:
            def __init__(self): self.done=False
            def wait(self,_):
                if self.done: return True
                self.done=True;return False
            def set(self): self.done=True
        self.controller.stop_watchdog=OneTick()
        self.clock.elapsed=72000
        writes=list(self.store.audit_writes)
        self.controller.start_watchdog();self.controller.watchdog.join(timeout=2)
        self.assertIsNotNone(self.controller.violation)
        self.assertEqual(self.store.audit_writes,writes)


if __name__=='__main__': unittest.main()
