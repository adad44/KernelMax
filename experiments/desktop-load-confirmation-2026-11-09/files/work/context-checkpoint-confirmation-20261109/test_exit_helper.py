import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evidence import Controller,Store,EvidenceError,GUARDS
from verify_exit import finalize_exit

SPEC=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104/protocol.json')


class ExitHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name);self.protocol=json.loads(SPEC.read_text())
        self.store=Store(self.work/'evidence',self.protocol);self.controller=Controller(self.store)
        self.controller.freeze({'synthetic_cpu_test':True})
        self.controller.open_epoch('one',identities={'synthetic_cpu_test':True},clock_domain='synthetic-boot')
        self.process={'pid':123,'birth':'synthetic CPU fixture'}
        self.controller.close('hard_stop',{**{key:True for key in GUARDS},'worker_process':self.process})

    def test_still_live_same_birth_cannot_exclude_cleanup(self):
        with patch('verify_exit.process_identity',return_value=self.process):
            with self.assertRaises(EvidenceError): finalize_exit(self.work,'one')
        self.assertFalse(self.controller.state()['epochs']['one']['exit_verified'])

    def test_verified_exit_of_failed_pass_is_reported_without_goal_claim(self):
        with patch('verify_exit.process_identity',return_value=None),patch('verify_exit.boot_identity',return_value='synthetic-boot'):
            report=finalize_exit(self.work,'one')
        self.assertTrue(report['worker_process_verified_gone'])
        self.assertFalse(report['five_percent_improvement_verified'])
        self.assertFalse(report['complete_locked_workload'])
        self.assertTrue(self.controller.state()['epochs']['one']['exit_verified'])


if __name__=='__main__': unittest.main()
