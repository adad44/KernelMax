"""Exercise a durable pause and resume with the real boot identity reader."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence import Controller, EvidenceError, GUARDS, Store
from runtime_integration import PREPARED, boot_identity
from verify_exit import finalize_exit


class BootSessionResumeTests(unittest.TestCase):
    def test_exit_verifier_and_resume_use_the_same_boot_session(self):
        protocol = json.loads((PREPARED / 'protocol.json').read_text())
        identities = {'synthetic_cpu_test': True}
        guards = dict.fromkeys(GUARDS, True)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            controller = Controller(Store(work / 'evidence', protocol))
            controller.freeze(identities)
            controller.open_epoch('e001', identities=identities, clock_domain=boot_identity())
            controller.close('user_pause', {**guards, 'worker_process': {'pid': 123, 'birth': 'synthetic CPU fixture'}})
            with patch('verify_exit.process_identity', return_value=None):
                report = finalize_exit(work, 'e001')
            self.assertTrue(report['worker_process_verified_gone'])
            self.assertFalse(report['complete_locked_workload'])
            self.assertTrue(controller.state()['epochs']['e001']['exit_verified'])
            resume = {'identities': identities, 'previous_process_verified_gone': True}
            with self.assertRaisesRegex(EvidenceError, 'clock domain changed'):
                controller.open_epoch('e002', clock_domain='boot-session:synthetic-other-boot', **resume)
            controller.open_epoch('e002', clock_domain=boot_identity(), **resume)
            self.assertEqual(controller.state()['current_epoch'], 'e002')
            controller.close('user_pause', guards)


if __name__ == '__main__':
    unittest.main()
