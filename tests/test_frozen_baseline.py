"""Portable publication audit: immutable data and actual arithmetic, no model load."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO/'reports/baselines/gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((REPORT/'raw.json').read_text())
        cls.report = json.loads((REPORT/'baseline.json').read_text())
        cls.publication = json.loads((REPORT/'publication.json').read_text())

    def test_frozen_data_and_independent_arithmetic(self):
        self.assertEqual(digest(REPORT/'raw.json'), self.publication['raw_sha256'])
        self.assertEqual(digest(REPORT/'baseline.json'), self.publication['baseline_sha256'])
        result = subprocess.run([sys.executable, '-B', str(REPO/'scripts/audit_baseline.py'),
            '--attempt', str(REPORT/'raw.json'), '--report', str(REPORT/'baseline.json')],
            capture_output=True, text=True, timeout=30, check=True)
        audit = json.loads(result.stdout)
        self.assertEqual(audit['independent_audit'], 'passed')
        self.assertEqual(audit['checked_requests'], 147)
        self.assertEqual(audit['official_samples'], 120)
        self.assertEqual(audit['excluded_diagnostic_requests'], 12)
        self.assertFalse(self.report['candidate_acceptance_enabled'])

    def test_measured_sources_and_snapshot_are_unchanged(self):
        for path, expected in self.raw['evaluator_source_sha256'].items():
            name = Path(path).name
            self.assertEqual(digest(REPORT/'protected_sources/evaluator'/name), expected, name)
            self.assertEqual(digest(REPO/'evaluator'/name), expected, name)
        for path, expected in self.raw['runtime_source_sha256'].items():
            relative = path.partition('/mlx_lm/')[2]
            self.assertTrue(relative, path)
            self.assertEqual(digest(REPORT/'protected_sources/mlx_lm'/relative), expected, relative)
        self.assertEqual(digest(REPO/'scripts/audit_baseline.py'),
                         digest(REPORT/'execution_sources/verify-baseline-evidence.py'))
        license_info = self.publication['mlx_lm_license']
        self.assertEqual(digest(REPORT/license_info['path']), license_info['sha256'])

    def test_all_historical_attempts_remain_excluded_and_hash_bound(self):
        entries = self.publication['historical_attempts']
        self.assertEqual(len(entries), 16)
        self.assertEqual(len({entry['run_id'] for entry in entries}), 16)
        self.assertEqual(len(list((REPORT/'historical_attempts').glob('*/attempt.json'))), 16)
        for entry in entries:
            path = REPORT/entry['path']
            self.assertEqual(digest(path), entry['sha256'])
            state = json.loads(path.read_text())
            self.assertEqual(state['run_id'], entry['run_id'])
            self.assertNotEqual(state['run_id'], self.raw['run_id'])
            self.assertEqual(state['status'], entry['saved_status'])
            self.assertTrue(entry['excluded_from_official'])
            if entry['classification'] == 'externally_interrupted_setup':
                interruption = json.loads((path.parent/'external-interruption.json').read_text())
                self.assertEqual(interruption['retained_raw_sha256'], entry['sha256'])
            else:
                self.assertIn(state['status'], ('failed', 'interrupted'))

    def test_execution_sources_and_service_freeze_restoration_order(self):
        pipeline = json.loads((REPORT/'durable-pipeline.json').read_text())
        by_name = {Path(path).name: sha for path, sha in pipeline['sources'].items()}
        for entry in self.publication['execution_sources']:
            path = REPORT/entry['path']
            self.assertEqual(digest(path), entry['sha256'])
            if entry['hash_recorded_in_original_pipeline']:
                self.assertEqual(entry['sha256'], by_name[path.name])
        frozen = REPORT/'background-service-lease-at-freeze.json'
        lease = json.loads(frozen.read_text())
        self.assertEqual(lease['status'], 'paused_until_official_baseline')
        self.assertEqual(digest(frozen), self.report['environment_session']['external_service_lease']['sha256'])
        self.assertEqual(pipeline['status'], 'complete')
        self.assertTrue(pipeline['background_services_restored'])
        after = json.loads((REPORT/'background-services-after.json').read_text())
        self.assertEqual(after['status'], 'restored')
        self.assertTrue(all(after['loaded'].values()))
        restoration_info = self.publication['restored_service_lease']
        restored_path = REPORT/restoration_info['path']
        self.assertEqual(digest(restored_path), restoration_info['sha256'])
        restored = json.loads(restored_path.read_text())
        self.assertEqual(restored['status'], 'restored')
        self.assertEqual(set(restored['restored']), set(restored['paused']))
        self.assertLess(pipeline['official_verified_at'], restored['restored_at'])
        self.assertLessEqual(restored['restored_at'], pipeline['finished_at'])

    def test_failed_dense_diagnostic_is_not_promoted(self):
        diagnostic = self.report['dense_reference_diagnostic']
        self.assertEqual(digest(REPORT/'dense_reference_diagnostic.json'), diagnostic['sha256'])
        self.assertTrue(diagnostic['failed_cases'])
        self.assertTrue(all(not row['passed'] for row in diagnostic['failed_cases']))
        self.assertEqual(diagnostic['failed_cases'][0]['layer'], 12)
        self.assertFalse(self.raw['contract']['acceptance']['enabled'])


if __name__ == '__main__':
    unittest.main()
