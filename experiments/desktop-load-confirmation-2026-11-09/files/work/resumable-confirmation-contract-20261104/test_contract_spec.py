import copy
import json
from pathlib import Path
import unittest
from contract_spec import validate


class ContractTests(unittest.TestCase):
    def setUp(self): self.protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_two_resume_budget_counts_all_additional_requests(self):
        result=validate(self.protocol)
        self.assertEqual(result['expected_starts_with_two_resumes'],958)
        self.assertEqual(result['start_margin'],42)
        self.assertFalse(result['timing_authorized'])

    def reject(self,path,value):
        data=copy.deepcopy(self.protocol)
        item=data
        for key in path[:-1]: item=item[key]
        item[path[-1]]=value
        with self.assertRaises(ValueError): validate(data)

    def test_old_800_cap_cannot_silently_be_reset_on_resume(self):
        self.reject(['bounds','max_starts_global'],800)

    def test_user_preparation_reply_cannot_become_timing_authorization(self):
        self.reject(['timing_authorized'],True)

    def test_all_original_contexts_and_independent_repeat_are_required(self):
        self.reject(['workload','contexts'],[512,2048])
        self.reject(['workload','comparison_blocks'],{'512':1,'2048':1,'4096':1})

    def test_old_reference_and_candidate_identities_are_bound(self):
        self.reject(['original_reference_sha256'],'0'*64)
        self.reject(['candidate_sha256'],'0'*64)
        self.reject(['new_candidate_allocation'],True)

    def test_confidence_and_goal_requirements_cannot_be_relaxed(self):
        self.reject(['statistics','aa_ci_must_include_one'],False)
        self.reject(['statistics','aa_max_ci_width'],0.05)
        self.reject(['statistics','minimum_goal_ratio'],1.04)
        self.reject(['statistics','bootstrap_resamples'],100)

    def test_failed_control_and_paging_rules_preserved(self):
        self.reject(['statistics','failed_control_may_be_retried_after_pause'],True)
        self.reject(['environment','paging_alone_stops_or_excludes'],True)

    def test_hard_stops_cannot_be_removed(self):
        self.reject(['environment','hard_stops'],['AC_loss'])

    def test_generation_and_drain_cannot_be_omitted(self):
        self.reject(['workload','outputs'],127)
        self.reject(['workload','candidate_async_submissions'],128)
        self.reject(['workload','first_token_and_final_gpu_drain_included'],False)

    def test_pairing_and_sample_selection_requirements_preserved(self):
        self.reject(['statistics','same_epoch_pairs_only'],False)
        self.reject(['statistics','pool_every_eligible_complete_pair'],False)
        self.reject(['statistics','old_samples_reclassified'],True)
        self.reject(['statistics','stop_early_when_gain_exceeds_target'],True)

    def test_global_wall_epoch_and_calendar_limits_are_finite(self):
        self.reject(['bounds','max_active_wall_seconds_global'],float('inf'))
        self.reject(['bounds','max_calendar_seconds'],0)
        self.reject(['bounds','max_epochs'],4)
        self.reject(['bounds','max_user_resumes'],True)


if __name__=='__main__': unittest.main()
