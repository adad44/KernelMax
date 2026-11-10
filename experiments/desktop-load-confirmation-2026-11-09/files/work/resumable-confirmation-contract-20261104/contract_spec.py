"""Review checks for a prepared contract; does not authorize GPU execution."""
import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(protocol):
    p=copy.deepcopy(protocol)
    def require(condition,message):
        if not condition: raise ValueError(message)
    require(p['schema_version']==1 and p['protocol']=='resumable_desktop_confirmation_v1','Unknown protocol')
    require(p['status']=='prepared_inactive' and p['authorization_scope']=='prepare_and_review_contract_only'
        and p['timing_authorized'] is False,'Preparation cannot authorize timing')
    require(p['new_candidate_allocation'] is False and p['historical_timing_imported'] is False
        and p['original_catalog_modified'] is False,'Candidate/history scope changed')
    require(p['candidate_id']=='cand_57c56c7943e0'
        and p['candidate_sha256']=='709f9a33eb632748e472f213157405dca706f42fd547d87494f950919065ef18',
        'Frozen candidate identity changed')
    require(p['original_reference_sha256']=='87230ee192aea2c097a063a7f062f7ad4b7fe1f98182ef118b80f0f04ccbb1a2','Original reference changed')
    w=p['workload'];b=p['bounds'];s=p['statistics'];e=p['environment']
    require(w['contexts']==[512,2048,4096] and w['full_numerical_contexts']==w['contexts']
        and w['outputs']==128 and w['warmups_per_lane']==5 and w['aa_pairs']==20
        and w['comparison_pairs_per_block']==20 and w['comparison_blocks']=={'512':2,'2048':1,'4096':1}
        and w['order']=='alternating_ab_ba' and w['seed']==42 and w['sampler']=='greedy','Locked workload changed')
    require(w['timed_forward_calls']==129 and w['candidate_async_submissions']==127
        and w['fresh_handoff_outputs_per_epoch']==8 and w['first_token_and_final_gpu_drain_included'] is True
        and w['fresh_cache_per_request'] is True and w['unconditional_same_lane_128_output_primer'] is True,
        'Generation/timing boundary changed')
    for key in ('max_starts_global','max_epochs','max_user_resumes','max_active_wall_seconds_global',
            'max_wall_seconds_per_worker','max_calendar_seconds','base_expected_starts'):
        require(type(b[key]) is int and b[key]>0,'Invalid bound: '+key)
    require(b['max_epochs']==3 and b['max_user_resumes']==2 and b['max_epochs']==b['max_user_resumes']+1,
        'Epoch bound changed')
    require(b['max_starts_global']==1000 and b['base_expected_starts']==706
        and b['max_active_wall_seconds_global']==72000 and b['max_wall_seconds_per_worker']==72000
        and b['max_calendar_seconds']==604800,'Declared global bounds changed')
    require(b['reserve_per_user_resume']=={'aa_and_warmups_with_primers':100,
        'candidate_block_warmups_with_primers':20,'fresh_handoff_traces':2,'abandoned_pair_starts_upper_bound':4},
        'Resume request accounting changed')
    expected=b['base_expected_starts']+b['max_user_resumes']*sum(b['reserve_per_user_resume'].values())
    require(expected<=b['max_starts_global'],'Resume allowance exceeds global start cap')
    require((b['fixed_idle_seconds'],b['continuous_nominal_seconds'],b['max_quiet_wait_seconds'])==(60,60,180)
        and (b['qualification_stable_seconds'],b['max_qualification_wait_seconds'])==(15,1200)
        and b['watchdog_seconds']==1,'Thermal/bound preparation changed')
    require((s['bootstrap_resamples'],s['bootstrap_lower_index'],s['bootstrap_upper_index'])==(10000,249,9749)
        and s['comparison_bootstrap_seed']=='prompt_length' and s['aa_bootstrap_seed']=='42+prompt_length'
        and s['aa_ci_must_include_one'] is True and s['aa_max_ci_width']==0.04
        and s['minimum_goal_ratio']==1.05 and s['eligible_pair_target_per_block']==20,'Statistical requirements changed')
    for key in ('same_epoch_pairs_only','pool_every_eligible_complete_pair','pair_order_index_is_global_within_block',
            'failed_control_is_sticky_for_context','finish_other_predeclared_blocks_after_statistical_failure'):
        require(s[key] is True,'Evidence preservation requirement changed: '+key)
    require(s['failed_control_may_be_retried_after_pause'] is False and s['old_samples_reclassified'] is False
        and s['stop_early_when_gain_exceeds_target'] is False,'Filtering/reroll rule changed')
    require(e['policy']=='retain_and_flag_desktop_load' and e['ordinary_apps_and_safari_allowed'] is True
        and e['paging_alone_stops_or_excludes'] is False,'Desktop policy changed')
    require(set(e['hard_stops'])=={'AC_loss','nonnominal_request_thermals','numerical_or_generation_failure',
        'GPU_or_runtime_failure','source_or_checkpoint_integrity_failure','declared_bounds'},'Hard stops changed')
    require(p['retention']['acceptance_enabled'] is False and len(p['activation_requirements'])>=5,'Activation boundary missing')
    return {'protocol_sha256':digest(p),'expected_starts_with_two_resumes':expected,
        'start_margin':b['max_starts_global']-expected,'timing_authorized':False}
