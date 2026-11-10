"""Offline coverage for bounded policy loops and paired policy comparisons."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from evaluator.schemas import EvaluationReport, ExperimentContract, RunStatus, load_contract
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.search import (
    CallbackTimeout,
    SearchPolicy,
    _invoke_with_timeout,
    run_search_loop,
    schedule_candidates,
    scheduling_controls,
)
from orchestrator.search_comparison import (
    ComparisonBudget,
    ComparisonRules,
    SearchEvidence,
    SearchGenerationQuota,
    SearchLimits,
    SearchPolicyProposal,
    SearchTask,
    TaskPartition,
    run_policy_comparison,
)
from orchestrator.candidate_manager import create_candidate_id
from orchestrator.state_machine import RunIdentity

ROOT = Path(__file__).resolve().parents[1]


def timeout_callback():
    time.sleep(5)


def generate_paths(policy, history):
    return [policy.controls["candidate_path"]]


def generate_paths_with_quota(policy, history, quota):
    quota.consume(1)
    return [policy.controls["candidate_path"]]


def slow_generate_paths(policy, history):
    time.sleep(5)
    return []


def stop_proposing(policy, history):
    return None


def generate_policy_paths(policy, history):
    return list(policy.controls["candidate_paths"])


def bump_policy(policy, history):
    return SearchPolicy.create(policy.policy_id, policy.revision + 1, policy.controls)


def propose_unseen_first(policy, history):
    controls = dict(policy.controls)
    controls["ranking"] = "unseen_first"
    controls["candidate_paths"] = list(controls["next_paths"])
    return SearchPolicy.create(policy.policy_id, policy.revision + 1, controls)


def propose_bad_ranking(policy, history):
    return SearchPolicy.create(
        policy.policy_id, policy.revision + 1, {"ranking": "agent_code"}
    )


def write_candidate(root: Path) -> Path:
    candidate_id = create_candidate_id()
    candidate = root / candidate_id
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text(json.dumps({
        "candidate_id": candidate_id,
        "name": "search test",
        "implementation_type": "mlx",
        "description": "test fixture",
    }), encoding="utf-8")
    for filename in ("wrapper.py", "implementation.py"):
        (candidate / filename).write_text("# inert test fixture\n", encoding="utf-8")
    return candidate


def fake_eval(run, frozen, contract, *, on_diagnostic_passed=None):
    return EvaluationReport(
        run_id=run.run_id,
        candidate=run.candidate,
        contract_sha256=run.contract_sha256,
        environment=None,
        correctness=(),
        performance=(),
        status=RunStatus.COMPLETED,
        errors=(),
        verdict=None,
    )


def comparison_proposer(incumbent, training):
    return SearchPolicyProposal(
        SearchPolicy.create(
            "challenger",
            incumbent.revision + 1,
            {"ranking": "new", "training_task_ids": [item.task.task_id for item in training]},
        ),
        "training outcomes indicate the old ordering wastes attempts",
        "improve first-verified-result rate without relaxing correctness gates",
    )


def hung_proposer(incumbent, training):
    time.sleep(5)
    return SearchPolicyProposal(
        SearchPolicy.create("late", incumbent.revision + 1, {}),
        "late proposal",
        "late expected benefit",
    )


def comparison_executor(policy, task, seed, limits, generation_quota):
    generation_quota.consume_attempt()
    generation_quota.consume(2)
    speed = 1.1 if policy.policy_id == "challenger" else 1.0
    return SearchEvidence(
        completed=True,
        environment_valid=True,
        eligible=True,
        speedup=speed,
        attempts_used=1,
        generation_units_used=2,
        best_run_id=f"{task.task_id}-{seed}-{policy.policy_id}",
        best_candidate_id=f"candidate-{policy.policy_id}",
        candidate_ids=(f"candidate-{policy.policy_id}",),
        run_ids=(f"{task.task_id}-{seed}-{policy.policy_id}",),
        report_ids=(f"report-{task.task_id}-{seed}-{policy.policy_id}",),
    )


def failed_gain_executor(policy, task, seed, limits, generation_quota):
    generation_quota.consume_attempt()
    generation_quota.consume(1)
    speed = 1.0 if policy.policy_id == "challenger" else 1.1
    return SearchEvidence(
        True, True, True, speed, 1, 1,
        best_run_id=f"{task.task_id}-{seed}-{policy.policy_id}",
        best_candidate_id=f"candidate-{policy.policy_id}",
        candidate_ids=(f"candidate-{policy.policy_id}",),
        run_ids=(f"{task.task_id}-{seed}-{policy.policy_id}",),
        report_ids=(f"report-{task.task_id}-{seed}-{policy.policy_id}",),
    )


def hung_executor(policy, task, seed, limits, generation_quota):
    time.sleep(5)
    return SearchEvidence(False, True, False, None, 0, 0)


def environment_failure_executor(policy, task, seed, limits, generation_quota):
    generation_quota.consume_attempt()
    return SearchEvidence(
        completed=False,
        environment_valid=False,
        eligible=False,
        speedup=None,
        attempts_used=1,
        generation_units_used=0,
        stop_reason="thermal interruption",
        failure_category="environment_thermal",
    )


def failing_executor(policy, task, seed, limits, generation_quota):
    generation_quota.consume_attempt()
    generation_quota.consume(1)
    raise RuntimeError("synthetic search executor failure")


def make_task(task_id, partition, index):
    digest = f"{index:064x}"
    return SearchTask(
        task_id, partition, f"{index + 500:064x}", digest,
        f"{index + 100:064x}", f"{200:064x}",
        f"{300:064x}", "generator-v1", f"{400:064x}",
    )


class OrchestratorSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract: ExperimentContract = load_contract(ROOT / "kernelmaxxing.yaml")
        self.journal = ExperimentJournal(self.root / "journal")

    def test_callback_timeout_terminates_worker(self):
        started = time.monotonic()
        with self.assertRaises(CallbackTimeout):
            _invoke_with_timeout(timeout_callback, (), 0.05)
        self.assertLess(time.monotonic() - started, 3)

    def test_run_search_loop_uses_deadline_safe_callbacks(self):
        candidates = self.root / "candidates"
        candidates.mkdir()
        candidate_id = create_candidate_id()
        candidate = candidates / candidate_id
        candidate.mkdir()
        (candidate / "manifest.json").write_text(json.dumps({
            "candidate_id": candidate_id,
            "name": "search test",
            "implementation_type": "mlx",
            "description": "test fixture",
        }), encoding="utf-8")
        for filename in ("wrapper.py", "implementation.py"):
            (candidate / filename).write_text("# inert test fixture\n", encoding="utf-8")
        policy = SearchPolicy.create("baseline", 1, {"candidate_path": str(candidate)})
        quota = SearchGenerationQuota(1, 1)
        result = run_search_loop(
            initial_policy=policy,
            candidate_factory=generate_paths_with_quota,
            policy_proposer=stop_proposing,
            max_rounds=1,
            evaluator=fake_eval,
            contract=self.contract,
            budget=ExperimentBudget(max_attempts=1),
            journal=self.journal,
            frozen_root=self.root / "frozen",
            candidates_root=candidates,
            callback_timeout_seconds=10,
            generation_quota=quota,
        )
        self.assertEqual(result.attempts_started, 1, result.summary())
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual((quota.used, quota.attempts_used), (1, 1))

    def test_run_search_loop_stops_on_hung_candidate_factory(self):
        result = run_search_loop(
            initial_policy=SearchPolicy.create("baseline", 1, {}),
            candidate_factory=slow_generate_paths,
            policy_proposer=stop_proposing,
            max_rounds=1,
            evaluator=fake_eval,
            contract=self.contract,
            budget=ExperimentBudget(max_attempts=1),
            journal=self.journal,
            frozen_root=self.root / "frozen-timeout",
            candidates_root=self.root / "empty-candidates",
            callback_timeout_seconds=0.05,
        )
        self.assertIn("CallbackTimeout", result.stop_reason)
        self.assertEqual(result.attempts_started, 0)

    def test_paired_comparison_alternates_and_confirms_before_promotion(self):
        incumbent = SearchPolicy.create("incumbent", 1, {"ranking": "old"})
        rules = ComparisonRules(
            minimum_paired_seeds=2,
            minimum_gain_fraction=0.01,
            maximum_task_regression_fraction=0.02,
            task_weights=(("val-a", 1.0), ("test-a", 1.0)),
        )
        budget = ComparisonBudget(
            max_elapsed_seconds=200,
            max_attempts=30,
            max_generation_units=60,
            per_search=SearchLimits(5, 1, 2),
        )
        result = run_policy_comparison(
            incumbent=incumbent,
            training_tasks=(make_task("train-a", TaskPartition.TRAINING, 1),),
            validation_tasks=(make_task("val-a", TaskPartition.VALIDATION, 2),),
            confirmation_tasks=(make_task("test-a", TaskPartition.CONFIRMATION, 3),),
            training_seeds=(11,),
            validation_seeds=(21, 22),
            confirmation_seeds=(31, 32),
            proposer=comparison_proposer,
            executor=comparison_executor,
            budget=budget,
            rules=rules,
            journal=self.journal,
        )
        self.assertEqual(result.validation_decision, "pass")
        self.assertEqual(result.confirmation_decision, "pass")
        self.assertEqual(result.final_decision, "promote")
        self.assertEqual(result.promoted_policy.policy_id, "challenger")
        self.assertEqual(result.challenger.controls["training_task_ids"], ["train-a"])
        self.assertTrue(all(item.task.partition is TaskPartition.TRAINING for item in result.training))
        policies = [item.policy_id for item in result.validation]
        self.assertEqual(policies, ["incumbent", "challenger", "challenger", "incumbent"])
        history = self.journal.read_history()
        self.assertTrue(any(item.get("session_event") == "confirmation_decision" for item in history))
        self.assertTrue(any(item.get("session_event") == "policy_promoted" for item in history))
        proposal_record = next(
            item for item in history if item.get("session_event") == "policy_proposed"
        )
        self.assertIn("rationale", proposal_record["payload"])
        self.assertIn("expected_benefit", proposal_record["payload"])

    def test_validation_failure_retains_incumbent_and_never_runs_confirmation(self):
        incumbent = SearchPolicy.create("incumbent", 1, {"ranking": "old"})
        result = run_policy_comparison(
            incumbent=incumbent,
            training_tasks=(make_task("train-b", TaskPartition.TRAINING, 4),),
            validation_tasks=(make_task("val-b", TaskPartition.VALIDATION, 5),),
            confirmation_tasks=(make_task("test-b", TaskPartition.CONFIRMATION, 6),),
            training_seeds=(31,),
            validation_seeds=(41, 42), confirmation_seeds=(43, 44),
            proposer=comparison_proposer,
            executor=failed_gain_executor,
            budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
            rules=ComparisonRules(task_weights=(("val-b", 1.0), ("test-b", 1.0))),
            journal=self.journal,
        )
        self.assertEqual(result.final_decision, "retain_incumbent")
        self.assertEqual(result.confirmation, ())
        self.assertIsNone(result.promoted_policy)

    def test_partition_leakage_and_insufficient_confirmation_budget_are_rejected(self):
        incumbent = SearchPolicy.create("incumbent", 1, {"ranking": "old"})
        common = make_task("same-task", TaskPartition.VALIDATION, 7)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_policy_comparison(
                incumbent=incumbent,
                training_tasks=(make_task("train-c", TaskPartition.TRAINING, 8),),
                validation_tasks=(common,),
                confirmation_tasks=(SearchTask("same-task", TaskPartition.CONFIRMATION,
                                               common.task_sha256, common.workload_sha256,
                                               common.starting_state_sha256,
                                               common.environment_sha256, common.contract_sha256,
                                               common.generator_version, common.model_configuration_sha256),),
                training_seeds=(51,), validation_seeds=(61, 62), confirmation_seeds=(63, 64),
                proposer=comparison_proposer, executor=comparison_executor,
                budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
                rules=ComparisonRules(), journal=self.journal,
            )
        shared_task = make_task("val-shared", TaskPartition.VALIDATION, 18)
        duplicated_content = SearchTask(
            "test-copy", TaskPartition.CONFIRMATION, shared_task.task_sha256,
            shared_task.workload_sha256, shared_task.starting_state_sha256,
            shared_task.environment_sha256, shared_task.contract_sha256,
            shared_task.generator_version, shared_task.model_configuration_sha256,
        )
        with self.assertRaisesRegex(ValueError, "contents must be disjoint"):
            run_policy_comparison(
                incumbent=incumbent,
                training_tasks=(make_task("train-copy", TaskPartition.TRAINING, 19),),
                validation_tasks=(shared_task,), confirmation_tasks=(duplicated_content,),
                training_seeds=(52,), validation_seeds=(62, 63), confirmation_seeds=(64, 65),
                proposer=comparison_proposer, executor=comparison_executor,
                budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
                rules=ComparisonRules(), journal=self.journal,
            )
        with self.assertRaisesRegex(ValueError, "reserved confirmation"):
            run_policy_comparison(
                incumbent=incumbent,
                training_tasks=(make_task("train-d", TaskPartition.TRAINING, 9),),
                validation_tasks=(make_task("val-d", TaskPartition.VALIDATION, 10),),
                confirmation_tasks=(make_task("test-d", TaskPartition.CONFIRMATION, 11),),
                training_seeds=(71,), validation_seeds=(81, 82), confirmation_seeds=(83, 84),
                proposer=comparison_proposer, executor=comparison_executor,
                budget=ComparisonBudget(200, 8, 60, SearchLimits(5, 1, 2)),
                rules=ComparisonRules(), journal=self.journal,
            )

    def test_generation_quota_stops_overrun_and_diagnostic_scores_baseline(self):
        quota = SearchGenerationQuota(1)
        quota.consume()
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            quota.consume()
        evidence = SearchEvidence(True, True, False, None, 1, 0, report_ids=("diagnostic-report",))
        self.assertEqual(evidence.score, 1.0)

    def test_comparison_executor_timeout_is_inconclusive_and_does_not_promote(self):
        result = run_policy_comparison(
            incumbent=SearchPolicy.create("incumbent", 1, {}),
            training_tasks=(make_task("train-timeout", TaskPartition.TRAINING, 12),),
            validation_tasks=(make_task("val-timeout", TaskPartition.VALIDATION, 13),),
            confirmation_tasks=(make_task("test-timeout", TaskPartition.CONFIRMATION, 14),),
            training_seeds=(101,), validation_seeds=(201, 202), confirmation_seeds=(203, 204),
            proposer=comparison_proposer, executor=hung_executor,
            budget=ComparisonBudget(10, 9, 0, SearchLimits(0.05, 1, 0), 0.1),
            rules=ComparisonRules(task_weights=(("val-timeout", 1.0), ("test-timeout", 1.0))),
            journal=self.journal,
        )
        self.assertEqual(result.final_decision, "inconclusive")
        self.assertIsNone(result.promoted_policy)
        self.assertIn("CallbackTimeout", result.stop_reason)
        self.assertEqual((result.attempts_used, result.generation_units_used), (1, 0))
        self.assertTrue(any(
            item.get("session_event") == "search_failure"
            for item in self.journal.read_history()
        ))

    def test_comparison_proposer_timeout_stops_before_validation(self):
        result = run_policy_comparison(
            incumbent=SearchPolicy.create("incumbent", 1, {}),
            training_tasks=(make_task("train-proposer-timeout", TaskPartition.TRAINING, 20),),
            validation_tasks=(make_task("val-proposer-timeout", TaskPartition.VALIDATION, 21),),
            confirmation_tasks=(make_task("test-proposer-timeout", TaskPartition.CONFIRMATION, 22),),
            training_seeds=(111,), validation_seeds=(211, 212), confirmation_seeds=(311, 312),
            proposer=hung_proposer, executor=comparison_executor,
            budget=ComparisonBudget(200, 9, 18, SearchLimits(15, 1, 2), 0.05),
            rules=ComparisonRules(task_weights=(("val-proposer-timeout", 1.0),
                                                 ("test-proposer-timeout", 1.0))),
            journal=self.journal,
        )
        self.assertEqual(result.final_decision, "inconclusive")
        self.assertEqual(result.validation, ())
        self.assertIn("CallbackTimeout", result.stop_reason)
        self.assertTrue(any(
            item.get("session_event") == "policy_proposal_failed"
            for item in self.journal.read_history()
        ), (result.stop_reason, self.journal.read_history()))

    def test_environment_failure_is_retained_and_blocks_confirmation(self):
        result = run_policy_comparison(
            incumbent=SearchPolicy.create("incumbent", 1, {}),
            training_tasks=(make_task("train-env", TaskPartition.TRAINING, 15),),
            validation_tasks=(make_task("val-env", TaskPartition.VALIDATION, 16),),
            confirmation_tasks=(make_task("test-env", TaskPartition.CONFIRMATION, 17),),
            training_seeds=(111,), validation_seeds=(211, 212), confirmation_seeds=(311, 312),
            proposer=comparison_proposer, executor=environment_failure_executor,
            budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
            rules=ComparisonRules(task_weights=(("val-env", 1.0), ("test-env", 1.0))),
            journal=self.journal,
        )
        self.assertEqual(result.final_decision, "inconclusive")
        self.assertEqual(result.validation_decision, "inconclusive")
        self.assertEqual(result.confirmation, ())
        self.assertTrue(all(
            item.evidence.failure_category == "environment_thermal"
            for item in result.validation
        ))

    def test_executor_failure_retains_charged_cost_and_failure_provenance(self):
        result = run_policy_comparison(
            incumbent=SearchPolicy.create("incumbent", 1, {}),
            training_tasks=(make_task("train-failure", TaskPartition.TRAINING, 23),),
            validation_tasks=(make_task("val-failure", TaskPartition.VALIDATION, 24),),
            confirmation_tasks=(make_task("test-failure", TaskPartition.CONFIRMATION, 25),),
            training_seeds=(121,), validation_seeds=(221, 222), confirmation_seeds=(321, 322),
            proposer=comparison_proposer, executor=failing_executor,
            budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
            rules=ComparisonRules(task_weights=(("val-failure", 1.0), ("test-failure", 1.0))),
            journal=self.journal,
        )
        self.assertEqual(result.final_decision, "inconclusive")
        self.assertEqual((result.attempts_used, result.generation_units_used), (1, 1))
        failure = next(
            item for item in self.journal.read_history()
            if item.get("session_event") == "search_failure"
        )
        self.assertEqual(failure["payload"]["failure_category"], "RuntimeError")
        self.assertEqual(failure["payload"]["task_id"], "train-failure")

    def test_schedule_prefers_unseen_and_caps_new_hypotheses(self):
        controls = scheduling_controls(SearchPolicy.create("baseline", 1, {
            "ranking": "unseen_first",
            "avoid_duplicates": True,
            "new_hypothesis_fraction": 0.5,
        }))
        decision = schedule_candidates(
            ["seen/a", "new/b", "new/c", "new/d"],
            seen_ids={"a"},
            attempts_remaining=4,
            controls=controls,
        )
        self.assertEqual([Path(path).name for path in decision.selected], ["b", "c"])
        self.assertEqual(decision.skipped_duplicates, ("a",))
        self.assertEqual(decision.deferred_candidates, ("d",))
        with self.assertRaisesRegex(ValueError, "ranking"):
            scheduling_controls(SearchPolicy.create("baseline", 1, {"ranking": "agent_code"}))

    def test_duplicate_avoidance_preserves_a_remaining_attempt(self):
        candidates = self.root / "candidates-dup"
        first = write_candidate(candidates)
        second = write_candidate(candidates)
        policy = SearchPolicy.create("baseline", 1, {
            "candidate_paths": [str(first), str(second)],
            "avoid_duplicates": True,
            "stop_after_unchanged_rounds": 1,
        })
        result = run_search_loop(
            initial_policy=policy,
            candidate_factory=generate_policy_paths,
            policy_proposer=bump_policy,
            max_rounds=3,
            evaluator=fake_eval,
            contract=self.contract,
            budget=ExperimentBudget(max_attempts=3),
            journal=self.journal,
            frozen_root=self.root / "frozen-dup",
            candidates_root=candidates,
            callback_timeout_seconds=10,
        )
        self.assertEqual(result.attempts_started, 2, result.summary())
        self.assertEqual(result.attempts_remaining, 1)
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(result.stop_reason, "policy stop: no new hypotheses")
        self.assertEqual(result.rounds[1].skipped_duplicates, (first.name, second.name))
        self.assertEqual(result.rounds[1].session.submissions, ())
        self.assertIsNone(result.promoted_policy)

    def test_unseen_first_spends_the_remaining_attempt_on_a_new_candidate(self):
        candidates = self.root / "candidates-rank"
        first = write_candidate(candidates)
        second = write_candidate(candidates)
        policy = SearchPolicy.create("baseline", 1, {
            "ranking": "input_order",
            "candidate_paths": [str(first)],
            "next_paths": [str(first), str(second)],
        })
        result = run_search_loop(
            initial_policy=policy,
            candidate_factory=generate_policy_paths,
            policy_proposer=propose_unseen_first,
            max_rounds=2,
            evaluator=fake_eval,
            contract=self.contract,
            budget=ExperimentBudget(max_attempts=2),
            journal=self.journal,
            frozen_root=self.root / "frozen-rank",
            candidates_root=candidates,
            callback_timeout_seconds=10,
        )
        self.assertEqual(len(result.rounds), 2, result.summary())
        second_round = result.rounds[1].session.submissions
        self.assertEqual(second_round[0].candidate_id, second.name)
        self.assertEqual(second_round[0].outcome.value, "completed_no_verdict")
        self.assertEqual(second_round[1].candidate_id, first.name)
        self.assertEqual(second_round[1].outcome.value, "budget_exhausted")

    def test_invalid_ranking_proposal_is_rejected_before_the_next_round(self):
        candidates = self.root / "candidates-bad"
        candidate = write_candidate(candidates)
        with self.assertRaisesRegex(ValueError, "ranking"):
            run_search_loop(
                initial_policy=SearchPolicy.create("baseline", 1, {"ranking": "agent_code"}),
                candidate_factory=generate_policy_paths,
                policy_proposer=stop_proposing,
                max_rounds=1,
                evaluator=fake_eval,
                contract=self.contract,
                budget=ExperimentBudget(max_attempts=1),
                journal=self.journal,
                frozen_root=self.root / "frozen-bad-initial",
                candidates_root=candidates,
            )
        self.assertEqual(self.journal.read_history(), [])
        result = run_search_loop(
            initial_policy=SearchPolicy.create("baseline", 1, {
                "candidate_paths": [str(candidate)],
            }),
            candidate_factory=generate_policy_paths,
            policy_proposer=propose_bad_ranking,
            max_rounds=2,
            evaluator=fake_eval,
            contract=self.contract,
            budget=ExperimentBudget(max_attempts=2),
            journal=self.journal,
            frozen_root=self.root / "frozen-bad-proposal",
            candidates_root=candidates,
            callback_timeout_seconds=10,
        )
        self.assertEqual(len(result.rounds), 1, result.summary())
        self.assertEqual(result.attempts_started, 1)
        self.assertIn("ranking", result.stop_reason)
        self.assertIsNone(result.promoted_policy)

    def test_comparison_elapsed_uses_the_injected_clock(self):
        ticks = {"value": 0.0}

        def clock():
            ticks["value"] += 0.25
            return ticks["value"]

        result = run_policy_comparison(
            incumbent=SearchPolicy.create("incumbent", 1, {"ranking": "old"}),
            training_tasks=(make_task("train-clock", TaskPartition.TRAINING, 26),),
            validation_tasks=(make_task("val-clock", TaskPartition.VALIDATION, 27),),
            confirmation_tasks=(make_task("test-clock", TaskPartition.CONFIRMATION, 28),),
            training_seeds=(131,),
            validation_seeds=(231, 232),
            confirmation_seeds=(331, 332),
            proposer=comparison_proposer,
            executor=comparison_executor,
            budget=ComparisonBudget(200, 30, 60, SearchLimits(5, 1, 2)),
            rules=ComparisonRules(task_weights=(("val-clock", 1.0), ("test-clock", 1.0))),
            journal=self.journal,
            clock=clock,
        )
        self.assertEqual(result.final_decision, "promote")
        self.assertGreater(result.elapsed_seconds, 0)
        self.assertLess(result.elapsed_seconds, 30)


if __name__ == "__main__":
    unittest.main()
