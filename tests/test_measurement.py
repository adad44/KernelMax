"""Regression checks for swap triage, pass bounds and phase attribution limits."""

import copy
import unittest
from types import SimpleNamespace

from evaluator.measurement import (
    BoundedExploration,
    ExplorationComplete,
    ObservedModel,
    PhaseMemoryObserver,
    assess_environment,
    memory_delta,
    memory_snapshot,
    parse_vm_stat,
)


def environment():
    return {
        "power": "Now drawing from 'AC Power'",
        "swap_used_mib": 100.0,
        "thermal": "No thermal warning level has been recorded\nNo performance warning level has been recorded",
        "power_settings": "lowpowermode         0",
        "thermal_state": 0,
        "swapins": 100,
        "swapouts": 100,
    }


VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages wired down: 200.
File-backed pages: 300.
Pages occupied by compressor: 50.
Swapins: 10.
Swapouts: 20.
Pageins: 30.
Pageouts: 40.
"""


class MeasurementPolicyTests(unittest.TestCase):
    def test_swap_affected_exploration_finishes_but_cannot_be_confirmation(self):
        before = environment()
        after = {**before, "swap_used_mib": 107.44, "swapouts": 1088}
        exploratory = assess_environment(before, after, mode="exploration")
        strict = assess_environment(before, after, mode="confirmation")
        self.assertTrue(exploratory["swap_affected"])
        self.assertTrue(exploratory["continue_pass"])
        self.assertFalse(exploratory["confirmation_sample_eligible"])
        self.assertEqual(exploratory["flags"], strict["flags"])
        self.assertFalse(strict["continue_pass"])
        self.assertFalse(strict["confirmation_sample_eligible"])

    def test_valid_exploratory_sample_alone_still_cannot_confirm(self):
        before = environment()
        self.assertFalse(
            assess_environment(before, before, mode="exploration")[
                "confirmation_sample_eligible"
            ]
        )
        self.assertTrue(
            assess_environment(before, before, mode="confirmation")[
                "confirmation_sample_eligible"
            ]
        )

    def test_swapins_remain_logged_under_paging_aware_confirmation(self):
        before = environment()
        self.assertTrue(
            assess_environment(before, {**before, "swapins": 200}, mode="confirmation")[
                "continue_pass"
            ]
        )

    def test_power_thermal_and_counter_failures_still_stop_exploration(self):
        before = environment()
        for changed in (
            {"power": "Now drawing from 'Battery Power'"},
            {"thermal_state": 1},
            {"swapouts": 99},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(
                    assess_environment(
                        before, {**before, **changed}, mode="exploration"
                    )["continue_pass"]
                )
        self.assertFalse(
            assess_environment(
                before, before, mode="exploration", thermal_states=[0, 1, 0]
            )["continue_pass"]
        )

    def test_mode_cannot_come_from_an_unknown_policy(self):
        with self.assertRaises(ValueError):
            assess_environment(environment(), environment(), mode="ignore")

    def test_predeclared_request_and_wall_bounds_stop_without_resetting_counts(self):
        tick = [0.0]
        policy = SimpleNamespace(max_requests=5, max_wall_time_seconds=20)
        bounded = BoundedExploration(
            policy, clock=lambda: tick[0], max_requests=2, max_seconds=10
        )
        bounded.begin_request()
        bounded.begin_request()
        with self.assertRaisesRegex(ExplorationComplete, "request"):
            bounded.begin_request()
        self.assertEqual(bounded.requests, 2)
        bounded = BoundedExploration(policy, clock=lambda: tick[0])
        tick[0] = 20
        with self.assertRaisesRegex(ExplorationComplete, "wall-time"):
            bounded.begin_request()
        self.assertEqual(bounded.requests, 0)

    def test_controller_cannot_exceed_contract_bounds(self):
        policy = SimpleNamespace(max_requests=5, max_wall_time_seconds=20)
        for kwargs in (
            {"max_requests": 0},
            {"max_requests": True},
            {"max_requests": 6},
            {"max_seconds": 21},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BoundedExploration(policy, **kwargs)


class PhaseMemoryTests(unittest.TestCase):
    def test_actual_page_size_drives_memory_and_swap_rate(self):
        vm = parse_vm_stat(VM)
        before = {"monotonic": 1.0, "vm": vm, "swap_used_bytes": 100}
        after = copy.deepcopy(before)
        after["monotonic"] = 3.0
        after["vm"]["counts"]["Swapouts"] += 4
        after["swap_used_bytes"] = 90
        result = memory_delta(before, after)
        self.assertEqual(result["Swapouts"]["pages_per_second"], 2)
        self.assertEqual(result["Swapouts"]["bytes_per_second"], 32768)
        self.assertEqual(result["swap_used_delta_bytes"], -10)
        self.assertEqual(result["Swapouts"]["pages"], 4)

    def test_missing_or_reset_counters_are_not_zero_swap(self):
        before = {"monotonic": 1.0, "vm": parse_vm_stat(VM), "swap_used_bytes": None}
        after = copy.deepcopy(before)
        after["monotonic"] = 2.0
        after["vm"]["counts"]["Swapouts"] = 1
        result = memory_delta(before, after)
        self.assertIsNone(result["Swapouts"]["bytes_per_second"])
        self.assertEqual(result["counter_errors"], ["Swapouts decreased"])
        self.assertIsNone(result["Compressions"]["pages"])
        self.assertIsNone(parse_vm_stat("")["page_size_bytes"])

    def test_read_only_snapshot_keeps_pressure_and_process_context(self):
        def read(args):
            if args == ["vm_stat"]:
                return VM
            if args[-1] == "vm.swapusage":
                return "total = 1024.00M used = 100.00M free = 924.00M (encrypted)"
            if args[-1] == "kern.memorystatus_vm_pressure_level":
                return "2"
            if args[0] == "ps":
                return "12"
            raise AssertionError(args)

        result = memory_snapshot(read=read)
        self.assertEqual(result["memory_pressure_level_raw"], 2)
        self.assertEqual(result["memory_bytes"]["Pages wired down"], 200 * 16384)
        self.assertEqual(result["process"]["rss_bytes"], 12 * 1024)
        self.assertEqual(result["swap_used_bytes"], 100 * 1024**2)
        self.assertEqual(result["errors"], [])

    def test_chunked_prompt_singleton_is_not_mislabeled_decode(self):
        calls = []

        def snapshot(_):
            calls.append(1)
            return {
                "monotonic": float(len(calls)),
                "vm": parse_vm_stat(VM),
                "swap_used_bytes": 100,
                "observation_wall_seconds": 0.001,
            }

        observer = PhaseMemoryObserver(19, snapshot=snapshot, interval_seconds=3600)
        observer.start()
        for count in (3, 3, 3, 3, 3, 3, 1):
            observer.before_model_call(count)
            self.assertEqual(observer.phase, "prefill")
        observer.before_model_call(1)
        self.assertEqual(observer.phase, "decode")
        result = observer.finish()
        self.assertEqual(
            [s["phase"] for s in result["samples"]],
            ["before_request", "prefill", "decode", "after_decode"],
        )
        self.assertEqual(set(result["phase_deltas"]), {"prefill", "decode"})
        self.assertFalse(result["confirmation_eligible"])

    def test_model_proxy_preserves_original_arguments_result_and_attributes(self):
        class Model:
            marker = object()

            def __call__(self, value, *, cache):
                self.received = (value, cache)
                return self.marker

        observer = SimpleNamespace(before_model_call=lambda count: None)
        model = Model()
        proxy = ObservedModel(model, observer)
        value = SimpleNamespace(shape=(1, 8))
        cache = object()
        self.assertIs(proxy(value, cache=cache), model.marker)
        self.assertEqual(model.received, (value, cache))
        self.assertIs(proxy.marker, model.marker)


if __name__ == "__main__":
    unittest.main()
