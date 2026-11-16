"""Counterexamples for resource accounting; no model or physical evaluator runs."""

import copy
import unittest

from large_qubit.section11_costs import request_budget, summarize_components, summarize_training


def lane():
    return {"main_by_trajectory": {"method__target": {
        "generation": {"records": 4, "recorded_wall_seconds": 10., "input_tokens": 20,
                       "output_tokens": 8, "wall_is_lower_bound": False, "numerical_work_is_lower_bound": False},
        "search": {"records": 5, "recorded_wall_seconds": 6., "evaluator_calls": 3,
                   "hamiltonian_action_seconds": 4., "wall_is_lower_bound": False, "numerical_work_is_lower_bound": False}}},
        "execution_attempts": [{"status": "finished", "wall_seconds": 12.,
            "recorded_algebraic_candidate_preparation_seconds": .5,
            "completed_algebraic_coordinate_fits_including_replay": 2,
            "partial_algebraic_work_may_be_unknown": False,
            "inference_sessions": [{"unknown_unflushed_model_cost": False,
                                    "cost": {"generation_wall_seconds": 10., "model_load_seconds": 1., "adapter_load_seconds": .2}}],
            "physics_worker_loads": [{"unknown_unflushed_setup_cost": False, "completed": {"wall_seconds": .3}}]}],
        "winner_integrity_overhead": {"method__target": {"uncertain_cost": False, "result": {"evaluator_calls": 1, "wall_seconds": .4}}}}


class ResourceAccountingTests(unittest.TestCase):
    def test_session_generation_crosschecks_without_double_charging(self):
        result = summarize_components({"lane_0": lane()})
        self.assertEqual(result["components"]["generation_seconds"], 10.)
        self.assertEqual(result["components"]["scoring_seconds"], 6.)
        self.assertEqual(result["setup_and_algebra"]["lane_process_seconds"], 12.)
        self.assertNotEqual(10. + 6., result["setup_and_algebra"]["lane_process_seconds"])
        self.assertEqual(result["components"]["physical_hamiltonian_actions"], 3)
        self.assertEqual(result["components"]["logical_requests"], 5)
        self.assertEqual(result["winner_verification"]["physical_hamiltonian_actions"], 1)

    def test_shared_trajectory_in_two_lanes_refused(self):
        with self.assertRaisesRegex(ValueError, "multiple lanes"):
            summarize_components({"lane_0": lane(), "lane_1": lane()})

    def test_session_disagreement_refused(self):
        value = lane()
        value["execution_attempts"][0]["inference_sessions"][0]["cost"]["generation_wall_seconds"] = 11.
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_components({"lane_0": value})

    def test_one_interrupted_lane_does_not_disable_other_crosscheck(self):
        unknown, complete = lane(), lane()
        unknown["execution_attempts"][0]["inference_sessions"][0]["unknown_unflushed_model_cost"] = True
        complete["main_by_trajectory"]["other__target"] = complete["main_by_trajectory"].pop("method__target")
        complete["execution_attempts"][0]["inference_sessions"][0]["cost"]["generation_wall_seconds"] = 11.
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_components({"lane_0": unknown, "lane_1": complete})

    def test_interrupted_and_failed_costs_remain_lower_bounds(self):
        value = lane()
        value["main_by_trajectory"]["method__target"]["search"]["wall_is_lower_bound"] = True
        value["execution_attempts"][0]["status"] = "interrupted"
        value["execution_attempts"][0]["inference_sessions"][0]["unknown_unflushed_model_cost"] = True
        value["execution_attempts"][0]["inference_sessions"][0]["cost"] = None
        result = summarize_components({"lane_0": value})
        self.assertTrue(result["components"]["wall_is_lower_bound"])
        self.assertTrue(result["setup_and_algebra"]["wall_is_lower_bound"])
        self.assertEqual(result["components"]["logical_requests"], 5)

    def test_failed_main_and_rejected_corpus_counted_once(self):
        result = request_budget({"cumulative_requests": 945, "ceiling": 1024},
                                {"failed_v1": 6538, "fresh_v2": 6552, "cumulative": 13090, "ceiling": 14730},
                                {"actual": 1019, "ceiling": 1024},
                                {"failed_attempt": 16, "recovery_attempt": 16382, "cumulative": 16398, "maximum_cumulative": 16400})
        self.assertEqual(result["actual"], 31452)
        self.assertEqual(result["amended_ceiling"], 33178)
        self.assertFalse(result["overhead_included"])

    def test_inconsistent_failed_count_refused(self):
        with self.assertRaisesRegex(ValueError, "Failed main"):
            request_budget({}, {}, {}, {"failed_attempt": 16, "recovery_attempt": 16382, "cumulative": 16382})

    def test_training_actual_replay_and_preflight_are_distinct(self):
        cost = {"additional_energy_requests": 0, "committed_exposure": {"optimizer_steps": 10, "examples_seen": 160},
                "actual_totals": {"optimizer_steps_completed": 12}, "uncommitted_or_replayed_work": {"optimizer_steps_completed": 2},
                "training_wall_seconds": 20., "model_load_seconds": 2., "attributable_wall_seconds": 22.,
                "assigned_device_hours": 22/3600, "wall_is_lower_bound": True,
                "peak_memory_is_lower_bound": True, "peak_allocated_bytes": 100, "peak_reserved_bytes": 120,
                "attempt_count": 2, "attempts": []}
        bench = {"additional_energy_requests": 0, "model_load_seconds": 1.,
                 "benchmark_wall_seconds_including_warmup_and_failed_candidates": 3.,
                 "disposable_actual_work": {"optimizer_steps_completed": 2}, "candidates": [{"success": False}, {"success": True}]}
        result = summarize_training({"fit": {"cost_summary": cost}}, {"fit": {"benchmark": bench}})
        self.assertEqual(result["attributable_seconds"], 22.)
        self.assertEqual(result["preflight_seconds_including_load"], 4.)
        self.assertTrue(result["wall_is_lower_bound"])
        self.assertEqual(result["fits"]["fit"]["committed_optimizer_steps"], 10)
        self.assertEqual(result["fits"]["fit"]["actual_work"]["optimizer_steps_completed"], 12)
        self.assertEqual(result["fits"]["fit"]["preflight_failed_candidates"], 1)
        wrong = copy.deepcopy(cost)
        wrong["attributable_wall_seconds"] = 24.
        with self.assertRaisesRegex(ValueError, "double counted"):
            summarize_training({"fit": {"cost_summary": wrong}}, {"fit": {"benchmark": bench}})


if __name__ == "__main__":
    unittest.main()
