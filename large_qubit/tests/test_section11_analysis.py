"""Independent synthetic endpoint/accounting counterexamples for Section 11."""

import copy
import unittest

from large_qubit.section11_analysis import METHODS, TARGETS, summarize


def fixture():
    errors = {"gpt2_xl_frozen": (.12, .14), "gpt2_xl_weighted_lora": (.08, .06),
              "gpt2_xl_uniform_lora": (.09, .07), "qwen3_4b_frozen": (.16, .18),
              "qwen3_4b_weighted_lora": (.07, .06), "uniform_random": (.15, .16),
              "training_circuit_retrieval": (.08, .05), "conventional_vqe": (.07, .08)}
    matrix = {"jobs": {}, "resources": {}, "methods": {}, "paired_comparisons": {},
              "failed_jobs": [], "missing_jobs": [], "all_endpoints_verified": True}
    for method in METHODS:
        matrix["methods"][method] = {"mean_verified_error_hartree": sum(errors[method]) / 2}
        for i, target in enumerate(TARGETS):
            key = f"{method}__{target}"
            energy, reference, hf = -185.2 + errors[method][i], -185.2, -185.
            request_count = 1023 if method == "conventional_vqe" else 1024
            curve = [{"logical_requests": n, "best_energy_hartree": hf if n == 1 else energy,
                      "actual_request": n <= request_count, "carried_after_stop": n > request_count}
                     for n in range(1, 1025)]
            matrix["jobs"][key] = {"status": "complete", "incumbent_verified": True,
                                    "best_verified_energy_hartree": energy,
                                    "best_verified_error_hartree": errors[method][i],
                                    "reference_energy_hartree": reference, "hf_energy_hartree": hf,
                                    "hf_improvement_hartree": hf - energy,
                                    "budget_curve": curve, "best_request": 2,
                                    "counts": {"logical_requests": request_count, "duplicate_requests": 1},
                                    "unused_allowance": 1024 - request_count,
                                    "best_query": {"parameters": [.1]} if method == "conventional_vqe" else {"ops": [1, 2]}}
            matrix["resources"][key] = {"native_family": "fixture",
                                        "resources_excluding_hf": {"depth": 5, "two_qubit_count": 4, "two_qubit_depth": 3, "gates": 10},
                                        "resources_including_hf": {"depth": 6, "two_qubit_count": 4, "two_qubit_depth": 3, "gates": 32}}
    for key, left, right, diff in [
        ("gpt_weighted_minus_frozen", "gpt2_xl_weighted_lora", "gpt2_xl_frozen", -.06),
        ("gpt_weighted_minus_uniform", "gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora", -.01),
        ("qwen_weighted_minus_frozen", "qwen3_4b_weighted_lora", "qwen3_4b_frozen", -.105),
        ("qwen_weighted_minus_gpt_weighted", "qwen3_4b_weighted_lora", "gpt2_xl_weighted_lora", -.005),
    ]:
        matrix["paired_comparisons"][key] = {"left_method": left, "right_method": right, "mean_difference_hartree": diff}
    return matrix


class DescriptiveAnalysisTests(unittest.TestCase):
    def test_paired_differences_use_same_target_and_mha(self):
        result = summarize(fixture(), [(1, 2)])
        paired = {r["comparison"]: r for r in result["paired_rows"]}
        r = paired["gpt_weighted_minus_frozen"]
        self.assertAlmostEqual(r["I01_difference_mha"], -40.)
        self.assertAlmostEqual(r["I02_difference_mha"], -80.)
        self.assertAlmostEqual(r["mean_difference_mha"], -60.)
        self.assertEqual(len(result["endpoint_rows"]), 16)
        self.assertEqual(len(result["curve_rows"]), 16384)
        self.assertTrue(all(r["winner_exactly_in_fitting_corpus"] for r in result["endpoint_rows"] if r["method"] != "conventional_vqe"))
        self.assertTrue(all(r["winner_exactly_in_fitting_corpus"] is None for r in result["endpoint_rows"] if r["method"] == "conventional_vqe"))

    def test_vqe_carried_point_does_not_create_request(self):
        result = summarize(fixture())
        rows = [r for r in result["endpoint_rows"] if r["method"] == "conventional_vqe"]
        self.assertEqual([r["actual_requests"] for r in rows], [1023, 1023])
        self.assertEqual([r["carried_allowance"] for r in rows], [1, 1])
        wrong = fixture()
        wrong["jobs"]["conventional_vqe__CO2_I01"]["budget_curve"][-1]["actual_request"] = True
        with self.assertRaisesRegex(ValueError, "Carried budget"):
            summarize(wrong)

    def test_missing_or_failed_trajectory_is_not_silently_excluded(self):
        missing = fixture()
        missing["jobs"].pop("gpt2_xl_frozen__CO2_I02")
        with self.assertRaisesRegex(ValueError, "sixteen"):
            summarize(missing)
        failed = fixture()
        failed["failed_jobs"] = ["gpt2_xl_frozen__CO2_I02"]
        with self.assertRaisesRegex(ValueError, "Incomplete outcomes"):
            summarize(failed)

    def test_unauthenticated_winner_is_refused(self):
        matrix = fixture()
        matrix["jobs"]["gpt2_xl_frozen__CO2_I01"]["incumbent_verified"] = False
        with self.assertRaisesRegex(ValueError, "Unverified"):
            summarize(matrix)

    def test_worsening_incumbent_and_wrong_paired_math_are_refused(self):
        matrix = fixture()
        matrix["jobs"]["gpt2_xl_frozen__CO2_I01"]["budget_curve"][5]["best_energy_hartree"] += .01
        with self.assertRaisesRegex(ValueError, "Incumbent worsened"):
            summarize(matrix)
        matrix = fixture()
        matrix["paired_comparisons"]["gpt_weighted_minus_frozen"]["mean_difference_hartree"] = .06
        with self.assertRaisesRegex(ValueError, "Paired arithmetic"):
            summarize(matrix)

    def test_curve_absolute_tolerance_is_not_scaled_by_nuclear_energy(self):
        matrix = fixture()
        # A 0.1 micro-Hartree drift is ten times the declared 1e-8 Ha
        # tolerance, but Python's default relative tolerance at -185 Ha
        # would incorrectly accept it.
        matrix["jobs"]["gpt2_xl_frozen__CO2_I01"]["budget_curve"][-1]["best_energy_hartree"] -= 1e-7
        with self.assertRaisesRegex(ValueError, "Curve and verified endpoint"):
            summarize(matrix)

    def test_missing_paired_comparison_is_refused(self):
        matrix = fixture()
        matrix["paired_comparisons"].pop("gpt_weighted_minus_uniform")
        with self.assertRaises(ValueError):
            summarize(matrix)


if __name__ == "__main__":
    unittest.main()
