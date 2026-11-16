"""Synthetic ledger fixtures only; these are not molecular energy results."""

import copy
import unittest

from large_qubit.common import digest
from large_qubit.endpoints import (PHYSICS_KEYS, aggregate_targets, model_geometry_payload,
                                   outcome_spec, summarize_trajectory)
from large_qubit.protocol import BENCHMARK_ID


def fixture(target="CO2_I01", method="training_circuit_retrieval"):
    bindings = {"benchmark_id": BENCHMARK_ID, "target_id": target, "method": method,
                "pipeline_seed": 22, "hamiltonian_sha256": "1" * 64,
                "orbital_alignment_sha256": "2" * 64, "catalogue_sha256": "3" * 64,
                "backend_sha256": "4" * 64, "precision": "complex128",
                "final_freeze_sha256": "5" * 64, "final_protocol_frozen": True}
    reference = {"schema_version": "large-qubit-sector-reference-v1",
                 "physics": {key: bindings[key] for key in PHYSICS_KEYS},
                 "solver_sha256": "6" * 64, "energy_hartree": -11.0, "n_alpha": 11,
                 "n_beta": 11, "residual_norm": 1e-10, "nuclear_repulsion_included": True}
    requests = [
        {"request_ordinal": 1, "kind": "hf", "circuit_sha256": "a" * 64, "status": "scored",
         "energy_hartree": -10.0, "cache_hit": False, "duplicate_of": None, "failure_reason": None},
        {"request_ordinal": 2, "kind": "candidate", "circuit_sha256": "b" * 64, "status": "scored",
         "energy_hartree": -10.5, "cache_hit": False, "duplicate_of": None, "failure_reason": None},
        {"request_ordinal": 3, "kind": "candidate", "circuit_sha256": "b" * 64, "status": "scored",
         "energy_hartree": -10.5, "cache_hit": True, "duplicate_of": 2, "failure_reason": None},
    ]
    trajectory = {"schema_version": "large-qubit-scored-trajectory-v1", "bindings": copy.deepcopy(bindings),
                  "status": "finished", "stop_reason": "retrieval_exhausted", "failure_reason": None,
                  "requests": requests, "completion_attempts": 0}
    verification = {"schema_version": "large-qubit-winner-verification-v1",
                    "bindings": copy.deepcopy(bindings), "request_ordinal": 2, "circuit_sha256": "b" * 64,
                    "independent_backend_sha256": "7" * 64, "energy_hartree": -10.5,
                    "normalization_error": 1e-12, "population_leakage": 1e-12}
    return trajectory, reference, verification, bindings


def summarize(data, pins=None):
    trajectory, reference, verification, bindings = data
    pins = pins or {"trajectory": digest(trajectory), "reference": digest(reference),
                    "verification": digest(verification) if verification is not None else None}
    return summarize_trajectory(trajectory, reference, verification, expected_bindings=bindings,
                                expected_artifact_sha256=pins)


def aggregate(summaries, pins=None):
    pins = pins or {alias: digest(summary) for alias, summary in summaries.items()}
    return aggregate_targets(summaries, expected_summary_sha256=pins)


class EndpointTest(unittest.TestCase):
    def test_hf_duplicates_cache_hits_and_early_incumbent_are_accounted_without_invented_requests(self):
        result = summarize(fixture())
        self.assertEqual(result["actual_logical_requests"], 3)
        self.assertEqual(result["endpoint_budget"], 1024)
        self.assertTrue(result["incumbent_carried_forward"])
        self.assertEqual(result["winning_request_ordinal"], 2)
        self.assertEqual((result["duplicate_requests"], result["cache_hits"]), (1, 1))
        self.assertEqual(result["best_verified_error_hartree"], 0.5)
        self.assertEqual(result["improvement_over_hf_hartree"], 0.5)

    def test_payload_modification_without_trusted_pin_change_is_rejected(self):
        data = fixture()
        pins = {"trajectory": digest(data[0]), "reference": digest(data[1]), "verification": digest(data[2])}
        data[0]["requests"].pop()
        with self.assertRaisesRegex(ValueError, "trusted manifest"):
            summarize(data, pins)
        data = fixture()
        data[0]["authenticated"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            summarize(data)

    def test_final_freeze_fingerprints_and_reference_sector_are_required(self):
        for edit in ("freeze", "fingerprint", "reference_sector", "nuclear", "residual"):
            data = fixture()
            if edit == "freeze":
                data[3]["final_protocol_frozen"] = False
            elif edit == "fingerprint":
                data[0]["bindings"]["hamiltonian_sha256"] = "8" * 64
            elif edit == "reference_sector":
                data[1]["n_alpha"] = True
            elif edit == "nuclear":
                data[1]["nuclear_repulsion_included"] = False
            else:
                data[1]["residual_norm"] = 1e-4
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                summarize(data)

    def test_missing_repeated_or_bool_ordinals_and_missing_hf_fail(self):
        for edit in ("gap", "repeat", "bool", "hf", "duplicate"):
            data = fixture()
            rows = data[0]["requests"]
            if edit == "gap":
                rows[1]["request_ordinal"] = 4
            elif edit == "repeat":
                rows[2]["request_ordinal"] = 2
            elif edit == "bool":
                rows[0]["request_ordinal"] = True
            elif edit == "hf":
                rows[0]["kind"] = "candidate"
            else:
                rows[2]["duplicate_of"] = None
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                summarize(data)

    def test_full_budget_claim_requires_all_requests_and_cap_cannot_be_refunded(self):
        data = fixture()
        data[0]["stop_reason"] = "request_cap"
        with self.assertRaisesRegex(ValueError, "Truncated"):
            summarize(data)
        rows = data[0]["requests"]
        for ordinal in range(4, 1025):
            rows.append(dict(rows[2], request_ordinal=ordinal))
        result = summarize(data)
        self.assertEqual(result["actual_logical_requests"], 1024)
        self.assertFalse(result["incumbent_carried_forward"])
        self.assertEqual(result["duplicate_requests"], 1022)
        rows.append(dict(rows[-1], request_ordinal=1025))
        with self.assertRaisesRegex(ValueError, "ledger"):
            summarize(data)

    def test_generator_early_stop_requires_actual_attempt_ceiling(self):
        data = fixture(method="gpt2_xl_frozen")
        data[0]["stop_reason"] = "attempt_ceiling"
        data[0]["completion_attempts"] = 2
        with self.assertRaisesRegex(ValueError, "ceiling"):
            summarize(data)
        data[0]["completion_attempts"] = 2048
        self.assertTrue(summarize(data)["incumbent_carried_forward"])

    def test_independent_winner_verification_is_not_a_success_boolean(self):
        for edit in ("wrong_winner", "same_backend", "energy", "norm", "missing"):
            data = fixture()
            if edit == "wrong_winner":
                data[2]["request_ordinal"] = 1
            elif edit == "same_backend":
                data[2]["independent_backend_sha256"] = data[3]["backend_sha256"]
            elif edit == "energy":
                data[2]["energy_hartree"] += 0.001
            elif edit == "norm":
                data[2]["normalization_error"] = 1e-4
            else:
                data = data[0], data[1], {"verified": True}, data[3]
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                summarize(data)

    def test_signed_small_negative_error_is_preserved_and_large_negative_refused(self):
        for offset in (-5e-9, -2e-8):
            data = fixture()
            for row in data[0]["requests"][1:]:
                row["energy_hartree"] = -11.0 + offset
            data[2]["energy_hartree"] = -11.0 + offset
            if offset > -1e-8:
                self.assertLess(summarize(data)["best_verified_error_hartree"], 0.0)
            else:
                with self.assertRaisesRegex(ValueError, "below reference"):
                    summarize(data)

    def test_nonfinite_and_boolean_energies_are_rejected(self):
        for value in (float("nan"), float("inf"), True, 10 ** 400):
            data = fixture()
            data[0]["requests"][1]["energy_hartree"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(data)

    def test_explicit_initialization_failure_is_retained_without_fabricated_incumbent(self):
        data = fixture()
        data[0].update(status="failed", stop_reason="failure", failure_reason="synthetic initialization failure",
                       requests=[])
        data = data[0], data[1], None, data[3]
        result = summarize(data)
        self.assertEqual(result["actual_logical_requests"], 0)
        self.assertIsNone(result["best_verified_error_hartree"])
        self.assertFalse(result["incumbent_carried_forward"])

    def test_two_fixed_target_mean_requires_both_and_preserves_failure(self):
        first = summarize(fixture("CO2_I01"))
        second = summarize(fixture("CO2_I02"))
        summaries = {"CO2_I01": first, "CO2_I02": second}
        self.assertEqual(aggregate(summaries)["mean_verified_error_hartree"], 0.5)
        with self.assertRaises(ValueError):
            aggregate({"CO2_I01": first})
        second["trajectory_status"] = "failed"
        second["failure_reason"] = "synthetic failure after incumbent"
        result = aggregate(summaries)
        self.assertIsNone(result["mean_verified_error_hartree"])
        self.assertEqual(result["failed_targets"], ["CO2_I02"])
        self.assertEqual(result["targets"]["CO2_I02"]["best_verified_error_hartree"], 0.5)

    def test_summary_mutation_cannot_reuse_original_trusted_pins(self):
        summaries = {alias: summarize(fixture(alias)) for alias in ("CO2_I01", "CO2_I02")}
        pins = {alias: digest(summary) for alias, summary in summaries.items()}
        summaries["CO2_I02"]["best_verified_error_hartree"] = 0.0
        with self.assertRaisesRegex(ValueError, "trusted manifest"):
            aggregate(summaries, pins)
        with self.assertRaises(TypeError):
            aggregate_targets(summaries)

    def test_finite_inputs_cannot_overflow_signed_errors_and_mean_is_overflow_safe(self):
        data = fixture()
        data[1]["energy_hartree"] = -1e308
        data[0]["requests"][0]["energy_hartree"] = 1e308
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            summarize(data)
        summaries = {alias: summarize(fixture(alias)) for alias in ("CO2_I01", "CO2_I02")}
        for summary in summaries.values():
            summary.update(best_verified_error_hartree=1e308, best_verified_energy_hartree=1e308,
                           reference_energy_hartree=0.0)
        self.assertEqual(aggregate(summaries)["mean_verified_error_hartree"], 1e308)

    def test_model_payload_contains_structure_only_and_rejects_reference_or_cache_injection(self):
        payload = model_geometry_payload("CO2_I01")
        self.assertEqual(payload["bond_length_angstrom"], 1.1)
        self.assertEqual(set(payload), {"benchmark_id", "geometry_id", "molecule", "elements",
                                        "coordinates_angstrom", "bond_length_angstrom", "coordinate_unit",
                                        "basis", "charge", "electrons", "spatial_orbitals", "qubits",
                                        "n_alpha", "n_beta", "mapping"})
        payload["coordinates_angstrom"][0][2] = 100.0
        self.assertEqual(model_geometry_payload("CO2_I01")["coordinates_angstrom"][0][2], -1.1)
        with self.assertRaises(TypeError):
            model_geometry_payload("CO2_I01", reference_energy=-11.0)
        self.assertTrue(outcome_spec()["hf_included"])


if __name__ == "__main__":
    unittest.main()
