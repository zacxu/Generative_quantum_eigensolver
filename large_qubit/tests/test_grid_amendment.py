"""Synthetic byte/JSON evidence for amendment provenance; no numerical physics."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from large_qubit.common import read_json, sha256_file
from large_qubit.definitions import build_definition
from large_qubit.registration import GRID_AMENDMENT_PATH, verify_grid_amendment


class GridAmendmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = "large_qubit/archives/section3_physical_v1"
        self.failed = "large_qubit/physics/section3_v1"
        self.private = "large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/physical_validation/section3_v1"
        self.old_sources = ("large_qubit/chemistry.py", "large_qubit/definitions.py", "large_qubit/orbitals.py",
                            "large_qubit/physical_definition.json", "large_qubit/section3_policies.json")
        old = copy.deepcopy(build_definition())
        old["definition_id"] = "co2_physical_definition_v1"
        old.pop("grid_amendment")
        old["geometries"]["CO2_T03"]["bond_length_angstrom"] = 1.40
        old["geometries"]["CO2_T03"]["coordinates_angstrom"][0][2] = -1.40
        old["geometries"]["CO2_T03"]["coordinates_angstrom"][2][2] = 1.40
        policies = {"scf_policy": {"method": "RHF", "stability_negative_eigenvalue_tolerance": 1e-7},
                    "orbital_conventions": {"anchor_geometry_id": "CO2_T02"}}
        for relative in self.old_sources[:3]:
            self.write(f"{self.archive}/{relative}", 'raise RuntimeError("Never execute archived source")\n')
        self.write(f"{self.archive}/large_qubit/physical_definition.json", old)
        self.write(f"{self.archive}/large_qubit/section3_policies.json", policies)
        self.write("large_qubit/section3_policies.json", policies)
        self.write(f"{self.archive}/snapshot.json", {
            "schema": "section3-pre-amendment-snapshot-v1",
            "bindings_sha256": {name: sha256_file(self.root / self.archive / name) for name in self.old_sources}})
        diagnostics = {
            "geometry_id": "CO2_T03", "all_passed": False, "physical_validation_passed": False,
            "diagnostics": {"geometry": old["geometries"]["CO2_T03"], "scf": {"converged": True},
                            "physical_validation_passed": False, "correlated_reference_computed": False,
                            "scf_policy": policies["scf_policy"],
                            "stability": {"all_channels_stable": False,
                                          "channels": {"external_RHF_to_UHF": {
                                              "stable": False, "minimum_eigenvalue": -0.01}}}}}
        self.write(f"{self.private}/failure_diagnostics.json", diagnostics)
        self.write(f"{self.failed}/failure.json", {
            "all_passed": False, "physical_validation_passed": False,
            "benchmark_id": "gqe-co2-sto3g-30q-v1", "failed_geometry_id": "CO2_T03",
            "completed_geometries": ["CO2_T02", "CO2_T01"],
            "diagnostics_path": f"{self.private}/failure_diagnostics.json",
            "diagnostics_sha256": sha256_file(self.root / self.private / "failure_diagnostics.json"),
            "wall_seconds": 3.0})
        self.write(f"{self.failed}/failure.sha256", sha256_file(self.root / self.failed / "failure.json") + "\n")
        for alias in ("CO2_T01", "CO2_T02"):
            self.write(f"{self.private}/{alias}.json", {"synthetic_metadata": True})
            self.write(f"{self.private}/{alias}.npz", b"synthetic bytes; never load as an array")
        paths = [f"{self.archive}/{name}" for name in self.old_sources] + [f"{self.archive}/snapshot.json",
            f"{self.failed}/failure.json", f"{self.failed}/failure.sha256", f"{self.private}/failure_diagnostics.json"]
        paths += [f"{self.private}/{alias}.{suffix}" for alias in ("CO2_T01", "CO2_T02") for suffix in ("json", "npz")]
        self.record = {
            "schema_version": "section3-grid-amendment-v2", "amendment_id": "training_T03_stable_RHF_v2",
            "benchmark_id": "gqe-co2-sto3g-30q-v1", "before_production": True, "geometry_id": "CO2_T03",
            "old_bond_length_angstrom": 1.40, "new_bond_length_angstrom": 1.35,
            "reason": "external RHF-to-UHF instability under the predeclared stable-RHF convention; not an invalid molecular Hamiltonian",
            "decision_inputs": "SCF convergence and stability only; no candidate-circuit or correlated-reference energies",
            "failed_minimum_external_hessian_eigenvalue": -0.01,
            "thresholds_or_SCF_method_changed": False, "training_stretch_range_reduced": True,
            "requires_all_six_geometries_to_pass_unchanged_checks": True,
            "unchanged_final_targets": {"CO2_I01": 1.10, "CO2_I02": 1.30},
            "incurred_costs": {"SCF_geometries_attempted": 3, "circuit_energy_requests": 0,
                               "failed": 1, "passed": 2, "wall_seconds": 3.0},
            "preserved_artifacts_sha256": {p: sha256_file(self.root / p) for p in paths}}
        prior_path = "large_qubit/amendments/section3_grid_v2.json"
        self.write(prior_path, self.record)
        prior = copy.deepcopy(self.record)
        old_archive, old_private, old_failed = self.archive, self.private, self.failed
        self.archive = "large_qubit/archives/section3_physical_v2"
        self.private = self.private.removesuffix("v1") + "v2"
        self.failed = self.failed.removesuffix("v1") + "v2"
        for name in self.old_sources:
            self.write(f"{self.archive}/{name}", (self.root / old_archive / name).read_bytes())
        old["definition_id"] = "co2_physical_definition_v2"
        old["geometries"]["CO2_T03"]["bond_length_angstrom"] = 1.35
        old["geometries"]["CO2_T03"]["coordinates_angstrom"][0][2] = -1.35
        old["geometries"]["CO2_T03"]["coordinates_angstrom"][2][2] = 1.35
        self.write(f"{self.archive}/large_qubit/physical_definition.json", old)
        self.write(f"{self.archive}/snapshot.json", {
            "schema": "section3-pre-amendment-snapshot-v2",
            "bindings_sha256": {name: sha256_file(self.root / self.archive / name) for name in self.old_sources}})
        diagnostics["diagnostics"]["geometry"] = old["geometries"]["CO2_T03"]
        diagnostics["diagnostics"]["stability"]["channels"]["external_RHF_to_UHF"]["minimum_eigenvalue"] = -0.005
        self.write(f"{self.private}/failure_diagnostics.json", diagnostics)
        failure = read_json(self.root / old_failed / "failure.json")
        failure.update(diagnostics_path=f"{self.private}/failure_diagnostics.json",
                       diagnostics_sha256=sha256_file(self.root / self.private / "failure_diagnostics.json"),
                       wall_seconds=4.0)
        self.write(f"{self.failed}/failure.json", failure)
        self.write(f"{self.failed}/failure.sha256", sha256_file(self.root / self.failed / "failure.json") + "\n")
        for alias in ("CO2_T01", "CO2_T02"):
            for suffix in ("json", "npz"):
                self.write(f"{self.private}/{alias}.{suffix}", (self.root / old_private / f"{alias}.{suffix}").read_bytes())
        current_paths = [f"{self.archive}/{name}" for name in self.old_sources] + [
            f"{self.archive}/snapshot.json", f"{self.failed}/failure.json", f"{self.failed}/failure.sha256",
            f"{self.private}/failure_diagnostics.json"]
        current_paths += [f"{self.private}/{alias}.{suffix}" for alias in ("CO2_T01", "CO2_T02") for suffix in ("json", "npz")]
        self.record.update(schema_version="section3-grid-amendment-v3", amendment_id="training_T03_stable_RHF_v3",
                           old_bond_length_angstrom=1.35, new_bond_length_angstrom=1.32,
                           failed_minimum_external_hessian_eigenvalue=-0.005,
                           prior_amendment={"path": prior_path, "sha256": sha256_file(self.root / prior_path)},
                           cumulative_failed_attempt_costs={"SCF_geometries_attempted": 6, "circuit_energy_requests": 0,
                                                            "failed": 2, "passed": 4, "wall_seconds": 7.0})
        self.record["incurred_costs"]["wall_seconds"] = 4.0
        self.record["preserved_artifacts_sha256"] = {
            **prior["preserved_artifacts_sha256"], prior_path: sha256_file(self.root / prior_path),
            **{p: sha256_file(self.root / p) for p in current_paths}}
        self.write(GRID_AMENDMENT_PATH, self.record)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")

    def repin(self):
        for relative in self.record["preserved_artifacts_sha256"]:
            self.record["preserved_artifacts_sha256"][relative] = sha256_file(self.root / relative)
        self.write(GRID_AMENDMENT_PATH, self.record)

    def test_all_27_chain_artifacts_are_verified_without_importing_scientific_code(self):
        report = verify_grid_amendment(self.root)
        self.assertEqual(report["preserved_artifacts_verified"], 27)
        self.assertEqual(report["prior_amendment_verification"]["preserved_artifacts_verified"], 13)
        self.assertEqual(report["cumulative_failed_attempt_costs"]["SCF_geometries_attempted"], 6)
        self.assertTrue(report["failed_attempt_preserved"])
        self.assertFalse(report["scientific_arrays_loaded"])
        script = ('import sys; from large_qubit.registration import verify_grid_amendment; '
                  'verify_grid_amendment(sys.argv[1]); '
                  'assert not ({"numpy", "scipy", "pyscf", "torch", "transformers"} & set(sys.modules))')
        subprocess.run([sys.executable, "-B", "-c", script, str(self.root)], check=True)

    def test_scientific_scope_feedback_flags_types_and_costs_cannot_be_changed(self):
        for field, value in (("before_production", False), ("thresholds_or_SCF_method_changed", True),
                             ("new_bond_length_angstrom", 1.36), ("old_bond_length_angstrom", True),
                             ("decision_inputs", "selected using target performance"),
                             ("failed_minimum_external_hessian_eigenvalue", 0.01)):
            record = copy.deepcopy(self.record)
            record[field] = value
            self.write(GRID_AMENDMENT_PATH, record)
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_grid_amendment(self.root)
        record = copy.deepcopy(self.record)
        record["incurred_costs"]["circuit_energy_requests"] = False
        self.write(GRID_AMENDMENT_PATH, record)
        with self.assertRaises(ValueError):
            verify_grid_amendment(self.root)

    def test_missing_extra_and_changed_preserved_artifacts_are_refused(self):
        record = copy.deepcopy(self.record)
        record["preserved_artifacts_sha256"].pop(next(iter(record["preserved_artifacts_sha256"])))
        self.write(GRID_AMENDMENT_PATH, record)
        with self.assertRaisesRegex(ValueError, "every original"):
            verify_grid_amendment(self.root)
        self.write(GRID_AMENDMENT_PATH, self.record)
        self.write(f"{self.private}/CO2_T01.npz", b"changed bytes")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            verify_grid_amendment(self.root)

    def test_unchanged_policy_flag_cannot_hide_a_threshold_change(self):
        policy = read_json(self.root / "large_qubit/section3_policies.json")
        policy["scf_policy"]["stability_negative_eigenvalue_tolerance"] = 0.02
        self.write("large_qubit/section3_policies.json", policy)
        with self.assertRaisesRegex(ValueError, "policy was changed"):
            verify_grid_amendment(self.root)

    def test_unrecorded_files_and_omitted_cumulative_costs_are_refused(self):
        self.write(f"{self.private}/unrecorded.json", {"hidden_scientific_result": True})
        with self.assertRaisesRegex(ValueError, "unrecorded files"):
            verify_grid_amendment(self.root)
        (self.root / self.private / "unrecorded.json").unlink()
        self.record["cumulative_failed_attempt_costs"]["SCF_geometries_attempted"] = 3
        self.write(GRID_AMENDMENT_PATH, self.record)
        with self.assertRaisesRegex(ValueError, "Cumulative failed-attempt costs"):
            verify_grid_amendment(self.root)

    def test_rewritten_artifact_pins_cannot_hide_a_different_instability(self):
        diagnostics = read_json(self.root / self.private / "failure_diagnostics.json")
        diagnostics["diagnostics"]["stability"]["channels"]["external_RHF_to_UHF"]["minimum_eigenvalue"] = 0.01
        self.write(f"{self.private}/failure_diagnostics.json", diagnostics)
        failure = read_json(self.root / self.failed / "failure.json")
        failure["diagnostics_sha256"] = sha256_file(self.root / self.private / "failure_diagnostics.json")
        self.write(f"{self.failed}/failure.json", failure)
        self.write(f"{self.failed}/failure.sha256", sha256_file(self.root / self.failed / "failure.json") + "\n")
        self.repin()
        with self.assertRaisesRegex(ValueError, "external-instability evidence"):
            verify_grid_amendment(self.root)


if __name__ == "__main__":
    unittest.main()
