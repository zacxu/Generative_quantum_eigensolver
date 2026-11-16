import copy
import json
import subprocess
import sys
import unittest

from large_qubit.common import digest
from large_qubit.definitions import (GEOMETRY_IDS, PURPOSES, TRAINING_PURPOSES,
                                     build_definition, build_seed_manifest, geometry,
                                     namespace_for, require_geometry_access,
                                     validate_definition)
from large_qubit.protocol import FITS, STREAMS, seed_for, training_conditions


class PhysicalDefinitionTest(unittest.TestCase):
    def test_six_geometries_have_exact_coordinates_and_partition_boundaries(self):
        expected = {
            "CO2_T01": ("training", 1.00), "CO2_T02": ("training", 1.20),
            "CO2_T03": ("training", 1.32), "CO2_V01": ("validation", 1.25),
            "CO2_I01": ("final_test", 1.10), "CO2_I02": ("final_test", 1.30),
        }
        for alias, (role, distance) in expected.items():
            with self.subTest(alias=alias):
                data = geometry(alias)
                self.assertEqual(data["partition"], role)
                self.assertEqual(data["bond_length_angstrom"], distance)
                self.assertEqual(data["coordinates_angstrom"],
                                 [[0.0, 0.0, -distance], [0.0, 0.0, 0.0], [0.0, 0.0, distance]])
                self.assertEqual(data["elements"], ["O", "C", "O"])
                self.assertEqual(data["coordinate_unit"], "angstrom")
        partitions = build_definition()["partitions"]
        self.assertEqual([len(partitions[r]) for r in ("training", "validation", "final_test")], [3, 1, 2])
        self.assertEqual(len(set(sum(partitions.values(), []))), 6)

    def test_preproduction_grid_amendment_preserves_failed_identity_and_thresholds(self):
        definition = build_definition()
        amendment = definition["grid_amendment"]
        self.assertEqual(definition["definition_id"], "co2_physical_definition_v3")
        self.assertEqual(amendment["previous_definition_id"], "co2_physical_definition_v2")
        self.assertEqual(amendment["old_bond_length_angstrom"], 1.35)
        self.assertEqual(amendment["new_bond_length_angstrom"], 1.32)
        self.assertEqual(amendment["evidence_path"], "large_qubit/amendments/section3_grid_v3.json")
        self.assertEqual(amendment["prior_amendment_path"], "large_qubit/amendments/section3_grid_v2.json")
        self.assertFalse(amendment["reference_or_candidate_energies_used_for_decision"])
        self.assertFalse(amendment["final_targets_changed"])
        self.assertFalse(amendment["SCF_or_stability_thresholds_changed"])
        self.assertEqual(definition["scf_policy"]["stability_minimum_eigenvalue_tolerance"], -1e-7)
        self.assertEqual(geometry("CO2_I01")["bond_length_angstrom"], 1.10)
        self.assertEqual(geometry("CO2_I02")["bond_length_angstrom"], 1.30)

    def test_full_space_population_and_reference_are_explicit(self):
        definition = build_definition()
        problem = definition["problem"]
        self.assertEqual((problem["electrons"], problem["spatial_orbitals"], problem["qubits"]), (22, 15, 30))
        self.assertEqual((problem["n_alpha"], problem["n_beta"]), (11, 11))
        self.assertEqual((problem["frozen_core_orbitals"], problem["tapered_qubits"]), (0, 0))
        occupations = definition["occupation_convention"]
        self.assertEqual(occupations["fixed_sector_dimension"], 1863225)
        self.assertEqual(sum(occupations["hf_occupations"][::2]), 11)
        self.assertEqual(sum(occupations["hf_occupations"][1::2]), 11)
        self.assertEqual(occupations["endianness"], "wire_0_most_significant_bit")
        self.assertFalse(occupations["reshape_interleaved_state_into_alpha_beta_array"])
        self.assertFalse(definition["reference_policy"]["total_spin_purity_asserted_from_fixed_spin_projection"])
        self.assertFalse(definition["reference_policy"]["values_present"])
        self.assertFalse(definition["execution_ready"])
        self.assertEqual(definition["status"], "frozen_physical_specification_realization_recorded_separately")
        self.assertEqual(definition["mo_convention"]["numerical_alignment_validation"],
                         "required_section_3_realization_bound_separately")
        self.assertFalse(definition["physical_realization_binding_policy"]["numeric_results_in_this_definition"])
        self.assertEqual(definition["scf_policy"]["choose_solution"],
                         "single_deterministic_minao_RHF_per_geometry_stop_if_unstable_no_reference_label_selection")

    def test_roundtrip_is_strict_and_builders_return_detached_data(self):
        definition = build_definition()
        self.assertIs(validate_definition(definition), definition)
        self.assertEqual(validate_definition(json.loads(json.dumps(definition))), definition)
        definition["geometries"]["CO2_T01"]["coordinates_angstrom"][0][2] = 100.0
        self.assertEqual(geometry("CO2_T01")["coordinates_angstrom"][0][2], -1.0)
        returned = geometry("CO2_T02")
        returned["elements"][0] = "N"
        self.assertEqual(geometry("CO2_T02")["elements"][0], "O")

    def test_amendments_bool_confusion_nonfinite_and_non_json_types_fail(self):
        mutations = [
            ("problem", "electrons", 20), ("problem", "charge", False),
            ("problem", "full_basis", 1), ("problem", "qubits", 30.0),
            ("reference_policy", "values_present", True),
            ("mo_convention", "energy_degeneracy_tolerance_hartree", float("nan")),
            ("mo_convention", "energy_degeneracy_tolerance_hartree", float("inf")),
        ]
        for section, field, value in mutations:
            with self.subTest(section=section, field=field, value=value):
                definition = build_definition()
                definition[section][field] = value
                with self.assertRaises(ValueError):
                    validate_definition(definition)
        for value in (None, [], {"extra": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_definition(value)
        definition = build_definition()
        definition["geometries"]["CO2_T01"]["elements"] = ("O", "C", "O")
        with self.assertRaisesRegex(ValueError, "plain JSON"):
            validate_definition(definition)
        definition = build_definition()
        definition["cycle"] = definition
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_definition(definition)

    def test_partition_access_matrix_and_heldout_metadata_only(self):
        for alias in GEOMETRY_IDS:
            role = geometry(alias)["partition"]
            for purpose in PURPOSES:
                allowed = purpose == "physical_validation" or (
                    role == "training" and purpose in TRAINING_PURPOSES) or purpose == role
                with self.subTest(alias=alias, purpose=purpose):
                    if allowed:
                        self.assertEqual(require_geometry_access(alias, purpose), role)
                    else:
                        with self.assertRaises(ValueError):
                            geometry(alias, purpose)
            # These metadata results never expose energy/reference/cache/model payloads.
            self.assertEqual(set(geometry(alias)), {"geometry_id", "partition", "bond_length_angstrom",
                                                   "elements", "coordinates_angstrom", "coordinate_unit",
                                                   "definition_status"})
        forbidden = ("reference", "reference_energy", "scoring_cache", "model_loading", "fitting")
        for purpose in forbidden:
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                geometry("CO2_I01", purpose)

    def test_alias_and_purpose_are_exact_not_coerced_or_path_interpreted(self):
        for alias in (None, True, 1, "CO2_I03", "co2_i01", "../CO2_T01", "CO2_T01 "):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                geometry(alias)
        for purpose in (None, True, " training", "../training"):
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                namespace_for("CO2_T01", purpose)

    def test_namespaces_separate_diagnostics_fitting_pilot_and_partitions(self):
        paths = [namespace_for("CO2_T01", p) for p in ("training", "diagnostics", "acquisition", "representation")]
        self.assertEqual(len(set(paths)), 4)
        self.assertNotEqual(namespace_for("CO2_I01"), namespace_for("CO2_I01", "final_test"))
        with self.assertRaises(ValueError):
            namespace_for("CO2_V01", "training")
        policy = build_definition()["access_policy"]
        self.assertFalse(policy["namespaces"]["cross_partition_cache_reuse"])
        self.assertFalse(policy["metadata_guard_is_execution_authorization"])
        self.assertIn("scoring_only", policy["namespaces"]["references"])
        self.assertIn("scoring_only", policy["namespaces"]["energy_caches"])

    def test_seed_manifest_reuses_registered_derivation_and_pairs_gpt_declarations(self):
        manifest = build_seed_manifest()
        fits = training_conditions()
        self.assertEqual(manifest["shared_streams"], {s: seed_for(s) for s in STREAMS})
        for alias in ("CO2_T01", "CO2_T02", "CO2_T03"):
            self.assertEqual(manifest["acquisition"][alias]["seed"], seed_for("acquisition", group=alias))
        for name in FITS:
            self.assertEqual(manifest["adapter_initialization"][name]["seed"], fits[name]["adapter_initialization_seed"])
        self.assertEqual(manifest["adapter_initialization"][FITS[0]], manifest["adapter_initialization"][FITS[1]])
        paired = manifest["gpt_pair"]["shared_numerical_declaration"]
        self.assertEqual(digest(paired), manifest["gpt_pair"]["shared_numerical_declaration_sha256"])
        self.assertEqual(paired["learning_rate"], 2e-4)
        self.assertIsNone(paired["optimizer_scheduler_clipping"])
        for alias, entry in manifest["vqe"].items():
            self.assertEqual(entry["seed"], seed_for("vqe", group=alias))

    def test_definition_import_has_no_scientific_or_model_side_effects(self):
        script = ('import sys; from large_qubit.definitions import build_definition; '
                  'build_definition(); '
                  'assert not ({"numpy", "scipy", "pyscf", "torch", "transformers", "pennylane"} & set(sys.modules))')
        subprocess.run([sys.executable, "-B", "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
