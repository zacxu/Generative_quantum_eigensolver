"""Section 3 physical definitions and partition guards; no energy or model code.

This freezes the physical specification, not numerical results or the final
experimental protocol. Section 3 SCF/orbital/integral realization is recorded in
separately bound evidence; references and evaluation still require Section 4.
"""

import copy
import math

from .common import canonical_bytes, digest, require
from .protocol import BENCHMARK_ID, FITS, METHODS, STREAMS, seed_for, training_conditions


DEFINITION_ID = "co2_physical_definition_v3"
GEOMETRY_IDS = ("CO2_T01", "CO2_T02", "CO2_T03", "CO2_V01", "CO2_I01", "CO2_I02")
TRAINING_PURPOSES = ("training", "diagnostics", "acquisition", "representation",
                     "weighting", "retrieval", "parent_selection")
PURPOSES = TRAINING_PURPOSES + ("validation", "final_test", "physical_validation")
_GRID = (
    ("CO2_T01", "training", 1.00), ("CO2_T02", "training", 1.20),
    ("CO2_T03", "training", 1.32), ("CO2_V01", "validation", 1.25),
    ("CO2_I01", "final_test", 1.10), ("CO2_I02", "final_test", 1.30),
)
_PURPOSE_NAMESPACE = {
    "training": "training_fitting", "weighting": "training_fitting",
    "retrieval": "training_fitting", "parent_selection": "training_fitting",
    "diagnostics": "training_diagnostics", "acquisition": "training_acquisition",
    "representation": "training_pilot", "validation": "validation",
    "final_test": "final_test", "physical_validation": "physical_metadata",
}


def _geometries():
    return {alias: {
        "geometry_id": alias, "partition": partition,
        "bond_length_angstrom": distance,
        "elements": ["O", "C", "O"],
        "coordinates_angstrom": [[0.0, 0.0, -distance], [0.0, 0.0, 0.0],
                                 [0.0, 0.0, distance]],
        "coordinate_unit": "angstrom",
        "definition_status": "fixed_grid_requires_passed_physical_realization",
    } for alias, partition, distance in _GRID}


def build_seed_manifest():
    """Resolve the existing benchmark-namespaced streams without RNG execution."""
    fits = training_conditions()
    paired = copy.deepcopy(fits[FITS[0]])
    for field in ("condition_id", "objective", "lambda_scale"):
        paired.pop(field)
    other = copy.deepcopy(fits[FITS[1]])
    for field in ("condition_id", "objective", "lambda_scale"):
        other.pop(field)
    require(canonical_bytes(paired) == canonical_bytes(other),
            "GPT numerical declarations or paired initialization/order differ")
    return {
        "pipeline_seed": 22,
        "benchmark_id": BENCHMARK_ID,
        "derivation": {
            "input": "canonical JSON array [benchmark_id,22,stream,group]",
            "encoding": "UTF-8; sorted keys; compact separators; ensure_ascii=true; no newline",
            "digest": "sha256", "bytes": "first 8, big-endian",
            "mask": "(1 << 63) - 1", "implementation": "large_qubit.protocol.seed_for",
        },
        "shared_streams": {stream: seed_for(stream) for stream in STREAMS},
        "acquisition": {alias: {"group": alias, "seed": seed_for("acquisition", group=alias)}
                        for alias, role, _ in _GRID if role == "training"},
        "adapter_initialization": {
            name: {"group": "gpt2_pair" if name in FITS[:2] else "qwen3_weighted",
                   "seed": fits[name]["adapter_initialization_seed"]} for name in FITS
        },
        "row_order": {"group": "shared", "seed": seed_for("row_order"),
                      "rule": "geometry_balanced_same_rows_same_order_for_all_fits"},
        "generation": {
            method: {alias: {"group": f"{method}:{alias}",
                             "seed": seed_for("generation", group=f"{method}:{alias}")}
                     for alias in GEOMETRY_IDS}
            for method in METHODS[:6]
        },
        "vqe": {alias: {"group": alias, "seed": seed_for("vqe", group=alias)}
                for alias in GEOMETRY_IDS},
        "gpt_pair": {
            "conditions": list(FITS[:2]), "initialization_group": "gpt2_pair",
            "shared_numerical_declaration": paired,
            "shared_numerical_declaration_sha256": digest(paired),
            "allowed_differences": ["condition_id", "objective", "lambda_scale"],
            "resolved_microbatch_optimizer_and_adapter_targets": "pending_before_fitting",
        },
        "generation_batch_and_rng_checkpoint_policy": "pending_executor_freeze_before_generation",
    }


def build_definition():
    """Return the complete immutable Section 3 specification as detached JSON data."""
    return {
        "schema_version": "large-qubit-physical-definition-v1",
        "definition_id": DEFINITION_ID, "benchmark_id": BENCHMARK_ID,
        "status": "frozen_physical_specification_realization_recorded_separately",
        "section": 3, "scientific_results_present": False, "execution_ready": False,
        "final_protocol_frozen": False,
        "problem": {
            "molecule": "CO2", "geometry": "linear_symmetric",
            "atom_order": ["O_left", "C", "O_right"],
            "atomic_numbers": [8, 6, 8], "charge": 0,
            "basis": "sto-3g", "full_basis": True, "cartesian": False,
            "spatial_orbitals": 15, "spin_orbitals": 30, "qubits": 30,
            "electrons": 22, "n_alpha": 11, "n_beta": 11,
            "frozen_core_orbitals": 0, "tapered_qubits": 0,
            "mapping": "jordan_wigner", "coordinate_unit": "angstrom",
            "molecular_axis": "laboratory_z", "origin": "carbon_nucleus",
        },
        "geometries": _geometries(),
        "partitions": {role: [alias for alias, actual, _ in _GRID if actual == role]
                       for role in ("training", "validation", "final_test")},
        "grid_amendment_policy": {
            "status": "fixed_definition_physical_realization_required_before_production",
            "permitted_reason": "numerical_or_physical_validation_failure_before_production",
            "required_record": ["new_version", "old_grid", "new_grid", "reason", "evidence",
                                "affected_bindings", "incurred_costs"],
            "final_performance_based_amendment": False,
            "overwrite_registration_or_suppress_negative_outcomes": False,
        },
        "grid_amendment": {
            "amendment_id": "training_T03_stable_RHF_v3",
            "previous_definition_id": "co2_physical_definition_v2",
            "current_definition_id": DEFINITION_ID,
            "geometry_id": "CO2_T03",
            "old_bond_length_angstrom": 1.35,
            "new_bond_length_angstrom": 1.32,
            "reason": "registered_RHF_to_UHF_stability_check_failed_before_production",
            "evidence_path": "large_qubit/amendments/section3_grid_v3.json",
            "prior_amendment_path": "large_qubit/amendments/section3_grid_v2.json",
            "preserved_failed_attempt": "large_qubit/physics/section3_v2",
            "preserved_specification": "large_qubit/archives/section3_physical_v2/snapshot.json",
            "reference_or_candidate_energies_used_for_decision": False,
            "final_targets_changed": False,
            "SCF_or_stability_thresholds_changed": False,
            "replacement_requires_all_registered_physical_checks": True,
        },
        "ao_convention": {
            "functions": "real_normalized_atom_centered_STO-3G_spherical",
            "atom_order": "O_left,C,O_right",
            "within_atom_order": "PySCF AO order; exact indexed labels recorded in every realization",
            "overlap": "real_symmetric_AO_overlap_S",
            "coefficient_layout": "AO_rows_spatial_MO_columns",
            "orthonormality": "C.T @ S @ C = identity",
            "required_realization_fields": ["provider_version", "basis_definition_sha256",
                                            "ao_labels", "ao_overlap_sha256", "mo_coefficients_sha256"],
        },
        "mo_convention": {
            "orbitals": "real_spatial_RHF; occupied_0_to_10_then_virtual_11_to_14",
            "alignment_anchor": "CO2_T02",
            "anchor_order": "energy_ascending_separately_occupied_and_virtual",
            "energy_degeneracy_tolerance_hartree": 1e-7,
            "degeneracy_grouping": "total_block_span_within_tolerance_no_transitive_chaining",
            "anchor_degenerate_basis": "ordered_AO_projection_pivots_S_metric_Gram_Schmidt",
            "AO_projection_threshold": 1e-10,
            "pivot_tie_break": "lowest_AO_index",
            "anchor_phase": "largest_absolute_AO_coefficient_positive_ties_lowest_AO_index",
            "coefficient_tie_tolerance": 1e-12,
            "target_assignment": "maximum_absolute_cross_AO_overlap_separately_occupied_and_virtual",
            "assignment_tie_break": "lexicographic_among_equal_optimal_assignments",
            "assignment_tie_tolerance": 1e-12,
            "minimum_alignment_singular_value": 1e-6,
            "S_orthonormality_maximum_absolute_error": 1e-8,
            "target_degenerate_alignment": "orthogonal_Procrustes_only_within_target_energy_degenerate_blocks",
            "cross_overlap": "C_anchor.T @ S_cross @ C_target",
            "target_phase": "positive_matched_anchor_overlap",
            "occupied_virtual_mixing": False,
            "integrals": "transform_all_one_and_two_electron_integrals_with_final_aligned_C",
            "two_electron_integral_layout": "full_four_index_chemists_notation_(pq|rs)",
            "held_out_reference_energy_feedback": False,
            "numerical_alignment_validation": "required_section_3_realization_bound_separately",
        },
        "occupation_convention": {
            "spatial_MO_indexing": "zero_based",
            "spin_orbitals": "alpha_even_beta_odd; alpha(p)=2*p; beta(p)=2*p+1",
            "qubit_wires": "wire_index_equals_spin_orbital_index",
            "basis_index": "sum(occupation[w] << (29-w) for w in range(30))",
            "endianness": "wire_0_most_significant_bit",
            "hf_occupations": [1] * 22 + [0] * 8,
            "hf_spatial_occupations": [2] * 11 + [0] * 4,
            "fixed_sector_dimension": math.comb(15, 11) ** 2,
            "alpha_beta_conversion": "explicit_determinant_permutation_and_fermionic_reordering_sign",
            "reshape_interleaved_state_into_alpha_beta_array": False,
            "determinant_ordering_validation": "pending_section_4",
        },
        "scf_policy": {
            "reference_determinant": "restricted_closed_shell_Hartree_Fock",
            "initial_guess": "minao",
            "convergence_energy_hartree": 1e-12,
            "convergence_gradient": 1e-9,
            "maximum_cycles": 200,
            "post_SCF_gradient_norm_maximum": 1e-8,
            "stability_channels": ["internal_RHF", "external_RHF_to_UHF", "external_real_to_complex"],
            "stability_Hessian_dimension_per_channel": 44,
            "stability_minimum_eigenvalue_tolerance": -1e-7,
            "choose_solution": "single_deterministic_minao_RHF_per_geometry_stop_if_unstable_no_reference_label_selection",
            "required_checks": ["SCF_converged", "internal_and_external_stability",
                                "electron_population", "AO_MO_orthonormality",
                                "record_chosen_solution_and_all_retries"],
            "unresolved_or_unstable_solution": "stop_and_record_before_production",
            "actual_solutions_and_stability_results": "required_section_3_realization_bound_separately",
        },
        "reference_policy": {
            "endpoint": "lowest_energy_in_fixed_Nalpha_11_Nbeta_11_sector",
            "hamiltonian": "same_full_basis_aligned_integral_Hamiltonian_as_candidates",
            "reported_energy": "electronic_expectation_plus_nuclear_repulsion_in_hartree",
            "total_spin_purity_asserted_from_fixed_spin_projection": False,
            "interpretation": "finite_basis_fixed_population_reference_not_experimental_ground_energy",
            "solver_and_residual_validation": "pending_section_4",
            "values_present": False,
        },
        "access_policy": {
            "training_only_purposes": list(TRAINING_PURPOSES),
            "validation_only_purposes": ["validation"],
            "final_test_only_purposes": ["final_test"],
            "all_geometry_metadata_purposes": ["physical_validation"],
            "physical_validation_grants": ["geometry_and_physical_conventions_only"],
            "physical_validation_never_grants": ["reference_energy_access", "scoring_cache_access",
                                                "model_loading", "model_selection", "training_labels"],
            "model_inputs_never_contain": ["reference_energies", "target_energy_feedback",
                                          "validation_or_final_test_scoring_caches"],
            "diagnostics_excluded_from": ["fitting", "weighting", "parent_selection", "retrieval"],
            "final_test_scoring": "after_final_experimental_manifest_freeze_only",
            "metadata_guard_is_execution_authorization": False,
            "namespaces": {
                "metadata_or_role_outputs": f"large_qubit/{BENCHMARK_ID}/<purpose_namespace>/<geometry_id>",
                "model_inputs": f"large_qubit/{BENCHMARK_ID}/model_inputs/<partition>/<geometry_id>",
                "references": f"large_qubit/scoring_only/{BENCHMARK_ID}/references/<partition>/<geometry_id>",
                "energy_caches": f"large_qubit/scoring_only/{BENCHMARK_ID}/caches/<partition>/<geometry_id>",
                "purpose_mapping": copy.deepcopy(_PURPOSE_NAMESPACE),
                "cross_partition_cache_reuse": False,
                "cross_method_same_partition_identical_query_reuse": "allowed_with_each_logical_request_charged",
            },
        },
        "endpoint_policy": {
            "primary": "best_verified_energy_error_at_B_1024_including_initial_HF",
            "target_reporting": "each_of_CO2_I01_CO2_I02_and_their_unweighted_mean",
            "also_report": ["budget_curves", "improvement_over_HF", "validity", "duplicates",
                            "circuit_resources", "measured_costs"],
            "reference_or_cache_use_for_development_selection": False,
        },
        "seeds": build_seed_manifest(),
        "physical_realization_binding_policy": {
            "section": 3,
            "artifact_binding": "separate_section_3_registration_and_validation_evidence",
            "required_checks": ["all_geometry_SCF_solutions_and_stability", "AO_MO_alignment",
                                "basis_and_population_counts", "aligned_one_and_two_electron_integrals"],
            "numeric_results_in_this_definition": False,
            "production_gate": "all_required_realization_checks_pass_and_match_registered_artifacts",
        },
        "pending_numerical_bindings": ["determinant_ordering_and_signs",
                                       "iterative_references_and_residuals", "evaluator_correctness",
                                       "measured_runtime_memory_and_VQE_feasibility"],
    }


def _require_plain_json(value, active=None):
    """Reject non-JSON types, cycles and nonfinite numbers before comparison."""
    active = set() if active is None else active
    kind = type(value)
    require(kind in (dict, list, str, int, float, bool, type(None)),
            "Physical definition must contain only plain JSON values")
    if kind is float:
        require(math.isfinite(value), "Physical definition contains nonfinite numbers")
    if kind not in (dict, list):
        return
    require(id(value) not in active, "Physical definition contains a cycle")
    active.add(id(value))
    try:
        values = value.values() if kind is dict else value
        if kind is dict:
            require(all(type(key) is str for key in value), "Definition object keys must be strings")
        for child in values:
            _require_plain_json(child, active)
    finally:
        active.remove(id(value))


def validate_definition(value):
    """Fail closed on any amended identity, bool/int confusion or fabricated result."""
    _require_plain_json(value)
    require(canonical_bytes(value) == canonical_bytes(build_definition()),
            "Physical definition differs; create a reviewed versioned amendment")
    return value


def require_geometry_access(alias, purpose="physical_validation"):
    """Check partition access for physical metadata; never authorize execution."""
    require(type(alias) is str and alias in GEOMETRY_IDS, "Unknown CO2 geometry alias")
    require(type(purpose) is str and purpose in PURPOSES, "Unknown geometry access purpose")
    role = _geometries()[alias]["partition"]
    expected = ("training" if purpose in TRAINING_PURPOSES else purpose)
    require(purpose == "physical_validation" or role == expected,
            f"Geometry {alias} in {role} cannot be used for {purpose}")
    return role


def geometry(alias, purpose="physical_validation"):
    """Return only detached coordinates/conventions permitted by a role guard."""
    require_geometry_access(alias, purpose)
    return _geometries()[alias]


def namespace_for(alias, purpose="physical_validation"):
    """Return an isolated portable namespace after validating alias and purpose."""
    require_geometry_access(alias, purpose)
    return f"large_qubit/{BENCHMARK_ID}/{_PURPOSE_NAMESPACE[purpose]}/{alias}"
