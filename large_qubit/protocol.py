"""Sections 1-6 decisions and Section7 researcher-run fit implementation.

The training-only evaluator and representation/VQE gates passed; remaining
partition references, training integration and executor identities remain unresolved.
The original Section 6 NO-GO and its full cost remain preserved separately.
No old Phase7 configuration or adapter is an executable CO2 configuration.
"""

import copy
import hashlib

from .common import canonical_bytes, require


BENCHMARK_ID = "gqe-co2-sto3g-30q-v1"
MODEL_REVISIONS = {
    "gpt2_xl": "15ea56dee5df4983c59b2538573817e1667135e2",
    "qwen3_4b": "1cfa9a7208912126459214e8b04321603b3df60c",
}
METHODS = (
    "gpt2_xl_frozen", "gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora",
    "qwen3_4b_frozen", "qwen3_4b_weighted_lora", "uniform_random",
    "training_circuit_retrieval", "conventional_vqe",
)
FITS = (METHODS[1], METHODS[2], METHODS[4])
PENDING = (
    "remaining_partition_references",
    "resolved_optimizer_and_numerical_settings", "three_fresh_final_adapters",
    "compact_validation_panel", "reproducible_dependency_environment",
    "resumable_tmux_executor_and_analysis", "final_experiment_freeze",
)
STREAMS = ("acquisition", "adapter_initialization", "row_order", "generation", "vqe")


def seed_for(stream, *, group="shared"):
    """Stable benchmark-namespaced streams, independent of Python hash randomization.

    Both GPT fits use the same initialization/order group. Generation can use a
    condition/target group once those execution identities are frozen.
    """
    require(stream in STREAMS and isinstance(group, str) and group.strip(),
            "Unknown stream or empty group")
    data = canonical_bytes([BENCHMARK_ID, 22, stream, group])
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") & ((1 << 63) - 1)


def training_conditions():
    common = {
        "initialization": "pinned_pretrained_base_plus_fresh_lora",
        "resume_from_historical_adapter": False,
        "pipeline_seed": 22, "corpus_role": "new_shared_co2_fitting_release",
        "passes": 5, "checkpoint_selection": "final",
        "learning_rate": 0.0002, "lora_rank": 16, "lora_alpha": 32,
        "lora_dropout": 0.05, "precision": "bf16", "effective_batch_size": 16,
        "row_order": "geometry_balanced",
        "likelihood": "complete_answer_grammar_masked_sequence_nll",
        "weight_application_count": 1,
        "microbatch_size": None, "gradient_accumulation": None,
        "optimizer_scheduler_clipping": None, "adapter_targets": None,
        "configuration_status": "declaration_not_executable_training_config",
    }
    result = {}
    for name, model, mode, scale in (
        (FITS[0], "gpt2_xl", "weighted", 0.25),
        (FITS[1], "gpt2_xl", "uniform", None),
        (FITS[2], "qwen3_4b", "weighted", 0.10),
    ):
        group = "gpt2_pair" if model == "gpt2_xl" else "qwen3_weighted"
        result[name] = dict(copy.deepcopy(common), condition_id=name, model_key=model,
                            model_revision=MODEL_REVISIONS[model], objective=mode,
                            lambda_scale=scale,
                            adapter_initialization_seed=seed_for("adapter_initialization", group=group),
                            row_order_seed=seed_for("row_order"))
    return result


def build_study():
    from .definitions import build_definition
    from .common import digest
    physical_definition = build_definition()
    return {
        "schema_version": "large-qubit-decisions-v2", "benchmark_id": BENCHMARK_ID,
        "status": "sections_1_6_complete_section7_implemented_training_pending", "final_protocol_frozen": False,
        "execution_ready": False,
        "section4_validation": {
            "manifest": "large_qubit/validation/section4_v1/manifest.json",
            "manifest_sha256": "232fbe1c86075244a17b8a194d6ae7e35d84ab8126e69ad528436c2c8ed977e0",
            "correctness_passed": True, "reference_converged": True,
            "reference_geometry": "CO2_T02", "section_4_complete": True,
            "feasibility_gate": "go_local_cpu_under_declared_window",
            "heldout_references_computed": False, "full_pipeline_ready": False,
        },
        "section5_validation": {
            "manifest": "large_qubit/validation/section5_v1/manifest.json",
            "manifest_sha256": "26d65cc10eb622d28585cb6503e23033004c10acd6496513b82b0d46ecec3155",
            "correctness_passed": True, "section_5_complete": True,
            "feasibility_gate": "go", "selected_length": 32,
            "catalogue": {
                "path": "large_qubit/catalogues/co2_screened_v1.json",
                "file_sha256": "fbeba16b348f4d150448548fc00fa12e68f09024e8f1055c3353abce39132221",
                "catalogue_sha256": "0a5fc7b1e21497e9fc4283a2e40829de8789f1e9f17e5f3726b878a549c4121c",
                "templates": 64, "discrete_ids_including_identity": 513,
            },
            "preparation": {
                "path": "large_qubit/evidence/sections_1_5_v1/preparation.json",
                "file_sha256": "52e1bd4fc0fb33c42a8743266910ef868e1e60ee6ba0ad7d223c43e72a0a0908",
                "complete_answer_context_certificates": 24,
            },
            "policy": {
                "path": "large_qubit/section5_policy.json",
                "file_sha256": "e481d50b33cd0e3ed8f4519c6985a81212bf4b0b0d12ec90fa52401f8a086c79",
                "policy_sha256": "c84d10ae1975ca0dc91715d4e29e96b0d06425407d621a2f4d568c33da5e0cc2",
            },
            "compact_vqe": {
                "family": "screened_compact_fermionic_excitation",
                "parameters": 24, "layout": "catalogue_vqe_template_indices_in_order",
                "optimizer": "five_point_excitation_coordinate_minimization",
                "initializations": ["HF_zero", "fixed_training_mean_perturbative_vector"],
                "initialization_binding": "preparation_private_screening_record",
                "pilot_requests_per_restart": [145, 145],
                "main_post_HF_request_caps_per_restart": [511, 512],
                "main_actual_post_HF_requests": 1022,
                "main_unused_tail_requests": 1,
                "duplicate_and_probe_queries_charged": True,
                "convergence_claimed": False,
            },
            "ordinary_requests": 909, "pilot_total_including_section4": 945,
            "remaining_pilot_allowance": 79, "separate_integrity_hamiltonian_actions": 9,
            "heldout_energy_or_reference_access": False, "full_pipeline_ready": False,
        },
        "section6_failed_attempt_v1": {
            "historical_snapshot": True, "superseded_by": "section6_validation",
            "manifest": "large_qubit/validation/section6_v1/manifest.json",
            "manifest_sha256": "cdf23f3493d1c583cbf4e46126412416ea0afe295247f699f1a4305845693737",
            "evidence_authenticated": True, "section_6_complete": False,
            "feasibility_gate": "no_go", "stop_reason": "target",
            "release": None, "released_rows_per_geometry": None,
            "retained_fitting_rows_per_geometry": 2048,
            "retained_diagnostics_per_geometry": 128,
            "ordinary_requests": 6538, "production_request_ceiling": 8192,
            "remaining_production_allowance": 1654,
            "failed_requests": 0, "abandoned_requests": 0,
            "charged_nonretained_fitting_requests": 7,
            "separate_integrity_hamiltonian_actions": 0,
            "preparation": {
                "path": "large_qubit/evidence/sections_1_6_v1/preparation.json",
                "file_sha256": "e70140bc847dca2d8e5fb138188fd93f1e0ae61a9709e6d51a5cd2c1ec0ccbea",
            },
            "policy": {
                "path": "large_qubit/section6_policy.json",
                "file_sha256": "486cbb228c40db3a68f7bb07f0edfd5146816b0cada305af8fde606e8407c008",
                "policy_sha256": "d1573c90682486ec7c28a32b597c4587d9cdff2e35923bf6b3de5f278fd130d0",
            },
            "failed_quality_check": {
                "geometry_id": "CO2_T03", "lambda_scale": 0.10,
                "ess_fraction": 0.045462919181968986,
                "minimum_ess_fraction": 0.05,
                "maximum_row_weight": 0.05793925698270687,
                "maximum_allowed_row_weight": 0.05,
            },
            "fallback_permitted": False,
            "next_action": "review_explicit_prospective_acquisition_amendment_before_release_or_fitting",
            "diagnostic_energies_used_for_quality": False,
            "heldout_energy_or_reference_access": False,
            "model_training_executed": False, "full_pipeline_ready": False,
        },
        "section6_validation": {
            "manifest": "large_qubit/validation/section6_v2/manifest.json",
            "manifest_sha256": "4724732634b8740560de28bf126834a939a1ac13af44448a23e435d0a0cd4fbb",
            "evidence_authenticated": True, "section_6_complete": True,
            "feasibility_gate": "go", "stop_reason": "target",
            "amended_confirmation": True,
            "amendment": {
                "path": "large_qubit/amendments/section6_acquisition_v2.json",
                "file_sha256": "611592dad5d900a536939ce4435f5a1982174d2e1a94df1f947c6a95357cc130",
            },
            "release": {
                "path": "large_qubit/releases/co2_corpus_v2/manifest.json",
                "manifest_sha256": "323fe5fa6d4830339aa9b14dc546364fe5d24765745f08c421e3ca562e2d3e2a",
                "fitting_rows_per_geometry": 2048, "diagnostics_per_geometry": 128,
                "identical_fitting_rows_for_all_three_fits": True,
                "retrieval_uses_fitting_rows_only": True,
                "prior_attempt_labels_imported": False,
            },
            "preparation": {
                "path": "large_qubit/evidence/sections_1_6_v2/preparation.json",
                "file_sha256": "12cf3780db66b74aeae55701f3a56ec40e696ddce50d51212e740004baeb15e2",
            },
            "policy": {
                "path": "large_qubit/section6_v2_policy.json",
                "file_sha256": "c7b7110d52688e0ae60c4738fd15cc247552d8cc448f1f09935c4b88b36cce52",
                "policy_sha256": "b8a3766565d649431aec5aff5604bb2eedbf54ee1d69c78142185b35ffb25441",
            },
            "fresh_ordinary_requests": 6552, "fresh_request_ceiling": 8192,
            "prior_failed_ordinary_requests": 6538,
            "cumulative_ordinary_requests": 13090,
            "maximum_cumulative_ordinary_requests": 14730,
            "remaining_confirmation_allowance": 1640,
            "failed_requests": 0, "abandoned_requests": 0,
            "separate_integrity_hamiltonian_actions": 12,
            "each_corpus_consumer_charged_full_prior_plus_fresh_and_verification_cost": True,
            "fixed_lambdas_and_quality_thresholds_unchanged": True,
            "diagnostic_energies_used_for_quality": False,
            "heldout_energy_or_reference_access": False,
            "model_training_executed": False, "full_pipeline_ready": False,
        },
        "problem": {"molecule": "CO2", "geometry": "linear_symmetric",
                    "basis": "sto-3g", "charge": 0, "electrons": 22,
                    "spatial_orbitals": 15, "qubits": 30, "n_alpha": 11, "n_beta": 11,
                    "mapping": "jordan_wigner", "frozen_core_orbitals": 0, "tapered_qubits": 0},
        "model_revisions": copy.deepcopy(MODEL_REVISIONS),
        "training_conditions": training_conditions(),
        "section7_implementation": {
            "status": "implemented_prepared_workflow_researcher_training_pending",
            "policy": "large_qubit/section7_policy.json",
            "preparation": "large_qubit/evidence/sections_1_7_v1/preparation.json",
            "token_cache": "large_qubit/training_data/section7_v1",
            "gpu_allocation": "three_H200_one_per_fit",
            "gpu_batch_configuration": "measured_and_frozen_by_researcher_before_fitting",
            "scientific_training_complete": False,
            "final_adapters_produced": False,
            "model_training_executed_during_implementation": False,
            "training_conditions_declaration_preserved_for_physical_lineage": True,
        },
        "required_methods": list(METHODS), "pipeline_seeds": [22],
        "targets": ["CO2_I01", "CO2_I02"],
        "target_definition_status": "fixed_section3_grid_realization_authenticated_separately",
        "physical_definition": {"path": "large_qubit/physical_definition.json",
                                "definition_sha256": digest(physical_definition),
                                "realization_manifest": "large_qubit/physics/section3_v3/manifest.json",
                                "final_experiment_freeze": False},
        "generation": {"temperature": 0.7, "constrained_complete_answer": True,
                       "qwen_enable_thinking": False, "target_energy_feedback": False,
                       "max_attempts_per_main_trajectory": 2048,
                       "context_tokens": 1024, "context_and_catalogue_preflight_pending": False,
                       "selected_circuit_length": 32,
                       "catalogue_path": "large_qubit/catalogues/co2_screened_v1.json",
                       "catalogue_sha256": "0a5fc7b1e21497e9fc4283a2e40829de8789f1e9f17e5f3726b878a549c4121c"},
        "budget": {"requests_per_main_trajectory": 1024, "hf_included": True,
                   "post_hf_requests_per_main_trajectory": 1023,
                   "duplicates_charged": True, "cache_hits_charged": True,
                   "main_trajectories": 16, "main_logical_requests": 16384},
        "research_questions": {
            "RQ1": {"pairs": [[METHODS[0], METHODS[1]], [METHODS[3], METHODS[4]]],
                    "claim": "descriptive_frozen_vs_weighted_per_backbone"},
            "RQ2": {"learned": list(FITS), "baselines": list(METHODS[5:]),
                    "outcomes": ["verified_energy_error", "logical_requests", "marginal_and_total_cost"],
                    "matched_family_optimizer_claim": False, "amortized_transfer_advantage_claim": False},
            "RQ3": {"targets": ["CO2_I01", "CO2_I02"], "claim": "two_fixed_unseen_interpolation_geometries"},
            "RQ4": {"pair": [METHODS[1], METHODS[2]], "claim": "gpt2_weighting_only",
                    "qwen_weighting_specific_claim": False},
        },
        "reuse_policy": {
            "base_models": "authenticate_existing_pinned_weights_and_tokenizers_locally",
            "historical_lih": "separately_labelled_pilot_and_development_evidence",
            "new_labels_and_adapters_required": True,
            "old_adapter_initialization": "deferred_separate_controlled_condition",
            "historical_cost_namespace": "historical_lih",
            "new_cost_namespace": BENCHMARK_ID,
            "new_code_and_outputs": "large_qubit/",
            "old_phase7_commands": "historical_lih_only",
        },
        "claim_limits": ["one_seed_descriptive_case_study", "no_seed_robustness_inference",
                         "no_quantum_advantage_claim", "no_classical_hardness_claim",
                         "no_gqe_superiority_from_width", "no_claim_five_passes_converged",
                         "no_qwen_weighting_specific_conclusion", "no_online_energy_feedback_training"],
        "deferred": ["qwen_uniform", "smollm3", "phi4", "scratch", "additional_seeds",
                     "extrapolation", "large_tuning_sweeps", "publication_extension"],
        "optional_40_qubit_gate": "required_results_and_submission_artifacts_complete_with_measured_headroom",
        "pending_bindings": {name: None for name in PENDING},
    }


def validate_study(study):
    """Fail closed on accidental changes, extra conditions, bool/int confusion.

    A scientific amendment needs a new version and review, not an ad hoc edit
    to this registration. Pending identities cannot be filled with LiH hashes.
    """
    require(canonical_bytes(study) == canonical_bytes(build_study()),
            "Study differs from the revised registered decisions; create a reviewed versioned amendment")
    return study


def planned_jobs(study):
    validate_study(study)
    return [{"job_id": f"{BENCHMARK_ID}_{method}_{target}_s{seed}",
             "method": method, "target": target, "pipeline_seed": seed,
             "logical_request_cap": 1024, "includes_hf": True,
             "state": "not_executable_pending_final_freeze"}
            for method in METHODS for target in study["targets"] for seed in study["pipeline_seeds"]]


def require_execution_ready(study):
    validate_study(study)
    raise ValueError("CO2 execution is not ready: remaining partition references, numerical settings, "
                     "fresh adapters, validation, executor and final freeze are pending")
