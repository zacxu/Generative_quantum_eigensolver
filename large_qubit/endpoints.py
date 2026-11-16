"""Section 3 scoring definitions and strict reduction of future frozen evidence.

No evaluator, reference solver, model input reader or experiment runner lives
here. Artifact hashes must come from a trusted evidence manifest bound to the
experimental freeze; a
caller-supplied success flag is never treated as authentication. SHA256 matching
establishes that binding only, not an external signature or scientific truth.
"""

import copy
import math
import re

from .common import canonical_bytes, digest, require
from .definitions import geometry
from .protocol import BENCHMARK_ID, METHODS


TARGETS = ("CO2_I01", "CO2_I02")
ENERGY_TOLERANCE = 1e-8
SYMMETRY_TOLERANCE = 1e-10
BINDING_KEYS = (
    "benchmark_id", "target_id", "method", "pipeline_seed", "hamiltonian_sha256",
    "orbital_alignment_sha256", "catalogue_sha256", "backend_sha256", "precision",
    "final_freeze_sha256", "final_protocol_frozen",
)
PHYSICS_KEYS = ("benchmark_id", "target_id", "hamiltonian_sha256", "orbital_alignment_sha256")


def outcome_spec():
    return {
        "schema_version": "large-qubit-outcome-spec-v1", "benchmark_id": BENCHMARK_ID,
        "targets": list(TARGETS), "pipeline_seed": 22,
        "budget": 1024, "hf_included": True, "duplicates_charged": True,
        "cache_hits_charged": True, "request_ordinals": "contiguous_one_based_HF_first",
        "primary": "lowest_successfully_scored_energy_with_independently_verified_winner_minus_reference",
        "energy_error": "signed_energy_minus_same_Hamiltonian_fixed_population_reference",
        "below_reference_refusal_tolerance_hartree": ENERGY_TOLERANCE,
        "small_negative_errors": "retain_signed_value_without_absolute_value_or_clipping",
        "early_terminal_trajectory": "carry_verified_incumbent_to_B1024_record_actual_requests_only",
        "mean": "unweighted_two_fixed_target_mean_only_when_both_trajectories_succeed",
        "failed_target": "retain_failure_and_incumbent_if_available_no_success_only_mean",
        "required_reports": ["per_target_results", "budget_curves", "HF_improvement", "validity",
                             "duplicates", "circuit_resources", "measured_costs"],
        "artifact_trust": "canonical_SHA256_pins_from_trusted_evidence_manifest_bound_to_final_freeze",
        "derived_energy_arithmetic": "finite_signed_differences_and_overflow_safe_two_target_mean",
        "aggregation_trust": "both_summary_SHA256_pins_required_from_trusted_evidence_manifest",
        "numerical_evaluator_and_full_resource_analysis": "pending_later_sections",
    }


def model_geometry_payload(alias):
    """Build structural model input only, without reading scoring/reference data."""
    data = geometry(alias, "physical_validation")
    return {"benchmark_id": BENCHMARK_ID, "geometry_id": alias,
            "molecule": "CO2", "elements": data["elements"],
            "coordinates_angstrom": data["coordinates_angstrom"],
            "bond_length_angstrom": data["bond_length_angstrom"],
            "coordinate_unit": "angstrom", "basis": "sto-3g", "charge": 0,
            "electrons": 22, "spatial_orbitals": 15, "qubits": 30,
            "n_alpha": 11, "n_beta": 11, "mapping": "jordan_wigner"}


def _keys(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), f"Invalid {name} fields")


def _number(value, name):
    require(type(value) in (int, float), f"{name} must be finite numeric data")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f"{name} must be finite numeric data")
    return value


def _hash(value, name):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{name} must be a lowercase SHA256 digest")


def _text(value, name):
    require(type(value) is str and bool(value.strip()), f"{name} must be nonempty text")


def _bindings(value):
    _keys(value, BINDING_KEYS, "caller-scoped bindings")
    require(value["benchmark_id"] == BENCHMARK_ID, "Wrong benchmark binding")
    require(value["target_id"] in TARGETS, "Endpoint requires a fixed final-test target")
    require(type(value["method"]) is str and value["method"] in METHODS, "Unknown method binding")
    require(type(value["pipeline_seed"]) is int and value["pipeline_seed"] == 22, "Wrong pipeline seed")
    require(value["final_protocol_frozen"] is True, "Final experimental freeze is required")
    require(value["precision"] in ("complex128", "float64_validated_against_complex128"),
            "Unsupported or unvalidated numeric precision")
    for name in BINDING_KEYS:
        if name.endswith("_sha256"):
            _hash(value[name], name)


def _authenticate(payload, expected_hash, name):
    _hash(expected_hash, f"expected {name} hash")
    try:
        actual = digest(payload)
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError(f"{name} must be finite canonical JSON data") from error
    require(actual == expected_hash, f"{name} does not match trusted manifest digest")


def summarize_trajectory(trajectory, reference, verification, *, expected_bindings,
                         expected_artifact_sha256):
    """Reduce authenticated final-test evidence, never execute or extend a search.

``expected_bindings`` and ``expected_artifact_sha256`` must be obtained from a
trusted evidence manifest bound to the final freeze, independently of these
payloads. An early terminal run
needs an explicit reason; a run claiming budget completion requires all 1,024
logical requests. An independent winner-verification artifact is required when
any request produced an energy, including an early failed run's incumbent.
"""
    _bindings(expected_bindings)
    _keys(expected_artifact_sha256, ("trajectory", "reference", "verification"), "artifact pins")
    _authenticate(trajectory, expected_artifact_sha256["trajectory"], "trajectory")
    _authenticate(reference, expected_artifact_sha256["reference"], "reference")
    _keys(reference, ("schema_version", "physics", "solver_sha256", "energy_hartree", "n_alpha",
                      "n_beta", "residual_norm", "nuclear_repulsion_included"), "reference")
    require(reference["schema_version"] == "large-qubit-sector-reference-v1", "Wrong reference schema")
    require(canonical_bytes(reference["physics"]) == canonical_bytes(
        {key: expected_bindings[key] for key in PHYSICS_KEYS}), "Reference physical identity mismatch")
    _hash(reference["solver_sha256"], "reference solver")
    for population in ("n_alpha", "n_beta"):
        require(type(reference[population]) is int and reference[population] == 11, "Wrong reference sector")
    require(reference["nuclear_repulsion_included"] is True, "Reference omits nuclear repulsion")
    reference_energy = _number(reference["energy_hartree"], "reference energy")
    residual = _number(reference["residual_norm"], "reference residual")
    require(0.0 <= residual <= ENERGY_TOLERANCE, "Reference residual is not converged")

    _keys(trajectory, ("schema_version", "bindings", "status", "stop_reason", "failure_reason",
                       "requests", "completion_attempts"), "trajectory")
    require(trajectory["schema_version"] == "large-qubit-scored-trajectory-v1", "Wrong trajectory schema")
    require(canonical_bytes(trajectory["bindings"]) == canonical_bytes(expected_bindings),
            "Trajectory fingerprint mismatch")
    requests = trajectory["requests"]
    require(type(requests) is list and len(requests) <= 1024, "Invalid logical request ledger")
    attempts = trajectory["completion_attempts"]
    require(type(attempts) is int and 0 <= attempts <= 2048, "Invalid completion attempt count")
    generator = expected_bindings["method"] in METHODS[:5]
    if generator:
        require(attempts >= max(0, len(requests) - 1), "Completion attempts disagree with requests")
    else:
        require(attempts == 0, "Non-generator method cannot claim completion attempts")

    status, reason = trajectory["status"], trajectory["stop_reason"]
    require(status in ("finished", "failed"), "Trajectory lacks terminal status")
    if status == "failed":
        require(reason == "failure", "Failed trajectory requires explicit failure stop reason")
        _text(trajectory["failure_reason"], "trajectory failure")
    else:
        require(trajectory["failure_reason"] is None and requests, "Finished trajectory must include HF")
        if reason == "request_cap":
            require(len(requests) == 1024, "Truncated ledger claims full request budget")
        elif reason == "attempt_ceiling":
            require(generator and attempts == 2048, "Attempt ceiling is not demonstrated")
        elif reason == "retrieval_exhausted":
            require(expected_bindings["method"] == "training_circuit_retrieval", "Invalid retrieval stop")
        elif reason == "optimizer_stopped":
            require(expected_bindings["method"] == "conventional_vqe", "Invalid optimizer stop")
        else:
            raise ValueError("Unknown or unrecorded early stopping reason")

    first_seen, scored, duplicate_count, cache_hits, failed_requests = {}, [], 0, 0, 0
    for ordinal, row in enumerate(requests, 1):
        _keys(row, ("request_ordinal", "kind", "circuit_sha256", "status", "energy_hartree",
                    "cache_hit", "duplicate_of", "failure_reason"), "scored request")
        require(type(row["request_ordinal"]) is int and row["request_ordinal"] == ordinal,
                "Request ordinals must be contiguous and unique")
        require(row["kind"] == ("hf" if ordinal == 1 else "candidate"), "First logical request must be HF")
        _hash(row["circuit_sha256"], "circuit identity")
        require(type(row["cache_hit"]) is bool, "Cache-hit flag must be boolean")
        cache_hits += int(row["cache_hit"])
        duplicate_of = first_seen.get(row["circuit_sha256"])
        if duplicate_of is None:
            require(row["duplicate_of"] is None, "Invented duplicate linkage")
            first_seen[row["circuit_sha256"]] = ordinal
        else:
            require(type(row["duplicate_of"]) is int and row["duplicate_of"] == duplicate_of,
                    "Duplicate request must identify its first actual occurrence")
            duplicate_count += 1
        require(row["status"] in ("scored", "failed"), "Unknown request status")
        if row["status"] == "failed":
            require(row["energy_hartree"] is None, "Failed query cannot supply a candidate energy")
            _text(row["failure_reason"], "query failure")
            failed_requests += 1
        else:
            energy = _number(row["energy_hartree"], "candidate energy")
            require(row["failure_reason"] is None, "Scored query has conflicting failure data")
            energy_error = _number(energy - reference_energy, "candidate energy error")
            require(energy_error >= -ENERGY_TOLERANCE,
                    "Candidate lies below reference beyond numerical tolerance")
            scored.append(row)
    if status == "finished":
        require(requests[0]["status"] == "scored", "Finished trajectory has no successful initial HF request")

    winner = min(scored, key=lambda row: (row["energy_hartree"], row["request_ordinal"])) if scored else None
    verified_energy = None
    if winner is None:
        require(status == "failed" and verification is None and expected_artifact_sha256["verification"] is None,
                "No-scored-energy outcome must be an explicit failure without invented verification")
    else:
        _authenticate(verification, expected_artifact_sha256["verification"], "winner verification")
        _keys(verification, ("schema_version", "bindings", "request_ordinal", "circuit_sha256",
                             "independent_backend_sha256", "energy_hartree", "normalization_error",
                             "population_leakage"), "winner verification")
        require(verification["schema_version"] == "large-qubit-winner-verification-v1", "Wrong verification schema")
        require(canonical_bytes(verification["bindings"]) == canonical_bytes(expected_bindings),
                "Winner verification fingerprint mismatch")
        require(type(verification["request_ordinal"]) is int
                and verification["request_ordinal"] == winner["request_ordinal"]
                and verification["circuit_sha256"] == winner["circuit_sha256"],
                "Independent verification does not identify the best scored circuit")
        _hash(verification["independent_backend_sha256"], "independent verifier identity")
        require(verification["independent_backend_sha256"] != expected_bindings["backend_sha256"],
                "Independent verification must identify a different implementation")
        verified_energy = _number(verification["energy_hartree"], "verified energy")
        disagreement = _number(verified_energy - winner["energy_hartree"], "verification energy disagreement")
        require(abs(disagreement) <= ENERGY_TOLERANCE,
                "Independent winning energy disagrees with scored energy")
        verified_error = _number(verified_energy - reference_energy, "verified energy error")
        require(verified_error >= -ENERGY_TOLERANCE,
                "Verified energy lies below reference beyond numerical tolerance")
        for check in ("normalization_error", "population_leakage"):
            value = _number(verification[check], check)
            require(0.0 <= value <= SYMMETRY_TOLERANCE, "Independent norm/population checks failed")

    hf_energy = (requests[0]["energy_hartree"] if requests and requests[0]["status"] == "scored" else None)
    improvement = (_number(hf_energy - verified_energy, "HF improvement")
                   if winner and hf_energy is not None else None)
    return {
        "schema_version": "large-qubit-trajectory-outcome-v1", "bindings": copy.deepcopy(expected_bindings),
        "trajectory_status": status, "stop_reason": reason, "failure_reason": trajectory["failure_reason"],
        "endpoint_budget": 1024, "actual_logical_requests": len(requests),
        "completion_attempts": attempts, "duplicate_requests": duplicate_count,
        "cache_hits": cache_hits, "failed_requests": failed_requests,
        "incumbent_carried_forward": winner is not None and len(requests) < 1024,
        "winning_request_ordinal": winner["request_ordinal"] if winner else None,
        "best_verified_energy_hartree": verified_energy,
        "best_verified_error_hartree": verified_error if winner else None,
        "hf_energy_hartree": hf_energy,
        "improvement_over_hf_hartree": improvement,
        "reference_energy_hartree": reference_energy, "reference_residual_norm": residual,
        "artifact_sha256": copy.deepcopy(expected_artifact_sha256),
        "scoring_only": True,
    }


def aggregate_targets(summaries, *, expected_summary_sha256):
    """Aggregate two trusted summaries, retaining failures without a partial mean.

Summary pins must come from the trusted evidence manifest independently of the
payloads. Neither a success flag nor summary-embedded artifact digests alone
authenticate a changed summary. This reducer does not execute the experiments.
"""
    _keys(summaries, TARGETS, "fixed-target summaries")
    _keys(expected_summary_sha256, TARGETS, "trusted summary pins")
    shared = None
    failures, errors = [], []
    for alias in TARGETS:
        summary = summaries[alias]
        _authenticate(summary, expected_summary_sha256[alias], f"{alias} summary")
        require(type(summary) is dict and summary.get("schema_version") == "large-qubit-trajectory-outcome-v1",
                "Invalid target summary")
        _bindings(summary.get("bindings"))
        bindings = summary["bindings"]
        require(bindings["target_id"] == alias, "Target summary alias mismatch")
        identity = {k: bindings[k] for k in ("benchmark_id", "method", "pipeline_seed", "final_freeze_sha256")}
        require(shared is None or identity == shared, "Target summaries belong to different experiments")
        shared = identity
        require(summary.get("endpoint_budget") == 1024 and summary.get("scoring_only") is True,
                "Target summary has wrong endpoint or access namespace")
        status = summary.get("trajectory_status")
        require(status in ("finished", "failed"), "Missing target terminal status")
        if status == "failed":
            _text(summary.get("failure_reason"), "target failure")
            failures.append(alias)
        else:
            error = _number(summary.get("best_verified_error_hartree"), "target energy error")
            energy = _number(summary.get("best_verified_energy_hartree"), "target energy")
            reference = _number(summary.get("reference_energy_hartree"), "target reference")
            difference = _number(energy - reference, "derived target energy error")
            require(error == difference and error >= -ENERGY_TOLERANCE,
                    "Target error is not the signed verified energy difference")
            errors.append(error)
    mean = (_number(errors[0] / 2.0 + errors[1] / 2.0, "two-target mean")
            if not failures else None)
    return {"schema_version": "large-qubit-two-target-outcome-v1", "identity": shared,
            "targets": copy.deepcopy(summaries), "failed_targets": failures,
            "primary_mean_available": not failures,
            "mean_verified_error_hartree": mean,
            "summary_sha256": copy.deepcopy(expected_summary_sha256),
            "interpretation": "descriptive_mean_over_two_fixed_targets_one_pipeline_seed"}
