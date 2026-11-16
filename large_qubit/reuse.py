"""Read-only reuse inventory for the separately identified CO2 study.

Only small public metadata/source files are read. This module never imports the
old experiment packages, loads model weights, scores circuits, or reruns an audit.
A saved audit is fingerprinted evidence of the audit that was previously run;
it is deliberately not reported as a fresh authentication of every raw result.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SCHEMA = "large-qubit-reuse-inventory-v1"
ORIGINAL_PROTOCOL = "phase7/development_protocol.json"
ORIGINAL_PROTOCOL_SHA256 = "f0fd8d52c74bd66007b726b30a96fdeb5f205646577b9b13560a0bde1d295bde"
EXPECTED_BINDING_COUNT = 129
EXPECTED_CONFIGURATION_COUNT = 24
MODEL_KEYS = ("gpt2_xl", "qwen3_4b")
SELECTED_METHODS = ("gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora", "qwen3_4b_weighted_lora")
MAX_METADATA_BYTES = 64 * 1024 * 1024
ASSESSMENT = "phase7/assessments/dev_v1_p5_s20/checkpoint_selection.json"
SAVED_VERIFICATION = "phase7/evidence/finalization_review_v1/step5_verification.json"
PILOT_REVIEW = "phase6/evidence/pilot_review_20261029_parallel_v1/review.json"
PILOT_AUDIT = "phase6/evidence/pilot_review_20261029_parallel_v1/step13_integrity_audit.json"
# These are source roles, not model-input permissions. Public reports may describe
# old training/validation findings; they cannot become new physical labels.
ARTIFACT_ROLES = {
    "phase1/README.md": "historical_foundations",
    "project_information.md": "course_requirements",
    "phase2/benchmark_manifest.json": "small_system_definition",
    "phase2/validation_report.json": "saved_metadata_audit",
    "phase3/validation_report.json": "saved_physics_validation",
    "phase3/completion_audit.json": "saved_physics_completion_audit",
    "phase4/evidence/vqe_validation.json": "saved_vqe_correctness_validation",
    "phase4/evidence/completion_audit.json": "saved_baseline_completion_audit",
    "phase5/release_manifest.json": "historical_lih_release_only",
    "phase5/data_study.md": "historical_acquisition_design_and_coverage",
    "phase5/evidence/completion_audit.json": "saved_corpus_completion_audit",
    "phase6/evidence/pilot_review_20261029_parallel_v1/review.md": "historical_lih_pilot_report",
    PILOT_REVIEW: "saved_lih_pilot_review",
    PILOT_AUDIT: "saved_lih_pilot_integrity_audit",
    SAVED_VERIFICATION: "saved_lih_development_authentication",
    ASSESSMENT: "historical_lih_selection_and_nonconvergence",
    "phase6/model_lock.json": "pretrained_revision_and_tokenizer_lock",
}


class ReuseError(ValueError):
    """A required historical identity or reuse boundary is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReuseError(message)


def _digest(document: Any) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _relative_path(workspace: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Expected a nonempty relative path")
    p = PurePosixPath(relative)
    _require(not p.is_absolute() and "\\" not in relative and ":" not in relative
             and p.as_posix() == relative and all(part not in (".", "..") for part in p.parts),
             f"Unsafe or noncanonical relative path: {relative}")
    resolved = (workspace / p).resolve()
    _require(resolved.is_relative_to(workspace), f"Path escapes workspace: {relative}")
    return resolved


def _bytes(workspace: Path, relative: str) -> bytes:
    path = _relative_path(workspace, relative)
    _require(path.is_file(), f"Required reuse artifact is missing: {relative}")
    before = path.stat()
    _require(before.st_size <= MAX_METADATA_BYTES,
             f"Expected metadata/source file, refusing large payload: {relative}")
    data = path.read_bytes()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             f"Artifact changed during read: {relative}")
    _require(len(data) == before.st_size, f"Artifact size changed during read: {relative}")
    return data


def _json(workspace: Path, relative: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key in {relative}: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ReuseError(f"Nonfinite JSON in {relative}: {value}")

    try:
        value = json.loads(_bytes(workspace, relative).decode("utf-8"),
                           parse_constant=reject, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReuseError(f"Unreadable JSON metadata: {relative}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {relative}")
    return value


def _fingerprint(workspace: Path, relative: str, *, expected: str | None = None) -> dict[str, Any]:
    data = _bytes(workspace, relative)
    actual = sha256(data).hexdigest()
    if expected is not None:
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                 f"Invalid expected SHA256: {relative}")
        _require(actual == expected, f"Historical artifact hash mismatch: {relative}")
    return {"path": relative, "sha256": actual, "bytes": len(data)}


def audit_original_protocol(workspace: str | Path) -> dict[str, Any]:
    """Authenticate the unchanged original registration, without importing it.

    The known original protocol hash anchors its own sidecar and registered
    paths. Configuration files must exist (the old verifier allowed absence).
    """
    root = Path(workspace).resolve()
    identity = _fingerprint(root, ORIGINAL_PROTOCOL, expected=ORIGINAL_PROTOCOL_SHA256)
    sidecar_path = "phase7/development_protocol.sha256"
    sidecar = _bytes(root, sidecar_path).decode("ascii").split()
    _require(bool(sidecar) and sidecar[0] == ORIGINAL_PROTOCOL_SHA256,
             "Original protocol checksum sidecar differs")
    protocol = _json(root, ORIGINAL_PROTOCOL)
    _require(protocol.get("schema_version") == "phase7-development-protocol-v1"
             and protocol.get("final_protocol_frozen") is False,
             "Unexpected original development protocol state")
    bindings = protocol.get("bindings_sha256")
    _require(isinstance(bindings, dict) and len(bindings) == EXPECTED_BINDING_COUNT,
             "Original registered binding count differs")
    verified = [_fingerprint(root, path, expected=expected)
                for path, expected in sorted(bindings.items())]
    registered_phase7_sources = {path for path in bindings
                                if path.startswith("phase7/") and path.endswith(".py")}
    live_phase7_sources = {path.relative_to(root).as_posix()
                          for path in (root / "phase7").rglob("*.py")
                          if not path.name.startswith("test_") and "__pycache__" not in path.parts}
    _require(live_phase7_sources == registered_phase7_sources,
             "Original recursively hashed Phase 7 source namespace changed")
    conditions = protocol.get("training_conditions")
    _require(isinstance(conditions, dict), "Original training conditions are missing")
    configurations = []
    seen = set()
    for method, condition in sorted(conditions.items()):
        for candidate in condition["configurations"]:
            relative = candidate["config_path"]
            _require(relative not in seen, "Duplicate registered configuration path")
            seen.add(relative)
            config = candidate["configuration"]
            _require(_digest(config) == candidate["configuration_sha256"],
                     f"Registered configuration digest differs: {relative}")
            _require(_json(root, relative) == config,
                     f"Registered configuration content differs: {relative}")
            configurations.append({**_fingerprint(root, relative), "method_id": method,
                                   "configuration_sha256": candidate["configuration_sha256"]})
    _require(len(configurations) == EXPECTED_CONFIGURATION_COUNT,
             "Original registered configuration count differs")
    return {"protocol": identity,
            "checksum_sidecar": _fingerprint(root, sidecar_path),
            "registered_binding_count": len(verified), "registered_bindings": verified,
            "registered_configuration_count": len(configurations),
            "registered_configurations": configurations,
            "final_protocol_frozen": False}


def _model_manifests(root: Path) -> list[dict[str, Any]]:
    lock = _json(root, "phase6/model_lock.json")
    lock_hash = _fingerprint(root, "phase6/model_lock.json")["sha256"]
    output = []
    for key in MODEL_KEYS:
        item = lock["models"][key]
        revision = item["revision"]
        _require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                 f"Unpinned pretrained revision: {key}")
        relative = f"phase6/cache/models/weight_manifests/{key}_{revision}.json"
        manifest = _json(root, relative)
        _require(manifest.get("schema") == "phase6-weight-download-v1"
                 and manifest.get("model_key") == key
                 and manifest.get("revision") == revision
                 and manifest.get("repo_id") == item["repo_id"]
                 and manifest.get("model_lock_sha256") == lock_hash,
                 f"Weight manifest metadata differs from pinned model: {key}")
        output.append({"model_key": key, "repo_id": item["repo_id"], "revision": revision,
                       "manifest": _fingerprint(root, relative),
                       "allowed_reuse": "pretrained_base_and_tokenizer_only",
                       "weight_payloads_authenticated_by_this_inventory": False,
                       "requires_separate_local_weight_authentication": True})
    return output


def _selected_metadata(root: Path, assessment: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for method in SELECTED_METHODS:
        selected = assessment["selected"][method]
        checkpoint = selected["checkpoint"]
        config = checkpoint["configuration"]
        _require(selected["method_id"] == method and config.get("pipeline_seed") == 11,
                 f"Unexpected historical selected condition: {method}")
        candidate = selected["candidate_id"]
        complete_pass = selected["checkpoint_pass"]
        _require(type(complete_pass) is int and complete_pass > 0,
                 "Historical complete pass must be a positive integer")
        expected_directory = f"phase7/runs/dev_v1/{method}_{candidate}/pass_{complete_pass:03d}"
        _require(checkpoint["path"] == expected_directory,
                 f"Historical checkpoint path does not match its condition: {method}")
        manifest_path = expected_directory + "/checkpoint_manifest.json"
        binding_path = f"phase7/runs/dev_v1/{method}_{candidate}/run_binding.json"
        manifest_record = _fingerprint(root, manifest_path, expected=checkpoint["manifest_sha256"])
        binding_record = _fingerprint(root, binding_path, expected=checkpoint["run_binding_sha256"])
        manifest = _json(root, manifest_path)
        binding = _json(root, binding_path)
        identity = checkpoint["identity_sha256"]
        _require(manifest.get("identity_sha256") == binding.get("identity_sha256") == identity
                 and _digest(binding["identity"]) == identity
                 and manifest.get("complete_pass") == checkpoint.get("complete_pass") == complete_pass
                 and binding["identity"]["configuration"] == config,
                 f"Historical checkpoint metadata chain differs: {method}")
        registered_path = f"phase7/configs/{method}_{candidate}.json"
        _require(_json(root, registered_path) == config,
                 f"Selection does not match registered configuration: {method}")
        output.append({"method_id": method, "candidate_id": candidate,
                       "complete_pass": complete_pass, "configuration": config,
                       "checkpoint_manifest": manifest_record, "run_binding": binding_record,
                       "allowed_reuse": "historical_hyperparameter_context_only",
                       "adapter_payloads_authenticated_by_this_inventory": False,
                       "usable_as_co2_initialization": False})
    return output


def build_reuse_inventory(workspace: str | Path) -> dict[str, Any]:
    """Build a deterministic inventory; write nothing and execute no experiments.

    This is a migration record, not an independent re-audit of old trajectories.
    It fails closed if required public artifacts are missing or inconsistent.
    """
    root = Path(workspace).resolve()
    original = audit_original_protocol(root)
    artifacts = [{**_fingerprint(root, path), "role": role,
                  "new_co2_physical_label_or_result": False}
                 for path, role in sorted(ARTIFACT_ROLES.items())]
    pilot = _json(root, PILOT_REVIEW)
    audit = _json(root, PILOT_AUDIT)
    verification = _json(root, SAVED_VERIFICATION)
    assessment = _json(root, ASSESSMENT)
    _require(pilot.get("schema") == "phase6-pilot-review-v1", "Unexpected pilot review schema")
    _require(all(pilot.get(field) is True for field in
                 ("step12_complete", "step13_complete", "step14_review_complete"))
             and pilot.get("phase6_empirical_correctness_gate") == "passed"
             and audit.get("audit_result") == "passed",
             "Saved Phase 6 pilot reports do not support completed steps and a passed correctness gate")
    _require(assessment.get("schema_version") == "phase7-checkpoint-selection-v1"
             and assessment.get("selection_is_final_freeze") is False,
             "Unexpected historical assessment state")
    _require(assessment.get("pretrained_allowance_passes") == 5
             and assessment.get("scratch_allowance_passes") == 20,
             "Expected historical 5/20-pass assessment")
    convergence = assessment["convergence"]["all_convergence_ready"]
    _require(convergence is False and verification.get("all_convergence_ready") is False,
             "Original 5/20 nonconvergence record differs; investigate before migrating")
    _require(verification.get("all_artifact_authentication_passed") is True
             and verification.get("saved_selection_matches_recomputed") is True
             and verification.get("saved_convergence_matches_recomputed") is True
             and verification.get("checkpoint_jobs") == 348
             and verification.get("rows_authenticated") == 696,
             "Saved development audit does not support the declared reuse inventory")
    return {
        "schema": SCHEMA,
        "scope": "public_metadata_and_registered_source_authentication_only",
        "original_registration": original,
        "historical_artifacts": artifacts,
        "pretrained_models": _model_manifests(root),
        "historical_selected_configurations": _selected_metadata(root, assessment),
        "saved_audit_claims": {
            "interpretation": "claims_from_fingerprinted_saved_reports_not_fresh_raw_result_authentication",
            "phase6": {"source": PILOT_REVIEW, "integrity_source": PILOT_AUDIT,
                       "scope": "LiH_seed11_training_geometry_pilot",
                       "step12_complete": pilot.get("step12_complete"),
                       "step13_complete": pilot.get("step13_complete"),
                       "step14_review_complete": pilot.get("step14_review_complete"),
                       "empirical_correctness_gate": pilot.get("phase6_empirical_correctness_gate"),
                       "saved_integrity_result": audit.get("audit_result"),
                       "saved_hash_references_checked": audit.get("hash_references_checked")},
            "phase7": {"source": SAVED_VERIFICATION,
                       "scope": "LiH_seed11_development_selection",
                       "checkpoint_jobs": verification["checkpoint_jobs"],
                       "alias_records_authenticated_in_saved_audit": verification["rows_authenticated"],
                       "all_convergence_ready": convergence,
                       "final_freeze_completed": False}},
        "migration_policy": {
            "allowed": ["pinned_pretrained_base_weights_and_tokenizers",
                        "source_bound_generic_components_with_new_physical_guards",
                        "small_system_correctness_fixtures",
                        "historical_LiH_results_labelled_with_original_scope",
                        "historical_hyperparameter_context_and_acquisition_design"],
            "forbidden_as_co2_inputs_or_results": ["LiH_trained_adapters",
                        "LiH_energies_or_references", "LiH_best_circuits_or_catalogue_IDs",
                        "LiH_run_bindings", "historical_validation_outcomes_as_new_target_results"],
            "fresh_co2_labels_required": True,
            "fresh_co2_adapter_fits_required": list(SELECTED_METHODS),
            "historical_costs": "preserve_separately_from_new_branch_expenditure",
            "new_code_namespace": "large_qubit/",
            "new_outputs_require": ["revised_plan_hash", "new_code_and_dependency_identities",
                                    "new_Hamiltonian_and_orbital_identity", "new_catalogue_identity",
                                    "new_release_and_checkpoint_identity", "partition_and_seed_identity"],
            "phase8_executor_reused": False,
            "original_phase7_commands_remain_old_protocol": True},
        "audit_work": {"raw_trajectories_reauthenticated": False,
                       "scientific_audits_rerun": False, "model_weights_loaded": False,
                       "weight_payloads_hashed": False, "training_started": False,
                       "energy_evaluations": 0, "network_access": False,
                       "checkpoint_initializations_performed": 0}}


def verify_reuse_inventory(document: dict[str, Any], workspace: str | Path) -> dict[str, Any]:
    """Rebuild and compare every role/path/value, rejecting omitted or relabelled evidence.

    The enclosing registration must additionally bind this inventory's own bytes.
    This function performs no large-payload or private-results authentication.
    """
    _require(isinstance(document, dict) and document.get("schema") == SCHEMA,
             "Unrecognized reuse inventory")
    expected = build_reuse_inventory(workspace)
    _require(_digest(document) == _digest(expected),
             "Reuse inventory differs from current files or declared reuse roles")
    return {"all_passed": True, "inventory_sha256": _digest(expected),
            "registered_bindings_verified": expected["original_registration"]["registered_binding_count"],
            "registered_configurations_verified": expected["original_registration"]["registered_configuration_count"],
            "historical_artifacts_fingerprinted": len(expected["historical_artifacts"]),
            "raw_results_reauthenticated": False,
            "original_convergence_ready": False, "co2_training_or_scoring_executed": False}
