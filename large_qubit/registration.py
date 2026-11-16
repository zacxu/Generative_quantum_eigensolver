"""Immutable registrations, with an anchored archive of the original foundation.

The archived Sections 1-2 sources remain independently verifiable after active
development. No implementation registration is the final experiment freeze.
"""

import json
import math
from pathlib import Path
import re
import shutil

from .common import digest, read_json, require, sha256_file, workspace_path
from .protocol import MODEL_REVISIONS, PENDING, build_study, planned_jobs, validate_study
from .reuse import build_reuse_inventory, verify_reuse_inventory
from .models import inspect_assets


ORIGINAL_REGISTRATION = "large_qubit/registrations/sections_1_2_v1"
ORIGINAL_MANIFEST_SHA256 = "8fe6d1fbc2acac1eae7962c5db72131085d96e85a769df74d70682ae1efeea2f"
ORIGINAL_ARCHIVE = "large_qubit/archives/sections_1_2_v1"
SECTION3_REGISTRATION = "large_qubit/registrations/sections_1_3_v1"
SECTION3_MANIFEST_SHA256 = "5d6cc474395d418307e2130df1916ab691a61392c9cda90ceb564d58ffb988a3"
SECTION3_ARCHIVE = "large_qubit/archives/sections_1_3_v1"
SECTION4_REGISTRATION = "large_qubit/registrations/sections_1_4_v1"
SECTION4_MANIFEST_SHA256 = "b257a2dc7e70c7f41c6228cc3669a1239ac71e9745398bf76b2097293e976f60"
SECTION4_ARCHIVE = "large_qubit/archives/sections_1_4_v1"
SECTION5_REGISTRATION = "large_qubit/registrations/sections_1_5_v1"
SECTION5_MANIFEST_SHA256 = "a22de16bfe94e7607d273a091ffed603714a99ea7069a24da2c172c1d8fc8cb8"
SECTION5_ARCHIVE = "large_qubit/archives/sections_1_5_v1"
SECTION6_REGISTRATION = "large_qubit/registrations/sections_1_6_v1"
SECTION6_MANIFEST_SHA256 = "f21d24f044a7fa4ef2c1609451fd513d05aba38b11562e5e99505ef1f7d11856"
SECTION6_ARCHIVE = "large_qubit/archives/sections_1_6_v1"
SECTION6_V2_REGISTRATION = "large_qubit/registrations/sections_1_6_v2"
SECTION6_V2_MANIFEST_SHA256 = "1007e51094a01537482041ac57e07a180ece667730fbaa40d778cd9ea64a6a23"
SECTION6_V2_ARCHIVE = "large_qubit/archives/sections_1_6_v2"
SECTION7_INITIAL_REGISTRATION = "large_qubit/registrations/sections_1_7_v1"
SECTION7_INITIAL_MANIFEST_SHA256 = "95a34afb3acec5316585ee053857440610acc8c27ea517ea0e12e98135d48f9b"
SECTION7_INITIAL_ARCHIVE = "large_qubit/archives/sections_1_7_v1"
SECTION7_GPU_DOCS_REGISTRATION = "large_qubit/registrations/sections_1_7_v1_gpu_docs_v1"
SECTION7_GPU_DOCS_MANIFEST_SHA256 = "66016d518f12958753f02a431f5463dc099df3e14f2741f00633d47510c8ebbf"
SECTION7_GPU_DOCS_ARCHIVE = "large_qubit/archives/sections_1_7_v1_gpu_docs_v1"
SECTION7_GPU_DOCS_AMENDMENT = "large_qubit/amendments/section7_gpu_docs_v1.json"
SECTION7_GPU_DOCS_AMENDMENT_SHA256 = "b26b534d271836df888837f0bc03b04087dc6173b631e004c68d10091137eb81"
SECTION7_INITIAL_PREPARATION_PATH = "large_qubit/evidence/sections_1_7_v1/preparation.json"
SECTION7_INITIAL_PREPARATION_SHA256 = "8a2b6bef5bb5a217b5abfcbcd33a2f74435b00c988fd3b89b150142abbbfb908"
SECTION7_INITIAL_IMPLEMENTATION_PATH = "large_qubit/evidence/sections_1_7_v1/implementation.json"
SECTION7_UUID_AMENDMENT = "large_qubit/amendments/section7_h200_uuid_v1.json"
SECTION7_UUID_AMENDMENT_SHA256 = "df359a7fdfa5d8f6a96ba7d24ff71e71d234b493734f513ddef5a6cd522b2ff9"
SECTION7_UUID_REGISTRATION = "large_qubit/registrations/sections_1_7_v2_gpu_uuid"
SECTION7_UUID_MANIFEST_SHA256 = "252b7b19e6e6bd4a26c0280c8e8e20757bcffe50aede41c251efc2ba8b3e84e6"
SECTION7_UUID_ARCHIVE = "large_qubit/archives/sections_1_7_v2_gpu_uuid"
SECTION7_GUIDE_AMENDMENT = "large_qubit/amendments/section7_guide_v1.json"
DEFAULT_REGISTRATION = "large_qubit/registrations/sections_1_7_v2_guide_v1"
REGISTRATION_ID = "sections_1_7_v2"
BOUND_DOCUMENTS = (
    "research_plan_revised.md", "requirements.txt", "phase6/requirements.txt",
    "large_qubit/study.json", "large_qubit/README.md", "large_qubit/claims.md",
)
PHYSICAL_DEFINITION_PATH = "large_qubit/physical_definition.json"
GRID_AMENDMENT_PATH = "large_qubit/amendments/section3_grid_v3.json"
SECTION3_DOCUMENTS = (
    PHYSICAL_DEFINITION_PATH, "large_qubit/section3_policies.json",
    "large_qubit/section3.md", "large_qubit/requirements-chemistry.txt",
    "large_qubit/amendments/section3_grid_v3.json",
)
PHYSICS_MANIFEST_PATH = "large_qubit/physics/section3_v3/manifest.json"
SECTION4_DOCUMENTS = ("large_qubit/section4.md", "large_qubit/section4_policy.json",
                      "large_qubit/scripts/run_section4_tmux.sh")
SECTION4_MANIFEST_PATH = "large_qubit/validation/section4_v1/manifest.json"
SECTION5_DOCUMENTS = ("large_qubit/section5_policy.json", "large_qubit/catalogues/co2_screened_v1.json",
                      "large_qubit/section5.md", "large_qubit/scripts/run_section5_tmux.sh")
SECTION5_MANIFEST_PATH = "large_qubit/validation/section5_v1/manifest.json"
SECTION6_DOCUMENTS = ("large_qubit/section6_policy.json", "large_qubit/section6.md",
                      "large_qubit/scripts/run_section6_tmux.sh")
SECTION6_PREPARATION_PATH = "large_qubit/evidence/sections_1_6_v1/preparation.json"
SECTION6_MANIFEST_PATH = "large_qubit/validation/section6_v1/manifest.json"
SECTION6_RELEASE_MANIFEST_PATH = "large_qubit/releases/co2_corpus_v1/manifest.json"
ACQUISITION_AMENDMENT_PATH = "large_qubit/amendments/section6_acquisition_v2.json"
ACQUISITION_AMENDMENT_SHA256 = "611592dad5d900a536939ce4435f5a1982174d2e1a94df1f947c6a95357cc130"
SECTION6_V2_DOCUMENTS = (ACQUISITION_AMENDMENT_PATH, "large_qubit/section6_v2_policy.json",
                         "large_qubit/section6_v2.md", "large_qubit/scripts/run_section6_v2_tmux.sh")
SECTION6_V2_PREPARATION_PATH = "large_qubit/evidence/sections_1_6_v2/preparation.json"
SECTION6_V2_MANIFEST_PATH = "large_qubit/validation/section6_v2/manifest.json"
SECTION6_V2_RELEASE_MANIFEST_PATH = "large_qubit/releases/co2_corpus_v2/manifest.json"
SECTION7_DOCUMENTS = ("large_qubit/section7_policy.json", "large_qubit/section7.md",
                      "large_qubit/scripts/run_section7_tmux.sh", "large_qubit/requirements-training.txt")
SECTION7_PREPARATION_PATH = "large_qubit/evidence/sections_1_7_gpu_uuid_v1/preparation.json"
SECTION7_IMPLEMENTATION_PATH = "large_qubit/evidence/sections_1_7_gpu_uuid_v1/implementation.json"
SECTION7_CACHE_MANIFEST_PATHS = (
    "large_qubit/training_data/section7_v1/gpt2_xl/manifest.json",
    "large_qubit/training_data/section7_v1/qwen3_4b/manifest.json",
)
REUSABLE_COMPONENTS = {
    "grammar": ["phase6/grammar.py"],
    "fitting_weights_and_complete_answer_loss": ["phase6/objective.py"],
    "native_lora_targets_and_adapter_setup": ["phase6/models.py"],
    "training_checkpoint_and_rng_patterns": ["phase6/pipeline.py", "phase6/train.py", "phase7/train.py"],
    "detached_controller_and_worker_patterns": ["phase7/run.py", "phase7/scripts/tmux_worker.sh"],
}


def current_bindings(workspace):
    root = Path(workspace).resolve(strict=True)
    package = workspace_path(root, "large_qubit", namespace="large_qubit")
    sources = [p.relative_to(root).as_posix() for p in sorted(package.rglob("*.py"))
               if "__pycache__" not in p.parts
               and p.relative_to(package).parts[0] not in
                   {"archives", "registrations", "evidence", "preflights", "training_runs", "training_data"}]
    require(sources and "large_qubit/protocol.py" in sources, "New source package missing")
    paths = (list(BOUND_DOCUMENTS) + list(SECTION3_DOCUMENTS) + list(SECTION4_DOCUMENTS)
             + list(SECTION5_DOCUMENTS) + sources)
    physics = workspace_path(root, PHYSICS_MANIFEST_PATH, namespace="large_qubit/physics")
    if physics.exists():
        paths.extend((PHYSICS_MANIFEST_PATH, PHYSICS_MANIFEST_PATH.removesuffix(".json") + ".sha256"))
    gate = workspace_path(root, SECTION4_MANIFEST_PATH, namespace="large_qubit/validation")
    if gate.exists():
        paths.extend((SECTION4_MANIFEST_PATH, SECTION4_MANIFEST_PATH.removesuffix(".json") + ".sha256"))
    section5 = workspace_path(root, SECTION5_MANIFEST_PATH, namespace="large_qubit/validation")
    if section5.exists():
        paths.extend((SECTION5_MANIFEST_PATH, SECTION5_MANIFEST_PATH.removesuffix(".json") + ".sha256"))
    # These artifacts are absent during implementation. Once registered, adding,
    # removing or changing any of them changes the active binding inventory.
    for relative in (*SECTION6_DOCUMENTS, SECTION6_PREPARATION_PATH,
                     *SECTION6_V2_DOCUMENTS, SECTION6_V2_PREPARATION_PATH,
                     *SECTION7_DOCUMENTS, SECTION7_PREPARATION_PATH,
                     SECTION7_IMPLEMENTATION_PATH, *SECTION7_CACHE_MANIFEST_PATHS,
                     SECTION7_GPU_DOCS_AMENDMENT, SECTION7_UUID_AMENDMENT, SECTION7_GUIDE_AMENDMENT):
        if workspace_path(root, relative, namespace="large_qubit").exists():
            paths.append(relative)
    for relative in SECTION7_CACHE_MANIFEST_PATHS:
        sidecar = relative.removesuffix(".json") + ".sha256"
        if workspace_path(root, sidecar, namespace="large_qubit/training_data").exists():
            paths.append(sidecar)
    for relative in (SECTION6_MANIFEST_PATH, SECTION6_RELEASE_MANIFEST_PATH,
                     SECTION6_V2_MANIFEST_PATH, SECTION6_V2_RELEASE_MANIFEST_PATH):
        if workspace_path(root, relative, namespace="large_qubit").exists():
            paths.extend((relative, relative.removesuffix(".json") + ".sha256"))
    return {p: sha256_file(workspace_path(root, p)) for p in sorted(set(paths))}


def _registration_files(root, registration):
    folder = workspace_path(root, registration, namespace="large_qubit/registrations")
    require(folder != root / "large_qubit/registrations", "Choose a registration subdirectory")
    manifest = workspace_path(root, f"{registration}/manifest.json",
                              namespace="large_qubit/registrations")
    sidecar = workspace_path(root, f"{registration}/manifest.sha256",
                             namespace="large_qubit/registrations")
    expected = sidecar.read_text(encoding="ascii").strip()
    require(expected == sha256_file(manifest), "Registration sidecar mismatch")
    return manifest, expected


def verify_original_registration(workspace):
    """Verify archived original source and live historical evidence, never new source.

    The known manifest digest anchors the archive. Rewriting the old manifest and
    its sidecar cannot legitimize a changed source snapshot or historical report.
    No archived Python is imported or executed.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, expected_digest = _registration_files(root, ORIGINAL_REGISTRATION)
    require(expected_digest == ORIGINAL_MANIFEST_SHA256,
            "Original registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("registration_id") == "sections_1_2_v1"
            and saved.get("schema_version") == "large-qubit-foundation-registration-v1"
            and saved.get("execution_ready") is False,
            "Unexpected original foundation registration")
    archive = workspace_path(root, ORIGINAL_ARCHIVE, namespace="large_qubit/archives")
    require(archive.is_dir(), "Original source archive is missing")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        # Validate both the original relative identity and its archived location.
        workspace_path(root, relative)
        path = workspace_path(root, f"{ORIGINAL_ARCHIVE}/{relative}",
                              namespace="large_qubit/archives")
        require(path.is_file(), f"Original archived binding is missing: {relative}")
        require(sha256_file(path) == expected,
                f"Original archived binding changed: {relative}")
        archived[relative] = expected
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Original pretrained model metadata changed")
    for component in saved["reusable_components"].values():
        for relative, expected in component["sources_sha256"].items():
            require(sha256_file(workspace_path(root, relative)) == expected,
                    f"Original reusable historical source changed: {relative}")
    return {
        "all_passed": True, "registration": ORIGINAL_REGISTRATION,
        "manifest_sha256": expected_digest, "bound_files_verified": len(archived),
        "source_verification_mode": "archived_original_sources_and_live_historical_evidence",
        "active_source_verified_as_original": False,
        "archive_root": ORIGINAL_ARCHIVE, "archived_bindings_sha256": archived,
        "historical_reuse_verification": reuse,
        "model_weight_bytes_verified": False, "execution_ready": False,
        "scope": "Original Sections 1-2 archive and historical evidence, not current active code",
    }


def validate_physical_definition(workspace):
    """Authenticate the static physical specification without scientific imports."""
    from .definitions import validate_definition

    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, PHYSICAL_DEFINITION_PATH, namespace="large_qubit")
    definition = validate_definition(read_json(path))
    return {"path": PHYSICAL_DEFINITION_PATH, "file_sha256": sha256_file(path),
            "definition_sha256": digest(definition), "definition": definition,
            "numerical_physics_validated": False, "scientific_results_present": False}


def verify_section3_registration(workspace):
    """Authenticate the archived Section 3 implementation and its live evidence.

    Archive bytes, original lineage and saved numerical-certificate payloads are
    checked without importing archived Python or interpreting numerical arrays.
    The report makes no claim that current Section 4 code is the old source.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, SECTION3_REGISTRATION)
    require(actual == SECTION3_MANIFEST_SHA256,
            "Section 3 registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section3-registration-v1"
            and saved.get("registration_id") == "sections_1_3_v1"
            and saved.get("execution_ready") is False, "Unexpected Section 3 registration")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        workspace_path(root, relative)
        archived_path = workspace_path(root, f"{SECTION3_ARCHIVE}/{relative}",
                                       namespace="large_qubit/archives")
        require(archived_path.is_file(), f"Section 3 archived binding is missing: {relative}")
        require(sha256_file(archived_path) == expected,
                f"Section 3 archived binding changed: {relative}")
        archived[relative] = expected
    original = verify_original_registration(root)
    require(digest(original) == digest(saved["original_registration_lineage"]),
            "Section 3 original-registration lineage changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Section 3 pretrained model metadata changed")
    amendment = verify_grid_amendment(root)
    require(digest(amendment) == digest(saved["grid_amendment_verification"]),
            "Section 3 preserved grid-amendment evidence changed")
    physics = saved["realized_physics_manifest"]
    payload_count = 0
    if physics is not None:
        path = workspace_path(root, physics["path"], namespace="large_qubit/physics")
        sidecar = workspace_path(root, physics["sidecar_path"], namespace="large_qubit/physics")
        require(sha256_file(path) == physics["manifest_sha256"]
                and sha256_file(sidecar) == physics["sidecar_sha256"]
                and sidecar.read_text(encoding="ascii").strip() == physics["manifest_sha256"],
                "Section 3 saved physical manifest or sidecar changed")
        certificate = read_json(path)
        require(certificate["all_passed"] is True, "Section 3 saved physical certificate did not pass")
        for relative, expected in certificate["files"].items():
            candidate = workspace_path(root, relative, namespace="large_qubit")
            require(candidate.is_file() and sha256_file(candidate) == expected,
                    f"Section 3 physical payload missing or changed: {relative}")
            payload_count += 1
    return {"all_passed": True, "registration": SECTION3_REGISTRATION,
            "manifest_sha256": actual, "bound_files_verified": len(archived),
            "source_verification_mode": "archived_sections_1_3_sources_and_live_saved_evidence",
            "active_source_verified_as_original": False,
            "archive_root": SECTION3_ARCHIVE, "archived_bindings_sha256": archived,
            "original_registration_lineage": original, "historical_reuse_verification": reuse,
            "physical_payload_files_verified": payload_count,
            "section_3_complete": physics is not None,
            "saved_section_3_physical_certificates_verified": physics is not None,
            "numerical_arrays_recomputed_by_registration": False,
            "execution_ready": False, "model_weight_bytes_verified": False,
            "scope": "Archived Sections 1-3 implementation and unchanged saved evidence, not active Section 4 code"}


def _verify_grid_amendment_record(workspace, amendment_path, version):
    """Authenticate the preproduction grid amendment and all preserved evidence.

    Read JSON and hash bytes only. Archived Python and numerical arrays are
    never imported or loaded. The active registration binds this report and
    the amendment document; this is not an external digital signature.
    """
    root = Path(workspace).resolve(strict=True)
    require(version in (2, 3), "Unknown reviewed grid amendment version")
    path = workspace_path(root, amendment_path, namespace="large_qubit/amendments")
    record = read_json(path)
    fields = {"schema_version", "amendment_id", "benchmark_id", "before_production", "geometry_id",
              "old_bond_length_angstrom", "new_bond_length_angstrom", "reason", "decision_inputs",
              "failed_minimum_external_hessian_eigenvalue", "thresholds_or_SCF_method_changed",
              "training_stretch_range_reduced", "requires_all_six_geometries_to_pass_unchanged_checks",
              "unchanged_final_targets", "incurred_costs", "preserved_artifacts_sha256"}
    if version > 2:
        fields |= {"prior_amendment", "cumulative_failed_attempt_costs"}
    require(type(record) is dict and set(record) == fields, "Grid amendment schema fields differ")
    fixed = {
        "schema_version": f"section3-grid-amendment-v{version}",
        "amendment_id": f"training_T03_stable_RHF_v{version}", "benchmark_id": "gqe-co2-sto3g-30q-v1",
        "before_production": True, "geometry_id": "CO2_T03",
        "old_bond_length_angstrom": 1.40 if version == 2 else 1.35,
        "new_bond_length_angstrom": 1.35 if version == 2 else 1.32,
        "reason": "external RHF-to-UHF instability under the predeclared stable-RHF convention; not an invalid molecular Hamiltonian",
        "decision_inputs": "SCF convergence and stability only; no candidate-circuit or correlated-reference energies",
        "thresholds_or_SCF_method_changed": False, "training_stretch_range_reduced": True,
        "requires_all_six_geometries_to_pass_unchanged_checks": True,
        "unchanged_final_targets": {"CO2_I01": 1.10, "CO2_I02": 1.30},
    }
    require(digest({name: record[name] for name in fixed}) == digest(fixed),
            "Grid amendment changes its registered scientific scope or feedback policy")
    eigenvalue = record["failed_minimum_external_hessian_eigenvalue"]
    require(type(eigenvalue) is float and math.isfinite(eigenvalue) and eigenvalue < -1e-7,
            "Grid amendment lacks a finite failed external-stability mode")
    costs = record["incurred_costs"]
    require(type(costs) is dict and set(costs) == {"SCF_geometries_attempted", "circuit_energy_requests",
                                                "failed", "passed", "wall_seconds"},
            "Grid amendment cost fields differ")
    require(digest({name: costs[name] for name in ("SCF_geometries_attempted", "circuit_energy_requests",
                                                 "failed", "passed")}) == digest({
        "SCF_geometries_attempted": 3, "circuit_energy_requests": 0, "failed": 1, "passed": 2}),
        "Grid amendment changed or omitted incurred logical costs")
    wall = costs["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
            "Grid amendment wall time must be finite and nonnegative")

    prior = None
    if version > 2:
        pointer = record["prior_amendment"]
        prior_path = f"large_qubit/amendments/section3_grid_v{version - 1}.json"
        require(type(pointer) is dict and set(pointer) == {"path", "sha256"}
                and pointer["path"] == prior_path, "Grid amendment prior-record identity differs")
        prior_file = workspace_path(root, prior_path, namespace="large_qubit/amendments")
        require(pointer["sha256"] == sha256_file(prior_file), "Prior grid amendment hash changed")
        prior = _verify_grid_amendment_record(root, prior_path, version - 1)
        cumulative = {name: prior["cumulative_failed_attempt_costs"][name] + costs[name] for name in costs}
        require(digest(record["cumulative_failed_attempt_costs"]) == digest(cumulative),
                "Cumulative failed-attempt costs omit or change earlier work")
    else:
        cumulative = costs
    archive = f"large_qubit/archives/section3_physical_v{version - 1}"
    old_sources = ("large_qubit/chemistry.py", "large_qubit/definitions.py", "large_qubit/orbitals.py",
                   "large_qubit/physical_definition.json", "large_qubit/section3_policies.json")
    failed = f"large_qubit/physics/section3_v{version - 1}"
    private = f"large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/physical_validation/section3_v{version - 1}"
    expected_paths = {f"{archive}/{name}" for name in old_sources} | {
        f"{archive}/snapshot.json", f"{failed}/failure.json", f"{failed}/failure.sha256",
        f"{private}/CO2_T01.json", f"{private}/CO2_T01.npz", f"{private}/CO2_T02.json",
        f"{private}/CO2_T02.npz", f"{private}/failure_diagnostics.json"}
    if prior is not None:
        expected_paths |= set(prior["preserved_artifacts_sha256"]) | {prior_path}
    preserved = record["preserved_artifacts_sha256"]
    require(type(preserved) is dict and set(preserved) == expected_paths,
            "Grid amendment must preserve exactly every original and prior-chain artifact")
    if prior is not None:
        require(all(preserved[name] == expected for name, expected in prior["preserved_artifacts_sha256"].items())
                and preserved[prior_path] == pointer["sha256"], "Prior-chain artifact hashes were changed")
    for prefix in (archive, failed, private):
        folder = workspace_path(root, prefix, namespace="large_qubit")
        actual_files = {item.relative_to(root).as_posix() for item in folder.rglob("*") if item.is_file()}
        declared_files = {name for name in expected_paths if name.startswith(prefix + "/")}
        require(actual_files == declared_files,
                f"Preserved amendment directory has missing or unrecorded files: {prefix}")
    for relative, expected in preserved.items():
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Malformed preserved artifact SHA256")
        artifact = workspace_path(root, relative, namespace="large_qubit")
        require(artifact.is_file() and sha256_file(artifact) == expected,
                f"Preserved grid amendment artifact missing or changed: {relative}")

    snapshot = read_json(root / archive / "snapshot.json")
    require(type(snapshot) is dict and set(snapshot) == {"schema", "bindings_sha256"}
            and snapshot["schema"] == f"section3-pre-amendment-snapshot-v{version - 1}"
            and set(snapshot["bindings_sha256"]) == set(old_sources), "Original physical snapshot differs")
    require(all(snapshot["bindings_sha256"][name] == preserved[f"{archive}/{name}"]
                for name in old_sources), "Archived source snapshot hashes differ from preserved artifacts")
    old_definition = read_json(root / archive / PHYSICAL_DEFINITION_PATH)
    require(old_definition["definition_id"] == f"co2_physical_definition_v{version - 1}"
            and old_definition["geometries"]["CO2_T03"]["bond_length_angstrom"] == record["old_bond_length_angstrom"],
            "Original physical definition no longer preserves its failed grid")
    policies = read_json(root / archive / "large_qubit/section3_policies.json")
    current_policies = read_json(workspace_path(root, "large_qubit/section3_policies.json", namespace="large_qubit"))
    require(all(digest(policies[name]) == digest(current_policies[name])
                for name in ("scf_policy", "orbital_conventions")),
            "SCF/stability or orbital policy was changed during the grid amendment")
    failure = read_json(root / failed / "failure.json")
    require((root / failed / "failure.sha256").read_text(encoding="ascii").strip()
            == preserved[f"{failed}/failure.json"], "Failed-attempt sidecar differs")
    require(failure["all_passed"] is False and failure["physical_validation_passed"] is False
            and failure["benchmark_id"] == record["benchmark_id"]
            and failure["failed_geometry_id"] == "CO2_T03"
            and failure["completed_geometries"] == ["CO2_T02", "CO2_T01"]
            and failure["diagnostics_path"] == f"{private}/failure_diagnostics.json"
            and failure["diagnostics_sha256"] == preserved[f"{private}/failure_diagnostics.json"]
            and failure["wall_seconds"] == wall, "Failed-attempt record and amendment disagree")
    diagnostics = read_json(root / private / "failure_diagnostics.json")
    require(diagnostics["geometry_id"] == "CO2_T03" and diagnostics["all_passed"] is False
            and diagnostics["physical_validation_passed"] is False, "Failure diagnostics status differs")
    details = diagnostics["diagnostics"]
    require(details["geometry"] == old_definition["geometries"]["CO2_T03"]
            and details["scf"]["converged"] is True
            and details["physical_validation_passed"] is False
            and details["correlated_reference_computed"] is False
            and digest(details["scf_policy"]) == digest(policies["scf_policy"]),
            "Grid amendment failure is not the preserved converged original RHF solution")
    channel = details["stability"]["channels"]["external_RHF_to_UHF"]
    require(details["stability"]["all_channels_stable"] is False and channel["stable"] is False
            and channel["minimum_eigenvalue"] == eigenvalue,
            "Grid amendment external-instability evidence differs")
    return {"all_passed": True, "path": amendment_path, "file_sha256": sha256_file(path),
            "amendment_id": record["amendment_id"], "preserved_artifacts_verified": len(preserved),
            "preserved_artifacts_sha256": preserved, "failed_attempt_preserved": True,
            "cumulative_failed_attempt_costs": cumulative,
            "prior_amendment_verification": prior,
            "thresholds_and_SCF_method_unchanged": True, "scientific_arrays_loaded": False}


def verify_grid_amendment(workspace):
    """Verify the entire reviewed amendment chain without scientific imports."""
    from .definitions import build_definition

    root = Path(workspace).resolve(strict=True)
    result = _verify_grid_amendment_record(root, GRID_AMENDMENT_PATH, 3)
    record = read_json(workspace_path(root, GRID_AMENDMENT_PATH, namespace="large_qubit/amendments"))
    active = build_definition()
    declaration = active["grid_amendment"]
    require(declaration["amendment_id"] == record["amendment_id"]
            and declaration["evidence_path"] == GRID_AMENDMENT_PATH
            and declaration["prior_amendment_path"] == record["prior_amendment"]["path"]
            and declaration["old_bond_length_angstrom"] == record["old_bond_length_angstrom"]
            and active["geometries"]["CO2_T03"]["bond_length_angstrom"] == record["new_bond_length_angstrom"],
            "Active physical definition differs from the grid amendment")
    return result


def physics_manifest_binding(workspace):
    """Optionally bind numerical evidence without evaluating or loading its arrays.

    A registered absent value cannot later be replaced with results without a new
    registration. Numerical correctness is assessed by the Section 3/4 checks.
    """
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, PHYSICS_MANIFEST_PATH, namespace="large_qubit/physics")
    if not path.exists():
        return None
    sidecar_path = PHYSICS_MANIFEST_PATH.removesuffix(".json") + ".sha256"
    sidecar = workspace_path(root, sidecar_path, namespace="large_qubit/physics")
    actual = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == actual,
            "Realized physics manifest sidecar mismatch")
    from .physics import verify_physics
    verification = verify_physics(root)
    return {"path": PHYSICS_MANIFEST_PATH, "manifest_sha256": actual,
            "sidecar_path": sidecar_path, "sidecar_sha256": sha256_file(sidecar),
            "numeric_arrays_loaded_by_registration": False,
            "numerical_correctness_established_by_registration": False,
            "saved_certificate_and_payload_verification": verification}


def section4_manifest_binding(workspace):
    """Bind an optional saved Section 4 gate without rerunning scientific work."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, SECTION4_MANIFEST_PATH, namespace="large_qubit/validation")
    if not path.exists():
        return None
    sidecar_path = SECTION4_MANIFEST_PATH.removesuffix(".json") + ".sha256"
    sidecar = workspace_path(root, sidecar_path, namespace="large_qubit/validation")
    actual = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == actual,
            "Section 4 gate manifest sidecar mismatch")
    from .section4 import verify_section4
    verification = verify_section4(root, section4_root=path.parent.relative_to(root).as_posix(),
                                   numerical_integrity=False)
    require(verification.get("all_passed") is True,
            "Section 4 gate certificate or payload authentication failed")
    return {"path": SECTION4_MANIFEST_PATH, "manifest_sha256": actual,
            "sidecar_path": sidecar_path, "sidecar_sha256": sha256_file(sidecar),
            "scientific_work_rerun_by_registration": False,
            "saved_certificate_and_payload_verification": verification}


def verify_section4_registration(workspace):
    """Verify archived Sections 1-4 source and unchanged live scientific evidence.

    Section 4's saved gate additionally authenticates its six live scientific
    sources. Archiving their old bytes does not permit changing that live gate
    implementation while continuing to claim its old source identity.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, SECTION4_REGISTRATION)
    require(actual == SECTION4_MANIFEST_SHA256,
            "Section 4 registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section4-registration-v1"
            and saved.get("registration_id") == "sections_1_4_v1"
            and saved.get("execution_ready") is False, "Unexpected Section 4 registration")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        workspace_path(root, relative)
        path = workspace_path(root, f"{SECTION4_ARCHIVE}/{relative}", namespace="large_qubit/archives")
        require(path.is_file(), f"Section 4 archived binding is missing: {relative}")
        require(sha256_file(path) == expected, f"Section 4 archived binding changed: {relative}")
        archived[relative] = expected
    original = verify_original_registration(root)
    section3 = verify_section3_registration(root)
    require(digest(original) == digest(saved["original_registration_lineage"])
            and digest(section3) == digest(saved["section3_registration_lineage"]),
            "Section 4 immutable earlier-registration lineage changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Section 4 pretrained model metadata changed")
    amendment = verify_grid_amendment(root)
    require(digest(amendment) == digest(saved["grid_amendment_verification"]),
            "Section 4 preserved grid-amendment evidence changed")
    physics = physics_manifest_binding(root)
    require(digest(physics) == digest(saved["realized_physics_manifest"]),
            "Section 4 saved physical realization changed")
    gate = section4_manifest_binding(root)
    require(digest(gate) == digest(saved["section4_gate_manifest"]),
            "Section 4 saved gate evidence or live scientific sources changed")
    report = gate["saved_certificate_and_payload_verification"] if gate is not None else {}
    return {"all_passed": True, "registration": SECTION4_REGISTRATION,
            "manifest_sha256": actual, "bound_files_verified": len(archived),
            "source_verification_mode": "archived_sections_1_4_sources_and_live_saved_evidence",
            "active_source_verified_as_original": False,
            "archive_root": SECTION4_ARCHIVE, "archived_bindings_sha256": archived,
            "original_registration_lineage": original, "section3_registration_lineage": section3,
            "historical_reuse_verification": reuse,
            "section_4_saved_gate_verified": gate is not None,
            "section_4_complete": report.get("section_4_complete") is True,
            "section_4_feasibility_gate": report.get("feasibility_gate", "not_measured"),
            "live_section4_scientific_sources_unchanged": gate is not None,
            "numerical_arrays_recomputed_by_registration": False,
            "execution_ready": False, "model_weight_bytes_verified": False,
            "scope": "Archived Sections 1-4 implementation and live unchanged gate sources/evidence, not active Section 5 code"}


def section5_manifest_binding(workspace):
    """Bind optional Section 5 certificates without rerunning scientific work."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, SECTION5_MANIFEST_PATH, namespace="large_qubit/validation")
    if not path.exists():
        return None
    sidecar_path = SECTION5_MANIFEST_PATH.removesuffix(".json") + ".sha256"
    sidecar = workspace_path(root, sidecar_path, namespace="large_qubit/validation")
    actual = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == actual,
            "Section 5 gate manifest sidecar mismatch")
    from .section5 import verify_section5
    verification = verify_section5(root, section5_root=path.parent.relative_to(root).as_posix(),
                                   numerical_integrity=False)
    require(verification.get("all_passed") is True,
            "Section 5 gate certificate or payload authentication failed")
    return {"path": SECTION5_MANIFEST_PATH, "manifest_sha256": actual,
            "sidecar_path": sidecar_path, "sidecar_sha256": sha256_file(sidecar),
            "scientific_work_rerun_by_registration": False,
            "saved_certificate_and_payload_verification": verification}


def verify_section5_registration(workspace):
    """Authenticate archived Sections 1-5 and unchanged live saved science.

    The 74-file archive preserves the old implementation. The Section 4 and 5
    gate verifiers still require their twelve live scientific sources and all
    saved payloads to retain the identities used for those results.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, SECTION5_REGISTRATION)
    require(actual == SECTION5_MANIFEST_SHA256,
            "Section 5 registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section5-registration-v1"
            and saved.get("registration_id") == "sections_1_5_v1"
            and saved.get("execution_ready") is False, "Unexpected Section 5 registration")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        workspace_path(root, relative)
        path = workspace_path(root, f"{SECTION5_ARCHIVE}/{relative}", namespace="large_qubit/archives")
        require(path.is_file(), f"Section 5 archived binding is missing: {relative}")
        require(sha256_file(path) == expected, f"Section 5 archived binding changed: {relative}")
        archived[relative] = expected
    original = verify_original_registration(root)
    section3 = verify_section3_registration(root)
    section4 = verify_section4_registration(root)
    require(digest(original) == digest(saved["original_registration_lineage"])
            and digest(section3) == digest(saved["section3_registration_lineage"])
            and digest(section4) == digest(saved["section4_registration_lineage"]),
            "Section 5 immutable earlier-registration lineage changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Section 5 pretrained model metadata changed")
    amendment = verify_grid_amendment(root)
    require(digest(amendment) == digest(saved["grid_amendment_verification"]),
            "Section 5 preserved grid-amendment evidence changed")
    physics = physics_manifest_binding(root)
    require(digest(physics) == digest(saved["realized_physics_manifest"]),
            "Section 5 saved physical realization changed")
    section4_gate = section4_manifest_binding(root)
    section5_gate = section5_manifest_binding(root)
    require(digest(section4_gate) == digest(saved["section4_gate_manifest"])
            and digest(section5_gate) == digest(saved["section5_gate_manifest"]),
            "Section 5 saved gates, payloads or live scientific sources changed")
    report = section5_gate["saved_certificate_and_payload_verification"] if section5_gate is not None else {}
    return {"all_passed": True, "registration": SECTION5_REGISTRATION,
            "manifest_sha256": actual, "bound_files_verified": len(archived),
            "source_verification_mode": "archived_sections_1_5_sources_and_live_saved_evidence",
            "active_source_verified_as_original": False,
            "archive_root": SECTION5_ARCHIVE, "archived_bindings_sha256": archived,
            "original_registration_lineage": original, "section3_registration_lineage": section3,
            "section4_registration_lineage": section4,
            "historical_reuse_verification": reuse,
            "section_5_saved_gate_verified": section5_gate is not None,
            "section_5_complete": report.get("section_5_complete") is True,
            "section_5_feasibility_gate": report.get("feasibility_gate", "not_measured"),
            "live_section4_and_section5_scientific_sources_unchanged": section4_gate is not None and section5_gate is not None,
            "numerical_arrays_recomputed_by_registration": False,
            "execution_ready": False, "model_weight_bytes_verified": False,
            "scope": "Archived Sections 1-5 implementation and live unchanged scientific sources/evidence, not active Section 6 code"}


def section6_manifest_binding(workspace):
    """Authenticate an optional Section 6 corpus gate without scoring queries."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, SECTION6_MANIFEST_PATH, namespace="large_qubit/validation")
    if not path.exists():
        return None
    sidecar_path = SECTION6_MANIFEST_PATH.removesuffix(".json") + ".sha256"
    sidecar = workspace_path(root, sidecar_path, namespace="large_qubit/validation")
    actual = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == actual,
            "Section 6 gate manifest sidecar mismatch")
    from .section6 import verify_section6
    verification = verify_section6(root, section6_root=path.parent.relative_to(root).as_posix(),
                                   numerical_integrity=False)
    require(verification.get("all_passed") is True,
            "Section 6 gate certificate or payload authentication failed")
    return {"path": SECTION6_MANIFEST_PATH, "manifest_sha256": actual,
            "sidecar_path": sidecar_path, "sidecar_sha256": sha256_file(sidecar),
            "scientific_work_rerun_by_registration": False,
            "saved_certificate_and_payload_verification": verification}


def verify_section6_registration(workspace):
    """Authenticate the archived v1 attempt and its unchanged live negative gate.

    No archived Python is imported. The v1 verifier still authenticates its three
    frozen live sources, while earlier verifiers protect the twelve older ones.
    Authentication of negative evidence does not complete the scientific section.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, SECTION6_REGISTRATION)
    require(actual == SECTION6_MANIFEST_SHA256,
            "Section 6 v1 registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section6-registration-v1"
            and saved.get("registration_id") == "sections_1_6_v1"
            and saved.get("execution_ready") is False, "Unexpected Section 6 v1 registration")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        workspace_path(root, relative)
        path = workspace_path(root, f"{SECTION6_ARCHIVE}/{relative}", namespace="large_qubit/archives")
        require(path.is_file(), f"Section 6 v1 archived binding is missing: {relative}")
        require(sha256_file(path) == expected, f"Section 6 v1 archived binding changed: {relative}")
        archived[relative] = expected
    lineages = {"original_registration_lineage": verify_original_registration(root),
                "section3_registration_lineage": verify_section3_registration(root),
                "section4_registration_lineage": verify_section4_registration(root),
                "section5_registration_lineage": verify_section5_registration(root)}
    require(all(digest(value) == digest(saved[key]) for key, value in lineages.items()),
            "Section 6 v1 immutable earlier-registration lineage changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Section 6 v1 pretrained model metadata changed")
    certificates = {"grid_amendment_verification": verify_grid_amendment(root),
                    "realized_physics_manifest": physics_manifest_binding(root),
                    "section4_gate_manifest": section4_manifest_binding(root),
                    "section5_gate_manifest": section5_manifest_binding(root),
                    "section6_gate_manifest": section6_manifest_binding(root)}
    require(all(digest(value) == digest(saved[key]) for key, value in certificates.items()),
            "Section 6 v1 saved science or live frozen sources changed")
    gate = certificates["section6_gate_manifest"]
    report = gate["saved_certificate_and_payload_verification"] if gate is not None else {}
    return {"all_passed": True, "registration": SECTION6_REGISTRATION,
            "registration_id": "sections_1_6_v1", "manifest_sha256": actual,
            "bound_files_verified": len(archived),
            "source_verification_mode": "archived_sections_1_6_v1_sources_and_live_saved_evidence",
            "active_source_verified_as_original": False,
            "archive_root": SECTION6_ARCHIVE, "archived_bindings_sha256": archived, **lineages,
            "historical_reuse_verification": reuse,
            "section_6_saved_gate_verified": gate is not None,
            "section_6_complete": report.get("section_6_complete") is True,
            "section_6_feasibility_gate": report.get("feasibility_gate", "not_measured"),
            "section_6_release_sha256": report.get("release_sha256"),
            "section_6_production_ordinary_requests": report.get("production_ordinary_requests"),
            "live_section4_section5_and_section6_v1_scientific_sources_unchanged":
                all(certificates[key] is not None for key in
                    ("section4_gate_manifest", "section5_gate_manifest", "section6_gate_manifest")),
            "numerical_arrays_recomputed_by_registration": False,
            "execution_ready": False, "model_weight_bytes_verified": False,
            "scope": "Archived attempted Section6 v1 snapshot and preserved negative evidence, not active v2 code"}


def verify_acquisition_amendment(workspace):
    """Authenticate the user's one-campaign authorization, never authorize work."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, ACQUISITION_AMENDMENT_PATH, namespace="large_qubit/amendments")
    require(sha256_file(path) == ACQUISITION_AMENDMENT_SHA256,
            "Section6 v2 authorization amendment differs from its immutable anchor")
    amendment = read_json(path)
    previous = amendment["previous_attempt"]
    campaign = amendment["prospective_campaign"]
    constraints = amendment["stopping_and_scope"]
    require(amendment["approved"] is True
            and amendment["authorization"]["verbatim_user_answer"] == "Approve one fresh v2 CPU campaign"
            and previous["registration_sha256"] == SECTION6_MANIFEST_SHA256
            and previous["ordinary_requests"] == 6538
            and previous["gate_sha256"] == sha256_file(workspace_path(root, previous["gate_path"],
                                                                         namespace="large_qubit/validation")),
            "Section6 v2 amendment authorization or prior negative evidence differs")
    policy = read_json(workspace_path(root, "large_qubit/section6_policy.json", namespace="large_qubit"))
    require(amendment["unchanged_quality_policy"] == policy["acquisition"]["quality_policy"]
            and campaign["fresh_ordinary_request_ceiling"] == 8192
            and campaign["maximum_cumulative_ordinary_requests"] == previous["ordinary_requests"] + 8192
            and campaign["maximum_wall_seconds"] == 86400
            and campaign["mixture"] == {"guided": .4, "uniform": .2, "local": .4}
            and constraints["authorized_fresh_campaigns"] == 1
            and constraints["automatic_repeated_campaigns"] is False
            and constraints["post_label_lambda_threshold_seed_or_mixture_retuning"] is False,
            "Section6 v2 amendment changes its approved budget or frozen scientific constraints")
    return {"all_passed": True, "path": ACQUISITION_AMENDMENT_PATH,
            "sha256": ACQUISITION_AMENDMENT_SHA256, "approved_fresh_campaigns": 1,
            "prior_attempt_ordinary_requests": 6538, "fresh_ordinary_request_ceiling": 8192,
            "maximum_cumulative_ordinary_requests": 14730, "scientific_quality_thresholds_unchanged": True,
            "execution_started_by_registration": False, "training_authorized_by_amendment": False}


def section6_v2_manifest_binding(workspace):
    """Authenticate optional amended-campaign evidence with zero new queries."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, SECTION6_V2_MANIFEST_PATH, namespace="large_qubit/validation")
    if not path.exists():
        return None
    sidecar_path = SECTION6_V2_MANIFEST_PATH.removesuffix(".json") + ".sha256"
    sidecar = workspace_path(root, sidecar_path, namespace="large_qubit/validation")
    actual = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == actual,
            "Section6 v2 gate manifest sidecar mismatch")
    from .section6_v2 import verify_section6
    verification = verify_section6(root, section6_root=path.parent.relative_to(root).as_posix(),
                                   numerical_integrity=False)
    require(verification.get("all_passed") is True,
            "Section6 v2 gate certificate or payload authentication failed")
    return {"path": SECTION6_V2_MANIFEST_PATH, "manifest_sha256": actual,
            "sidecar_path": sidecar_path, "sidecar_sha256": sha256_file(sidecar),
            "scientific_work_rerun_by_registration": False,
            "saved_certificate_and_payload_verification": verification}


def verify_section6_v2_registration(workspace):
    """Authenticate archived completed Sections1-6 plus unchanged live science.

    The six registration anchors preserve all historical implementation stages.
    Saved v2 sources and released data remain live and immutable for their gate;
    new training implementation files are never treated as old source bytes.
    """
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, SECTION6_V2_REGISTRATION)
    require(actual == SECTION6_V2_MANIFEST_SHA256,
            "Section6 v2 registration manifest differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section6-registration-v2"
            and saved.get("registration_id") == "sections_1_6_v2"
            and saved.get("execution_ready") is False, "Unexpected Section6 v2 registration")
    archived = {}
    for relative, expected in sorted(saved["bindings_sha256"].items()):
        workspace_path(root, relative)
        path = workspace_path(root, f"{SECTION6_V2_ARCHIVE}/{relative}", namespace="large_qubit/archives")
        require(path.is_file(), f"Section6 v2 archived binding is missing: {relative}")
        require(sha256_file(path) == expected, f"Section6 v2 archived binding changed: {relative}")
        archived[relative] = expected
    lineages = {"original_registration_lineage": verify_original_registration(root),
                "section3_registration_lineage": verify_section3_registration(root),
                "section4_registration_lineage": verify_section4_registration(root),
                "section5_registration_lineage": verify_section5_registration(root),
                "section6_v1_registration_lineage": verify_section6_registration(root)}
    require(all(digest(value) == digest(saved[key]) for key, value in lineages.items()),
            "Section6 v2 immutable earlier-registration lineage changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    assets = {key: inspect_assets(root, key, full_hash=False)
              for key in saved["base_asset_metadata_verification"]}
    require(digest(assets) == digest(saved["base_asset_metadata_verification"]),
            "Section6 v2 pretrained model metadata changed")
    certificates = {"grid_amendment_verification": verify_grid_amendment(root),
                    "realized_physics_manifest": physics_manifest_binding(root),
                    "section4_gate_manifest": section4_manifest_binding(root),
                    "section5_gate_manifest": section5_manifest_binding(root),
                    "section6_v1_gate_manifest": section6_manifest_binding(root),
                    "section6_gate_manifest": section6_v2_manifest_binding(root),
                    "section6_acquisition_amendment": verify_acquisition_amendment(root)}
    require(all(digest(value) == digest(saved[key]) for key, value in certificates.items()),
            "Section6 v2 saved release, gates, amendment or live frozen sources changed")
    gate = certificates["section6_gate_manifest"]
    report = gate["saved_certificate_and_payload_verification"] if gate is not None else {}
    return {"all_passed": True, "registration": SECTION6_V2_REGISTRATION,
            "registration_id": "sections_1_6_v2", "manifest_sha256": actual,
            "bound_files_verified": len(archived),
            "source_verification_mode": "archived_sections_1_6_v2_sources_and_live_saved_evidence",
            "active_source_verified_as_original": False,
            "archive_root": SECTION6_V2_ARCHIVE, "archived_bindings_sha256": archived, **lineages,
            "historical_reuse_verification": reuse,
            "section_6_saved_gate_verified": gate is not None,
            "section_6_complete": report.get("section_6_complete") is True,
            "section_6_feasibility_gate": report.get("feasibility_gate", "not_measured"),
            "section_6_release_sha256": report.get("release_sha256"),
            "section_6_production_ordinary_requests": report.get("production_ordinary_requests"),
            "section_6_cumulative_ordinary_requests": report.get("cumulative_ordinary_requests"),
            "live_section4_section5_and_section6_scientific_sources_unchanged":
                all(certificates[key] is not None for key in
                    ("section4_gate_manifest", "section5_gate_manifest", "section6_v1_gate_manifest", "section6_gate_manifest")),
            "numerical_arrays_recomputed_by_registration": False,
            "execution_ready": False, "model_weight_bytes_verified": False,
            "scope": "Archived completed Sections1-6 and unchanged live released science, not active Section7 training code"}


def section7_implementation_binding(workspace):
    """Authenticate implementation readiness; never run production preflights/fit.

    Preparation and the local fixture certificate predate this registration.
    Mutable user-run GPU preflights and training outputs are deliberately outside
    these bindings and cannot change the recorded implementation-only scope.
    """
    root = Path(workspace).resolve(strict=True)
    prep = workspace_path(root, SECTION7_PREPARATION_PATH, namespace="large_qubit/evidence")
    implementation = workspace_path(root, SECTION7_IMPLEMENTATION_PATH, namespace="large_qubit/evidence")
    if not prep.exists() and not implementation.exists():
        return None
    require(prep.is_file() and implementation.is_file(),
            "Section7 readiness needs preparation and its local implementation certificate")
    from .section7 import verify_section7
    verification = verify_section7(root)
    require(verification.get("all_passed") is True and verification.get("implementation_ready") is True
            and verification.get("production_training_executed") is False
            and verification.get("production_gpu_preflight_executed") is False
            and verification.get("production_optimizer_steps_executed") == 0
            and verification.get("model_weights_loaded") is False
            and verification.get("execution_ready") is False,
            "Section7 preparation/cache authentication failed or claims production execution")
    certificate = read_json(implementation)
    require(certificate.get("schema_version") == "section7-implementation-certificate-v1"
            and certificate.get("all_passed") is True and certificate.get("implementation_ready") is True
            and certificate.get("production_training_executed") is False
            and certificate.get("production_gpu_preflight_executed") is False
            and type(certificate.get("production_optimizer_steps_executed")) is int
            and certificate["production_optimizer_steps_executed"] == 0
            and certificate.get("model_weights_loaded") is False
            and type(certificate.get("trained_adapters")) is int and certificate["trained_adapters"] == 0
            and certificate.get("server_connected") is False and certificate.get("execution_ready") is False,
            "Section7 implementation certificate must not claim training, GPU preflights or final readiness")
    unit = certificate["unit_tests"]
    require(unit["all_passed"] is True and type(unit["tests_run"]) is int and unit["tests_run"] > 0
            and type(unit["skipped"]) is int and 0 <= unit["skipped"] < unit["tests_run"]
            and certificate["preparation"]["path"] == SECTION7_PREPARATION_PATH
            and certificate["preparation"]["sha256"] == sha256_file(prep)
            and digest(certificate["preparation"]["verification"]) == digest(verification),
            "Section7 local fixture or preparation certificate identity differs")
    unit_path = workspace_path(root, unit["receipt"], namespace="large_qubit/evidence/sections_1_7_gpu_uuid_v1")
    require(sha256_file(unit_path) == unit["sha256"], "Section7 saved local fixture test receipt changed")
    tested = read_json(unit_path)
    require(tested["all_passed"] is True and tested["exit_code"] == 0
            and tested["tests_run"] == unit["tests_run"]
            and tested["skipped"] == unit["skipped"]
            and tested["production_training_executed"] is False
            and tested["production_gpu_preflight_executed"] is False
            and tested["production_optimizer_steps_executed"] == 0
            and tested["model_weights_loaded"] is False and tested["server_connected"] is False,
            "Section7 saved fixture evidence must authenticate implementation without production work")
    return {"preparation_path": SECTION7_PREPARATION_PATH, "preparation_sha256": sha256_file(prep),
            "implementation_certificate_path": SECTION7_IMPLEMENTATION_PATH,
            "implementation_certificate_sha256": sha256_file(implementation),
            "saved_preparation_and_cache_verification": verification,
            "implementation_ready": True, "production_training_executed": False,
            "production_gpu_preflight_executed": False, "production_optimizer_steps_executed": 0,
            "trained_adapters": 0, "training_not_run_scope": "as_of_saved_implementation_certificate",
            "source_or_data_prepared_by_registration": False, "execution_ready": False}


def _section7_archived_snapshot(workspace, registration, anchor, archive, count):
    """Authenticate an exact legacy inventory without importing its Python."""
    root = Path(workspace).resolve(strict=True)
    manifest, actual = _registration_files(root, registration)
    require(actual == anchor, "Legacy Section7 registration differs from its immutable anchor")
    saved = read_json(manifest)
    require(saved.get("schema_version") == "large-qubit-section7-registration-v1"
            and saved.get("registration_id") == "sections_1_7_v1"
            and saved.get("execution_ready") is False, "Legacy Section7 registration identity differs")
    bindings = saved["bindings_sha256"]
    require(type(bindings) is dict and len(bindings) == count,
            "Legacy Section7 archived binding count differs")
    for relative, expected in bindings.items():
        workspace_path(root, relative)
        archived = workspace_path(root, f"{archive}/{relative}",
                                  namespace="large_qubit/archives")
        require(archived.is_file(), f"Legacy Section7 archived binding is missing: {relative}")
        require(sha256_file(archived) == expected, f"Legacy Section7 archived binding changed: {relative}")
    _verify_section7_saved_preparation(root, saved, archive)
    return root, saved, actual


def _verify_section7_saved_preparation(root, saved, archive):
    """Reconstruct the old deterministic proof from pinned bytes, not old code.

    Bound sources come from the archive. Historical sources omitted from that
    inventory must still exist live with their original hashes. The unchanged
    fast cache reader authenticates the allowlisted native caches and metadata.
    """
    bindings = saved["bindings_sha256"]

    def source(relative, expected):
        workspace_path(root, relative)
        if relative in bindings:
            require(bindings[relative] == expected, "Legacy preparation and archived binding disagree")
            path = workspace_path(root, f"{archive}/{relative}", namespace="large_qubit/archives")
        else:
            path = workspace_path(root, relative)
        require(path.is_file() and sha256_file(path) == expected,
                f"Legacy Section7 prepared source or evidence changed: {relative}")
        return path

    binding = saved["section7_implementation"]
    require(binding is not None and binding["preparation_path"] == SECTION7_INITIAL_PREPARATION_PATH
            and binding["preparation_sha256"] == SECTION7_INITIAL_PREPARATION_SHA256
            and binding["implementation_certificate_path"] == SECTION7_INITIAL_IMPLEMENTATION_PATH,
            "Legacy Section7 preparation identity differs")
    prep_path = source(binding["preparation_path"], binding["preparation_sha256"])
    prep = read_json(prep_path)
    require(prep["schema_version"] == "section7-preparation-v1"
            and prep["production_model_weights_loaded"] is False
            and prep["production_training_executed"] is False
            and prep["production_gpu_preflight_executed"] is False
            and type(prep["energy_requests"]) is int and prep["energy_requests"] == 0
            and prep["prior_registration_sha256"] == SECTION6_V2_MANIFEST_SHA256,
            "Legacy Section7 preparation claims a different scope")
    prior, prior_sha = _registration_files(root, SECTION6_V2_REGISTRATION)
    require(prior_sha == SECTION6_V2_MANIFEST_SHA256, "Legacy Section7 prior registration changed")
    source(prep["policy_path"], prep["policy_sha256"])
    source(prep["deep_native_cache_audit"]["path"], prep["deep_native_cache_audit"]["sha256"])
    require(type(prep["source_bindings"]) is dict and prep["source_bindings"],
            "Legacy Section7 prepared sources are missing")
    for relative, expected in prep["source_bindings"].items():
        source(relative, expected)
    for relative in ("large_qubit/training_data.py", "large_qubit/training_cache_access.py"):
        require(sha256_file(workspace_path(root, relative)) == prep["source_bindings"][relative],
                "Legacy cache authentication requires its unchanged live reader")
    from .training_cache_access import verify_training_cache
    require(set(prep["token_caches"]) == set(MODEL_REVISIONS), "Legacy Section7 requires both native caches")
    checks = {}
    for key, cache in prep["token_caches"].items():
        require(cache["manifest_path"] == f"large_qubit/training_data/section7_v1/{key}/manifest.json",
                "Legacy Section7 cache routing changed")
        source(cache["manifest_path"], cache["manifest_sha256"])
        checks[key] = verify_training_cache(root, cache["manifest_path"], cache["manifest_sha256"],
                                           expected_release_sha256=prep["release_sha256"])
        require(checks[key]["all_passed"] is True, "Legacy Section7 cache authentication failed")
    prepared = {"all_passed": True, "implementation_ready": True,
        "scope": "prepared_fixed_fit_workflow_not_completed_training_or_GPU_preflight",
        "preparation_path": SECTION7_INITIAL_PREPARATION_PATH,
        "preparation_sha256": binding["preparation_sha256"], "policy_sha256": prep["policy_sha256"],
        "token_caches": prep["token_caches"], "cache_verification": checks,
        "source_bindings": prep["source_bindings"], "production_training_executed": False,
        "production_gpu_preflight_executed": False, "production_optimizer_steps_executed": 0,
        "model_weights_loaded": False, "energy_requests": 0, "execution_ready": False}
    require(digest(prepared) == digest(binding["saved_preparation_and_cache_verification"]),
            "Legacy Section7 saved preparation/cache proof differs")
    certificate = read_json(source(binding["implementation_certificate_path"],
                                   binding["implementation_certificate_sha256"]))
    require(certificate["schema_version"] == "section7-implementation-certificate-v1"
            and certificate["all_passed"] is True and certificate["implementation_ready"] is True
            and certificate["production_training_executed"] is False
            and certificate["production_gpu_preflight_executed"] is False
            and type(certificate["production_optimizer_steps_executed"]) is int
            and certificate["production_optimizer_steps_executed"] == 0
            and certificate["model_weights_loaded"] is False and certificate["server_connected"] is False
            and type(certificate["trained_adapters"]) is int and certificate["trained_adapters"] == 0
            and certificate["execution_ready"] is False
            and certificate["preparation"] == {"path": SECTION7_INITIAL_PREPARATION_PATH,
                "sha256": binding["preparation_sha256"], "verification": prepared},
            "Legacy Section7 implementation certificate changed its proof or execution scope")
    unit = certificate["unit_tests"]
    unit_path = source(unit["receipt"], unit["sha256"])
    workspace_path(root, unit["receipt"], namespace="large_qubit/evidence/sections_1_7_v1")
    tested = read_json(unit_path)
    require(unit["all_passed"] is True and tested["all_passed"] is True and tested["exit_code"] == 0
            and type(unit["tests_run"]) is int and tested["tests_run"] == unit["tests_run"] > 0
            and type(unit["skipped"]) is int and 0 <= tested["skipped"] == unit["skipped"] < unit["tests_run"]
            and tested["production_training_executed"] is False
            and tested["production_gpu_preflight_executed"] is False
            and type(tested["production_optimizer_steps_executed"]) is int
            and tested["production_optimizer_steps_executed"] == 0
            and tested["model_weights_loaded"] is False and tested["server_connected"] is False,
            "Legacy Section7 local fixture evidence changed or claims production execution")
    expected = {"preparation_path": SECTION7_INITIAL_PREPARATION_PATH,
        "preparation_sha256": binding["preparation_sha256"],
        "implementation_certificate_path": SECTION7_INITIAL_IMPLEMENTATION_PATH,
        "implementation_certificate_sha256": binding["implementation_certificate_sha256"],
        "saved_preparation_and_cache_verification": prepared, "implementation_ready": True,
        "production_training_executed": False, "production_gpu_preflight_executed": False,
        "production_optimizer_steps_executed": 0, "trained_adapters": 0,
        "training_not_run_scope": "as_of_saved_implementation_certificate",
        "source_or_data_prepared_by_registration": False, "execution_ready": False}
    require(digest(expected) == digest(binding), "Legacy Section7 implementation binding differs")


def verify_section7_initial_registration(workspace):
    """Verify original source/preparation archives and unchanged native cache bytes."""
    root, saved, actual = _section7_archived_snapshot(workspace, SECTION7_INITIAL_REGISTRATION,
        SECTION7_INITIAL_MANIFEST_SHA256, SECTION7_INITIAL_ARCHIVE, 128)
    bindings = saved["bindings_sha256"]
    # Preserve this deterministic report because the anchored docs revision saved it.
    return {"all_passed": True, "registration": SECTION7_INITIAL_REGISTRATION,
            "registration_id": saved["registration_id"], "manifest_sha256": actual,
            "bound_files_verified": len(bindings), "archive_root": SECTION7_INITIAL_ARCHIVE,
            "source_verification_mode": "archived_original_section7_sources_and_live_prepared_inputs",
            "active_source_verified_as_original": False, "live_prepared_inputs_verified": True,
            "section_7_implementation_ready": True, "section_7_status": "IMPLEMENTATION_READY",
            "section_7_training_complete": False, "section_7_training_not_run": True,
            "section_7_training_not_run_scope": "as_of_saved_implementation_certificate",
            "section_7_production_gpu_preflight_verified": False, "section_7_trained_adapters": 0,
            "execution_ready": False,
            "scope": "Original implementation snapshot and unchanged prepared inputs; no GPU execution or fit authentication"}


def section7_documentation_binding(workspace):
    """Bind the authorized GPU-selection instructions without changing fit inputs."""
    root = Path(workspace).resolve(strict=True)
    amendment_path = workspace_path(root, SECTION7_GPU_DOCS_AMENDMENT,
                                    namespace="large_qubit/amendments")
    if not amendment_path.exists():
        return None
    require(sha256_file(amendment_path) == SECTION7_GPU_DOCS_AMENDMENT_SHA256,
            "Section7 documentation amendment differs from its immutable anchor")
    amendment = read_json(amendment_path)
    original = verify_section7_initial_registration(root)
    require(amendment.get("schema_version") == "section7-gpu-selection-documentation-amendment-v1"
            and amendment.get("same_implementation_id") == "sections_1_7_v1"
            and amendment.get("successor_registration") == SECTION7_GPU_DOCS_REGISTRATION
            and amendment.get("scientific_settings_changed") is False
            and amendment.get("training_code_or_policy_changed") is False
            and amendment.get("gpu_execution_authorized_by_this_documentation_request") is False
            and amendment.get("original_registration") == {
                "path": SECTION7_INITIAL_REGISTRATION + "/manifest.json",
                "sha256": SECTION7_INITIAL_MANIFEST_SHA256,
                "bound_files": original["bound_files_verified"], "archive_root": SECTION7_INITIAL_ARCHIVE}
            and amendment.get("training_preparation") == {
                "path": SECTION7_INITIAL_PREPARATION_PATH,
                "sha256": SECTION7_INITIAL_PREPARATION_SHA256},
            "Section7 documentation amendment changes its declared scope or prepared inputs")
    return {"amendment_path": SECTION7_GPU_DOCS_AMENDMENT,
            "amendment_sha256": SECTION7_GPU_DOCS_AMENDMENT_SHA256,
            "original_registration": original, "same_implementation_id": "sections_1_7_v1",
            "scientific_settings_changed": False, "training_code_or_policy_changed": False,
            "gpu_execution_authorized_by_this_documentation_request": False}


def verify_section7_gpu_docs_registration(workspace):
    """Verify the 130-file GPU-selection documentation snapshot as history."""
    root, saved, actual = _section7_archived_snapshot(workspace, SECTION7_GPU_DOCS_REGISTRATION,
        SECTION7_GPU_DOCS_MANIFEST_SHA256, SECTION7_GPU_DOCS_ARCHIVE, 130)
    documentation = section7_documentation_binding(root)
    require(documentation is not None
            and digest(documentation) == digest(saved["section7_documentation_revision"]),
            "Legacy Section7 documentation lineage changed")
    return {"all_passed": True, "registration": SECTION7_GPU_DOCS_REGISTRATION,
        "registration_id": "sections_1_7_v1", "manifest_sha256": actual,
        "bound_files_verified": len(saved["bindings_sha256"]), "archive_root": SECTION7_GPU_DOCS_ARCHIVE,
        "source_verification_mode": "archived_section7_gpu_docs_sources_and_saved_preparation_with_live_cache_authentication",
        "active_source_verified_as_original": False, "archived_preparation_verified": True,
        "section7_documentation_revision": documentation,
        "section_7_implementation_ready": True, "section_7_status": "IMPLEMENTATION_READY",
        "section_7_training_complete": False, "section_7_training_not_run": True,
        "section_7_training_not_run_scope": "as_of_saved_implementation_certificate",
        "section_7_production_gpu_preflight_verified": False, "section_7_trained_adapters": 0,
        "execution_ready": False,
        "scope": "Historical GPU-selection docs snapshot and saved preparation; no current-source or GPU execution claim"}


def section7_uuid_compatibility_binding(workspace):
    """Bind the technical repair and both preserved Section7 implementation stages."""
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, SECTION7_UUID_AMENDMENT, namespace="large_qubit/amendments")
    if not path.exists():
        return None
    require(sha256_file(path) == SECTION7_UUID_AMENDMENT_SHA256,
            "Section7 UUID compatibility amendment differs from its immutable anchor")
    amendment = read_json(path)
    require(amendment["schema_version"] == "section7-h200-uuid-compatibility-amendment-v1"
            and amendment["successor_registration"] == SECTION7_UUID_REGISTRATION
            and amendment["successor_preparation"] == SECTION7_PREPARATION_PATH
            and amendment["successor_implementation_id"] == REGISTRATION_ID
            and amendment["scientific_settings_changed"] is False
            and amendment["data_or_token_caches_changed"] is False
            and amendment["training_objective_changed"] is False and amendment["policy_changed"] is False
            and amendment["new_preflight_and_run_namespace"] == "section7_gpu_uuid_v1"
            and amendment["production_gpu_execution_authorized_by_this_repair"] is False
            and amendment["model_downloads_required"] is False
            and amendment["original_registration"] == {
                "path": SECTION7_GPU_DOCS_REGISTRATION + "/manifest.json",
                "sha256": SECTION7_GPU_DOCS_MANIFEST_SHA256, "bound_files": 130,
                "archive": SECTION7_GPU_DOCS_ARCHIVE}
            and amendment["original_implementation_registration"] == {
                "path": SECTION7_INITIAL_REGISTRATION + "/manifest.json",
                "sha256": SECTION7_INITIAL_MANIFEST_SHA256, "bound_files": 128,
                "archive": SECTION7_INITIAL_ARCHIVE}
            and amendment["original_preparation"] == {
                "path": SECTION7_INITIAL_PREPARATION_PATH, "sha256": SECTION7_INITIAL_PREPARATION_SHA256},
            "Section7 UUID amendment changed its technical-only scope or immutable lineage")
    original = verify_section7_initial_registration(root)
    documentation = verify_section7_gpu_docs_registration(root)
    return {"all_passed": True, "amendment_path": SECTION7_UUID_AMENDMENT,
        "amendment_sha256": SECTION7_UUID_AMENDMENT_SHA256,
        "original_implementation_registration": original, "original_registration": documentation,
        "successor_registration": SECTION7_UUID_REGISTRATION, "successor_implementation_id": REGISTRATION_ID,
        "successor_preparation": SECTION7_PREPARATION_PATH,
        "scientific_settings_changed": False, "data_or_token_caches_changed": False,
        "training_objective_changed": False, "policy_changed": False,
        "production_gpu_execution_authorized_by_this_repair": False,
        "model_downloads_required": False, "execution_ready": False}


def make_registration(workspace):
    root = Path(workspace).resolve(strict=True)
    lineage = verify_original_registration(root)
    section3_lineage = verify_section3_registration(root)
    section4_lineage = verify_section4_registration(root)
    section5_lineage = verify_section5_registration(root)
    section6_lineage = verify_section6_registration(root)
    section6_v2_lineage = verify_section6_v2_registration(root)
    acquisition_amendment = verify_acquisition_amendment(root)
    physical_definition = validate_physical_definition(root)
    grid_amendment = verify_grid_amendment(root)
    realized_physics = physics_manifest_binding(root)
    section4_gate = section4_manifest_binding(root)
    section5_gate = section5_manifest_binding(root)
    section6_v1_gate = section6_manifest_binding(root)
    section6_gate = section6_v2_manifest_binding(root)
    section7 = section7_implementation_binding(root)
    documentation = section7_documentation_binding(root)
    compatibility = section7_uuid_compatibility_binding(root)
    study = validate_study(read_json(root / "large_qubit/study.json"))
    inventory = build_reuse_inventory(root)
    verify_reuse_inventory(inventory, root)
    assets = {key: inspect_assets(root, key, full_hash=False) for key in MODEL_REVISIONS}
    components = {role: {"reuse_mode": "reference_for_new_adaptation", "co2_ready": False,
                         "sources_sha256": {path: sha256_file(workspace_path(root, path)) for path in paths}}
                  for role, paths in REUSABLE_COMPONENTS.items()}
    document = {
        "schema_version": "large-qubit-section7-registration-v2",
        "registration_id": REGISTRATION_ID,
        "registration_kind": "section7_implementation_readiness_not_production_training_or_final_freeze",
        "benchmark_id": study["benchmark_id"], "study": study,
        "bindings_sha256": current_bindings(root),
        "reuse_inventory": inventory, "base_asset_metadata_verification": assets,
        "original_registration_lineage": lineage,
        "section3_registration_lineage": section3_lineage,
        "section4_registration_lineage": section4_lineage,
        "section5_registration_lineage": section5_lineage,
        "section6_v1_registration_lineage": section6_lineage,
        "section6_v2_registration_lineage": section6_v2_lineage,
        "section6_acquisition_amendment": acquisition_amendment,
        "physical_definition": physical_definition,
        "grid_amendment_verification": grid_amendment,
        "realized_physics_manifest": realized_physics,
        "section4_gate_manifest": section4_gate,
        "section5_gate_manifest": section5_gate,
        "section6_gate_manifest": section6_gate,
        "section6_v1_gate_manifest": section6_v1_gate,
        "section7_implementation": section7,
        "reusable_components": components,
        "model_weight_bytes_authenticated_by_this_registration": False,
        "weight_authentication_policy": "full_tensor_hash_required_before_model_load; tokenizer_only_checks_authenticate_metadata_and_templates",
        "dependency_scope": "isolated_chemistry_requirements_and_recorded_backend_bound_training_environment_pending",
        "execution_ready": False, "pending_scientific_bindings": {name: None for name in PENDING},
    }
    if documentation is not None:
        document["section7_documentation_revision"] = documentation
    if compatibility is not None:
        document["section7_uuid_compatibility_amendment"] = compatibility
        document["section7_original_registration_lineage"] = compatibility["original_implementation_registration"]
        document["section7_gpu_docs_registration_lineage"] = compatibility["original_registration"]
    return document


def register(workspace, output=DEFAULT_REGISTRATION):
    """Explicit one-time write. Never overwrite a registration or old artifacts."""
    root = Path(workspace).resolve(strict=True)
    folder = workspace_path(root, output, namespace="large_qubit/registrations")
    require(folder != root / "large_qubit/registrations", "Choose a registration subdirectory")
    require(not folder.exists(), "Registration already exists; verify it instead of overwriting")
    require(output not in (ORIGINAL_REGISTRATION, SECTION3_REGISTRATION, SECTION4_REGISTRATION,
                           SECTION5_REGISTRATION, SECTION6_REGISTRATION, SECTION6_V2_REGISTRATION,
                           SECTION7_INITIAL_REGISTRATION, SECTION7_GPU_DOCS_REGISTRATION, SECTION7_UUID_REGISTRATION),
            "An immutable historical registration cannot be recreated")
    if output == DEFAULT_REGISTRATION and (root / SECTION7_GUIDE_AMENDMENT).exists():
        from .documentation_registration import make_documentation_registration
        document = make_documentation_registration(root)
    else:
        document = make_registration(root)
    # mkdir(exist_ok=False) is the final no-clobber check if another writer races.
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    try:
        manifest = folder / "manifest.json"
        with manifest.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        with (folder / "manifest.sha256").open("x", encoding="ascii", newline="\n") as handle:
            handle.write(sha256_file(manifest) + "\n")
    except BaseException:
        # Remove only the directory created by this call, after validating scope.
        require(folder.resolve().is_relative_to((root / "large_qubit/registrations").resolve()),
                "Refuse cleanup outside registrations")
        shutil.rmtree(folder)
        raise
    return {"registered": True, "path": folder.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest), "execution_ready": False}


def verify_registration(workspace, registration=DEFAULT_REGISTRATION):
    root = Path(workspace).resolve(strict=True)
    if registration == ORIGINAL_REGISTRATION:
        return verify_original_registration(root)
    if registration == SECTION3_REGISTRATION:
        return verify_section3_registration(root)
    if registration == SECTION4_REGISTRATION:
        return verify_section4_registration(root)
    if registration == SECTION5_REGISTRATION:
        return verify_section5_registration(root)
    if registration == SECTION6_REGISTRATION:
        return verify_section6_registration(root)
    if registration == SECTION6_V2_REGISTRATION:
        return verify_section6_v2_registration(root)
    if registration == SECTION7_INITIAL_REGISTRATION:
        return verify_section7_initial_registration(root)
    if registration == SECTION7_GPU_DOCS_REGISTRATION:
        return verify_section7_gpu_docs_registration(root)
    if registration == SECTION7_UUID_REGISTRATION:
        from .documentation_registration import verify_predecessor_registration
        return verify_predecessor_registration(root)
    manifest, expected_digest = _registration_files(root, registration)
    saved = read_json(manifest)
    if "section7_guide_documentation_revision" in saved:
        from .documentation_registration import verify_documentation_registration
        return verify_documentation_registration(root, registration)
    # Rebuild from current source/data/metadata, never trust a rewritten sidecar alone.
    expected = make_registration(root)
    require(digest(saved) == digest(expected),
            "Registered decisions, source, dependency declarations, model metadata or historical artifacts changed")
    reuse = verify_reuse_inventory(saved["reuse_inventory"], root)
    realized = expected["realized_physics_manifest"]
    gate = expected["section4_gate_manifest"]
    gate_report = gate["saved_certificate_and_payload_verification"] if gate is not None else {}
    correctness = gate_report.get("correctness_passed") is True
    reference = gate_report.get("reference_converged") is True
    section5 = expected["section5_gate_manifest"]
    section5_report = section5["saved_certificate_and_payload_verification"] if section5 is not None else {}
    section6 = expected["section6_gate_manifest"]
    section6_report = section6["saved_certificate_and_payload_verification"] if section6 is not None else {}
    section7 = expected["section7_implementation"]
    return {"all_passed": True, "registration": registration,
            "registration_id": saved["registration_id"],
            "manifest_sha256": expected_digest, "bound_files_verified": len(saved["bindings_sha256"]),
            "source_verification_mode": "current_section7_uuid_compatible_sources_with_eight_anchored_registration_archives",
            "original_registration_lineage": saved["original_registration_lineage"],
            "section3_registration_lineage": saved["section3_registration_lineage"],
            "section4_registration_lineage": saved["section4_registration_lineage"],
            "section5_registration_lineage": saved["section5_registration_lineage"],
            "section6_v1_registration_lineage": saved["section6_v1_registration_lineage"],
            "section6_v2_registration_lineage": saved["section6_v2_registration_lineage"],
            "section6_acquisition_amendment": saved["section6_acquisition_amendment"],
            "section7_documentation_revision": expected.get("section7_documentation_revision"),
            "section7_original_registration_lineage": expected.get("section7_original_registration_lineage"),
            "section7_gpu_docs_registration_lineage": expected.get("section7_gpu_docs_registration_lineage"),
            "section7_uuid_compatibility_amendment": expected.get("section7_uuid_compatibility_amendment"),
            "section_3_physical_definition_verified": True,
            "section_3_complete": realized is not None,
            "saved_section_3_physical_certificates_verified": realized is not None,
            "numerical_arrays_recomputed_by_registration": False,
            "section_4_saved_gate_verified": gate is not None,
            "section_4_complete": gate_report.get("section_4_complete") is True,
            "section_4_feasibility_gate": gate_report.get("feasibility_gate", "not_measured"),
            "reference_and_circuit_evaluator_validated": correctness and reference,
            "section_5_saved_gate_verified": section5 is not None,
            "section_5_complete": section5_report.get("section_5_complete") is True,
            "section_5_feasibility_gate": section5_report.get("feasibility_gate", "not_measured"),
            "section_6_saved_gate_verified": section6 is not None,
            "section_6_complete": section6_report.get("section_6_complete") is True,
            "section_6_feasibility_gate": section6_report.get("feasibility_gate", "not_measured"),
            "section_6_release_sha256": section6_report.get("release_sha256"),
            "section_6_production_ordinary_requests": section6_report.get("production_ordinary_requests"),
            "section_6_prior_failed_attempt_ordinary_requests": 6538,
            "section_6_cumulative_ordinary_requests":
                None if section6 is None else 6538 + section6_report.get("production_ordinary_requests", 0),
            "section_7_implementation_ready": section7 is not None,
            "section_7_status": "IMPLEMENTATION_READY" if section7 is not None else "PENDING",
            "section_7_training_complete": False,
            "section_7_training_not_run": True,
            "section_7_training_not_run_scope": "as_of_saved_implementation_certificate",
            "section_7_production_gpu_preflight_verified": False,
            "section_7_trained_adapters": 0,
            "historical_reuse_verification": reuse,
            "model_weight_bytes_verified": False, "execution_ready": False,
            "scope": "Section7 implementation snapshot with preserved completed Sections1-6; no production training or final experiment freeze"}


def status(workspace, registration=DEFAULT_REGISTRATION):
    verification = verify_registration(workspace, registration)
    study = build_study()
    return {"benchmark_id": study["benchmark_id"], "foundation_verified": verification["all_passed"],
            "registration": registration,
            "source_verification_mode": verification["source_verification_mode"],
            "section_3_physical_definition_verified": verification.get("section_3_physical_definition_verified", False),
            "section_3_complete": verification.get("section_3_complete", False),
            "saved_section_3_physical_certificates_verified": verification.get("saved_section_3_physical_certificates_verified", False),
            "section_4_saved_gate_verified": verification.get("section_4_saved_gate_verified", False),
            "section_4_complete": verification.get("section_4_complete", False),
            "section_4_feasibility_gate": verification.get("section_4_feasibility_gate", "not_measured"),
            "reference_and_circuit_evaluator_validated": verification.get("reference_and_circuit_evaluator_validated", False),
            "section_5_saved_gate_verified": verification.get("section_5_saved_gate_verified", False),
            "section_5_complete": verification.get("section_5_complete", False),
            "section_5_feasibility_gate": verification.get("section_5_feasibility_gate", "not_measured"),
            "section_6_saved_gate_verified": verification.get("section_6_saved_gate_verified", False),
            "section_6_complete": verification.get("section_6_complete", False),
            "section_6_feasibility_gate": verification.get("section_6_feasibility_gate", "not_measured"),
            "section_6_release_sha256": verification.get("section_6_release_sha256"),
            "section_6_production_ordinary_requests": verification.get("section_6_production_ordinary_requests"),
            "section_7_implementation_ready": verification.get("section_7_implementation_ready", False),
            "section_7_status": verification.get("section_7_status", "PENDING"),
            "section_7_training_complete": False,
            "section_7_training_not_run": True,
            "section_7_training_not_run_scope": "as_of_saved_implementation_certificate",
            "section_7_production_gpu_preflight_verified": False,
            "section_7_trained_adapters": 0,
            "final_protocol_frozen": False, "execution_ready": False,
            "planned_fits": len(study["training_conditions"]),
            "planned_methods": len(study["required_methods"]),
            "planned_target_trajectories": len(planned_jobs(study)),
            "maximum_main_logical_requests_including_hf": study["budget"]["main_logical_requests"],
            "pending": list(PENDING),
            "next_action": "User-run H200 memory/throughput preflights and three fresh fits; no production training executed by implementation checks",
            "historical_lih_phase7_convergence_passed": False,
            "historical_audit_replayed": False,
            "cost_reporting": "Keep historical_lih separate from gqe-co2-sto3g-30q-v1",
            "model_weight_bytes_verified_by_status": False}
