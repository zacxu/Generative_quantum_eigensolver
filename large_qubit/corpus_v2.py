"""Fresh v2 CO2 release with preserved v1 costs and isolated consumer views.

Release authentication establishes reproducibility against a caller's trusted
SHA256 pin; it is not an OS permission boundary or external signature. No
Hamiltonian, reference, model, Torch or optimizer is loaded by this module.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType

from .baselines import ROW_FIELDS
from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .definitions import require_geometry_access
from .grammar import _checked_values, operator_limit
from .protocol import BENCHMARK_ID, FITS


TRAINING = ("CO2_T01", "CO2_T02", "CO2_T03")
ELIGIBLE_ROW_FIELDS = frozenset(ROW_FIELDS | {"request_ordinal"})
RELEASE_ROOT = "large_qubit/releases/co2_corpus_v2"
CORPUS_ROOT = "large_qubit/corpora/co2_corpus_v2"
RELEASE_PATH = RELEASE_ROOT + "/manifest.json"
POLICY_PATH = "large_qubit/section6_v2_policy.json"
PRIVATE_ROOT = f"large_qubit/scoring_only/{BENCHMARK_ID}/section6/section6_v2"
IQR_FLOOR_HA = 1e-6
CONSUMERS = (*FITS, "training_circuit_retrieval")
SOURCE_PATHS = ("large_qubit/section6_v2.py", "large_qubit/acquisition_v2.py", "large_qubit/corpus_v2.py",
                "phase6/objective.py")
SECTION5_SOURCES = tuple(f"large_qubit/{name}.py" for name in
                        ("catalogue", "prompts", "grammar", "baselines", "vqe", "section5"))
_SHA = re.compile(r"[0-9a-f]{64}")
PRIOR_GATE_PATH = "large_qubit/validation/section6_v1/manifest.json"
PRIOR_GATE_SHA256 = "cdf23f3493d1c583cbf4e46126412416ea0afe295247f699f1a4305845693737"
PRIOR_REVIEW_PATH = "large_qubit/evidence/sections_1_6_v1/negative_gate_review.json"
PRIOR_REVIEW_SHA256 = "3f2fbc29eb42a1604f25580529281463613b66cafc4ca0992a70d8bbb02410f3"
PRIOR_ORDINARY_REQUESTS = 6538
CUMULATIVE_REQUEST_CEILING = 14730
AMENDMENT_PATH = "large_qubit/amendments/section6_acquisition_v2.json"


def _number(value, name, *, minimum=None):
    require(type(value) in (int, float), f"Invalid {name}")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error
    require(math.isfinite(result) and (minimum is None or result >= minimum), f"Invalid {name}")
    return result


def _integer(value, name, *, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            f"Invalid {name}")
    return value


def _hash(value, name):
    require(type(value) is str and _SHA.fullmatch(value), f"Invalid {name} SHA256")
    return value


@dataclass(frozen=True)
class GeometryWeights:
    row_ids: tuple
    weights: tuple
    multipliers: tuple
    _report: dict

    @property
    def report(self):
        return deepcopy(self._report)


def geometry_weights(rows, *, mode, lambda_scale=None):
    """Port Phase6's linear-quantile exponential/IQR rule to CO2 training rows.

    Supply one complete released fitting geometry in release order. Diagnostics,
    mixed geometries, duplicate IDs and nonfinite labels fail without filtering.
    Uniform row sampling is unchanged; ``n_g*w_gi`` enters the loss once only.
    The report contains private training-label statistics and must not be copied
    wholesale into public summaries or model prompts.
    """
    import numpy as np
    require(not isinstance(rows, (str, bytes, Mapping)), "A complete fitting row sequence is required")
    rows = list(rows)
    if not rows or any(not isinstance(row, Mapping) or row.get("role") != "fitting" for row in rows):
        raise PermissionError("Weights accept fitting rows only, without diagnostic filtering")
    require(all(set(row) in (set(ROW_FIELDS), set(ELIGIBLE_ROW_FIELDS)) for row in rows),
            "Unexpected fitting row fields")
    aliases = {row.get("geometry_id") for row in rows}
    require(len(aliases) == 1, "Compute weights within one complete fitting geometry")
    alias = next(iter(aliases))
    require_geometry_access(alias, "weighting")
    ids = tuple(row.get("row_id") for row in rows)
    require(all(type(value) is str and value for value in ids) and len(set(ids)) == len(ids),
            "Unique complete fitting row IDs are required")
    catalogues = {row.get("catalogue_sha256") for row in rows}
    require(len(catalogues) == 1, "Mixed fitting catalogues")
    _hash(next(iter(catalogues)), "fitting catalogue")
    energies = np.asarray([_number(row.get("energy_hartree"), "fitting label") for row in rows], dtype=np.float64)
    require(mode in ("weighted", "uniform"), "Unknown objective mode")
    q25, q75 = np.quantile(energies, [.25, .75], method="linear")
    minimum, iqr = float(np.min(energies)), float(q75 - q25)
    _number(iqr, "fitting IQR", minimum=0.)
    dispersion = max(iqr, IQR_FLOOR_HA)
    n = len(rows)
    if mode == "weighted":
        scale = _number(lambda_scale, "explicit positive weighting lambda", minimum=0.)
        require(scale > 0, "Weighted training requires a positive lambda")
        tau = scale * dispersion
        require(math.isfinite(tau) and tau > 0, "Invalid or underflowed training temperature")
        with np.errstate(under="ignore"):
            unnormalized = np.exp(-(energies - minimum) / tau)
        normalizer = float(unnormalized.sum())
        require(math.isfinite(normalizer) and normalizer > 0, "Invalid exponential normalizer")
        weights = unnormalized / normalizer
    else:
        require(lambda_scale is None, "Uniform SFT has no energy-weighting lambda")
        tau, normalizer = None, None
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    multipliers = n * weights
    ess = float(1.0 / np.dot(weights, weights))
    report = {"alias": alias, "pipeline_seed": 22, "population": "fitting_only", "mode": mode,
              "rows": n, "lambda_scale": None if mode == "uniform" else float(lambda_scale),
              "fitting_minimum_energy_ha": minimum, "fitting_iqr_ha": iqr,
              "iqr_floor_ha": IQR_FLOOR_HA, "dispersion_ha": dispersion, "tau_ha": tau,
              "shifted_exponential_normalizer": normalizer, "effective_sample_size": ess,
              "ess_fraction": ess / n, "maximum_weight": float(np.max(weights)),
              "zero_weights_from_underflow": int(np.count_nonzero(weights == 0)),
              "ess_below_descriptive_5_percent": bool(ess < .05 * n),
              "lambda_selected_automatically": False, "rows_filtered": 0,
              "sampling": "Uniform geometry visits and uniform shuffled rows; no weight-proportional sampling",
              "multiplier": "n_g*w_gi exactly once; no sum-of-weights minibatch divisor",
              "interpretation": "ESS is stored-row weight concentration, not independent information"}
    return GeometryWeights(ids, tuple(map(float, weights)), tuple(map(float, multipliers)), report)


def _validated_rows(rows, catalogue):
    maximum = operator_limit(catalogue)
    require(type(rows) in (list, tuple), "An eligible acquisition-row sequence is required")
    result, ids, ordinals = [], set(), set()
    circuits = {alias: set() for alias in TRAINING}
    for row in rows:
        require(type(row) is dict and set(row) == set(ELIGIBLE_ROW_FIELDS), "Invalid eligible-row schema")
        alias = row["geometry_id"]
        require_geometry_access(alias, "acquisition")
        require(row["role"] in ("fitting", "diagnostic"), "Only eligible fitting/diagnostic rows are releasable")
        require(type(row["row_id"]) is str and row["row_id"] and row["row_id"] not in ids,
                "Eligible row IDs must be globally unique")
        marker = "F" if row["role"] == "fitting" else "D"
        require(re.fullmatch(re.escape(alias) + "_v2_" + marker + r"[0-9]{6}", row["row_id"])
                and 1 <= int(row["row_id"][-6:]) <= 8192,
                "Only fresh role-specific v2 acquisition row identities are eligible")
        ordinal = _integer(row["request_ordinal"], "global actual request ordinal", minimum=4, maximum=8192)
        require(ordinal not in ordinals, "Two eligible rows cannot claim one actual request")
        ids.add(row["row_id"]); ordinals.add(ordinal)
        require(row["catalogue_sha256"] == catalogue["sha256"], "Eligible row catalogue differs")
        ops = _checked_values(row["ops"], maximum=maximum, length=32)
        key = tuple(ops)
        require(key not in circuits[alias], "Diagnostic/fitting overlap or duplicate eligible circuit")
        circuits[alias].add(key)
        _number(row["energy_hartree"], "eligible training label")
        result.append(deepcopy(row))
    return sorted(result, key=lambda row: row["request_ordinal"])


def select_balanced_rows(rows, *, catalogue, stop_reason):
    """Select the first balanced acquisition-order prefix, never an energy rank.

    All eligible raw fitting rows are returned, including poor or parent-rejected
    local proposals. Quality assessment follows prefix selection independently;
    failed quality cannot trigger a smaller release.
    """
    require(stop_reason in ("target", "budget", "schedule"), "Unknown acquisition stop reason")
    ordered = _validated_rows(rows, catalogue)
    fitting = {alias: [row for row in ordered if row["geometry_id"] == alias and row["role"] == "fitting"]
               for alias in TRAINING}
    diagnostic = {alias: [row for row in ordered if row["geometry_id"] == alias and row["role"] == "diagnostic"]
                  for alias in TRAINING}
    require(all(len(diagnostic[alias]) == 128 for alias in TRAINING), "Reserve exactly128 diagnostics per geometry")
    require(all(fitting[alias] for alias in TRAINING), "Every training geometry requires fitting rows")
    require(max(row["request_ordinal"] for group in diagnostic.values() for row in group)
            < min(row["request_ordinal"] for group in fitting.values() for row in group),
            "All diagnostics must finish before any fitting acquisition")
    if all(len(fitting[alias]) >= 2048 for alias in TRAINING):
        count = 2048
    else:
        require(stop_reason in ("budget", "schedule") and all(len(fitting[alias]) >= 1024 for alias in TRAINING),
                "A balanced1024 prefix requires budget/schedule prevention of the full2048 release")
        count = 1024
    return {"rows_per_geometry": count, "stop_reason": stop_reason,
            "fitting_by_geometry": {alias: fitting[alias][:count] for alias in TRAINING},
            "diagnostic_by_geometry": diagnostic, "raw_fitting_by_geometry": fitting}


def _read_rows(path):
    import json
    from .common import _pairs
    def reject(_):
        raise ValueError("Nonfinite corpus row JSON")
    data = Path(path).read_bytes()
    require(not data or data.endswith(b"\n"), "Frozen corpus JSONL has an incomplete final line")
    result = []
    for line in data.decode("ascii").splitlines():
        require(bool(line), "Blank corpus JSONL record")
        result.append(json.loads(line, object_pairs_hook=_pairs, parse_constant=reject))
    return result


def _descriptor(root, relative, *, namespace):
    path = workspace_path(root, relative, namespace=namespace)
    require(path.is_file(), f"Missing bound artifact: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _descriptor_shape(root, value, *, namespace):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"}, "Invalid artifact descriptor")
    _integer(value["bytes"], "artifact bytes")
    _hash(value["sha256"], "artifact")
    return workspace_path(root, value["path"], namespace=namespace)


def _check_descriptor(root, value, *, namespace):
    _descriptor_shape(root, value, namespace=namespace)
    require(value == _descriptor(root, value["path"], namespace=namespace), "Bound artifact changed")
    return workspace_path(root, value["path"], namespace=namespace)


def _bindings(root, policy_path):
    paths = {"section5_gate": "large_qubit/validation/section5_v1/manifest.json",
             "section5_preparation": "large_qubit/evidence/sections_1_5_v1/preparation.json",
             "section5_policy": "large_qubit/section5_policy.json",
             "catalogue": "large_qubit/catalogues/co2_screened_v1.json",
             "physical_definition": "large_qubit/physical_definition.json",
             "section3_physics": "large_qubit/physics/section3_v3/manifest.json",
             "acquisition_amendment": AMENDMENT_PATH,
             "section6_policy": policy_path}
    require(policy_path == POLICY_PATH, "Use the declared Section6 policy path")
    return {"source_sha256": {name: sha256_file(workspace_path(root, name)) for name in SOURCE_PATHS},
            "artifacts": {key: _descriptor(root, relative, namespace="large_qubit")
                          for key, relative in paths.items()}}


def _validate_bindings(root, bindings, catalogue):
    require(type(bindings) is dict and set(bindings) == {"source_sha256", "artifacts"}, "Invalid release bindings")
    require(canonical_bytes(bindings) == canonical_bytes(_bindings(root, POLICY_PATH)),
            "Release policy, source or parent lineage changed")
    artifacts = bindings["artifacts"]
    require(canonical_bytes(read_json(root / artifacts["catalogue"]["path"])) == canonical_bytes(catalogue),
            "Release catalogue differs from its frozen file")
    gate = read_json(root / artifacts["section5_gate"]["path"])
    prep = read_json(root / artifacts["section5_preparation"]["path"])
    physics = read_json(root / artifacts["section3_physics"]["path"])
    require(type(gate.get("source_bindings")) is dict and set(gate["source_bindings"]) == set(SECTION5_SOURCES)
            and canonical_bytes(gate["source_bindings"]) == canonical_bytes(prep.get("source_bindings")),
            "Frozen Section5 dependency-source declarations differ")
    for relative, fingerprint in gate["source_bindings"].items():
        _hash(fingerprint, "frozen Section5 dependency")
        require(sha256_file(workspace_path(root, relative, namespace="large_qubit")) == fingerprint,
                "Frozen Section5 dependency source changed")
    require(prep.get("reused_grammar_sha256") == sha256_file(root / "phase6/grammar.py"),
            "Preserved generic answer grammar changed")
    require(gate["section_5_complete"] is True and gate["feasibility_gate"] == "go"
            and gate["decision"]["selected_length"] == 32 and gate["catalogue_sha256"] == catalogue["sha256"]
            and prep["catalogue_sha256"] == catalogue["sha256"]
            and prep["candidate_energy_requests"] == 0 and prep["model_training_executed"] is False,
            "A released corpus requires the completed frozen Section5 representation")
    require(physics["all_passed"] is True and physics["definition_sha256"]
            == digest(read_json(root / artifacts["physical_definition"]["path"])), "Physical-definition lineage differs")
    require(catalogue["training_hamiltonian_bindings"] == {
        alias: physics["geometries"][alias]["hamiltonian_fingerprint"] for alias in TRAINING},
        "Released training Hamiltonians differ from physical realization")


def _quality_scope(report, count, catalogue):
    from .acquisition_v2 import acquisition_policy, quality_policy
    require(type(report) is dict and report.get("all_passed") is True
            and report.get("benchmark_id") == BENCHMARK_ID
            and type(report.get("rows_per_geometry")) is int and report["rows_per_geometry"] == count
            and report.get("catalogue_sha256") == catalogue["sha256"]
            and report.get("policy_sha256") == digest(acquisition_policy(catalogue, length=32))
            and report.get("quality_policy") == quality_policy()
            and report.get("diagnostic_energies_used_for_quality") is False
            and report.get("heldout_or_reference_access") is False
            and report.get("model_training_executed") is False,
            "Release quality must pass the predeclared rule on this exact balanced prefix")
    require(type(report.get("groups")) is dict and set(report["groups"]) == set(TRAINING),
            "Quality must describe every training geometry")
    for group in report["groups"].values():
        require(type(group) is dict and group.get("passed") is True
                and type(group.get("checks")) is dict and group["checks"]
                and all(type(value) is bool and value for value in group["checks"].values()),
                "A failed quality result cannot become a released corpus")
    from .section4 import _reject_public_energy_values
    _reject_public_energy_values(report)


def _quality_check(report, selection, catalogue):
    _quality_scope(report, selection["rows_per_geometry"], catalogue)
    for alias in TRAINING:
        rows = selection["fitting_by_geometry"][alias]
        group = report["groups"][alias]
        require(group.get("passed") is True and type(group.get("checks")) is dict
                and group["checks"] and all(type(value) is bool and value for value in group["checks"].values())
                and type(group.get("fitting_rows")) is int and group["fitting_rows"] == len(rows)
                and type(group.get("diagnostic_rows_reserved")) is int and group["diagnostic_rows_reserved"] == 128
                and group.get("fitting_row_order_sha256") == digest([
                    [row["request_ordinal"], row["row_id"], row["ops"], row["energy_hartree"]] for row in rows]),
                "Quality result is not bound to the selected acquisition-order rows")


def _weighting_spec():
    return {"source": "phase6/objective.py:geometry_weights", "iqr_floor_ha": IQR_FLOOR_HA,
            "quantiles": "numpy_linear_25_75", "GPT_weighted_lambda": .25,
            "Qwen_weighted_lambda": .10, "uniform_multiplier": 1,
            "multiplier": "n_g*w_gi_once", "weighted_sampling": False,
            "population": "released_fitting_prefix_only"}


def _cost(value):
    require(type(value) is dict and set(value) == {"logical_energy_requests", "wall_seconds"},
            "Record ordinary acquisition requests and wall time separately")
    _integer(value["logical_energy_requests"], "all-started ordinary acquisition requests", minimum=3, maximum=8192)
    require(_number(value["wall_seconds"], "acquisition wall time", minimum=0.) > 0,
            "Acquisition wall time must be positive")
    return deepcopy(value)


def _verification_cost(value):
    value = {"hamiltonian_actions": 0, "wall_seconds": 0.0} if value is None else value
    require(type(value) is dict and set(value) == {"hamiltonian_actions", "wall_seconds"},
            "Record verification H actions and wall time explicitly")
    _integer(value["hamiltonian_actions"], "separate verification H actions")
    _number(value["wall_seconds"], "separate verification wall time", minimum=0.)
    return deepcopy(value)


def prior_attempt_cost(workspace):
    """Authenticate public v1 failure provenance without reading its labels.

    The failed attempt remains a cost of the amended experiment. Its energies,
    circuits, diagnostic payloads and acquisition parents are never opened here.
    Trusted immutable pins also prevent a caller from inventing cheaper history.
    """
    root = Path(workspace).resolve(strict=True)
    gate_path = workspace_path(root, PRIOR_GATE_PATH, namespace="large_qubit/validation")
    review_path = workspace_path(root, PRIOR_REVIEW_PATH, namespace="large_qubit/evidence")
    require(sha256_file(gate_path) == PRIOR_GATE_SHA256
            and gate_path.with_suffix(".sha256").read_text(encoding="ascii").strip() == PRIOR_GATE_SHA256
            and sha256_file(review_path) == PRIOR_REVIEW_SHA256,
            "Prior failed-attempt gate or independent review changed")
    gate, review = read_json(gate_path), read_json(review_path)
    require(gate["benchmark_id"] == BENCHMARK_ID and gate["feasibility_gate"] == "no_go"
            and gate["section_6_complete"] is False and gate["release"] is None
            and type(gate["ordinary_requests"]) is int and gate["ordinary_requests"] == PRIOR_ORDINARY_REQUESTS
            and gate["completed_requests"] == PRIOR_ORDINARY_REQUESTS
            and gate["failed_requests"] == gate["abandoned_requests"] == 0
            and gate["separate_integrity_hamiltonian_actions"] == 0
            and gate["heldout_energy_or_reference_access"] is False
            and gate["model_training_executed"] is False,
            "Expected the immutable completed v1 negative acquisition attempt")
    wall = _number(gate["recorded_active_wall_seconds"], "prior active acquisition wall", minimum=0.)
    require(wall > 0 and review["audit_checks_passed"] is True
            and review["scientific_gate"] == "no_go" and review["section_6_complete"] is False
            and review["saved_gate"] == {"path": PRIOR_GATE_PATH, "sha256": PRIOR_GATE_SHA256}
            and review["costs"]["ordinary_logical_requests"] == PRIOR_ORDINARY_REQUESTS
            and review["costs"]["recorded_active_wall_seconds"] == wall
            and review["costs"]["separate_label_integrity_hamiltonian_actions"] == 0,
            "Prior cost review does not match its negative gate")
    expected_sources = {"large_qubit/section6.py", "large_qubit/acquisition.py", "large_qubit/corpus.py"}
    require(type(gate["source_bindings"]) is dict and set(gate["source_bindings"]) == expected_sources,
            "Missing immutable v1 source provenance")
    for relative, fingerprint in gate["source_bindings"].items():
        require(sha256_file(workspace_path(root, relative, namespace="large_qubit"))
                == _hash(fingerprint, "prior immutable source"), "Frozen v1 acquisition source changed")
    return {"gate": {"path": PRIOR_GATE_PATH, "sha256": PRIOR_GATE_SHA256},
            "independent_review": {"path": PRIOR_REVIEW_PATH, "sha256": PRIOR_REVIEW_SHA256},
            "logical_energy_requests": PRIOR_ORDINARY_REQUESTS, "wall_seconds": wall,
            "separate_verification_cost": {"hamiltonian_actions": 0, "wall_seconds": 0.0}}


def _prior_cost(root, value):
    expected = prior_attempt_cost(root)
    require(canonical_bytes(value) == canonical_bytes(expected),
            "Every consumer must retain the authenticated full failed-v1 acquisition cost")
    return deepcopy(expected)


def _cumulative_cost(prior, fresh):
    count = prior["logical_energy_requests"] + fresh["logical_energy_requests"]
    _integer(count, "cumulative prior and fresh ordinary requests", maximum=CUMULATIVE_REQUEST_CEILING)
    return {"logical_energy_requests": count,
            "wall_seconds": _number(prior["wall_seconds"] + fresh["wall_seconds"], "cumulative acquisition wall")}


def _flatten(groups):
    return sorted([deepcopy(row) for alias in TRAINING for row in groups[alias]],
                  key=lambda row: row["request_ordinal"])


def _group_metadata(selection):
    groups = {}
    for alias in TRAINING:
        fit, diagnostic = selection["fitting_by_geometry"][alias], selection["diagnostic_by_geometry"][alias]
        groups[alias] = {"fitting_rows": len(fit), "diagnostic_rows": len(diagnostic),
            "all_raw_fitting_rows": len(selection["raw_fitting_by_geometry"][alias])}
        for role, rows in (("fitting", fit), ("diagnostic", diagnostic)):
            groups[alias][role + "_row_ids_sha256"] = digest([row["row_id"] for row in rows])
            groups[alias][role + "_request_ordinals_sha256"] = digest([row["request_ordinal"] for row in rows])
    return groups


def _preserve_incomplete_directories(root, directories):
    """Rename partial outputs to unused siblings; never delete or overwrite.

    Every resolved absolute source and destination is checked against its own
    releases/corpora namespace before any Windows directory move is attempted.
    The executor serializes finalization with its acquisition lock.
    """
    from uuid import uuid4
    present = [(directory, namespace) for directory, namespace in directories if directory.exists()]
    token = uuid4().hex
    moves = []
    for directory, namespace in present:
        boundary = workspace_path(root, namespace, namespace=namespace).resolve(strict=True)
        original = workspace_path(root, directory.relative_to(root).as_posix(), namespace=namespace)
        require(original.is_dir() and not original.is_symlink(), "A partial output must be a real directory")
        original_resolved = original.resolve(strict=True)
        require(original_resolved != boundary and original_resolved.is_relative_to(boundary),
                "Partial output source escaped its intended namespace")
        preserved = original.with_name(f"{original.name}_incomplete_{token}")
        preserved_resolved = preserved.resolve()
        require(preserved_resolved.is_relative_to(boundary) and preserved_resolved.parent == original_resolved.parent
                and preserved_resolved != original_resolved and not preserved.exists() and not preserved.is_symlink(),
                "Partial output preservation target escaped its namespace or already exists")
        moves.append((original, preserved))
    for original, preserved in moves:
        original.rename(preserved)


def build_release(workspace, *, eligible_rows_path, catalogue, quality_report, stop_reason,
                  acquisition_cost, prior_acquisition_cost, raw_artifacts, verification_cost=None, policy_path=POLICY_PATH,
                  release_root=RELEASE_ROOT, corpus_root=CORPUS_ROOT):
    """Finalize or safely resume one release using the same frozen arguments.

    Ordinary cost includes every started request, HF, diagnostics, rejected local
    candidates, duplicates and retries. Verification remains separate overhead;
    its time and the full failed-v1 cost enter every consumer's attributable cost.
    Prior cost provenance never supplies labels or parents to this fresh release.
    All eligible raw fitting rows remain in their bound raw acquisition file.
    A complete identical release is authenticated and returned without writes.
    Incomplete output directories are preserved as uniquely named siblings
    before rebuilding. A complete release or readable partial manifest with
    different frozen inputs is refused without moving or rewriting anything.
    """
    root = Path(workspace).resolve(strict=True)
    destination = workspace_path(root, release_root, namespace="large_qubit/releases")
    payload_root = workspace_path(root, corpus_root, namespace="large_qubit/corpora")
    require(release_root == RELEASE_ROOT and corpus_root == CORPUS_ROOT,
            "Fresh v2 data must stay in the declared v2 release and corpus namespaces")
    source = workspace_path(root, eligible_rows_path, namespace=PRIVATE_ROOT)
    rows = _read_rows(source)
    selection = select_balanced_rows(rows, catalogue=catalogue, stop_reason=stop_reason)
    _quality_check(quality_report, selection, catalogue)
    cost, verification = _cost(acquisition_cost), _verification_cost(verification_cost)
    require(verification["hamiltonian_actions"] >= 12 and verification["wall_seconds"] > 0,
            "A v2 release requires all twelve separately charged label checks")
    prior = _prior_cost(root, prior_acquisition_cost)
    cumulative = _cumulative_cost(prior, cost)
    require(len(rows) + 3 <= cost["logical_energy_requests"]
            and max(row["request_ordinal"] for row in rows) <= cost["logical_energy_requests"],
            "Eligible rows cannot exceed all charged requests including the three HFs")
    require(type(raw_artifacts) is dict and eligible_rows_path in raw_artifacts,
            "Bind every immutable raw artifact, including all eligible rows")
    for relative, fingerprint in raw_artifacts.items():
        _hash(fingerprint, "raw artifact")
        require(sha256_file(workspace_path(root, relative, namespace=PRIVATE_ROOT)) == fingerprint,
                "Raw acquisition artifact changed before release")
    bindings = _bindings(root, policy_path)
    _validate_bindings(root, bindings, catalogue)
    role_rows = {"fitting": _flatten(selection["fitting_by_geometry"]),
                 "diagnostic": _flatten(selection["diagnostic_by_geometry"])}
    payloads = {role: b"".join(canonical_bytes(row) + b"\n" for row in values)
                for role, values in role_rows.items()}
    from hashlib import sha256
    files = {role: {"path": f"{corpus_root}/{role}.jsonl", "bytes": len(data), "sha256": sha256(data).hexdigest()}
             for role, data in payloads.items()}
    manifest = {
        "schema_version": "co2-corpus-release-v2", "benchmark_id": BENCHMARK_ID,
        "release_id": Path(release_root).name, "release_root": release_root, "corpus_root": corpus_root,
        "pipeline_seed": 22, "length": 32, "catalogue_sha256": catalogue["sha256"],
        "rows_per_geometry": selection["rows_per_geometry"], "diagnostics_per_geometry": 128,
        "selection": {"stop_reason": stop_reason, "order": "global_actual_request_ordinal_acquisition_prefix",
                      "energy_ranking_used": False, "quality_failure_fallback_allowed": False,
                      "all_eligible_raw_fitting_retained": True},
        "groups": _group_metadata(selection), "files": files, "bindings": bindings,
        "eligible_rows": _descriptor(root, eligible_rows_path, namespace=PRIVATE_ROOT),
        "raw_artifacts_sha256": dict(raw_artifacts), "quality_report": deepcopy(quality_report),
        "acquisition_cost": cost, "separate_verification_cost": verification,
        "prior_acquisition_cost": prior, "cumulative_acquisition_cost": cumulative,
        "campaign_production_request_ceiling": 8192,
        "cumulative_production_request_ceiling": CUMULATIVE_REQUEST_CEILING,
        "prior_attempt_labels_imported": False,
        "attributable_total_wall_seconds": _number(cumulative["wall_seconds"] + verification["wall_seconds"], "total offline wall"),
        "consumers": list(CONSUMERS), "shared_physical_acquisition_counted_once_globally": True,
        "source_cost_attribution": "full_prior_failed_and_fresh_acquisition_plus_verification_cost_per_corpus_using_method",
        "diagnostics_excluded_from_fitting_weighting_parent_selection_and_retrieval": True,
        "weighting": _weighting_spec(),
        "model_training_executed": False, "heldout_or_reference_access": False,
    }
    manifest_path = destination / "manifest.json"
    sidecar_path = destination / "manifest.sha256"
    completed_files = (manifest_path, sidecar_path, payload_root / "fitting.jsonl", payload_root / "diagnostic.jsonl")
    if all(path.is_file() for path in completed_files):
        existing_hash = sha256_file(manifest_path)
        try:
            complete_marker = sidecar_path.read_text(encoding="ascii").strip() == existing_hash
        except UnicodeError:
            complete_marker = False
        if complete_marker:
            verify_release(root, release_path=manifest_path.relative_to(root).as_posix(),
                           expected_release_sha256=existing_hash)
            existing = read_json(manifest_path)
            require(canonical_bytes(existing) == canonical_bytes(manifest),
                    "Existing release uses different frozen finalization inputs")
            return {"manifest_path": manifest_path.relative_to(root).as_posix(),
                    "manifest_sha256": existing_hash, "manifest": existing}
    if manifest_path.is_file():
        try:
            partial_manifest = read_json(manifest_path)
        except (ValueError, UnicodeError):
            partial_manifest = None
        require(partial_manifest is None or canonical_bytes(partial_manifest) == canonical_bytes(manifest),
                "Partial release manifest uses different frozen finalization inputs")
    _preserve_incomplete_directories(root, ((destination, "large_qubit/releases"),
                                          (payload_root, "large_qubit/corpora")))
    destination.mkdir(parents=True)
    payload_root.mkdir(parents=True)
    for role, data in payloads.items():
        with (payload_root / f"{role}.jsonl").open("xb") as handle:
            handle.write(data)
    with manifest_path.open("xb") as handle:
        handle.write(canonical_bytes(manifest) + b"\n")
    fingerprint = sha256_file(manifest_path)
    with sidecar_path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(fingerprint + "\n")
    return {"manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": fingerprint, "manifest": manifest}


def _authenticate(workspace, release_path, expected_release_sha256):
    root = Path(workspace).resolve(strict=True)
    _hash(expected_release_sha256, "trusted release pin")
    path = workspace_path(root, release_path, namespace="large_qubit/releases")
    require(release_path == RELEASE_PATH and sha256_file(path) == expected_release_sha256,
            "Release differs from the caller's trusted pin")
    sidecar = path.with_suffix(".sha256")
    require(sidecar.read_text(encoding="ascii").strip() == expected_release_sha256, "Release sidecar differs")
    saved = read_json(path)
    expected_fields = {"schema_version", "benchmark_id", "release_id", "release_root", "corpus_root",
        "pipeline_seed", "length", "catalogue_sha256", "rows_per_geometry", "diagnostics_per_geometry",
        "selection", "groups", "files", "bindings", "eligible_rows", "raw_artifacts_sha256", "quality_report",
        "acquisition_cost", "separate_verification_cost", "attributable_total_wall_seconds", "consumers",
        "prior_acquisition_cost", "cumulative_acquisition_cost", "campaign_production_request_ceiling",
        "cumulative_production_request_ceiling", "prior_attempt_labels_imported",
        "shared_physical_acquisition_counted_once_globally", "source_cost_attribution",
        "diagnostics_excluded_from_fitting_weighting_parent_selection_and_retrieval", "weighting",
        "model_training_executed", "heldout_or_reference_access"}
    require(type(saved) is dict and set(saved) == expected_fields, "Release manifest fields differ")
    require(saved["schema_version"] == "co2-corpus-release-v2" and saved["benchmark_id"] == BENCHMARK_ID
            and saved["release_root"] == path.parent.relative_to(root).as_posix()
            and saved["corpus_root"] == CORPUS_ROOT
            and saved["release_id"] == path.parent.name and type(saved["pipeline_seed"]) is int
            and saved["pipeline_seed"] == 22 and type(saved["length"]) is int and saved["length"] == 32
            and type(saved["rows_per_geometry"]) is int and saved["rows_per_geometry"] in (1024, 2048)
            and type(saved["diagnostics_per_geometry"]) is int and saved["diagnostics_per_geometry"] == 128,
            "Release identity/population differs")
    require(saved["consumers"] == list(CONSUMERS)
            and saved["shared_physical_acquisition_counted_once_globally"] is True
            and saved["source_cost_attribution"] == "full_prior_failed_and_fresh_acquisition_plus_verification_cost_per_corpus_using_method"
            and saved["prior_attempt_labels_imported"] is False
            and saved["diagnostics_excluded_from_fitting_weighting_parent_selection_and_retrieval"] is True
            and saved["model_training_executed"] is False and saved["heldout_or_reference_access"] is False,
            "Release access/cost scope differs")
    require(canonical_bytes(saved["weighting"]) == canonical_bytes(_weighting_spec()),
            "Release weighting declarations differ from fixed fitting-only mathematics")
    selection = saved["selection"]
    require(type(selection) is dict and selection.get("stop_reason") in ("target", "budget", "schedule")
            and canonical_bytes(selection) == canonical_bytes({"stop_reason": selection["stop_reason"],
                "order": "global_actual_request_ordinal_acquisition_prefix", "energy_ranking_used": False,
                "quality_failure_fallback_allowed": False, "all_eligible_raw_fitting_retained": True}),
            "Release prefix-selection declarations differ")
    catalogue = read_json(root / "large_qubit/catalogues/co2_screened_v1.json")
    operator_limit(catalogue)
    require(saved["catalogue_sha256"] == catalogue["sha256"], "Release catalogue identity differs")
    _validate_bindings(root, saved["bindings"], catalogue)
    _quality_scope(saved["quality_report"], saved["rows_per_geometry"], catalogue)
    cost, verification = _cost(saved["acquisition_cost"]), _verification_cost(saved["separate_verification_cost"])
    require(verification["hamiltonian_actions"] >= 12 and verification["wall_seconds"] > 0,
            "Released v2 data lacks twelve separately charged label checks")
    prior = _prior_cost(root, saved["prior_acquisition_cost"])
    cumulative = _cumulative_cost(prior, cost)
    require(canonical_bytes(saved["cumulative_acquisition_cost"]) == canonical_bytes(cumulative)
            and type(saved["campaign_production_request_ceiling"]) is int
            and saved["campaign_production_request_ceiling"] == 8192
            and type(saved["cumulative_production_request_ceiling"]) is int
            and saved["cumulative_production_request_ceiling"] == CUMULATIVE_REQUEST_CEILING,
            "Cumulative accounting omits failed history or conflates the campaign request caps")
    require(_number(saved["attributable_total_wall_seconds"], "total offline cost")
            == cumulative["wall_seconds"] + verification["wall_seconds"],
            "Total offline cost omits/duplicates prior acquisition or fresh verification")
    require(set(saved["files"]) == {"fitting", "diagnostic"}
            and type(saved["groups"]) is dict and set(saved["groups"]) == set(TRAINING), "Incomplete released populations")
    payload_root = workspace_path(root, saved["corpus_root"], namespace="large_qubit/corpora")
    require(payload_root != root / "large_qubit/corpora", "Use a versioned corpus payload namespace")
    for role in ("fitting", "diagnostic"):
        _descriptor_shape(root, saved["files"][role], namespace=saved["corpus_root"])
        require(saved["files"][role]["path"] == f"{saved['corpus_root']}/{role}.jsonl", "Cross-role payload routing")
    _descriptor_shape(root, saved["eligible_rows"], namespace=PRIVATE_ROOT)
    require(type(saved["raw_artifacts_sha256"]) is dict
            and saved["raw_artifacts_sha256"].get(saved["eligible_rows"]["path"]) == saved["eligible_rows"]["sha256"],
            "Missing immutable all-eligible-rows binding")
    for relative, fingerprint in saved["raw_artifacts_sha256"].items():
        workspace_path(root, relative, namespace=PRIVATE_ROOT)
        _hash(fingerprint, "raw acquisition artifact")
    group_fields = {"fitting_rows", "diagnostic_rows", "all_raw_fitting_rows",
                    "fitting_row_ids_sha256", "fitting_request_ordinals_sha256",
                    "diagnostic_row_ids_sha256", "diagnostic_request_ordinals_sha256"}
    for group in saved["groups"].values():
        require(type(group) is dict and set(group) == group_fields, "Released group descriptor fields differ")
        require(type(group["fitting_rows"]) is int and group["fitting_rows"] == saved["rows_per_geometry"]
                and type(group["diagnostic_rows"]) is int and group["diagnostic_rows"] == 128,
                "Released group counts differ")
        _integer(group["all_raw_fitting_rows"], "all raw fitting count", minimum=saved["rows_per_geometry"])
        for field in group_fields - {"fitting_rows", "diagnostic_rows", "all_raw_fitting_rows"}:
            _hash(group[field], "group row-order identity")
    require(sha256_file(path) == expected_release_sha256, "Release changed during authentication")
    return root, saved, catalogue


def _role_rows(root, saved, catalogue, role):
    path = _check_descriptor(root, saved["files"][role], namespace=saved["corpus_root"])
    original = _read_rows(path)
    rows = _validated_rows(original, catalogue)
    require(canonical_bytes(original) == canonical_bytes(rows), "Released payload must retain acquisition order")
    require(all(row["role"] == role for row in rows), "Cross-role row contamination")
    count = saved["rows_per_geometry"] if role == "fitting" else 128
    require(len(rows) == count * len(TRAINING), "Released row count differs")
    for alias in TRAINING:
        selected = [row for row in rows if row["geometry_id"] == alias]
        group = saved["groups"][alias]
        require(len(selected) == group[role + "_rows"] == count
                and digest([row["row_id"] for row in selected]) == group[role + "_row_ids_sha256"]
                and digest([row["request_ordinal"] for row in selected]) == group[role + "_request_ordinals_sha256"],
                "Released group order/identity differs")
    return rows


def verify_release(workspace, *, expected_release_sha256, release_path=RELEASE_PATH):
    """Maintenance validation of both roles and raw prefix; zero H actions.

    The Section6 executor additionally replays acquisition decisions and quality
    against fresh-HF ledger records. This function authenticates that report and
    its selected-row binding; it never solves/evaluates a physical state.
    """
    root, saved, catalogue = _authenticate(workspace, release_path, expected_release_sha256)
    raw = _check_descriptor(root, saved["eligible_rows"], namespace=PRIVATE_ROOT)
    require(type(saved["raw_artifacts_sha256"]) is dict
            and saved["raw_artifacts_sha256"].get(saved["eligible_rows"]["path"]) == saved["eligible_rows"]["sha256"],
            "Missing all-eligible-rows raw binding")
    for relative, fingerprint in saved["raw_artifacts_sha256"].items():
        _hash(fingerprint, "raw acquisition artifact")
        require(sha256_file(workspace_path(root, relative, namespace=PRIVATE_ROOT)) == fingerprint,
                "Immutable raw acquisition artifact changed")
    raw_rows = _read_rows(raw)
    selection = select_balanced_rows(raw_rows, catalogue=catalogue, stop_reason=saved["selection"]["stop_reason"])
    require(selection["rows_per_geometry"] == saved["rows_per_geometry"]
            and canonical_bytes(_group_metadata(selection)) == canonical_bytes(saved["groups"]),
            "Released prefix/group metadata differs from complete raw acquisition")
    require(canonical_bytes(saved["selection"]) == canonical_bytes({"stop_reason": selection["stop_reason"],
        "order": "global_actual_request_ordinal_acquisition_prefix", "energy_ranking_used": False,
        "quality_failure_fallback_allowed": False, "all_eligible_raw_fitting_retained": True}), "Prefix-selection policy differs")
    for role, key in (("fitting", "fitting_by_geometry"), ("diagnostic", "diagnostic_by_geometry")):
        require(canonical_bytes(_role_rows(root, saved, catalogue, role)) == canonical_bytes(_flatten(selection[key])),
                "Released labels differ from the balanced raw acquisition prefix")
    _quality_check(saved["quality_report"], selection, catalogue)
    require(len(raw_rows) + 3 <= saved["acquisition_cost"]["logical_energy_requests"]
            and max(row["request_ordinal"] for row in raw_rows) <= saved["acquisition_cost"]["logical_energy_requests"],
            "Released accounting omits started queries or the three HFs")
    return {"all_passed": True, "release_sha256": expected_release_sha256,
            "rows_per_geometry": saved["rows_per_geometry"], "fitting_rows": 3 * saved["rows_per_geometry"],
            "diagnostic_rows": 384, "raw_prefix_verified": True, "quality_report_selected_rows_bound": True,
            "quality_recomputed_from_HF_ledger": False, "hamiltonian_actions": 0,
            "model_training_executed": False}


@dataclass(frozen=True)
class AuthenticatedRelease:
    consumer_id: str
    purpose: str
    release_sha256: str
    _workspace: Path
    _release_path: str

    def _rows(self, alias, role):
        if alias is not None:
            require_geometry_access(alias, "diagnostics" if role == "diagnostic" else "training")
        root, saved, catalogue = _authenticate(self._workspace, self._release_path, self.release_sha256)
        rows = _role_rows(root, saved, catalogue, role)
        return [row for row in rows if alias is None or row["geometry_id"] == alias]

    @property
    def source_acquisition_cost(self):
        _, saved, _ = _authenticate(self._workspace, self._release_path, self.release_sha256)
        return {"release_sha256": self.release_sha256,
                "logical_energy_requests": saved["cumulative_acquisition_cost"]["logical_energy_requests"],
                "wall_seconds": saved["attributable_total_wall_seconds"]}

    @property
    def campaign_acquisition_cost(self):
        _, saved, _ = _authenticate(self._workspace, self._release_path, self.release_sha256)
        return deepcopy(saved["acquisition_cost"])

    @property
    def prior_acquisition_cost(self):
        _, saved, _ = _authenticate(self._workspace, self._release_path, self.release_sha256)
        return deepcopy(saved["prior_acquisition_cost"])

    @property
    def cumulative_acquisition_cost(self):
        _, saved, _ = _authenticate(self._workspace, self._release_path, self.release_sha256)
        return deepcopy(saved["cumulative_acquisition_cost"])

    @property
    def separate_verification_cost(self):
        _, saved, _ = _authenticate(self._workspace, self._release_path, self.release_sha256)
        return deepcopy(saved["separate_verification_cost"])

    def fitting_rows(self, alias):
        if self.purpose != "fitting" or self.consumer_id not in FITS:
            raise PermissionError("Only a fitting-supervision view can expose fitting rows")
        return tuple(MappingProxyType({**row, "ops": tuple(row["ops"])})
                     for row in self._rows(alias, "fitting"))

    def diagnostic_rows(self, alias):
        if self.purpose != "diagnostics" or self.consumer_id not in FITS:
            raise PermissionError("Diagnostics require their separate explicit diagnostic view")
        return tuple(MappingProxyType({**row, "ops": tuple(row["ops"])})
                     for row in self._rows(alias, "diagnostic"))

    def retrieval_rows(self, alias=None):
        if self.purpose != "retrieval" or self.consumer_id != "training_circuit_retrieval":
            raise PermissionError("Only the fitting-source retrieval condition may expose retrieval rows")
        return tuple({key: deepcopy(row[key]) for key in ROW_FIELDS} for row in self._rows(alias, "fitting"))

    def weights(self, alias):
        if self.purpose != "fitting" or self.consumer_id not in FITS:
            raise PermissionError("Diagnostic and retrieval views cannot define training weights")
        mode = "uniform" if self.consumer_id == "gpt2_xl_uniform_lora" else "weighted"
        scale = None if mode == "uniform" else (.25 if self.consumer_id == "gpt2_xl_weighted_lora" else .10)
        return geometry_weights(self.fitting_rows(alias), mode=mode, lambda_scale=scale)


def load_release(workspace, *, consumer_id, expected_release_sha256, release_path=RELEASE_PATH, purpose=None):
    """Authenticate metadata only; no diagnostic/raw/fitting labels read here.

    Subsequent role-specific methods open only that role's payload. Retrieval
    receives detached six-field rows compatible with baselines.py; fitting and
    diagnostic supervision receive immutable seven-field row mappings.
    """
    if consumer_id not in CONSUMERS:
        raise PermissionError("This condition has no authenticated corpus access")
    default = "retrieval" if consumer_id == "training_circuit_retrieval" else "fitting"
    purpose = default if purpose is None else purpose
    if ((consumer_id == "training_circuit_retrieval" and purpose != "retrieval")
            or (consumer_id in FITS and purpose not in ("fitting", "diagnostics"))):
        raise PermissionError("Corpus purpose differs from its declared consumer")
    root, _, _ = _authenticate(workspace, release_path, expected_release_sha256)
    return AuthenticatedRelease(consumer_id, purpose, expected_release_sha256, root, release_path)
