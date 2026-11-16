"""Discrete random and training-only retrieval proposal rules, without scoring.

These functions propose circuits. HF and every subsequent attempted request,
including duplicate circuits/cache hits, are charged by the future score ledger.
Retrieval source labels may be used only from the authenticated fitting release;
the release loader/authentication belongs to Section 6, not these selectors.
"""

from copy import deepcopy
from decimal import Decimal
import math
import random
import re

from .common import digest, require
from .definitions import geometry, require_geometry_access
from .grammar import _checked_values, operator_limit, requested_length


TRAINING_ALIASES = ("CO2_T01", "CO2_T02", "CO2_T03")
MAX_POST_HF_REQUESTS = 1023
ROW_FIELDS = {"row_id", "geometry_id", "role", "ops", "energy_hartree", "catalogue_sha256"}
_SHA = re.compile("[0-9a-f]{64}")


def _integer(value, name, lower=0, upper=None):
    require(type(value) is int and value >= lower and (upper is None or value <= upper),
            f"Invalid {name}")
    return value


def _finite(value, name):
    require(type(value) in (int, float), f"Invalid {name}")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error
    require(math.isfinite(result), f"Nonfinite {name}")
    return result


def uniform_random(catalogue, *, length, count, seed):
    """Draw each position independently and exactly uniformly, including ID 0.

Python's randrange uses integer rejection sampling, avoiding modulo bias. No
novelty filter, cancellation filter or adaptive target feedback changes this
distribution. The caller still charges repeated proposals as logical queries.
"""
    requested_length(length)
    maximum = operator_limit(catalogue)
    _integer(count, "post-HF candidate count", upper=MAX_POST_HF_REQUESTS)
    _integer(seed, "random seed", upper=(1 << 63) - 1)
    generator = random.Random(seed)
    circuits = [[generator.randrange(maximum + 1) for _ in range(length)]
                for _ in range(count)]
    return {
        "method": "uniform_random", "catalogue_sha256": catalogue["sha256"],
        "length": length, "seed": seed, "circuits": circuits,
        "distribution": "independent_uniform_integer_IDs_including_Identity",
        "operator_count": maximum + 1, "identity_id": 0,
        "generator": "python_random_MT19937_randrange_integer_rejection",
        "post_HF_candidate_count": count, "HF_requests": 1,
        "maximum_total_logical_requests": count + 1,
        "duplicates_and_cache_hits_charged": True,
        "source_acquisition_cost": {"logical_energy_requests": 0, "wall_seconds": 0.0},
        "amortized_transfer_cost_advantage_claim": False,
    }


def nearest_training_geometry(target_alias):
    """Exact decimal geometry distances make midpoint tie-breaking stable."""
    target = geometry(target_alias, "physical_validation")
    target_r = Decimal(str(target["bond_length_angstrom"]))
    choices = []
    for alias in TRAINING_ALIASES:
        source = geometry(alias, "retrieval")
        source_r = Decimal(str(source["bond_length_angstrom"]))
        choices.append((abs(source_r - target_r), source_r, alias))
    return min(choices)[2]


def _source_cost(value):
    require(type(value) is dict and set(value) == {
        "release_sha256", "logical_energy_requests", "wall_seconds"},
        "Provide the full shared release's attributable acquisition cost")
    require(type(value["release_sha256"]) is str
            and _SHA.fullmatch(value["release_sha256"]), "Missing authenticated release identity")
    _integer(value["logical_energy_requests"], "source acquisition requests", lower=1)
    require(_finite(value["wall_seconds"], "source acquisition wall time") > 0,
            "Source acquisition wall time must be positive")
    return deepcopy(value)


def retrieval_candidates(target_alias, rows, *, catalogue, length,
                         acquisition_cost, max_candidates=MAX_POST_HF_REQUESTS):
    """Sort only nearest-source fitting labels, never target evaluation results.

Each row has precisely ROW_FIELDS; role is 'fitting' or 'diagnostic'. Diagnostics
are counted and excluded without consulting their energy or circuit. Missing
fitting rows at the nearest geometry produce an empty list, not a farther-source
fallback. Equal source energies sort by circuit tuple and then lexical row ID.
Only source labels are accepted; the signature has no target-energy argument.
"""
    requested_length(length)
    maximum = operator_limit(catalogue)
    _integer(max_candidates, "retrieval post-HF cap", upper=MAX_POST_HF_REQUESTS)
    cost = _source_cost(acquisition_cost)
    require(type(rows) in (list, tuple), "A source-release row sequence is required")
    source_alias = nearest_training_geometry(target_alias)
    seen_rows, selected = set(), []
    fitting_count = diagnostic_count = other_geometry_count = 0
    for row in rows:
        require(type(row) is dict and set(row) == ROW_FIELDS,
                "Retrieval accepts only the declared source-release row fields")
        row_id, alias = row["row_id"], row["geometry_id"]
        require(type(row_id) is str and row_id and row_id not in seen_rows,
                "Retrieval row IDs must be unique nonempty strings")
        seen_rows.add(row_id)
        require_geometry_access(alias, "retrieval")
        require(row["catalogue_sha256"] == catalogue["sha256"], "Source catalogue mismatch")
        require(row["role"] in ("fitting", "diagnostic"), "Unknown source row role")
        if row["role"] == "diagnostic":
            diagnostic_count += 1
            continue
        fitting_count += 1
        # Other training geometries cannot reorder or supply a fallback for the
        # nearest source. Their energy labels are not inspected by this selector.
        if alias != source_alias:
            other_geometry_count += 1
            continue
        ops = tuple(_checked_values(row["ops"], maximum=maximum, length=length))
        energy = _finite(row["energy_hartree"], "source fitting energy")
        selected.append((energy, ops, row_id))
    selected.sort()
    chosen = selected[:max_candidates]
    # This fingerprint binds selection order, not an external signature. Release
    # authentication must establish its identity before this selector is called.
    ordering_digest = digest([[energy, list(ops), row_id] for energy, ops, row_id in selected])
    return {
        "method": "training_circuit_retrieval", "target_id": target_alias,
        "source_geometry_id": source_alias, "catalogue_sha256": catalogue["sha256"],
        "length": length,
        "candidates": [{"row_id": row_id, "geometry_id": source_alias, "ops": list(ops)}
                       for _, ops, row_id in chosen],
        "order_rule": "nearest_training_R_then_lower_R_tie_then_source_energy_circuit_tuple_row_ID",
        "source_order_sha256": ordering_digest,
        "source_fitting_rows_seen": fitting_count,
        "excluded_diagnostic_rows": diagnostic_count,
        "excluded_other_training_geometry_rows": other_geometry_count,
        "eligible_nearest_source_rows": len(selected),
        "post_HF_candidate_count": len(chosen), "post_HF_candidate_cap": max_candidates,
        "HF_requests": 1, "maximum_total_logical_requests": len(chosen) + 1,
        "duplicates_and_cache_hits_charged": True,
        "target_based_reordering": False,
        "source_acquisition_cost": cost,
        "source_cost_attribution": "full_shared_release_acquisition_cost_per_corpus_using_method",
        "shared_physical_source_acquisition_counted_once_globally": True,
        "amortized_transfer_cost_advantage_claim": False,
    }


def baseline_spec():
    return {
        "schema": "co2-discrete-baselines-v1", "post_HF_request_cap": MAX_POST_HF_REQUESTS,
        "random": "independent_exact_uniform_IDs_0_through_512_same_legal_discrete_family",
        "retrieval": "nearest_registered_training_R_lower_R_tie_then_fitting_source_energy_ops_tuple_row_ID",
        "retrieval_diagnostics": "excluded_without_reading_labels_or_circuits",
        "nearest_source_empty": "no_farther_geometry_fallback",
        "target_based_reordering": False, "HF_requests": 1,
        "duplicates_and_cache_hits_charged": True,
        "source_acquisition_cost": "full_shared_release_cost_attributable_to_each_corpus_using_method",
        "global_physical_source_acquisition": "counted_once",
        "amortized_transfer_advantage_claim": False,
        "release_authentication": "required_in_Section6_loader_before_selection",
    }
