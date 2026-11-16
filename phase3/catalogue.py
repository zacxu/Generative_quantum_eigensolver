"""Deterministic, versioned whole-excitation operator catalogues.

Wire zero is the most-significant computational-basis bit. Occupied reference
spin orbitals are range(electrons); even wires are alpha and odd wires beta.
The gate-angle convention is PennyLane's phi (rotation uses phi/2):
https://docs.pennylane.ai/en/stable/code/api/pennylane.SingleExcitation.html
https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitation.html
"""

from collections.abc import Iterable, Mapping
from hashlib import sha256
from itertools import combinations
import json
from numbers import Integral


ANGLES = (-0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1)
SCHEMA_VERSION = "whole-excitation-catalogue-v1"


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, excluding booleans.")
    return int(value)


def canonical_json(payload):
    """The exact UTF-8 JSON serialization used to compute catalogue hashes."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _catalogue_payload(electrons, qubits):
    electrons = _integer(electrons, "electrons")
    qubits = _integer(qubits, "qubits")
    if not 2 <= qubits <= 62 or qubits % 2:
        raise ValueError("qubits must be an even integer between 2 and 62.")
    if not 0 <= electrons <= qubits:
        raise ValueError("electrons must lie between zero and qubits.")
    occupied = range(electrons)
    virtual = range(electrons, qubits)
    singles = sorted(
        (i, a) for i in occupied for a in virtual if i % 2 == a % 2
    )
    doubles = sorted(
        (i, j, a, b)
        for i, j in combinations(occupied, 2)
        for a, b in combinations(virtual, 2)
        if (i % 2 + j % 2) == (a % 2 + b % 2)
    )
    entries = [{"id": 0, "name": "Identity", "wires": [], "angle": 0.0}]
    for name, wire_tuples in (
        ("SingleExcitation", singles), ("DoubleExcitation", doubles)
    ):
        for wires in wire_tuples:
            for angle in ANGLES:
                entries.append({
                    "id": len(entries), "name": name,
                    "wires": list(wires), "angle": angle,
                })
    return {
        "schema_version": SCHEMA_VERSION,
        "electrons": electrons,
        "qubits": qubits,
        "angles": list(ANGLES),
        "ordering": (
            "Identity ID 0; lexicographic single wire tuples, then lexicographic "
            "double wire tuples; ascending angles within every tuple."
        ),
        "execution_order": "Operator IDs execute from left to right.",
        "bit_order": "Wire 0 is the most-significant basis bit.",
        "spin_convention": "Even wire = alpha; odd wire = beta.",
        "reference_occupied_wires": list(occupied),
        "rotation_convention": (
            "low=01/high=10 for singles; low=0011/high=1100 for doubles; "
            "out_low=c*in_low-s*in_high; out_high=s*in_low+c*in_high; "
            "c=cos(angle/2),s=sin(angle/2)."
        ),
        "counts": {
            "singles": len(singles), "doubles": len(doubles),
            "operators": len(entries),
        },
        "entries": entries,
    }


def build_catalogue(electrons, qubits):
    """Return a JSON-native catalogue plus its canonical payload SHA-256.

    The hash excludes only the top-level sha256 field. No PennyLane call is
    involved in enumeration. A whole excitation gate is one catalogue entry,
    rather than one of its individual Pauli-evolution factors.
    """
    payload = _catalogue_payload(electrons, qubits)
    return {**payload, "sha256": sha256(canonical_json(payload).encode("utf-8")).hexdigest()}


def validate_catalogue(catalogue):
    """Reject modified hashes, ordering, gate definitions, types, or metadata."""
    if not isinstance(catalogue, Mapping):
        raise ValueError("catalogue must be a mapping.")
    if "electrons" not in catalogue or "qubits" not in catalogue:
        raise ValueError("catalogue needs electrons and qubits.")
    expected = build_catalogue(catalogue["electrons"], catalogue["qubits"])
    try:
        actual_text = canonical_json(dict(catalogue))
    except (TypeError, ValueError) as error:
        raise ValueError("catalogue is not canonical JSON data.") from error
    if actual_text != canonical_json(expected):
        raise ValueError("catalogue differs from its declared deterministic specification.")
    return True


def validate_operator_ids(ops, catalogue):
    """Return immutable builtin-int IDs, rejecting bools, floats, and bad ranges."""
    validate_catalogue(catalogue)
    if isinstance(ops, (str, bytes, Mapping)) or not isinstance(ops, Iterable):
        raise ValueError("ops must be an iterable of integer operator IDs.")
    result = tuple(_integer(value, "operator ID") for value in ops)
    size = len(catalogue["entries"])
    if any(value < 0 or value >= size for value in result):
        raise ValueError(f"operator IDs must lie in [0,{size - 1}].")
    return result
