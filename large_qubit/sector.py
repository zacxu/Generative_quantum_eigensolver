"""Bounded fixed-population states and exact excitation rotations.

Flat state amplitudes use interleaved alpha/beta Jordan-Wigner occupations
(wire 0 most significant), indexed alpha-string-major/beta-string-minor.
PySCF CI arrays instead use spin-blocked creation order. ``to_ci``/``from_ci``
apply the explicit fermionic reordering phase; a reshape alone is incorrect.
No full Hilbert-space object is built except explicitly capped small fixtures.
"""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Integral, Real
import sys

import numpy as np

from .common import require


FULL_SPACE_FIXTURE_QUBIT_CAP = 12
NORM_TOLERANCE = 1e-10


def _integer(value, name, minimum=0):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral)
            and value >= minimum, f"{name} must be an integer >= {minimum}, excluding booleans")
    return int(value)


def _strings(norb, population):
    return np.array(sorted(sum(1 << p for p in occupied)
                           for occupied in combinations(range(norb), population)), dtype=np.uint64)


@dataclass(frozen=True)
class GatePairs:
    source: np.ndarray
    target: np.ndarray
    phase: object

    @property
    def nbytes(self):
        return self.source.nbytes + self.target.nbytes + (0 if self.phase is None else self.phase.nbytes)


class SectorBasis:
    """Exact N-alpha/N-beta basis, with lazy byte-bounded excitation maps.

    Gate methods return fresh arrays. Only float64 and complex128 states are
    accepted: adopting real arithmetic for production additionally requires
    evaluator-level comparison with complex128. No gate cache stores states.
    """

    def __init__(self, norb, n_alpha, n_beta, *, cache_budget_bytes=64 * 1024 * 1024,
                 max_dimension=10_000_000):
        self.norb = _integer(norb, "norb", 1)
        require(self.norb <= 31, "Sector occupation indexing supports at most 62 qubits")
        self.nalpha = self.n_alpha = _integer(n_alpha, "n_alpha")
        self.nbeta = self.n_beta = _integer(n_beta, "n_beta")
        require(self.nalpha <= self.norb and self.nbeta <= self.norb, "Population exceeds orbital count")
        self.n_qubits = 2 * self.norb
        self.shape = (math.comb(self.norb, self.nalpha), math.comb(self.norb, self.nbeta))
        self.dimension = math.prod(self.shape)
        require(self.dimension <= _integer(max_dimension, "max_dimension", 1),
                "Sector dimension exceeds explicit allocation limit")
        self.cache_budget_bytes = _integer(cache_budget_bytes, "cache_budget_bytes")
        self.alpha_strings = _strings(self.norb, self.nalpha)
        self.beta_strings = _strings(self.norb, self.nbeta)
        alpha_jw = np.zeros(self.shape[0], dtype=np.uint64)
        beta_jw = np.zeros(self.shape[1], dtype=np.uint64)
        for p in range(self.norb):
            alpha_jw |= ((self.alpha_strings >> np.uint64(p)) & np.uint64(1)) << np.uint64(self.n_qubits - 1 - 2 * p)
            beta_jw |= ((self.beta_strings >> np.uint64(p)) & np.uint64(1)) << np.uint64(self.n_qubits - 2 - 2 * p)
        self.sector_indices = (alpha_jw[:, None] | beta_jw[None, :]).reshape(-1)
        self.jw_indices = self.sector_indices
        parity = np.zeros(self.shape, dtype=np.uint8)
        for p in range(1, self.norb):
            occupied_alpha = ((self.alpha_strings >> np.uint64(p)) & np.uint64(1)).astype(np.uint8)
            preceding_beta = (np.bitwise_count(self.beta_strings & np.uint64((1 << p) - 1)) & 1).astype(np.uint8)
            parity ^= occupied_alpha[:, None] & preceding_beta[None, :]
        self._ci_phase = (1 - 2 * parity.astype(np.int8)).reshape(-1)
        for array in (self.alpha_strings, self.beta_strings, self.sector_indices, self._ci_phase):
            array.setflags(write=False)
        self._gate_cache = OrderedDict()
        self._cache_bytes = self._hits = self._misses = self._evictions = 0

    def validate_state(self, state, *, require_normalized=True):
        value = np.asarray(state)
        require(value.shape == (self.dimension,), "Expected a flat physical sector state")
        require(value.dtype in (np.dtype("float64"), np.dtype("complex128")),
                "State precision must be float64 or complex128")
        require(np.all(np.isfinite(value)), "State amplitudes must be finite")
        if require_normalized:
            norm = float(np.vdot(value, value).real)
            require(math.isfinite(norm) and abs(norm - 1.0) <= NORM_TOLERANCE,
                    "Sector state is not normalized")
        return value

    def hf_state(self, dtype=np.complex128):
        dtype = np.dtype(dtype)
        require(dtype in (np.dtype("float64"), np.dtype("complex128")), "HF precision must be float64 or complex128")
        result = np.zeros(self.dimension, dtype=dtype)
        result[0] = 1
        return result

    def to_ci(self, state):
        value = self.validate_state(state, require_normalized=False)
        return (value * self._ci_phase).reshape(self.shape)

    def from_ci(self, ci):
        value = np.asarray(ci)
        require(value.shape == self.shape, "CI array has the wrong alpha/beta string dimensions")
        return self.validate_state(value.reshape(-1), require_normalized=False) * self._ci_phase

    def to_full(self, state):
        require(self.n_qubits <= FULL_SPACE_FIXTURE_QUBIT_CAP, "Full-space conversion is restricted to small fixtures")
        value = self.validate_state(state, require_normalized=False)
        result = np.zeros(1 << self.n_qubits, dtype=value.dtype)
        result[self.sector_indices] = value
        return result

    def from_full(self, full):
        require(self.n_qubits <= FULL_SPACE_FIXTURE_QUBIT_CAP, "Full-space conversion is restricted to small fixtures")
        value = np.asarray(full)
        require(value.shape == (1 << self.n_qubits,), "Full-space state dimensions differ")
        require(value.dtype in (np.dtype("float64"), np.dtype("complex128")) and np.all(np.isfinite(value)),
                "Full-space state must have finite float64/complex128 amplitudes")
        inside = value[self.sector_indices]
        # An explicit complement check avoids cancellation in total-minus-sector norms.
        outside = np.ones(len(value), dtype=bool)
        outside[self.sector_indices] = False
        leakage = float(np.vdot(value[outside], value[outside]).real)
        require(leakage <= NORM_TOLERANCE, "Full-space state leaks outside the fixed populations")
        return inside.copy()

    def _template(self, wires, family):
        require(family in ("whole_qubit", "fermionic"), "Unknown excitation family")
        require(not isinstance(wires, (str, bytes)), "Excitation wires must be integer indices")
        try:
            wires = tuple(_integer(w, "wire") for w in wires)
        except TypeError as error:
            raise ValueError("Excitation wires must be an iterable") from error
        require(len(wires) in (2, 4) and len(set(wires)) == len(wires)
                and all(w < self.n_qubits for w in wires), "Excitations require two or four distinct valid wires")
        half = len(wires) // 2
        require(sum(w % 2 for w in wires[:half]) == sum(w % 2 for w in wires[half:]),
                "Excitation leaves the declared spin populations")
        return family, wires

    def _compile_pairs(self, family, wires):
        half = len(wires) // 2
        masks = [1 << (self.n_qubits - 1 - w) for w in wires]
        union = sum(masks)
        occupied = masks[half:] if family == "whole_qubit" else masks[:half]
        source = np.flatnonzero((self.sector_indices & np.uint64(union)) == np.uint64(sum(occupied)))
        row, col = source // self.shape[1], source % self.shape[1]
        alpha_toggle = sum(1 << (w // 2) for w in wires if w % 2 == 0)
        beta_toggle = sum(1 << (w // 2) for w in wires if w % 2 == 1)
        alpha = self.alpha_strings[row] ^ np.uint64(alpha_toggle)
        beta = self.beta_strings[col] ^ np.uint64(beta_toggle)
        target_row, target_col = np.searchsorted(self.alpha_strings, alpha), np.searchsorted(self.beta_strings, beta)
        require(np.all(target_row < self.shape[0]) and np.all(target_col < self.shape[1]),
                "Excitation mapping left the fixed-population basis")
        require(np.array_equal(self.alpha_strings[target_row], alpha) and np.array_equal(self.beta_strings[target_col], beta),
                "Excitation target strings are absent from the fixed-population basis")
        target = target_row * self.shape[1] + target_col
        phase = None
        if family == "fermionic":
            occupation = self.sector_indices[source].copy()
            phase = np.ones(len(source), dtype=np.int8)
            # Right-to-left factors: annihilate first-half wires in listed
            # order, then create last-half wires in listed order. Thus doubles
            # have T=a_b^dagger a_a^dagger a_j a_i, matching Phase 4 VQE.
            for wire in wires:
                parity = np.bitwise_count(occupation >> np.uint64(self.n_qubits - wire)) & 1
                phase *= 1 - 2 * parity.astype(np.int8)
                occupation ^= np.uint64(1 << (self.n_qubits - 1 - wire))
            require(np.array_equal(occupation, self.sector_indices[target]), "Fermionic ladder target mismatch")
        dtype = np.int32 if self.dimension <= np.iinfo(np.int32).max else np.int64
        source, target = source.astype(dtype), target.astype(dtype)
        for array in (source, target, phase):
            if array is not None:
                array.setflags(write=False)
        return GatePairs(source, target, phase)

    def gate_pairs(self, wires, *, family="whole_qubit"):
        key = self._template(wires, family)
        if key in self._gate_cache:
            self._hits += 1
            self._gate_cache.move_to_end(key)
            return self._gate_cache[key][0]
        self._misses += 1
        pairs = self._compile_pairs(*key)
        cost = self._cache_cost(key, pairs)
        if pairs.nbytes and cost <= self.cache_budget_bytes:
            while self._cache_bytes + cost > self.cache_budget_bytes:
                _, evicted = self._gate_cache.popitem(last=False)
                self._cache_bytes -= evicted[1]
                self._evictions += 1
            self._gate_cache[key] = (pairs, cost)
            self._cache_bytes += cost
        return pairs

    @staticmethod
    def _cache_cost(key, pairs):
        # Conservatively include array headers, dataclass/key objects and an
        # OrderedDict-node allowance, in addition to every retained data byte.
        arrays = (pairs.source, pairs.target) + (() if pairs.phase is None else (pairs.phase,))
        return (sum(max(sys.getsizeof(a), a.nbytes + 128) for a in arrays)
                + sys.getsizeof(pairs) + sys.getsizeof(key) + sys.getsizeof(key[1])
                + sum(sys.getsizeof(w) for w in key[1]) + sys.getsizeof(key[0]) + 128)

    @staticmethod
    def _rotate(value, pairs, theta):
        require(not isinstance(theta, (bool, np.bool_)) and isinstance(theta, Real) and math.isfinite(theta),
                "Excitation angle must be a finite real scalar")
        result = value.copy()
        source, target = value[pairs.source], value[pairs.target]
        cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
        signed_sine = sine if pairs.phase is None else sine * pairs.phase
        result[pairs.source] = cosine * source - signed_sine * target
        result[pairs.target] = signed_sine * source + cosine * target
        return result

    def apply_gate(self, state, wires, theta, *, family="whole_qubit"):
        value = self.validate_state(state)
        result = self._rotate(value, self.gate_pairs(wires, family=family), theta)
        return self.validate_state(result)

    def apply_circuit(self, state, gates, *, family="whole_qubit"):
        """Apply ``[(wires,theta), ...]`` with norm/finite guards at both ends."""
        result = self.validate_state(state).copy()
        for wires, theta in gates:
            result = self._rotate(result, self.gate_pairs(wires, family=family), theta)
        return self.validate_state(result)

    def diagnostics(self, state):
        value = self.validate_state(state, require_normalized=False)
        norm = float(np.vdot(value, value).real)
        return {"state_norm": norm, "norm_error": abs(norm - 1.0), "n_alpha": self.nalpha,
                "n_beta": self.nbeta, "sector_dimension": self.dimension,
                "sector_leakage_probability": 0.0, "population_conservation": "by_fixed_basis_construction"}

    def cache_info(self):
        return {"entries": len(self._gate_cache), "bytes": self._cache_bytes,
                "array_payload_bytes": sum(pairs.nbytes for pairs, _ in self._gate_cache.values()),
                "budget_bytes": self.cache_budget_bytes, "hits": self._hits,
                "misses": self._misses, "evictions": self._evictions,
                "base_index_bytes": self.sector_indices.nbytes + self._ci_phase.nbytes
                + self.alpha_strings.nbytes + self.beta_strings.nbytes}

    def clear_cache(self):
        self._gate_cache.clear()
        self._cache_bytes = 0
