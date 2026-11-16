"""Independent dense occupation/ladder fixtures; no molecular calculations."""

from itertools import combinations
import unittest

import numpy as np
from scipy.linalg import expm

from large_qubit.sector import SectorBasis


def dense_whole(state, wires, theta, nqubits):
    # Enumerate computational words independently of the sector mapping.
    result = state.copy()
    half = len(wires) // 2
    for index in range(len(state)):
        bits = [(index >> (nqubits - 1 - w)) & 1 for w in range(nqubits)]
        if [bits[w] for w in wires] != [0] * half + [1] * half:
            continue
        other = index
        for wire in wires:
            other ^= 1 << (nqubits - 1 - wire)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        result[index] = c * state[index] - s * state[other]
        result[other] = s * state[index] + c * state[other]
    return result


def annihilator(wire, nqubits):
    matrix = np.zeros((1 << nqubits, 1 << nqubits))
    for column in range(len(matrix)):
        word = format(column, f"0{nqubits}b")
        if word[wire] == "1":
            row = column ^ (1 << (nqubits - 1 - wire))
            matrix[row, column] = (-1) ** word[:wire].count("1")
    return matrix


def dense_fermionic(state, wires, theta, nqubits):
    half = len(wires) // 2
    operator = np.eye(len(state))
    for position, wire in enumerate(wires):
        factor = annihilator(wire, nqubits)
        if position >= half:
            factor = factor.T
        operator = factor @ operator
    return expm((theta / 2) * (operator - operator.T)) @ state


class SectorBasisTest(unittest.TestCase):
    def test_string_order_indexing_and_fermionic_ci_phase_are_independent(self):
        basis = SectorBasis(3, 2, 1)
        self.assertEqual(basis.alpha_strings.tolist(), [3, 5, 6])
        self.assertEqual(basis.beta_strings.tolist(), [1, 2, 4])
        physical = np.arange(1, 10, dtype=np.float64)
        ci = basis.to_ci(physical)
        for a, alpha in enumerate(basis.alpha_strings.tolist()):
            for b, beta in enumerate(basis.beta_strings.tolist()):
                interleaved = [("a", p) if w % 2 == 0 else ("b", p)
                               for p in range(3) for w in (2 * p, 2 * p + 1)
                               if ((alpha if w % 2 == 0 else beta) >> p) & 1]
                blocked = sorted(interleaved, key=lambda x: (x[0], x[1]))
                permutation = [blocked.index(x) for x in interleaved]
                inversions = sum(permutation[i] > permutation[j] for i in range(len(permutation))
                                 for j in range(i + 1, len(permutation)))
                expected = physical[a * 3 + b] * (-1) ** inversions
                self.assertEqual(ci[a, b], expected)
                word = "".join(str(((alpha if w % 2 == 0 else beta) >> (w // 2)) & 1) for w in range(6))
                self.assertEqual(int(basis.sector_indices[a * 3 + b]), int(word, 2))
        np.testing.assert_array_equal(basis.from_ci(ci), physical)
        self.assertEqual(basis.to_ci(basis.hf_state())[0, 0], -1)

    def test_all_small_populations_roundtrip_and_full_leakage_guard(self):
        rng = np.random.default_rng(9)
        for nalpha in range(4):
            for nbeta in range(4):
                basis = SectorBasis(3, nalpha, nbeta)
                state = rng.normal(size=basis.dimension) + 1j * rng.normal(size=basis.dimension)
                state /= np.linalg.norm(state)
                np.testing.assert_array_equal(basis.from_full(basis.to_full(state)), state)
                np.testing.assert_array_equal(basis.from_ci(basis.to_ci(state)), state)
                self.assertLess(basis.diagnostics(state)["norm_error"], 1e-14)
        basis = SectorBasis(2, 1, 1)
        bad = basis.to_full(basis.hf_state())
        bad[0] = 0.1
        with self.assertRaisesRegex(ValueError, "leaks"):
            basis.from_full(bad)

    def test_whole_gates_match_independent_dense_semantics_for_complex_states(self):
        basis = SectorBasis(4, 2, 2)
        rng = np.random.default_rng(34)
        state = rng.normal(size=basis.dimension) + 1j * rng.normal(size=basis.dimension)
        state /= np.linalg.norm(state)
        gates = [((0, 6), .371), ((1, 7), -.231), ((0, 1, 6, 7), .487), ((2, 4, 0, 6), -.149)]
        for wires, theta in gates:
            with self.subTest(wires=wires):
                actual = basis.apply_gate(state, wires, theta)
                expected = dense_whole(basis.to_full(state), wires, theta, 8)
                np.testing.assert_allclose(basis.to_full(actual), expected, atol=1e-13)
                np.testing.assert_allclose(basis.apply_gate(actual, wires, -theta), state, atol=1e-13)
        before = state.copy()
        circuit = basis.apply_circuit(state, gates)
        expected = basis.to_full(state)
        for wires, theta in gates:
            expected = dense_whole(expected, wires, theta, 8)
        np.testing.assert_allclose(basis.to_full(circuit), expected, atol=1e-13)
        np.testing.assert_array_equal(state, before)

    def test_fermionic_gates_match_independent_ladder_exponential_and_differ_from_whole(self):
        basis = SectorBasis(3, 1, 1)
        rng = np.random.default_rng(117)
        state = rng.normal(size=basis.dimension) + 1j * rng.normal(size=basis.dimension)
        state /= np.linalg.norm(state)
        for wires in ((0, 4), (1, 5), (0, 1, 4, 5), (0, 3, 2, 5)):
            with self.subTest(wires=wires):
                actual = basis.apply_gate(state, wires, .621, family="fermionic")
                expected = dense_fermionic(basis.to_full(state), wires, .621, 6)
                np.testing.assert_allclose(basis.to_full(actual), expected, atol=1e-13)
        whole = basis.apply_gate(state, (0, 4), .621)
        fermionic = basis.apply_gate(state, (0, 4), .621, family="fermionic")
        self.assertGreater(np.linalg.norm(whole - fermionic), .01)

    def test_real_path_matches_complex_path_and_conserves_norm(self):
        basis = SectorBasis(4, 2, 2)
        gates = [((0, 6), .271), ((0, 1, 6, 7), -.443), ((1, 7), .17)]
        for family in ("whole_qubit", "fermionic"):
            real = basis.apply_circuit(basis.hf_state(np.float64), gates, family=family)
            complex_state = basis.apply_circuit(basis.hf_state(np.complex128), gates, family=family)
            self.assertEqual(real.dtype, np.dtype("float64"))
            np.testing.assert_allclose(real, complex_state, atol=1e-14)
            self.assertLess(abs(np.vdot(real, real) - 1), 1e-14)

    def test_lazy_cache_has_hard_byte_budget_eviction_and_no_state_payload(self):
        basis = SectorBasis(4, 2, 2, cache_budget_bytes=1000)
        first = basis.gate_pairs((0, 6))
        self.assertLessEqual(basis.cache_info()["bytes"], 1000)
        basis.gate_pairs((0, 6))
        self.assertEqual(basis.cache_info()["hits"], 1)
        basis.gate_pairs((1, 7))
        self.assertGreater(basis.cache_info()["evictions"], 0)
        basis.gate_pairs((0, 1, 6, 7), family="fermionic")
        self.assertLessEqual(basis.cache_info()["bytes"], 1000)
        self.assertFalse(first.source.flags.writeable)
        basis.clear_cache()
        self.assertEqual(basis.cache_info()["bytes"], 0)
        uncached = SectorBasis(4, 2, 2, cache_budget_bytes=0)
        uncached.gate_pairs((0, 6))
        self.assertEqual(uncached.cache_info()["entries"], 0)

    def test_resource_precision_population_and_type_guards(self):
        for args in ((True, 1, 1), (3, -1, 1), (3, 4, 1), (32, 0, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SectorBasis(*args)
        with self.assertRaisesRegex(ValueError, "allocation limit"):
            SectorBasis(15, 11, 11, max_dimension=100)
        large = SectorBasis(15, 11, 11, cache_budget_bytes=0)
        self.assertEqual(large.dimension, 1863225)
        with self.assertRaisesRegex(ValueError, "small fixtures"):
            large.to_full(large.hf_state())
        basis = SectorBasis(3, 1, 1)
        for bad in (np.ones(basis.dimension, dtype=np.float32), np.ones(basis.dimension),
                    np.full(basis.dimension, np.nan)):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                basis.apply_gate(bad, (0, 4), .2)
        for wires, theta in (((0, 1), .2), ((0, 0), .2), ((0, 6), .2), ((0, 4), True), ((0, 4), np.inf)):
            with self.subTest(wires=wires, theta=theta), self.assertRaises(ValueError):
                basis.apply_gate(basis.hf_state(), wires, theta)


if __name__ == "__main__":
    unittest.main()
