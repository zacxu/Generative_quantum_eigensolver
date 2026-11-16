"""Independent small Fock-space checks of historical integral recovery."""

from itertools import product
import unittest

import numpy as np

from large_qubit.regression import extract_spatial_integrals, independent_pauli_matrix


def ladder(nqubits, wire, create):
    matrix = np.zeros((1 << nqubits,) * 2, dtype=np.complex128)
    mask = 1 << (nqubits - 1 - wire)
    for column in range(1 << nqubits):
        occupied = bool(column & mask)
        if occupied == create:
            continue
        parity = sum(bool(column & (1 << (nqubits - 1 - previous))) for previous in range(wire))
        matrix[column ^ mask, column] = (-1) ** parity
    return matrix


def small_hamiltonian(h1, eri, nuclear):
    n = len(h1)
    destroy = [ladder(2 * n, wire, False) for wire in range(2 * n)]
    create = [value.conj().T for value in destroy]
    matrix = nuclear * np.eye(1 << (2 * n), dtype=np.complex128)
    for p, q in product(range(n), repeat=2):
        for spin in (0, 1):
            matrix += h1[p, q] * create[2*p+spin] @ destroy[2*q+spin]
    for p, q, r, s in product(range(n), repeat=4):
        for spin, other in product((0, 1), repeat=2):
            matrix += .5 * eri[p, q, r, s] * (create[2*p+spin] @ create[2*r+other]
                                                  @ destroy[2*s+other] @ destroy[2*q+spin])
    return matrix


def decompose_paulis(matrix):
    local = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
             "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    nqubits = int(np.log2(len(matrix)))
    terms = []
    for word in product("IXYZ", repeat=nqubits):
        operator = np.array([[1]])
        for letter in word:
            operator = np.kron(operator, local[letter])
        coefficient = np.trace(operator.conj().T @ matrix) / len(matrix)
        if abs(coefficient) > 1e-12:
            terms.append({"word": "".join(word), "real": float(coefficient.real),
                          "imag": float(coefficient.imag)})
    return {"physical_definition": {"n_qubits": nqubits, "n_spatial_orbitals": nqubits // 2},
            "pauli_terms": terms}


class HistoricalRegressionTests(unittest.TestCase):
    def test_recovered_integrals_match_independent_ladder_hamiltonian(self):
        h1 = np.array([[-1.3, .17], [.17, -.4]])
        factors = np.array([[[.9, .11], [.11, .4]], [[.1, -.08], [-.08, .3]]])
        eri = np.einsum("apq,ars->pqrs", factors, factors)
        nuclear = 1.27
        full = small_hamiltonian(h1, eri, nuclear)
        serialized = decompose_paulis(full)
        recovered_full = independent_pauli_matrix(serialized).toarray()
        np.testing.assert_allclose(recovered_full, full, atol=1e-12, rtol=0)
        actual_h1, actual_eri, actual_nuclear = extract_spatial_integrals(serialized)
        np.testing.assert_allclose(actual_h1, h1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual_eri, eri, atol=1e-12, rtol=0)
        self.assertAlmostEqual(actual_nuclear, nuclear, places=12)

    def test_spin_dependent_operator_is_rejected(self):
        alpha_number = ladder(4, 0, True) @ ladder(4, 0, False)
        with self.assertRaisesRegex(ValueError, "not spin-independent"):
            extract_spatial_integrals(decompose_paulis(alpha_number))

    def test_full_space_allocation_is_bounded_before_building_paulis(self):
        serialized = {"physical_definition": {"n_qubits": 30, "n_spatial_orbitals": 15},
                      "pauli_terms": []}
        with self.assertRaisesRegex(ValueError, "at most 12 wires"):
            independent_pauli_matrix(serialized)

    def test_imaginary_hermiticity_error_is_rejected(self):
        serialized = {"physical_definition": {"n_qubits": 2, "n_spatial_orbitals": 1},
                      "pauli_terms": [{"word": "XI", "real": 0, "imag": .2}]}
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            independent_pauli_matrix(serialized)


if __name__ == "__main__":
    unittest.main()
