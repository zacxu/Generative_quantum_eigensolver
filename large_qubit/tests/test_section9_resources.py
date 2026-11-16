"""Independent 4--6-qubit unitary checks; no Hamiltonians or energies."""

from copy import deepcopy
import math
import unittest

import numpy as np

from large_qubit.common import digest
from large_qubit.sector import SectorBasis
from large_qubit.section9_resources import excitation_decomposition, resource_report
from large_qubit.tests._section5_fixtures import catalogue_fixture


def _annihilator(wire, n_qubits):
    result = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=complex)
    for column in range(len(result)):
        bits = format(column, f'0{n_qubits}b')
        if bits[wire] == '1':
            result[column ^ (1 << (n_qubits - 1 - wire)), column] = (-1) ** bits[:wire].count('1')
    return result


def _native_unitary(wires, theta, family, n_qubits):
    size, half = 1 << n_qubits, len(wires) // 2
    operator = np.zeros((size, size), dtype=complex)
    if family == 'whole_qubit':
        for source in range(size):
            bits = format(source, f'0{n_qubits}b')
            if [int(bits[w]) for w in wires] == [0] * half + [1] * half:
                target = source
                for wire in wires:
                    target ^= 1 << (n_qubits - 1 - wire)
                operator[target, source] = 1.
    else:
        operator = np.eye(size, dtype=complex)
        for position, wire in enumerate(wires):
            ladder = _annihilator(wire, n_qubits)
            if position >= half:
                ladder = ladder.conj().T
            operator = ladder @ operator
    generator = operator - operator.conj().T
    # K^3=-K for an excitation's disjoint active basis pairs. This gives the
    # exact exponential without SciPy or the Pauli implementation under test.
    return (np.eye(size) + math.sin(theta / 2) * generator
            + (1 - math.cos(theta / 2)) * (generator @ generator))


def _apply_common(state, sequence, n_qubits):
    result = np.array(state, dtype=complex, copy=True)
    indices = np.arange(1 << n_qubits)
    for gate in sequence:
        name, wires = gate['gate'], gate['wires']
        if name == 'cx':
            control, target = (1 << (n_qubits - 1 - w) for w in wires)
            permutation = indices.copy()
            active = (indices & control) != 0
            permutation[active] ^= target
            result = result[permutation]
            continue
        mask = 1 << (n_qubits - 1 - wires[0])
        low = indices[(indices & mask) == 0]
        high = low | mask
        if name == 'x':
            matrix = np.array([[0, 1], [1, 0]])
        elif name == 'h':
            matrix = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        elif name == 'rx':
            c, s = math.cos(gate['angle_radians'] / 2), math.sin(gate['angle_radians'] / 2)
            matrix = np.array([[c, -1j * s], [-1j * s, c]])
        elif name == 'rz':
            matrix = np.diag([np.exp(-.5j * gate['angle_radians']),
                              np.exp(.5j * gate['angle_radians'])])
        else:
            raise AssertionError(name)
        a, b = result[low].copy(), result[high].copy()
        result[low] = matrix[0, 0] * a + matrix[0, 1] * b
        result[high] = matrix[1, 0] * a + matrix[1, 1] * b
    return result


class Section9ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()

    def test_complete_unitaries_both_families_contiguous_and_spectator_parities(self):
        cases = [(4, (0, 2)), (4, (0, 1, 2, 3)),
                 (6, (0, 4)), (6, (1, 5)), (6, (0, 1, 4, 5)),
                 (6, (0, 3, 2, 5)), (6, (3, 0, 5, 2))]
        for n_qubits, wires in cases:
            for family in ('whole_qubit', 'fermionic'):
                for theta in (.371, -.731):
                    with self.subTest(n_qubits=n_qubits, wires=wires, family=family, theta=theta):
                        report = excitation_decomposition(wires, theta, family=family, n_qubits=n_qubits)
                        actual = _apply_common(np.eye(1 << n_qubits), report['common_gate_sequence'], n_qubits)
                        expected = _native_unitary(wires, theta, family, n_qubits)
                        # Direct equality, not equality modulo an arbitrary
                        # global phase, covers every spectator occupation.
                        np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=0.)

    def test_decomposition_matches_native_sector_for_complex_superpositions(self):
        basis = SectorBasis(3, 1, 1)
        rng = np.random.default_rng(703)
        state = rng.normal(size=basis.dimension) + 1j * rng.normal(size=basis.dimension)
        state /= np.linalg.norm(state)
        gates = [((0, 4), .417), ((0, 3, 2, 5), -.318), ((3, 0, 5, 2), .209)]
        for family in ('whole_qubit', 'fermionic'):
            actual = basis.to_full(state)
            for wires, theta in gates:
                report = excitation_decomposition(wires, theta, family=family, n_qubits=6)
                actual = _apply_common(actual, report['common_gate_sequence'], 6)
            expected = basis.to_full(basis.apply_circuit(state, gates, family=family))
            np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=0.)

    def test_jordan_wigner_strings_are_present_only_in_fermionic_family(self):
        whole = excitation_decomposition((0, 4), .2, family='whole_qubit', n_qubits=6)
        fermion = excitation_decomposition((0, 4), .2, family='fermionic', n_qubits=6)
        self.assertTrue(all(row['pauli_word'][1:4] == 'III' for row in whole['pauli_terms']))
        self.assertTrue(all(row['pauli_word'][1:4] == 'ZZZ' for row in fermion['pauli_terms']))
        whole_matrix = _apply_common(np.eye(64), whole['common_gate_sequence'], 6)
        fermion_matrix = _apply_common(np.eye(64), fermion['common_gate_sequence'], 6)
        self.assertGreater(np.linalg.norm(whole_matrix - fermion_matrix), .1)

    def test_exact_zero_is_removed_without_epsilon_threshold(self):
        for theta in (0., -0.):
            value = excitation_decomposition((0, 1, 4, 5), theta, family='fermionic', n_qubits=6)
            self.assertTrue(value['exact_zero_omitted'])
            self.assertEqual(value['common_gate_sequence'], [])
        tiny = excitation_decomposition((0, 1, 4, 5), 1e-20, family='fermionic', n_qubits=6)
        self.assertFalse(tiny['exact_zero_omitted'])
        self.assertTrue(tiny['common_gate_sequence'])

    def test_hf_report_counts_preparation_separately(self):
        query = {'kind': 'hf', 'geometry_id': 'CO2_I01'}
        report = resource_report(query, self.catalogue)
        self.assertEqual(report['query'], query)
        self.assertEqual(report['query_sha256'], digest(query))
        self.assertEqual(report['resources_excluding_hf']['gates'], 0)
        self.assertEqual(report['resources_excluding_hf']['depth'], 0)
        self.assertEqual(report['resources_including_hf']['gates'], 22)
        self.assertEqual(report['resources_including_hf']['depth'], 1)
        self.assertEqual(report['resources_including_hf']['two_qubit_count'], 0)
        self.assertEqual([row['wires'][0] for row in report['hf_preparation_sequence']], list(range(22)))

    def test_discrete_double_count_depth_and_input_identity_are_explicit(self):
        template = next(i for i, row in enumerate(self.catalogue['templates']) if row['kind'] == 'double')
        query = {'kind': 'discrete', 'ops': [1 + 8 * template] + [0] * 31, 'geometry_id': 'CO2_I02'}
        before = deepcopy(query)
        report = resource_report(query, self.catalogue)
        self.assertEqual(query, before)
        self.assertEqual(report['native_family'], 'whole_qubit')
        self.assertEqual(len(report['native_recipe']), 32)
        self.assertEqual(sum(x.get('exact_identity_omitted', False) for x in report['native_recipe']), 31)
        self.assertEqual(len(report['native_recipe'][0]['pauli_terms']), 8)
        self.assertEqual(report['two_qubit_count'], 8 * 6)
        self.assertEqual(report['resources_excluding_hf']['gates'], 8 * 15)
        self.assertEqual(report['resources_including_hf']['gates'], 8 * 15 + 22)
        self.assertEqual(report['resources_including_hf']['two_qubit_count'], report['two_qubit_count'])
        self.assertEqual(report['common_sequence_sha256'], digest(report['common_gate_sequence']))
        self.assertGreater(report['depth'], 0)
        self.assertFalse(report['hardware_advantage_claimed'])

    def test_vqe_exact_parameter_order_and_fermionic_recipe(self):
        parameters = [.12345678901234567] + [0.] * 23
        report = resource_report({'kind': 'vqe', 'parameters': parameters, 'geometry_id': 'CO2_I01'}, self.catalogue)
        self.assertEqual(report['native_family'], 'fermionic')
        self.assertEqual(len(report['native_recipe']), 24)
        self.assertEqual([r['template_index'] for r in report['native_recipe']], self.catalogue['vqe_template_indices'])
        self.assertEqual(report['native_recipe'][0]['angle_hex'], parameters[0].hex())
        self.assertTrue(all(r['exact_zero_omitted'] for r in report['native_recipe'][1:]))
        first = report['native_recipe'][0]
        self.assertEqual(first['wires'], self.catalogue['templates'][0]['wires'])
        self.assertEqual(first['common_sequence_range'], [0, len(report['common_gate_sequence'])])
        self.assertFalse(any(r['gate'] == 'x' for r in report['common_gate_sequence']))

    def test_invalid_native_queries_and_angles_refused(self):
        for bad in ({'kind': 'discrete', 'ops': [0] * 31},
                    {'kind': 'discrete', 'ops': [True] * 32},
                    {'kind': 'vqe', 'parameters': [0.] * 23},
                    {'kind': 'vqe', 'parameters': [math.nan] * 24},
                    {'kind': 'vqe', 'parameters': [True] * 24},
                    {'kind': 'hf', 'energy': -100.}, {'kind': 'hf', 'geometry_id': 'other'}):
            with self.subTest(query=bad), self.assertRaises(ValueError):
                resource_report(bad, self.catalogue)
        for bad_theta in (True, math.nan, math.inf):
            with self.assertRaises(ValueError):
                excitation_decomposition((0, 2), bad_theta, family='fermionic', n_qubits=4)
        with self.assertRaises(ValueError):
            excitation_decomposition((0, 0), .2, family='whole_qubit', n_qubits=4)


if __name__ == '__main__':
    unittest.main()

