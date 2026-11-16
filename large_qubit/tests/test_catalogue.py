"""No native chemistry: exact small-fixture coupling/sign and catalogue checks."""

import copy
import itertools
import unittest

import numpy as np

from large_qubit.catalogue import (ANGLES, TRAINING_ALIASES, build_full_catalogue,
                                   build_screened_catalogue, circuit_gates,
                                   decode_operator, encode_operator, factored_recipe,
                                   full_templates, random_circuit, screening_policy,
                                   template_metrics, validate_catalogue,
                                   validate_operator_ids)
from large_qubit.common import digest
from large_qubit.sector import SectorBasis
from large_qubit.tests.test_evaluator import dense_integral_fixture, random_integrals


def synthetic_training():
    result = {}
    for ordinal, alias in enumerate(TRAINING_ALIASES):
        _, eri = random_integrals(15, seed=117 + ordinal)
        eri *= .001
        desired_fock = np.diag(np.r_[np.linspace(-15., -1., 11), np.linspace(.5, 2., 4)])
        h1 = desired_fock - 2 * np.einsum('pqii->pq', eri[:, :, :11, :11]) + np.einsum('piiq->pq', eri[:, :11, :11, :])
        result[alias] = {'geometry_id': alias, 'partition': 'training',
                         'hamiltonian_fingerprint': digest({'synthetic_fixture': alias}),
                         'h1': h1, 'eri': eri}
    return result


def physical_sector_matrix(h1, eri, nalpha):
    """Convert an independent blocked-fermion matrix by explicit inversion count."""
    norb = len(h1)
    strings = sorted(itertools.combinations(range(norb), nalpha),
                     key=lambda occupied: sum(1 << orbital for orbital in occupied))
    occupations, phases = [], []
    for alpha in strings:
        for beta in strings:
            blocked = [2 * orbital for orbital in alpha] + [2 * orbital + 1 for orbital in beta]
            inversions = sum(blocked[i] > blocked[j] for i in range(len(blocked))
                             for j in range(i + 1, len(blocked)))
            phases.append((-1) ** inversions)
            occupations.append(frozenset(blocked))
    blocked_matrix = dense_integral_fixture(h1, eri, 0., (nalpha, nalpha))
    phases = np.asarray(phases)
    return phases[:, None] * blocked_matrix * phases[None, :], occupations


class FullCatalogueTest(unittest.TestCase):
    def test_exact_full_counts_and_spin_conservation(self):
        catalogue = build_full_catalogue()
        self.assertEqual(catalogue['counts'], {'singles': 88, 'doubles': 2596,
                                              'templates': 2684, 'operators': 21473})
        for template in catalogue['templates']:
            wires = template['wires']
            half = len(wires) // 2
            self.assertTrue(all(wire < 22 for wire in wires[:half]))
            self.assertTrue(all(wire >= 22 for wire in wires[half:]))
            self.assertEqual(sum(wire % 2 for wire in wires[:half]),
                             sum(wire % 2 for wire in wires[half:]))
        validate_catalogue(catalogue)

    def test_id_angle_roundtrip_and_identity_are_fresh(self):
        catalogue = build_full_catalogue(4, 8)
        self.assertEqual(decode_operator(0, catalogue)['name'], 'Identity')
        for template in catalogue['templates']:
            for angle_index, angle in enumerate(ANGLES):
                value = encode_operator(template['template_id'], angle_index, catalogue)
                decoded = decode_operator(value, catalogue)
                self.assertEqual(decoded['wires'], template['wires'])
                self.assertEqual(decoded['angle'], angle)
                self.assertEqual(decoded['template_id'], template['template_id'])

    def test_invalid_ids_and_incomplete_circuits_are_rejected(self):
        catalogue = build_full_catalogue(2, 4)
        for value in (-1, catalogue['max_operator_id'] + 1, True, 1., '1'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_operator_ids([value], catalogue)
        with self.assertRaises(ValueError):
            validate_operator_ids([1, 2], catalogue, length=3)
        with self.assertRaises(ValueError):
            validate_operator_ids('123', catalogue)

    def test_semantic_tampering_is_rejected_even_with_recomputed_checksum(self):
        catalogue = build_full_catalogue(2, 4)
        catalogue['templates'][0]['wires'] = [0, 3]
        catalogue['sha256'] = digest({key: value for key, value in catalogue.items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'semantics'):
            validate_catalogue(catalogue)


class ExactHFCouplingTest(unittest.TestCase):
    def test_all_single_double_couplings_match_independent_second_quantization(self):
        h1, eri = random_integrals(4, seed=516)
        matrix, occupations = physical_sector_matrix(h1, eri, 2)
        addresses = {occupied: index for index, occupied in enumerate(occupations)}
        reference = set(range(4))
        records = template_metrics(h1, eri, nalpha=2)
        self.assertEqual(len(records), len(full_templates(4, 8)))
        for record in records:
            wires = record['wires']
            half = len(wires) // 2
            target = frozenset((reference - set(wires[:half])) | set(wires[half:]))
            expected = matrix[addresses[target], 0]
            self.assertAlmostEqual(record['coupling'], expected, places=11,
                                   msg=str(wires))

    def test_derivative_signs_match_both_independent_matrix_gate_families(self):
        h1, eri = random_integrals(4, seed=516)
        matrix, _ = physical_sector_matrix(h1, eri, 2)
        basis = SectorBasis(4, 2, 2)
        initial, epsilon = basis.hf_state(), 1e-6
        for record in template_metrics(h1, eri, nalpha=2):
            for family, derivative_key in [('whole_qubit', 'whole_qubit_hf_derivative'),
                                            ('fermionic', 'fermionic_hf_derivative')]:
                plus = basis.apply_gate(initial, record['wires'], epsilon, family=family)
                minus = basis.apply_gate(initial, record['wires'], -epsilon, family=family)
                derivative = (np.vdot(plus, matrix @ plus).real - np.vdot(minus, matrix @ minus).real) / (2 * epsilon)
                self.assertAlmostEqual(derivative, record[derivative_key], delta=1e-7,
                                       msg=f'{family} {record["wires"]}')

    def test_bad_integral_symmetry_is_rejected(self):
        h1, eri = random_integrals(4)
        eri[0, 1, 2, 3] += .1
        with self.assertRaisesRegex(ValueError, 'symmetries'):
            template_metrics(h1, eri, nalpha=2)


class ScreenedCatalogueTest(unittest.TestCase):
    def setUp(self):
        self.inputs = synthetic_training()

    def test_deterministic_64_and_24_cover_all_wires_without_energy_requests(self):
        catalogue, report = build_screened_catalogue(self.inputs)
        repeated, repeated_report = build_screened_catalogue(dict(reversed(list(self.inputs.items()))))
        self.assertEqual(catalogue, repeated)
        self.assertEqual(report, repeated_report)
        self.assertEqual(catalogue['counts']['templates'], 64)
        self.assertEqual(catalogue['counts']['operators'], 513)
        self.assertEqual(catalogue['covered_wires'], list(range(30)))
        self.assertEqual(catalogue['covered_spatial_orbitals'], list(range(15)))
        compact = catalogue['vqe_template_indices']
        self.assertEqual(len(compact), 24)
        self.assertEqual({wire for index in compact for wire in catalogue['templates'][index]['wires']}, set(range(30)))
        self.assertEqual(report['logical_energy_requests'], 0)
        self.assertEqual(len(report['vqe_initial_angles']), 24)
        self.assertTrue(all(abs(angle) <= .1 for angle in report['vqe_initial_angles']))
        self.assertTrue(all('coupling' not in template and 'score' not in template
                            for template in catalogue['templates']))
        self.assertLessEqual(len(report['coverage_selection_order']), 24)
        validate_catalogue(catalogue)

    def test_screening_forbids_held_out_and_incomplete_training_inputs(self):
        for altered in ({alias: value for alias, value in self.inputs.items() if alias != 'CO2_T03'},
                        {**self.inputs, 'CO2_I01': self.inputs['CO2_T01']}):
            with self.assertRaisesRegex(ValueError, 'three training'):
                build_screened_catalogue(altered)
        altered = copy.deepcopy(self.inputs)
        altered['CO2_T01']['partition'] = 'final_test'
        with self.assertRaisesRegex(ValueError, 'partition'):
            build_screened_catalogue(altered)

    def test_zero_couplings_and_nonstationary_hf_cannot_fake_screened_coverage(self):
        altered = copy.deepcopy(self.inputs)
        for data in altered.values():
            data['eri'][:] = 0
            data['h1'] = np.diag(np.r_[np.linspace(-15., -1., 11), np.linspace(.5, 2., 4)])
        with self.assertRaisesRegex(ValueError, 'Too few'):
            build_screened_catalogue(altered)
        altered = copy.deepcopy(self.inputs)
        altered['CO2_T01']['h1'][0, 11] += .01
        altered['CO2_T01']['h1'][11, 0] += .01
        with self.assertRaisesRegex(ValueError, 'stationarity'):
            build_screened_catalogue(altered)

    def test_fixed_sizes_random_draws_and_factored_recipe(self):
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            build_screened_catalogue(self.inputs, vqe_template_count=16)
        catalogue, _ = build_screened_catalogue(self.inputs)
        left = random_circuit(np.random.default_rng(22), 64, catalogue)
        right = random_circuit(np.random.default_rng(22), 64, catalogue)
        self.assertEqual(left, right)
        self.assertEqual(validate_operator_ids(left, catalogue, length=64), left)
        self.assertEqual(len(factored_recipe(catalogue)['templates']), 64)
        self.assertEqual(circuit_gates([0, 1, 512], catalogue),
                         [(catalogue['templates'][0]['wires'], ANGLES[0]),
                          (catalogue['templates'][-1]['wires'], ANGLES[-1])])
        self.assertFalse(screening_policy()['held_out_access'])


if __name__ == '__main__':
    unittest.main()
