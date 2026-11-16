"""Synthetic orbital/integral tests; these are not CO2 energy results."""

import unittest

import numpy as np

from large_qubit.orbitals import (align_orbitals, canonicalize_anchor,
                                 closed_shell_hf_energy, orbital_conventions,
                                 transform_integrals, _lexicographic_assignment)


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


class OrbitalGaugeTest(unittest.TestCase):
    def test_anchor_resolves_arbitrary_exact_degenerate_rotations_in_metric(self):
        rng = np.random.default_rng(432)
        a = rng.normal(size=(6, 6))
        s = a.T @ a + np.eye(6)
        w, v = np.linalg.eigh(s)
        inverse_root = (v / np.sqrt(w)) @ v.T
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        c = inverse_root @ q
        energies = [-6, -2, -2, 1, 3, 3]
        occupations = [2, 2, 2, 0, 0, 0]
        u = np.eye(6)
        u[1:3, 1:3] = rotation(0.927)
        u[4:6, 4:6] = rotation(-1.121)
        u[:, 0] *= -1
        first, transform, metadata = canonicalize_anchor(c, energies, s, occupations)
        second, _, _ = canonicalize_anchor(c @ u, energies, s, occupations)
        np.testing.assert_allclose(first, second, atol=1e-12)
        np.testing.assert_allclose(c @ transform, first, atol=1e-12)
        np.testing.assert_allclose(first.T @ s @ first, np.eye(6), atol=1e-12)
        self.assertEqual(metadata['occupation_order'], occupations)

    def test_cross_geometry_permutation_sign_and_degenerate_rotation_recovered(self):
        c = np.eye(6)
        energies = np.array([-6., -2., -2., 1., 3., 3.])
        occupations = np.array([2, 2, 2, 0, 0, 0])
        u = np.eye(6)
        u[1:3, 1:3] = rotation(0.927)
        u[4:6, 4:6] = rotation(-1.121)
        u[:, [0, 3]] *= -1
        permutation = [3, 2, 0, 5, 1, 4]
        target = c @ u[:, permutation]
        # Deliberately not equal to the target AO overlap.  This represents
        # nonorthogonal overlaps between AO sets on two displaced geometries.
        cross = np.diag([.99, .98, .98, .97, .96, .96])
        aligned, transform, metadata = align_orbitals(
            c, target, cross, np.eye(6), energies[permutation], occupations[permutation])
        np.testing.assert_allclose(aligned, c, atol=1e-12)
        np.testing.assert_allclose(target @ transform, c, atol=1e-12)
        self.assertFalse(metadata['occupied_virtual_mixing'])
        np.testing.assert_allclose(metadata['matched_overlap_diagonal'], np.diag(cross))

    def test_alignment_preserves_occupation_sectors_even_when_cross_scores_prefer_mixing(self):
        cross = np.array([[.2, .9], [.9, .2]])
        aligned, _, metadata = align_orbitals(np.eye(2), np.eye(2), cross, np.eye(2),
                                             [-1, 1], [2, 0])
        np.testing.assert_allclose(aligned, np.eye(2))
        self.assertEqual(metadata['input_assignment'], [0, 1])

    def test_ambiguous_disconnected_subspace_fails(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            align_orbitals(np.eye(2), np.eye(2), np.zeros((2, 2)), np.eye(2), [-1, 1], [2, 0])

    def test_degeneracy_threshold_does_not_chain(self):
        _, _, metadata = canonicalize_anchor(np.eye(4), [-1, -1 + 0.75e-7,
                                                        -1 + 1.5e-7, 1], np.eye(4), [2, 2, 2, 0])
        self.assertEqual([len(block['output_indices']) for block in metadata['blocks']], [2, 1, 1])

    def test_assignment_ties_are_lexicographic_but_suboptimal_lex_order_is_rejected(self):
        self.assertEqual(_lexicographic_assignment(np.ones((3, 3))), [0, 1, 2])
        self.assertEqual(_lexicographic_assignment(np.array([[1, 2, 0], [2, 1, 0], [0, 0, 3]])), [1, 0, 2])

    def test_invalid_metric_complex_and_open_shell_inputs_fail(self):
        for c, s, occ in [(np.eye(2), np.diag([1., -1.]), [2, 0]),
                          (np.eye(2, dtype=complex), np.eye(2), [2, 0]),
                          (np.eye(2), np.eye(2), [1, 1]),
                          (2 * np.eye(2), np.eye(2), [2, 0])]:
            with self.subTest(coeff=c, metric=s, occupations=occ):
                with self.assertRaises(ValueError):
                    canonicalize_anchor(c, [-1, 1], s, occ)

    def test_bit_order_and_gauge_policy_are_explicit(self):
        conventions = orbital_conventions()
        self.assertEqual(conventions['anchor_geometry_id'], 'CO2_T02')
        self.assertEqual(conventions['hf_occupation_bits'], '1' * 22 + '0' * 8)
        self.assertIn('most significant', conventions['jordan_wigner_bit_order'])


class IntegralConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(774)
        a = self.rng.normal(size=(4, 4))
        self.h = (a + a.T) / 2
        b = self.rng.normal(size=(4, 4, 5))
        b = (b + b.swapaxes(0, 1)) / 2
        self.eri = np.einsum('pqk,rsk->pqrs', b, b)

    def test_full_eri_transform_preserves_correlated_complex_two_electron_energy(self):
        # One alpha and one beta electron.  This tests the entire ERI tensor,
        # including entries absent from the HF contraction.
        q, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        h, eri = transform_integrals(self.h, self.eri, q)
        psi = self.rng.normal(size=(4, 4)) + 1j * self.rng.normal(size=(4, 4))
        psi /= np.linalg.norm(psi)
        transformed_psi = q.T @ psi @ q

        def energy(one, two, state):
            action = one @ state + state @ one.T + np.einsum('prqs,rs->pq', two, state)
            return np.vdot(state, action)

        self.assertAlmostEqual(energy(self.h, self.eri, psi).real,
                               energy(h, eri, transformed_psi).real, places=12)
        self.assertLess(abs(energy(h, eri, transformed_psi).imag), 1e-12)
        # Only transforming h would be a material and detectable bug.
        self.assertGreater(abs(energy(h, self.eri, transformed_psi).real
                               - energy(self.h, self.eri, psi).real), 1e-3)

    def test_occupied_and_virtual_rotations_preserve_hf_energy_and_nuclear_constant(self):
        q = np.eye(4)
        q[:2, :2] = rotation(.371)
        q[2:, 2:] = rotation(-.841)
        h, eri = transform_integrals(self.h, self.eri, q)
        original = closed_shell_hf_energy(self.h, self.eri, 2, 14.7)
        aligned = closed_shell_hf_energy(h, eri, 2, 14.7)
        self.assertAlmostEqual(original, aligned, places=12)
        self.assertAlmostEqual(aligned - closed_shell_hf_energy(h, eri, 2), 14.7, places=12)

    def test_integral_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            transform_integrals(self.h, self.eri.reshape(16, 16), np.eye(4))
        bad = self.h.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            transform_integrals(bad, self.eri, np.eye(4))


if __name__ == '__main__':
    unittest.main()
