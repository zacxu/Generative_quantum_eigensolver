"""Bounded SCF/stability contracts, independent of an installed PySCF backend."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from large_qubit.chemistry import (PhysicalValidationError, inspect_response_hessian,
                                  prepare_geometry, scf_policy)
from large_qubit.definitions import geometry


class ResponseStabilityTest(unittest.TestCase):
    def test_full_response_hessian_detects_unstable_direction(self):
        # A probe vector or one favourable eigenmode cannot certify stability.
        h = np.diag([.3, -.002, .7, 1.1])
        metadata = inspect_response_hessian(lambda vector: h @ vector, 4)
        self.assertFalse(metadata['stable'])
        self.assertEqual(metadata['minimum_eigenvalue'], -.002)
        self.assertLess(metadata['maximum_eigenpair_residual'], 1e-12)

    def test_certifies_all_modes_with_declared_negative_roundoff_tolerance(self):
        h = np.array([[.1, .2], [.2, .6]])
        metadata = inspect_response_hessian(lambda vector: h @ vector, 2)
        self.assertTrue(metadata['stable'])
        np.testing.assert_allclose(metadata['eigenvalues'], np.linalg.eigvalsh(h))
        near_zero = inspect_response_hessian(lambda vector: -1e-9 * vector, 2)
        self.assertTrue(near_zero['stable'])

    def test_rejects_invalid_nonsymmetric_and_unbounded_kernels(self):
        for action, dimension in [(lambda x: np.zeros_like(x), 45),
                                  (lambda x: np.array([[1., 2.], [0., 1.]]) @ x, 2),
                                  (lambda x: np.full_like(x, np.nan), 2),
                                  (lambda x: 1j * x, 2)]:
            with self.subTest(dimension=dimension):
                with self.assertRaises(ValueError):
                    inspect_response_hessian(action, dimension)

    def test_policy_requires_internal_unrestricted_and_complex_stability(self):
        policy = scf_policy()
        self.assertEqual(set(policy['stability_channels']),
                         {'internal_RHF', 'external_RHF_to_UHF', 'external_real_to_complex'})
        self.assertIn('stop', policy['failure_policy'])
        self.assertEqual(policy['maximum_cycles'], 200)
        self.assertFalse(policy['correlated_reference_computed'])


class ChemistryBoundaryTest(unittest.TestCase):
    def test_non_anchor_cannot_be_realized_without_training_anchor(self):
        with self.assertRaisesRegex(ValueError, 'CO2_T02 anchor'):
            prepare_geometry(geometry('CO2_I01', purpose='physical_validation'))

    def test_geometry_cannot_be_silently_changed(self):
        definition = geometry('CO2_T02', purpose='physical_validation')
        definition['coordinates_angstrom'][0][2] = -1.21
        with self.assertRaisesRegex(ValueError, 'Undeclared geometry'):
            prepare_geometry(definition)

    def test_backend_missing_does_not_claim_physics_validation(self):
        with patch('large_qubit.chemistry.importlib.import_module', side_effect=ImportError('not installed')):
            with self.assertRaisesRegex(RuntimeError, 'no SCF or physical validation has run'):
                prepare_geometry(geometry('CO2_T02', purpose='physical_validation'))

    def test_diagnostic_errors_retain_failed_checks(self):
        diagnostics = {'physical_validation_passed': False, 'unstable_mode': -.02}
        error = PhysicalValidationError('SCF unstable', diagnostics)
        self.assertIs(error.diagnostics, diagnostics)

    def test_other_pyscf_version_is_rejected_before_native_construction(self):
        with patch('large_qubit.chemistry.importlib.import_module',
                   return_value=SimpleNamespace(__version__='2.12.0')):
            with self.assertRaisesRegex(ValueError, 'validated 2.11.0'):
                prepare_geometry(geometry('CO2_T02', purpose='physical_validation'))


if __name__ == '__main__':
    unittest.main()
