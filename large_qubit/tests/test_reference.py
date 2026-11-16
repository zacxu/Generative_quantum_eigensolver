"""Iterative reference gates on independent bounded matrix fixtures."""

import importlib.util
import unittest

import numpy as np

from large_qubit.reference import ReferenceFailure, solve_reference


class TinyOperatorFixture:
    """Independent test operator; production reference code receives no matrix."""

    def __init__(self, matrix, spin_squared=None):
        self.matrix = np.asarray(matrix, dtype=float)
        self.dimension = len(matrix)
        self.shape = (self.dimension, 1)
        self.hamiltonian_identity = 'tiny-test-only-operator'
        self.backend_metadata = {'backend': 'independent-tiny-fixture'}
        self._stats = dict(apply_h_calls=0, complex_actions=0, real_actions=0,
                           kernel_contractions=0, hamiltonian_action_seconds=0.)
        self.spin_squared = np.zeros_like(self.matrix) if spin_squared is None else spin_squared

    def statistics(self):
        return dict(self._stats)

    def apply_h(self, state):
        self._stats['apply_h_calls'] += 1
        self._stats['complex_actions'] += 1
        self._stats['kernel_contractions'] += 2
        return (self.matrix @ state.reshape(-1)).reshape(self.shape).astype(complex)

    def apply_h_real(self, state):
        self._stats['apply_h_calls'] += 1
        self._stats['real_actions'] += 1
        self._stats['kernel_contractions'] += 1
        return (self.matrix @ state.reshape(-1)).reshape(self.shape)

    def validate_real_path(self, **kwargs):
        vector = np.ones(self.shape) / np.sqrt(self.dimension)
        np.testing.assert_allclose(self.apply_h(vector), self.apply_h_real(vector))
        return {'all_passed': True}

    def spin_diagnostics(self, state):
        ss = self.spin_squared @ state.reshape(-1)
        expected = np.vdot(state.reshape(-1), ss).real
        return {'s_squared_expectation': float(expected),
                's_squared_variance': float(np.vdot(ss, ss).real - expected ** 2),
                'total_spin_enforced_by_solver': False}


class IterativeReferenceTest(unittest.TestCase):
    def test_lowest_root_matches_complete_fixture_spectrum(self):
        rng = np.random.default_rng(422)
        a = rng.normal(size=(8, 8))
        h = (a + a.T) / 2
        outcome = solve_reference(TinyOperatorFixture(h), ncv=8)
        self.assertTrue(outcome['report']['all_passed'])
        self.assertAlmostEqual(outcome['energy'], np.linalg.eigvalsh(h)[0], places=11)
        self.assertLess(outcome['report']['maximum_complex128_residual_norm_ha'], 1e-8)
        self.assertEqual(outcome['report']['ordinary_circuit_energy_requests'], 0)
        self.assertFalse(outcome['report']['rigorous_global_ground_certificate'])

    def test_random_full_sector_starts_find_lower_triplet_missed_by_hf_start(self):
        # |00> and |11> are closed-shell singlets.  The antisymmetric open-shell
        # combination has S=1 and energy -1, below the HF eigenstate at +3.
        h = np.array([[3., 0., 0., 0.], [0., 0., 1., 0.],
                      [0., 1., 0., 0.], [0., 0., 0., 4.]])
        ss = np.array([[0., 0., 0., 0.], [0., 1., -1., 0.],
                       [0., -1., 1., 0.], [0., 0., 0., 0.]])
        outcome = solve_reference(TinyOperatorFixture(h, ss), ncv=4)
        self.assertAlmostEqual(outcome['energy'], -1., places=12)
        self.assertAlmostEqual(outcome['report']['spin_diagnostics']['s_squared_expectation'], 2., places=12)
        self.assertFalse(outcome['report']['spin_diagnostics']['total_spin_enforced_by_solver'])

    def test_single_determinant_sector_has_explicit_exact_action_path(self):
        outcome = solve_reference(TinyOperatorFixture([[5.7]]))
        self.assertAlmostEqual(outcome['energy'], 5.7)
        self.assertEqual(outcome['state'].dtype, np.complex128)
        self.assertEqual(outcome['report']['actual_ncv'], 1)

    def test_matvec_budget_failure_keeps_unconverged_report(self):
        with self.assertRaises(ReferenceFailure) as raised:
            solve_reference(TinyOperatorFixture(np.diag(np.arange(20.))), max_matvecs=4)
        self.assertFalse(raised.exception.report['all_passed'])
        self.assertFalse(raised.exception.report['converged'])
        self.assertIn('budget_exhausted', raised.exception.report['failure_reason'])
        self.assertGreaterEqual(raised.exception.report['hamiltonian_actions'], 4)

    def test_residual_gate_cannot_be_bypassed_by_real_solver_convergence(self):
        class IncorrectComplexAction(TinyOperatorFixture):
            def apply_h(self, state):
                return super().apply_h(state) + np.roll(state, 1, axis=0)

            def validate_real_path(self, **kwargs):
                # Test adversarially that direct residual validation still
                # refuses an incorrectly certified optimized real path.
                return {'all_passed': True}

        with self.assertRaises(ReferenceFailure) as raised:
            solve_reference(IncorrectComplexAction(np.diag([1., 2., 3., 4.])), ncv=4)
        self.assertIn('residual_exceeds', raised.exception.report['failure_reason'])

    def test_memory_and_scientific_tolerances_cannot_silently_expand(self):
        fixture = TinyOperatorFixture(np.diag(np.arange(5.)))
        for settings in ({'max_krylov_bytes': 1}, {'residual_tolerance': 1e-5}, {'starts': 1}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    solve_reference(fixture, **settings)

    @unittest.skipUnless(importlib.util.find_spec('pyscf'), 'PySCF native reference check requires Linux chemistry environment')
    def test_native_fixed_sector_reference_can_have_triplet_ground_state(self):
        from large_qubit.evaluator import IntegralEvaluator
        from large_qubit.tests.test_evaluator import dense_integral_fixture
        h1 = np.zeros((2, 2))
        factors = np.array([[[2., 0., 0.], [0., 0., 1.]],
                            [[0., 0., 1.], [0., 2., 0.]]])
        eri = np.einsum('pqk,rsk->pqrs', factors, factors)
        evaluator = IntegralEvaluator(h1, eri, 1.7, 2, (1, 1))
        outcome = solve_reference(evaluator, ncv=4)
        expected = dense_integral_fixture(h1, eri, 1.7, (1, 1))
        self.assertAlmostEqual(outcome['energy'], np.linalg.eigvalsh(expected)[0], places=11)
        self.assertAlmostEqual(outcome['report']['spin_diagnostics']['s_squared_expectation'], 2., places=11)


if __name__ == '__main__':
    unittest.main()
