"""Independent tiny-sector second-quantized checks for the native action."""

import importlib.util
import itertools
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from large_qubit.evaluator import IntegralEvaluator, evaluator_policy


HAS_PYSCF = importlib.util.find_spec('pyscf') is not None


def dense_integral_fixture(h1, eri, nuclear, populations):
    """Test-only dense H from explicit creation/annihilation operations.

    Blocked alpha-before-beta spin-orbital order, ascending occupied strings.
    The production evaluator never calls this independent implementation.
    """
    norb = len(h1)
    strings = [sorted(sum(1 << p for p in occupied)
                      for occupied in itertools.combinations(range(norb), count))
               for count in populations]
    determinants = [alpha | (beta << norb) for alpha in strings[0] for beta in strings[1]]
    addresses = {value: index for index, value in enumerate(determinants)}
    result = np.eye(len(determinants)) * nuclear

    def operated(det, operators):
        sign = 1
        for creation, index in operators:
            bit = 1 << index
            occupied = bool(det & bit)
            if creation == occupied:
                return None
            sign *= -1 if (det & (bit - 1)).bit_count() % 2 else 1
            det ^= bit
        return det, sign

    terms = []
    for p, q in itertools.product(range(norb), repeat=2):
        if h1[p, q] != 0:
            for spin in range(2):
                terms.append((h1[p, q], [(False, q + spin * norb), (True, p + spin * norb)]))
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        if eri[p, q, r, s] != 0:
            for sigma, tau in itertools.product(range(2), repeat=2):
                terms.append((.5 * eri[p, q, r, s], [
                    (False, q + sigma * norb), (False, s + tau * norb),
                    (True, r + tau * norb), (True, p + sigma * norb)]))
    for column, determinant in enumerate(determinants):
        for coefficient, operators in terms:
            applied = operated(determinant, operators)
            if applied is not None:
                target, sign = applied
                result[addresses[target], column] += coefficient * sign
    return result


def random_integrals(norb, seed=112):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(norb, norb))
    h1 = (raw + raw.T) / 2
    factors = rng.normal(size=(norb, norb, 3))
    factors = (factors + factors.swapaxes(0, 1)) / 2
    eri = np.einsum('pqk,rsk->pqrs', factors, factors)
    return h1, eri


class EvaluatorInputTest(unittest.TestCase):
    def test_bad_inputs_are_rejected_before_loading_native_backend(self):
        h1, eri = random_integrals(2)
        for first, second, nuclear, norb, populations, kwargs in [
                (h1, eri, np.nan, 2, (1, 1), {}),
                (h1, eri, 0., 2, (True, 1), {}),
                (h1.astype(complex), eri, 0., 2, (1, 1), {}),
                (h1, eri.reshape(4, 4), 0., 2, (1, 1), {}),
                (h1, eri, 0., 2, (1, 1), {'max_dimension': 3}),
                (h1, eri, 0., 2, (0, 0), {})]:
            with self.subTest(populations=populations, norb=norb):
                with patch('large_qubit.evaluator._backend', side_effect=AssertionError('native load forbidden')):
                    with self.assertRaises(ValueError):
                        IntegralEvaluator(first, second, nuclear, norb, populations, **kwargs)

    def test_wrong_backend_release_rejected(self):
        h1, eri = random_integrals(2)
        with patch('large_qubit.evaluator.importlib.import_module',
                   return_value=SimpleNamespace(__version__='2.12.0')):
            with self.assertRaisesRegex(ValueError, 'Unsupported PySCF'):
                IntegralEvaluator(h1, eri, 0., 2, (1, 1))

    def test_policy_uses_full_integrals_and_conjugating_inner_product(self):
        policy = evaluator_policy()
        self.assertIn('fac=0.5', policy['operator'])
        self.assertIn('numpy.vdot', policy['energy_inner_product'])
        self.assertFalse(policy['dense_sector_matrix_constructed'])


@unittest.skipUnless(HAS_PYSCF, 'PySCF native tests require the isolated Linux chemistry environment')
class NativeActionTest(unittest.TestCase):
    def test_all_tiny_sector_columns_match_independent_fermionic_construction(self):
        for norb, populations in [(2, (1, 1)), (3, (2, 1)), (3, (2, 2))]:
            with self.subTest(norb=norb, populations=populations):
                h1, eri = random_integrals(norb)
                evaluator = IntegralEvaluator(h1, eri, 1.73, norb, populations)
                expected = dense_integral_fixture(h1, eri, 1.73, populations)
                columns = []
                for index in range(evaluator.dimension):
                    state = np.zeros(evaluator.shape, dtype=complex)
                    state.reshape(-1)[index] = 1
                    columns.append(evaluator.apply_h(state).reshape(-1))
                np.testing.assert_allclose(np.column_stack(columns), expected, atol=1e-11)
                np.testing.assert_allclose(evaluator.diagonal(), expected.diagonal(), atol=1e-11)
                self.assertFalse(evaluator.diagonal().flags.writeable)

    def test_complex_action_energy_linearity_and_nuclear_constant_once(self):
        h1, eri = random_integrals(3)
        evaluator = IntegralEvaluator(h1, eri, 2.31, 3, (1, 2))
        electronic = dense_integral_fixture(h1, eri, 0., (1, 2))
        rng = np.random.default_rng(82)
        state = rng.normal(size=evaluator.shape) + 1j * rng.normal(size=evaluator.shape)
        state /= np.linalg.norm(state)
        expected = electronic @ state.reshape(-1) + 2.31 * state.reshape(-1)
        np.testing.assert_allclose(evaluator.apply_h(state).reshape(-1), expected, atol=1e-11)
        self.assertAlmostEqual(evaluator.energy(state), np.vdot(state.reshape(-1), expected).real, places=11)
        np.testing.assert_allclose(evaluator.apply_h(2j * state), 2j * evaluator.apply_h(state), atol=1e-11)
        self.assertGreater(abs(np.dot(state.reshape(-1), expected) - np.vdot(state.reshape(-1), expected)), 1e-3)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            evaluator.energy(state * 2)

    def test_real_path_requires_and_passes_complex128_validation(self):
        h1, eri = random_integrals(2)
        evaluator = IntegralEvaluator(h1, eri, .4, 2, (1, 1))
        with self.assertRaisesRegex(ValueError, 'Validate'):
            evaluator.apply_h_real(evaluator.hf_state(dtype=np.float64))
        self.assertTrue(evaluator.validate_real_path()['all_passed'])
        np.testing.assert_allclose(evaluator.apply_h_real(evaluator.hf_state(dtype=np.float64)),
                                   evaluator.apply_h(evaluator.hf_state()), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'imaginary'):
            evaluator.apply_h_real(evaluator.hf_state())

    def test_hf_energy_equals_independent_closed_shell_contraction(self):
        from large_qubit.orbitals import closed_shell_hf_energy
        h1, eri = random_integrals(4)
        evaluator = IntegralEvaluator(h1, eri, 3.1, 4, (2, 2))
        self.assertAlmostEqual(evaluator.energy(evaluator.hf_state()),
                               closed_shell_hf_energy(h1, eri, 2, 3.1), places=12)

    def test_flat_ci_reshape_and_invalid_states_rejected(self):
        h1, eri = random_integrals(2)
        evaluator = IntegralEvaluator(h1, eri, 0., 2, (1, 1))
        with self.assertRaisesRegex(ValueError, 'two-dimensional'):
            evaluator.apply_h(np.ones(4))
        state = evaluator.hf_state()
        state[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            evaluator.apply_h(state)

    def test_integrals_are_copied_and_native_work_is_counted(self):
        h1, eri = random_integrals(2)
        evaluator = IntegralEvaluator(h1, eri, 0., 2, (1, 1))
        baseline = evaluator.apply_h(evaluator.hf_state())
        h1[:] = 1000
        eri[:] = 1000
        np.testing.assert_allclose(evaluator.apply_h(evaluator.hf_state()), baseline)
        state = np.array([[1., 0], [0, 1j]]) / np.sqrt(2)
        evaluator.apply_h(state)
        stats = evaluator.statistics()
        self.assertEqual(stats['apply_h_calls'], 3)
        self.assertEqual(stats['kernel_contractions'], 4)
        self.assertEqual(stats['dense_sector_matrix_bytes_allocated'], 0)
        self.assertGreater(stats['integral_and_link_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
