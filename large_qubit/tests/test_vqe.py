import math
import unittest

import numpy as np

from large_qubit.vqe import OFFSETS, coordinate_minimum, optimize_compact


class CompactOptimizerTest(unittest.TestCase):
    def test_quartic_minimum_matches_independent_dense_grid_for_generic_curves(self):
        rng = np.random.default_rng(22)
        grid = np.linspace(-.25, .25, 10001)
        for _ in range(25):
            coefficients = rng.normal(size=5)
            def energy(theta):
                return float(coefficients @ [1, np.cos(theta), np.sin(theta),
                                              np.cos(theta / 2), np.sin(theta / 2)])
            center = float(rng.uniform(-.25, .25))
            result = coordinate_minimum([energy(center + offset) for offset in OFFSETS], center=center)
            self.assertLessEqual(result['predicted_energy'], min(map(energy, grid)) + 1e-9)
            self.assertAlmostEqual(result['predicted_energy'], energy(result['theta']), places=12)

    def test_generic_spectator_rotation_needs_half_frequencies(self):
        # Independent Hermitian matrix and exp of a rank-two rotation generator,
        # with populated spectator support, yield all five Fourier terms.
        from scipy.linalg import expm
        rng = np.random.default_rng(4)
        h = rng.normal(size=(4, 4)); h = h + h.T
        generator = np.zeros((4, 4)); generator[0, 1] = -.5; generator[1, 0] = .5
        state = rng.normal(size=4); state /= np.linalg.norm(state)
        def energy(theta):
            vector = expm(theta * generator) @ state
            return float(vector @ h @ vector)
        result = coordinate_minimum([energy(offset) for offset in OFFSETS])
        self.assertAlmostEqual(result['predicted_energy'], energy(result['theta']), places=12)
        self.assertGreater(np.linalg.norm(result['coefficients'][3:]), 1e-3)

    def test_budget_duplicates_and_initializations_are_counted(self):
        calls = []
        def objective(parameters, metadata):
            calls.append((parameters, metadata))
            return float(2 - math.cos(parameters[0] - .13) - math.cos(parameters[1] + .08))
        result = optimize_compact(objective, [0., 0.], max_requests=14)
        self.assertEqual(result['requests'], 13)
        self.assertEqual(len(calls), 13)
        self.assertEqual(result['unused_requests'], 1)
        self.assertEqual(result['completed_sweeps'], 1)
        self.assertLess(result['energy'], result['initial_energy'])
        self.assertFalse(result['convergence_claimed'])

    def test_failed_identity_cannot_be_called_a_valid_optimizer(self):
        with self.assertRaisesRegex(ValueError, 'five-term'):
            optimize_compact(lambda p, _: p[0] ** 4, [0.], max_requests=7)

    def test_full_main_restart_caps_keep_one_unspent_call_without_extra_search(self):
        requests = []
        for cap in (511, 512):
            count = [0]
            def energy(parameters, metadata):
                count[0] += 1
                return float(sum(1 - math.cos(value - .04) for value in parameters))
            result = optimize_compact(energy, [0.] * 24, max_requests=cap)
            self.assertEqual(result['requests'], count[0])
            self.assertLessEqual(count[0], cap)
            self.assertLess(result['energy'], result['initial_energy'])
            requests.append(count[0])
        self.assertEqual(requests, [511, 511])
        self.assertEqual(sum(requests), 1022)


if __name__ == '__main__':
    unittest.main()
