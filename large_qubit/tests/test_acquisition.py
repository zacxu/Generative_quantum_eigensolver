"""Metadata-only sampler replay and fitting-only corpus quality tests."""

from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from large_qubit.acquisition import (AcquisitionSampler, COMPONENT_CYCLE,
    DIAGNOSTIC_ROWS, REQUEST_QUOTAS, TRAINING_ALIASES, acquisition_policy, quality_report)
from large_qubit.catalogue import build_screened_catalogue
from large_qubit.tests.test_catalogue import synthetic_training


def fitting_row(alias, catalogue, ops, ordinal, energy=-100.05):
    return {'row_id': f'{alias}_F{ordinal:06d}', 'geometry_id': alias, 'role': 'fitting',
            'ops': list(ops), 'energy_hartree': energy, 'catalogue_sha256': catalogue['sha256'],
            'request_ordinal': ordinal}


class AcquisitionSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.screening = build_screened_catalogue(synthetic_training())

    def sampler(self, alias='CO2_T01', **kwargs):
        return AcquisitionSampler(alias, self.catalogue, self.screening, **kwargs)

    def test_reservation_is_deterministic_unique_independent_and_not_mutable(self):
        sampler = self.sampler()
        left, right = sampler.reserve_diagnostics(), self.sampler().reserve_diagnostics()
        self.assertEqual(left, right)
        self.assertEqual(len(left), 128)
        self.assertEqual(len({tuple(row['ops']) for row in left}), 128)
        self.assertEqual(sum(row['component'] == 'uniform' for row in left), 64)
        self.assertEqual(sum(row['component'] == 'guided' for row in left), 64)
        self.assertNotEqual(left, self.sampler('CO2_T02').reserve_diagnostics())
        left[0]['ops'][0] = 999999
        self.assertNotEqual(left, sampler.reserve_diagnostics())

    def test_fixed_mixture_and_resume_replay_use_fitting_parents_only(self):
        sampler, parents, observed = self.sampler(), [], []
        for ordinal in range(1, 11):
            plan = sampler.propose(ordinal, parents)
            observed.append(plan['component'])
            self.assertEqual(plan, self.sampler().propose(ordinal, parents))
            if plan['component'] == 'local':
                self.assertIn(plan['parent_row_id'], {row['row_id'] for row in parents})
                self.assertTrue(plan['recipe']['retain_if_novel_even_when_nonimproving'])
            if tuple(plan['ops']) not in {tuple(row['ops']) for row in parents}:
                parents.append(fitting_row('CO2_T01', self.catalogue, plan['ops'], 387 + 3 * ordinal))
        self.assertEqual(tuple(observed), COMPONENT_CYCLE)
        self.assertEqual(observed.count('guided'), 6)
        self.assertEqual(observed.count('uniform'), 2)
        self.assertEqual(observed.count('local'), 2)

    def test_guided_every_circuit_preserves_mandatory_all_wire_cover(self):
        sampler = self.sampler()
        for ordinal in (1, 3, 5, 6, 8, 10, 21, 31):
            plan = sampler.propose(ordinal, [])
            template_ids = {(operator - 1) // 8 for operator in plan['ops']}
            self.assertEqual(len(template_ids), 32)
            self.assertEqual({wire for index in template_ids for wire in self.catalogue['templates'][index]['wires']},
                             set(range(30)))
            self.assertTrue(all(1 <= operator <= 512 for operator in plan['ops']))

    def test_duplicate_and_reserved_proposals_are_not_silently_filtered(self):
        sampler = self.sampler()
        first = sampler.propose(11, [])
        observed = fitting_row('CO2_T01', self.catalogue, first['ops'], 400)
        self.assertEqual(sampler.propose(11, [observed]), first)
        diagnostic = sampler.reserve_diagnostics()[0]['ops']
        with patch.object(sampler, '_guided', return_value=(diagnostic, {'fixture': 'reserved collision'})):
            self.assertEqual(sampler.propose(1, [])['ops'], diagnostic)

    def test_role_alias_future_parent_and_quota_guards_fail_closed(self):
        sampler = self.sampler()
        initial = sampler.propose(1, [])
        row = fitting_row('CO2_T01', self.catalogue, initial['ops'], 400)
        with self.assertRaisesRegex(ValueError, 'future'):
            sampler.propose(1, [row])
        for role, alias in (('diagnostic', 'CO2_T01'), ('pilot', 'CO2_T01'), ('fitting', 'CO2_I01')):
            changed = {**row, 'role': role, 'geometry_id': alias}
            with self.subTest(role=role, alias=alias), self.assertRaises(ValueError):
                sampler.propose(4, [changed])
        with self.assertRaisesRegex(ValueError, 'quota'):
            sampler.propose(REQUEST_QUOTAS['CO2_T01'] - 128, [])
        for alias in ('CO2_V01', 'CO2_I01', 'CO2_I02'):
            with self.assertRaises(ValueError):
                self.sampler(alias)

    def test_local_parent_selection_is_bounded_best32_with_deterministic_ties(self):
        sampler, rng = self.sampler(), random.Random(772)
        parents = [fitting_row('CO2_T01', self.catalogue,
                              [rng.randrange(513) for _ in range(32)], 400 + index, float(index))
                   for index in range(40)]
        plan = sampler.propose(44, parents)
        self.assertEqual(plan['component'], 'local')
        self.assertIn(plan['parent_row_id'], {row['row_id'] for row in parents[:32]})
        self.assertGreaterEqual(len(plan['recipe']['mutations']), 1)
        self.assertLessEqual(len(plan['recipe']['mutations']), 3)

    def test_policy_accounts_for_every_hf_diagnostic_duplicate_and_local_request(self):
        policy = acquisition_policy(self.catalogue)
        self.assertEqual(sum(policy['geometry_request_quotas'].values()), 8192)
        self.assertEqual(policy['fitting_goal_per_geometry'], 2048)
        self.assertTrue(policy['poor_or_nonimproving_local_rows_retained'])
        self.assertFalse(policy['model_training_executed'])
        self.assertEqual(policy['quality_policy']['minimum_weight_ESS_fraction'], .05)


class FittingQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.screening = build_screened_catalogue(synthetic_training())
        cls.fitting, cls.diagnostics = {}, {}
        for index, alias in enumerate(TRAINING_ALIASES):
            sampler = AcquisitionSampler(alias, cls.catalogue, cls.screening)
            rng = random.Random(824 + index)
            cls.fitting[alias] = [fitting_row(alias, cls.catalogue,
                [rng.randrange(513) for _ in range(32)], 388 + index + 3 * ordinal)
                for ordinal in range(1024)]
            cls.diagnostics[alias] = [{'row_id': f'{alias}_D{ordinal + 1:03d}',
                'geometry_id': alias, 'role': 'diagnostic', 'ops': proposal['ops'],
                'energy_hartree': -99., 'catalogue_sha256': cls.catalogue['sha256'],
                'request_ordinal': 4 + index * 128 + ordinal}
                for ordinal, proposal in enumerate(sampler.reserve_diagnostics())]
        cls.hf = {alias: -100. for alias in TRAINING_ALIASES}

    def assess(self, fitting=None, diagnostics=None, hf=None):
        return quality_report(self.fitting if fitting is None else fitting,
                              self.diagnostics if diagnostics is None else diagnostics,
                              self.hf if hf is None else hf, self.catalogue)

    def test_balanced_fitting_prefix_quality_passes_and_uses_both_fixed_weights(self):
        report = self.assess()
        self.assertTrue(report['all_passed'])
        self.assertEqual(report['rows_per_geometry'], 1024)
        self.assertFalse(report['diagnostic_energies_used_for_quality'])
        for group in report['groups'].values():
            self.assertEqual(group['useful_rows'], 1024)
            self.assertEqual(set(group['weight_concentration']), {'0.25', '0.1'})
            self.assertGreater(group['mean_normalized_hamming_distance'], .9)

    def test_diagnostic_numeric_labels_are_never_read_for_quality(self):
        diagnostics = deepcopy(self.diagnostics)
        class ForbiddenNumeric:
            def __float__(self):
                raise AssertionError('Diagnostic labels must not affect fitting quality')
        for rows in diagnostics.values():
            for row in rows:
                row['energy_hartree'] = ForbiddenNumeric()
        self.assertEqual(self.assess(diagnostics=diagnostics), self.assess())

    def test_collapsed_fixed_weight_tail_fails_without_changing_lambdas(self):
        fitting = deepcopy(self.fitting)
        fitting['CO2_T01'][0]['energy_hartree'] = -101.
        report = self.assess(fitting=fitting)
        self.assertFalse(report['all_passed'])
        self.assertFalse(report['groups']['CO2_T01']['checks']['fixed_weight_concentration'])
        self.assertEqual(report['quality_policy']['fixed_weight_lambdas'], [.25, .1])

    def test_weak_hf_improvements_and_low_string_diversity_fail_independently(self):
        fitting = deepcopy(self.fitting)
        for row in fitting['CO2_T01']:
            row['energy_hartree'] = -100.001
        report = self.assess(fitting=fitting)
        self.assertFalse(report['groups']['CO2_T01']['checks']['useful_tail'])
        self.assertFalse(report['groups']['CO2_T01']['checks']['best_HF_improvement'])
        fitting = deepcopy(self.fitting)
        for index, row in enumerate(fitting['CO2_T01']):
            row['ops'] = [index % 513, index // 513] + [1] * 30
        report = self.assess(fitting=fitting)
        self.assertFalse(report['groups']['CO2_T01']['checks']['circuit_diversity'])

    def test_diagnostics_overlap_unbalanced_and_unsorted_prefixes_are_refused(self):
        fitting = deepcopy(self.fitting)
        fitting['CO2_T01'][0]['ops'] = list(self.diagnostics['CO2_T01'][0]['ops'])
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.assess(fitting=fitting)
        fitting = deepcopy(self.fitting)
        fitting['CO2_T01'] = fitting['CO2_T01'][:-1]
        with self.assertRaisesRegex(ValueError, 'balanced'):
            self.assess(fitting=fitting)
        fitting = deepcopy(self.fitting)
        fitting['CO2_T01'][0], fitting['CO2_T01'][1] = fitting['CO2_T01'][1], fitting['CO2_T01'][0]
        with self.assertRaisesRegex(ValueError, 'acquisition order'):
            self.assess(fitting=fitting)


if __name__ == '__main__':
    unittest.main()
