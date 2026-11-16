"""Fresh v2 streams and unchanged quality math; no physical labels or scoring."""

from copy import deepcopy
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from large_qubit import acquisition as frozen
from large_qubit import acquisition_v2 as current
from large_qubit.catalogue import build_screened_catalogue
from large_qubit.common import digest, sha256_file
from large_qubit.protocol import seed_for
from large_qubit.tests.test_catalogue import synthetic_training


EXPECTED_CYCLE = ('guided', 'uniform', 'local', 'guided', 'local',
                  'guided', 'uniform', 'local', 'guided', 'local')
FROZEN_SHA = '3a0a493c557e267296a2f96573408feb604059d2e969e6566e9188a0554930dd'


def row(alias, catalogue, ops, index, *, ordinal=None, energy=-100.05, role='fitting'):
    return {'row_id': f'{alias}_v2_{"F" if role == "fitting" else "D"}{index:06d}',
            'geometry_id': alias, 'role': role, 'ops': list(ops), 'energy_hartree': energy,
            'catalogue_sha256': catalogue['sha256'], 'request_ordinal': index if ordinal is None else ordinal}


class AcquisitionV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.screening = build_screened_catalogue(synthetic_training())

    def sampler(self, alias='CO2_T01'):
        return current.AcquisitionSampler(alias, self.catalogue, self.screening)

    def test_frozen_source_and_quality_bounds_are_exactly_preserved(self):
        path = Path(frozen.__file__)
        self.assertEqual(sha256_file(path), FROZEN_SHA)
        before = frozen.acquisition_policy(self.catalogue)
        after = current.acquisition_policy(self.catalogue)
        self.assertEqual(current.quality_policy(), frozen.quality_policy())
        self.assertEqual(after['quality_policy']['fixed_weight_lambdas'], [.25, .1])
        self.assertEqual(after['quality_policy']['minimum_weight_ESS_fraction'], .05)
        self.assertEqual(after['quality_policy']['maximum_single_row_weight'], .05)
        self.assertEqual(after['fitting_mixture'], {'guided': .4, 'uniform': .2, 'local': .4})
        self.assertEqual(after['fitting_component_cycle'], list(EXPECTED_CYCLE))
        self.assertEqual(after['acquisition_id'], 'co2-corpus-acquisition-v2')
        self.assertIn('_v2_F', after.pop('row_id_convention'))
        for field in ('schema_version', 'acquisition_id', 'fitting_mixture', 'fitting_component_cycle'):
            after[field] = before[field]
        self.assertEqual(after, before)
        self.assertEqual(sha256_file(path), FROZEN_SHA)

    def test_fresh_namespace_seeds_and_reservations_are_deterministic(self):
        for alias in current.TRAINING_ALIASES:
            sampler = self.sampler(alias)
            reservations = sampler.reserve_diagnostics()
            old = frozen.AcquisitionSampler(alias, self.catalogue, self.screening).reserve_diagnostics()
            self.assertEqual(reservations, self.sampler(alias).reserve_diagnostics())
            self.assertEqual(len(reservations), 128)
            self.assertEqual(len({tuple(item['ops']) for item in reservations}), 128)
            self.assertEqual(sum(item['component'] == 'uniform' for item in reservations), 64)
            self.assertEqual(sum(item['component'] == 'guided' for item in reservations), 64)
            for index, item in enumerate(reservations, 1):
                self.assertEqual(item['seed'], seed_for('acquisition',
                    group=f'co2-corpus-acquisition-v2:{alias}:diagnostic:{index}'))
                self.assertNotEqual(item['seed'], old[index - 1]['seed'])
            self.assertNotEqual(reservations, old)
            reservations[0]['ops'][0] = 99999
            self.assertNotEqual(reservations, sampler.reserve_diagnostics())

    def test_fixed_40_20_40_cycle_replays_using_only_fresh_v2_parents(self):
        sampler, parents, observed = self.sampler(), [], []
        for ordinal in range(1, 21):
            plan = sampler.propose(ordinal, parents)
            self.assertEqual(plan, self.sampler().propose(ordinal, parents))
            self.assertEqual(plan['seed'], seed_for('acquisition',
                group=f'co2-corpus-acquisition-v2:CO2_T01:fitting:{ordinal}'))
            observed.append(plan['requested_component'])
            if plan['component'] == 'local':
                self.assertIn(plan['parent_row_id'], {item['row_id'] for item in parents})
                self.assertIn('_v2_F', plan['parent_row_id'])
                self.assertTrue(plan['recipe']['retain_if_novel_even_when_nonimproving'])
            if tuple(plan['ops']) not in {tuple(item['ops']) for item in parents}:
                parents.append(row('CO2_T01', self.catalogue, plan['ops'], len(parents) + 1,
                                   ordinal=388 + 3 * ordinal))
        self.assertEqual(observed, list(EXPECTED_CYCLE) * 2)
        self.assertEqual({name: observed.count(name) for name in current.COMPONENT_CYCLE},
                         {'guided': 8, 'uniform': 4, 'local': 8})

    def test_empty_v2_local_slot_has_no_old_parent_and_v1_globals_cannot_change_v2(self):
        sampler = self.sampler()
        original = sampler.propose(3, [])
        self.assertEqual(original['requested_component'], 'local')
        self.assertEqual(original['component'], 'guided_no_fitting_parent')
        self.assertIsNone(original['parent_row_id'])
        self.assertNotIn('parent_energy_hartree', original['recipe'])
        with patch.object(frozen, 'ACQUISITION_ID', 'forbidden-old-global'), \
                patch.object(frozen, 'COMPONENT_CYCLE', ('uniform',)), \
                patch.object(frozen.AcquisitionSampler, 'propose', side_effect=AssertionError('no old sampler state')):
            self.assertEqual(self.sampler().propose(3, []), original)
        self.assertEqual(sha256_file(Path(frozen.__file__)), FROZEN_SHA)

    def test_old_diagnostic_heldout_future_or_malformed_parent_rows_are_rejected(self):
        sampler = self.sampler()
        first = sampler.propose(1, [])
        fresh = row('CO2_T01', self.catalogue, first['ops'], 1, ordinal=388)
        self.assertEqual(sampler.propose(3, [fresh])['parent_row_id'], fresh['row_id'])
        bad = [dict(fresh, row_id='CO2_T01_F000001'),
               dict(fresh, row_id='CO2_T01_v1_F000001'),
               dict(fresh, row_id='CO2_T01_v2_D000001'),
               dict(fresh, row_id='CO2_T01_v2_F000000'),
               dict(fresh, role='diagnostic'), dict(fresh, geometry_id='CO2_I01')]
        for value in bad:
            with self.subTest(row_id=value['row_id'], role=value['role']), self.assertRaises(ValueError):
                sampler.propose(3, [value])
        with self.assertRaisesRegex(ValueError, 'future'):
            sampler.propose(1, [fresh])

    def test_local_best32_and_one_to_three_edit_recipe_are_unchanged(self):
        rng = random.Random(912)
        parents = [row('CO2_T01', self.catalogue, [rng.randrange(513) for _ in range(32)], index + 1,
                       ordinal=388 + 3 * index, energy=float(index)) for index in range(40)]
        plan = self.sampler().propose(43, parents)
        self.assertEqual(plan['component'], 'local')
        self.assertIn(plan['parent_row_id'], {item['row_id'] for item in parents[:32]})
        self.assertGreaterEqual(len(plan['recipe']['mutations']), 1)
        self.assertLessEqual(len(plan['recipe']['mutations']), 3)
        self.assertEqual(len(plan['ops']), 32)
        self.assertTrue(all(0 <= op <= 512 for op in plan['ops']))


class QualityV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.screening = build_screened_catalogue(synthetic_training())
        cls.fitting, cls.diagnostics = {}, {}
        for index, alias in enumerate(current.TRAINING_ALIASES):
            rng = random.Random(601 + index)
            sampler = current.AcquisitionSampler(alias, cls.catalogue, cls.screening)
            cls.fitting[alias] = [row(alias, cls.catalogue, [rng.randrange(513) for _ in range(32)], number + 1,
                                     ordinal=388 + index + 3 * number) for number in range(1024)]
            cls.diagnostics[alias] = [row(alias, cls.catalogue, plan['ops'], number + 1,
                ordinal=4 + index + 3 * number, role='diagnostic', energy=-99.)
                for number, plan in enumerate(sampler.reserve_diagnostics())]
        cls.hf = {alias: -100. for alias in current.TRAINING_ALIASES}

    def assess(self, fitting=None, diagnostics=None):
        return current.quality_report(self.fitting if fitting is None else fitting,
            self.diagnostics if diagnostics is None else diagnostics, self.hf, self.catalogue)

    def test_v2_quality_identity_and_frozen_weight_math_match_v1(self):
        report = self.assess()
        baseline = frozen.quality_report(self.fitting, self.diagnostics, self.hf, self.catalogue)
        self.assertTrue(report['all_passed'])
        self.assertEqual(report['schema_version'], 'co2-fitting-prefix-quality-v2')
        self.assertEqual(report['acquisition_id'], current.ACQUISITION_ID)
        self.assertEqual(report['policy_sha256'], digest(current.acquisition_policy(self.catalogue)))
        self.assertEqual(report['quality_policy'], baseline['quality_policy'])
        for alias in current.TRAINING_ALIASES:
            group, previous = report['groups'][alias], baseline['groups'][alias]
            self.assertEqual(group['weight_concentration'], previous['weight_concentration'])
            self.assertEqual(group['checks'], previous['checks'])
            self.assertEqual(group['useful_rows'], 1024)
            self.assertEqual(set(group['weight_concentration']), {'0.25', '0.1'})

    def test_quality_never_reads_diagnostic_labels_and_rejects_v1_row_provenance(self):
        class ForbiddenNumeric:
            def __float__(self):
                raise AssertionError('Diagnostic labels cannot select quality or weights')
        diagnostics = deepcopy(self.diagnostics)
        for group in diagnostics.values():
            for item in group:
                item['energy_hartree'] = ForbiddenNumeric()
        self.assertEqual(self.assess(diagnostics=diagnostics), self.assess())
        fitting = deepcopy(self.fitting)
        fitting['CO2_T01'][0]['row_id'] = 'CO2_T01_F000001'
        with self.assertRaisesRegex(ValueError, 'v2 campaign'):
            self.assess(fitting=fitting)
        diagnostics = deepcopy(self.diagnostics)
        diagnostics['CO2_T01'][0]['row_id'] = 'CO2_T01_D000001'
        with self.assertRaisesRegex(ValueError, 'v2 campaign'):
            self.assess(diagnostics=diagnostics)

    def test_collapsed_tail_still_fails_both_predeclared_concentration_bounds(self):
        fitting = deepcopy(self.fitting)
        fitting['CO2_T03'][0]['energy_hartree'] = -101.
        report = self.assess(fitting=fitting)
        self.assertFalse(report['all_passed'])
        self.assertFalse(report['groups']['CO2_T03']['checks']['fixed_weight_concentration'])
        for values in report['groups']['CO2_T03']['weight_concentration'].values():
            self.assertLess(values['ess_fraction'], .05)
            self.assertGreater(values['maximum_weight'], .05)
        self.assertEqual(report['quality_policy'], frozen.quality_policy())


if __name__ == '__main__':
    unittest.main()
