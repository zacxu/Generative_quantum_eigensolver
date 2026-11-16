"""Tamper-resistant pilot evidence fixtures; no CO2 or native energy execution."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section5
from large_qubit.catalogue import build_screened_catalogue
from large_qubit.common import canonical_bytes, digest, read_json, sha256_file
from large_qubit.tests.test_catalogue import synthetic_training
from large_qubit.vqe import optimize_compact


def norm_receipt():
    return {'state_norm': 1., 'norm_error': 0., 'n_alpha': 11, 'n_beta': 11,
            'sector_dimension': 1863225, 'sector_leakage_probability': 0.,
            'population_conservation': 'by_fixed_basis_construction'}


def participation_receipt(active=True):
    occupations = [1.] * 22 + [0.] * 8
    if active:
        for occupied, virtual in zip((0, 2, 4, 6), (22, 24, 26, 28)):
            occupations[occupied] -= .01
            occupations[virtual] += .01
    delta = [abs(value - (1. if wire < 22 else 0.)) for wire, value in enumerate(occupations)]
    wires = [wire for wire, change in enumerate(delta) if change > 1e-6]
    return {'wire_occupations': occupations, 'HF_occupation_changes': delta,
            'active_wires': wires, 'active_spatial_orbitals': sorted({wire // 2 for wire in wires}),
            'effectively_fixed_wires': sorted(set(range(30)) - set(wires)), 'threshold': 1e-6,
            'participation_is_not_a_chemical_accuracy_claim': True}


def geometry_fixture(alias, catalogue, screening, policy, *, active=True):
    """Analytic scalar responses to optimizer queries, with a complete fake ledger."""
    hf = -10. - section5.TRAINING.index(alias)
    ledger, seen, calls = [], {}, 0

    def append(category, family, gates, recipe, energy):
        nonlocal calls
        calls += 1
        key = digest({'geometry': alias, 'family': family, 'gates': gates})
        ledger.append({'event': 'request_started', 'ordinal': calls, 'geometry_id': alias,
                       'category': category, 'family': family, 'recipe': recipe, 'gates': gates,
                       'query_sha256': key, 'duplicate_of': seen.get(key)})
        seen.setdefault(key, calls)
        ledger.append({'event': 'request_completed', 'ordinal': calls, 'geometry_id': alias,
                       'query_sha256': key, 'energy_hartree': float(energy), 'norm': norm_receipt(),
                       'state_preparation_seconds': .001, 'wall_seconds': .01, 'peak_rss_bytes': 1000})
        return float(energy)

    append('HF', 'whole_qubit', [], {'HF': True}, hf)
    discrete, private_discrete = {}, {}
    participation = participation_receipt(active)
    for length in (32, 64):
        rows = []
        for plan in section5.discrete_proposals(catalogue, screening, alias, length):
            value = append('expressivity', 'whole_qubit', section5._discrete_gates(plan['ops'], catalogue),
                           {'length': length, **plan}, hf - .001 * (plan['sample'] + 1))
            rows.append({**plan, 'energy_hartree': value})
        winner = min(rows, key=lambda row: (row['energy_hartree'], tuple(row['ops']), row['sample']))
        covered = sorted({wire for row in rows for gate in section5._discrete_gates(row['ops'], catalogue)
                          for wire in gate['wires']})
        improving = {tuple(row['ops']) for row in rows if hf - row['energy_hartree'] >= .0001}
        passed = (hf - winner['energy_hartree'] >= .001 and len(improving) >= 2
                  and len(participation['active_spatial_orbitals']) >= 8 and covered == list(range(30)))
        discrete[str(length)] = {'passed': passed, 'best_HF_improvement_ha': hf - winner['energy_hartree'],
            'distinct_improving_circuits': len(improving), 'circuits_evaluated': 6,
            'covered_wires_union': covered, 'best_state_participation': copy.deepcopy(participation)}
        private_discrete[str(length)] = {'rows': rows, 'winner': winner}
    runs = []
    for restart, initial in zip(policy['vqe']['restarts'],
                                ([0.] * policy['compact_vqe_parameters'], screening['vqe_initial_angles'])):
        def evaluate(parameters, step):
            value = hf + sum(.01 * (1 - math.cos(theta)) - .002 * math.sin(theta) for theta in parameters)
            return append('VQE', 'fermionic', section5._vqe_gates(parameters, catalogue),
                          {'restart': restart, 'parameters': parameters, 'optimizer_step': step}, value)
        run = optimize_compact(evaluate, initial, max_requests=policy['vqe']['pilot_requests_per_restart'],
                               bounds=tuple(policy['vqe']['proposal_angle_bounds']))
        run['restart'] = restart
        runs.append(run)
    winner = min(runs, key=lambda row: (row['energy'], tuple(row['parameters']), row['restart']))
    required = max(.001, .5 * max(0., max(row['best_HF_improvement_ha'] for row in discrete.values())))
    vqe = {'passed': hf - winner['energy'] >= required, 'best_HF_improvement_ha': hf - winner['energy'],
           'required_improvement_ha': required, 'restarts': 2, 'parameters': policy['compact_vqe_parameters'],
           'ordinary_queries': 2 * policy['vqe']['pilot_requests_per_restart'],
           'maximum_Fourier_identity_error_ha': max(row['interpolation_error'] for run in runs for row in run['history']),
           'state_participation': copy.deepcopy(participation), 'convergence_claimed': False}
    fingerprint = catalogue['training_hamiltonian_bindings'][alias]
    outcome = {'geometry_id': alias, 'hf_energy_hartree': hf, 'discrete': private_discrete,
               'vqe_runs': runs, 'vqe_winner': winner, 'hamiltonian_fingerprint': fingerprint,
               'HF_energy_discrepancy_ha': 0.,
               'participation': {label: copy.deepcopy(participation) for label in ('discrete_32', 'discrete_64', 'VQE')}}
    statistics = {'apply_h_calls': calls + 3, 'complex_actions': calls + 3, 'real_actions': 0,
                  'kernel_contractions': calls + 3, 'hamiltonian_action_seconds': 1.,
                  'initialization_seconds': .1, 'integral_and_link_bytes': 1000,
                  'reference_diagonal_bytes': 0, 'complex128_state_bytes': 1863225 * 16,
                  'float64_state_bytes': 1863225 * 8, 'dense_sector_matrix_bytes_allocated': 0,
                  'real_path_validation': None}
    public = {'geometry_id': alias, 'discrete': discrete, 'vqe': vqe, 'ordinary_requests': calls,
              'peak_rss_bytes': 1000, 'hamiltonian_fingerprint': fingerprint,
              'HF_energy_discrepancy_ha': 0., 'separate_integrity_hamiltonian_actions': 3,
              'total_evaluator_statistics': statistics,
              'saved_winner_integrity': [{'winner': label, 'absolute_energy_discrepancy_ha': 0.,
                                          'normalization': norm_receipt()}
                                         for label in ('discrete_32', 'discrete_64', 'VQE')]}
    return ledger, outcome, public


class Section5EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.screening = build_screened_catalogue(synthetic_training())
        cls.policy = section5.build_section5_policy()
        cls.fixtures = {alias: geometry_fixture(alias, cls.catalogue, cls.screening, cls.policy)
                        for alias in section5.TRAINING}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = 'large_qubit/validation/section5_fixture'
        self.private = f'large_qubit/scoring_only/{section5.BENCHMARK_ID}/section5/section5_fixture'
        for source in section5.SOURCES:
            self.write(source, '# synthetic bound source; never imported\n')
        self.write(section5.CATALOGUE_PATH, self.catalogue)
        self.write(section5.PREPARATION_PATH, {'synthetic_fixture': True})
        self.write('large_qubit/validation/section4_v1/manifest.json', {'synthetic_parent': True})
        results = {}
        for alias, (ledger, outcome, result) in copy.deepcopy(self.fixtures).items():
            self.write_ledger(alias, ledger)
            self.write(f'{self.private}/{alias}_outcomes.json', outcome)
            results[alias] = result
        gate = section5.decision(results)
        self.manifest = {'schema_version': 'section5-validation-v1', 'benchmark_id': section5.BENCHMARK_ID,
            'created_utc': '2026-11-06T00:00:00+00:00', 'output': self.output, 'restricted_output': self.private,
            'policy': self.policy, 'policy_sha256': digest(self.policy),
            'catalogue_sha256': self.catalogue['sha256'],
            'catalogue_file_sha256': sha256_file(self.root / section5.CATALOGUE_PATH),
            'preparation_file_sha256': sha256_file(self.root / section5.PREPARATION_PATH),
            'source_bindings': section5._source_bindings(self.root),
            'section4_manifest_sha256': sha256_file(self.root / 'large_qubit/validation/section4_v1/manifest.json'),
            'results': results, 'decision': gate, 'ordinary_requests': section5.ORDINARY_REQUESTS,
            'pilot_total_including_section4': section5.TOTAL_PILOT_REQUESTS,
            'remaining_pilot_allowance': section5.REMAINING_PILOT_REQUESTS,
            'separate_integrity_hamiltonian_actions': 9, 'correctness_passed': True,
            'section_5_complete': gate['status'] == 'go', 'feasibility_gate': gate['status'],
            'model_training_executed': False, 'heldout_energy_or_reference_access': False,
            'execution_ready': False, 'wall_seconds': 20., 'files': {}}
        self.repin()
        prepared = patch('large_qubit.section5.verify_preparation',
                         return_value=(self.policy, self.catalogue, self.screening, {}))
        parent = patch('large_qubit.section4.verify_section4', return_value={'all_passed': True})
        prepared.start(); parent.start()
        self.addCleanup(prepared.stop); self.addCleanup(parent.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode('utf-8') if type(value) is str else canonical_bytes(value) + b'\n')

    def write_ledger(self, alias, rows):
        self.write(f'{self.private}/{alias}.jsonl', ''.join(canonical_bytes(row).decode('ascii') + '\n' for row in rows))

    def ledger(self, alias='CO2_T01'):
        return [json.loads(line) for line in (self.root / self.private / f'{alias}.jsonl').read_text().splitlines()]

    def repin(self):
        self.manifest['files'] = {path.relative_to(self.root).as_posix(): sha256_file(path)
                                  for path in sorted((self.root / self.private).rglob('*')) if path.is_file()}
        self.write(f'{self.output}/manifest.json', self.manifest)
        self.write(f'{self.output}/manifest.sha256', sha256_file(self.root / self.output / 'manifest.json') + '\n')

    def verify(self, **kwargs):
        return section5.verify_section5(self.root, self.output, **kwargs)

    def test_complete_fixture_replays_all_909_requests_without_native_energy(self):
        with patch('large_qubit.section4.load_co2_evaluator', side_effect=AssertionError('No physics in saved replay')):
            result = self.verify()
        self.assertTrue(result['all_passed'])
        self.assertTrue(result['section_5_complete'])
        self.assertTrue(result['VQE_replayed_from_charged_responses'])
        self.assertEqual(result['ordinary_requests'], 909)
        self.assertEqual(result['additional_hamiltonian_actions'], 0)
        self.assertEqual(result['winning_state_preparations_replayed'], 0)

    def test_rehashed_public_go_flip_cannot_override_private_no_go(self):
        alias = 'CO2_T01'
        ledger, outcome, result = geometry_fixture(alias, self.catalogue, self.screening, self.policy, active=False)
        self.write_ledger(alias, ledger)
        self.write(f'{self.private}/{alias}_outcomes.json', outcome)
        self.manifest['results'][alias] = result
        gate = section5.decision(self.manifest['results'])
        self.manifest.update(decision=gate, feasibility_gate=gate['status'], section_5_complete=False)
        self.repin()
        self.assertEqual(self.verify()['feasibility_gate'], 'no_go')
        for length in ('32', '64'):
            self.manifest['results'][alias]['discrete'][length]['passed'] = True
            self.manifest['results'][alias]['discrete'][length]['best_state_participation'] = participation_receipt(True)
        gate = section5.decision(self.manifest['results'])
        self.manifest.update(decision=gate, feasibility_gate='go', section_5_complete=True)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'public discrete gate'):
            self.verify()

    def test_rehashed_omitted_charged_request_is_rejected(self):
        rows = self.ledger()
        self.write_ledger('CO2_T01', rows[:-2])
        self.repin()
        with self.assertRaisesRegex(ValueError, 'Every issued'):
            self.verify()

    def test_rehashed_optimizer_recipe_and_restart_initialization_are_rejected(self):
        rows = self.ledger()
        first_vqe = 2 * 13
        rows[first_vqe]['recipe']['parameters'][0] = .1
        self.write_ledger('CO2_T01', rows)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'charged request recipe'):
            self.verify()

    def test_rehashed_optimizer_history_cannot_invent_an_incumbent(self):
        relative = f'{self.private}/CO2_T01_outcomes.json'
        outcome = read_json(self.root / relative)
        outcome['vqe_runs'][0]['history'][0]['best_energy'] -= 1.
        self.write(relative, outcome)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'deterministically replayed VQE'):
            self.verify()

    def test_sixth_probe_must_satisfy_five_sample_fourier_identity(self):
        rows = self.ledger()
        # HF +12 discrete +initial +five samples; this completion is the first
        # independent proposal check (one-based request ordinal20).
        changed_key = rows[2 * 19 + 1]['query_sha256']
        for row in rows:
            if row['event'] == 'request_completed' and row['query_sha256'] == changed_key:
                row['energy_hartree'] += .01
        self.write_ledger('CO2_T01', rows)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'five-term'):
            self.verify()

    def test_participation_active_lists_must_follow_all_30_occupations(self):
        relative = f'{self.private}/CO2_T01_outcomes.json'
        outcome = read_json(self.root / relative)
        outcome['participation']['discrete_32']['active_spatial_orbitals'] = list(range(15))
        self.write(relative, outcome)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'derived participation'):
            self.verify()

    def test_finite_types_privacy_scope_sources_and_decision_are_strict(self):
        original = copy.deepcopy(self.manifest)
        mutations = [
            lambda value: value.update(model_training_executed=0),
            lambda value: value.update(ordinary_requests=True),
            lambda value: value.update(wall_seconds=True),
            lambda value: value['decision'].update(selected_length=64),
            lambda value: value.update(reference_energy_ha=-100.),
            lambda value: value.update(restricted_output=self.private + '_other'),
        ]
        for mutate in mutations:
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            self.repin()
            with self.subTest(manifest=self.manifest['decision']), self.assertRaises(ValueError):
                self.verify()
        self.manifest = copy.deepcopy(original)
        self.repin()
        self.write('large_qubit/vqe.py', '# modified bound source')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.verify()

    def test_hf_zero_vqe_request_cannot_be_shifted_from_hf(self):
        rows = self.ledger()
        rows[2 * 13 + 1]['energy_hartree'] -= .001
        self.write_ledger('CO2_T01', rows)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'inconsistent|Zero-angle'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
