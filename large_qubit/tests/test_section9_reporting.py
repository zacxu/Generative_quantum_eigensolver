"""Synthetic ledger/report fixtures only; no model or scientific evaluation."""

from copy import deepcopy
import unittest

from large_qubit.common import digest
from large_qubit.protocol import METHODS
from large_qubit.section8_policy import attempt_seed
from large_qubit.section9_reporting import trajectory_report, aggregate_reports, verify_reference, TARGETS


def identity(alias):
    return {'partition': 'final_test', 'geometry_id': alias, 'hamiltonian': 'fixture',
            'orbitals': 'fixture', 'catalogue': 'fixture', 'backend': 'fixture',
            'precision': {'state': 'complex128', 'integrals': 'float64'}}


def reference(alias, energy=-20.):
    run = {'converged': True, 'energy_ha': energy, 'complex128_residual_norm_ha': 1e-10,
           'norm_squared': 1.}
    return {'geometry_id': alias, 'partition': 'final_test', 'cache_identity': identity(alias),
        'energy_hartree': energy, 'report': {'all_passed': True, 'converged': True,
            'independent_starts': 2, 'runs': [run, deepcopy(run)]},
        'wall_seconds': 3., 'evaluator_calls': 12}


def snapshot(method='uniform_random', alias='CO2_I01', count=1024, energy=-10.2):
    requests, attempts = [], []
    for ordinal in range(1, count + 1):
        query = {'geometry_id': alias, 'kind': 'hf'} if ordinal == 1 else (
            {'geometry_id': alias, 'kind': 'vqe', 'parameters': [.1] * 24}
            if method == 'conventional_vqe' else {'geometry_id': alias, 'kind': 'discrete', 'ops': [0] * 32})
        linked = None
        if ordinal > 1 and method in METHODS[:5]:
            linked = len(attempts) + 1
            output = {'valid': True, 'ops': [0] * 32, 'condition_id': method, 'geometry_id': alias,
                      'raw_text': 'fixture', 'token_ids': [1] * 70, 'input_tokens': 3,
                      'output_tokens': 70, 'wall_seconds': .2}
            attempts.append({'ordinal': linked, 'call_id': f'attempt_{linked:04d}',
                'input': {'seed': attempt_seed(method, alias, linked)}, 'output': output,
                'valid': True, 'status': 'completed', 'failure_reason': None})
        scalar = -10. if ordinal == 1 else energy
        evaluation = {'energy_hartree': scalar, 'geometry_id': alias, 'partition': 'final_test',
            'query_sha256': digest(query), 'cache_identity': identity(alias), 'norm': 1.,
            'populations': [11, 11], 'evaluator_calls': 1, 'kernel_contractions': 2,
            'hamiltonian_action_seconds': .05, 'wall_seconds': .1}
        requests.append({'ordinal': ordinal, 'call_id': 'hf' if ordinal == 1 else f'candidate_{ordinal}',
            'kind': 'hf' if ordinal == 1 else 'candidate', 'query': query,
            'status': 'completed', 'energy_hartree': scalar, 'cache_hit': False,
            'duplicate_of': 2 if ordinal > 2 else None, 'attempt_ordinal': linked,
            'evaluation': evaluation, 'failure_reason': None})
    return {'identity': {'method': method, 'geometry_id': alias, 'cache_identity': identity(alias)},
        'requests': requests, 'attempts': attempts, 'head_sha256': 'fixture',
        'logical_requests': len(requests), 'completion_attempts': len(attempts)}


def verification(state):
    rows = [r for r in state['requests'] if r['status'] == 'completed']
    best = min(rows, key=lambda r: (r['energy_hartree'], r['ordinal']))
    return {**deepcopy(best['evaluation']), 'fresh_basis_and_evaluator': True,
            'independent_backend_claimed': False}


class Section9ReportingTests(unittest.TestCase):
    def report(self, state, *, failure=None, ref=None, check=True):
        method, alias = state['identity']['method'], state['identity']['geometry_id']
        return trajectory_report(state, method=method, geometry_id=alias,
            verification=verification(state) if check else None,
            reference=reference(alias) if ref is None else ref, failure=failure)

    def test_complete_baseline_and_vqe_caps_with_exact_carried_curve(self):
        for method, count in [('uniform_random', 1024), ('training_circuit_retrieval', 1024), ('conventional_vqe', 1023)]:
            with self.subTest(method=method):
                report = self.report(snapshot(method=method, count=count))
                self.assertEqual(report['status'], 'complete')
                self.assertTrue(report['primary_endpoint_available'])
                self.assertEqual(report['logical_requests'], count)
                self.assertEqual(report['unused_allowance'], 1024 - count)
                self.assertEqual(len(report['budget_curve']), 1024)
                self.assertEqual(report['budget_curve'][-1]['best_energy_hartree'], -10.2)
                self.assertEqual(report['budget_curve'][-1]['actual_request'], count == 1024)
                self.assertEqual(report['numerical_cost']['search']['evaluator_calls'], count)
                self.assertFalse(report['performance_threshold_applied'])

    def test_llm_all_invalid_attempt_cap_is_retained_as_negative_outcome(self):
        method, alias = METHODS[0], TARGETS[0]
        state = snapshot(method, alias, count=1)
        state['attempts'] = [{'ordinal': i, 'call_id': f'attempt_{i}',
            'input': {'seed': attempt_seed(method, alias, i)}, 'status': 'completed', 'valid': False,
            'output': {'valid': False, 'raw_text': 'bad', 'token_ids': [1], 'input_tokens': 3,
                       'output_tokens': 1, 'wall_seconds': .001, 'error_kind': 'invalid_output'}}
            for i in range(1, 2049)]
        state['completion_attempts'] = 2048
        report = self.report(state)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['counts']['invalid_completion_attempts'], 2048)
        self.assertEqual(report['logical_requests'], 1)
        self.assertTrue(report['primary_endpoint_available'])
        self.assertEqual(report['best_request'], 1)
        self.assertEqual(report['unused_allowance'], 1023)

    def test_failed_early_incumbent_is_verified_and_carried_with_failure_visible(self):
        state = snapshot(count=3)
        record = deepcopy(state['requests'][-1])
        record.update(ordinal=4, call_id='failed', status='failed', energy_hartree=None,
            failure_reason='fixture worker stopped', evaluation={'wall_seconds': .3, 'evaluator_calls': 1,
                'kernel_contractions': 2, 'hamiltonian_action_seconds': .1, 'partial': True})
        state['requests'].append(record)
        state['logical_requests'] += 1
        report = self.report(state, failure={'reason': 'fixture worker stopped'})
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['primary_endpoint_available'])
        self.assertEqual(report['counts']['failed_requests'], 1)
        self.assertEqual(report['unused_allowance'], 1020)
        self.assertEqual(report['best_query'], state['requests'][1]['query'])
        self.assertTrue(report['budget_curve'][-1]['carried_after_stop'])
        self.assertTrue(report['numerical_cost']['search']['wall_is_lower_bound'])

    def test_physical_norm_query_cache_and_energy_faults_withhold_primary_without_erasing_raw_best(self):
        clean = snapshot(count=3)
        mutations = [lambda r: r['evaluation'].update(norm=.9),
                     lambda r: r['evaluation'].update(query_sha256='different'),
                     lambda r: r['evaluation'].update(cache_identity={'different': True}),
                     lambda r: r['evaluation'].update(energy_hartree=-11.)]
        for mutate in mutations:
            state = deepcopy(clean)
            mutate(state['requests'][1])
            report = self.report(state, failure={'reason': 'physical integrity failure'})
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['primary_endpoint_available'])
            self.assertIsNone(report['best_verified_error_hartree'])
            self.assertEqual(report['best_energy_hartree'], -10.2)
            self.assertTrue(report['integrity_failures'])

    def test_bad_partial_physical_norm_also_withholds_endpoint(self):
        state = snapshot(count=2)
        bad = deepcopy(state['requests'][1])
        bad.update(ordinal=3, status='failed', energy_hartree=None,
                   evaluation={'norm': .7, 'wall_seconds': .1})
        state['requests'].append(bad)
        state['logical_requests'] = 3
        report = self.report(state, failure={'reason': 'normalization failure'})
        self.assertFalse(report['primary_endpoint_available'])
        self.assertTrue(any('partial failure' in message for message in report['integrity_failures']))

    def test_missing_checks_or_blocked_reference_never_invent_verified_endpoint(self):
        state = snapshot(count=2)
        report = trajectory_report(state, method='uniform_random', geometry_id=TARGETS[0],
                                   reference=None, verification=None, failure={'reason': 'reference blocked'})
        self.assertEqual(report['reference_status'], 'missing')
        self.assertEqual(report['verification_status'], 'missing')
        self.assertFalse(report['primary_endpoint_available'])
        self.assertIsNone(report['best_verified_error_hartree'])
        self.assertEqual(report['best_energy_hartree'], -10.2)

    def test_invalid_reference_and_winner_check_are_reported_scientific_failures(self):
        state = snapshot(count=2)
        ref = reference(TARGETS[0])
        ref['report']['runs'][1]['complex128_residual_norm_ha'] = .01
        report = self.report(state, ref=ref)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['reference_status'], 'invalid')
        self.assertFalse(report['primary_endpoint_available'])
        check = verification(state)
        check['fresh_basis_and_evaluator'] = False
        report = trajectory_report(state, method='uniform_random', geometry_id=TARGETS[0],
            verification=check, reference=reference(TARGETS[0]), failure={'reason': 'integrity check failed'})
        self.assertEqual(report['verification_status'], 'invalid')
        self.assertFalse(report['primary_endpoint_available'])

    def test_small_negative_signed_error_is_kept_and_large_below_reference_is_refused(self):
        state = snapshot(count=2, energy=-20. - 5e-9)
        report = self.report(state, failure={'reason': 'early stop'})
        self.assertTrue(report['primary_endpoint_available'])
        self.assertLess(report['best_verified_error_hartree'], 0.)
        state = snapshot(count=2, energy=-20. - 1e-5)
        report = self.report(state, failure={'reason': 'early stop'})
        self.assertFalse(report['primary_endpoint_available'])
        self.assertTrue(report['integrity_failures'])

    def test_cache_origin_proves_physics_but_producer_cost_is_not_recharged(self):
        state = snapshot(count=3)
        origin = state['requests'][2]['evaluation']
        origin['wall_seconds'] = 99.
        state['requests'][2]['cache_hit'] = True
        state['requests'][2]['evaluation'] = {'energy_hartree': -10.2, 'wall_seconds': 0.,
            'evaluator_calls': 0, 'origin_evaluation': origin}
        report = self.report(state, failure={'reason': 'early stop'})
        self.assertTrue(report['primary_endpoint_available'])
        self.assertEqual(report['cache_hits'], 1)
        self.assertEqual(report['numerical_cost']['search']['evaluator_calls'], 2)
        self.assertAlmostEqual(report['numerical_cost']['search']['recorded_wall_seconds'], .2)

    def test_pending_missing_receipts_are_explicit_lower_bounds(self):
        state = snapshot(count=2)
        state['requests'][1].update(status='pending', energy_hartree=None, evaluation=None)
        report = trajectory_report(state, method='uniform_random', geometry_id=TARGETS[0])
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['numerical_cost']['search']['wall_is_lower_bound'])
        self.assertTrue(report['numerical_cost']['search']['numerical_work_is_lower_bound'])
        self.assertEqual(report['numerical_cost']['search']['missing_cost_records'], 1)

    def test_wrong_job_or_seed_is_a_binding_refusal(self):
        state = snapshot(METHODS[0], count=2)
        with self.assertRaisesRegex(ValueError, 'identity'):
            trajectory_report(state, method=METHODS[1], geometry_id=TARGETS[0])
        state['attempts'][0]['input']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.report(state)

    def test_failed_verified_incumbents_are_retained_without_primary_or_paired_means(self):
        reports = {}
        for index, method in enumerate(METHODS):
            for alias in TARGETS:
                state = snapshot(method, alias, count=2, energy=-10.2 + index * .01)
                report = self.report(state, failure={'reason': 'synthetic fixed early termination'})
                reports[report['job_id']] = report
        result = aggregate_reports(reports, resources={'fixture': True}, costs={'scope': 'separate'})
        self.assertEqual(len(result['jobs']), 16)
        self.assertEqual(len(result['failed_jobs']), 16)
        self.assertTrue(result['all_endpoints_verified'])
        self.assertFalse(result['methods'][METHODS[0]]['primary_mean_available'])
        self.assertIsNone(result['methods'][METHODS[0]]['mean_verified_error_hartree'])
        self.assertEqual(result['methods'][METHODS[0]]['failed_targets'], list(TARGETS))
        pairs = result['paired_comparisons']
        self.assertEqual(len(pairs), 4)
        comparison = pairs['gpt_weighted_minus_uniform']
        self.assertIsNone(comparison['mean_difference_hartree'])
        self.assertAlmostEqual(comparison['target_error_differences_hartree'][TARGETS[0]], -.01)
        self.assertTrue(comparison['target_status'][TARGETS[0]]['includes_failed_incumbent'])
        self.assertEqual(comparison['target_status'][TARGETS[0]]['left_status'], 'failed')
        self.assertFalse(result['failure_exclusion_applied'])
        self.assertEqual(result['costs'], {'scope': 'separate'})

    def test_primary_means_require_both_completed_verified_targets(self):
        reports = {}
        for index, method in enumerate(METHODS):
            for alias in TARGETS:
                count = 1023 if method == 'conventional_vqe' else 1024
                report = self.report(snapshot(method, alias, count=count, energy=-10.2 + index * .01))
                reports[report['job_id']] = report
        result = aggregate_reports(reports)
        self.assertTrue(result['methods'][METHODS[0]]['primary_mean_available'])
        self.assertAlmostEqual(result['paired_comparisons']['gpt_weighted_minus_uniform']['mean_difference_hartree'], -.01)
        failed = self.report(snapshot(METHODS[1], TARGETS[0], count=2, energy=-10.19),
                             failure={'reason': 'fixed synthetic early failure'})
        reports[failed['job_id']] = failed
        result = aggregate_reports(reports)
        method = result['methods'][METHODS[1]]
        self.assertFalse(method['primary_mean_available'])
        self.assertIsNone(method['mean_verified_error_hartree'])
        self.assertIsNotNone(method['target_errors_hartree'][TARGETS[0]])
        self.assertIsNotNone(method['target_errors_hartree'][TARGETS[1]])
        self.assertEqual(method['target_status'][TARGETS[0]], 'failed')
        self.assertIsNone(result['paired_comparisons']['gpt_weighted_minus_uniform']['mean_difference_hartree'])

    def test_missing_targets_do_not_produce_partial_means_and_sealing_requires_all_jobs(self):
        report = self.report(snapshot(count=2), failure={'reason': 'early stop'})
        with self.assertRaisesRegex(ValueError, '16'):
            aggregate_reports({report['job_id']: report})
        status = aggregate_reports({report['job_id']: report}, require_complete=False)
        self.assertEqual(len(status['missing_jobs']), 15)
        self.assertIsNone(status['methods']['uniform_random']['mean_verified_error_hartree'])
        self.assertFalse(status['all_endpoints_verified'])

    def test_reference_must_match_target_and_both_independent_starts(self):
        self.assertTrue(verify_reference(reference(TARGETS[0]), identity(TARGETS[0]))['all_passed'])
        with self.assertRaises(ValueError):
            verify_reference(reference(TARGETS[0]), identity(TARGETS[1]))
        bad = reference(TARGETS[0])
        bad['report']['runs'].pop()
        with self.assertRaises(ValueError):
            verify_reference(bad, identity(TARGETS[0]))
        malformed = reference(TARGETS[0])
        malformed['report'] = None
        reduced = self.report(snapshot(count=2), ref=malformed)
        self.assertEqual(reduced['reference_status'], 'invalid')
        self.assertEqual(reduced['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

