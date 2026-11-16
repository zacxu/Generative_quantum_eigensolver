"""CPU-only synthetic main-controller mechanics; never scientific execution."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9 as s
from large_qubit.common import digest, read_json
from large_qubit.protocol import METHODS
from large_qubit.section9_policy import policy, jobs, attempt_seed
from large_qubit.section9_ledger import TrajectoryLedger, ScalarCache, OrderedScorer
from large_qubit.section9_scoring import ScoringError

ROOT = Path(__file__).resolve().parents[2]
ALIAS = 'CO2_I01'
IDENTITY = {'partition': 'final_test', 'geometry_id': ALIAS, 'hamiltonian': 'fixture',
            'orbitals': 'fixture', 'catalogue': 'fixture', 'backend': 'analytic_fixture',
            'precision': {'state': 'complex128', 'integrals': 'float64'}}


def evaluate(query):
    if query['kind'] == 'hf':
        energy = -90.
    elif query['kind'] == 'vqe':
        energy = -90. - .001 * sum(math.cos(x) for x in query['parameters'])
    else:
        energy = -90.1
    return {'energy_hartree': energy, 'geometry_id': ALIAS, 'partition': 'final_test',
        'norm': 1., 'populations': [11, 11], 'query_sha256': digest(query), 'cache_identity': IDENTITY,
        'evaluator_calls': 1, 'kernel_contractions': 1, 'wall_seconds': .001, 'fixture_only': True}


class Generator:
    def __init__(self, *, invalid=False):
        self.invalid, self.seeds = invalid, []

    def generate(self, alias, catalogue, seed):
        self.seeds.append(seed)
        return {'valid': not self.invalid, 'ops': [0] * 32, 'raw_text': 'fixture', 'token_ids': [1],
                'wall_seconds': .001, 'input_tokens': 1, 'output_tokens': 1, 'fixture_only': True}


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalogue = read_json(ROOT / 'large_qubit/catalogues/co2_screened_v1.json')
        self.handles = []

    def tearDown(self):
        for handle in reversed(self.handles):
            handle.close()
        self.temp.cleanup()

    def components(self, method, callback=evaluate):
        ledger = TrajectoryLedger(self.root / 'ledger.sqlite', s.ledger_identity('fixture', method, ALIAS, IDENTITY))
        cache = ScalarCache(self.root / 'cache.sqlite')
        self.handles.extend((ledger, cache))
        return ledger, OrderedScorer(ledger, cache, IDENTITY, callback)

    def test_all_sixteen_jobs_and_exact_frozen_budgets(self):
        p = policy()
        self.assertEqual(len(jobs()), 16)
        self.assertEqual(len(set(p['job_order'])), 16)
        self.assertEqual(p['scientific_protocol']['requests_per_trajectory'], 1024)
        self.assertEqual(p['scientific_protocol']['max_completion_attempts'], 2048)
        self.assertEqual(p['scientific_protocol']['vqe_restart_caps'], [511, 512])
        self.assertEqual(p['vqe_actual_requests_without_failures'], 1023)
        self.assertEqual(p['inherited_inference']['inference_batch_size'], 1)
        self.assertFalse(p['performance_selection'])

    def test_no_execution_without_explicit_opt_in(self):
        for call in (s.prepare, s.run_matrix):
            with self.assertRaisesRegex(ValueError, 'explicit'):
                call(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_main_gate_precedes_gpu_and_scoring(self):
        with patch.object(s.platform, 'system', return_value='Linux'), \
             patch.object(s, 'verify_preparation', side_effect=ValueError('missing frozen Section8')), \
             self.assertRaisesRegex(ValueError, 'missing frozen Section8'):
            s.run_matrix(self.root, execute=True)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_preparation_missing_freeze_never_loads_models_or_target_arrays(self):
        with patch('large_qubit.section9_registration.verify', return_value={'registration_sha256': 'fixture'}), \
             patch('large_qubit.section8.verify_final_freeze', side_effect=ValueError('freeze incomplete')), \
             self.assertRaisesRegex(ValueError, 'freeze incomplete'):
            s.prepare(self.root, execute=True)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_gpu_first_allocation_can_change_but_precision_cannot(self):
        expected = {'gpu_uuid': 'old', 'gpu_name': 'NVIDIA H200', 'precision': 'bf16', 'batch_size': 1}
        actual = dict(expected, gpu_uuid='new')
        s.runtime_matches_validation(actual, expected)
        for key, value in (('precision', 'fp32'), ('batch_size', 2), ('gpu_name', 'NVIDIA H100')):
            with self.assertRaises(ValueError):
                s.runtime_matches_validation(dict(actual, **{key: value}), expected)
        s.save(self.root / 'runtime.json', actual)
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            s.save(self.root / 'runtime.json', expected)

    def test_target_ledger_identity_rejects_validation_and_cross_target(self):
        for wrong in (dict(IDENTITY, partition='validation'), dict(IDENTITY, geometry_id='CO2_I02')):
            with self.assertRaises(ValueError):
                s.ledger_identity('fixture', METHODS[0], ALIAS, wrong)
        value = s.ledger_identity('fixture', METHODS[0], ALIAS, IDENTITY)
        self.assertIn(ALIAS, value['job_id'])
        self.assertIn(value['job_id'], value['queue_id'])

    def test_overhead_completed_result_is_reused_without_second_call(self):
        result = s.overhead_once(self.root, 'reference', lambda: {'value': 1})
        self.assertEqual(result, s.overhead_once(self.root, 'reference', lambda: self.fail('unexpected retry')))

    def test_interrupted_overhead_is_unknown_and_never_retried(self):
        s.save(self.root / 'reference_started.json', {'utc': s.utc()})
        result = s.overhead_once(self.root, 'reference', lambda: self.fail('unknown work was retried'))
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['uncertain_cost'])
        self.assertIsNone(result['result'])

    def test_overhead_partial_failure_keeps_known_cost(self):
        def fail():
            raise ScoringError('fixture failure', report={'evaluator_calls': 3, 'wall_seconds': .2})
        result = s.overhead_once(self.root, 'winner_verification', fail)
        self.assertEqual(result['failure_report']['evaluator_calls'], 3)
        self.assertEqual(result, s.overhead_once(self.root, 'winner_verification', lambda: self.fail('retry')))

    def test_overhead_storage_error_is_not_scientific_failure(self):
        def fail():
            raise OSError('fixture quota')
        with self.assertRaisesRegex(OSError, 'quota'):
            s.overhead_once(self.root, 'reference', fail)
        self.assertTrue((self.root / 'reference_started.json').exists())
        self.assertFalse((self.root / 'reference.json').exists())

    def test_abandoned_generation_keeps_budget_and_next_ordinal_seed(self):
        ledger, scorer = self.components(METHODS[0])
        ledger.reserve_attempt('attempt_0001', {'seed': attempt_seed(METHODS[0], ALIAS, 1)})
        ledger.recover_pending('fixture process interruption')
        gen = Generator(invalid=True)
        with scorer:
            result = s.execute_trajectory(ledger, scorer, METHODS[0], ALIAS,
                                          catalogue=self.catalogue, generator=gen)
        state = result['snapshot']
        self.assertEqual((state['logical_requests'], state['completion_attempts']), (1, 2048))
        self.assertEqual(len(gen.seeds), 2047)
        self.assertEqual(gen.seeds[0], attempt_seed(METHODS[0], ALIAS, 2))
        self.assertEqual(state['attempts'][0]['status'], 'abandoned')

    def test_saved_valid_answer_admitted_without_regeneration(self):
        ledger, scorer = self.components(METHODS[0])
        seed = attempt_seed(METHODS[0], ALIAS, 1)
        ledger.reserve_attempt('attempt_0001', {'seed': seed})
        ledger.finish_attempt(1, Generator().generate(ALIAS, self.catalogue, seed), valid=True)
        class Stop:
            def generate(self, *args):
                raise KeyboardInterrupt('fixture abrupt stop')
        with scorer, self.assertRaises(KeyboardInterrupt):
            s.execute_trajectory(ledger, scorer, METHODS[0], ALIAS, catalogue=self.catalogue, generator=Stop())
        state = ledger.snapshot()
        self.assertEqual(state['logical_requests'], 2)
        self.assertEqual(state['requests'][1]['attempt_ordinal'], 1)
        self.assertEqual(state['completion_attempts'], 2)
        self.assertEqual(state['pending_attempts'], 1)

    def test_pending_physical_call_is_never_implicitly_rescored(self):
        ledger, scorer = self.components(METHODS[0])
        scorer.close()
        ledger.reserve_request('hf', {'geometry_id': ALIAS, 'kind': 'hf'}, kind='hf')
        ledger.recover_pending()
        with self.assertRaisesRegex(ValueError, 'Stopped'):
            OrderedScorer(ledger, self.handles[-1], IDENTITY, lambda q: self.fail('rescore'))
        self.assertEqual(ledger.frontier()['logical_requests'], 1)

    def test_vqe_caps_all_probe_energies_and_exact_completed_replay(self):
        ledger, scorer = self.components('conventional_vqe')
        progress, work = [], []
        with scorer:
            first = s.execute_trajectory(ledger, scorer, 'conventional_vqe', ALIAS,
                catalogue=self.catalogue, initial_angles=[.01] * 24,
                optimizer_progress=lambda r, p: progress.append((r, p)),
                optimizer_work=lambda r, i, e, p: work.append((r, i, e, p)))
        self.assertEqual(first['snapshot']['logical_requests'], 1023)
        self.assertEqual([r['requests'] for r in first['optimizer']], [511, 511])
        self.assertEqual(len(progress), 170)
        self.assertEqual(sum(w[2] == 'fit_started' for w in work), 170)
        self.assertEqual(sum(w[2] == 'fit_completed' for w in work), 170)
        self.assertTrue(all(w[3]['algebraic_candidate_preparation_seconds'] >= 0
                            for w in work if w[2] == 'fit_completed'))
        self.assertEqual(first['snapshot']['completion_attempts'], 0)
        with OrderedScorer(ledger, self.handles[-1], IDENTITY, lambda q: self.fail('replayed physical evaluation')) as scorer:
            second = s.execute_trajectory(ledger, scorer, 'conventional_vqe', ALIAS,
                catalogue=self.catalogue, initial_angles=[.01] * 24)
        self.assertEqual(first, second)

    def test_vqe_interpolation_incident_is_retained_and_terminal(self):
        def nonlinear(query):
            result = evaluate(query)
            if query['kind'] == 'vqe':
                result['energy_hartree'] = -90. + sum(math.exp(x) for x in query['parameters'])
            return result
        ledger, scorer = self.components('conventional_vqe', nonlinear)
        work = []
        with scorer, self.assertRaisesRegex(ValueError, 'five-term'):
            s.execute_trajectory(ledger, scorer, 'conventional_vqe', ALIAS,
                catalogue=self.catalogue, initial_angles=[.01] * 24,
                optimizer_work=lambda r, i, e, p: work.append(e))
        self.assertEqual(work, ['fit_started', 'fit_completed'])
        self.assertTrue(ledger.frontier()['halted'])
        self.assertEqual(ledger.frontier()['logical_requests'], 8)
        self.assertIn('Numerical incident:', ledger.frontier()['stop_reason'])

    def test_unknown_method_refuses_before_hf(self):
        ledger, scorer = self.components(METHODS[0])
        with scorer, self.assertRaises(ValueError):
            s.execute_trajectory(ledger, scorer, 'unknown', ALIAS, catalogue=self.catalogue)
        self.assertEqual(ledger.frontier()['logical_requests'], 0)

    def test_implementation_registration_refuses_source_drift_without_rewriting(self):
        from large_qubit import section9_registration as registration
        from large_qubit.section9_policy import REGISTRATION, SOURCES
        for name in SOURCES + ('research_plan_revised.md',):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source', encoding='utf-8')
        with patch.object(registration, 'predecessor', return_value={'fixture_only': True}):
            self.assertTrue(registration.register(self.root)['all_passed'])
            before = (self.root / REGISTRATION).read_bytes()
            (self.root / SOURCES[0]).write_text('changed source', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                registration.register(self.root)
            self.assertEqual((self.root / REGISTRATION).read_bytes(), before)

    def test_preflight_costs_keep_failed_candidate_work_separate_from_fits(self):
        import json
        from large_qubit.common import sha256_file
        from large_qubit.section7 import PREFLIGHT_ROOT
        path = self.root / PREFLIGHT_ROOT / 'fixture.json'
        path.parent.mkdir(parents=True)
        benchmark = {'model_load_seconds': 2., 'benchmark_wall_seconds_including_warmup_and_failed_candidates': 7.,
                     'candidates': [{'state': 'oom', 'actual_work': {'forward_attempts': 1}}]}
        path.write_text(json.dumps({'benchmark': benchmark}))
        freeze = path.parent / 'freeze.json'
        freeze.write_text(json.dumps({'preflight_reports': {'fixture_fit': {
            'path': path.relative_to(self.root).as_posix(), 'sha256': sha256_file(path)}}}))
        frozen = {'final_adapters_and_training': {'numerical_freeze_sha256': sha256_file(freeze)}}
        result = s.training_preflight_overhead(self.root, frozen)
        self.assertEqual(result['fits']['fixture_fit']['benchmark'], benchmark)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            s.training_preflight_overhead(self.root, frozen)


if __name__ == '__main__':
    unittest.main()
