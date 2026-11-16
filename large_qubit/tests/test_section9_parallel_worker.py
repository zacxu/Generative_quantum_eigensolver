"""Four-GPU lane admission/recovery using real ledgers and CPU-only fake science."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from large_qubit import section9, section9_parallel_worker as worker
from large_qubit.common import digest, sha256_file
from large_qubit.protocol import FITS, METHODS
from large_qubit.section8 import load, save
from large_qubit.section8_policy import CATALOGUE
from large_qubit.section9_policy import attempt_seed, job_id
from large_qubit.section9_parallel_policy import PREPARATION, OUTPUT_ROOT, MATRIX, lane_spec, lane_specs, policy

ROOT = Path(__file__).resolve().parents[2]


def physical_identity(alias):
    return {'geometry_id': alias, 'partition': 'final_test', 'fixture_only': True,
        'cache_identity': {'partition': 'final_test', 'geometry_id': alias, 'hamiltonian': 'fixture_' + alias,
            'orbitals': 'fixture', 'catalogue': 'fixture', 'backend': 'constant_fixture',
            'precision': {'state': 'complex128', 'integrals': 'float64'}}}


def evaluation(query):
    return {'geometry_id': query['geometry_id'], 'partition': 'final_test', 'energy_hartree': -90.,
        'norm': 1., 'populations': [11, 11], 'query_sha256': digest(query),
        'cache_identity': physical_identity(query['geometry_id'])['cache_identity'],
        'evaluator_calls': 1, 'kernel_contractions': 1, 'hamiltonian_action_seconds': .00001,
        'wall_seconds': .00002, 'fixture_only': True}


class LaneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / OUTPUT_ROOT / MATRIX
        self.prepared = {'final_freeze_sha256': 'f' * 64, 'release_sha256': 'r' * 64,
                         'training': {'fits': {fit: {'adapter_path': 'fixture/' + fit} for fit in FITS}}}
        save(self.root / PREPARATION, self.prepared)
        self.allocation = {spec['lane_id']: {'gpu_uuid': 'GPU-fixture-' + spec['lane_id'],
            'runtime': {'gpu_name': 'NVIDIA H200', 'gpu_uuid': 'GPU-fixture-' + spec['lane_id'],
                        'fixture_only': True}} for spec in lane_specs()}
        save(self.dest / 'gpu_allocation.json', {'schema_version': 'fixture', 'lanes': self.allocation})
        save(self.dest / 'binding.json', {'preparation_sha256': sha256_file(self.root / PREPARATION),
            'policy': policy(), 'final_freeze_sha256': self.prepared['final_freeze_sha256'],
            'allocation_sha256': sha256_file(self.dest / 'gpu_allocation.json')})
        for alias in ('CO2_I01', 'CO2_I02'):
            target = self.dest / 'targets' / alias
            identity = physical_identity(alias)
            save(target / 'physics_identity.json', identity)
            save(target / 'reference.json', {'status': 'completed', 'uncertain_cost': False, 'result': {
                'geometry_id': alias, 'partition': 'final_test', 'energy_hartree': -91.,
                'cache_identity': identity['cache_identity'], 'evaluator_calls': 2,
                'wall_seconds': .0001, 'fixture_only': True,
                'report': {'all_passed': True, 'converged': True, 'independent_starts': 2,
                    'hamiltonian_actions': 2, 'wall_seconds': .0001,
                    'runs': [{'converged': True, 'energy_ha': -91.,
                              'complex128_residual_norm_ha': 0., 'norm_squared': 1.} for _ in range(2)]}}})
        catalogue = self.root / CATALOGUE
        catalogue.parent.mkdir(parents=True)
        catalogue.write_bytes((ROOT / CATALOGUE).read_bytes())
        from large_qubit.section5 import SCREENING_PATH
        screening = self.root / SCREENING_PATH
        screening.parent.mkdir(parents=True)
        screening.write_text('{"vqe_initial_angles":' + str([0.] * 24) + '}', encoding='utf-8')
        self.events = {'model_loads': [], 'switches': [], 'physics_loads': [], 'scores': [],
                       'verifications': [], 'generation_seeds': [], 'closed_physics': 0, 'closed_models': 0}

    def fixtures(self, lane_id='lane_0', *, early_engine=True):
        events, allocation = self.events, self.allocation[lane_id]
        parent = types.ModuleType('large_qubit.section9_parallel')
        parent.verify_preparation = Mock(return_value=self.prepared)
        parent.verify_binding = Mock(return_value=(self.prepared, {'lanes': self.allocation}))
        parent.directory = lambda root: root / OUTPUT_ROOT / MATRIX

        class Physics:
            def __init__(self, root, python, alias, **kwargs):
                self.alias, self.identity = alias, physical_identity(alias)
                events['physics_loads'].append((alias, kwargs))

            def reference(self):
                raise AssertionError('A lane must never recompute the parent target reference')

            def score(self, query):
                events['scores'].append(query)
                return evaluation(query)

            def verify(self, query):
                events['verifications'].append(query)
                return {**evaluation(query), 'fresh_basis_and_evaluator': True,
                        'independent_backend_claimed': False}

            def close(self):
                events['closed_physics'] += 1

        class Generator:
            def __init__(self, root, key, **kwargs):
                self.model_key = key
                self.metadata = {'runtime_identity': allocation['runtime'], 'fixture_only': True}
                self.condition = None
                events['model_loads'].append(key)

            def switch(self, method, **kwargs):
                self.condition = method
                events['switches'].append(method)

            def generate(self, alias, catalogue, seed):
                events['generation_seeds'].append((self.condition, alias, seed))
                return {'valid': True, 'ops': [0] * 32, 'condition_id': self.condition,
                    'geometry_id': alias, 'attempt_seed': seed, 'input_tokens': 1, 'output_tokens': 1,
                    'wall_seconds': .00001, 'fixture_only': True}

            def statistics(self):
                return {'fixture_only': True, 'base_loads': 1, 'wall_seconds': .0001}

            def close(self):
                events['closed_models'] += 1

        class Release:
            source_acquisition_cost = {'ordinary_requests': 1, 'fixture_only': True}
            def retrieval_rows(self):
                return []

        def pin(root, destination):
            save(destination / 'gpu_runtime.json', allocation['runtime'])
            return allocation['runtime']

        stack = ExitStack()
        stack.enter_context(patch.dict('sys.modules', {'large_qubit.section9_parallel': parent}))
        stack.enter_context(patch.object(worker.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(worker, 'pin_runtime', side_effect=pin))
        stack.enter_context(patch('large_qubit.training._run_lock', return_value=nullcontext()))
        stack.enter_context(patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()))
        stack.enter_context(patch('large_qubit.section9_scoring.PhysicsClient', Physics))
        stack.enter_context(patch('large_qubit.section8_inference.ResidentGenerator', Generator))
        stack.enter_context(patch('large_qubit.corpus_v2.load_release', return_value=Release()))
        stack.enter_context(patch('large_qubit.baselines.retrieval_candidates',
            return_value={'candidates': [{'ops': [0] * 32} for _ in range(1023)], 'fixture_only': True}))
        stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': allocation['gpu_uuid']}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        if early_engine:
            stack.enter_context(patch.object(worker, 'execute_trajectory', side_effect=self.short_engine))
        return stack

    @staticmethod
    def short_engine(ledger, scorer, method, alias, **kwargs):
        scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
        if method in METHODS[:5]:
            ordinal = ledger.frontier()['completion_attempts'] + 1
            seed = attempt_seed(method, alias, ordinal)
            call_id = f'attempt_{ordinal:04d}'
            ledger.reserve_attempt(call_id, {'seed': seed})
            result = kwargs['generator'].generate(alias, kwargs['catalogue'], seed)
            ledger.finish_attempt(ordinal, result, valid=True)
            scorer.submit(call_id, {'geometry_id': alias, 'kind': 'discrete', 'ops': result['ops']},
                          attempt_ordinal=ordinal).result()
        ledger.halt('Numerical incident: fixture early terminal outcome')
        raise ValueError('fixture early terminal outcome')

    def test_frozen_controller_is_reused_without_budget_or_rng_reimplementation(self):
        self.assertIs(worker.execute_trajectory, section9.execute_trajectory)
        for spec in lane_specs():
            for method in spec['methods']:
                identity = worker.ledger_identity('a' * 64, spec['lane_id'], method, spec['target'],
                                                  physical_identity(spec['target'])['cache_identity'])
                self.assertEqual(identity['cache_scope'], 'lane_local')
                self.assertIn(spec['lane_id'], identity['queue_id'])
        with self.assertRaisesRegex(ValueError, 'outside its fixed lane'):
            worker.ledger_identity('a' * 64, 'lane_0', 'qwen3_4b_frozen', 'CO2_I01',
                                   physical_identity('CO2_I01')['cache_identity'])

    def test_requires_explicit_execution_and_authentication_before_gpu(self):
        with patch.object(worker, '_context') as context, self.assertRaisesRegex(ValueError, '--execute'):
            worker.run_lane(self.root, 'lane_0')
        context.assert_not_called()
        with patch.object(worker.platform, 'system', return_value='Linux'), \
             patch.object(worker, '_context', side_effect=ValueError('freeze absent')), \
             patch.object(worker, 'pin_runtime') as pin, self.assertRaisesRegex(ValueError, 'freeze absent'):
            worker.run_lane(self.root, 'lane_0', execute=True)
        pin.assert_not_called()

    def test_wrong_gpu_mask_refuses_before_runtime_pin_or_lane_creation(self):
        with self.fixtures(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-wrong'}), \
             patch.object(worker, 'pin_runtime') as pin, self.assertRaisesRegex(ValueError, 'allocated GPU UUID'):
            worker.run_lane(self.root, 'lane_0', execute=True)
        pin.assert_not_called()
        self.assertFalse((self.dest / 'lanes').exists())
        self.assertEqual(self.events['model_loads'], [])
        self.assertEqual(self.events['physics_loads'], [])

    def test_failed_reference_preserves_all_lane_jobs_without_physics_or_models(self):
        path = self.dest / 'targets/CO2_I01/reference.json'
        path.unlink()  # Fixture-only replacement before any lane work.
        save(path, {'status': 'failed', 'error': 'fixture reference failure', 'result': None, 'uncertain_cost': True})
        with self.fixtures():
            result = worker.run_lane(self.root, 'lane_0', execute=True)
            worker.verify_lane(self.root, 'lane_0')
        self.assertTrue(result['lane_execution_complete'])
        self.assertEqual(len(result['reports']), 5)
        self.assertFalse(result['all_trajectories_successful'])
        self.assertEqual(self.events['physics_loads'], [])
        self.assertEqual(self.events['model_loads'], [])
        self.assertTrue(all(r['logical_requests'] == 0 and r['status'] == 'failed' for r in result['reports'].values()))

    def test_all_lanes_own_disjoint_jobs_models_caches_and_pinned_seeds(self):
        reports, cache_paths = {}, []
        for spec in lane_specs():
            with self.fixtures(spec['lane_id']):
                result = worker.run_lane(self.root, spec['lane_id'], execute=True)
            reports.update(result['reports'])
            cache_paths.append(self.dest / 'lanes' / spec['lane_id'] / 'scalar_cache.sqlite')
            self.assertFalse(result['costs']['shared_costs_charged_here'])
            self.assertEqual(set(result['reports']), set(spec['job_ids']))
        self.assertEqual(len(reports), 16)
        self.assertEqual(self.events['model_loads'], ['gpt2_xl', 'gpt2_xl', 'qwen3_4b', 'qwen3_4b'])
        self.assertEqual(len(set(cache_paths)), 4)
        self.assertTrue(all(p.is_file() for p in cache_paths))
        for method, alias, seed in self.events['generation_seeds']:
            self.assertEqual(seed, attempt_seed(method, alias, 1))
        self.assertTrue(all(k['threads'] == 1 and k['execute'] for _, k in self.events['physics_loads']))
        self.assertEqual(len(self.events['verifications']), 16)
        self.assertEqual(self.events['closed_models'], 4)

    def test_completed_lane_skips_gpu_model_and_physics_even_when_mask_changes(self):
        with self.fixtures():
            worker.run_lane(self.root, 'lane_0', execute=True)
            before = {key: len(value) for key, value in self.events.items() if isinstance(value, list)}
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(worker, 'pin_runtime') as pin:
                result = worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
            pin.assert_not_called()
        self.assertTrue(result['lane_execution_complete'])
        self.assertEqual(before, {key: len(value) for key, value in self.events.items() if isinstance(value, list)})

    def test_interrupted_attempt_is_charged_on_explicit_resume_and_seed_advances(self):
        interrupted = [False]
        def engine(ledger, scorer, method, alias, **kwargs):
            if not interrupted[0]:
                interrupted[0] = True
                ledger.reserve_attempt('attempt_0001', {'seed': attempt_seed(method, alias, 1)})
                raise KeyboardInterrupt('fixture interruption after generation intent')
            return self.short_engine(ledger, scorer, method, alias, **kwargs)
        with self.fixtures(early_engine=False), patch.object(worker, 'execute_trajectory', side_effect=engine):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_lane(self.root, 'lane_0', execute=True)
            with self.assertRaisesRegex(ValueError, '--resume'):
                worker.run_lane(self.root, 'lane_0', execute=True)
            result = worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
        report = result['reports'][job_id('gpt2_xl_frozen', 'CO2_I01')]
        self.assertEqual(report['completion_attempts'], 2)
        self.assertEqual(report['counts']['abandoned_completion_attempts'], 1)
        self.assertEqual(self.events['generation_seeds'][0][2], attempt_seed('gpt2_xl_frozen', 'CO2_I01', 2))
        self.assertEqual(len(result['costs']['execution_attempts']), 2)

    def test_missing_initialized_cache_is_never_recreated(self):
        with self.fixtures():
            worker.run_lane(self.root, 'lane_0', execute=True)
            dest = self.dest / 'lanes/lane_0'
            hashes = {p: sha256_file(p) for p in dest.glob('*/ledger.sqlite')}
            (dest / 'completed.json').unlink()
            (dest / 'scalar_cache.sqlite').unlink()
            with self.assertRaisesRegex(ValueError, 'lane scalar cache is missing'):
                worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
            self.assertFalse((dest / 'scalar_cache.sqlite').exists())
            self.assertTrue(all(sha256_file(p) == value for p, value in hashes.items()))

    def test_missing_initialized_job_ledger_cannot_reset_the_budget(self):
        with self.fixtures():
            worker.run_lane(self.root, 'lane_0', execute=True)
            dest = self.dest / 'lanes/lane_0'
            folder = dest / job_id('gpt2_xl_frozen', 'CO2_I01')
            identity_sha, report_sha = sha256_file(folder / 'identity.json'), sha256_file(folder / 'report.json')
            (dest / 'completed.json').unlink()
            (folder / 'ledger.sqlite').unlink()
            with self.assertRaisesRegex(ValueError, 'lane job ledger is missing'):
                worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
            self.assertFalse((folder / 'ledger.sqlite').exists())
            self.assertEqual(sha256_file(folder / 'identity.json'), identity_sha)
            self.assertEqual(sha256_file(folder / 'report.json'), report_sha)

    def test_physical_failure_stays_charged_without_replacement_and_other_jobs_continue(self):
        calls = []
        def score(query):
            calls.append(query)
            if len(calls) == 1:
                from large_qubit.section9_scoring import ScoringError
                raise ScoringError('fixture numerical failure', report={
                    'geometry_id': query['geometry_id'], 'stage': 'score',
                    'cost_category': 'ordinary_energy_request', 'query_sha256': digest(query),
                    'evaluator_calls': 1, 'wall_seconds': .0001, 'fixture_only': True})
            return evaluation(query)
        with self.fixtures(), patch('large_qubit.section9_scoring.PhysicsClient.score', side_effect=score):
            result = worker.run_lane(self.root, 'lane_0', execute=True)
        first = result['reports'][job_id('gpt2_xl_frozen', 'CO2_I01')]
        self.assertEqual(first['status'], 'failed')
        self.assertEqual(first['logical_requests'], 1)
        self.assertEqual(first['counts']['failed_requests'], 1)
        self.assertEqual(first['completion_attempts'], 0)
        self.assertFalse(first['primary_endpoint_available'])
        self.assertEqual(len(result['reports']), 5)
        self.assertTrue(all(r['status'] == 'failed' for r in result['reports'].values()))
        self.assertTrue(result['reports'][job_id('gpt2_xl_weighted_lora', 'CO2_I01')]['primary_endpoint_available'])

    def test_changed_parent_reference_is_rejected_without_additional_work(self):
        with self.fixtures():
            worker.run_lane(self.root, 'lane_0', execute=True)
            path = self.dest / 'targets/CO2_I01/reference.json'
            receipt = load(path)
            receipt['extra_unregistered_field'] = True
            path.unlink()
            save(path, receipt)
            with self.assertRaisesRegex(ValueError, 'Lane binding changed'):
                worker.verify_lane(self.root, 'lane_0')

    def test_partial_seal_resume_authenticates_and_skips_every_finished_job(self):
        actual_save, seal_path = worker.save, self.dest / 'lanes/lane_0/completed.json'
        def interrupted_save(path, value):
            if Path(path) == seal_path:
                raise RuntimeError('fixture seal interruption')
            return actual_save(path, value)
        with self.fixtures():
            with patch.object(worker, 'save', side_effect=interrupted_save), \
                 self.assertRaisesRegex(RuntimeError, 'seal interruption'):
                worker.run_lane(self.root, 'lane_0', execute=True)
            before = {key: len(value) for key, value in self.events.items() if isinstance(value, list)}
            result = worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
        self.assertTrue(result['lane_execution_complete'])
        self.assertEqual(before, {key: len(value) for key, value in self.events.items() if isinstance(value, list)})
        self.assertEqual(len(result['costs']['execution_attempts']), 2)

    def test_parent_death_setup_checks_both_sides_of_linux_registration(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with patch.object(worker.platform, 'system', return_value='Linux'), \
             patch.object(worker.os, 'getppid', side_effect=[123, 123]), patch('ctypes.CDLL', return_value=libc):
            worker._arm_parent_death_signal(123)
        libc.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)
        with patch.object(worker.platform, 'system', return_value='Linux'), \
             patch.object(worker.os, 'getppid', side_effect=[123, 1]), patch('ctypes.CDLL', return_value=libc), \
             self.assertRaisesRegex(ValueError, 'died during lane startup'):
            worker._arm_parent_death_signal(123)

    def test_interrupted_attempt_initialization_retains_unknown_cost(self):
        destination = self.dest / 'lanes/lane_0'
        (destination / 'execution_attempts/attempt_0001').mkdir(parents=True)
        costs = worker.retained_lane_costs(self.root, destination, {})
        self.assertEqual(len(costs['execution_attempts']), 1)
        self.assertIsNone(costs['execution_attempts'][0]['started'])
        self.assertEqual(costs['execution_attempts'][0]['status'], 'interrupted')
        self.assertTrue(costs['execution_attempts'][0]['partial_algebraic_work_may_be_unknown'])


if __name__ == '__main__':
    unittest.main()
