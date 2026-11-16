"""Real main coordinator/ledgers, with fake models and constant scalar physics.

The full fixture exercises all sixteen original budgets, one interrupted
generation reservation and a failed final seal. No chemistry/model is loaded.
"""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9 as main
from large_qubit.common import digest, sha256_file
from large_qubit.protocol import FITS, METHODS
from large_qubit.section8 import save, load
from large_qubit.section8_policy import CATALOGUE, FINAL_FREEZE
from large_qubit.section9_policy import PREPARATION, TARGETS, jobs, job_id


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = {'gpu_name': 'NVIDIA H200', 'gpu_uuid': 'GPU-fixture-only', 'fixture_only': True}


def cache_identity(alias):
    return {'partition': 'final_test', 'geometry_id': alias, 'hamiltonian': 'fixture_' + alias,
            'orbitals': 'fixture', 'catalogue': 'fixture', 'backend': 'constant_fixture',
            'precision': {'state': 'complex128', 'integrals': 'float64'}}


def evaluation(query):
    alias = query['geometry_id']
    return {'geometry_id': alias, 'partition': 'final_test',
            'energy_hartree': -90. if alias == 'CO2_I01' else -89.,
            'norm': 1., 'populations': [11, 11], 'query_sha256': digest(query),
            'cache_identity': cache_identity(alias), 'evaluator_calls': 1,
            'kernel_contractions': 1, 'hamiltonian_action_seconds': .00001,
            'wall_seconds': .00002, 'fixture_only': True}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = {'physics_loads': [], 'references': [], 'scores': [], 'verifications': [],
                       'model_loads': [], 'generation_interrupted': False}
        self.prepared = {'final_freeze_sha256': 'f' * 64, 'release_sha256': 'r' * 64,
                         'training': {'fits': {fit: {'adapter_path': 'fixture/' + fit} for fit in FITS}}}
        save(self.root / PREPARATION, self.prepared)
        save(self.root / FINAL_FREEZE, {
            'final_adapters_and_training': self.prepared['training'],
            'costs': {'source_acquisition_charged_once': {'fixture_only': True, 'ordinary_requests': 1},
                      'training_costs': {'fixture_only': True}, 'total_validation_logical_requests': 1019}})
        catalogue = self.root / CATALOGUE
        catalogue.parent.mkdir(parents=True)
        catalogue.write_bytes((ROOT / CATALOGUE).read_bytes())
        from large_qubit.section5 import SCREENING_PATH
        screening = self.root / SCREENING_PATH
        screening.parent.mkdir(parents=True)
        screening.write_text('{"vqe_initial_angles":' + str([0.] * 24) + '}', encoding='utf-8')

    def fixture_components(self, *, interrupt_generation=False, fail_references=False):
        events = self.events

        class Physics:
            def __init__(self, root, python, alias, **kwargs):
                self.alias = alias
                self.identity = {'geometry_id': alias, 'partition': 'final_test',
                                 'cache_identity': cache_identity(alias), 'fixture_only': True}
                events['physics_loads'].append(alias)

            def reference(self):
                events['references'].append(self.alias)
                if fail_references:
                    from large_qubit.section9_scoring import ScoringError
                    raise ScoringError('fixture reference failure', report={
                        'geometry_id': self.alias, 'stage': 'reference', 'evaluator_calls': 0,
                        'wall_seconds': .0001, 'fixture_only': True})
                energy = -91. if self.alias == 'CO2_I01' else -90.
                return {'geometry_id': self.alias, 'partition': 'final_test', 'energy_hartree': energy,
                    'cache_identity': cache_identity(self.alias), 'evaluator_calls': 2,
                    'wall_seconds': .0001, 'fixture_only': True,
                    'report': {'all_passed': True, 'converged': True, 'independent_starts': 2,
                        'hamiltonian_actions': 2, 'wall_seconds': .0001,
                        'runs': [{'converged': True, 'energy_ha': energy,
                                  'complex128_residual_norm_ha': 0., 'norm_squared': 1.} for _ in range(2)]}}

            def score(self, query):
                events['scores'].append(dict(query))
                return evaluation(query)

            def verify(self, query):
                events['verifications'].append(dict(query))
                return {**evaluation(query), 'fresh_basis_and_evaluator': True,
                        'independent_backend_claimed': False}

            def close(self):
                pass

        class Generator:
            def __init__(self, root, key, **kwargs):
                self.model_key, self.calls, self.condition = key, 0, None
                self.metadata = {'runtime_identity': dict(RUNTIME), 'fixture_only': True}
                events['model_loads'].append(key)

            def switch(self, condition, **kwargs):
                self.condition = condition

            def generate(self, alias, catalogue, seed):
                self.calls += 1
                if interrupt_generation and not events['generation_interrupted'] and self.calls == 3:
                    events['generation_interrupted'] = True
                    raise KeyboardInterrupt('fixture interruption after durable generation reservation')
                return {'valid': True, 'ops': [0] * 32, 'condition_id': self.condition,
                    'geometry_id': alias, 'attempt_seed': seed, 'input_tokens': 1, 'output_tokens': 1,
                    'wall_seconds': .00001, 'fixture_only': True}

            def statistics(self):
                return {'attempts': self.calls, 'base_loads': 1, 'wall_seconds': .0001,
                        'fixture_only': True}

            def close(self):
                pass

        class Release:
            source_acquisition_cost = {'ordinary_requests': 1, 'wall_seconds': 1., 'fixture_only': True}

            def retrieval_rows(self):
                return []

        stack = ExitStack()
        stack.enter_context(patch.object(main.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(main, 'verify_preparation', return_value=self.prepared))
        stack.enter_context(patch.object(main, 'controller_runtime', return_value={'fixture_only': True}))
        stack.enter_context(patch.object(main, 'training_preflight_overhead',
                                        return_value={'fixture_only': True}, create=True))
        stack.enter_context(patch('large_qubit.training._run_lock', return_value=nullcontext()))
        stack.enter_context(patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()))
        def pin(root, destination):
            save(destination / 'gpu_runtime.json', dict(RUNTIME))
            return dict(RUNTIME)
        stack.enter_context(patch.object(main, 'pin_runtime', side_effect=pin))
        stack.enter_context(patch('large_qubit.section9_scoring.PhysicsClient', Physics))
        stack.enter_context(patch('large_qubit.section8_inference.ResidentGenerator', Generator))
        stack.enter_context(patch('large_qubit.corpus_v2.load_release', return_value=Release()))
        stack.enter_context(patch('large_qubit.baselines.retrieval_candidates',
            return_value={'candidates': [{'ops': [0] * 32} for _ in range(1023)], 'fixture_only': True}))
        stack.enter_context(patch.dict(main.os.environ))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_missing_freeze_refuses_before_gpu_or_target_access(self):
        with patch.object(main.platform, 'system', return_value='Linux'), \
             patch.object(main, 'verify_preparation', side_effect=ValueError('final freeze absent')), \
             patch.object(main, 'pin_runtime') as gpu, \
             patch('large_qubit.section9_scoring.PhysicsClient') as physics, \
             patch('large_qubit.section8_inference.ResidentGenerator') as model, \
             self.assertRaisesRegex(ValueError, 'final freeze absent'):
            main.run_matrix(self.root, execute=True)
        gpu.assert_not_called()
        physics.assert_not_called()
        model.assert_not_called()
        self.assertFalse(main.directory(self.root).exists())

    def test_failed_target_references_account_for_all_jobs_without_model_loading(self):
        with self.fixture_components(fail_references=True):
            result = main.run_matrix(self.root, execute=True)
        self.assertTrue(result['matrix_execution_complete'])
        self.assertFalse(result['all_trajectories_successful'])
        self.assertEqual(self.events['references'], list(TARGETS))
        self.assertEqual(self.events['model_loads'], [])
        self.assertEqual(self.events['scores'], [])
        dest = main.directory(self.root)
        for method, alias in jobs():
            report = load(dest / job_id(method, alias) / 'report.json')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['logical_requests'], 0)
            self.assertFalse(report['primary_endpoint_available'])

    def test_numerical_vqe_failure_is_terminal_and_keeps_verified_incumbent(self):
        def early_fixture_engine(ledger, scorer, method, alias, **kwargs):
            scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
            if method == 'conventional_vqe':
                ledger.halt('Numerical incident: fixture Fourier identity mismatch')
                raise ValueError('fixture Fourier identity mismatch')
            ledger.halt('Fixture early stop; other method engines have separate full-budget coverage')
            return {'snapshot': ledger.snapshot(), 'optimizer': []}
        with self.fixture_components(), patch.object(main, 'execute_trajectory', side_effect=early_fixture_engine):
            result = main.run_matrix(self.root, execute=True)
        self.assertTrue(result['matrix_execution_complete'])
        self.assertFalse(result['all_trajectories_successful'])
        dest = main.directory(self.root)
        for alias in TARGETS:
            report = load(dest / job_id('conventional_vqe', alias) / 'report.json')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['logical_requests'], 1)
            self.assertIn('Numerical incident:', report['failure']['reason'])
            self.assertTrue(report['primary_endpoint_available'])
            self.assertEqual(report['best_verified_error_hartree'], 1.)

    def test_missing_initialized_cache_refuses_without_recreating_or_resetting_ledgers(self):
        with self.fixture_components(fail_references=True):
            main.run_matrix(self.root, execute=True)
            dest = main.directory(self.root)
            ledgers = {job_id(m, a): sha256_file(dest / job_id(m, a) / 'ledger.sqlite')
                       for m, a in jobs()}
            # Remove only fixture files to simulate a lost cache after initialization.
            (dest / 'completed.json').unlink()
            (dest / 'scalar_cache.sqlite').unlink()
            before = {k: len(self.events[k]) for k in ('references', 'scores', 'model_loads')}
            with self.assertRaisesRegex(ValueError, 'initialized scalar cache is missing'):
                main.run_matrix(self.root, execute=True, resume=True)
            self.assertFalse((dest / 'scalar_cache.sqlite').exists())
            self.assertTrue((dest / 'cache_initialized.json').exists())
            for name, expected in ledgers.items():
                self.assertEqual(sha256_file(dest / name / 'ledger.sqlite'), expected)
            for key, count in before.items():
                self.assertEqual(len(self.events[key]), count)

    def test_missing_initialized_job_ledger_refuses_without_resetting_its_budget(self):
        with self.fixture_components(fail_references=True):
            main.run_matrix(self.root, execute=True)
            dest = main.directory(self.root)
            folder = dest / job_id(*jobs()[0])
            identity_sha = sha256_file(folder / 'identity.json')
            report_sha = sha256_file(folder / 'report.json')
            # Remove only fixture files to simulate a lost ledger with its identity retained.
            (dest / 'completed.json').unlink()
            (folder / 'ledger.sqlite').unlink()
            before = {k: len(self.events[k]) for k in ('references', 'scores', 'model_loads')}
            with self.assertRaisesRegex(ValueError, 'initialized job ledger is missing'):
                main.run_matrix(self.root, execute=True, resume=True)
            self.assertFalse((folder / 'ledger.sqlite').exists())
            self.assertEqual(sha256_file(folder / 'identity.json'), identity_sha)
            self.assertEqual(sha256_file(folder / 'report.json'), report_sha)
            for key, count in before.items():
                self.assertEqual(len(self.events[key]), count)

    def test_full_sixteen_budgets_generation_resume_seal_resume_and_completed_skip(self):
        dest = main.directory(self.root)
        actual_save = main.save
        fail_seal = [True]
        def save_with_one_seal_crash(path, value):
            if Path(path) == dest / 'completed.json' and fail_seal[0]:
                fail_seal[0] = False
                raise RuntimeError('fixture crash before final matrix seal')
            return actual_save(path, value)
        with self.fixture_components(interrupt_generation=True), \
             patch.object(main, 'save', side_effect=save_with_one_seal_crash):
            with self.assertRaises(KeyboardInterrupt):
                main.run_matrix(self.root, execute=True)
            self.assertTrue(self.events['generation_interrupted'])
            first_folder = dest / job_id(METHODS[0], TARGETS[0])
            self.assertFalse((first_folder / 'report.json').exists())
            with self.assertRaisesRegex(RuntimeError, 'before final matrix seal'):
                main.run_matrix(self.root, execute=True, resume=True)
            self.assertFalse((dest / 'completed.json').exists())
            self.assertEqual(len(self.events['references']), 2)
            self.assertEqual(len(self.events['verifications']), 16)
            self.assertEqual(sum((dest / job_id(m, a) / 'report.json').exists() for m, a in jobs()), 16)
            before = {key: len(self.events[key]) for key in ('scores', 'references', 'verifications', 'model_loads')}
            result = main.run_matrix(self.root, execute=True, resume=True)
            self.assertTrue(result['matrix_execution_complete'])
            self.assertTrue(result['all_trajectories_successful'])
            for key, count in before.items():
                self.assertEqual(len(self.events[key]), count, 'Sealing resume repeated ' + key)
            # The sealed fast path authenticates existing artifacts without even rebuilding CPU clients.
            before = {key: len(values) for key, values in self.events.items() if isinstance(values, list)}
            self.assertTrue(main.run_matrix(self.root, execute=True)['all_passed'])
            for key, count in before.items():
                self.assertEqual(len(self.events[key]), count, 'Completed matrix repeated ' + key)
        reports = {job_id(m, a): load(dest / job_id(m, a) / 'report.json') for m, a in jobs()}
        self.assertEqual(len(reports), 16)
        self.assertEqual(sum(r['logical_requests'] for r in reports.values()), 16382)
        for method, alias in jobs():
            report = reports[job_id(method, alias)]
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['logical_requests'], 1023 if method == 'conventional_vqe' else 1024)
            self.assertEqual(report['best_verified_error_hartree'], 1.)
            self.assertEqual(report['hf_improvement_hartree'], 0.)
            self.assertTrue(report['primary_endpoint_available'])
            self.assertTrue((dest / job_id(method, alias) / 'circuit_resources.json').exists())
        interrupted = reports[job_id(METHODS[0], TARGETS[0])]
        self.assertEqual(interrupted['counts']['abandoned_completion_attempts'], 1)
        self.assertEqual(interrupted['completion_attempts'], 1024)
        self.assertEqual(len(self.events['model_loads']), 3)  # GPT reload after interruption, then Qwen.
        self.assertEqual(len(list((dest / 'execution_attempts').glob('attempt_*'))), 3)
        self.assertEqual(len(list((dest / 'execution_attempts').glob('attempt_*/costs.json'))), 2)


if __name__ == '__main__':
    unittest.main()
