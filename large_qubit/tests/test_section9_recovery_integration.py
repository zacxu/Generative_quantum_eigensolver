"""Real recovery ledgers/accounting with preserved references and CPU-only fake science."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_recovery as main, section9_recovery_worker as worker
from large_qubit.common import canonical_bytes, sha256_file
from large_qubit.protocol import FITS
from large_qubit.section8 import save, load
from large_qubit.section8_policy import CATALOGUE, FINAL_FREEZE
from large_qubit.section9_recovery_policy import REGISTRATION, PREPARATION, FAILED_ROOT, lane_specs
from large_qubit.tests import test_section9_parallel as coordinator_fixtures
from large_qubit.tests import test_section9_recovery_worker as lane_fixtures

ROOT = Path(__file__).resolve().parents[2]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = main.directory(self.root)
        self.allocation = coordinator_fixtures.allocation()
        self.events = {'references': [], 'scores': [], 'verifications': [],
                       'models': [], 'generations': [], 'lane_launches': []}
        self.prepared = {'final_freeze_sha256': 'f' * 64, 'release_sha256': 'r' * 64,
            'training': {'fits': {fit: {'adapter_path': 'fixture/' + fit} for fit in FITS}}}
        registration = self.root / REGISTRATION
        registration.parent.mkdir(parents=True)
        registration.write_bytes(b'CPU integration fixture registration\n')
        catalogue = self.root / CATALOGUE
        catalogue.parent.mkdir(parents=True)
        catalogue.write_bytes((ROOT / CATALOGUE).read_bytes())
        from large_qubit.section5 import SCREENING_PATH
        screening = self.root / SCREENING_PATH
        screening.parent.mkdir(parents=True)
        screening.write_bytes(canonical_bytes({'vqe_initial_angles': [0.] * 24}))
        from large_qubit.section7 import PREFLIGHT_ROOT
        numerical = self.root / PREFLIGHT_ROOT / 'freeze.json'
        numerical.parent.mkdir(parents=True)
        numerical.write_bytes(canonical_bytes({'preflight_reports': {}}))
        save(self.root / FINAL_FREEZE, {
            'costs': {'source_acquisition_charged_once': {'requests': 1, 'fixture_only': True},
                      'validation_panels': {'fixture_panel': {'requests': 1}}},
            'final_adapters_and_training': {
                'numerical_freeze_sha256': sha256_file(numerical),
                'fits': self.prepared['training']['fits'], 'fixture_only': True}})

        original = self.root / FAILED_ROOT
        save(original / 'gpu_allocation.json', self.allocation)
        refs = {}
        for alias in ('CO2_I01', 'CO2_I02'):
            target = original / 'targets' / alias
            identity = lane_fixtures.physical_identity(alias)
            save(target / 'physics_identity.json', identity)
            save(target / 'reference.json', {'status': 'completed', 'uncertain_cost': False, 'result': {
                'geometry_id': alias, 'partition': 'final_test', 'energy_hartree': -91.,
                'cache_identity': identity['cache_identity'], 'evaluator_calls': 2,
                'wall_seconds': .0001, 'fixture_only': True,
                'report': {'all_passed': True, 'converged': True, 'independent_starts': 2,
                    'hamiltonian_actions': 2, 'wall_seconds': .0001,
                    'runs': [{'converged': True, 'energy_ha': -91.,
                              'complex128_residual_norm_ha': 0., 'norm_squared': 1.} for _ in range(2)]}}})
            refs[alias] = sha256_file(target / 'reference.json')
        failed_costs = original / 'execution_attempts/attempt_0001/costs.json'
        save(failed_costs, {'lanes': {s['lane_id']: {'fixture_only': True, 'preserved_setup_seconds': 1.}
                                     for s in lane_specs()},
                            'coordinator_attempts': [{'fixture_only': True, 'status': 'finished'}]})
        save(original / 'completed.json', {'costs_path': failed_costs.relative_to(self.root).as_posix(),
                                          'fixture_only': True})
        self.failed = {'logical_requests': 16, 'reference_sha256': refs,
                       'physical_cost_retained': {'evaluator_calls': 16, 'kernel_contractions': 16,
                           'hamiltonian_action_seconds': 16., 'recorded_wall_seconds': 16.},
                       'fixture_only': True}
        self.prepared['failed_campaign'] = self.failed
        save(self.root / PREPARATION, self.prepared)
        self.original_inventory = {p.relative_to(original).as_posix(): sha256_file(p)
                                   for p in original.rglob('*') if p.is_file()}

    def assert_preserved_original(self):
        original = self.root / FAILED_ROOT
        self.assertEqual(self.original_inventory, {p.relative_to(original).as_posix(): sha256_file(p)
                         for p in original.rglob('*') if p.is_file()})

    def fixtures(self):
        events, allocation = self.events, self.allocation

        class Physics:
            def __init__(self, root, python, alias, **kwargs):
                self.alias, self.identity = alias, lane_fixtures.recovery_identity(alias)

            def reference(self):
                events['references'].append(self.alias)
                raise AssertionError('Recovery must never solve another target reference')

            def score(self, query):
                events['scores'].append(query)
                return lane_fixtures.evaluation(query)

            def verify(self, query):
                events['verifications'].append(query)
                return {**lane_fixtures.evaluation(query), 'fresh_basis_and_evaluator': True,
                        'independent_backend_claimed': False}

            def close(self):
                pass

        def current_runtime():
            return next(row['runtime'] for row in allocation['lanes'].values()
                        if row['gpu_uuid'] == os.environ['CUDA_VISIBLE_DEVICES'])

        class Generator:
            def __init__(self, root, key, **kwargs):
                self.model_key, self.condition = key, None
                self.metadata = {'runtime_identity': current_runtime(), 'fixture_only': True}
                events['models'].append(key)

            def switch(self, method, **kwargs):
                self.condition = method

            def generate(self, alias, catalogue, seed):
                events['generations'].append((self.condition, alias, seed))
                return {'valid': True, 'ops': [0] * 32, 'condition_id': self.condition,
                    'geometry_id': alias, 'attempt_seed': seed, 'input_tokens': 1, 'output_tokens': 1,
                    'wall_seconds': .00001, 'fixture_only': True}

            def statistics(self):
                return {'fixture_only': True, 'base_loads': 1, 'wall_seconds': .0001}

            def close(self):
                pass

        class Release:
            source_acquisition_cost = {'ordinary_requests': 1, 'fixture_only': True}

            def retrieval_rows(self):
                return []

        def pin(root, destination):
            value = current_runtime()
            save(destination / 'gpu_runtime.json', value)
            return value

        stack = ExitStack()
        stack.enter_context(patch.object(main.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(main, 'verify_preparation', return_value=self.prepared))
        stack.enter_context(patch.object(main, 'discover_allocation', return_value=allocation))
        stack.enter_context(patch.object(main, 'assert_no_legacy'))
        stack.enter_context(patch('large_qubit.section9_recovery_registration.failed_campaign', return_value=self.failed))
        stack.enter_context(patch.object(main, 'controller_runtime', return_value={'fixture_only': True}))
        stack.enter_context(patch.object(main, 'interruption_signals', return_value=nullcontext()))
        stack.enter_context(patch.object(worker, 'pin_runtime', side_effect=pin))
        stack.enter_context(patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()))
        stack.enter_context(patch('large_qubit.section9_recovery_scoring.PhysicsClient', Physics))
        stack.enter_context(patch('large_qubit.section8_inference.ResidentGenerator', Generator))
        stack.enter_context(patch('large_qubit.corpus_v2.load_release', return_value=Release()))
        stack.enter_context(patch('large_qubit.baselines.retrieval_candidates',
            return_value={'candidates': [{'ops': [0] * 32}] * 1023, 'fixture_only': True}))
        stack.enter_context(patch.object(worker, 'execute_trajectory',
                                        side_effect=lane_fixtures.LaneTests.short_engine))
        stack.enter_context(patch.object(main, 'launch_lanes', side_effect=self.launch_lanes))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def launch_lanes(self, root, dest, attempt, allocation, chemistry_python):
        """Sequential transport fixture; actual lane controller and all receipts are real."""
        exits = {}
        for spec in lane_specs():
            name = spec['lane_id']
            folder = dest / 'lanes' / name
            if (folder / 'completed.json').exists():
                worker.verify_lane(root, name)
                continue
            self.events['lane_launches'].append(name)
            receipt = attempt / 'workers' / name
            save(receipt / 'launch.json', {'lane_id': name, 'fixture_only': True,
                'gpu_uuid': allocation['lanes'][name]['gpu_uuid']})
            save(receipt / 'started.json', {'pid': None, 'fixture_only': True})
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': allocation['lanes'][name]['gpu_uuid']}):
                worker.run_lane(root, name, execute=True, resume=folder.exists(),
                                chemistry_python=chemistry_python)
            save(receipt / 'exit.json', {'exit_code': 0, 'fixture_only': True})
            exits[name] = 0
        return exits

    def test_all_sixteen_outcomes_seal_with_real_accounting_and_no_success_only_mean(self):
        with self.fixtures():
            result = main.run_matrix(self.root, execute=True)
            self.assertTrue(main.verify_matrix(self.root)['all_passed'])
            before = {key: list(value) for key, value in self.events.items()}
            self.assertTrue(main.status(self.root)['matrix_execution_complete'])
            self.assertEqual(self.events, before)
        self.assertTrue(result['matrix_execution_complete'])
        self.assertEqual(result['trajectories_accounted'], 16)
        self.assertFalse(result['all_trajectories_successful'])
        self.assertEqual(self.events['references'], [])
        self.assertEqual(len(self.events['verifications']), 16)
        self.assertEqual(len(self.events['models']), 4)
        report = load(self.dest / 'matrix_report.json')
        self.assertEqual(len(report['jobs']), 16)
        self.assertEqual(len(report['failed_jobs']), 16)
        self.assertTrue(all(v['mean_verified_error_hartree'] is None for v in report['methods'].values()))
        self.assertEqual(sum(v['logical_requests'] for v in report['jobs'].values()), 26)
        complete = load(self.dest / 'completed.json')
        costs = load(self.root / complete['costs_path'])
        self.assertEqual(len(costs['target_references_charged_once']), 2)
        self.assertEqual(len(costs['coordinator_attempts']), 1)
        self.assertEqual(len(costs['lanes']), 4)
        self.assertTrue(all(not value['shared_costs_charged_here'] for value in costs['lanes'].values()))
        self.assertTrue(costs['lane_local_caches_may_increase_physical_work'])
        self.assertEqual(costs['ordinary_request_accounting'], {
            'failed_attempt': 16, 'recovery_attempt': 26, 'cumulative': 42,
            'maximum_cumulative': 16400, 'recovery_maximum': 16384})
        self.assertEqual(costs['additional_reference_solves'], 0)
        self.assertEqual(costs['preserved_failed_main_attempt']['logical_requests'], 16)
        self.assertEqual(costs['preserved_failed_main_attempt']['physical_cost_retained'], self.failed['physical_cost_retained'])
        self.assert_preserved_original()
        for alias in ('CO2_I01', 'CO2_I02'):
            self.assertEqual(sha256_file(self.dest / 'targets' / alias / 'reference.json'),
                             self.failed['reference_sha256'][alias])
            self.assertEqual(load(self.dest / 'targets' / alias / 'reuse.json')['additional_reference_calls'], 0)

    def test_parent_seal_crash_resume_skips_completed_lanes_and_preserves_both_attempts(self):
        actual_save = main.save

        def fail_seal(path, value):
            if Path(path) == self.dest / 'completed.json':
                raise RuntimeError('fixture parent seal interruption')
            return actual_save(path, value)

        with self.fixtures():
            with patch.object(main, 'save', side_effect=fail_seal), \
                 self.assertRaisesRegex(RuntimeError, 'parent seal interruption'):
                main.run_matrix(self.root, execute=True)
            self.assertFalse((self.dest / 'completed.json').exists())
            self.assertTrue(all((self.dest / 'lanes' / spec['lane_id'] / 'completed.json').exists()
                                for spec in lane_specs()))
            before = {key: list(value) for key, value in self.events.items()}
            result = main.run_matrix(self.root, execute=True, resume=True)
            self.assertTrue(result['all_passed'])
            self.assertEqual(self.events, before)
            complete = load(self.dest / 'completed.json')
            costs = load(self.root / complete['costs_path'])
            self.assertEqual([row['status'] for row in costs['coordinator_attempts']], ['failed', 'finished'])
            self.assertEqual(len(costs['coordinator_attempts'][0]['workers']), 4)
            self.assertEqual(costs['coordinator_attempts'][1]['workers'], {})
            self.assertTrue(all(len(value['execution_attempts']) == 1 for value in costs['lanes'].values()))
            # A completed resume becomes a read-only verify, with no third attempt.
            self.assertTrue(main.run_matrix(self.root, execute=True, resume=True)['all_passed'])
            self.assertEqual(self.events, before)
            self.assertEqual(len(list((self.dest / 'execution_attempts').glob('attempt_*'))), 2)
            self.assertEqual(costs['ordinary_request_accounting']['cumulative'], 42)
            self.assertEqual(costs['additional_reference_solves'], 0)
            self.assert_preserved_original()

    def test_parent_inventory_rejects_tampered_lane_evidence_without_new_science(self):
        with self.fixtures():
            main.run_matrix(self.root, execute=True)
            before = {key: list(value) for key, value in self.events.items()}
            report = next((self.dest / 'lanes' / 'lane_0').glob('*/report.json'))
            report.write_bytes(report.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'matrix inventory changed'):
                main.verify_matrix(self.root)
            with self.assertRaisesRegex(ValueError, 'matrix inventory changed'):
                main.run_matrix(self.root, execute=True, resume=True)
            self.assertEqual(self.events, before)

    def test_changed_preserved_reference_refuses_before_any_lane_or_science(self):
        original = self.root / FAILED_ROOT / 'targets/CO2_I01/reference.json'
        original.write_bytes(original.read_bytes() + b' ')
        with self.fixtures(), self.assertRaisesRegex(ValueError, 'Preserved target reference changed'):
            main.run_matrix(self.root, execute=True)
        self.assertTrue(all(not value for value in self.events.values()))


if __name__ == '__main__':
    unittest.main()


