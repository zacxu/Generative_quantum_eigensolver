"""CPU fixtures for GPU epoch binding, safe pauses and unchanged durable work."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from large_qubit import section9, section9_control_worker as worker
from large_qubit import section9_recovery_worker as original_worker
from large_qubit.common import sha256_file
from large_qubit.section8 import load, save
from large_qubit.section9_ledger import ScalarCache
from large_qubit.section9_recovery_policy import OUTPUT_ROOT, MATRIX, lane_specs
from large_qubit.tests import test_section9_recovery_worker as legacy


class ControlWorkerTests(unittest.TestCase):
    short_engine = staticmethod(legacy.LaneTests.short_engine)

    def setUp(self):
        legacy.LaneTests.setUp(self)
        self.origin = copy.deepcopy(load(self.dest / 'gpu_allocation.json'))
        self.history = [(sha256_file(self.dest / 'gpu_allocation.json'), self.origin)]

    def execution(self, lane_id):
        epoch, allocation = self.history[-1]
        return {'registration_sha256': 'c' * 64,
                'origin_binding_sha256': sha256_file(self.dest / 'binding.json'),
                'allocation_epoch_sha256': epoch,
                'allocation': allocation['lanes'][lane_id]}

    def fixtures(self, lane_id='lane_0', *, early_engine=True):
        parent = types.ModuleType('large_qubit.section9_control')
        parent.verify_binding = Mock(return_value=(self.prepared, {'lanes': self.allocation}))
        parent.directory = lambda root: root / OUTPUT_ROOT / MATRIX
        parent.verify_reused_references = __import__(
            'large_qubit.section9_recovery', fromlist=['verify_reused_references']).verify_reused_references
        parent.allocation_history = lambda root: self.history
        parent.registered_control_sha = lambda root: 'c' * 64
        parent.execution_binding = lambda root, lane: self.execution(lane)
        with patch.object(legacy, 'worker', worker):
            stack = legacy.LaneTests.fixtures(self, lane_id, early_engine=early_engine)
        stack.enter_context(patch.dict('sys.modules', {'large_qubit.section9_control': parent}))
        return stack

    def migrate(self):
        self.allocation = copy.deepcopy(self.allocation)
        for lane_id, allocation in self.allocation.items():
            uuid = 'GPU-new-' + lane_id
            allocation['gpu_uuid'] = allocation['runtime']['gpu_uuid'] = uuid
        self.history.append(('e' * 64, {'schema_version': 'fixture', 'lanes': self.allocation}))

    def test_original_controller_ledger_and_lane_bindings_are_identical(self):
        self.assertIs(worker.execute_trajectory, section9.execute_trajectory)
        for spec in lane_specs():
            self.assertEqual(worker._binding(self.root, self.dest, spec),
                             original_worker._binding(self.root, self.dest, spec))
            for method in spec['methods']:
                args = ('a' * 64, spec['lane_id'], method, spec['target'],
                        legacy.physical_identity(spec['target'])['cache_identity'])
                self.assertEqual(worker.ledger_identity(*args), original_worker.ledger_identity(*args))

    def test_new_gpu_records_actual_runtime_without_rewriting_original(self):
        original_bytes = (self.dest / 'gpu_allocation.json').read_bytes()
        self.migrate()
        with self.fixtures():
            result = worker.run_lane(self.root, 'lane_0', execute=True)
        lane = self.dest / 'lanes/lane_0'
        self.assertTrue(result['all_passed'])
        self.assertEqual((self.dest / 'gpu_allocation.json').read_bytes(), original_bytes)
        self.assertEqual(load(lane / 'gpu_runtime.json'), self.origin['lanes']['lane_0']['runtime'])
        attempt = next((lane / 'execution_attempts').glob('attempt_*'))
        self.assertEqual(load(attempt / 'gpu_runtime.json'), self.allocation['lane_0']['runtime'])
        self.assertEqual(load(attempt / 'control_binding.json'), self.execution('lane_0'))
        self.assertEqual(load(lane / 'completed.json')['allocation_epoch_sha256'], 'e' * 64)

    def test_unregistered_epoch_and_altered_runtime_are_refused(self):
        lane = self.dest / 'lanes/lane_0'
        attempt = lane / 'execution_attempts/attempt_0001'
        value = self.execution('lane_0')
        value['allocation_epoch_sha256'] = 'x' * 64
        save(attempt / 'control_binding.json', value)
        with self.fixtures(), self.assertRaisesRegex(ValueError, 'outside the authenticated GPU history'):
            worker.verify_execution_history(self.root, self.dest, lane, 'lane_0')
        (attempt / 'control_binding.json').unlink()
        save(attempt / 'control_binding.json', self.execution('lane_0'))
        wrong = dict(self.allocation['lane_0']['runtime'], gpu_uuid='GPU-unregistered')
        save(attempt / 'gpu_runtime.json', wrong)
        with self.fixtures(), self.assertRaisesRegex(ValueError, 'differs from its allocation epoch'):
            worker.verify_execution_history(self.root, self.dest, lane, 'lane_0')

    def test_old_completed_lane_is_verified_and_skipped_byte_for_byte(self):
        with legacy.LaneTests.fixtures(self):
            original_worker.run_lane(self.root, 'lane_0', execute=True)
        lane = self.dest / 'lanes/lane_0'
        before = {p.relative_to(lane): p.read_bytes() for p in lane.rglob('*') if p.is_file()}
        self.migrate()
        self.events['scores'].clear()
        self.events['model_loads'].clear()
        with self.fixtures(), patch.object(worker, 'pin_runtime') as pin:
            checked = worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
        self.assertTrue(checked['all_passed'])
        pin.assert_not_called()
        self.assertEqual(self.events['scores'], [])
        self.assertEqual(self.events['model_loads'], [])
        after = {p.relative_to(lane): p.read_bytes() for p in lane.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_existing_epoch_refuses_before_gpu_startup(self):
        lane = self.dest / 'lanes/lane_0'
        attempt = lane / 'execution_attempts/attempt_0001'
        invalid = dict(self.execution('lane_0'), allocation_epoch_sha256='x' * 64)
        save(attempt / 'control_binding.json', invalid)
        with self.fixtures(), patch.object(worker, 'pin_runtime') as pin, \
             patch('large_qubit.section7._gpu_operation_lock') as gpu_lock, \
             self.assertRaisesRegex(ValueError, 'outside the authenticated GPU history'):
            worker.run_lane(self.root, 'lane_0', execute=True, resume=True)
        pin.assert_not_called()
        gpu_lock.assert_not_called()
        self.assertEqual(self.events['scores'], [])
        self.assertEqual(self.events['model_loads'], [])

    def test_cooperative_lane_pause_saves_cost_without_a_terminal_seal(self):
        def request_pause(ledger, scorer, method, alias, **kwargs):
            scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
            marker = worker.pause_path(self.root)
            save(marker, {'reason': 'fixture requested pause'})
            ledger.reserve_attempt('attempt_0001', {'seed': 1})
            self.fail('Paused attempt must not be admitted')
        with self.fixtures(), patch.object(worker, 'execute_trajectory', side_effect=request_pause):
            result = worker.run_lane(self.root, 'lane_0', execute=True)
        self.assertTrue(result['lane_paused'])
        lane = self.dest / 'lanes/lane_0'
        self.assertFalse((lane / 'completed.json').exists())
        attempt = next((lane / 'execution_attempts').glob('attempt_*'))
        self.assertTrue(load(attempt / 'finished.json')['paused'])
        self.assertFalse((attempt / 'failed.json').exists())
        self.assertEqual(self.events['closed_models'], 1)
        self.assertEqual(self.events['closed_physics'], 1)
        costs = worker.retained_lane_costs(self.root, lane, {})
        self.assertEqual(costs['execution_attempts'][0]['status'], 'finished')


class PausableControllerTests(unittest.TestCase):
    def test_pause_drains_already_admitted_physics_without_cancelling_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pause = root / 'pause_requested.json'
            alias = 'CO2_I01'
            identity = worker.ledger_identity('a' * 64, 'lane_0', 'gpt2_xl_frozen', alias,
                                               legacy.physical_identity(alias)['cache_identity'])
            ledger = worker.PausableLedger(root / 'ledger.sqlite', identity, pause_marker=pause)
            cache = ScalarCache(root / 'cache.sqlite')
            entered, released = threading.Event(), threading.Event()
            calls = []

            def evaluate(query):
                calls.append(query)
                entered.set()
                if not released.wait(3):
                    raise AssertionError('Fixture request failed to drain')
                return legacy.evaluation(query)

            timer = None
            try:
                with self.assertRaises(worker.PauseRequested):
                    with worker.PausableScorer(ledger, cache, identity['cache_identity'], evaluate,
                                               pause_marker=pause) as scorer:
                        scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf')
                        self.assertTrue(entered.wait(3))
                        save(pause, {'fixture': 'pause while physical work is active'})
                        timer = threading.Timer(.05, released.set)
                        timer.start()
                        scorer.submit('never_admitted', {'geometry_id': alias, 'kind': 'discrete', 'ops': [0] * 32})
                snap = ledger.snapshot()
                self.assertEqual(len(calls), 1)
                self.assertEqual(snap['logical_requests'], 1)
                self.assertEqual(snap['completed_requests'], 1)
                self.assertEqual(snap['pending_requests'], 0)
                self.assertEqual(snap['abandoned_requests'], 0)
                self.assertFalse(snap['halted'])
            finally:
                released.set()
                if timer is not None:
                    timer.join()
                ledger.close()
                cache.close()

    def test_generated_answer_survives_pause_and_resume_without_extra_energy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pause = root / 'pause_requested.json'
            alias, method = 'CO2_I01', 'gpt2_xl_frozen'
            identity = worker.ledger_identity('a' * 64, 'lane_0', method, alias,
                                               legacy.physical_identity(alias)['cache_identity'])
            ledger = worker.PausableLedger(root / 'ledger.sqlite', identity, pause_marker=pause)
            cache = ScalarCache(root / 'cache.sqlite')
            generated, evaluated = [], []

            class Generator:
                def generate(self, geometry, catalogue, seed):
                    generated.append(seed)
                    save(pause, {'fixture_pause_after_generation': len(generated)})
                    return {'valid': True, 'ops': [len(generated) - 1] * 32,
                            'attempt_seed': seed, 'geometry_id': geometry, 'condition_id': method,
                            'wall_seconds': 0.0, 'input_tokens': 1, 'output_tokens': 1}

            def evaluate(query):
                evaluated.append(query)
                return legacy.evaluation(query)

            try:
                for expected_attempts, expected_requests in ((1, 1), (2, 2)):
                    with self.assertRaises(worker.PauseRequested):
                        with worker.PausableScorer(ledger, cache, identity['cache_identity'], evaluate,
                                                   pause_marker=pause) as scorer:
                            section9.execute_trajectory(ledger, scorer, method, alias,
                                                        catalogue={}, generator=Generator())
                    snapshot = ledger.snapshot()
                    self.assertEqual(snapshot['completion_attempts'], expected_attempts)
                    self.assertEqual(snapshot['logical_requests'], expected_requests)
                    self.assertEqual(snapshot['completed_requests'], expected_requests)
                    self.assertEqual(snapshot['pending_requests'], 0)
                    self.assertEqual(snapshot['pending_attempts'], 0)
                    self.assertEqual(snapshot['abandoned_requests'], 0)
                    self.assertEqual(snapshot['failed_requests'], 0)
                    self.assertFalse(snapshot['halted'])
                    self.assertFalse(snapshot['finished'])
                    self.assertEqual(len(evaluated), expected_requests)
                    pause.unlink()
                self.assertEqual(len(generated), 2)
                self.assertEqual(snapshot['requests'][1]['query']['ops'], [0] * 32)
                self.assertEqual(snapshot['attempts'][0]['output']['attempt_seed'], generated[0])
                self.assertNotEqual(generated[0], generated[1])
            finally:
                ledger.close()
                cache.close()


if __name__ == '__main__':
    unittest.main()
