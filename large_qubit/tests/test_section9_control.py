"""CPU-only checks for continuation on reassigned GPUs without resetting science."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_control as control
from large_qubit.common import sha256_file
from large_qubit.section8 import save, load
from large_qubit.section9_ledger import TrajectoryLedger, verify_ledger
from large_qubit.section9_recovery_policy import PREPARATION, lane_specs, job_id
from large_qubit.section9_recovery_worker import ledger_identity
from large_qubit.tests.test_section9_recovery import allocation
from large_qubit.tests.test_section9_recovery_worker import physical_identity


def changed_allocation():
    result = allocation()
    for slot, spec in enumerate(lane_specs(), 11):
        row = result['lanes'][spec['lane_id']]
        row['gpu_uuid'] = 'GPU-00000000-0000-0000-0000-' + str(slot).zfill(12)
        row['runtime']['gpu_uuid'] = row['gpu_uuid']
    return result


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = control.directory(self.root)
        self.prepared = {'final_freeze_sha256': 'f' * 64, 'fixture_only': True}
        save(self.root / PREPARATION, self.prepared)
        save(self.root / control.REGISTRATION, {'fixture_only': True})
        save(self.dest / 'gpu_allocation.json', allocation())
        save(self.dest / 'binding.json', {'fixture_only': True})
        self.ledgers = []
        for spec in lane_specs():
            lane = self.dest / 'lanes' / spec['lane_id']
            save(lane / 'binding.json', {'fixture_only': True})
            save(lane / 'gpu_runtime.json', allocation()['lanes'][spec['lane_id']]['runtime'])
            target = self.dest / 'targets' / spec['target'] / 'physics_identity.json'
            if not target.exists():
                save(target, physical_identity(spec['target']))
            # All sixteen trajectories have genuine authenticated, quiescent ledgers.
            for method in spec['methods']:
                folder = lane / job_id(method, spec['target'])
                identity = ledger_identity(sha256_file(self.root / PREPARATION), spec['lane_id'],
                    method, spec['target'], physical_identity(spec['target'])['cache_identity'])
                save(folder / 'identity.json', identity)
                ledger = TrajectoryLedger(folder / 'ledger.sqlite', identity)
                ledger.reserve_request('hf', {'geometry_id': spec['target'], 'kind': 'hf'}, kind='hf')
                ledger.complete_request(1, -90.)
                ledger.close()
                self.ledgers.append((folder / 'ledger.sqlite', identity))

    def gates(self, proposed=None):
        stack = ExitStack()
        stack.enter_context(patch.object(control.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(control, 'verify_binding', side_effect=lambda root:
            (self.prepared, control.allocation_history(root)[-1][1])))
        stack.enter_context(patch.object(control.original, 'discover_allocation',
                                         return_value=proposed or changed_allocation()))
        stack.enter_context(patch.object(control.original, 'interruption_signals', return_value=nullcontext()))
        return stack

    def record_epoch(self, proposed=None):
        with self.gates(proposed):
            return control.reassign(self.root, execute=True)

    def test_all_runtime_fields_except_physical_uuid_must_remain_identical(self):
        self.assertEqual(control.compatible_allocation(changed_allocation(), allocation()), changed_allocation())
        for field, value in (('gpu_name', 'NVIDIA A100'), ('precision', 'fp32'),
                             ('device', 'cuda:1'), ('fixture_only', False)):
            bad = changed_allocation()
            bad['lanes']['lane_2']['runtime'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.compatible_allocation(bad, allocation())

    def test_reassignment_preserves_every_origin_file_and_all_sixteen_budgets(self):
        before = {p: p.read_bytes() for p in self.dest.rglob('*') if p.is_file()}
        result = self.record_epoch()
        self.assertEqual(result['epoch'], 1)
        self.assertEqual(result['scientific_requests_added'], 0)
        self.assertFalse(result['model_weights_loaded'])
        self.assertTrue(all(path.read_bytes() == contents for path, contents in before.items()))
        history = control.allocation_history(self.root)
        self.assertEqual([a for _, a in history], [allocation(), changed_allocation()])
        receipt = load(self.dest / 'gpu_transitions/epoch_0001.json')
        self.assertEqual(len(receipt['paused_frontiers']), 16)
        self.assertEqual(sum(v['logical_requests'] for v in receipt['paused_frontiers'].values()), 16)
        bound = control.execution_binding(self.root, 'lane_2')
        self.assertEqual(bound['allocation_epoch_sha256'], history[1][0])
        self.assertEqual(bound['allocation'], changed_allocation()['lanes']['lane_2'])

    def test_ledger_may_advance_after_transition_without_losing_authenticated_prefix(self):
        self.record_epoch()
        path, identity = self.ledgers[0]
        ledger = TrajectoryLedger(path, identity)
        ledger.reserve_request('candidate_2', {'kind': 'fixture', 'ordinal': 2})
        ledger.complete_request(2, -90.1)
        ledger.close()
        self.assertEqual(len(control.allocation_history(self.root)), 2)
        self.assertEqual(verify_ledger(path, identity)['logical_requests'], 2)

    def test_self_consistent_rewritten_ledger_still_fails_saved_prefix(self):
        self.record_epoch()
        path, identity = self.ledgers[0]
        original_bytes = path.read_bytes()
        path.unlink()
        replacement = TrajectoryLedger(path, identity)
        replacement.reserve_request('hf', {'kind': 'different-HF-query'}, kind='hf')
        replacement.complete_request(1, -89.)
        replacement.close()
        self.assertNotEqual(path.read_bytes(), original_bytes)
        self.assertEqual(verify_ledger(path, identity)['logical_requests'], 1)
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            control.allocation_history(self.root)

    def test_missing_ledger_never_reinitializes_or_refills_its_budget(self):
        self.record_epoch()
        path, _ = self.ledgers[0]
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            control.allocation_history(self.root)
        self.assertFalse(path.exists())

    def test_multiple_epochs_chain_and_missing_middle_epoch_is_rejected(self):
        first = self.record_epoch()
        second = self.record_epoch(allocation())
        self.assertEqual([first['epoch'], second['epoch']], [1, 2])
        self.assertEqual(len(control.allocation_history(self.root)), 3)
        path = self.dest / 'gpu_transitions/epoch_0001.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'history has a gap'):
            control.allocation_history(self.root)

    def test_last_used_epoch_cannot_disappear_even_when_no_later_epoch_exists(self):
        self.record_epoch()
        value = control.execution_binding(self.root, 'lane_0')
        value['allocation'] = changed_allocation()
        save(self.dest / 'execution_attempts/attempt_0001/control_binding.json', value)
        self.assertEqual(len(control.allocation_history(self.root)), 2)
        (self.dest / 'gpu_transitions/epoch_0001.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing GPU allocation epoch'):
            control.allocation_history(self.root)

    def test_coordinator_binding_authenticates_registration_origin_and_full_allocation(self):
        self.record_epoch()
        valid = control.execution_binding(self.root, 'lane_0')
        valid['allocation'] = changed_allocation()
        path = self.dest / 'execution_attempts/attempt_0001/control_binding.json'
        save(path, valid)
        self.assertEqual(len(control.allocation_history(self.root)), 2)
        for field, replacement in (('registration_sha256', '0' * 64),
                                   ('origin_binding_sha256', '0' * 64),
                                   ('allocation', allocation())):
            invalid = copy.deepcopy(valid)
            invalid[field] = replacement
            path.unlink()
            save(path, invalid)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Coordinator execution binding'):
                control.allocation_history(self.root)
        path.unlink()
        save(path, valid)
        self.assertEqual(len(control.allocation_history(self.root)), 2)

    def test_worker_and_coordinator_use_the_same_operational_pause_flag(self):
        from large_qubit.section9_control_worker import pause_path
        self.assertEqual(pause_path(self.root), self.root / control.PAUSE_FLAG)

    def test_transition_tampering_fails_even_with_a_valid_envelope(self):
        self.record_epoch()
        path = self.dest / 'gpu_transitions/epoch_0001.json'
        value = load(path)
        value['previous_epoch_sha256'] = '0' * 64
        path.unlink()
        save(path, value)
        with self.assertRaisesRegex(ValueError, 'identity or history changed'):
            control.allocation_history(self.root)

    def test_busy_matrix_or_orphan_lane_prevents_gpu_probe(self):
        from large_qubit.training import _run_lock
        for suffix in ('matrix', 'lane_3'):
            with self.subTest(lease=suffix), self.gates(), \
                 _run_lock(self.root / control.OPERATIONS / suffix), \
                 patch.object(control.original, 'discover_allocation') as probe:
                with self.assertRaisesRegex(ValueError, 'run lock'):
                    control.reassign(self.root, execute=True)
                probe.assert_not_called()
        self.assertFalse((self.dest / 'gpu_transitions').exists())

    def test_pending_request_or_attempt_prevents_probe_and_epoch(self):
        for kind in ('request', 'attempt'):
            path, identity = self.ledgers[0 if kind == 'request' else 1]
            ledger = TrajectoryLedger(path, identity)
            if kind == 'request':
                ledger.reserve_request('candidate_2', {'kind': 'pending'})
            else:
                ledger.reserve_attempt('attempt_0001', {'seed': 22})
            ledger.close()
            with self.subTest(kind=kind), self.gates(), \
                 patch.object(control.original, 'discover_allocation') as probe:
                with self.assertRaisesRegex(ValueError, 'has not drained'):
                    control.reassign(self.root, execute=True)
                probe.assert_not_called()
        self.assertFalse((self.dest / 'gpu_transitions').exists())

    def test_pending_nonrepeatable_winner_verification_blocks_reassignment(self):
        save(self.ledgers[0][0].parent / 'winner_verification_started.json', {'fixture_only': True})
        with self.gates(), patch.object(control.original, 'discover_allocation') as probe:
            with self.assertRaisesRegex(ValueError, 'nonrepeatable verification'):
                control.reassign(self.root, execute=True)
            probe.assert_not_called()

    def test_completed_campaign_or_unchanged_allocation_is_not_restarted(self):
        with self.gates(allocation()), self.assertRaisesRegex(ValueError, 'unchanged'):
            control.reassign(self.root, execute=True)
        save(self.dest / 'completed.json', {'fixture_only': True})
        with self.gates(), patch.object(control.original, 'discover_allocation') as probe:
            with self.assertRaisesRegex(ValueError, 'Completed matrices'):
                control.reassign(self.root, execute=True)
            probe.assert_not_called()

    def test_resume_validates_allocation_before_clearing_pause_or_starting_workers(self):
        save(self.root / control.PAUSE_FLAG, {'fixture_only': True, 'utc': 'fixture'})
        flag = (self.root / control.PAUSE_FLAG).read_bytes()
        with self.gates(changed_allocation()), patch.object(control, 'launch_lanes') as launch:
            with self.assertRaisesRegex(ValueError, 'Selected GPUs differ'):
                control.resume(self.root, execute=True)
            launch.assert_not_called()
        self.assertEqual((self.root / control.PAUSE_FLAG).read_bytes(), flag)
        self.assertFalse((self.dest / 'execution_attempts').exists())

    def test_resume_does_not_clear_pause_when_a_generation_is_still_pending(self):
        save(self.root / control.PAUSE_FLAG, {'fixture_only': True})
        path, identity = self.ledgers[0]
        ledger = TrajectoryLedger(path, identity)
        ledger.reserve_attempt('attempt_0001', {'seed': 22})
        ledger.close()
        with self.gates(allocation()), patch.object(control, 'launch_lanes') as launch, \
             patch.object(control.original, 'discover_allocation') as probe:
            with self.assertRaisesRegex(ValueError, 'has not drained'):
                control.resume(self.root, execute=True)
            probe.assert_not_called()
            launch.assert_not_called()
        self.assertTrue((self.root / control.PAUSE_FLAG).exists())
        self.assertEqual(verify_ledger(path, identity)['pending_attempts'], 1)

    def test_clean_pause_writes_receipt_without_sealing_and_archives_prior_pause(self):
        old_request = {'fixture_only': True, 'request': 'initial pause'}
        save(self.root / control.PAUSE_FLAG, old_request)
        def pause_lanes(root, dest, attempt, allocation_value, chemistry):
            self.assertFalse((root / control.PAUSE_FLAG).exists())
            self.assertEqual(load(attempt / 'resumed_pause_request.json'), old_request)
            save(root / control.PAUSE_FLAG, {'fixture_only': True, 'request': 'new pause'})
            for spec in lane_specs():
                save(dest / 'lanes' / spec['lane_id'] / 'execution_attempts/attempt_0001/paused.json',
                     {'lane_paused': True})
        with self.gates(allocation()), patch.object(control, 'verify_reused_references'), \
             patch.object(control, 'launch_lanes', side_effect=pause_lanes):
            result = control.resume(self.root, execute=True)
        self.assertTrue(result['paused'])
        self.assertFalse(result['matrix_execution_complete'])
        attempt = self.dest / 'execution_attempts/attempt_0001'
        self.assertEqual(len(load(attempt / 'paused.json')['frontiers']), 16)
        self.assertTrue((attempt / 'finished.json').exists())
        self.assertTrue((self.root / control.PAUSE_FLAG).exists())
        self.assertFalse((self.dest / 'completed.json').exists())
        self.assertEqual(sum(verify_ledger(p, i)['logical_requests'] for p, i in self.ledgers), 16)

    def test_returning_unfinished_without_pause_is_an_infrastructure_failure(self):
        with self.gates(allocation()), patch.object(control, 'verify_reused_references'), \
             patch.object(control, 'launch_lanes'):
            with self.assertRaisesRegex(ValueError, 'without a pause'):
                control.resume(self.root, execute=True)
        self.assertFalse((self.dest / 'completed.json').exists())
        self.assertTrue((self.dest / 'execution_attempts/attempt_0001/failed.json').exists())

    def test_existing_completion_verifies_without_probe_or_new_attempt(self):
        save(self.dest / 'completed.json', {'fixture_only': True})
        expected = {'all_passed': True, 'matrix_execution_complete': True}
        with self.gates(), patch.object(control, 'verify_matrix', return_value=expected), \
             patch.object(control.original, 'discover_allocation') as probe:
            self.assertEqual(control.resume(self.root, execute=True), expected)
            probe.assert_not_called()
        self.assertFalse((self.dest / 'execution_attempts').exists())

    def test_no_authorization_cannot_touch_gpu_or_leases(self):
        with patch.object(control, '_leases') as leases, \
             patch.object(control.original, 'discover_allocation') as probe:
            for action in (control.resume, control.reassign):
                with self.assertRaisesRegex(ValueError, '--execute'):
                    action(self.root)
            leases.assert_not_called()
            probe.assert_not_called()


class ControlIntegrationTests(unittest.TestCase):
    """Actual coordinator, workers, caches, ledger replay and cost seals; fake science."""

    def setUp(self):
        from large_qubit.tests.test_section9_recovery_integration import IntegrationTests
        from large_qubit import section9_recovery_worker
        IntegrationTests.setUp(self)
        save(self.root / control.REGISTRATION, {'fixture_only': True})
        self.transport_worker = section9_recovery_worker
        self.bootstrap = False

    def launch_lanes(self, root, dest, attempt, allocation_value, chemistry_python):
        failures = []
        for spec in lane_specs():
            name = spec['lane_id']
            folder = dest / 'lanes' / name
            if (folder / 'completed.json').exists():
                self.transport_worker.verify_lane(root, name)
                continue
            self.events['lane_launches'].append(name)
            receipt = attempt / 'workers' / name
            save(receipt / 'launch.json', {'lane_id': name, 'fixture_only': True,
                'gpu_uuid': allocation_value['lanes'][name]['gpu_uuid']})
            save(receipt / 'started.json', {'pid': None, 'fixture_only': True})
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': allocation_value['lanes'][name]['gpu_uuid']}):
                try:
                    self.transport_worker.run_lane(root, name, execute=True, resume=folder.exists(),
                                                   chemistry_python=chemistry_python)
                except KeyboardInterrupt as error:
                    if not self.bootstrap:
                        raise
                    failures.append(error)
            save(receipt / 'exit.json', {'exit_code': 2 if self.bootstrap else 0, 'fixture_only': True})
        if failures:
            raise KeyboardInterrupt('fixture initial coordinator interruption')
        return {s['lane_id']: 0 for s in lane_specs()}

    def test_pause_reassign_resume_preserves_all_sixteen_trajectories_and_costs(self):
        from large_qubit import section9_recovery as original, section9_recovery_worker as legacy_worker
        from large_qubit import section9_control_worker as worker
        from large_qubit.tests.test_section9_recovery_integration import IntegrationTests
        from large_qubit.tests.test_section9_recovery_worker import LaneTests

        def initial_hf_then_interrupt(ledger, scorer, method, alias, **kwargs):
            scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
            raise KeyboardInterrupt('fixture interruption after durable HF')

        def pause_before_attempt(ledger, scorer, method, alias, **kwargs):
            # The HF request already exists and must replay without charging again.
            scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
            save(self.root / control.PAUSE_FLAG, {'fixture_only': True, 'request': 'pause before new work'})
            ledger.reserve_attempt('attempt_0001', {'seed': 22})
            raise AssertionError('A new generation attempt must not be admitted after pause')

        def pin(root, destination):
            runtime = next(row['runtime'] for row in self.allocation['lanes'].values()
                           if row['gpu_uuid'] == os.environ['CUDA_VISIBLE_DEVICES'])
            save(destination / 'gpu_runtime.json', runtime)
            return runtime

        with IntegrationTests.fixtures(self), \
             patch('large_qubit.section9_control_registration.verify', return_value={'all_passed': True}), \
             patch.object(worker, 'pin_runtime', side_effect=pin), \
             patch.object(control, 'launch_lanes', side_effect=self.launch_lanes):
            self.bootstrap = True
            with patch.object(legacy_worker, 'execute_trajectory', side_effect=initial_hf_then_interrupt):
                with self.assertRaisesRegex(KeyboardInterrupt, 'initial coordinator interruption'):
                    original.run_matrix(self.root, execute=True)
            self.bootstrap = False
            self.transport_worker = worker
            self.assertEqual(sum(v['logical_requests'] for v in control._lane_frontiers(self.root).values()), 4)
            origins = {p: p.read_bytes() for p in self.dest.rglob('*')
                       if p.is_file() and p.name in ('binding.json', 'gpu_runtime.json', 'gpu_allocation.json')}
            with patch.object(worker, 'execute_trajectory', side_effect=pause_before_attempt):
                paused = control.resume(self.root, execute=True)
            self.assertTrue(paused['paused'])
            self.assertFalse((self.dest / 'completed.json').exists())
            self.assertEqual(sum(v['logical_requests'] for v in control._lane_frontiers(self.root).values()), 4)
            self.assertEqual(sum(v['completion_attempts'] for v in control._lane_frontiers(self.root).values()), 0)

            self.allocation.clear()
            self.allocation.update(changed_allocation())
            moved = control.reassign(self.root, execute=True)
            self.assertEqual(moved['epoch'], 1)
            self.assertEqual(moved['scientific_requests_added'], 0)
            with patch.object(worker, 'execute_trajectory', side_effect=LaneTests.short_engine):
                complete = control.resume(self.root, execute=True)
            self.assertTrue(complete['all_passed'])
            self.assertTrue(complete['matrix_execution_complete'])
            self.assertEqual(complete['trajectories_accounted'], 16)
            self.assertFalse(complete['all_trajectories_successful'])  # Intentional short fixture outcomes.
            self.assertTrue(control.verify_matrix(self.root)['all_passed'])
            before_events = copy.deepcopy(self.events)
            self.assertTrue(control.resume(self.root, execute=True)['all_passed'])
            self.assertEqual(self.events, before_events)
            self.assertEqual(self.events['references'], [])
            self.assertEqual(len(self.events['verifications']), 16)
            self.assertTrue(all(path.read_bytes() == data for path, data in origins.items()))
            IntegrationTests.assert_preserved_original(self)

        receipt = load(self.dest / 'completed.json')
        costs = load(self.root / receipt['costs_path'])
        self.assertEqual(costs['ordinary_request_accounting'], {
            'failed_attempt': 16, 'recovery_attempt': 26, 'cumulative': 42,
            'maximum_cumulative': 16400, 'recovery_maximum': 16384})
        self.assertEqual(costs['additional_reference_solves'], 0)
        self.assertEqual(len(costs['coordinator_attempts']), 3)
        self.assertEqual(len(costs['gpu_control']['allocation_epochs']), 2)
        self.assertFalse(costs['gpu_control']['scientific_budgets_refilled'])
        self.assertEqual(len(costs['gpu_control']['pause_requests']), 1)
        self.assertEqual(len(costs['lanes']), 4)
        self.assertTrue(all(len(v['execution_attempts']) == 3 for v in costs['lanes'].values()))
        self.assertEqual(len(load(self.dest / 'matrix_report.json')['jobs']), 16)
        self.assertFalse((self.root / control.PAUSE_FLAG).exists())


if __name__ == '__main__':
    unittest.main()
