"""CPU fixtures for recovery admission, immutable references and supervision."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_recovery as main
from large_qubit.common import sha256_file
from large_qubit.section8 import save, load
from large_qubit.section9_recovery_policy import REGISTRATION, PREPARATION, lane_specs, policy


def allocation():
    rows = {}
    for spec in lane_specs():
        uuid = 'GPU-00000000-0000-0000-0000-' + str(spec['gpu_slot'] + 1).zfill(12)
        rows[spec['lane_id']] = {'gpu_uuid': uuid, 'runtime': {'gpu_uuid': uuid,
            'gpu_name': 'NVIDIA H200', 'device': 'cuda:0', 'fixture_only': True}}
    return {'schema_version': 'section9-four-gpu-allocation-v1', 'lanes': rows}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = {'final_freeze_sha256': 'f' * 64, 'fixture_only': True}
        save(self.root / PREPARATION, self.prepared)
        path = self.root / REGISTRATION
        path.parent.mkdir(parents=True)
        path.write_text('fixture', encoding='ascii')
        self.dest = main.directory(self.root)
        save(self.root / main.FAILED_ROOT / 'gpu_allocation.json', allocation())

    def gates(self):
        stack = ExitStack()
        stack.enter_context(patch.object(main.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(main, 'verify_preparation', return_value=self.prepared))
        stack.enter_context(patch.object(main, 'assert_no_legacy'))
        stack.enter_context(patch.object(main, 'discover_allocation', return_value=allocation()))
        stack.enter_context(patch.object(main, 'controller_runtime', return_value={'fixture_only': True}))
        stack.enter_context(patch.object(main, 'interruption_signals', return_value=nullcontext()))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_four_unique_devices_and_matching_uuid_required(self):
        self.assertEqual(main.validate_allocation(allocation()), allocation())
        for mutation in ('duplicate', 'missing', 'model', 'uuid'):
            value = allocation()
            if mutation == 'duplicate':
                value['lanes']['lane_3'] = copy.deepcopy(value['lanes']['lane_0'])
            elif mutation == 'missing':
                del value['lanes']['lane_3']
            elif mutation == 'model':
                value['lanes']['lane_2']['runtime']['gpu_name'] = 'NVIDIA A100'
            else:
                value['lanes']['lane_2']['runtime']['gpu_uuid'] = value['lanes']['lane_0']['gpu_uuid']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                main.validate_allocation(value)

    def test_unprepared_or_no_authorization_cannot_probe_or_score(self):
        with patch.object(main, 'discover_allocation') as gpu, patch.object(main, 'build_references') as refs:
            with self.assertRaisesRegex(ValueError, '--execute'):
                main.run_matrix(self.root)
            with patch.object(main.platform, 'system', return_value='Linux'), \
                 patch.object(main, 'verify_preparation', side_effect=ValueError('missing freeze')):
                with self.assertRaisesRegex(ValueError, 'missing freeze'):
                    main.run_matrix(self.root, execute=True)
            gpu.assert_not_called()
            refs.assert_not_called()

    def test_bad_fourth_gpu_stops_before_any_reference_or_guard(self):
        with self.gates(), patch.object(main, 'discover_allocation', side_effect=ValueError('bad fourth GPU')), \
             patch.object(main, 'build_references') as refs:
            with self.assertRaisesRegex(ValueError, 'bad fourth GPU'):
                main.run_matrix(self.root, execute=True)
            refs.assert_not_called()
            self.assertFalse(self.dest.exists())
            self.assertFalse((self.root / main.FAILED_ROOT / 'binding.json').exists())

    def test_changed_gpu_assignment_refuses_before_science(self):
        with self.gates(), patch.object(main, 'build_references', side_effect=RuntimeError('stop before science')):
            with self.assertRaisesRegex(RuntimeError, 'stop before science'):
                main.run_matrix(self.root, execute=True)
        changed = allocation()
        changed['lanes']['lane_0'], changed['lanes']['lane_1'] = changed['lanes']['lane_1'], changed['lanes']['lane_0']
        with self.gates(), patch.object(main, 'discover_allocation', return_value=changed), \
             patch.object(main, 'build_references') as refs:
            with self.assertRaisesRegex(ValueError, 'original four GPU UUIDs'):
                main.run_matrix(self.root, execute=True, resume=True)
            refs.assert_not_called()

    def test_busy_orphan_lane_refuses_before_new_allocation(self):
        from large_qubit.training import _run_lock
        with _run_lock(self.root / main.OPERATIONS / 'lane_2'), self.gates(), \
             patch.object(main, 'discover_allocation') as gpu:
            with self.assertRaisesRegex(ValueError, 'run lock'):
                main.run_matrix(self.root, execute=True)
            gpu.assert_not_called()
            self.assertFalse(self.dest.exists())

    def test_allocation_and_guard_precede_references_and_lane_launch(self):
        events = []
        def refs(root, dest, prepared, attempt, chemistry):
            self.assertEqual(load(dest / 'gpu_allocation.json'), allocation())
            self.assertEqual(load(dest / 'binding.json')['allocation_sha256'], sha256_file(dest / 'gpu_allocation.json'))
            self.assertEqual(load(dest / 'binding.json')['failed_completion_sha256'], main.FAILED_COMPLETION_SHA)
            events.append('references')
        def launch(*args):
            events.append('launch')
            raise RuntimeError('fixture infrastructure failure')
        with self.gates(), patch.object(main, 'build_references', side_effect=refs), \
             patch.object(main, 'launch_lanes', side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, 'fixture infrastructure'):
                main.run_matrix(self.root, execute=True)
        self.assertEqual(events, ['references', 'launch'])
        self.assertFalse((self.dest / 'completed.json').exists())
        failures = list((self.dest / 'execution_attempts').glob('*/failed.json'))
        self.assertEqual(len(failures), 1)
        self.assertFalse(load(failures[0])['automatic_retry'])

    def test_supervisor_launches_four_before_polling_and_keeps_healthy_siblings(self):
        save(self.dest / 'gpu_allocation.json', allocation())
        events, processes = [], []
        class Process:
            def __init__(inner, argv, **kwargs):
                inner.lane = argv[argv.index('--lane') + 1]
                inner.pid, inner.returncode = 100 + len(processes), None
                inner.polls = 0
                events.append('launch:' + inner.lane)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], allocation()['lanes'][inner.lane]['gpu_uuid'])
                self.assertTrue(kwargs['start_new_session'])
                processes.append(inner)
            def poll(inner):
                self.assertEqual(len(processes), 4)
                inner.polls += 1
                events.append('poll:' + inner.lane)
                # One fails immediately; the other three finish later.
                if inner.lane == 'lane_0' or inner.polls > 1:
                    inner.returncode = 2 if inner.lane == 'lane_0' else 0
                return inner.returncode
            def wait(inner):
                return inner.returncode
        attempt = self.dest / 'execution_attempts' / 'attempt_0001'
        with patch.object(main.subprocess, 'Popen', Process), patch.object(main.time, 'sleep'), \
             patch.object(main, '_stop_processes') as stop, \
             patch.object(main, '_signal_group') as cleanup, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'explicitly resume'):
                main.launch_lanes(self.root, self.dest, attempt, allocation(), 'fixture')
            # Cleanup only happens after healthy siblings have completed.
            self.assertTrue(all(p.returncode is not None for p in processes))
            stop.assert_called_once()
            cleanup.assert_called_once_with(100, 9)
        self.assertEqual(events[:4], ['launch:lane_' + str(i) for i in range(4)])
        self.assertEqual(len(list(attempt.glob('workers/*/exit.json'))), 4)

    def test_abort_cleans_child_group_even_when_leader_already_exited(self):
        from unittest.mock import Mock
        dead_leader = Mock(pid=123, returncode=2)
        dead_leader.poll.return_value = 2
        with patch.object(main, '_signal_group') as send:
            main._stop_processes([dead_leader])
        self.assertEqual([call.args for call in send.call_args_list], [(123, main.signal.SIGTERM), (123, 9)])
        dead_leader.wait.assert_called_once()

    def test_supervisor_authenticates_completed_lane_and_skips_it(self):
        save(self.dest / 'gpu_allocation.json', allocation())
        for spec in lane_specs():
            save(self.dest / 'lanes' / spec['lane_id'] / 'completed.json', {'fixture_only': True})
        with patch('large_qubit.section9_recovery_worker.verify_lane', return_value={'all_passed': True}) as verify, \
             patch.object(main.subprocess, 'Popen') as process:
            main.launch_lanes(self.root, self.dest, self.dest / 'attempt', allocation(), 'fixture')
            self.assertEqual(verify.call_count, 4)
            process.assert_not_called()

    def test_readonly_status_does_not_initialize_any_outputs(self):
        before = set(self.root.rglob('*'))
        with patch.object(main, 'discover_allocation') as gpu, patch.object(main, 'build_references') as refs:
            value = main.status(self.root)
            self.assertEqual(value['logical_requests_completed_or_charged'], 0)
            self.assertEqual(sum(len(v['jobs']) for v in value['lanes'].values()), 16)
            gpu.assert_not_called()
            refs.assert_not_called()
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_recovery_requires_original_gpu_order_and_runtime(self):
        main.require_same_devices(self.root, allocation())
        for mode in ('reordered', 'environment'):
            value = allocation()
            if mode == 'reordered':
                value['lanes']['lane_0'], value['lanes']['lane_1'] = value['lanes']['lane_1'], value['lanes']['lane_0']
            else:
                value['lanes']['lane_0']['runtime']['precision'] = 'fp32'
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'original four GPU UUIDs'):
                main.require_same_devices(self.root, value)

    def test_references_reused_without_physics_and_prior_bytes_untouched(self):
        prepared = {'failed_campaign': {'reference_sha256': {}}}
        before = {}
        for alias in main.TARGETS:
            folder = self.root / main.FAILED_ROOT / 'targets' / alias
            save(folder / 'reference.json', {'status': 'completed', 'result': {'fixture': alias}})
            save(folder / 'physics_identity.json', {'cache_identity': {'geometry_id': alias}})
            prepared['failed_campaign']['reference_sha256'][alias] = sha256_file(folder / 'reference.json')
            before[alias] = (folder / 'reference.json').read_bytes()
        with patch('large_qubit.section9_recovery_scoring.PhysicsClient') as client, \
             patch('large_qubit.section9_reporting.verify_reference') as verify:
            main.build_references(self.root, self.dest, prepared, self.dest / 'attempt', 'unused')
            main.build_references(self.root, self.dest, prepared, self.dest / 'attempt', 'unused')
            main.verify_reused_references(self.root, prepared)
            client.assert_not_called()
            self.assertEqual(verify.call_count, 4)
        for alias in main.TARGETS:
            self.assertEqual((self.dest / 'targets' / alias / 'reference.json').read_bytes(), before[alias])
            self.assertEqual((self.root / main.FAILED_ROOT / 'targets' / alias / 'reference.json').read_bytes(), before[alias])
            self.assertEqual(load(self.dest / 'targets' / alias / 'reuse.json')['additional_reference_calls'], 0)
        target = self.root / main.FAILED_ROOT / 'targets' / main.TARGETS[0] / 'reference.json'
        target.write_bytes(target.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            main.build_references(self.root, self.dest, prepared, self.dest / 'attempt', 'unused')
        with self.assertRaisesRegex(ValueError, 'differs from preserved'):
            main.verify_reused_references(self.root, prepared)

    def test_authentication_of_failed_outcomes_returns_nonzero(self):
        failed = {'all_passed': True, 'matrix_execution_complete': True, 'all_trajectories_successful': False}
        with patch.object(main, 'verify_matrix', return_value=failed), redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(['matrix-verify']), 3)
        with patch.object(main, 'verify_matrix', return_value={**failed, 'all_trajectories_successful': True}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(['matrix-verify']), 0)


if __name__ == '__main__':
    unittest.main()



