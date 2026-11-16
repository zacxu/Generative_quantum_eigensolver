"""CPU fixtures for four-device admission, concurrent supervision and recovery."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_parallel as main
from large_qubit.common import sha256_file
from large_qubit.section8 import save, load
from large_qubit.section9_parallel_policy import REGISTRATION, PREPARATION, lane_specs, policy


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

    def gates(self):
        stack = ExitStack()
        stack.enter_context(patch.object(main.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(main, 'verify_preparation', return_value=self.prepared))
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

    def test_legacy_evidence_is_never_overwritten(self):
        old = self.root / main.LEGACY_DIRECTORY
        old.mkdir(parents=True)
        (old / 'binding.json').write_text('legacy evidence')
        with self.assertRaisesRegex(ValueError, 'second campaign'):
            main.reserve_campaign(self.root)
        self.assertEqual((old / 'binding.json').read_text(), 'legacy evidence')

    def test_durable_guard_rejects_original_run_and_resume_initialization(self):
        main.reserve_campaign(self.root)
        main.assert_no_legacy(self.root)
        old = self.root / main.LEGACY_DIRECTORY
        self.assertTrue(old.exists())
        self.assertFalse((old / 'binding.json').exists())
        # Original executor requires !exists for run and binding.json for resume.
        self.assertEqual(set(p.name for p in old.iterdir()), {main.LEGACY_MARKER})
        original = (old / main.LEGACY_MARKER).read_bytes()
        main.reserve_campaign(self.root)
        self.assertEqual(original, (old / main.LEGACY_MARKER).read_bytes())

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
            self.assertFalse((self.root / main.LEGACY_DIRECTORY).exists())

    def test_changed_gpu_assignment_refuses_before_science(self):
        with self.gates(), patch.object(main, 'build_references', side_effect=RuntimeError('stop before science')):
            with self.assertRaisesRegex(RuntimeError, 'stop before science'):
                main.run_matrix(self.root, execute=True)
        changed = allocation()
        changed['lanes']['lane_0'], changed['lanes']['lane_1'] = changed['lanes']['lane_1'], changed['lanes']['lane_0']
        with self.gates(), patch.object(main, 'discover_allocation', return_value=changed), \
             patch.object(main, 'build_references') as refs:
            with self.assertRaisesRegex(ValueError, 'Preserve existing artifact'):
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

    def test_references_skip_completed_and_never_retry_uncertain_work(self):
        for alias in main.TARGETS:
            folder = self.dest / 'targets' / alias
            save(folder / 'reference_started.json', {'utc': 'fixture'})
            save(folder / 'physics_identity.json', {'cache_identity': {'fixture': alias}})
        save(self.dest / 'targets' / main.TARGETS[0] / 'reference.json',
             {'status': 'completed', 'result': {'fixture': True}})
        with patch('large_qubit.section9_scoring.PhysicsClient') as physics, \
             patch('large_qubit.section9_reporting.verify_reference') as verify:
            main.build_references(self.root, self.dest, self.prepared, self.dest / 'attempt', 'fixture')
            physics.assert_not_called()
            verify.assert_called_once()
        result = load(self.dest / 'targets' / main.TARGETS[1] / 'reference.json')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['uncertain_cost'])

    def test_allocation_and_guard_precede_references_and_lane_launch(self):
        events = []
        def refs(root, dest, prepared, attempt, chemistry):
            self.assertEqual(load(dest / 'gpu_allocation.json'), allocation())
            self.assertEqual(load(dest / 'binding.json')['allocation_sha256'], sha256_file(dest / 'gpu_allocation.json'))
            self.assertTrue((root / main.LEGACY_DIRECTORY / main.LEGACY_MARKER).exists())
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
        with patch('large_qubit.section9_parallel_worker.verify_lane', return_value={'all_passed': True}) as verify, \
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


if __name__ == '__main__':
    unittest.main()
