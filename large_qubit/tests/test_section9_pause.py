"""CPU-only tests of authenticated stopping and unchanged scientific frontiers."""
from dataclasses import replace
from contextlib import nullcontext
import io
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

from large_qubit import section9_pause as subject
from large_qubit.section8 import load, save
from large_qubit.section9_ledger import TrajectoryLedger
from large_qubit.section9_recovery_policy import PREPARATION, lane_specs
from large_qubit.section9_recovery_worker import ledger_identity
from large_qubit.common import sha256_file


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.spec = lane_specs()[0]
        self.lane = self.root / subject.MATRIX / 'lanes' / self.spec['lane_id']
        save(self.lane / 'binding.json', {'fixture': True})
        save(self.root / PREPARATION, {'fixture': True})
        self.cache = {'geometry_id': self.spec['target'], 'partition': 'final_test'}
        save(self.root / subject.MATRIX / 'targets' / self.spec['target'] / 'physics_identity.json',
             {'cache_identity': self.cache})
        self.job = self.lane / self.spec['job_ids'][0]
        self.identity = ledger_identity(sha256_file(self.root / PREPARATION), self.spec['lane_id'],
                                        self.spec['methods'][0], self.spec['target'], self.cache)

    def ledger(self):
        value = TrajectoryLedger(self.job / 'ledger.sqlite', self.identity)
        save(self.job / 'identity.json', self.identity)
        return value

    def boundary(self):
        return subject.frozen_boundary(self.root, self.spec['lane_id'])


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux /proc contract')
class BoundaryTests(Fixture):
    def test_clean_frontier_is_read_only(self):
        ledger = self.ledger()
        before = ledger.snapshot()
        ledger.close()
        observed = self.boundary()
        self.assertTrue(observed['clean'])
        self.assertEqual(observed['jobs'][self.spec['job_ids'][0]]['ledger_head'], before['head_sha256'])
        reopened = self.ledger()
        self.assertEqual(reopened.snapshot(), before)
        reopened.close()

    def test_pending_attempt_is_busy(self):
        ledger = self.ledger()
        ledger.reserve_attempt('attempt_0001', {'seed': 1})
        ledger.close()
        self.assertFalse(self.boundary()['clean'])

    def test_natural_completion_requires_verified_seal_and_no_live_children(self):
        ledger = self.ledger()
        ledger.close()
        save(self.lane / 'completed.json', {'fixture': True})
        worker = subject.Process(111, 1, 100, 111, 'Z', os.getuid(), '', ())
        signals = FakeSignals()
        signals.dead.add(worker.pid)
        verified = {'all_passed': True, 'lane_execution_complete': True,
                    'all_trajectories_successful': True}
        with patch.object(subject, '_processes', return_value={}), \
             patch('large_qubit.section9_recovery_worker.verify_lane', return_value=verified) as verifier:
            result = subject._natural_completion(self.root, self.spec['lane_id'], worker, signals)
        verifier.assert_called_once_with(self.root, self.spec['lane_id'])
        self.assertTrue(result['naturally_completed'])
        self.assertEqual(result['completed_sha256'], sha256_file(self.lane / 'completed.json'))
        child = replace(worker, pid=112, parent=111, state='S')
        with patch.object(subject, '_processes', return_value={112: child}):
            with self.assertRaisesRegex(ValueError, 'live child processes'):
                subject._natural_completion(self.root, self.spec['lane_id'], worker, signals)

    def test_natural_completion_refuses_missing_seal(self):
        worker = subject.Process(111, 1, 100, 111, 'Z', os.getuid(), '', ())
        signals = FakeSignals()
        signals.dead.add(worker.pid)
        with patch.object(subject, '_processes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'without a completion seal'):
                subject._natural_completion(self.root, self.spec['lane_id'], worker, signals)

    def test_pending_energy_is_busy(self):
        ledger = self.ledger()
        ledger.reserve_request('hf', {'kind': 'hf'}, kind='hf')
        ledger.close()
        self.assertFalse(self.boundary()['clean'])

    def test_persistent_wait_drains_real_generation_and_energy_frontiers(self):
        ledger = self.ledger()
        self.addCleanup(ledger.close)
        ledger.reserve_request('hf', {'kind': 'hf'}, kind='hf')
        ledger.complete_request(1, 0.)
        ledger.reserve_attempt('attempt_0001', {'seed': 1})
        ledger.finish_attempt(1, {'valid': True, 'ops': [1]}, valid=True)
        ledger.reserve_request('candidate', {'ops': [1]}, attempt_ordinal=1)
        ledger.reserve_attempt('attempt_0002', {'seed': 2})
        signals, clock, snapshots = FakeSignals(), [0.], []
        coordinator = subject.Process(100, 1, 99, 99, 'S', os.getuid(), str(self.root),
                                      ('python', '-m', 'large_qubit.section9_recovery'))
        worker = replace(coordinator, pid=101, start_ticks=2, parent=100, group=101)
        def freeze(root, process, backend, stopped, deadline):
            backend.send(process, signal.SIGSTOP)
            stopped[(process.pid, process.start_ticks)] = process
            return [process]
        def advance(unused):
            clock[0] += 61
            # The generation result can finish while an earlier energy call
            # remains pending. Neither incomplete frontier may be terminated.
            if clock[0] == 61:
                ledger.finish_attempt(2, {'valid': False, 'raw': 'fixture invalid'}, valid=False)
            else:
                ledger.complete_request(2, -1.)
            snapshots.append(ledger.frontier())
        with patch.object(subject, '_lane_processes', return_value={self.spec['lane_id']: worker}), \
             patch.object(subject, '_freeze_lane', side_effect=freeze), \
             patch.object(subject.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(subject.time, 'sleep', side_effect=advance):
            result = subject._legacy_pause(self.root, coordinator, self.root, self.root / 'receipt',
                                           0, signals=signals)
        self.assertTrue(result['paused'])
        self.assertEqual(clock[0], 122)
        self.assertEqual(snapshots[0]['pending_requests'], 1)
        after = ledger.snapshot()
        self.assertEqual(after['head_sha256'], snapshots[-1]['head_sha256'])
        self.assertEqual((after['logical_requests'], after['completion_attempts']), (2, 2))
        self.assertEqual((after['pending_requests'], after['pending_attempts'],
                          after['failed_requests'], after['abandoned_requests']), (0, 0, 0, 0))

    def test_winner_verification_intent_is_busy_until_outcome(self):
        ledger = self.ledger()
        ledger.close()
        save(self.job / 'winner_verification_started.json', {'utc': 'fixture'})
        self.assertFalse(self.boundary()['clean'])
        save(self.job / 'winner_verification.json', {'status': 'completed'})
        self.assertTrue(self.boundary()['clean'])

    def test_preexisting_failure_is_not_repaired(self):
        ledger = self.ledger()
        ledger.halt('fixture incident')
        ledger.close()
        with self.assertRaisesRegex(ValueError, 'existing failed trajectory'):
            self.boundary()

    def test_durable_identity_mismatch_is_refused(self):
        ledger = self.ledger()
        ledger.close()
        with patch.object(subject, 'load', side_effect=lambda path: {'invalid': True}
                          if Path(path).name == 'identity.json' else load(path)):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                self.boundary()

    def test_sqlite_uncommitted_reservation_is_not_a_clean_boundary(self):
        ledger = self.ledger()
        ledger.close()
        code = "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN IMMEDIATE'); print('locked',flush=True); sys.stdin.readline(); c.rollback()"
        child = subprocess.Popen([sys.executable, '-B', '-c', code, str(self.job / 'ledger.sqlite')],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            # A SELECT can still see the old clean committed frontier. The
            # external pause must notice the writer lock instead.
            self.assertTrue(subject._sqlite_writer_locks(self.lane, {child.pid}))
            self.assertEqual(self.boundary()['reason'], 'SQLite write transaction is in progress')
        finally:
            child.communicate('\n', timeout=5)


class FakeSignals:
    def __init__(self):
        self.events = []
        self.dead = set()
        self.is_stopped = set()
    def pin(self, process):
        self.events.append(('pin', process.pid))
    def send(self, process, number):
        self.events.append((number, process.pid))
        if number == signal.SIGTERM:
            self.dead.add(process.pid)
        if number == signal.SIGSTOP:
            self.is_stopped.add(process.pid)
        if number == signal.SIGCONT:
            self.is_stopped.discard(process.pid)
        return True
    def stopped(self, process):
        return process.pid in self.is_stopped
    def alive(self, process):
        return process.pid not in self.dead
    def close(self):
        self.events.append(('close',))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.receipt = self.root / 'receipt'
        self.coordinator = subject.Process(100, 1, 99, 99, 'S', 1, str(self.root), ('python', '-m', 'large_qubit.section9_recovery'))
        self.worker = subject.Process(101, 2, 100, 101, 'S', 1, str(self.root), ('python', '-m', 'large_qubit.section9_recovery_worker'))

    def test_pause_requires_explicit_execution_before_process_inspection(self):
        with patch.object(subject, 'discover') as discovery:
            with self.assertRaisesRegex(ValueError, 'explicit --execute'):
                subject.pause(self.root)
            discovery.assert_not_called()

    def test_clean_legacy_pause_terminates_only_parents_and_preserves_heads(self):
        signals = FakeSignals()
        boundary = {'clean': True, 'jobs': {'job': {'ledger_head': 'same'}}}
        def freeze(root, worker, backend, stopped, deadline):
            backend.send(worker, signal.SIGSTOP)
            stopped[(worker.pid, worker.start_ticks)] = worker
            return [worker]
        with patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker}), \
             patch.object(subject, '_freeze_lane', side_effect=freeze), \
             patch.object(subject, 'frozen_boundary', return_value=boundary):
            result = subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, 10, signals=signals)
        self.assertTrue(result['paused'])
        term = [e[1] for e in signals.events if e[0] == signal.SIGTERM]
        self.assertEqual(term, [101, 100])
        self.assertEqual(signals.is_stopped, set())
        self.assertTrue(load(self.receipt / 'ready.json')['sqlite_write_transactions_absent'])

    def test_busy_timeout_restores_running_processes_without_termination(self):
        signals = FakeSignals()
        def freeze(root, worker, backend, stopped, deadline):
            backend.send(worker, signal.SIGSTOP)
            stopped[(worker.pid, worker.start_ticks)] = worker
            return [worker]
        with patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker}), \
             patch.object(subject, '_freeze_lane', side_effect=freeze), \
             patch.object(subject, 'frozen_boundary', return_value={'clean': False, 'reason': 'pending request'}):
            with self.assertRaisesRegex(ValueError, 'clean pause boundary'):
                subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, .02, signals=signals)
        self.assertFalse(any(e[0] == signal.SIGTERM for e in signals.events))
        self.assertEqual(signals.is_stopped, set())
        self.assertFalse(load(self.receipt / 'failed.json')['termination_committed'])

    def test_persistent_pause_retains_ready_lane_beyond_old_timeout(self):
        """Continuous work may stay pending longer than the old 60-second probe."""
        signals = FakeSignals()
        later = replace(self.worker, pid=102, group=102)
        clock, observed = [0.], {'lane_0': 0, 'lane_1': 0}
        clean = {'clean': True, 'jobs': {'job': {'ledger_head': 'same'}}}
        def freeze(root, worker, backend, stopped, deadline):
            backend.send(worker, signal.SIGSTOP)
            stopped[(worker.pid, worker.start_ticks)] = worker
            return [worker]
        def boundary(root, lane, processes=()):
            observed[lane] += 1
            if lane == 'lane_1' and observed[lane] == 1:
                return {'clean': False, 'reason': 'admitted generation and scoring still active'}
            return clean
        def advance(unused):
            clock[0] += 61
            self.assertIn(self.worker.pid, signals.is_stopped)
        with patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker, 'lane_1': later}), \
             patch.object(subject, '_freeze_lane', side_effect=freeze), \
             patch.object(subject, 'frozen_boundary', side_effect=boundary), \
             patch.object(subject.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(subject.time, 'sleep', side_effect=advance):
            result = subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, 0, signals=signals)
        self.assertTrue(result['paused'])
        self.assertGreater(clock[0], 60)
        self.assertEqual(signals.events.count((signal.SIGSTOP, self.worker.pid)), 1)
        self.assertEqual(signals.is_stopped, set())

    def test_naturally_exited_lane_does_not_wait_to_stop_a_dead_pid(self):
        signals = FakeSignals()
        signals.dead.add(self.worker.pid)
        clean = {'clean': True, 'jobs': {'job': {'ledger_head': 'same'}}}
        with patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker}), \
             patch.object(subject, '_natural_completion', return_value=dict(clean, naturally_completed=True)) as completed, \
             patch.object(subject, 'frozen_boundary', return_value=clean):
            result = subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, 0, signals=signals)
        completed.assert_called_once_with(self.root, 'lane_0', self.worker, signals)
        self.assertTrue(result['paused'])
        self.assertFalse(any(e == (signal.SIGTERM, self.worker.pid) for e in signals.events))
        self.assertEqual(signals.is_stopped, set())

    def test_exit_race_during_stop_is_reported_without_waiting(self):
        signals = FakeSignals()
        def send(process, number):
            signals.dead.add(process.pid)
            return False
        with patch.object(signals, 'send', side_effect=send):
            with self.assertRaises(subject.ProcessExited):
                subject._stop(signals, self.worker, float('inf'), {})

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux pause interruption signals')
    def test_committed_shutdown_finishes_before_accepting_cancellation(self):
        signals = FakeSignals()
        original_send = signals.send
        boundary = {'clean': True, 'jobs': {}}
        def send(process, number):
            if process.pid == self.worker.pid and number == signal.SIGTERM:
                os.kill(os.getpid(), signal.SIGTERM)
            return original_send(process, number)
        def freeze(root, process, backend, stopped, deadline):
            backend.send(process, signal.SIGSTOP)
            stopped[(process.pid, process.start_ticks)] = process
            return [process]
        with subject._pause_interruptions(), \
             patch.object(signals, 'send', side_effect=send), \
             patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker}), \
             patch.object(subject, '_freeze_lane', side_effect=freeze), \
             patch.object(subject, 'frozen_boundary', return_value=boundary):
            result = subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, 0, signals=signals)
        self.assertTrue(result['paused'])
        self.assertEqual(signals.dead, {self.worker.pid, self.coordinator.pid})

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux pause interruption signals')
    def test_interruptions_roll_back_all_temporary_stops(self):
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(number=number):
                signals = FakeSignals()
                before = signal.getsignal(number)
                def freeze(root, worker, backend, stopped, deadline):
                    backend.send(worker, signal.SIGSTOP)
                    stopped[(worker.pid, worker.start_ticks)] = worker
                    return [worker]
                def interrupted(*unused):
                    os.kill(os.getpid(), number)
                with patch.object(subject, '_lane_processes', return_value={'lane_0': self.worker}), \
                     patch.object(subject, '_freeze_lane', side_effect=freeze), \
                     patch.object(subject, 'frozen_boundary', side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt), subject._pause_interruptions():
                        subject._legacy_pause(self.root, self.coordinator, self.root,
                                              self.receipt / str(number), 0, signals=signals)
                self.assertEqual(signals.is_stopped, set())
                self.assertFalse(any(e[0] == signal.SIGTERM for e in signals.events))
                self.assertEqual(signal.getsignal(number), before)
                self.assertFalse(load(self.receipt / str(number) / 'failed.json')['termination_committed'])

    def test_inspection_error_restores_coordinator_without_termination(self):
        signals = FakeSignals()
        with patch.object(subject, '_lane_processes', side_effect=ValueError('foreign worker')):
            with self.assertRaisesRegex(ValueError, 'foreign worker'):
                subject._legacy_pause(self.root, self.coordinator, self.root, self.receipt, 10, signals=signals)
        self.assertFalse(any(e[0] == signal.SIGTERM for e in signals.events))
        self.assertEqual(signals.is_stopped, set())

    def test_foreign_group_member_is_never_signalled(self):
        signals = FakeSignals()
        intruder = replace(self.worker, pid=102, parent=101, argv=('python', '-m', 'other.module'))
        stopped = {}
        with patch.object(subject, '_processes', return_value={101: self.worker, 102: intruder}):
            with self.assertRaisesRegex(ValueError, 'unauthenticated process'):
                subject._freeze_lane(self.root, self.worker, signals, stopped, time.monotonic() + 5)
        self.assertFalse(any(len(e) > 1 and e[1] == 102 for e in signals.events))

    def test_preexisting_stop_is_not_taken_over(self):
        signals = FakeSignals()
        signals.is_stopped.add(self.worker.pid)
        stopped = {}
        with self.assertRaisesRegex(ValueError, 'already stopped elsewhere'):
            subject._stop(signals, self.worker, time.monotonic() + 5, stopped)
        self.assertEqual(stopped, {})
        self.assertFalse(any(e[0] == signal.SIGCONT for e in signals.events))

    def test_cooperative_timeout_leaves_durable_request_without_signals(self):
        signals = FakeSignals()
        with patch.object(subject, 'LinuxSignals', return_value=signals), \
             patch.object(subject, '_processes', return_value={}):
            result = subject._cooperative_pause(self.root, self.coordinator, self.root / 'attempt',
                                                self.receipt, .01)
        self.assertFalse(result['paused'])
        self.assertTrue(result['pause_requested'])
        self.assertEqual(load(self.root / subject.REQUEST)['schema_version'], 'section9-pause-request-v1')
        self.assertFalse(any(isinstance(e[0], int) for e in signals.events))

    def test_cooperative_exit_without_pause_receipt_is_not_success(self):
        signals = FakeSignals()
        signals.dead.add(self.coordinator.pid)
        with patch.object(subject, 'LinuxSignals', return_value=signals), \
             patch.object(subject, '_processes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'without a clean pause'):
                subject._cooperative_pause(self.root, self.coordinator, self.root / 'attempt', self.receipt, .01)

    def test_cooperative_pause_requires_saved_controller_outcome(self):
        signals = FakeSignals()
        signals.dead.add(self.coordinator.pid)
        attempt = self.root / 'attempt'
        save(attempt / 'paused.json', {'paused': True})
        with patch.object(subject, 'LinuxSignals', return_value=signals), \
             patch.object(subject, '_processes', return_value={}):
            result = subject._cooperative_pause(self.root, self.coordinator, attempt, self.receipt, .01)
        self.assertTrue(result['paused'])
        self.assertFalse(result['matrix_completed'])


class PausedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.attempt = self.root / subject.MATRIX / 'execution_attempts' / 'attempt_0001'
        save(self.attempt / 'started.json', {'pid': 123, 'utc': 'fixture'})
        self.frontiers = {'fixture-ledger': {'head_sha256': 'unchanged'}}
        self.control = SimpleNamespace(_leases=Mock(side_effect=lambda root: nullcontext()),
                                       _lane_frontiers=Mock(return_value=self.frontiers))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'large_qubit.section9_control': self.control}).start()
        patch.object(subject, '_processes', return_value={}).start()

    def native_receipt(self):
        save(self.attempt / 'paused.json', {'paused': True, 'frontiers': self.frontiers})
        save(self.root / subject.REQUEST, {'schema_version': 'section9-pause-request-v1'})

    def test_native_pause_is_idempotent_after_tmux_exits(self):
        self.native_receipt()
        before = sorted(str(path) for path in self.root.rglob('*'))
        result = subject._already_paused(self.root)
        self.assertTrue(result['paused'])
        self.assertTrue(result['already_paused'])
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
        self.control._leases.assert_called_once_with(self.root)

    def test_status_reports_authenticated_pause_after_helper_exit(self):
        self.native_receipt()
        result = subject.pause_status(self.root)
        self.assertTrue(result['paused'])
        self.assertFalse(result['pause_command_running'])
        self.assertEqual(result['last_state'], 'paused')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux owned-process matching')
    def test_pause_helper_does_not_masquerade_as_a_running_coordinator(self):
        self.native_receipt()
        helper = subject.Process(456, 1, 1, 1, 'S', os.getuid(), str(self.root),
                                 ('python', '-m', 'large_qubit.section9_control', 'pause', '--execute'))
        with patch.object(subject, '_processes', return_value={helper.pid: helper}):
            self.assertTrue(subject._already_paused(self.root)['paused'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux owned-process matching')
    def test_coordinator_options_before_action_still_count_as_running_work(self):
        process = subject.Process(123, 1, 1, 1, 'S', os.getuid(), str(self.root),
                                  ('python', '-m', 'large_qubit.section9_control',
                                   '--root', str(self.root), 'resume', '--execute'))
        self.assertTrue(subject._is_work_process(process, self.root))

    def test_status_does_not_trust_a_reused_pause_helper_pid(self):
        receipt = self.root / subject.CONTROL / 'pauses' / 'pause_fixture'
        save(receipt / 'started.json', {'utc': 'fixture', 'pause_pid': 123, 'pause_start_ticks': 42})
        process = subject.Process(123, 43, 1, 1, 'S', os.getuid() if hasattr(os, 'getuid') else 0,
                                  str(self.root), ('python', '-m', 'large_qubit.section9_control', 'pause'))
        with patch.object(subject, '_process', return_value=process):
            result = subject.pause_status(self.root)
        self.assertFalse(result['paused'])
        self.assertFalse(result['pause_command_running'])
        self.assertEqual(result['last_state'], 'interrupted')

    def test_newer_resumed_attempt_invalidates_older_pause(self):
        self.native_receipt()
        save(self.attempt.with_name('attempt_0002') / 'started.json', {'pid': 456, 'utc': 'later'})
        self.assertIsNone(subject._already_paused(self.root))

    def test_changed_native_frontier_is_refused(self):
        self.native_receipt()
        self.control._lane_frontiers.return_value = {'changed': True}
        with self.assertRaisesRegex(ValueError, 'durable paused frontier changed'):
            subject._already_paused(self.root)

    def test_legacy_receipt_requires_same_latest_attempt_and_frontier(self):
        folder = self.root / subject.CONTROL / 'pauses' / 'pause_fixture'
        save(folder / 'started.json', {'utc': 'fixture', 'execution_attempt': self.attempt.relative_to(self.root).as_posix()})
        saved = {'clean': True, 'jobs': {'job': {'ledger_head': 'same'}}}
        save(folder / 'completed.json', {'paused': True, 'mode': 'authenticated_legacy_boundary',
                                         'lane_boundaries': {'lane_0': saved}})
        with patch.object(subject, 'frozen_boundary', return_value=saved):
            self.assertTrue(subject._already_paused(self.root)['already_paused'])
        with patch.object(subject, 'frozen_boundary', return_value={'clean': True, 'jobs': {}}):
            with self.assertRaisesRegex(ValueError, 'changed after the saved external pause'):
                subject._already_paused(self.root)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux ownership contract')
    def test_live_owned_runner_prevents_already_paused_acknowledgement(self):
        self.native_receipt()
        process = subject.Process(123, 1, 1, 1, 'S', os.getuid(), str(self.root),
                                  ('python', '-m', 'large_qubit.section9_control', 'resume'))
        with patch.object(subject, '_processes', return_value={123: process}):
            self.assertIsNone(subject._already_paused(self.root))
        self.control._leases.assert_not_called()


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('tmux'), 'Local Linux tmux fixture')
class TmuxIntegrationTests(unittest.TestCase):
    def test_real_tmux_coordinator_lanes_and_physics_children_stop_at_clean_frontier(self):
        """Temporary namespaced fake modules perform no model/chemistry work."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            package = root / 'large_qubit'
            package.mkdir()
            (package / '__init__.py').write_text('')
            common = '''import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path.cwd()
def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps({'payload':payload,'sha256':digest}))
def terminate(*unused):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,terminate)
'''
            coordinator = common + '''attempt=root/'large_qubit/main/section9_recovery_v1/matrix_0001/execution_attempts/attempt_0001'
save(attempt/'started.json',{'pid':os.getpid(),'utc':'fixture'})
children=[]
try:
    for n in range(4):
        lane='lane_'+str(n)
        argv=[sys.executable,'-B','-m','large_qubit.section9_recovery_worker','run','--root',str(root),'--lane',lane,'--execute','--parent-pid',str(os.getpid())]
        save(attempt/'workers'/lane/'launch.json',{'argv':argv})
        child=subprocess.Popen(argv,cwd=root,start_new_session=True)
        children.append(child)
        save(attempt/'workers'/lane/'started.json',{'pid':child.pid})
    while not all((root/('ready_lane_'+str(n))).exists() for n in range(4)):
        time.sleep(.01)
    (root/'fixture_ready').write_text('ready')
    while True:
        time.sleep(.1)
except KeyboardInterrupt:
    for child in children:
        child.wait(timeout=5)
'''
            worker = common + '''lane=sys.argv[sys.argv.index('--lane')+1]
child=subprocess.Popen([sys.executable,'-B','-m','large_qubit.section9_recovery_scoring','--worker','--root',str(root)],stdin=subprocess.PIPE,text=True,cwd=root)
(root/('ready_'+lane)).write_text('ready')
try:
    while True:
        time.sleep(.1)
except KeyboardInterrupt:
    child.communicate('stop\\n',timeout=5)
'''
            (package / 'section9_recovery.py').write_text(coordinator)
            (package / 'section9_recovery_worker.py').write_text(worker)
            (package / 'section9_recovery_scoring.py').write_text('import sys; sys.stdin.readline()\n')
            save(root / PREPARATION, {'fixture': True})
            for spec in lane_specs():
                lane = root / subject.MATRIX / 'lanes' / spec['lane_id']
                save(lane / 'binding.json', {'fixture': True})
                cache = {'geometry_id': spec['target'], 'partition': 'final_test'}
                save(root / subject.MATRIX / 'targets' / spec['target'] / 'physics_identity.json', {'cache_identity': cache})
                identity = ledger_identity(sha256_file(root / PREPARATION), spec['lane_id'],
                                           spec['methods'][0], spec['target'], cache)
                job = lane / spec['job_ids'][0]
                ledger = TrajectoryLedger(job / 'ledger.sqlite', identity)
                ledger.close()
                save(job / 'identity.json', identity)
            socket = 's9_pause_test_' + uuid.uuid4().hex
            session = 'co2-s9-pause-fixture'
            command = shlex.join([sys.executable, '-B', '-m', 'large_qubit.section9_recovery', '--root', str(root), '--execute'])
            launcher = root / 'fixture_launcher.sh'
            # Match the real __worker launcher: bash owns the tmux pane and
            # waits for its coordinator child. tmux automatically continues a
            # directly stopped pane leader, so direct exec is a different topology.
            launcher.write_text('#!/bin/bash\n' + command + '\nexit $?\n')
            actual_run = subprocess.run
            try:
                actual_run(['tmux', '-L', socket, 'new-session', '-d', '-s', session, '-c', str(root),
                            shlex.join(['bash', str(launcher)])], check=True)
                subject._until(lambda: (root / 'fixture_ready').exists(), time.monotonic() + 15, 'Synthetic tmux fixture did not initialize')
                def isolated_run(argv, **kwargs):
                    return actual_run(['tmux', '-L', socket] + argv[1:] if argv[0] == 'tmux' else argv, **kwargs)
                with patch.dict(subject.SESSIONS, {session: 'large_qubit.section9_recovery'}, clear=True), \
                     patch.object(subject.subprocess, 'run', side_effect=isolated_run):
                    found, process, attempt = subject.discover(root)
                    self.assertEqual(found, session)
                    result = subject._legacy_pause(root, process, attempt, root / 'pause_receipt', 10)
                self.assertTrue(result['paused'])
                self.assertEqual(set(result['lane_boundaries']), {'lane_0', 'lane_1', 'lane_2', 'lane_3'})
                self.assertFalse(any(p.cwd == str(root) for p in subject._processes().values()))
            finally:
                # Cleanup also covers a deliberately failing fixture without
                # leaving its detached synthetic lane sessions behind.
                backend = subject.LinuxSignals()
                try:
                    owned = [p for p in subject._processes().values() if p.cwd == str(root)]
                    for process in owned:
                        backend.pin(process)
                        backend.send(process, signal.SIGCONT)
                    for process in owned:
                        if process.module in subject.WORKERS:
                            backend.send(process, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while any(backend.alive(p) for p in owned if p.module in subject.WORKERS) and time.monotonic() < deadline:
                        time.sleep(.025)
                    for process in owned:
                        if backend.alive(process):
                            backend.send(process, signal.SIGTERM)
                finally:
                    backend.close()
                actual_run(['tmux', '-L', socket, 'kill-server'], capture_output=True, check=False)


@unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'pidfd_open'), 'Linux pidfd contract')
class LinuxSignalTests(unittest.TestCase):
    def test_concurrent_pause_refuses_before_discovery_or_signals(self):
        from large_qubit.training import _run_lock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            verification = Mock()
            with _run_lock(root / subject.CONTROL / 'pause_command'), \
                 patch.dict(sys.modules, {'large_qubit.section9_control': SimpleNamespace(verify_binding=verification)}), \
                 patch.object(subject, 'discover') as discovery:
                with self.assertRaisesRegex(ValueError, 'holds this run lock'):
                    subject.pause(root, execute=True)
                verification.assert_not_called()
                discovery.assert_not_called()

    def test_authentication_progress_precedes_expensive_verification(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder).resolve(), io.StringIO()
            def verify(unused):
                self.assertIn('authenticating', output.getvalue())
                raise ValueError('fixture verification stop')
            with redirect_stdout(output), \
                 patch.dict(sys.modules, {'large_qubit.section9_control': SimpleNamespace(verify_binding=verify)}):
                with self.assertRaisesRegex(ValueError, 'fixture verification stop'):
                    subject.pause(root, execute=True, timeout_seconds=0)

    def test_real_pidfd_stop_continue_and_exact_parent_termination(self):
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
        signals = subject.LinuxSignals()
        try:
            process = subject._process(child.pid)
            subject._stop(signals, process, time.monotonic() + 5)
            self.assertTrue(signals.stopped(process))
            signals.send(process, signal.SIGCONT)
            signals.send(process, signal.SIGTERM)
            child.wait(timeout=5)
            self.assertFalse(signals.alive(process))
        finally:
            signals.close()
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_pid_reuse_identity_change_is_rejected_before_signal(self):
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
        signals = subject.LinuxSignals()
        try:
            process = subject._process(child.pid)
            with patch.object(subject, '_process', return_value=replace(process, start_ticks=process.start_ticks + 1)):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    signals.pin(process)
            self.assertIsNone(child.poll())
        finally:
            signals.close()
            child.terminate()
            child.wait(timeout=5)

    def test_pinned_dead_pid_cannot_redirect_signal_to_new_identity(self):
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
        signals = subject.LinuxSignals()
        try:
            process = subject._process(child.pid)
            signals.pin(process)
            child.terminate()
            child.wait(timeout=5)
            # Even if /proc later exposes a replacement process under the PID,
            # the existing pidfd refers solely to the exited incarnation.
            with patch.object(subject, '_process', return_value=replace(process, start_ticks=process.start_ticks + 1)):
                self.assertFalse(signals.alive(process))
                self.assertFalse(signals.send(process, signal.SIGTERM))
        finally:
            signals.close()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
