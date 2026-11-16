"""Explicit, researcher-controlled pause without interrupting scientific requests.

The amended runner cooperates through a durable request. The original runner
has no pause hook, so it is stopped only at an authenticated, transaction-free
boundary. No signal is sent to an unverified PID or to an entire process group.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from .common import require, sha256_file
from .section8 import load, save, utc
from .section9_recovery_policy import PREPARATION, lane_specs

CONTROL = 'large_qubit/operations/section9_recovery_v1/control'
MATRIX = 'large_qubit/main/section9_recovery_v1/matrix_0001'
REQUEST = CONTROL + '/pause_requested.json'
SESSIONS = {'co2-s9-recovery-matrix_0001': 'large_qubit.section9_recovery',
            'co2-s9-control-matrix_0001': 'large_qubit.section9_control'}
WORKERS = {'large_qubit.section9_recovery_worker', 'large_qubit.section9_control_worker'}
SCORERS = {'large_qubit.section9_recovery_scoring'}
PAUSE_SIGNALS = tuple(getattr(signal, name) for name in ('SIGINT', 'SIGTERM', 'SIGHUP') if hasattr(signal, name))


class ProcessExited(ValueError):
    """An owned process exited while the pause was locating its boundary."""


@contextmanager
def _pause_interruptions():
    """Turn ordinary termination into rollback, including detached-job HUP.

    Ignore a second interruption while the first is unwinding temporary stops.
    SIGKILL cannot be handled and must not be used to cancel this operation.
    """
    numbers = PAUSE_SIGNALS
    previous = {number: signal.getsignal(number) for number in numbers}
    def interrupted(number, unused):
        for controlled in numbers:
            signal.signal(controlled, signal.SIG_IGN)
        raise KeyboardInterrupt('Pause command interrupted by signal ' + str(number))
    try:
        for number in numbers:
            signal.signal(number, interrupted)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@dataclass(frozen=True)
class Process:
    pid: int
    start_ticks: int
    parent: int
    group: int
    state: str
    uid: int
    cwd: str
    argv: tuple

    @property
    def module(self):
        return self.argv[self.argv.index('-m') + 1] if '-m' in self.argv and self.argv.index('-m') + 1 < len(self.argv) else None


def _process(pid):
    """Read identity before and after other /proc fields to reject PID reuse."""
    path = Path('/proc') / str(pid)
    def stat():
        value = (path / 'stat').read_text()
        return value[value.rfind(')') + 2:].split()
    first = stat()
    state = first[0]
    if state == 'Z':
        return Process(pid, int(first[19]), int(first[1]), int(first[2]), state,
                       path.stat().st_uid, '', ())
    argv = tuple(x.decode('utf-8', errors='strict') for x in (path / 'cmdline').read_bytes().split(b'\0') if x)
    cwd, uid = str((path / 'cwd').resolve(strict=True)), path.stat().st_uid
    second = stat()
    require(first[19] == second[19], 'Process identity changed while inspecting /proc')
    return Process(pid, int(first[19]), int(second[1]), int(second[2]), second[0], uid, cwd, argv)


def _processes():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            process = _process(int(path.name))
            if process.uid == os.getuid():
                result[process.pid] = process
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def _argument(process, name):
    values = [process.argv[i + 1] for i, value in enumerate(process.argv[:-1]) if value == name]
    require(len(values) <= 1, 'Repeated process argument: ' + name)
    return values[0] if values else None


def _root_matches(process, root):
    explicit = _argument(process, '--root')
    return process.uid == os.getuid() and process.cwd == str(root) and (
        explicit is None or (Path(process.cwd) / explicit).resolve() == root)


def _is_work_process(process, root):
    if process.module not in set(SESSIONS.values()) | WORKERS | SCORERS or not _root_matches(process, root):
        return False
    if process.module in WORKERS | SCORERS:
        return True
    if process.module in SESSIONS.values():
        arguments = process.argv[process.argv.index('-m') + 2:]
        return any(action in arguments for action in ('run', 'resume'))
    return False


def _descends(process, ancestor, processes):
    seen = set()
    while process.parent in processes and process.parent not in seen:
        if process.parent == ancestor:
            return True
        seen.add(process.parent)
        process = processes[process.parent]
    return False


def discover(root):
    """Resolve a unique exact tmux session and its project-owned coordinator."""
    processes = _processes()
    found = []
    for session, module in SESSIONS.items():
        response = subprocess.run(['tmux', 'list-panes', '-t', '=' + session, '-F', '#{pane_pid}'],
                                  capture_output=True, text=True, check=False)
        if response.returncode:
            continue
        panes = {int(value) for value in response.stdout.split() if value.isdecimal()}
        require(len(panes) == 1, 'Expected exactly one pane in the Section 9 session')
        matches = [p for p in processes.values() if p.module == module and _root_matches(p, root)
                   and any(_descends(p, pane, processes) or p.pid == pane for pane in panes)]
        require(len(matches) == 1, 'Cannot identify the exact live Section 9 coordinator')
        require(module != 'large_qubit.section9_recovery' or matches[0].pid not in panes,
                'Legacy pause requires the documented bash launcher; tmux automatically continues a stopped pane leader')
        found.append((session, matches[0]))
    require(len(found) == 1, 'Expected exactly one running Section 9 recovery/control tmux session')
    session, coordinator = found[0]
    require(coordinator.state not in ('T', 't', 'Z'), 'Coordinator was already stopped or exited')
    attempts = []
    for path in (root / MATRIX / 'execution_attempts').glob('attempt_*/started.json'):
        if load(path).get('pid') == coordinator.pid:
            attempts.append(path.parent)
    require(len(attempts) == 1, 'Coordinator PID does not match exactly one authenticated execution receipt')
    return session, coordinator, attempts[0]


class LinuxSignals:
    """pidfds keep every signal attached to the inspected process incarnation."""
    def __init__(self):
        require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
                'Safe pause requires Linux pidfd_open and pidfd_send_signal')
        self.handles = {}

    def pin(self, process):
        key = (process.pid, process.start_ticks)
        if key not in self.handles:
            fd = os.pidfd_open(process.pid)
            try:
                current = _process(process.pid)
                require(current.start_ticks == process.start_ticks and current.uid == process.uid
                        and current.argv == process.argv and current.cwd == process.cwd,
                        'Process identity changed before pinning its pidfd')
            except BaseException:
                os.close(fd)
                raise
            self.handles[key] = fd

    def send(self, process, number):
        self.pin(process)
        try:
            signal.pidfd_send_signal(self.handles[(process.pid, process.start_ticks)], number)
            return True
        except ProcessLookupError:
            return False

    def alive(self, process):
        try:
            current = _process(process.pid)
            return current.start_ticks == process.start_ticks and current.state != 'Z'
        except (FileNotFoundError, ProcessLookupError):
            return False

    def stopped(self, process):
        if not self.alive(process):
            return False
        states = []
        for path in (Path('/proc') / str(process.pid) / 'task').iterdir():
            try:
                value = (path / 'stat').read_text()
                states.append(value[value.rfind(')') + 2:].split()[0])
            except FileNotFoundError:
                continue
        return bool(states) and all(state in ('T', 't') for state in states)

    def close(self):
        for fd in self.handles.values():
            os.close(fd)
        self.handles.clear()


def _until(predicate, deadline, message):
    while not predicate():
        require(time.monotonic() < deadline, message)
        time.sleep(.025)


def _stop(signals, process, deadline, tracked=None):
    if not signals.alive(process):
        raise ProcessExited('Owned process exited before it could be stopped')
    require(process.state not in ('T', 't', 'Z'), 'Refusing to take over a process already stopped elsewhere')
    require(not signals.stopped(process), 'Refusing to take over a process already stopped elsewhere')
    if tracked is not None:
        tracked[(process.pid, process.start_ticks)] = process
    signals.send(process, signal.SIGSTOP)
    _until(lambda: signals.stopped(process) or not signals.alive(process),
           min(deadline, time.monotonic() + 30), 'Timed out stopping owned process threads')
    if not signals.alive(process):
        raise ProcessExited('Owned process exited while it was being stopped')


def _sqlite_writer_locks(folder, pids):
    """A clean snapshot alone does not exclude an uncommitted reservation.

    With every process thread stopped, no SQLite write lock means no commit can
    become visible when a pending termination signal is delivered. Inspect
    SQLite sidecar inodes as WAL write locks reside in the shared-memory file.
    """
    identities = set()
    for path in folder.rglob('*'):
        if path.is_file() and '.sqlite' in path.name:
            stat = path.stat()
            identities.add((os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino))
    writers = []
    for line in Path('/proc/locks').read_text().splitlines():
        parts = line.replace(' -> ', ' ').split()
        if len(parts) < 8 or parts[3] != 'WRITE':
            continue
        device = parts[5].split(':')
        require(len(device) == 3, 'Cannot interpret Linux file-lock identity')
        identity = (int(device[0], 16), int(device[1], 16), int(device[2]))
        if identity in identities:
            # Unknown or foreign writers also make the boundary unsafe.
            writers.append({'pid': int(parts[4]), 'inode': list(identity), 'kind': parts[1]})
    return writers


def frozen_boundary(root, lane_id, processes=()):
    """Read authenticated durable state while all lane threads are stopped."""
    from .section9_ledger import verify_ledger
    from .section9_recovery_worker import ledger_identity
    spec = next(s for s in lane_specs() if s['lane_id'] == lane_id)
    destination = root / MATRIX / 'lanes' / lane_id
    if not (destination / 'binding.json').exists():
        return {'clean': False, 'reason': 'lane initialization is still publishing its binding'}
    if _sqlite_writer_locks(destination, {p.pid for p in processes}):
        return {'clean': False, 'reason': 'SQLite write transaction is in progress'}
    physical = load(root / MATRIX / 'targets' / spec['target'] / 'physics_identity.json')
    prepared_sha = sha256_file(root / PREPARATION)
    heads = {}
    for method, name in zip(spec['methods'], spec['job_ids']):
        folder = destination / name
        ledger = folder / 'ledger.sqlite'
        if not ledger.exists():
            require(not (folder / 'identity.json').exists(), 'Job identity exists without its ledger')
            continue
        expected = ledger_identity(prepared_sha, lane_id, method, spec['target'], physical['cache_identity'])
        if (folder / 'identity.json').exists():
            require(load(folder / 'identity.json') == expected, 'Lane job identity differs')
        # Frozen processes cannot release a lock. Refuse quickly instead of
        # waiting for the journal verifier's ordinary 60-second SQLite timeout.
        try:
            connection = sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=.05)
            try:
                connection.execute('SELECT MAX(ordinal) FROM events').fetchone()
            finally:
                connection.close()
        except sqlite3.OperationalError as error:
            if 'locked' in str(error) or 'no such table' in str(error):
                return {'clean': False, 'reason': 'SQLite initialization or transaction is in progress'}
            raise
        snapshot = verify_ledger(ledger, expected)
        require(not snapshot['halted'] and not snapshot['failed_requests'] and not snapshot['abandoned_requests'],
                'An existing failed trajectory needs diagnosis; pause cannot repair it')
        if snapshot['pending_requests'] or snapshot['pending_attempts']:
            return {'clean': False, 'reason': 'admitted generation or energy requests are still active'}
        if (folder / 'winner_verification_started.json').exists() and not (folder / 'winner_verification.json').exists():
            return {'clean': False, 'reason': 'winner integrity verification is still active'}
        if (folder / 'winner_verification.json').exists():
            require(load(folder / 'winner_verification.json')['status'] == 'completed',
                    'Winner verification already failed; pause cannot repair it')
        if (folder / 'report.json').exists():
            require(load(folder / 'report.json')['ledger_head'] == snapshot['head_sha256'],
                    'Terminal report does not match its ledger')
        heads[name] = {'ledger_head': snapshot['head_sha256'],
                       'logical_requests': snapshot['logical_requests'],
                       'completion_attempts': snapshot['completion_attempts']}
    return {'clean': True, 'jobs': heads}


def _lane_processes(root, coordinator, attempt):
    current = _processes()
    active = [p for p in current.values() if p.parent == coordinator.pid and p.module in WORKERS]
    require(all(_root_matches(p, root) for p in active), 'Worker root differs from the coordinator')
    result = {}
    for spec in lane_specs():
        lane_id = spec['lane_id']
        matches = [p for p in active if _argument(p, '--lane') == lane_id]
        require(len(matches) <= 1, 'More than one process owns a fixed lane')
        if not matches:
            require((root / MATRIX / 'lanes' / lane_id / 'completed.json').exists(),
                    'Lane is still starting or stopped unexpectedly; retry pause after checking its log')
            continue
        worker = matches[0]
        folder = attempt / 'workers' / lane_id
        require(load(folder / 'started.json')['pid'] == worker.pid
                and tuple(load(folder / 'launch.json')['argv']) == worker.argv,
                'Live worker does not match its authenticated launch receipt')
        require(worker.group == worker.pid and _argument(worker, '--parent-pid') == str(coordinator.pid),
                'Worker session or parent binding differs')
        result[lane_id] = worker
    require(len(active) == len(result), 'Coordinator has an unrecognized lane worker')
    return result


def _freeze_lane(root, worker, signals, stopped, deadline):
    _stop(signals, worker, deadline, stopped)
    members = [p for p in _processes().values() if p.group == worker.group and p.pid != worker.pid]
    for child in members:
        require(child.parent == worker.pid and child.module in SCORERS and _root_matches(child, root)
                and '--worker' in child.argv, 'Lane process group contains an unauthenticated process')
        _stop(signals, child, deadline, stopped)
    actual = {(p.pid, p.start_ticks) for p in _processes().values() if p.group == worker.group}
    expected = {(p.pid, p.start_ticks) for p in [worker] + members}
    require(actual == expected, 'Lane process group membership changed while freezing')
    return [worker] + members


def _continue(signals, members, stopped):
    for process in reversed(members):
        signals.send(process, signal.SIGCONT)
        stopped.pop((process.pid, process.start_ticks), None)


def _natural_completion(root, lane_id, worker, signals):
    """Accept an exited lane only after its seal and child cleanup authenticate."""
    require(not signals.alive(worker), 'Natural completion requires an exited worker')
    require(not any(p.group == worker.group and p.state != 'Z' for p in _processes().values()),
            'Exited lane still has live child processes; inspect its log before pausing')
    destination = root / MATRIX / 'lanes' / lane_id
    require((destination / 'completed.json').exists(),
            'Lane exited without a completion seal; inspect its log before pausing')
    print(json.dumps({'lane': lane_id, 'pause_boundary': 'authenticating_completed_lane'}), flush=True)
    from .section9_recovery_worker import verify_lane
    result = verify_lane(root, lane_id)
    require(result['all_passed'] is True and result['lane_execution_complete'] is True
            and result['all_trajectories_successful'] is True,
            'Naturally completed lane failed authentication or contains failed trajectories')
    boundary = frozen_boundary(root, lane_id)
    require(boundary['clean'], 'Naturally completed lane has an incomplete durable frontier')
    return dict(boundary, naturally_completed=True,
                completed_sha256=sha256_file(destination / 'completed.json'))


def _legacy_pause(root, coordinator, attempt, receipt, timeout_seconds, *, signals=None):
    signals = signals or LinuxSignals()
    stopped, lanes, boundary, members, last_print = {}, {}, {}, {}, {}
    deadline = float('inf') if timeout_seconds == 0 else time.monotonic() + timeout_seconds
    committed, shutdown_handlers = False, None
    try:
        # Pin before the first stop so rollback can signal an exact incarnation.
        signals.pin(coordinator)
        _stop(signals, coordinator, deadline, stopped)
        lanes = _lane_processes(root, coordinator, attempt)
        while len(boundary) < len(lanes):
            for lane_id, worker in lanes.items():
                if lane_id in boundary:
                    continue
                require(time.monotonic() < deadline, 'No clean pause boundary within the requested timeout')
                try:
                    members[lane_id] = _freeze_lane(root, worker, signals, stopped, deadline)
                except (ProcessExited, ProcessLookupError, FileNotFoundError):
                    # A lane may finish between two probes while the parent
                    # coordinator is held. Its zombie PID is not a live worker.
                    require(not signals.alive(worker), 'A chemistry child exited unexpectedly')
                    boundary[lane_id] = _natural_completion(root, lane_id, worker, signals)
                    members[lane_id] = []
                    stopped.pop((worker.pid, worker.start_ticks), None)
                    print(json.dumps({'lane': lane_id, 'pause_boundary': 'completed'}), flush=True)
                    continue
                result = frozen_boundary(root, lane_id, members[lane_id])
                if result['clean']:
                    boundary[lane_id] = result
                    print(json.dumps({'lane': lane_id, 'pause_boundary': 'ready'}), flush=True)
                else:
                    _continue(signals, members[lane_id], stopped)
                    if time.monotonic() - last_print.get(lane_id, float('-inf')) >= 10:
                        print(json.dumps({'lane': lane_id, 'pause_boundary': 'waiting', 'reason': result['reason']}), flush=True)
                        last_print[lane_id] = time.monotonic()
            if len(boundary) < len(lanes):
                time.sleep(min(.2, max(0., deadline - time.monotonic())))
        save(receipt / 'ready.json', {'utc': utc(), 'coordinator_pid': coordinator.pid,
             'lane_boundaries': boundary, 'no_pending_attempts_or_requests': True,
             'sqlite_write_transactions_absent': True})
        # Only now may processes exit. The chemistry subprocess remains alive
        # for the worker's ordinary cleanup; it receives no termination signal.
        # Cancellation is safe before this point. Once termination starts,
        # finish bounded cleanup rather than waking a half-stopped coordinator.
        shutdown_handlers = {number: signal.getsignal(number) for number in PAUSE_SIGNALS}
        for number in PAUSE_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        committed = True
        for lane_id, worker in lanes.items():
            if boundary[lane_id].get('naturally_completed'):
                continue
            signals.send(worker, signal.SIGTERM)
            _continue(signals, members[lane_id], stopped)
        _until(lambda: all(not signals.alive(p) for group in members.values() for p in group),
               time.monotonic() + 60, 'Clean workers are still releasing resources; no force-kill was attempted')
        signals.send(coordinator, signal.SIGTERM)
        _continue(signals, [coordinator], stopped)
        _until(lambda: not signals.alive(coordinator), time.monotonic() + 60,
               'Coordinator cleanup is still running; no force-kill was attempted')
        for lane_id, previous in boundary.items():
            require(frozen_boundary(root, lane_id)['jobs'] == previous['jobs'],
                    'Ledger frontier changed during clean shutdown; inspect before resuming')
        result = {'paused': True, 'pause_requested': True, 'mode': 'authenticated_legacy_boundary',
                  'lane_boundaries': boundary, 'scientific_work_started': False,
                  'receipt_path': receipt.relative_to(root).as_posix()}
        save(receipt / 'completed.json', dict(result, utc=utc()))
        return result
    except BaseException as error:
        # Before commit every process resumes normally. After commit signals
        # already delivered cannot be undone; still release every temporary
        # stop and explicitly retain the incomplete-cleanup diagnostic.
        for process in list(stopped.values())[::-1]:
            try:
                signals.send(process, signal.SIGCONT)
            except (OSError, ValueError):
                pass
        save(receipt / 'failed.json', {'utc': utc(), 'error': str(error),
             'termination_committed': committed, 'temporary_stops_released': True,
             'no_force_kill': True})
        raise
    finally:
        signals.close()
        if shutdown_handlers is not None:
            for number, handler in shutdown_handlers.items():
                signal.signal(number, handler)


def _cooperative_pause(root, coordinator, attempt, receipt, timeout_seconds):
    request = root / REQUEST
    if request.exists():
        value = load(request)
        require(value.get('schema_version') == 'section9-pause-request-v1', 'Unexpected existing pause request')
    else:
        value = {'schema_version': 'section9-pause-request-v1', 'utc': utc(),
                 'requested_by_pid': os.getpid(), 'receipt_path': receipt.relative_to(root).as_posix()}
        save(request, value)
    save(receipt / 'request.json', value)
    signals = LinuxSignals()
    try:
        signals.pin(coordinator)
        deadline = float('inf') if timeout_seconds == 0 else time.monotonic() + timeout_seconds
        while signals.alive(coordinator) and time.monotonic() < deadline:
            time.sleep(.2)
        running = [p for p in _processes().values() if p.module in WORKERS and _root_matches(p, root)]
        exited = not signals.alive(coordinator) and not running
        paused = exited and (attempt / 'paused.json').exists() and load(attempt / 'paused.json').get('paused') is True
        completed = exited and (root / MATRIX / 'completed.json').exists() and load(
            root / MATRIX / 'completed.json').get('all_trajectories_successful') is True
        require(not exited or paused or completed,
                'Coordinator exited without a clean pause or successful matrix receipt; inspect its log')
        result = {'paused': paused, 'pause_requested': True, 'mode': 'cooperative',
                  'matrix_completed': completed,
                  'request_path': REQUEST, 'receipt_path': receipt.relative_to(root).as_posix(),
                  'scientific_work_started': False}
        save(receipt / ('completed.json' if paused or completed else 'waiting.json'), dict(result, utc=utc()))
        return result
    finally:
        signals.close()


def _already_paused(root):
    """Idempotent read-only acknowledgement after the tmux session has exited.

    The newest coordinator attempt must itself be the paused attempt. An older
    pause receipt never authorizes a later resumed or crashed attempt.
    """
    from .section9_control import _leases, _lane_frontiers
    if any(_is_work_process(p, root) for p in _processes().values()):
        return None
    with _leases(root):
        # The leases close the discovery/resume race. Check again after taking
        # all coordinator and lane leases before trusting durable state.
        if any(_is_work_process(p, root) for p in _processes().values()):
            return None
        attempts = sorted((root / MATRIX / 'execution_attempts').glob('attempt_*'))
        if not attempts or (root / MATRIX / 'completed.json').exists():
            return None
        latest = attempts[-1]
        native = latest / 'paused.json'
        if native.exists() and (root / REQUEST).exists():
            request, saved = load(root / REQUEST), load(native)
            require(request.get('schema_version') == 'section9-pause-request-v1'
                    and saved.get('paused') is True and saved['frontiers'] == _lane_frontiers(root),
                    'Saved pause request or durable paused frontier changed')
            return {'paused': True, 'already_paused': True, 'pause_requested': True,
                    'mode': 'cooperative', 'request_path': REQUEST,
                    'receipt_path': native.relative_to(root).as_posix(),
                    'scientific_work_started': False}
        latest_name = latest.relative_to(root).as_posix()
        matching = []
        for completed in (root / CONTROL / 'pauses').glob('pause_*/completed.json'):
            saved = load(completed)
            if saved.get('mode') != 'authenticated_legacy_boundary' or saved.get('paused') is not True:
                continue
            started = load(completed.parent / 'started.json')
            if started.get('execution_attempt') == latest_name:
                matching.append((started['utc'], completed, saved))
        if not matching:
            return None
        _, completed, saved = max(matching, key=lambda value: value[0])
        _lane_frontiers(root)  # Authenticate all initialized jobs, including sealed lanes.
        for lane_id, previous in saved['lane_boundaries'].items():
            current = frozen_boundary(root, lane_id)
            require(current['clean'] and current['jobs'] == previous['jobs'],
                    'Ledger frontier changed after the saved external pause')
        return {'paused': True, 'already_paused': True, 'pause_requested': True,
                'mode': 'authenticated_legacy_boundary',
                'receipt_path': completed.relative_to(root).as_posix(),
                'scientific_work_started': False}


def pause_status(root):
    """Read pause progress without signaling processes or loading models."""
    root = Path(root).resolve(strict=True)
    previous = _already_paused(root)
    if previous is not None:
        return dict(previous, pause_command_running=False, last_state='paused')
    receipts = []
    for path in (root / CONTROL / 'pauses').glob('pause_*/started.json'):
        value = load(path)
        receipts.append((value['utc'], path.parent, value))
    if not receipts:
        return {'paused': False, 'pause_requested': (root / REQUEST).exists(),
                'pause_command_running': False, 'last_state': 'not_requested'}
    _, receipt, started = max(receipts, key=lambda value: value[0])
    running = False
    if type(started.get('pause_pid')) is int:
        try:
            process = _process(started['pause_pid'])
            running = (process.start_ticks == started.get('pause_start_ticks')
                       and process.state != 'Z' and _root_matches(process, root)
                       and process.module in ('large_qubit.section9_control', 'large_qubit.section9_pause')
                       and (process.module == 'large_qubit.section9_pause' or 'pause' in process.argv))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    state = 'waiting' if running else 'interrupted'
    for name in ('waiting', 'ready', 'failed', 'completed'):
        if (receipt / (name + '.json')).exists():
            state = name
    return {'paused': False, 'pause_requested': running or (root / REQUEST).exists(),
            'pause_command_running': running, 'last_state': state,
            'receipt_path': receipt.relative_to(root).as_posix()}


def pause(root, *, execute=False, timeout_seconds=60):
    require(execute is True, 'Pausing the server run requires explicit --execute')
    require(platform.system() == 'Linux', 'Pause must run on the Linux GPU server')
    require(type(timeout_seconds) in (int, float) and 0 <= timeout_seconds <= 3600,
            'Pause timeout must be 0 (wait until paused), or at most 3600 seconds')
    root = Path(root).resolve(strict=True)
    from .training import _run_lock
    from .section9_control import verify_binding
    print(json.dumps({'pause_stage': 'authenticating', 'paused': False,
          'message': 'Checking registered files and saved artifacts before touching processes.'}), flush=True)
    with _pause_interruptions(), _run_lock(root / CONTROL / 'pause_command'):
        verify_binding(root)  # Authentication only: no model loading or target evaluation.
        previous = _already_paused(root)
        if previous is not None:
            return previous
        session, coordinator, attempt = discover(root)
        from .section9_operations_registration import amendment_sha
        own = _process(os.getpid())
        receipt = root / CONTROL / 'pauses' / ('pause_' + uuid.uuid4().hex)
        save(receipt / 'started.json', {'utc': utc(), 'session': session,
             'coordinator_pid': coordinator.pid, 'coordinator_start_ticks': coordinator.start_ticks,
             'coordinator_argv': list(coordinator.argv), 'timeout_seconds': timeout_seconds,
             'matrix_path': MATRIX, 'execution_attempt': attempt.relative_to(root).as_posix(),
             'pause_pid': own.pid, 'pause_start_ticks': own.start_ticks,
             'operations_amendment_sha256': amendment_sha(root)})
        print(json.dumps({'pause_stage': 'waiting_for_safe_boundary', 'paused': False,
              'wait_until_paused': timeout_seconds == 0,
              'receipt_path': receipt.relative_to(root).as_posix()}), flush=True)
        if coordinator.module == 'large_qubit.section9_control':
            return _cooperative_pause(root, coordinator, attempt, receipt, timeout_seconds)
        return _legacy_pause(root, coordinator, attempt, receipt, timeout_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=60)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(pause(args.root, execute=args.execute, timeout_seconds=args.timeout_seconds), indent=2))
        return 0
    except KeyboardInterrupt as error:
        print('Pause interrupted: temporary stops were released; check status before reassigning GPUs. '
              + str(error), file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
