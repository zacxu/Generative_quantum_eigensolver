"""Pause and reassign devices within the existing Section9 recovery campaign."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import time

from .common import require, sha256_file, workspace_path
from .section8 import save, load, utc, _new_attempt
from . import section9_recovery as original
from .section9_recovery_policy import (PREPARATION, OUTPUT_ROOT, MATRIX, TARGETS,
    policy, lane_specs, jobs, job_id)
from .section9_control_registration import REGISTRATION

directory = original.directory
validate_allocation = original.validate_allocation
verify_reused_references = original.verify_reused_references
_inventory = original._inventory
_stop_processes = original._stop_processes
_signal_group = original._signal_group
OPERATIONS = original.OPERATIONS
PAUSE_FLAG = OPERATIONS + '/control/pause_requested.json'


def registered_control_sha(root):
    return sha256_file(root / REGISTRATION)


def _lane_frontiers(root):
    """Only inspect a quiescent matrix while all coordinator/lane leases are held."""
    from .section9_ledger import verify_ledger
    from .section9_recovery_worker import ledger_identity
    dest = directory(root)
    rows = {}
    for spec in lane_specs():
        lane = dest / 'lanes' / spec['lane_id']
        require((lane / 'binding.json').exists() and (lane / 'gpu_runtime.json').exists(),
                'Wait until all four original lanes have initialized before reassignment')
        physical = load(dest / 'targets' / spec['target'] / 'physics_identity.json')
        for method in spec['methods']:
            folder = lane / job_id(method, spec['target'])
            if not (folder / 'identity.json').exists():
                require(not (folder / 'ledger.sqlite').exists(), 'Unbound ledger requires inspection')
                continue
            expected = ledger_identity(sha256_file(root / PREPARATION), spec['lane_id'], method,
                                       spec['target'], physical['cache_identity'])
            require(load(folder / 'identity.json') == expected, 'Origin trajectory identity changed')
            snap = verify_ledger(folder / 'ledger.sqlite', expected)
            require(snap['pending_requests'] == 0 and snap['pending_attempts'] == 0,
                    'Pause has not drained all requests and generation attempts; preserve uncertain work')
            for intent in folder.glob('*_started.json'):
                require(intent.with_name(intent.name.replace('_started.json', '.json')).exists(),
                        'Uncertain nonrepeatable verification prevents reassignment')
            path = (folder / 'ledger.sqlite').relative_to(root).as_posix()
            rows[path] = {key: snap[key] for key in ('head_sha256', 'events_count',
                'identity_sha256', 'logical_requests', 'completion_attempts')}
        completed = lane / 'completed.json'
        if completed.exists():
            rows[completed.relative_to(root).as_posix()] = {'completed_sha256': sha256_file(completed)}
    return rows


def _verify_frontier_prefixes(root, rows):
    from .section9_ledger import verify_ledger
    for name, expected in rows.items():
        path = workspace_path(root, name, namespace=OUTPUT_ROOT)
        if 'completed_sha256' in expected:
            require(sha256_file(path) == expected['completed_sha256'], 'Completed lane changed after GPU reassignment')
            continue
        identity = load(path.parent / 'identity.json')
        snap = verify_ledger(path, identity)
        require(snap['identity_sha256'] == expected['identity_sha256']
                and snap['events_count'] >= expected['events_count']
                and snap['logical_requests'] >= expected['logical_requests']
                and snap['completion_attempts'] >= expected['completion_attempts'],
                'GPU reassignment cannot reset a ledger or its counters')
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            row = connection.execute('SELECT sha256 FROM events WHERE ordinal=?',
                                     (expected['events_count'],)).fetchone()
        finally:
            connection.close()
        require(row == (expected['head_sha256'],), 'Paused ledger prefix changed after reassignment')


def compatible_allocation(value, baseline):
    validate_allocation(value)
    validate_allocation(baseline)
    for spec in lane_specs():
        key = spec['lane_id']
        actual, expected = value['lanes'][key]['runtime'], baseline['lanes'][key]['runtime']
        require({k:v for k,v in actual.items() if k != 'gpu_uuid'} ==
                {k:v for k,v in expected.items() if k != 'gpu_uuid'},
                'Only physical H200 UUIDs may change; every numerical runtime field must match')
    return value


def allocation_history(root):
    dest = directory(root)
    baseline = validate_allocation(load(dest / 'gpu_allocation.json'))
    history = [(sha256_file(dest / 'gpu_allocation.json'), baseline)]
    paths = sorted((dest / 'gpu_transitions').glob('epoch_*.json'))
    for number, path in enumerate(paths, 1):
        require(path.name == f'epoch_{number:04d}.json', 'GPU allocation history has a gap')
        receipt = load(path)
        require(receipt['schema_version'] == 'section9-allocation-epoch-v1'
                and receipt['epoch'] == number
                and receipt['previous_epoch_sha256'] == history[-1][0]
                and receipt['registration_sha256'] == registered_control_sha(root)
                and receipt['origin_binding_sha256'] == sha256_file(dest / 'binding.json')
                and receipt['previous_allocation'] == history[-1][1],
                'GPU allocation transition identity or history changed')
        compatible_allocation(receipt['allocation'], baseline)
        _verify_frontier_prefixes(root, receipt['paused_frontiers'])
        history.append((sha256_file(path), receipt['allocation']))
    known = {sha for sha, _ in history}
    for path in dest.glob('**/execution_attempts/attempt_*/control_binding.json'):
        binding = load(path)
        require(binding['allocation_epoch_sha256'] in known,
                'An execution attempt references a missing GPU allocation epoch')
        if path.parent.parent == dest / 'execution_attempts':
            require(set(binding) == {'registration_sha256', 'origin_binding_sha256',
                                     'allocation_epoch_sha256', 'allocation'}
                    and binding['registration_sha256'] == registered_control_sha(root)
                    and binding['origin_binding_sha256'] == sha256_file(dest / 'binding.json')
                    and binding['allocation'] == dict(history)[binding['allocation_epoch_sha256']],
                    'Coordinator execution binding differs from its authenticated allocation epoch')
    return history


def verify_binding(root):
    from .section9_control_registration import verify
    verify(root)
    prepared, baseline = original.verify_binding(root)
    return prepared, allocation_history(root)[-1][1]


def execution_binding(root, lane_id):
    history = allocation_history(root)
    return {'registration_sha256': registered_control_sha(root),
            'origin_binding_sha256': sha256_file(directory(root) / 'binding.json'),
            'allocation_epoch_sha256': history[-1][0],
            'allocation': history[-1][1]['lanes'][lane_id]}


def _leases(root, *, include_lanes=True):
    from .training import _run_lock
    leases = ExitStack()
    try:
        for name in ('section9_v1', 'section9_parallel_v1', 'section9_recovery_v1'):
            leases.enter_context(_run_lock(root / 'large_qubit/operations' / name / 'matrix'))
        if include_lanes:
            for spec in lane_specs():
                leases.enter_context(_run_lock(root / OPERATIONS / spec['lane_id']))
        return leases
    except BaseException:
        leases.close()
        raise


def reassign(workspace, *, execute=False):
    require(execute is True, 'GPU reassignment requires --execute')
    require(platform.system() == 'Linux', 'Reassign GPUs in the pinned Linux server environment')
    root = Path(workspace).resolve(strict=True)
    with _leases(root):
        prepared, previous = verify_binding(root)
        dest = directory(root)
        require(not (dest / 'completed.json').exists(), 'Completed matrices cannot be reassigned or rerun')
        frontiers = _lane_frontiers(root)
        history = allocation_history(root)
        proposed = compatible_allocation(original.discover_allocation(root), history[0][1])
        require(proposed != previous, 'Selected allocation is unchanged; use resume on these devices')
        number = len(history)
        receipt = {'schema_version': 'section9-allocation-epoch-v1', 'epoch': number, 'utc': utc(),
            'registration_sha256': registered_control_sha(root),
            'origin_binding_sha256': sha256_file(dest / 'binding.json'),
            'previous_epoch_sha256': history[-1][0], 'previous_allocation': previous,
            'allocation': proposed, 'paused_frontiers': frontiers,
            'scientific_requests_added': 0, 'model_weights_loaded': False,
            'reason': 'researcher requested less contended allocated H200 devices'}
        path = dest / 'gpu_transitions' / f'epoch_{number:04d}.json'
        save(path, receipt)
        return {'all_passed': True, 'gpu_reassignment_recorded': True,
                'epoch': number, 'epoch_sha256': sha256_file(path),
                'gpu_uuids': [proposed['lanes'][s['lane_id']]['gpu_uuid'] for s in lane_specs()],
                'scientific_requests_added': 0, 'model_weights_loaded': False}


def launch_lanes(root, dest, attempt, allocation, chemistry_python):
    from .section9_control_worker import verify_lane
    processes, handles, exits = {}, [], {}
    try:
        for spec in lane_specs():
            lane_id = spec['lane_id']
            lane = dest / 'lanes' / lane_id
            if (lane / 'completed.json').exists():
                verify_lane(root, lane_id)
                continue
            folder = attempt / 'workers' / lane_id
            argv = [sys.executable, '-u', '-B', '-m', 'large_qubit.section9_control_worker',
                'run', '--root', str(root), '--lane', lane_id, '--execute',
                '--parent-pid', str(os.getpid()), '--chemistry-python', str(chemistry_python)]
            if lane.exists():
                argv.append('--resume')
            save(folder / 'launch.json', {'utc': utc(), 'lane_id': lane_id, 'argv': argv,
                'gpu_uuid': allocation['lanes'][lane_id]['gpu_uuid'],
                'allocation_sha256': sha256_file(dest / 'gpu_allocation.json')})
            handle = (folder / 'worker.log').open('xb')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=allocation['lanes'][lane_id]['gpu_uuid'])
            process = subprocess.Popen(argv, cwd=root, env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            processes[lane_id] = process
            save(folder / 'started.json', {'utc': utc(), 'pid': process.pid})
        while len(exits) < len(processes):
            for lane_id, process in processes.items():
                code = process.poll()
                if code is not None and lane_id not in exits:
                    exits[lane_id] = code
                    if code != 0:
                        # SIGKILL/OOM may bypass worker cleanup. Its remaining
                        # children must not outlive the lease and inflate the
                        # registered four-physics-worker bound on resume.
                        _signal_group(process.pid, getattr(signal, 'SIGKILL', 9))
                    save(attempt / 'workers' / lane_id / 'exit.json', {'utc': utc(), 'exit_code': code})
                    print(json.dumps({'lane': lane_id, 'exit_code': code}), flush=True)
            if len(exits) < len(processes):
                time.sleep(.25)
        require(all(code == 0 for code in exits.values()),
                'One or more lanes stopped: ' + json.dumps(exits) + '; inspect logs, then explicitly resume')
        return exits
    except BaseException:
        _stop_processes(list(processes.values()))
        for lane_id, process in processes.items():
            path = attempt / 'workers' / lane_id / 'exit.json'
            if not path.exists():
                save(path, {'utc': utc(), 'exit_code': process.returncode, 'coordinator_interrupted': True})
        raise
    finally:
        for handle in handles:
            handle.close()



def status(workspace):
    root = Path(workspace).resolve(strict=True)
    dest = directory(root)
    from .section9_ledger import verify_ledger
    result = {'implementation_prepared': (root / REGISTRATION).exists(),
        'preparation_exists': (root / PREPARATION).exists(), 'matrix_execution_complete': False,
        'scope': 'read_only_no_model_loading_or_energy_requests', 'lanes': {},
        'logical_requests_completed_or_charged': 0, 'maximum_logical_requests': 16384,
        'preserved_failed_attempt_requests': 16, 'cumulative_maximum_logical_requests': 16400}
    for spec in lane_specs():
        lane = dest / 'lanes' / spec['lane_id']
        rows = {}
        for name in spec['job_ids']:
            folder = lane / name
            if not (folder / 'identity.json').exists():
                rows[name] = {'state': 'not_run'}
                continue
            snap = verify_ledger(folder / 'ledger.sqlite', load(folder / 'identity.json'))
            rows[name] = {'state': load(folder / 'report.json')['status'] if (folder / 'report.json').exists() else 'partial',
                **{k: snap[k] for k in ('logical_requests', 'completion_attempts', 'completed_requests',
                                       'failed_requests', 'abandoned_requests', 'pending_requests')}}
            result['logical_requests_completed_or_charged'] += snap['logical_requests']
        result['lanes'][spec['lane_id']] = {'jobs': rows, 'completed_receipt_exists': (lane / 'completed.json').exists()}
    result['cumulative_logical_requests_completed_or_charged'] = 16 + result['logical_requests_completed_or_charged']
    if (dest / 'completed.json').exists():
        result.update(verify_matrix(root))
    result['pause_requested'] = (root / PAUSE_FLAG).exists()
    attempts = sorted((dest / 'execution_attempts').glob('attempt_*'))
    result['paused_receipt_exists'] = bool(attempts and (attempts[-1] / 'paused.json').exists())
    if (dest / 'gpu_allocation.json').exists():
        history = allocation_history(root)
        result['allocation_epoch'] = len(history) - 1
        result['effective_gpu_uuids'] = [history[-1][1]['lanes'][s['lane_id']]['gpu_uuid'] for s in lane_specs()]
    return result


def collect_lanes(root):
    from .section9_control_worker import verify_lane
    lanes = {s['lane_id']: verify_lane(root, s['lane_id']) for s in lane_specs()}
    reports, resources = {}, {}
    for lane in lanes.values():
        require(not (set(reports) & set(lane['reports'])), 'Trajectory appears in multiple lanes')
        reports.update(lane['reports'])
        resources.update(lane['resources'])
    require(set(reports) == {job_id(m, a) for m, a in jobs()}, 'Exactly 16 original trajectories required')
    return lanes, reports, resources


def retained_costs(root, dest, lanes):
    value = original.retained_costs(root, dest, lanes)
    value['gpu_control'] = {
        'registration_sha256': registered_control_sha(root),
        'allocation_epochs': [{'sha256': sha, 'allocation': allocation}
                              for sha, allocation in allocation_history(root)],
        'pause_requests': [{
            'path': p.relative_to(root).as_posix(), 'request': load(p)}
            for p in sorted((dest / 'execution_attempts').glob('*/resumed_pause_request.json'))],
        'scientific_budgets_refilled': False,
        'reload_costs': 'retained in per-lane execution attempts; do not add twice'}
    return value


def verify_matrix(workspace):
    root = Path(workspace).resolve(strict=True)
    dest = directory(root)
    completed = load(dest / 'completed.json')
    if completed.get('schema_version') != 'section9-control-completion-v1':
        require(not (dest / 'gpu_transitions').exists(), 'Original completion cannot contain GPU transitions')
        return original.verify_matrix(root)
    prepared, _ = verify_binding(root)
    verify_reused_references(root, prepared)
    require(completed['inventory'] == _inventory(root, dest)
            and completed['binding_sha256'] == sha256_file(dest / 'binding.json')
            and completed['control_registration_sha256'] == registered_control_sha(root)
            and completed['allocation_epoch_sha256'] == allocation_history(root)[-1][0],
            'Controlled matrix inventory or allocation history changed')
    lanes, reports, resources = collect_lanes(root)
    from .section9_reporting import aggregate_reports
    require(load(dest / 'matrix_report.json') == aggregate_reports(reports, resources=resources),
            'Controlled matrix aggregate changed')
    path = workspace_path(root, completed['costs_path'], namespace=OUTPUT_ROOT)
    require(load(path) == retained_costs(root, dest, lanes), 'Controlled cumulative costs changed')
    successful = all(r['status'] == 'complete' for r in reports.values())
    require(completed['matrix_execution_complete'] is True and completed['trajectories_accounted'] == 16
            and completed['all_trajectories_successful'] == successful
            and completed['lane_completion_sha256'] == {k:v['completed_sha256'] for k,v in lanes.items()}
            and completed['pipeline_seeds'] == [22] and completed['training_executed'] is False,
            'Controlled completion flags changed')
    return {'all_passed': True, 'matrix_execution_complete': True, 'trajectories_accounted': 16,
            'all_trajectories_successful': successful, 'completed_sha256': sha256_file(dest / 'completed.json'),
            'ordinary_request_accounting': load(path)['ordinary_request_accounting']}


def resume(workspace, *, execute=False, chemistry_python=None):
    require(execute is True, 'Resuming evaluation requires --execute')
    require(platform.system() == 'Linux', 'Use the pinned Linux server environments')
    root = Path(workspace).resolve(strict=True)
    from .training import _run_lock
    chemistry_python = Path(chemistry_python or root / '.venv-chemistry/bin/python')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    with _leases(root, include_lanes=False), original.interruption_signals():
        prepared, allocation = verify_binding(root)
        dest = directory(root)
        if (dest / 'completed.json').exists():
            return verify_matrix(root)
        with ExitStack() as lane_leases:
            for spec in lane_specs():
                lane_leases.enter_context(_run_lock(root / OPERATIONS / spec['lane_id']))
            _lane_frontiers(root)
            actual = original.discover_allocation(root)
            require(actual == allocation,
                    'Selected GPUs differ from the current recorded allocation; use reassign while paused first')
            attempt = _new_attempt(dest)
            save(attempt / 'control_binding.json', {
                'registration_sha256': registered_control_sha(root),
                'allocation_epoch_sha256': allocation_history(root)[-1][0],
                'allocation': allocation, 'origin_binding_sha256': sha256_file(dest / 'binding.json')})
            flag = root / PAUSE_FLAG
            if flag.exists():
                save(attempt / 'resumed_pause_request.json', load(flag))
                flag.unlink()  # Operational flag only; the immutable request receipt is retained.
        started = time.monotonic()
        try:
            verify_reused_references(root, prepared)
            launch_lanes(root, dest, attempt, allocation, chemistry_python)
            unfinished = [s for s in lane_specs()
                          if not (dest / 'lanes' / s['lane_id'] / 'completed.json').exists()]
            if unfinished:
                require((root / PAUSE_FLAG).exists(), 'An unfinished lane returned without a pause request')
                for spec in unfinished:
                    lane_attempts = sorted((dest / 'lanes' / spec['lane_id'] / 'execution_attempts').glob('attempt_*'))
                    require(lane_attempts and load(lane_attempts[-1] / 'paused.json')['lane_paused'] is True,
                            'An unfinished lane did not record a clean pause')
                with ExitStack() as lane_leases:
                    for spec in lane_specs():
                        lane_leases.enter_context(_run_lock(root / OPERATIONS / spec['lane_id']))
                    frontiers = _lane_frontiers(root)
                receipt = {'utc': utc(), 'paused': True, 'matrix_execution_complete': False,
                           'frontiers': frontiers, 'wall_seconds': time.monotonic() - started}
                save(attempt / 'paused.json', receipt)
                save(attempt / 'finished.json', receipt)
                return {'all_passed': True, 'paused': True, 'matrix_execution_complete': False,
                        'scope': 'all admitted work drained; resume preserves ledgers and budgets'}
            lanes, reports, resources = collect_lanes(root)
            from .section9_reporting import aggregate_reports
            save(dest / 'matrix_report.json', aggregate_reports(reports, resources=resources))
            successful = all(r['status'] == 'complete' for r in reports.values())
            save(attempt / 'finished.json', {'utc': utc(), 'wall_seconds': time.monotonic() - started,
                'accounting_complete': True, 'all_trajectories_successful': successful})
            path = attempt / 'costs.json'
            save(path, retained_costs(root, dest, lanes))
            save(dest / 'completed.json', {'schema_version': 'section9-control-completion-v1',
                'matrix_execution_complete': True, 'trajectories_accounted': 16,
                'all_trajectories_successful': successful,
                'binding_sha256': sha256_file(dest / 'binding.json'),
                'control_registration_sha256': registered_control_sha(root),
                'allocation_epoch_sha256': allocation_history(root)[-1][0],
                'lane_completion_sha256': {k:v['completed_sha256'] for k,v in lanes.items()},
                'costs_path': path.relative_to(root).as_posix(), 'inventory': _inventory(root, dest),
                'training_executed': False, 'pipeline_seeds': [22], 'performance_selection': False})
            return verify_matrix(root)
        except BaseException as error:
            if not (dest / 'completed.json').exists():
                save(attempt / 'failed.json', {'utc': utc(), 'error': str(error),
                    'wall_seconds': time.monotonic() - started, 'automatic_retry': False})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'register', 'status', 'pause', 'reassign', 'resume', 'matrix-verify'))
    parser.add_argument('--root', default='.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--chemistry-python')
    parser.add_argument('--timeout-seconds', type=float, default=60)
    args = parser.parse_args(argv)
    try:
        if args.action in ('verify', 'register'):
            from . import section9_control_registration
            result = getattr(section9_control_registration, args.action)(args.root)
        elif args.action == 'pause':
            from .section9_pause import pause
            result = pause(args.root, execute=args.execute, timeout_seconds=args.timeout_seconds)
        elif args.action == 'reassign':
            result = reassign(args.root, execute=args.execute)
        elif args.action == 'resume':
            result = resume(args.root, execute=args.execute, chemistry_python=args.chemistry_python)
        elif args.action == 'status':
            result = status(args.root)
        else:
            result = verify_matrix(args.root)
        print(json.dumps(result, indent=2))
        return 3 if result.get('matrix_execution_complete') and result.get('all_trajectories_successful') is False else 0
    except Exception as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

