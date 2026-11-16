"""Explicitly launched four-GPU main matrix; imports and status execute no science."""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from .common import require, sha256_file, workspace_path
from .section8 import save, load, utc, controller_runtime, _new_attempt
from .section8_policy import FINAL_FREEZE, CATALOGUE
from .section9 import overhead_once, runtime_matches_validation, training_preflight_overhead
from .section9_parallel_policy import (REGISTRATION, PREPARATION, OUTPUT_ROOT, MATRIX,
    TARGETS, policy, lane_specs, jobs, job_id)

LEGACY_DIRECTORY = 'large_qubit/main/section9_v1/matrix_0001'
LEGACY_MARKER = 'parallel_executor_selected.json'
OPERATIONS = 'large_qubit/operations/section9_parallel_v1'


def directory(root):
    return workspace_path(root, OUTPUT_ROOT + '/' + MATRIX, namespace=OUTPUT_ROOT)


def choice(root):
    return {'executor': 'section9_parallel_v1', 'matrix': MATRIX,
            'registration_sha256': sha256_file(root / REGISTRATION),
            'scientific_requests_in_this_guard': 0}


def assert_no_legacy(root):
    """The original immutable entrypoint refuses an existing unbound directory."""
    old = workspace_path(root, LEGACY_DIRECTORY, namespace='large_qubit/main')
    if old.exists():
        require(old.is_dir() and {p.name for p in old.iterdir()} == {LEGACY_MARKER}
                and load(old / LEGACY_MARKER) == choice(root),
                'Original Section9 matrix already exists; preserve it and do not start a second campaign')


def reserve_campaign(root):
    # Caller holds the SAME lock as the original executor. This durable guard
    # remains after completion or parent death; both old run and resume refuse.
    assert_no_legacy(root)
    save(root / LEGACY_DIRECTORY / LEGACY_MARKER, choice(root))


def _preparation_value(root):
    from .section9_parallel_registration import verify
    from .section8 import verify_final_freeze
    from .section8_policy import PREPARATION as VALIDATION_PREPARATION
    registered = verify(root)
    frozen_check = verify_final_freeze(root)
    frozen = load(root / FINAL_FREEZE)
    require(frozen['protocol']['main'] == policy()['scientific_protocol'], 'Frozen main protocol differs')
    assert_no_legacy(root)
    return {'schema_version': 'section9-four-gpu-preparation-v1', 'policy': policy(),
        'registration_sha256': registered['registration_sha256'],
        'final_freeze_sha256': frozen_check['freeze_sha256'],
        'training': frozen['final_adapters_and_training'], 'contexts': frozen['prompt_contexts'],
        'release_sha256': load(root / VALIDATION_PREPARATION)['release_sha256'],
        'catalogue_sha256': sha256_file(root / CATALOGUE),
        'scientific_energy_requests': 0, 'model_weights_loaded': False, 'training_executed': False}


def prepare(workspace, *, execute=False):
    require(execute is True, 'Parallel preparation requires explicit --execute')
    root = Path(workspace).resolve(strict=True)
    value = _preparation_value(root)
    save(workspace_path(root, PREPARATION, namespace='large_qubit/preflights'), value)
    return {'all_passed': True, 'preparation_sha256': sha256_file(root / PREPARATION),
            'scientific_energy_requests': 0, 'model_weights_loaded': False}


def verify_preparation(workspace):
    root = Path(workspace).resolve(strict=True)
    value = load(root / PREPARATION)
    require(value == _preparation_value(root), 'Parallel preparation or prior evidence changed')
    return value


def validate_allocation(value):
    from .gpu_identity import normalize_gpu_uuid
    require(set(value) == {'schema_version', 'lanes'}
            and value['schema_version'] == 'section9-four-gpu-allocation-v1'
            and set(value['lanes']) == {s['lane_id'] for s in lane_specs()}, 'Four GPU lane assignments required')
    identities = []
    for spec in lane_specs():
        row = value['lanes'][spec['lane_id']]
        require(set(row) == {'gpu_uuid', 'runtime'}, 'Unexpected GPU allocation fields')
        uuid = normalize_gpu_uuid(row['gpu_uuid'])
        require(row['gpu_uuid'] == uuid and row['runtime']['gpu_uuid'] == uuid
                and 'H200' in row['runtime'].get('gpu_name', ''), 'Each lane requires an identifiable H200')
        identities.append(uuid)
    require(len(set(identities)) == 4, 'Four distinct physical GPU UUIDs are required')
    return value


def gpu_probe(workspace, *, execute=False):
    """One visible device, no model loading and no target array access."""
    require(execute is True, 'GPU probe requires --execute')
    root = Path(workspace).resolve(strict=True)
    verify_preparation(root)
    import torch
    from .section8_inference import _runtime_identity
    from .section8_policy import OUTPUT_ROOT as VALIDATION_ROOT
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Probe needs one visible H200')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = _runtime_identity(torch, 0)
    panel = root / VALIDATION_ROOT / load(root / FINAL_FREEZE)['selected_panel']
    for model in ('gpt2_xl', 'qwen3_4b'):
        runtime_matches_validation(actual, load(panel / (model + '_runtime.json')))
    return actual


def discover_allocation(root):
    """Resolve indices through CUDA itself, then check each isolated runtime."""
    from .gpu_identity import normalize_gpu_uuid
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4,
            'Expose exactly four allocated H200s with CUDA_VISIBLE_DEVICES')
    uuids = []
    for i in range(4):
        properties = torch.cuda.get_device_properties(i)
        require('H200' in properties.name, 'Every assigned GPU must be an H200')
        uuids.append(normalize_gpu_uuid(getattr(properties, 'uuid', None)))
    require(len(set(uuids)) == 4, 'Four distinct GPU UUIDs required')
    rows = {}
    # Probe before references or model loads, so a bad fourth device consumes no science.
    for spec, uuid in zip(lane_specs(), uuids):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        result = subprocess.run([sys.executable, '-B', '-m', 'large_qubit.section9_parallel',
            'gpu-probe', '--execute', '--root', str(root)], cwd=root, env=env,
            capture_output=True, text=True, check=False)
        require(result.returncode == 0, 'GPU probe failed: ' + result.stderr.strip())
        runtime = json.loads(result.stdout)
        require(runtime['gpu_uuid'] == uuid, 'Isolated probe resolved a different GPU')
        rows[spec['lane_id']] = {'gpu_uuid': uuid, 'runtime': runtime}
    return validate_allocation({'schema_version': 'section9-four-gpu-allocation-v1', 'lanes': rows})


def verify_binding(root):
    prepared = verify_preparation(root)
    dest = directory(root)
    allocation = validate_allocation(load(dest / 'gpu_allocation.json'))
    binding = load(dest / 'binding.json')
    require(binding['preparation_sha256'] == sha256_file(root / PREPARATION)
            and binding['policy'] == policy()
            and binding['final_freeze_sha256'] == prepared['final_freeze_sha256']
            and binding['allocation_sha256'] == sha256_file(dest / 'gpu_allocation.json')
            and load(root / LEGACY_DIRECTORY / LEGACY_MARKER) == choice(root),
            'Parallel campaign binding or allocation changed')
    return prepared, allocation


def build_references(root, dest, prepared, attempt, chemistry_python):
    from .section9_scoring import PhysicsClient, ScoringError
    from .section9_reporting import verify_reference
    for alias in TARGETS:
        folder = dest / 'targets' / alias
        if (folder / 'reference.json').exists():
            receipt = load(folder / 'reference.json')
            if receipt['status'] == 'completed':
                verify_reference(receipt['result'], load(folder / 'physics_identity.json')['cache_identity'])
            continue
        # A saved intent but no outcome must become unknown failure without a
        # second reference solve, including when setup/model work was interrupted.
        if (folder / 'reference_started.json').exists():
            overhead_once(folder, 'reference', lambda: (_ for _ in ()).throw(AssertionError('no retry')))
            continue
        marker = attempt / 'reference_setup' / alias
        save(marker / 'started.json', {'utc': utc(), 'geometry_id': alias})
        started = time.monotonic()
        client = PhysicsClient(root, chemistry_python, alias, execute=True, threads=1,
                              expected_freeze_sha256=prepared['final_freeze_sha256'])
        try:
            save(marker / 'finished.json', {'utc': utc(), 'wall_seconds': time.monotonic() - started})
            save(folder / 'physics_identity.json', client.identity)
            def compute():
                value = client.reference()
                try:
                    verify_reference(value, client.identity['cache_identity'])
                except ValueError as error:
                    raise ScoringError('Reference integrity gate failed: ' + str(error), report=value) from error
                return value
            overhead_once(folder, 'reference', compute)
        finally:
            client.close()


@contextmanager
def interruption_signals():
    def interrupted(number, frame):
        raise KeyboardInterrupt('Coordinator received signal ' + str(number))
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_processes(processes):
    """Signal whole lane groups, including chemistry subprocesses; never retry."""
    for process in processes:
        _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(.1)
    for process in processes:
        # A dead leader can leave a live chemistry child in its session group.
        _signal_group(process.pid, getattr(signal, 'SIGKILL', 9))
    for process in processes:
        process.wait()


def _signal_group(pid, number):
    try:
        os.killpg(pid, number)
    except ProcessLookupError:
        pass


def launch_lanes(root, dest, attempt, allocation, chemistry_python):
    from .section9_parallel_worker import verify_lane
    processes, handles, exits = {}, [], {}
    try:
        for spec in lane_specs():
            lane_id = spec['lane_id']
            lane = dest / 'lanes' / lane_id
            if (lane / 'completed.json').exists():
                verify_lane(root, lane_id)
                continue
            folder = attempt / 'workers' / lane_id
            argv = [sys.executable, '-u', '-B', '-m', 'large_qubit.section9_parallel_worker',
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


def _inventory(root, dest):
    result = {}
    for p in sorted(dest.rglob('*')):
        if not p.is_file() or p == dest / 'completed.json' or p.name.endswith(('-wal', '-shm')):
            continue
        require(not p.is_symlink(), 'Parallel inventory cannot contain symbolic links')
        result[p.relative_to(root).as_posix()] = sha256_file(p)
    return result


def retained_costs(root, dest, lanes):
    frozen = load(root / FINAL_FREEZE)
    attempts = []
    for p in sorted((dest / 'execution_attempts').glob('attempt_*')):
        row = {'path': p.relative_to(root).as_posix(),
               'started': load(p / 'started.json') if (p / 'started.json').exists() else None,
               'status': 'interrupted', 'reference_setup': {}, 'workers': {}}
        for outcome in ('finished', 'failed'):
            if (p / (outcome + '.json')).exists():
                row.update(status=outcome, outcome=load(p / (outcome + '.json')))
        for alias in TARGETS:
            setup = p / 'reference_setup' / alias
            if (setup / 'started.json').exists():
                row['reference_setup'][alias] = {name: load(setup / (name + '.json'))
                    if (setup / (name + '.json')).exists() else None for name in ('started', 'finished')}
        for lane in lane_specs():
            folder = p / 'workers' / lane['lane_id']
            if folder.exists():
                row['workers'][lane['lane_id']] = {name: load(folder / (name + '.json'))
                    if (folder / (name + '.json')).exists() else None for name in ('launch', 'started', 'exit')}
        attempts.append(row)
    return {'source_acquisition_charged_once': frozen['costs']['source_acquisition_charged_once'],
        'three_training_fits_charged_once': frozen['final_adapters_and_training'],
        'training_preflight_overhead_charged_once': training_preflight_overhead(root, frozen),
        'all_validation_attempts': {k: v for k, v in frozen['costs'].items()
                                  if k not in ('source_acquisition_charged_once', 'training_costs')},
        'target_references_charged_once': {alias: load(dest / 'targets' / alias / 'reference.json') for alias in TARGETS},
        'lanes': {k: v['costs'] for k, v in lanes.items()}, 'coordinator_attempts': attempts,
        'overlap': 'Four concurrent lanes, each overlapping generation and scoring; component times are not elapsed wall time',
        'unknown_interrupted_work_retained': True, 'lane_local_caches_may_increase_physical_work': True}


def collect_lanes(root):
    from .section9_parallel_worker import verify_lane
    lanes = {s['lane_id']: verify_lane(root, s['lane_id']) for s in lane_specs()}
    reports, resources = {}, {}
    for lane in lanes.values():
        require(not (set(reports) & set(lane['reports'])), 'Trajectory appears in multiple lanes')
        reports.update(lane['reports'])
        resources.update(lane['resources'])
    require(set(reports) == {job_id(m, a) for m, a in jobs()}, 'Exactly 16 fixed trajectories required')
    return lanes, reports, resources


def verify_matrix(workspace):
    root = Path(workspace).resolve(strict=True)
    verify_binding(root)
    dest = directory(root)
    completed = load(dest / 'completed.json')
    require(completed['inventory'] == _inventory(root, dest)
            and completed['binding_sha256'] == sha256_file(dest / 'binding.json'), 'Parallel matrix inventory changed')
    lanes, reports, resources = collect_lanes(root)
    from .section9_reporting import aggregate_reports
    require(load(dest / 'matrix_report.json') == aggregate_reports(reports, resources=resources), 'Parallel aggregate changed')
    costs_path = workspace_path(root, completed['costs_path'], namespace=OUTPUT_ROOT)
    require(load(costs_path) == retained_costs(root, dest, lanes), 'Parallel cumulative costs changed')
    successful = all(r['status'] == 'complete' for r in reports.values())
    require(completed['matrix_execution_complete'] is True and completed['trajectories_accounted'] == 16
            and completed['all_trajectories_successful'] == successful
            and completed['lane_completion_sha256'] == {k: v['completed_sha256'] for k, v in lanes.items()}
            and completed['training_executed'] is False and completed['pipeline_seeds'] == [22],
            'Parallel completion flags changed')
    return {'all_passed': True, 'matrix_execution_complete': True, 'trajectories_accounted': 16,
            'all_trajectories_successful': successful, 'completed_sha256': sha256_file(dest / 'completed.json'),
            'scope': 'read_only_all_outcomes_including_failures; one seed, two targets'}


def run_matrix(workspace, *, chemistry_python=None, execute=False, resume=False):
    require(execute is True, 'Parallel main matrix requires explicit --execute')
    require(platform.system() == 'Linux', 'Run in the pinned Linux server environments')
    root = Path(workspace).resolve(strict=True)
    prepared = verify_preparation(root)  # No CUDA or target access before this gate.
    dest = directory(root)
    if (dest / 'completed.json').exists():
        return verify_matrix(root)
    from .training import _run_lock
    chemistry_python = Path(chemistry_python or root / '.venv-chemistry/bin/python')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    with _run_lock(root / 'large_qubit/operations/section9_v1/matrix'), \
         _run_lock(root / OPERATIONS / 'matrix'), interruption_signals():
        # Orphan children retain their lane leases until cleanup. Refuse a new
        # coordinator before any new receipts, references or GPU allocations.
        with ExitStack() as leases:
            for spec in lane_specs():
                leases.enter_context(_run_lock(root / OPERATIONS / spec['lane_id']))
        existed = dest.exists()
        require(not existed or resume, 'Existing parallel matrix requires --resume')
        if existed:
            require((dest / 'gpu_allocation.json').exists(), 'Interrupted initialization lacks allocation; preserve and inspect')
        assert_no_legacy(root)
        allocation = discover_allocation(root)
        save(dest / 'gpu_allocation.json', allocation)
        reserve_campaign(root)
        save(dest / 'binding.json', {'preparation_sha256': sha256_file(root / PREPARATION),
            'policy': policy(), 'final_freeze_sha256': prepared['final_freeze_sha256'],
            'allocation_sha256': sha256_file(dest / 'gpu_allocation.json'), 'controller_runtime': controller_runtime()})
        attempt = _new_attempt(dest)
        started = time.monotonic()
        try:
            build_references(root, dest, prepared, attempt, chemistry_python)
            launch_lanes(root, dest, attempt, allocation, chemistry_python)
            lanes, reports, resources = collect_lanes(root)
            from .section9_reporting import aggregate_reports
            save(dest / 'matrix_report.json', aggregate_reports(reports, resources=resources))
            save(attempt / 'finished.json', {'utc': utc(), 'wall_seconds': time.monotonic() - started, 'success': True})
            cost_path = attempt / 'costs.json'
            save(cost_path, retained_costs(root, dest, lanes))
            save(dest / 'completed.json', {'schema_version': 'section9-four-gpu-completion-v1',
                'matrix_execution_complete': True, 'trajectories_accounted': 16,
                'all_trajectories_successful': all(r['status'] == 'complete' for r in reports.values()),
                'binding_sha256': sha256_file(dest / 'binding.json'),
                'lane_completion_sha256': {k: v['completed_sha256'] for k, v in lanes.items()},
                'costs_path': cost_path.relative_to(root).as_posix(), 'inventory': _inventory(root, dest),
                'training_executed': False, 'pipeline_seeds': [22], 'performance_selection': False})
            return verify_matrix(root)
        except BaseException as error:
            if not (dest / 'completed.json').exists():
                save(attempt / 'failed.json', {'utc': utc(), 'error': str(error),
                    'wall_seconds': time.monotonic() - started, 'automatic_retry': False})
            raise


def status(workspace):
    root = Path(workspace).resolve(strict=True)
    dest = directory(root)
    from .section9_ledger import verify_ledger
    result = {'implementation_prepared': (root / REGISTRATION).exists(),
        'preparation_exists': (root / PREPARATION).exists(), 'matrix_execution_complete': False,
        'scope': 'read_only_no_model_loading_or_energy_requests', 'lanes': {},
        'logical_requests_completed_or_charged': 0, 'maximum_logical_requests': 16384}
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
    if (dest / 'completed.json').exists():
        result.update(verify_matrix(root))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'register', 'verify', 'prepare', 'run', 'status', 'matrix-verify', 'gpu-probe'))
    parser.add_argument('--root', default='.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--chemistry-python')
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            result = policy()
        elif args.action in ('register', 'verify'):
            from . import section9_parallel_registration
            result = getattr(section9_parallel_registration, args.action)(args.root)
        elif args.action == 'prepare':
            result = prepare(args.root, execute=args.execute)
        elif args.action == 'run':
            result = run_matrix(args.root, execute=args.execute, resume=args.resume, chemistry_python=args.chemistry_python)
        elif args.action == 'status':
            result = status(args.root)
        elif args.action == 'gpu-probe':
            result = gpu_probe(args.root, execute=args.execute)
        else:
            result = verify_matrix(args.root)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
