"""Continue an authenticated recovery lane with additive GPU allocation history.

Each lane owns one target evaluator, one resident backbone and one scalar cache.
The coordinator owns target references and shared costs. Imports do no science.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

from .common import digest, read_json, require, sha256_file, workspace_path
from .protocol import METHODS
from .section8 import save, load, utc, _new_attempt
from .section8_policy import CATALOGUE
from .section9 import execute_trajectory, pin_runtime, overhead_once, _inventory, _failure
from .section9_policy import job_id
from .section9_recovery_policy import PREPARATION, OUTPUT_ROOT, MATRIX, policy, lane_spec, lane_specs
from .section9_ledger import TrajectoryLedger as _BaseLedger, OrderedScorer as _BaseScorer


class PauseRequested(BaseException):
    """Cooperative scheduling event, never a failed scientific request."""


def pause_path(root):
    return Path(root) / 'large_qubit/operations/section9_recovery_v1/control/pause_requested.json'


def check_pause(path):
    if path is not None and Path(path).exists():
        raise PauseRequested('Researcher requested a pause before new work admission')


class PausableLedger(_BaseLedger):
    def __init__(self, *args, pause_marker=None, **kwargs):
        self.pause_marker = pause_marker
        super().__init__(*args, **kwargs)

    def reserve_attempt(self, call_id, payload=None):
        if self.lookup_attempt(call_id) is None:
            check_pause(self.pause_marker)
        return super().reserve_attempt(call_id, payload)


class PausableScorer(_BaseScorer):
    def __init__(self, *args, pause_marker=None, **kwargs):
        self.pause_marker = pause_marker
        super().__init__(*args, **kwargs)

    def submit(self, call_id, query, **kwargs):
        if self.ledger.lookup_request(call_id) is None:
            check_pause(self.pause_marker)
        return super().submit(call_id, query, **kwargs)


def _context(workspace, lane_id):
    """Authenticate immutable inputs before GPU, model or target-array access."""
    from .section9_control import verify_binding, directory, verify_reused_references
    root = Path(workspace).resolve(strict=True)
    spec = lane_spec(lane_id)
    prepared, allocation_document = verify_binding(root)
    verify_reused_references(root, prepared, aliases=(spec['target'],))
    destination = directory(root)
    allocation = allocation_document['lanes']
    target = destination / 'targets' / spec['target']
    physical = load(target / 'physics_identity.json')
    identity = physical['cache_identity']
    require(identity['geometry_id'] == spec['target'] and identity['partition'] == 'final_test',
            'Lane target physics identity differs')
    reference = load(target / 'reference.json')
    require(reference['status'] in ('completed', 'failed'), 'Target reference is not terminal')
    if reference['status'] == 'completed':
        from .section9_reporting import verify_reference
        verify_reference(reference['result'], identity)
    return root, destination, spec, prepared, allocation[lane_id], physical, reference


def ledger_identity(preparation_sha, lane_id, method, alias, cache_identity):
    spec = lane_spec(lane_id)
    require(method in spec['methods'] and alias == spec['target'], 'Job is outside its fixed lane')
    require(cache_identity['geometry_id'] == alias and cache_identity['partition'] == 'final_test',
            'Lane ledger requires the exact final-test physics identity')
    name = job_id(method, alias)
    return {'schema_version': 'section9-recovery-trajectory-identity-v1',
        'preparation_sha256': preparation_sha, 'policy_sha256': digest(policy()),
        'method': method, 'geometry_id': alias, 'job_id': name, 'lane_id': lane_id,
        'queue_id': 'section9_recovery_v1/' + lane_id + '/' + name + '/ordered_cpu_queue',
        'cache_scope': 'lane_local', 'cache_identity': cache_identity}


def _binding(root, destination, spec):
    target = destination / 'targets' / spec['target']
    return {'schema_version': 'section9-recovery-lane-binding-v1',
        'preparation_sha256': sha256_file(root / PREPARATION),
        'parent_binding_sha256': sha256_file(destination / 'binding.json'),
        'allocation_sha256': sha256_file(destination / 'gpu_allocation.json'),
        'reference_sha256': sha256_file(target / 'reference.json'),
        'physics_identity_sha256': sha256_file(target / 'physics_identity.json'),
        'lane': spec, 'policy_sha256': digest(policy()), 'cache_scope': 'lane_local'}


def verify_execution_history(root, parent, destination, lane_id):
    """Authenticate actual runtimes without rewriting the original lane receipt."""
    from .section9_control import allocation_history, registered_control_sha
    history = dict(allocation_history(root))
    registered = registered_control_sha(root)
    bound = []
    for attempt in sorted((destination / 'execution_attempts').glob('attempt_*')):
        marker = attempt / 'control_binding.json'
        if not marker.exists():
            require(not (attempt / 'gpu_runtime.json').exists(),
                    'Unbound per-attempt GPU runtime cannot be authenticated')
            continue
        value = load(marker)
        require(set(value) == {'registration_sha256', 'origin_binding_sha256',
                              'allocation_epoch_sha256', 'allocation'},
                'Unexpected GPU control execution binding fields')
        epoch = value['allocation_epoch_sha256']
        require(value['registration_sha256'] == registered
                and value['origin_binding_sha256'] == sha256_file(parent / 'binding.json')
                and epoch in history
                and value['allocation'] == history[epoch]['lanes'][lane_id],
                'Lane execution allocation is outside the authenticated GPU history')
        runtime_path = attempt / 'gpu_runtime.json'
        if runtime_path.exists():
            require(load(runtime_path) == value['allocation']['runtime'],
                    'Lane execution runtime differs from its allocation epoch')
        else:
            require(not (attempt / 'finished.json').exists()
                    and not list(attempt.glob('*_load_started.json'))
                    and not (attempt / 'physics_loads').exists(),
                    'Executed lane attempt lacks its authenticated runtime')
        bound.append((attempt, value))
    return bound


def retained_lane_costs(root, destination, reports):
    """Only lane costs; shared collection, fits and references stay with parent."""
    executions = []
    for attempt in sorted((destination / 'execution_attempts').glob('attempt_*')):
        receipt = {'path': attempt.relative_to(root).as_posix(), 'status': 'interrupted',
                   'started': load(attempt / 'started.json') if (attempt / 'started.json').exists() else None,
                   'inference_sessions': []}
        for name in ('finished.json', 'failed.json'):
            if (attempt / name).exists():
                receipt.update(load(attempt / name), status=name[:-5])
        for marker in sorted(attempt.glob('*_load_started.json')):
            prefix = marker.name[:-len('_load_started.json')]
            done = attempt / (prefix + '_inference_cost.json')
            receipt['inference_sessions'].append({'session': prefix, 'load_intent': load(marker),
                'cost': load(done) if done.exists() else None,
                'unknown_unflushed_model_cost': not done.exists()})
        starts = sorted((attempt / 'optimizer_algebraic_work').rglob('*_fit_started.json'))
        updates = sorted((attempt / 'optimizer_algebraic_work').rglob('*_fit_completed.json'))
        receipt.update(started_algebraic_coordinate_fits_including_replay=len(starts),
            completed_algebraic_coordinate_fits_including_replay=len(updates),
            recorded_algebraic_candidate_preparation_seconds=sum(
                load(p)['algebraic_candidate_preparation_seconds'] for p in updates),
            algebraic_work_allowance_per_vqe_trajectory=170,
            partial_algebraic_work_may_be_unknown=len(starts) != len(updates) or receipt['status'] != 'finished')
        receipt['physics_worker_loads'] = []
        for marker in sorted((attempt / 'physics_loads').glob('*_started.json')):
            done = marker.with_name(marker.name.replace('_started.json', '_completed.json'))
            receipt['physics_worker_loads'].append({'started': load(marker),
                'completed': load(done) if done.exists() else None,
                'unknown_unflushed_setup_cost': not done.exists()})
        executions.append(receipt)
    return {'main_by_trajectory': {name: {k: v for k, v in report['numerical_cost'].items()
                                        if k in ('search', 'generation')} for name, report in reports.items()},
        'winner_integrity_overhead': {name: load(destination / name / 'winner_verification.json')
                                     for name in reports
                                     if (destination / name / 'winner_verification.json').exists()},
        'execution_attempts': executions, 'shared_costs_charged_here': False,
        'totals_are_known_lower_bounds_if_interrupted': True,
        'overlap': 'generation and scoring overlap; component times are not elapsed wall time',
        'inference_accounting': 'session totals crosscheck trajectory generation; do not add them twice'}


def _verify_job(root, destination, spec, method, physical, reference, catalogue):
    from .section9_ledger import verify_ledger
    from .section9_reporting import trajectory_report
    from .section9_resources import resource_report
    alias, lane_id = spec['target'], spec['lane_id']
    folder = destination / job_id(method, alias)
    identity = ledger_identity(sha256_file(root / PREPARATION), lane_id, method, alias, physical['cache_identity'])
    require(load(folder / 'identity.json') == identity, 'Lane trajectory identity differs')
    report = load(folder / 'report.json')
    snapshot = verify_ledger(folder / 'ledger.sqlite', identity, expected_head_sha256=report['ledger_head'])
    check = load(folder / 'winner_verification.json') if (folder / 'winner_verification.json').exists() else {}
    checked = trajectory_report(snapshot, method=method, geometry_id=alias,
        verification=check.get('result'), reference=reference.get('result'), failure=load(folder / 'failure.json'))
    require(report == checked and report['status'] in ('complete', 'failed'),
            'Lane trajectory report differs or is not terminal')
    require(load(folder / 'job_status.json') == {'job_id': job_id(method, alias), 'status': report['status'],
        'exit_code': 0 if report['status'] == 'complete' else 2,
        'started_utc': snapshot['created_utc'], 'finished_utc': snapshot['last_event_utc'],
        'ledger_head': snapshot['head_sha256'], 'queue_id': identity['queue_id']}, 'Lane job exit receipt differs')
    if snapshot['incumbent'] is not None:
        require(load(folder / 'circuit_resources.json') == resource_report(snapshot['incumbent']['query'], catalogue),
                'Lane winning-circuit resource report differs')
    return checked


def verify_lane(workspace, lane_id):
    root, parent, spec, prepared, allocation, physical, reference = _context(workspace, lane_id)
    destination = parent / 'lanes' / lane_id
    completed = load(destination / 'completed.json')
    require(load(destination / 'binding.json') == _binding(root, parent, spec), 'Lane binding changed')
    original = load(parent / 'gpu_allocation.json')['lanes'][lane_id]
    require(load(destination / 'gpu_runtime.json') == original['runtime'], 'Original lane GPU runtime changed')
    history = verify_execution_history(root, parent, destination, lane_id)
    if completed['schema_version'] == 'section9-control-lane-completion-v1':
        require(bool(history), 'Controlled lane completion lacks an execution binding')
        last = history[-1][1]
        require(completed['control_registration_sha256'] == last['registration_sha256']
                and completed['allocation_epoch_sha256'] == last['allocation_epoch_sha256'],
                'Controlled lane completion allocation changed')
    else:
        require(completed['schema_version'] == 'section9-recovery-lane-completion-v1' and not history,
                'Legacy lane seal cannot contain unregistered control execution')
    require(completed['binding_sha256'] == sha256_file(destination / 'binding.json')
            and completed['inventory'] == _inventory(root, destination), 'Sealed lane artifact inventory differs')
    catalogue = read_json(root / CATALOGUE)
    reports = {job_id(m, spec['target']): _verify_job(root, destination, spec, m, physical, reference, catalogue)
               for m in spec['methods']}
    from .section9_ledger import ScalarCache
    cache = ScalarCache(destination / 'scalar_cache.sqlite', create=False, read_only=True)
    try:
        cache.verify(expected_sha256=completed['scalar_cache_sha256'])
    finally:
        cache.close()
    costs_path = workspace_path(root, completed['costs_path'], namespace=OUTPUT_ROOT)
    require(costs_path.is_relative_to(destination / 'execution_attempts'), 'Lane cost receipt is outside its lane')
    costs = retained_lane_costs(root, destination, reports)
    require(load(costs_path) == costs, 'Lane cumulative costs differ')
    require(completed['lane_id'] == lane_id and completed['lane_execution_complete'] is True
            and completed['trajectories_accounted'] == len(spec['methods'])
            and completed['all_trajectories_successful'] == all(r['status'] == 'complete' for r in reports.values()),
            'Lane completion flags differ')
    resources = {n: load(destination / n / 'circuit_resources.json') for n in reports
                 if (destination / n / 'circuit_resources.json').exists()}
    return {'all_passed': True, 'lane_id': lane_id, 'lane_execution_complete': True,
        'completed_sha256': sha256_file(destination / 'completed.json'),
        'all_trajectories_successful': completed['all_trajectories_successful'],
        'reports': reports, 'resources': resources, 'costs': costs}


def run_lane(workspace, lane_id, *, chemistry_python=None, execute=False, resume=False):
    require(execute is True, 'Parallel lane requires explicit --execute')
    require(platform.system() == 'Linux', 'Run parallel lanes in the pinned Linux server environments')
    root, parent, spec, prepared, allocation, physical, reference = _context(workspace, lane_id)
    destination, alias = parent / 'lanes' / lane_id, spec['target']
    if (destination / 'completed.json').exists():
        return verify_lane(root, lane_id)
    verify_execution_history(root, parent, destination, lane_id)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == allocation['gpu_uuid'],
            'Expose only this lane allocated GPU UUID using CUDA_VISIBLE_DEVICES')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from .training import _run_lock
    from .section7 import _gpu_operation_lock
    from .section8_inference import ResidentGenerator
    from .section9_recovery_scoring import PhysicsClient, compatible_reference_identity
    from .section9_ledger import ScalarCache
    from .section9_reporting import trajectory_report
    from .section9_resources import resource_report
    with _run_lock(root / 'large_qubit/operations/section9_recovery_v1' / lane_id), _gpu_operation_lock(root, 'cuda:0'):
        require(not destination.exists() or resume, 'Existing lane requires --resume; no overwrite or new seed')
        require(not destination.exists() or (destination / 'binding.json').exists(),
                'Existing lane lacks its binding; preserve interrupted initialization')
        destination.mkdir(parents=True, exist_ok=True)
        save(destination / 'binding.json', _binding(root, parent, spec))
        original = load(parent / 'gpu_allocation.json')['lanes'][lane_id]
        save(destination / 'gpu_runtime.json', original['runtime'])
        verify_execution_history(root, parent, destination, lane_id)
        attempt, started = _new_attempt(destination), time.monotonic()
        from .section9_control import execution_binding
        execution = execution_binding(root, lane_id)
        save(attempt / 'control_binding.json', execution)
        generator, client, cache, ledger = None, None, None, None
        load_number, reports = 0, {}
        catalogue = read_json(root / CATALOGUE)

        def close_generator():
            nonlocal generator
            if generator is not None:
                previous, generator = generator, None
                try:
                    save(attempt / (previous.model_key + '_inference_cost.json'), previous.statistics())
                finally:
                    previous.close()

        def close_physics():
            nonlocal client
            previous, client = client, None
            if previous is not None:
                previous.close()

        def physics():
            nonlocal client, load_number
            if client is None:
                load_number += 1
                prefix = attempt / 'physics_loads' / f'{load_number:04d}_{alias}'
                save(prefix.with_name(prefix.name + '_started.json'), {'utc': utc(), 'geometry_id': alias})
                loading = time.monotonic()
                client = PhysicsClient(root, chemistry_python or root / '.venv-chemistry/bin/python',
                    alias, execute=True, threads=1, expected_freeze_sha256=prepared['final_freeze_sha256'])
                require(compatible_reference_identity(client.identity, physical),
                        'Recovery worker scientific identity differs from preserved reference physics')
                save(prefix.with_name(prefix.name + '_completed.json'),
                    {'utc': utc(), 'geometry_id': alias, 'wall_seconds': time.monotonic() - loading,
                     'physics_identity': client.identity})
            return client

        try:
            runtime = pin_runtime(root, attempt)
            require(runtime == execution['allocation']['runtime'] == allocation['runtime']
                    and runtime['gpu_uuid'] == allocation['gpu_uuid'],
                    'Lane runtime differs from its authenticated allocation epoch')
            check_pause(pause_path(root))
            cache_path, cache_marker = destination / 'scalar_cache.sqlite', destination / 'cache_initialized.json'
            prior_jobs = any((destination / name / 'identity.json').exists()
                             or (destination / name / 'ledger.sqlite').exists() for name in spec['job_ids'])
            require(not (cache_marker.exists() or prior_jobs) or cache_path.exists(),
                    'Previously initialized lane scalar cache is missing; never silently recreate it')
            cache = ScalarCache(cache_path, create=not cache_path.exists())
            save(cache_marker, {'schema_version': 'section9-recovery-cache-initialization-v1',
                'lane_id': lane_id, 'cache_scope': 'lane_local', 'cache_path': cache_path.relative_to(root).as_posix()})
            for method in spec['methods']:
                check_pause(pause_path(root))
                name = job_id(method, alias)
                folder = destination / name
                folder.mkdir(exist_ok=True)
                identity = ledger_identity(sha256_file(root / PREPARATION), lane_id, method, alias, physical['cache_identity'])
                ledger_path = folder / 'ledger.sqlite'
                require(not (folder / 'identity.json').exists() or ledger_path.exists(),
                        'Previously initialized lane job ledger is missing; never reset its budget')
                ledger = PausableLedger(ledger_path, identity, create=not ledger_path.exists(),
                                        pause_marker=pause_path(root))
                save(folder / 'identity.json', identity)
                if (folder / 'report.json').exists():
                    reports[name] = _verify_job(root, destination, spec, method, physical, reference, catalogue)
                    ledger.close()
                    ledger = None
                    continue
                if ledger.frontier()['pending_requests'] or ledger.frontier()['pending_attempts']:
                    ledger.recover_pending('Lane interrupted; durable reservations remain charged')
                failure = _failure(ledger.snapshot())
                if reference['status'] != 'completed':
                    failure = {'kind': 'reference_failed', 'reason': reference.get('error', 'Reference unavailable')}
                    if not ledger.frontier()['halted']:
                        ledger.halt(failure['reason'])
                if failure is None and not ledger.frontier()['finished']:
                    # Start the CPU evaluator before loading a resident GPU model.
                    # Correct process-local peak measurement also covers later restarts.
                    physics()
                    if method in METHODS[:5]:
                        if generator is None:
                            key = spec['model_key']
                            save(attempt / (key + '_load_started.json'), {'utc': utc(), 'model_key': key})
                            generator = ResidentGenerator(root, key, device='cuda:0', execute=True)
                            require(generator.metadata['runtime_identity'] == runtime,
                                    'Loaded lane generator runtime differs')
                            save(attempt / (key + '_inference_metadata.json'), generator.metadata)
                        generator.switch(method, adapter_path=prepared['training']['fits'].get(method, {}).get('adapter_path'))
                    else:
                        close_generator()
                    retrieval, angles = None, None
                    if method == 'training_circuit_retrieval':
                        from .corpus_v2 import load_release
                        from .baselines import retrieval_candidates
                        release = load_release(root, consumer_id=method, expected_release_sha256=prepared['release_sha256'])
                        source = retrieval_candidates(alias, list(release.retrieval_rows()), catalogue=catalogue,
                            length=32, acquisition_cost=release.source_acquisition_cost, max_candidates=1023)
                        save(folder / 'retrieval_source.json', source)
                        retrieval = source['candidates']
                    if method == 'conventional_vqe':
                        from .section5 import SCREENING_PATH
                        angles = read_json(root / SCREENING_PATH)['vqe_initial_angles']
                    def progress(restart, record):
                        index = record.get('coordinate_updates', record.get('requests', 0))
                        save(folder / 'optimizer_progress' / f'restart_{restart}_{index:04d}.json', record)
                        save(attempt / 'optimizer_replay_work' / name / f'restart_{restart}_{index:04d}.json', record)
                    def algebraic(restart, index, event, record):
                        save(attempt / 'optimizer_algebraic_work' / name / f'restart_{restart}_{index:04d}_{event}.json', record)
                    try:
                        with PausableScorer(ledger, cache, physical['cache_identity'], physics().score,
                                            max_pending=2, pause_marker=pause_path(root)) as scorer:
                            result = execute_trajectory(ledger, scorer, method, alias, catalogue=catalogue,
                                generator=generator, retrieval=retrieval, initial_angles=angles,
                                optimizer_progress=progress, optimizer_work=algebraic)
                        if method == 'conventional_vqe':
                            save(folder / 'optimizer.json', {'restarts': result['optimizer'],
                                'energy_calls_charged_in_ledger': True, 'analytic_gradient_calls': 0,
                                'algebraic_coordinate_fits': sum(len(x['history']) for x in result['optimizer'])})
                    except Exception as error:
                        snapshot = ledger.snapshot()
                        scientific = any(r['status'] == 'failed' and isinstance(r.get('evaluation'), dict)
                            and (r['evaluation'].get('cost_category') == 'ordinary_energy_request'
                                 or (r['evaluation'].get('stage') == 'score'
                                     and r['evaluation'].get('transport_failure') is True)) for r in snapshot['requests'])
                        numerical = str(snapshot.get('stop_reason', '')).startswith('Numerical incident: ')
                        if not scientific and not numerical:
                            raise error
                        failure = _failure(snapshot)
                        close_physics()
                    if failure is None:
                        ledger.finish('Frozen request or attempt allowance reached')
                snapshot, verification = ledger.snapshot(), None
                best = snapshot['incumbent']
                if best is not None:
                    check_pause(pause_path(root))
                    check = overhead_once(folder, 'winner_verification', lambda: physics().verify(best['query']))
                    if check['status'] == 'completed':
                        verification = check['result']
                    else:
                        failure = dict(failure or {}, integrity_verification_failed=check.get('error'))
                        close_physics()
                    save(folder / 'circuit_resources.json', resource_report(best['query'], catalogue))
                report = trajectory_report(snapshot, method=method, geometry_id=alias,
                    verification=verification, reference=reference.get('result'), failure=failure)
                require(report['status'] in ('complete', 'failed'), 'Lane produced a nonterminal trajectory')
                save(folder / 'failure.json', failure)
                save(folder / 'job_status.json', {'job_id': name, 'status': report['status'],
                    'exit_code': 0 if report['status'] == 'complete' else 2,
                    'started_utc': snapshot['created_utc'], 'finished_utc': snapshot['last_event_utc'],
                    'ledger_head': snapshot['head_sha256'], 'queue_id': identity['queue_id']})
                save(folder / 'report.json', report)
                reports[name] = report
                ledger.close()
                ledger = None
                print(json.dumps({'lane': lane_id, 'job': name, 'status': report['status'],
                    'logical_requests': snapshot['logical_requests'], 'completion_attempts': snapshot['completion_attempts']}), flush=True)
            close_generator()
            cache_report = cache.verify()
            cache.close()
            cache = None
            close_physics()
            save(attempt / 'finished.json', {'utc': utc(), 'success': True, 'wall_seconds': time.monotonic() - started})
            cost_path = attempt / 'costs.json'
            save(cost_path, retained_lane_costs(root, destination, reports))
            save(destination / 'completed.json', {'schema_version': 'section9-control-lane-completion-v1',
                'lane_id': lane_id, 'lane_execution_complete': True, 'trajectories_accounted': len(spec['methods']),
                'all_trajectories_successful': all(r['status'] == 'complete' for r in reports.values()),
                'binding_sha256': sha256_file(destination / 'binding.json'),
                'costs_path': cost_path.relative_to(root).as_posix(),
                'scalar_cache_sha256': cache_report['sha256'], 'inventory': _inventory(root, destination),
                'control_registration_sha256': execution['registration_sha256'],
                'allocation_epoch_sha256': execution['allocation_epoch_sha256'],
                'training_executed': False, 'pipeline_seeds': [22], 'performance_selection': False})
            return verify_lane(root, lane_id)
        except PauseRequested:
            # PausableScorer.__exit__ has drained all already admitted physics work.
            # Completed but unscored generated answers remain in the durable ledger.
            close_generator()
            close_physics()
            frontier = ledger.frontier() if ledger is not None else None
            require(frontier is None or (frontier['pending_requests'] == 0
                    and frontier['pending_attempts'] == 0 and not frontier['halted']),
                    'Pause did not reach a clean durable frontier; preserve and inspect the recorded work')
            receipt = {'utc': utc(), 'lane_id': lane_id, 'lane_paused': True,
                       'pending_requests': 0, 'pending_attempts': 0,
                       'ledger_head': frontier['head_sha256'] if frontier else None,
                       'wall_seconds': time.monotonic() - started}
            save(attempt / 'paused.json', receipt)
            save(attempt / 'finished.json', {**receipt, 'success': True, 'paused': True})
            return {'all_passed': True, 'lane_id': lane_id, 'lane_paused': True,
                    'lane_execution_complete': False, 'paused_receipt':
                    (attempt / 'paused.json').relative_to(root).as_posix()}
        except BaseException as error:
            cleanup_error = None
            try:
                close_generator()
            except Exception as problem:
                cleanup_error = str(problem)
            if not (destination / 'completed.json').exists():
                try:
                    save(attempt / 'failed.json', {'utc': utc(), 'error': str(error),
                        'scientific_failure_report': getattr(error, 'report', None),
                        'cleanup_error': cleanup_error, 'wall_seconds': time.monotonic() - started})
                except Exception as problem:
                    print('Lane failure receipt could not be persisted: ' + str(problem), file=sys.stderr, flush=True)
            raise
        finally:
            if ledger is not None:
                ledger.close()
            if cache is not None:
                cache.close()
            close_physics()


def _arm_parent_death_signal(parent_pid):
    """Linux workers terminate if their coordinator dies, including a setup race."""
    require(platform.system() == 'Linux', 'Parent-death supervision requires Linux')
    require(type(parent_pid) is int and parent_pid > 1 and os.getppid() == parent_pid,
            'Parallel coordinator parent is absent or changed')
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, 'Cannot install coordinator parent-death signal')
    require(os.getppid() == parent_pid, 'Parallel coordinator died during lane startup')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'verify'))
    parser.add_argument('--root', default='.')
    parser.add_argument('--lane', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--chemistry-python')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args(argv)
    try:
        if args.action == 'run':
            def terminate(signum, frame):
                raise KeyboardInterrupt('Lane terminated by coordinator signal')
            signal.signal(signal.SIGTERM, terminate)
            if args.parent_pid is not None:
                _arm_parent_death_signal(args.parent_pid)
            result = run_lane(args.root, args.lane, chemistry_python=args.chemistry_python,
                              execute=args.execute, resume=args.resume)
        else:
            result = verify_lane(args.root, args.lane)
        # Detailed trajectories are retained in immutable receipts, not repeated in logs.
        print(json.dumps({k: v for k, v in result.items() if k not in ('reports', 'resources', 'costs')}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
