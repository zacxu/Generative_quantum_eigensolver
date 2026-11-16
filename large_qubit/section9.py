"""User-launched, frozen 16-trajectory main matrix. Imports execute no science."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

from .common import digest, read_json, require, sha256_file, workspace_path
from .protocol import METHODS, seed_for
from .section8 import save, load, utc, controller_runtime, _new_attempt
from .section8_policy import FINAL_FREEZE, CATALOGUE
from .section9_policy import (REGISTRATION, PREPARATION, OUTPUT_ROOT, MATRIX, TARGETS,
                             policy, jobs, job_id, attempt_seed)


def directory(root):
    return workspace_path(root, OUTPUT_ROOT + '/' + MATRIX, namespace=OUTPUT_ROOT)


def prepare(workspace, *, execute=False):
    require(execute is True, 'Main preparation requires explicit --execute')
    root = Path(workspace).resolve(strict=True)
    from .section9_registration import verify
    from .section8 import verify_final_freeze
    from .section8_policy import PREPARATION as VALIDATION_PREPARATION
    registered = verify(root)
    frozen = verify_final_freeze(root)  # Must precede model access or target array access.
    value = load(root / FINAL_FREEZE)
    require(value['protocol']['main'] == policy()['scientific_protocol'], 'Frozen main protocol differs')
    prepared = {'schema_version': 'section9-preparation-v1', 'policy': policy(),
        'registration_sha256': registered['registration_sha256'],
        'final_freeze_sha256': frozen['freeze_sha256'],
        'training': value['final_adapters_and_training'], 'contexts': value['prompt_contexts'],
        'release_sha256': load(root / VALIDATION_PREPARATION)['release_sha256'],
        'catalogue_sha256': sha256_file(root / CATALOGUE),
        'scientific_energy_requests': 0, 'model_weights_loaded': False, 'training_executed': False}
    save(workspace_path(root, PREPARATION, namespace='large_qubit/preflights'), prepared)
    return {'all_passed': True, 'preparation_sha256': sha256_file(root / PREPARATION),
            'main_matrix_complete': False, 'next': 'explicit user-controlled main-matrix launch'}


def verify_preparation(workspace):
    root = Path(workspace).resolve(strict=True)
    from .section9_registration import verify
    from .section8 import verify_final_freeze
    registered = verify(root)
    frozen = verify_final_freeze(root)
    prepared, value = load(root / PREPARATION), load(root / FINAL_FREEZE)
    from .section8_policy import PREPARATION as VALIDATION_PREPARATION
    require(prepared['policy'] == policy()
            and prepared['registration_sha256'] == registered['registration_sha256']
            and prepared['final_freeze_sha256'] == frozen['freeze_sha256']
            and prepared['training'] == value['final_adapters_and_training']
            and prepared['contexts'] == value['prompt_contexts']
            and prepared['release_sha256'] == load(root / VALIDATION_PREPARATION)['release_sha256']
            and prepared['catalogue_sha256'] == sha256_file(root / CATALOGUE),
            'Main preparation, frozen protocol or completed adapters changed')
    return prepared


def ledger_identity(preparation_sha, method, alias, cache_identity):
    name = job_id(method, alias)
    require(cache_identity['geometry_id'] == alias and cache_identity['partition'] == 'final_test',
            'Main ledger requires the exact final-test cache identity')
    return {'schema_version': 'section9-trajectory-identity-v1',
        'preparation_sha256': preparation_sha, 'policy_sha256': digest(policy()),
        'method': method, 'geometry_id': alias, 'job_id': name,
        'queue_id': MATRIX + '/' + name + '/ordered_cpu_queue', 'cache_identity': cache_identity}


def execute_trajectory(ledger, scorer, method, alias, *, catalogue, generator=None,
                       retrieval=None, initial_angles=None, optimizer_progress=None, optimizer_work=None):
    """Pure controller mechanics, injectable for CPU fixtures; never changes budgets."""
    job_id(method, alias)
    state = ledger.frontier()
    require(not state['pending_requests'] and not state['pending_attempts'] and not state['halted'],
            'Recover the durable frontier before admitting work')
    scorer.submit('hf', {'geometry_id': alias, 'kind': 'hf'}, kind='hf').result()
    optimizer = []
    if method in METHODS[:5]:
        require(generator is not None, 'A resident generator is required')
        pending = []
        for ordinal in range(1, 2049):
            if ledger.frontier()['logical_requests'] >= 1024:
                break
            call_id = f'attempt_{ordinal:04d}'
            previous = ledger.lookup_attempt(call_id)
            if previous is not None:
                if previous['status'] != 'completed' or not previous['valid']:
                    continue
                output = previous['output']
            else:
                seed = attempt_seed(method, alias, ordinal)
                actual = ledger.reserve_attempt(call_id, {'seed': seed})
                require(actual['ordinal'] == ordinal, 'Attempt frontier changed')
                try:
                    output = generator.generate(alias, catalogue, seed)
                except Exception as error:
                    ledger.finish_attempt(ordinal, {'error': str(error)}, valid=False,
                                          failure_reason='inference infrastructure failure')
                    raise
                ledger.finish_attempt(ordinal, output, valid=output['valid'],
                    failure_reason=None if output['valid'] else 'invalid complete answer')
                require(output.get('error_kind') != 'infrastructure',
                        'Inference infrastructure failure recorded; inspect before explicit resume')
            if output['valid']:
                pending.append(scorer.submit(call_id, {'geometry_id': alias, 'kind': 'discrete',
                    'ops': output['ops']}, attempt_ordinal=ordinal))
                if len(pending) >= 2:
                    pending.pop(0).result()
        for future in pending:
            future.result()
    elif method == 'uniform_random':
        from .baselines import uniform_random
        rows = uniform_random(catalogue, length=32, count=1023,
            seed=seed_for('generation', group=f'{method}:{alias}'))['circuits']
        for i, ops in enumerate(rows):
            scorer.submit(f'random_{i:04d}', {'geometry_id': alias, 'kind': 'discrete', 'ops': ops}).result()
    elif method == 'training_circuit_retrieval':
        require(retrieval is not None and len(retrieval) == 1023, 'Exactly 1023 authenticated retrieval rows required')
        for i, row in enumerate(retrieval):
            scorer.submit(f'retrieval_{i:04d}', {'geometry_id': alias, 'kind': 'discrete', 'ops': row['ops']}).result()
    else:
        from .vqe import optimize_compact, OFFSETS
        require(initial_angles is not None and len(initial_angles) == 24, 'Pinned training-only VQE vector required')
        for restart, (cap, initial) in enumerate(zip((511, 512), ([0.] * 24, initial_angles))):
            fit_started = None
            def evaluate(parameters, metadata):
                nonlocal fit_started
                if metadata['stage'] == 'coordinate_proposal' and optimizer_work is not None:
                    optimizer_work(restart, metadata['local_request'], 'fit_completed', {
                        'coordinate': metadata['coordinate'], 'sweep': metadata['sweep'],
                        'proposal_parameters': parameters, 'additional_energy_requests': 0,
                        'algebraic_candidate_preparation_seconds': time.perf_counter() - fit_started})
                value = scorer.submit(f'vqe_{restart}_{metadata["local_request"]:04d}',
                    {'geometry_id': alias, 'kind': 'vqe', 'parameters': parameters}).result()['energy_hartree']
                if metadata['stage'] == 'coordinate_sample' and metadata['offset'] == OFFSETS[-1]:
                    if optimizer_work is not None:
                        optimizer_work(restart, metadata['local_request'] + 1, 'fit_started', {
                            'coordinate': metadata['coordinate'], 'sweep': metadata['sweep']})
                    fit_started = time.perf_counter()
                return value
            def progress(value):
                if optimizer_progress is not None:
                    optimizer_progress(restart, value)
            try:
                result = optimize_compact(evaluate, initial, max_requests=cap, progress=progress)
            except ValueError as error:
                if str(error) == 'Excitation coordinate does not satisfy the five-term energy identity':
                    ledger.halt('Numerical incident: ' + str(error))
                raise
            optimizer.append(result)
    return {'snapshot': ledger.snapshot(), 'optimizer': optimizer}


def runtime_matches_validation(actual, expected):
    """The first main allocation may differ; arithmetic, libraries and device class may not."""
    require('H200' in actual.get('gpu_name', ''), 'Main inference requires the frozen H200 device class')
    require({k: v for k, v in actual.items() if k != 'gpu_uuid'} ==
            {k: v for k, v in expected.items() if k != 'gpu_uuid'},
            'Main numerical inference environment differs from the validated environment')


def pin_runtime(root, destination):
    """Called only after the final-freeze gate, before loading any model weights."""
    import torch
    from .section8_inference import _runtime_identity
    from .section8_policy import OUTPUT_ROOT as VALIDATION_ROOT
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expose exactly one allocated H200 using CUDA_VISIBLE_DEVICES')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = _runtime_identity(torch, 0)
    frozen = load(root / FINAL_FREEZE)
    panel = root / VALIDATION_ROOT / frozen['selected_panel']
    for model in ('gpt2_xl', 'qwen3_4b'):
        runtime_matches_validation(actual, load(panel / (model + '_runtime.json')))
    save(destination / 'gpu_runtime.json', actual)  # Changed UUID on resume refuses here.
    return actual


def overhead_once(folder, stem, call):
    """Intent before work, immutable outcome, no hidden retry after uncertain completion."""
    result_path, marker = folder / (stem + '.json'), folder / (stem + '_started.json')
    if result_path.exists():
        return load(result_path)
    if marker.exists():
        receipt = {'status': 'failed', 'error': 'Interrupted overhead; execution extent unknown',
                   'uncertain_cost': True, 'result': None, 'failure_report': None}
    else:
        save(marker, {'utc': utc(), 'cost_category': stem})
        from .section9_scoring import ScoringError
        try:
            result = call()
            receipt = {'status': 'completed', 'result': result, 'uncertain_cost': False}
        except ScoringError as error:
            receipt = {'status': 'failed', 'error': str(error), 'failure_report': error.report,
                       'result': None, 'uncertain_cost': True}
    save(result_path, receipt)
    return receipt


def _inventory(root, destination):
    result = {}
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or path == destination / 'completed.json' or path.name.endswith(('-wal', '-shm')):
            continue
        require(not path.is_symlink(), 'Main inventory cannot contain symbolic links')
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def _failure(snapshot):
    if snapshot['failed_requests'] or snapshot['abandoned_requests'] or snapshot['halted']:
        kinds = []
        for row in snapshot.get('requests', []):
            if row['status'] == 'failed':
                evidence = row.get('evaluation') or {}
                kinds.append('physical_or_numerical_failure' if evidence.get('cost_category') == 'ordinary_energy_request'
                             else 'uncertain_physical_transport' if evidence.get('transport_failure')
                             else 'infrastructure_failure')
        if str(snapshot.get('stop_reason', '')).startswith('Numerical incident:'):
            kinds.append('numerical_incident')
        if snapshot['abandoned_requests']:
            kinds.append('interrupted_or_cancelled_physical_work')
        return {'kind': 'terminated_without_refill', 'incident_categories': sorted(set(kinds)),
                'reason': snapshot['stop_reason'],
                'failed_requests': snapshot['failed_requests'], 'abandoned_requests': snapshot['abandoned_requests']}
    return None


def training_preflight_overhead(root, frozen):
    """Read the separate disposable GPU measurements already bound by Section 7."""
    from .section7 import PREFLIGHT_ROOT
    path = root / PREFLIGHT_ROOT / 'freeze.json'
    require(sha256_file(path) == frozen['final_adapters_and_training']['numerical_freeze_sha256'],
            'Training numerical freeze changed in cost accounting')
    numerical = read_json(path)
    costs = {}
    for fit, binding in numerical['preflight_reports'].items():
        report_path = workspace_path(root, binding['path'], namespace='large_qubit/preflights')
        require(sha256_file(report_path) == binding['sha256'], 'Training preflight cost receipt changed')
        costs[fit] = {'source': binding, 'benchmark': read_json(report_path)['benchmark']}
    return {'fits': costs, 'scope': 'disposable preflight measurements including warmup and failed candidates; separate from production fits'}


def _retained_costs(root, destination, reports):
    """Keep categories separate; acquisition and three fits enter the global total once."""
    frozen = load(root / FINAL_FREEZE)
    overhead = {}
    for alias in TARGETS:
        folder = destination / 'targets' / alias
        if (folder / 'reference.json').exists():
            overhead[alias + '/reference'] = load(folder / 'reference.json')
    for name in reports:
        path = destination / name / 'winner_verification.json'
        if path.exists():
            overhead[name + '/winner_verification'] = load(path)
    executions = []
    for attempt in sorted((destination / 'execution_attempts').glob('attempt_*')):
        receipt = {'path': attempt.relative_to(root).as_posix(), 'status': 'interrupted',
                   'started': load(attempt / 'started.json'), 'inference_sessions': []}
        for name in ('finished.json', 'failed.json'):
            if (attempt / name).exists():
                receipt.update(load(attempt / name), status=name[:-5])
        for session in sorted(attempt.glob('*_load_started.json')):
            prefix = session.name[:-len('_load_started.json')]
            done = attempt / (prefix + '_inference_cost.json')
            receipt['inference_sessions'].append({'session': prefix,
                'load_intent': load(session), 'cost': load(done) if done.exists() else None,
                'unknown_unflushed_model_cost': not done.exists()})
        starts = sorted((attempt / 'optimizer_algebraic_work').rglob('*_fit_started.json'))
        updates = sorted((attempt / 'optimizer_algebraic_work').rglob('*_fit_completed.json'))
        receipt['started_algebraic_coordinate_fits_including_replay'] = len(starts)
        receipt['completed_algebraic_coordinate_fits_including_replay'] = len(updates)
        receipt['recorded_algebraic_candidate_preparation_seconds'] = sum(
            load(p)['algebraic_candidate_preparation_seconds'] for p in updates)
        receipt['algebraic_work_allowance_per_vqe_trajectory'] = 170
        receipt['partial_algebraic_work_may_be_unknown'] = len(starts) != len(updates) or receipt['status'] != 'finished'
        receipt['algebraic_timing_scope'] = 'five-point solve, stationary-root search and proposal preparation; excludes energy evaluations'
        receipt['physics_worker_loads'] = []
        for marker in sorted((attempt / 'physics_loads').glob('*_started.json')):
            result = marker.with_name(marker.name.replace('_started.json', '_completed.json'))
            receipt['physics_worker_loads'].append({'started': load(marker),
                'completed': load(result) if result.exists() else None,
                'unknown_unflushed_setup_cost': not result.exists()})
        executions.append(receipt)
    return {'source_acquisition_charged_once': frozen['costs']['source_acquisition_charged_once'],
        'three_training_fits_charged_once': frozen['final_adapters_and_training'],
        'training_preflight_overhead_charged_once': training_preflight_overhead(root, frozen),
        'all_validation_attempts': {k: v for k, v in frozen['costs'].items()
                                    if k not in ('source_acquisition_charged_once', 'training_costs')},
        'main_by_trajectory': {name: {k: v for k, v in r.get('numerical_cost', {}).items()
                                     if k in ('search', 'generation')} for name, r in reports.items()},
        'main_reference_and_integrity_overhead': overhead, 'execution_attempts': executions,
        'method_shared_cost_attribution': {m: {
            'source_acquisition': m in ('gpt2_xl_weighted_lora', 'gpt2_xl_uniform_lora',
                                        'qwen3_4b_weighted_lora', 'training_circuit_retrieval'),
            'fit': m if m in frozen['final_adapters_and_training']['fits'] else None} for m in METHODS},
        'totals_are_known_lower_bounds_if_interrupted': True,
        'overlap': 'generation and scoring overlap; component times are not elapsed wall time',
        'inference_accounting': 'attempt times belong to trajectories; model session totals are crosschecks, not additional generation charges',
        'corpus_accounting': 'shared collection is incurred once globally; method attribution is not an extra copy'}


def run_matrix(workspace, *, chemistry_python=None, execute=False, resume=False):
    require(execute is True, 'Main matrix requires explicit --execute')
    require(platform.system() == 'Linux', 'Run the main matrix in the pinned Linux server environments')
    root = Path(workspace).resolve(strict=True)
    prepared = verify_preparation(root)  # No GPU/target access before this gate.
    dest = directory(root)
    if (dest / 'completed.json').exists():
        return verify_matrix(root)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from .training import _run_lock
    from .section7 import _gpu_operation_lock
    from .section8_inference import ResidentGenerator
    from .section9_scoring import PhysicsClient, ScoringError
    from .section9_ledger import TrajectoryLedger, OrderedScorer, ScalarCache
    from .section9_reporting import trajectory_report, aggregate_reports, verify_reference
    from .section9_resources import resource_report
    with _run_lock(root / 'large_qubit/operations/section9_v1/matrix'), _gpu_operation_lock(root, 'cuda:0'):
        existed = dest.exists()
        require(not existed or resume, 'Existing main matrix requires --resume; no overwrite or new seed')
        require(not existed or (dest / 'binding.json').exists(),
                'Existing matrix lacks its binding; preserve and inspect interrupted initialization')
        dest.mkdir(parents=True, exist_ok=True)
        save(dest / 'binding.json', {'preparation_sha256': sha256_file(root / PREPARATION),
            'policy': policy(), 'controller_runtime': controller_runtime(),
            'final_freeze_sha256': prepared['final_freeze_sha256']})
        runtime = pin_runtime(root, dest)
        attempt = _new_attempt(dest)
        started = time.monotonic()
        generator, clients, cache, ledger = None, {}, None, None
        physics_load_number = 0
        catalogue = read_json(root / CATALOGUE)
        reports, references = {}, {}

        def close_generator():
            nonlocal generator
            if generator is not None:
                previous = generator
                generator = None
                try:
                    save(attempt / (previous.model_key + '_inference_cost.json'), previous.statistics())
                finally:
                    previous.close()

        def close_physics(alias):
            value = clients.pop(alias, None)
            if value is not None:
                value.close()

        def physics(alias):
            nonlocal physics_load_number
            if alias not in clients:
                physics_load_number += 1
                load_prefix = attempt / 'physics_loads' / f'{physics_load_number:04d}_{alias}'
                save(load_prefix.with_name(load_prefix.name + '_started.json'), {'utc': utc(), 'geometry_id': alias})
                loading_started = time.monotonic()
                client = PhysicsClient(root, chemistry_python or root / '.venv-chemistry/bin/python',
                    alias, execute=True, threads=1, expected_freeze_sha256=prepared['final_freeze_sha256'])
                clients[alias] = client
                save(load_prefix.with_name(load_prefix.name + '_completed.json'), {'utc': utc(),
                    'geometry_id': alias, 'wall_seconds': time.monotonic() - loading_started})
                folder = dest / 'targets' / alias
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / 'physics_identity.json', client.identity)
            return clients[alias]

        def physical_failure_or_raise(error, snapshot):
            """Never convert a quota/hash/cache/storage error into a scientific outcome."""
            failed = [r for r in snapshot['requests'] if r['status'] == 'failed']
            scientific = any(isinstance(r.get('evaluation'), dict)
                             and (r['evaluation'].get('cost_category') == 'ordinary_energy_request'
                                  or (r['evaluation'].get('stage') == 'score'
                                      and r['evaluation'].get('transport_failure') is True))
                             for r in failed)
            numerical = str(snapshot.get('stop_reason', '')).startswith('Numerical incident: ')
            if not scientific and not numerical:
                raise error
            return _failure(snapshot)

        try:
            # Two resident Hamiltonians; references and every later physical call are serialized.
            for alias in TARGETS:
                client = physics(alias)
                target_folder = dest / 'targets' / alias
                def checked_reference():
                    value = client.reference()
                    try:
                        verify_reference(value, client.identity['cache_identity'])
                    except ValueError as error:
                        raise ScoringError('Reference integrity gate failed: ' + str(error), report=value) from error
                    return value
                reference = overhead_once(target_folder, 'reference', checked_reference)
                if reference['status'] == 'completed':
                    verify_reference(reference['result'], client.identity['cache_identity'])
                else:
                    close_physics(alias)
                references[alias] = reference
            cache_path, cache_marker = dest / 'scalar_cache.sqlite', dest / 'cache_initialized.json'
            prior_jobs = any((dest / job_id(m, a) / 'identity.json').exists()
                             or (dest / job_id(m, a) / 'ledger.sqlite').exists() for m, a in jobs())
            require(not (cache_marker.exists() or prior_jobs) or cache_path.exists(),
                    'Previously initialized scalar cache is missing; never silently recreate it')
            cache = ScalarCache(cache_path, create=not cache_path.exists())
            save(cache_marker, {'schema_version': 'section9-cache-initialization-v1',
                                'cache_path': cache_path.relative_to(root).as_posix()})
            for method, alias in jobs():
                name = job_id(method, alias)
                folder = dest / name
                folder.mkdir(exist_ok=True)
                cache_identity = load(dest / 'targets' / alias / 'physics_identity.json')['cache_identity']
                identity = ledger_identity(sha256_file(root / PREPARATION), method, alias, cache_identity)
                ledger_path = folder / 'ledger.sqlite'
                require(not (folder / 'identity.json').exists() or ledger_path.exists(),
                        'Previously initialized job ledger is missing; never reset its budget')
                ledger = TrajectoryLedger(ledger_path, identity, create=not ledger_path.exists())
                save(folder / 'identity.json', identity)
                if (folder / 'report.json').exists():
                    reports[name] = _verify_job(root, dest, method, alias, references[alias], catalogue)
                    ledger.close()
                    ledger = None
                    continue
                reference = references[alias]
                if ledger.frontier()['pending_requests'] or ledger.frontier()['pending_attempts']:
                    ledger.recover_pending('Controller interrupted; durable reservations remain charged')
                failure = _failure(ledger.snapshot())
                if reference['status'] != 'completed':
                    failure = {'kind': 'reference_failed', 'reason': reference.get('error', 'Reference unavailable')}
                    if not ledger.frontier()['halted']:
                        ledger.halt(failure['reason'])
                if failure is None and not ledger.frontier()['finished']:
                    if method in METHODS[:5]:
                        key = 'gpt2_xl' if method.startswith('gpt2') else 'qwen3_4b'
                        if generator is None or generator.model_key != key:
                            close_generator()
                            save(attempt / (key + '_load_started.json'), {'utc': utc(), 'model_key': key})
                            generator = ResidentGenerator(root, key, device='cuda:0', execute=True)
                            require(generator.metadata['runtime_identity'] == runtime,
                                    'Loaded generator runtime changed from pinned main runtime')
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
                        # One immutable record per completed algebraic update, including partial runs.
                        index = record.get('coordinate_updates', record.get('requests', 0))
                        save(folder / 'optimizer_progress' / f'restart_{restart}_{index:04d}.json', record)
                        save(attempt / 'optimizer_replay_work' / name / f'restart_{restart}_{index:04d}.json', record)
                    def algebraic(restart, index, event, record):
                        save(attempt / 'optimizer_algebraic_work' / name / f'restart_{restart}_{index:04d}_{event}.json', record)
                    try:
                        with OrderedScorer(ledger, cache, cache_identity, physics(alias).score, max_pending=2) as scorer:
                            result = execute_trajectory(ledger, scorer, method, alias, catalogue=catalogue,
                                generator=generator, retrieval=retrieval, initial_angles=angles,
                                optimizer_progress=progress, optimizer_work=algebraic)
                        if method == 'conventional_vqe':
                            save(folder / 'optimizer.json', {'restarts': result['optimizer'],
                                'energy_calls_charged_in_ledger': True, 'analytic_gradient_calls': 0,
                                'algebraic_coordinate_fits': sum(len(x['history']) for x in result['optimizer'])})
                    except Exception as error:
                        failure = physical_failure_or_raise(error, ledger.snapshot())
                        close_physics(alias)
                    if failure is None:
                        ledger.finish('Frozen request or attempt allowance reached')
                snapshot = ledger.snapshot()
                best = snapshot['incumbent']
                verification = None
                if best is not None:
                    check = overhead_once(folder, 'winner_verification', lambda: physics(alias).verify(best['query']))
                    if check['status'] == 'completed':
                        verification = check['result']
                    else:
                        failure = dict(failure or {}, integrity_verification_failed=check.get('error'))
                        close_physics(alias)
                    save(folder / 'circuit_resources.json', resource_report(best['query'], catalogue))
                report = trajectory_report(snapshot, method=method, geometry_id=alias,
                    verification=verification, reference=reference.get('result'), failure=failure)
                save(folder / 'failure.json', failure)
                save(folder / 'job_status.json', {'job_id': name, 'status': report['status'],
                    'exit_code': 0 if report['status'] == 'complete' else 2,
                    'started_utc': snapshot['created_utc'], 'finished_utc': snapshot['last_event_utc'],
                    'ledger_head': snapshot['head_sha256'], 'queue_id': identity['queue_id']})
                save(folder / 'report.json', report)
                reports[name] = report
                ledger.close()
                ledger = None
                print(json.dumps({'job': name, 'status': report['status'],
                    'logical_requests': snapshot['logical_requests'], 'completion_attempts': snapshot['completion_attempts']}), flush=True)
            close_generator()
            cache_report = cache.verify()
            cache.close()
            cache = None
            for alias in list(clients):
                close_physics(alias)
            save(attempt / 'finished.json', {'utc': utc(), 'success': True, 'wall_seconds': time.monotonic() - started})
            resources = {n: load(dest / n / 'circuit_resources.json') for n in reports
                         if (dest / n / 'circuit_resources.json').exists()}
            aggregate = aggregate_reports(reports, resources=resources)
            save(dest / 'matrix_report.json', aggregate)
            costs = _retained_costs(root, dest, reports)
            # A crash during sealing may add a later controller attempt. Keep
            # each cost snapshot in that attempt instead of overwriting a total.
            cost_path = attempt / 'costs.json'
            save(cost_path, costs)
            summary = {'schema_version': 'section9-main-completion-v1', 'matrix': MATRIX,
                'matrix_execution_complete': True, 'trajectories_accounted': 16,
                'all_trajectories_successful': all(r['status'] == 'complete' for r in reports.values()),
                'binding_sha256': sha256_file(dest / 'binding.json'),
                'costs_path': cost_path.relative_to(root).as_posix(),
                'scalar_cache_sha256': cache_report['sha256'], 'inventory': _inventory(root, dest),
                'training_executed': False, 'pipeline_seeds': [22], 'performance_selection': False}
            save(dest / 'completed.json', summary)
            return verify_matrix(root)
        except BaseException as error:
            cleanup_error = None
            try:
                close_generator()
            except Exception as problem:
                cleanup_error = str(problem)
            if not (dest / 'completed.json').exists():
                try:
                    save(attempt / 'failed.json', {'utc': utc(), 'error': str(error),
                        'scientific_failure_report': getattr(error, 'report', None),
                        'cleanup_error': cleanup_error, 'wall_seconds': time.monotonic() - started})
                except Exception as problem:
                    print('Failure receipt could not be persisted: ' + str(problem), file=sys.stderr, flush=True)
            raise
        finally:
            if ledger is not None:
                ledger.close()
            if cache is not None:
                cache.close()
            for alias in list(clients):
                close_physics(alias)


def _verify_job(root, dest, method, alias, reference, catalogue):
    from .section9_ledger import verify_ledger
    from .section9_reporting import trajectory_report
    from .section9_resources import resource_report
    folder = dest / job_id(method, alias)
    cache_identity = load(dest / 'targets' / alias / 'physics_identity.json')['cache_identity']
    identity = ledger_identity(sha256_file(root / PREPARATION), method, alias, cache_identity)
    require(load(folder / 'identity.json') == identity, 'Main trajectory identity differs')
    report = load(folder / 'report.json')
    snapshot = verify_ledger(folder / 'ledger.sqlite', identity, expected_head_sha256=report['ledger_head'])
    check = load(folder / 'winner_verification.json') if (folder / 'winner_verification.json').exists() else {}
    checked = trajectory_report(snapshot, method=method, geometry_id=alias,
        verification=check.get('result'), reference=reference.get('result'), failure=load(folder / 'failure.json'))
    require(report == checked, 'Main trajectory report differs from durable ledger')
    require(load(folder / 'job_status.json') == {'job_id': job_id(method, alias), 'status': report['status'],
        'exit_code': 0 if report['status'] == 'complete' else 2,
        'started_utc': snapshot['created_utc'], 'finished_utc': snapshot['last_event_utc'],
        'ledger_head': snapshot['head_sha256'], 'queue_id': identity['queue_id']}, 'Job status or exit receipt differs')
    if snapshot['incumbent'] is not None:
        require(load(folder / 'circuit_resources.json') == resource_report(snapshot['incumbent']['query'], catalogue),
                'Winning-circuit common-gate resource report differs')
    return checked


def verify_matrix(workspace):
    root = Path(workspace).resolve(strict=True)
    prepared = verify_preparation(root)
    dest = directory(root)
    completed, binding = load(dest / 'completed.json'), load(dest / 'binding.json')
    require(binding['policy'] == policy() and binding['preparation_sha256'] == sha256_file(root / PREPARATION)
            and binding['final_freeze_sha256'] == prepared['final_freeze_sha256'], 'Main matrix binding changed')
    require(completed['binding_sha256'] == sha256_file(dest / 'binding.json')
            and completed['inventory'] == _inventory(root, dest), 'Sealed main artifact inventory differs')
    from .section9_reporting import verify_reference, aggregate_reports
    from .section9_ledger import ScalarCache
    references = {}
    for alias in TARGETS:
        folder = dest / 'targets' / alias
        reference = load(folder / 'reference.json')
        if reference['status'] == 'completed':
            verify_reference(reference['result'], load(folder / 'physics_identity.json')['cache_identity'])
        references[alias] = reference
    catalogue = read_json(root / CATALOGUE)
    reports = {job_id(m, a): _verify_job(root, dest, m, a, references[a], catalogue) for m, a in jobs()}
    resources = {n: load(dest / n / 'circuit_resources.json') for n in reports
                 if (dest / n / 'circuit_resources.json').exists()}
    require(load(dest / 'matrix_report.json') == aggregate_reports(reports, resources=resources), 'Main aggregate differs')
    costs_path = workspace_path(root, completed['costs_path'], namespace=OUTPUT_ROOT)
    require(load(costs_path) == _retained_costs(root, dest, reports), 'Main cumulative costs differ')
    cache = ScalarCache(dest / 'scalar_cache.sqlite', create=False, read_only=True)
    try:
        cache.verify(expected_sha256=completed['scalar_cache_sha256'])
    finally:
        cache.close()
    require(completed['matrix_execution_complete'] is True and completed['trajectories_accounted'] == 16
            and completed['all_trajectories_successful'] == all(r['status'] == 'complete' for r in reports.values()),
            'Main completion flags differ from all 16 outcomes')
    return {'all_passed': True, 'matrix_execution_complete': True, 'trajectories_accounted': 16,
            'all_trajectories_successful': completed['all_trajectories_successful'],
            'completed_sha256': sha256_file(dest / 'completed.json'),
            'scope': 'read_only_all_outcomes_including_failures; one seed, two targets'}


def status(workspace):
    root = Path(workspace).resolve(strict=True)
    dest = directory(root)
    from .section9_ledger import verify_ledger
    result = {'implementation_prepared': (root / REGISTRATION).exists(),
        'section8_freeze_exists': (root / FINAL_FREEZE).exists(),
        'preparation_exists': (root / PREPARATION).exists(), 'matrix_execution_complete': False,
        'scope': 'read_only_no_model_loading_or_energy_requests', 'jobs': {}}
    for method, alias in jobs():
        name = job_id(method, alias)
        folder = dest / name
        if not (folder / 'identity.json').exists():
            result['jobs'][name] = {'state': 'not_run'}
            continue
        snap = verify_ledger(folder / 'ledger.sqlite', load(folder / 'identity.json'))
        result['jobs'][name] = {'state': load(folder / 'report.json')['status'] if (folder / 'report.json').exists() else 'partial',
            **{k: snap[k] for k in ('logical_requests', 'completion_attempts', 'completed_requests',
                                   'failed_requests', 'abandoned_requests', 'pending_requests', 'cache_hits')}}
    if (dest / 'completed.json').exists():
        result.update(verify_matrix(root))
    result['logical_requests_completed_or_charged'] = sum(r.get('logical_requests', 0) for r in result['jobs'].values())
    result['maximum_logical_requests'] = 16384
    result['terminal_trajectories'] = sum(r['state'] in ('complete', 'failed') for r in result['jobs'].values())
    result['total_trajectories'] = 16
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'register', 'verify', 'prepare', 'run', 'status', 'matrix-verify'))
    parser.add_argument('--root', default='.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--chemistry-python')
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            result = policy()
        elif args.action in ('register', 'verify'):
            from . import section9_registration
            result = getattr(section9_registration, args.action)(args.root)
        elif args.action == 'prepare':
            result = prepare(args.root, execute=args.execute)
        elif args.action == 'run':
            result = run_matrix(args.root, execute=args.execute, resume=args.resume, chemistry_python=args.chemistry_python)
        elif args.action == 'status':
            result = status(args.root)
        else:
            result = verify_matrix(args.root)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
