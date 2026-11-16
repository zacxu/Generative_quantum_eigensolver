"""User-run compact validation and final protocol freeze. No implicit execution.

All production paths require --execute. Preparation needs the three completed
Section 7 fits; fixtures and implementation verification need neither GPU nor
scientific energies. Final target scoring is deliberately not a Section 8 action.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from importlib import metadata as package_metadata
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import uuid

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .protocol import METHODS, FITS, seed_for
from .section8_policy import (REGISTRATION, PREPARATION, OUTPUT_ROOT, FINAL_FREEZE,
    GEOMETRY, CATALOGUE, policy, attempt_seed)


def utc():
    return datetime.now(timezone.utc).isoformat()


def controller_runtime():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': {name: package_metadata.version(name) for name in
                ('numpy', 'torch', 'transformers', 'peft', 'safetensors', 'tokenizers')}}


def save(path, value):
    """No-clobber publication; incomplete writes stay separately identifiable."""
    path = Path(path)
    envelope = {'payload': value, 'sha256': digest(value)}
    if path.exists():
        require(load(path) == value, 'Preserve existing artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with pending.open('xb') as handle:
        handle.write(canonical_bytes(envelope) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(pending, path)  # Atomic no-overwrite publication on the same filesystem.
    pending.unlink()
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load(path):
    value = read_json(path)
    require(set(value) == {'payload', 'sha256'} and digest(value['payload']) == value['sha256'],
            'Artifact checksum differs: ' + str(path))
    return value['payload']


def training_bindings(root):
    from .section7 import status_section7, RUN_ROOT, PREFLIGHT_ROOT
    report = status_section7(root)
    require(report['training_complete'], 'Finish and authenticate all three Section7 fits before Section8 preparation')
    result = {}
    for fit in FITS:
        verified = report['fits'][fit]['verification']
        directory = RUN_ROOT + '/' + fit
        result[fit] = {'directory': directory, 'adapter_path': directory + '/checkpoints/final/adapter',
            'identity_sha256': verified['identity_sha256'],
            'final_checkpoint_sha256': verified['final_checkpoint_sha256'],
            'completed_sha256': sha256_file(root / directory / 'completed.json'),
            'counters': verified['counters'], 'cost_summary': verified['cost_summary']}
    return {'fits': result, 'numerical_freeze_sha256': sha256_file(root / PREFLIGHT_ROOT / 'freeze.json')}


def prepare(workspace, *, execute=False):
    require(execute is True, 'Preparation requires explicit --execute')
    root = Path(workspace).resolve(strict=True)
    from .section8_registration import verify
    registered = verify(root)
    trained = training_bindings(root)
    from .prompts import load_prompt_tokenizer, build_prompt
    catalogue = read_json(root / CATALOGUE)
    contexts = {}
    for key in ('gpt2_xl', 'qwen3_4b'):
        tokenizer = load_prompt_tokenizer(root, key, execute=True)
        contexts[key] = {alias: build_prompt(tokenizer, key, alias, catalogue, 32)['context_certificate']
                         for alias in (GEOMETRY, 'CO2_I01', 'CO2_I02')}
    # Only energy-free physical metadata enter the future target prompts.
    from .section7 import RELEASE_SHA256
    value = {'schema_version': 'section8-preparation-v1',
        'registration_sha256': registered['registration_sha256'], 'policy': policy(),
        'training': trained, 'contexts': contexts, 'release_sha256': RELEASE_SHA256,
        'catalogue_sha256': sha256_file(root / CATALOGUE),
        'model_weights_loaded': False, 'training_executed': False,
        'scientific_energy_requests': 0, 'final_target_energy_access': False}
    save(workspace_path(root, PREPARATION, namespace='large_qubit/preflights'), value)
    return {'all_passed': True, 'preparation_sha256': sha256_file(root / PREPARATION),
            'validation_complete': False, 'next': 'run user-controlled Section8 panel'}


def verify_preparation(workspace):
    root = Path(workspace).resolve(strict=True)
    from .section8_registration import verify
    registered = verify(root)
    value = load(root / PREPARATION)
    require(value['registration_sha256'] == registered['registration_sha256']
            and value['policy'] == policy() and value['training'] == training_bindings(root)
            and value['catalogue_sha256'] == sha256_file(root / CATALOGUE),
            'Section8 preparation, final adapters or fixed policy changed')
    return value


def panel_directory(root, panel):
    require(isinstance(panel, str) and re.fullmatch(r'panel_[0-9]{4}', panel),
            'Use a panel name such as panel_0001')
    return workspace_path(root, OUTPUT_ROOT + '/' + panel, namespace=OUTPUT_ROOT)


def ledger_identity(preparation_sha, method, cache_identity):
    require(method in METHODS, 'Unknown validation method')
    return {'schema_version': 'section8-trajectory-identity-v1',
        'preparation_sha256': preparation_sha, 'policy_sha256': digest(policy()),
        'method': method, 'geometry_id': GEOMETRY, 'cache_identity': cache_identity}


def verify_reference(reference, cache_identity):
    """Recheck saved numerical evidence without solving the Hamiltonian again."""
    report = reference['report']
    require(reference.get('geometry_id') == GEOMETRY and reference.get('partition') == 'validation'
            and reference.get('cache_identity') == cache_identity
            and report['all_passed'] is True and report['converged'] is True
            and report['independent_starts'] == 2 and len(report['runs']) == 2,
            'Validation reference identity or convergence evidence differs')
    energies = [run['energy_ha'] for run in report['runs']]
    require(all(math.isfinite(e) for e in energies)
            and math.isfinite(reference['energy_hartree'])
            and abs(reference['energy_hartree'] - min(energies)) <= 1e-12
            and max(energies) - min(energies) <= 1e-8
            and all(run['converged'] is True
                    and 0 <= run['complex128_residual_norm_ha'] <= 1e-8
                    and abs(run['norm_squared'] - 1.) <= 1e-10 for run in report['runs']),
            'Validation reference residual, normalization or independent-start agreement failed')


def technical_checks(snapshot, *, method, verification, reference_energy):
    """No favourable energy, validity-rate or method-ranking threshold."""
    records = snapshot['requests']
    completed = [r for r in records if r['status'] == 'completed']
    require(completed and records[0]['kind'] == 'hf', 'The first physical request must be HF')
    require(snapshot['logical_requests'] <= 128 and snapshot['completion_attempts'] <= 256,
            'Validation budget exceeded')
    require(not any(snapshot[k] for k in ('failed_requests', 'abandoned_requests', 'pending_requests')),
            'Failed or uncertain physical work is a technical NO-GO; preserve it and document a fresh panel')
    require(not snapshot['pending_attempts'], 'Generation frontier is incomplete')
    require(all(math.isfinite(r['energy_hartree']) for r in completed), 'Nonfinite energy')
    for record in completed:
        evaluation = record.get('evaluation', {})
        evidence = evaluation.get('origin_evaluation', evaluation)
        require(evidence.get('norm') is not None and abs(evidence['norm'] - 1.) <= 1e-10
                and evidence.get('populations') == [11, 11]
                and evidence.get('query_sha256') == digest(record['query'])
                and evidence.get('cache_identity') == snapshot['identity']['cache_identity'],
                'Physical request lacks matching norm, population or query evidence')
        if record.get('attempt_ordinal') is not None:
            attempt = snapshot['attempts'][record['attempt_ordinal'] - 1]
            require(attempt['status'] == 'completed' and attempt['valid']
                    and attempt['output']['ops'] == record['query'].get('ops')
                    and attempt['input']['seed'] == attempt_seed(method, GEOMETRY, attempt['ordinal']),
                    'Scored circuit differs from its recorded deterministic generation attempt')
    best = min(completed, key=lambda r: (r['energy_hartree'], r['ordinal']))
    require(verification.get('query_sha256') == digest(best['query'])
            and verification.get('cache_identity') == snapshot['identity']['cache_identity']
            and verification.get('fresh_basis_and_evaluator') is True
            and verification.get('populations') == [11, 11]
            and abs(verification.get('norm', 0.) - 1.) <= 1e-10,
            'Winner verification lacks fresh matching physical evidence')
    require(abs(best['energy_hartree'] - verification['energy_hartree']) <= 1e-8,
            'Independent winning-circuit reconstruction disagrees')
    require(best['energy_hartree'] >= reference_energy - 1e-8,
            'Candidate lies below the resolved same-Hamiltonian reference beyond tolerance')
    if method == 'conventional_vqe':
        require(snapshot['logical_requests'] == 123, 'Validation VQE must use the frozen truncated restart schedule')
    elif method in ('uniform_random', 'training_circuit_retrieval'):
        require(snapshot['logical_requests'] == 128, 'Discrete baseline must exhaust its request allowance')
    else:
        require(snapshot['logical_requests'] == 128 or snapshot['completion_attempts'] == 256,
                'Generator stopped before its request or attempt cap')
    incumbent = float('inf')
    curve = []
    for record in records:
        incumbent = min(incumbent, record['energy_hartree'])
        curve.append({'logical_requests': record['ordinal'], 'best_energy_hartree': incumbent})
    return {'all_passed': True, 'method': method, 'best_request': best['ordinal'],
        'best_query': best['query'], 'best_energy_hartree': best['energy_hartree'],
        'reference_energy_hartree': reference_energy,
        'best_error_hartree': best['energy_hartree'] - reference_energy,
        'hf_improvement_hartree': records[0]['energy_hartree'] - best['energy_hartree'],
        'logical_requests': snapshot['logical_requests'], 'unused_allowance': 128 - snapshot['logical_requests'],
        'completion_attempts': snapshot['completion_attempts'], 'cache_hits': snapshot['cache_hits'],
        'duplicates': snapshot['duplicate_requests'], 'ledger_head': snapshot['head_sha256'],
        'budget_curve': curve, 'endpoint_carried_to_budget_128': True,
        'valid_completion_attempts': sum(a['status'] == 'completed' and a['valid'] for a in snapshot['attempts']),
        'invalid_or_abandoned_completion_attempts': sum(a['status'] != 'completed' or not a['valid'] for a in snapshot['attempts']),
        'negative_outcomes_retained': True, 'performance_threshold_applied': False}


def execute_trajectory(ledger, scorer, method, *, catalogue, generator=None, retrieval=None,
                       initial_angles=None):
    """Common budgeted admission engine. Injectable components enable CPU fixtures."""
    before = ledger.snapshot()
    if before['pending_requests'] or before['pending_attempts']:
        ledger.recover_pending('process interrupted before durable result')
    state = ledger.snapshot()
    require(not state['failed_requests'] and not state['abandoned_requests'],
            'Uncertain/failed physical requests require a documented fresh panel; no automatic rescore')
    scorer.submit('hf', {'kind': 'hf'}, kind='hf').result()
    if method in METHODS[:5]:
        require(generator is not None, 'A resident generator is required')
        pending = []
        for ordinal in range(1, 257):
            state = ledger.snapshot()
            if state['logical_requests'] >= 128:
                break
            previous = {a['ordinal']: a for a in state['attempts']}.get(ordinal)
            if previous is not None:
                if previous['status'] != 'completed' or not previous['valid']:
                    continue
                output = previous['output']
            else:
                seed = attempt_seed(method, GEOMETRY, ordinal)
                actual = ledger.reserve_attempt(f'attempt_{ordinal:04d}', {'seed': seed})
                require(actual['ordinal'] == ordinal, 'Attempt frontier changed')
                try:
                    output = generator.generate(GEOMETRY, catalogue, seed)
                except Exception as error:
                    ledger.finish_attempt(ordinal, {'error': str(error)}, valid=False,
                                          failure_reason='inference failure')
                    raise
                ledger.finish_attempt(ordinal, output, valid=output['valid'],
                                      failure_reason=None if output['valid'] else 'invalid complete answer')
                require(output.get('error_kind') != 'infrastructure',
                        'Inference infrastructure failure recorded; inspect the error before explicit resume')
            if output['valid']:
                pending.append(scorer.submit(f'attempt_{ordinal:04d}',
                    {'kind': 'discrete', 'ops': output['ops']}, attempt_ordinal=ordinal))
                if len(pending) >= 2:
                    pending.pop(0).result()
        for future in pending:
            future.result()
    elif method == 'uniform_random':
        from .baselines import uniform_random
        proposals = uniform_random(catalogue, length=32, count=127,
            seed=seed_for('generation', group=f'{method}:{GEOMETRY}'))['circuits']
        for i, ops in enumerate(proposals):
            scorer.submit(f'random_{i:04d}', {'kind': 'discrete', 'ops': ops}).result()
    elif method == 'training_circuit_retrieval':
        require(retrieval is not None and len(retrieval) == 127, 'Authenticated retrieval source required')
        for i, row in enumerate(retrieval):
            scorer.submit(f'retrieval_{i:04d}', {'kind': 'discrete', 'ops': row['ops']}).result()
    elif method == 'conventional_vqe':
        from .vqe import optimize_compact
        require(initial_angles is not None and len(initial_angles) == 24, 'Pinned training-only VQE vector required')
        for restart, (cap, initial) in enumerate(zip((63, 64), ([0.] * 24, initial_angles))):
            def evaluate(parameters, metadata):
                return scorer.submit(f'vqe_{restart}_{metadata["local_request"]:04d}',
                    {'kind': 'vqe', 'parameters': parameters}).result()['energy_hartree']
            optimize_compact(evaluate, initial, max_requests=cap)
    else:
        raise ValueError('Unknown method')
    return ledger.snapshot()


def _new_attempt(directory):
    base = directory / 'execution_attempts'
    base.mkdir(exist_ok=True)
    for ordinal in range(1, 10000):
        target = base / f'attempt_{ordinal:04d}'
        try:
            target.mkdir()
            save(target / 'started.json', {'utc': utc(), 'pid': os.getpid()})
            return target
        except FileExistsError:
            continue
    raise ValueError('Execution attempt counter exhausted')


def run_panel(workspace, *, panel='panel_0001', chemistry_python=None,
              execute=False, resume=False, rerun_reason=None):
    require(execute is True, 'Scientific validation requires explicit --execute')
    require(platform.system() == 'Linux', 'Run validation in the pinned Linux server environments')
    root = Path(workspace).resolve(strict=True)
    prepared = verify_preparation(root)  # Refuses before GPU/model access if any fit is unfinished.
    require(not (root / FINAL_FREEZE).exists(), 'Final protocol already frozen; preserve the sealed panel')
    directory = panel_directory(root, panel)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from .training import _run_lock
    from .section7 import _gpu_operation_lock
    with _run_lock(root / 'large_qubit/operations/section8_v1/panel'), _gpu_operation_lock(root, 'cuda:0'):
        existed = directory.exists()
        require(not existed or resume, 'Existing panel requires --resume; never overwrite validation')
        earlier = sorted(p.name for p in (root / OUTPUT_ROOT).glob('panel_[0-9][0-9][0-9][0-9]')
                         if p.is_dir() and p.name != panel)
        require(not earlier or panel > earlier[-1],
                'A later panel has superseded this one; preserve earlier panels as read-only evidence')
        require(existed or not earlier or (isinstance(rerun_reason, str) and len(rerun_reason.strip()) >= 12),
                'A fresh rerun requires --rerun-reason describing the infrastructure incident')
        if not existed:
            expected_number = int(earlier[-1].split('_')[1]) + 1 if earlier else 1
            require(panel == f'panel_{expected_number:04d}', 'Use consecutive panel IDs; preserve earlier attempts')
            for prior in earlier:
                if (panel_directory(root, prior) / 'completed.json').exists():
                    verify_panel(root, prior, check_preparation=False)
                    raise ValueError('A technically successful panel already exists; freeze it. '
                                     'Scientific redesign requires a separately registered version')
        directory.mkdir(parents=True, exist_ok=True)
        binding = {'preparation_sha256': sha256_file(root / PREPARATION), 'policy': policy(),
            'panel': panel, 'superseded_panels': earlier, 'rerun_reason': rerun_reason,
            'geometry_id': GEOMETRY, 'final_target_access': False, 'controller_runtime': controller_runtime()}
        if existed:
            saved = load(directory / 'binding.json')
            require(saved['preparation_sha256'] == binding['preparation_sha256']
                    and saved['policy'] == policy()
                    and saved['controller_runtime'] == binding['controller_runtime'], 'Panel identity or controller runtime changed')
            binding = saved
        else:
            save(directory / 'binding.json', binding)
        if (directory / 'completed.json').exists():
            return verify_panel(root, panel, check_preparation=False)
        attempt = _new_attempt(directory)
        started = time.monotonic()
        from .section8_scoring import PhysicsClient
        from .section8_inference import ResidentGenerator
        from .section8_ledger import TrajectoryLedger, ScalarCache, OrderedScorer
        chemistry_python = chemistry_python or str(root / '.venv-chemistry/bin/python')
        generator = None
        try:
            with PhysicsClient(root, chemistry_python, execute=True, threads=1) as physics:
                save(directory / 'physics_identity.json', physics.identity)
                cache_identity = physics.identity['cache_identity']
                reference_path = directory / 'reference.json'
                if not reference_path.exists():
                    marker = directory / 'reference_started.json'
                    require(not marker.exists(), 'Interrupted reference work is preserved; document a fresh panel')
                    save(marker, {'utc': utc(), 'cost_category': 'reference_overhead'})
                    save(reference_path, physics.reference())
                reference = load(reference_path)
                verify_reference(reference, cache_identity)
                reference_energy = reference['energy_hartree']
                catalogue = read_json(root / CATALOGUE)
                cache = ScalarCache(directory / 'scalar_cache.sqlite')
                results = {}
                for method in METHODS:
                    folder = directory / method
                    folder.mkdir(exist_ok=True)
                    identity = ledger_identity(binding['preparation_sha256'], method, cache_identity)
                    save(folder / 'identity.json', identity)
                    ledger = TrajectoryLedger(folder / 'ledger.sqlite', identity,
                        request_budget=128, attempt_budget=256)
                    current = ledger.snapshot()
                    if current['pending_requests'] or current['pending_attempts']:
                        ledger.recover_pending('controller restarted after interrupted work')
                    require(not ledger.snapshot()['abandoned_requests'],
                            'Interrupted physical work was charged; document a fresh panel')
                    if method in METHODS[:5] and not (folder / 'report.json').exists():
                        key = 'gpt2_xl' if method.startswith('gpt2') else 'qwen3_4b'
                        if generator is None or generator.model_key != key:
                            if generator is not None:
                                save(attempt / (generator.model_key + '_inference_cost.json'), generator.statistics())
                                generator.close()
                                generator = None
                            generator = ResidentGenerator(root, key, device='cuda:0', execute=True)
                            save(directory / (key + '_runtime.json'), generator.metadata['runtime_identity'])
                            save(attempt / (key + '_inference_metadata.json'), generator.metadata)
                        adapter = prepared['training']['fits'].get(method, {}).get('adapter_path')
                        generator.switch(method, adapter_path=adapter)
                    retrieval = None
                    angles = None
                    if method == 'training_circuit_retrieval':
                        from .corpus_v2 import load_release
                        from .baselines import retrieval_candidates
                        release = load_release(root, consumer_id=method,
                            expected_release_sha256=prepared['release_sha256'])
                        source = retrieval_candidates(GEOMETRY, list(release.retrieval_rows()),
                            catalogue=catalogue, length=32, acquisition_cost=release.source_acquisition_cost,
                            max_candidates=127)
                        save(folder / 'retrieval_source.json', source)
                        retrieval = source['candidates']
                    elif method == 'conventional_vqe':
                        from .section5 import SCREENING_PATH
                        angles = read_json(root / SCREENING_PATH)['vqe_initial_angles']
                    with OrderedScorer(ledger, cache, cache_identity, physics.score, max_pending=2) as scorer:
                        if not (folder / 'report.json').exists():
                            execute_trajectory(ledger, scorer, method, catalogue=catalogue,
                                generator=generator, retrieval=retrieval, initial_angles=angles)
                    snapshot = ledger.snapshot()
                    complete = [r for r in snapshot['requests'] if r['status'] == 'completed']
                    require(complete, 'Missing HF result')
                    best = min(complete, key=lambda r: (r['energy_hartree'], r['ordinal']))
                    if not (folder / 'winner_verification.json').exists():
                        marker = folder / 'verification_started.json'
                        require(not marker.exists(), 'Interrupted winner check retained; document a fresh panel')
                        save(marker, {'utc': utc(), 'cost_category': 'independent_integrity_overhead'})
                        save(folder / 'winner_verification.json', physics.verify(best['query']))
                    checked = technical_checks(snapshot, method=method,
                        verification=load(folder / 'winner_verification.json'), reference_energy=reference_energy)
                    save(folder / 'report.json', checked)
                    results[method] = checked
                    if hasattr(ledger, 'close'):
                        ledger.close()
                if generator is not None:
                    save(attempt / (generator.model_key + '_inference_cost.json'), generator.statistics())
                    generator.close()
                    generator = None
                cache_report = cache.verify()
                if hasattr(cache, 'close'):
                    cache.close()
                save(attempt / 'finished.json', {'utc': utc(), 'success': True,
                    'wall_seconds': time.monotonic() - started})
                summary = {'schema_version': 'section8-validation-completion-v1',
                    'panel': panel, 'all_passed': True, 'methods': results,
                    'logical_requests': sum(r['logical_requests'] for r in results.values()),
                    'maximum_requests': 1024, 'validation_complete': True,
                    'reference_sha256': sha256_file(reference_path),
                    'scalar_cache_sha256': cache_report['sha256'],
                    'binding_sha256': sha256_file(directory / 'binding.json'),
                    'performance_selection': False, 'training_executed': False,
                    'final_target_energy_access': False}
                save(directory / 'completed.json', summary)
            return verify_panel(root, panel, check_preparation=False)
        except BaseException as error:
            if generator is not None:
                save(attempt / (generator.model_key + '_failed_inference_cost.json'), generator.statistics())
                generator.close()
            save(attempt / 'failed.json', {'utc': utc(), 'error': str(error),
                'scientific_failure_report': getattr(error, 'report', None),
                'wall_seconds': time.monotonic() - started,
                'uncertain_work': 'see durable request/attempt reservations and reference/integrity start records'})
            raise


def verify_panel(workspace, panel='panel_0001', *, check_preparation=True):
    root = Path(workspace).resolve(strict=True)
    if check_preparation:
        verify_preparation(root)
    directory = panel_directory(root, panel)
    binding = load(directory / 'binding.json')
    require(binding['policy'] == policy() and binding['preparation_sha256'] == sha256_file(root / PREPARATION),
            'Panel preparation or policy changed')
    completed = load(directory / 'completed.json')
    require(completed['binding_sha256'] == sha256_file(directory / 'binding.json')
            and completed['reference_sha256'] == sha256_file(directory / 'reference.json'),
            'Completed panel bindings changed')
    reference = load(directory / 'reference.json')
    from .section8_ledger import verify_ledger, ScalarCache
    cache_identity = load(directory / 'physics_identity.json')['cache_identity']
    verify_reference(reference, cache_identity)
    results = {}
    for method in METHODS:
        folder = directory / method
        identity = ledger_identity(binding['preparation_sha256'], method, cache_identity)
        require(load(folder / 'identity.json') == identity, 'Trajectory identity changed')
        report = load(folder / 'report.json')
        snapshot = verify_ledger(folder / 'ledger.sqlite', identity,
            request_budget=128, attempt_budget=256, expected_head_sha256=report['ledger_head'])
        checked = technical_checks(snapshot, method=method,
            verification=load(folder / 'winner_verification.json'), reference_energy=reference['energy_hartree'])
        require(report == checked and completed['methods'][method] == checked, 'Method report differs from ledger')
        results[method] = checked
    require(completed['all_passed'] is True and set(completed['methods']) == set(METHODS)
            and completed['logical_requests'] == sum(r['logical_requests'] for r in results.values()) <= 1024,
            'Panel aggregate is incomplete or exceeds its allowance')
    ScalarCache(directory / 'scalar_cache.sqlite', create=False, read_only=True).verify(
        expected_sha256=completed['scalar_cache_sha256'])
    return {'all_passed': True, 'validation_complete': True, 'panel': panel,
        'logical_requests': completed['logical_requests'], 'methods': results,
        'completed_sha256': sha256_file(directory / 'completed.json'), 'final_protocol_frozen': False}


def _panel_inventory(root):
    """Bind every prior failed/replayed attempt as incurred development expenditure."""
    inventory = {}
    for path in sorted((root / OUTPUT_ROOT).rglob('*')):
        if path.is_file() and not path.name.endswith(('-shm', '-wal')):
            require(not path.is_symlink(), 'Panel inventory cannot contain symbolic links')
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    return inventory


def validation_costs(root):
    """Count incurred requests in every retained panel, including failed reruns."""
    from .section8_ledger import verify_ledger
    panels = {}
    for directory in sorted((root / OUTPUT_ROOT).glob('panel_[0-9][0-9][0-9][0-9]')):
        row = {'logical_requests': 0, 'completion_attempts': 0,
            'failed_requests': 0, 'abandoned_requests': 0, 'pending_requests': 0,
            'cache_hits': 0, 'physical_hamiltonian_actions': 0,
            'physical_costs_with_unknown_portions': 0,
            'recorded_scoring_seconds': 0., 'recorded_generation_seconds': 0.,
            'reference_hamiltonian_actions': 0, 'winner_verification_hamiltonian_actions': 0,
            'recorded_reference_seconds': 0., 'recorded_winner_verification_seconds': 0.,
            'incomplete_overhead_records': [], 'execution_attempts': []}
        for method in METHODS:
            folder = directory / method
            if not (folder / 'identity.json').exists():
                continue
            state = verify_ledger(folder / 'ledger.sqlite', load(folder / 'identity.json'))
            for key in ('logical_requests', 'completion_attempts', 'failed_requests',
                        'abandoned_requests', 'pending_requests', 'cache_hits'):
                row[key] += state[key]
            for request in state['requests']:
                evaluation = request.get('evaluation') or {}
                row['physical_hamiltonian_actions'] += evaluation.get('evaluator_calls', 0)
                row['recorded_scoring_seconds'] += evaluation.get('wall_seconds', 0.)
                if request['status'] != 'completed':
                    row['physical_costs_with_unknown_portions'] += 1
            for attempt in state['attempts']:
                row['recorded_generation_seconds'] += (attempt.get('output') or {}).get('wall_seconds', 0.)
            if (folder / 'winner_verification.json').exists():
                check = load(folder / 'winner_verification.json')
                row['winner_verification_hamiltonian_actions'] += check.get('evaluator_calls', 0)
                row['recorded_winner_verification_seconds'] += check.get('wall_seconds', 0.)
            elif (folder / 'verification_started.json').exists():
                row['incomplete_overhead_records'].append(method + '/verification_started.json')
        if (directory / 'reference.json').exists():
            reference = load(directory / 'reference.json')
            row['reference_hamiltonian_actions'] = reference['report'].get('hamiltonian_actions', 0)
            row['recorded_reference_seconds'] = reference['report'].get('wall_seconds', 0.)
        elif (directory / 'reference_started.json').exists():
            row['incomplete_overhead_records'].append('reference_started.json')
        for attempt in sorted((directory / 'execution_attempts').glob('attempt_*')):
            receipt = {'path': attempt.relative_to(root).as_posix(), 'status': 'interrupted'}
            for filename in ('finished.json', 'failed.json'):
                if (attempt / filename).exists():
                    receipt.update(load(attempt / filename))
                    receipt['status'] = filename[:-5]
            receipt['inference_costs'] = [load(p) for p in sorted(attempt.glob('*inference_cost.json'))]
            partial = receipt.get('scientific_failure_report') or {}
            if (not (directory / 'reference.json').exists()
                    and partial.get('cost_category') == 'reference_solver_overhead'):
                row['reference_hamiltonian_actions'] += partial.get('hamiltonian_actions', 0)
                row['recorded_reference_seconds'] += partial.get('wall_seconds', 0.)
            row['execution_attempts'].append(receipt)
        panels[directory.name] = row
    from .section7 import RELEASE_SHA256
    from .corpus_v2 import load_release
    release = load_release(root, consumer_id='training_circuit_retrieval', expected_release_sha256=RELEASE_SHA256)
    return {'panels': panels, 'total_validation_logical_requests': sum(p['logical_requests'] for p in panels.values()),
        'source_acquisition_charged_once': release.source_acquisition_cost,
        'training_costs': load(root / PREPARATION)['training'],
        'scope': 'all retained validation attempts plus distinct source and training costs; unknown interrupted work is explicit',
        'physical_cost_totals_are_known_lower_bounds_when_work_is_incomplete': True,
        'overlapping_times_must_not_be_summed_as_elapsed_wall': True}


def freeze(workspace, *, panel='panel_0001', execute=False):
    require(execute is True, 'Final freeze requires explicit --execute')
    root = Path(workspace).resolve(strict=True)
    from .training import _run_lock
    with _run_lock(root / 'large_qubit/operations/section8_v1/panel'):
        if (root / FINAL_FREEZE).exists():
            return verify_final_freeze(root)
        panels = sorted(p.name for p in (root / OUTPUT_ROOT).glob('panel_[0-9][0-9][0-9][0-9]') if p.is_dir())
        require(panels and panels[-1] == panel, 'Freeze the latest declared panel; no performance-based panel selection')
        verified = verify_panel(root, panel)
        prepared = load(root / PREPARATION)
        value = {'schema_version': 'section8-final-experiment-freeze-v1',
            'section_8_complete': True, 'final_protocol_frozen': True,
            'selected_panel': panel, 'validation': verified,
            'registration_sha256': sha256_file(root / REGISTRATION),
            'preparation_sha256': sha256_file(root / PREPARATION),
            'protocol': policy(), 'final_adapters_and_training': prepared['training'],
            'prompt_contexts': prepared['contexts'], 'all_validation_artifacts': _panel_inventory(root),
            'costs': validation_costs(root),
            'all_prior_panel_costs_retained': True, 'final_target_energy_access': False,
            'execution_ready': False,
            'remaining_section9_work': ['final-target references isolated from generators',
                'explicit main-matrix launch and analysis using the frozen protocol'],
            'scientific_scope': 'one seed; two fixed targets; no convergence or positive-performance requirement'}
        save(workspace_path(root, FINAL_FREEZE, namespace='large_qubit/freezes'), value)
        return verify_final_freeze(root)


def verify_final_freeze(workspace):
    root = Path(workspace).resolve(strict=True)
    value = load(root / FINAL_FREEZE)
    prepared = verify_preparation(root)
    require(value['registration_sha256'] == sha256_file(root / REGISTRATION)
            and value['preparation_sha256'] == sha256_file(root / PREPARATION)
            and value['protocol'] == policy() and value['all_validation_artifacts'] == _panel_inventory(root),
            'Frozen protocol, inputs or validation history changed')
    require(value['validation'] == verify_panel(root, value['selected_panel'], check_preparation=False),
            'Frozen panel changed')
    require(value['costs'] == validation_costs(root), 'Frozen cumulative costs changed')
    require(value['final_adapters_and_training'] == prepared['training']
            and value['prompt_contexts'] == prepared['contexts']
            and value['all_prior_panel_costs_retained'] is True
            and value['final_target_energy_access'] is False
            and value['execution_ready'] is False, 'Frozen completion scope or input bindings changed')
    require(value['section_8_complete'] is True and value['final_protocol_frozen'] is True,
            'Section8 completion flags missing')
    return {'all_passed': True, 'section_8_complete': True, 'final_protocol_frozen': True,
            'freeze_sha256': sha256_file(root / FINAL_FREEZE), 'execution_ready': False,
            'scope': 'Section8 complete; no final target execution authorized by this verifier'}


def status(workspace, panel='panel_0001'):
    root = Path(workspace).resolve(strict=True)
    directory = panel_directory(root, panel)
    from .section8_ledger import verify_ledger
    result = {'implementation_prepared': (root / REGISTRATION).exists(),
        'preparation_exists': (root / PREPARATION).exists(), 'panel': panel,
        'validation_complete': False, 'final_protocol_frozen': False,
        'scope': 'read_only_no_model_loading_or_energy_requests', 'methods': {}}
    for method in METHODS:
        folder = directory / method
        if not (folder / 'identity.json').exists():
            result['methods'][method] = {'state': 'not_run'}
            continue
        s = verify_ledger(folder / 'ledger.sqlite', load(folder / 'identity.json'),
                          request_budget=128, attempt_budget=256)
        result['methods'][method] = {'state': 'complete' if (folder / 'report.json').exists() else 'partial',
            **{k: s[k] for k in ('logical_requests', 'completion_attempts', 'completed_requests',
                'failed_requests', 'abandoned_requests', 'pending_requests')}}
    if (directory / 'completed.json').exists():
        result['validation_complete'] = verify_panel(root, panel)['all_passed']
    if (root / FINAL_FREEZE).exists():
        result['final_protocol_frozen'] = verify_final_freeze(root)['all_passed']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'verify', 'prepare', 'run', 'status', 'freeze', 'freeze-verify'), nargs='?', default='plan')
    parser.add_argument('--root', default='.')
    parser.add_argument('--panel', default='panel_0001')
    parser.add_argument('--chemistry-python')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rerun-reason')
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            value = {'policy': policy(), 'training_or_validation_started': False,
                     'first_scientific_action': 'run --execute after prepare --execute'}
        elif args.action == 'verify':
            from .section8_registration import verify
            value = verify(args.root)
        elif args.action == 'prepare':
            value = prepare(args.root, execute=args.execute)
        elif args.action == 'run':
            value = run_panel(args.root, panel=args.panel, chemistry_python=args.chemistry_python,
                execute=args.execute, resume=args.resume, rerun_reason=args.rerun_reason)
        elif args.action == 'status':
            value = status(args.root, args.panel)
        elif args.action == 'freeze':
            value = freeze(args.root, panel=args.panel, execute=args.execute)
        else:
            value = verify_final_freeze(args.root)
        print(json.dumps(value, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
