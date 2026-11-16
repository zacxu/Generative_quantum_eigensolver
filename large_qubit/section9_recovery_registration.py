"""Immutable additive registration and eligibility check for one HF-only recovery."""
from pathlib import Path
from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .section9_operations_registration import source_hash
from .section9_recovery_policy import (REGISTRATION, PREDECESSOR, PREDECESSOR_SHA,
    FAILED_ROOT, FAILED_COMPLETION_SHA, SOURCES, TARGETS, lane_specs, job_id, policy)


def predecessor(workspace):
    root = Path(workspace).resolve(strict=True)
    require(sha256_file(root / PREDECESSOR) == PREDECESSOR_SHA,
            'Original four-GPU Section9 registration changed')
    from .section9_parallel_registration import verify as verify_original
    checked = verify_original(root)
    require(checked['all_passed'], 'Original four-GPU implementation closure failed')
    return {'manifest_sha256': PREDECESSOR_SHA, 'scientific_closure_authenticated': True,
            'original_section9_sources_unchanged': True}


def failed_campaign(workspace):
    """Authenticate preserved failure bytes without changing ledgers or loading models."""
    from .section8 import load
    from .section9_parallel import _inventory, validate_allocation
    from .section9_ledger import verify_ledger
    from .section9_reporting import verify_reference
    root = Path(workspace).resolve(strict=True)
    folder = workspace_path(root, FAILED_ROOT, namespace='large_qubit/main')
    completion = folder / 'completed.json'
    require(sha256_file(completion) == FAILED_COMPLETION_SHA,
            'Recovery requires the exact preserved failed campaign')
    saved = load(completion)
    require(saved['inventory'] == _inventory(root, folder), 'Preserved failed campaign inventory changed')
    require(saved['matrix_execution_complete'] is True
            and saved['all_trajectories_successful'] is False
            and saved['trajectories_accounted'] == 16
            and saved['training_executed'] is False and saved['pipeline_seeds'] == [22],
            'Recovery only accepts the sealed 16-trajectory failure')
    allocation = load(folder / 'gpu_allocation.json')
    validate_allocation(allocation)
    reference_sha = {}
    for alias in TARGETS:
        target = folder / 'targets' / alias
        physical, reference = load(target / 'physics_identity.json'), load(target / 'reference.json')
        require(reference['status'] == 'completed', 'Recovery requires both completed target references')
        identity = physical['cache_identity']
        require(identity['geometry_id'] == alias and identity['partition'] == 'final_test',
                'Preserved reference target identity changed')
        verify_reference(reference['result'], identity)
        reference_sha[alias] = sha256_file(target / 'reference.json')
    checked_jobs = []
    physical_cost = {'evaluator_calls': 0, 'kernel_contractions': 0,
                     'hamiltonian_action_seconds': 0.0, 'recorded_wall_seconds': 0.0}
    for lane in lane_specs():
        for method in lane['methods']:
            name = job_id(method, lane['target'])
            job = folder / 'lanes' / lane['lane_id'] / name
            report, identity = load(job / 'report.json'), load(job / 'identity.json')
            snapshot = verify_ledger(job / 'ledger.sqlite', identity,
                                     expected_head_sha256=report['ledger_head'])
            require(identity['method'] == method and identity['geometry_id'] == lane['target']
                    and identity['lane_id'] == lane['lane_id'], 'Preserved failure job identity changed')
            require(report['status'] == 'failed' and snapshot['logical_requests'] == 1
                    and snapshot['failed_requests'] == 1 and snapshot['completed_requests'] == 0
                    and snapshot['completion_attempts'] == 0 and snapshot['incumbent'] is None
                    and snapshot['pending_requests'] == 0 and snapshot['abandoned_requests'] == 0
                    and snapshot['pending_attempts'] == 0 and snapshot['attempts'] == []
                    and snapshot['cache_hits'] == 0 and snapshot['duplicate_requests'] == 0
                    and len(snapshot['requests']) == 1,
                    'Recovery eligibility requires exactly one failed HF request and no candidates per trajectory')
            require(all(report['counts'][key] == snapshot[key] for key in (
                        'logical_requests', 'failed_requests', 'completed_requests', 'completion_attempts',
                        'pending_requests', 'abandoned_requests', 'cache_hits', 'duplicate_requests'))
                    and report['logical_requests'] == snapshot['logical_requests']
                    and report['completion_attempts'] == 0,
                    'Preserved report counts differ from the authenticated failed ledger')
            request = snapshot['requests'][0]
            require(request['kind'] == 'hf' and request['status'] == 'failed'
                    and request['query'] == {'kind': 'hf', 'geometry_id': lane['target']}
                    and request['energy_hartree'] is None
                    and request['evaluation']['error'] == 'Target scorer exceeds the 4 GiB process RSS ceiling',
                    'Recovery requires the specific first-HF memory-guard failure')
            cost = request['evaluation']
            require(cost['evaluator_calls'] == 1 and cost['kernel_contractions'] == 1,
                    'Failed first-HF physical work accounting changed')
            for key in ('evaluator_calls', 'kernel_contractions', 'hamiltonian_action_seconds'):
                physical_cost[key] += cost[key]
            physical_cost['recorded_wall_seconds'] += cost['wall_seconds']
            checked_jobs.append(name)
    require(len(checked_jobs) == 16 and len(set(checked_jobs)) == 16,
            'Recovery requires all 16 distinct failed trajectories')
    return {'path': FAILED_ROOT, 'completed_sha256': FAILED_COMPLETION_SHA,
            'inventory_sha256': digest(saved['inventory']), 'trajectories': len(checked_jobs),
            'logical_requests': 16, 'completion_attempts': 0, 'successful_ordinary_requests': 0,
            'all_first_requests_hf': True, 'reference_sha256': reference_sha,
            'physical_cost_retained': physical_cost,
            'final_freeze_sha256': load(folder / 'binding.json')['final_freeze_sha256'],
            'allocation_sha256': sha256_file(folder / 'gpu_allocation.json'),
            'costs_sha256': sha256_file(workspace_path(root, saved['costs_path'], namespace=FAILED_ROOT))}


def specification(workspace):
    root = Path(workspace).resolve(strict=True)
    return {'schema_version': 'section9-four-gpu-infrastructure-recovery-registration-v1',
            'predecessor': predecessor(root), 'failed_campaign': failed_campaign(root),
            'policy': policy(),
            'bindings_sha256': {name: source_hash(root, name) for name in SOURCES},
            'research_plan_sha256': sha256_file(root / 'research_plan_revised.md'),
            'implementation_prepared': True, 'main_matrix_complete': False,
            'training_executed': False, 'scientific_evaluation_executed': False,
            'execution_ready': False, 'requires_completed_section8_freeze': True,
            'requires_exact_preserved_hf_only_failure': True}


def register(workspace):
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, REGISTRATION, namespace='large_qubit/registrations')
    if path.exists():
        return verify(root)
    value = specification(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n')
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha256_file(path) + '\n')
    return verify(root)


def verify(workspace):
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, REGISTRATION, namespace='large_qubit/registrations')
    sha = sha256_file(path)
    require(path.with_suffix('.sha256').read_text().strip() == sha,
            'Section9 recovery registration checksum changed')
    require(digest(read_json(path)) == digest(specification(root)),
            'Recovery implementation or preserved evidence changed')
    return {'all_passed': True, 'implementation_prepared': True, 'registration_sha256': sha,
            'main_matrix_complete': False, 'execution_ready': False,
            'scope': 'read_only_implementation_and_preserved_failure_authentication'}
