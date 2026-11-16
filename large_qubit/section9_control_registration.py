"""Additive execution-control registration; original scientific identities stay fixed."""
from pathlib import Path

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .section9_operations_registration import source_hash

REGISTRATION = 'large_qubit/registrations/sections_1_9_gpu_control_v1/manifest.json'
PREDECESSOR = 'large_qubit/registrations/sections_1_9_recovery_v1/manifest.json'
PREDECESSOR_SHA = '9509abe5569c71ea72bfeb7102488ab570b3d9b3bb5ec1217fbf1cfd2133ae72'
SOURCES = tuple('large_qubit/' + name for name in (
    'section9_control.py', 'section9_control_worker.py', 'section9_pause.py',
    'section9_control_registration.py', 'section9_gpu_control.md',
    'scripts/run_section9_control_tmux.sh',
    'tests/test_section9_control.py', 'tests/test_section9_control_worker.py',
    'tests/test_section9_pause.py', 'tests/test_section9_control_registration.py'))


def policy():
    """Only scheduling may change; this is not a new scientific campaign."""
    return {
        'schema_version': 'section9-gpu-execution-control-policy-v1',
        'scope': 'pause and change allocated GPUs within the existing recovery matrix',
        'authorization': 'researcher requested reassignment because allocated GPUs are under heavy load',
        'preserved_predecessor_sha256': PREDECESSOR_SHA,
        'output': 'large_qubit/main/section9_recovery_v1/matrix_0001',
        'scientific_identity': 'retain original preparation, lane, ledger, cache and frozen protocol identities',
        'unchanged': ['methods', 'targets', 'pipeline_seed', 'attempt_seeds', 'request_budgets',
                      'completion_attempt_budgets', 'model_weights', 'adapters', 'prompt_contexts',
                      'precision', 'decoding', 'CPU_scoring', 'reference_receipts'],
        'gpu_equivalence': 'four distinct H200 UUIDs; every runtime field except gpu_uuid must match the frozen environment',
        'allocation_provenance': 'append authenticated allocation epochs; preserve original allocation and runtime receipts',
        'pause_admission': 'no migration while an execution lease or uncertain physical request remains active',
        'legacy_takeover': 'stop only at verified quiescent frontiers; busy work continues until a safe boundary',
        'inflight_accounting': 'preserve all reservations, completed work, failures, charges and unknown interrupted setup costs',
        'completed_lanes': 'retain sealed outputs unchanged and skip their execution',
        'resume': 'continue existing durable ledgers and caches; never reset a trajectory or refill a failed budget',
        'reassignment_selection': 'GPU availability only; no outcome-based selection or candidate replacement',
        'execution_authorization': 'explicit researcher pause, reassign and resume commands',
        'training_executed': False,
    }


def predecessor(workspace):
    root = Path(workspace).resolve(strict=True)
    require(sha256_file(root / PREDECESSOR) == PREDECESSOR_SHA,
            'Original Section9 recovery registration changed')
    from .section9_recovery_registration import verify as verify_recovery
    checked = verify_recovery(root)
    require(checked['all_passed'] is True, 'Original Section9 recovery closure failed')
    return {'path': PREDECESSOR, 'manifest_sha256': PREDECESSOR_SHA,
            'scientific_closure_authenticated': True, 'original_sources_unchanged': True}


def specification(workspace):
    root = Path(workspace).resolve(strict=True)
    return {
        'schema_version': 'section9-gpu-execution-control-registration-v1',
        'predecessor': predecessor(root), 'policy': policy(),
        'bindings_sha256': {name: source_hash(root, name) for name in SOURCES},
        'research_plan_sha256': sha256_file(root / 'research_plan_revised.md'),
        'implementation_prepared': True, 'scientific_evaluation_executed': False,
        'training_executed': False, 'execution_ready': False,
        'requires_existing_recovery_matrix': True,
    }


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
    actual = sha256_file(path)
    require(path.with_suffix('.sha256').read_text().strip() == actual,
            'Section9 GPU control registration checksum changed')
    require(digest(read_json(path)) == digest(specification(root)),
            'GPU control implementation or original recovery evidence changed')
    return {'all_passed': True, 'implementation_prepared': True,
            'registration_sha256': actual, 'training_executed': False,
            'scientific_evaluation_executed': False, 'execution_ready': False,
            'scope': 'read_only_execution_control_and_predecessor_authentication'}
