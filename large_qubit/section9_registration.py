"""New implementation registration; never rewrites the Section 8 closure."""
from pathlib import Path
from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .section9_policy import REGISTRATION, PREDECESSOR, PREDECESSOR_SHA, SOURCES, policy
from .section9_operations_registration import source_hash


def predecessor(workspace):
    root = Path(workspace).resolve(strict=True)
    require(sha256_file(root / PREDECESSOR) == PREDECESSOR_SHA, 'Section8 registration changed')
    from .section8_registration import verify
    checked = verify(root)
    require(checked['all_passed'], 'Section8 implementation closure failed')
    return {'manifest_sha256': PREDECESSOR_SHA, 'scientific_closure_authenticated': True,
            'running_section8_sources_unchanged': True}


def specification(workspace):
    root = Path(workspace).resolve(strict=True)
    return {'schema_version': 'section9-implementation-registration-v1',
            'predecessor': predecessor(root), 'policy': policy(),
            'bindings_sha256': {n: source_hash(root, n) for n in SOURCES},
            'research_plan_sha256': sha256_file(root / 'research_plan_revised.md'),
            'implementation_prepared': True, 'main_matrix_complete': False,
            'training_executed': False, 'scientific_evaluation_executed': False,
            'execution_ready': False, 'requires_completed_section8_freeze': True}


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
    require(path.with_suffix('.sha256').read_text().strip() == sha, 'Section9 registration checksum changed')
    require(digest(read_json(path)) == digest(specification(root)), 'Section9 implementation or prior evidence changed')
    return {'all_passed': True, 'implementation_prepared': True, 'registration_sha256': sha,
            'main_matrix_complete': False, 'execution_ready': False,
            'scope': 'read_only_implementation_authentication'}
