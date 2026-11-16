"""Additive implementation registration, preserving the running Section 7."""
from pathlib import Path
from .common import canonical_bytes, read_json, require, sha256_file, workspace_path, digest
from .section8_policy import (REGISTRATION, PREDECESSOR, PREDECESSOR_SHA,
    ARCHIVE, SOURCES, policy)


def predecessor(workspace):
    root = Path(workspace).resolve(strict=True)
    require(sha256_file(root / PREDECESSOR) == PREDECESSOR_SHA, 'Section7 registration changed')
    saved = read_json(root / PREDECESSOR)
    changes = {}
    for name, expected in saved['bindings_sha256'].items():
        archived = workspace_path(root, ARCHIVE + '/' + name, namespace=ARCHIVE)
        require(sha256_file(archived) == expected, 'Registered Section7 archive changed: ' + name)
        actual = sha256_file(workspace_path(root, name))
        if actual != expected:
            require(name == 'large_qubit/section7.md', 'Running Section7 input changed: ' + name)
            changes[name] = {'registered': expected, 'current': actual}
    # Reauthenticate the inherited scientific closure; the only live documentary
    # difference is separately bound above. Never rewrite the old registration.
    from .documentation_registration import verify_predecessor_registration
    proof = verify_predecessor_registration(root)
    require(proof['all_passed'], 'Prior scientific closure failed')
    return {'manifest_sha256': PREDECESSOR_SHA, 'archive': ARCHIVE,
            'archived_bindings': len(saved['bindings_sha256']), 'documentation_changes': changes,
            'scientific_closure_authenticated': True}


def specification(workspace):
    root = Path(workspace).resolve(strict=True)
    return {'schema_version': 'section8-implementation-registration-v1',
        'predecessor': predecessor(root), 'policy': policy(),
        'bindings_sha256': {name: sha256_file(workspace_path(root, name)) for name in SOURCES},
        'research_plan_sha256': sha256_file(root / 'research_plan_revised.md'),
        'implementation_prepared': True, 'validation_complete': False,
        'training_executed': False, 'scientific_validation_executed': False,
        'final_experiment_frozen': False, 'execution_ready': False}


def register(workspace):
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, REGISTRATION, namespace='large_qubit/registrations')
    if path.exists():
        return verify(workspace)
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
    require(path.with_suffix('.sha256').read_text().strip() == actual, 'Section8 registration checksum changed')
    require(digest(read_json(path)) == digest(specification(root)), 'Section8 implementation or prior evidence changed')
    return {'all_passed': True, 'implementation_prepared': True,
            'registration_sha256': actual, 'validation_complete': False,
            'execution_ready': False, 'scope': 'read_only_implementation_authentication'}
