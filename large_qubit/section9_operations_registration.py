"""Authenticate operational fixes and archived guides without changing science.

The four original manifests remain byte-for-byte fixed. Their explicitly
superseded files are checked from exact archives; all other original sources
are still checked at their live paths. Replacement executable sources have a
separate immutable registration. The current human runbook is intentionally
outside the executable closure.
"""
from pathlib import Path

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path

REGISTRATION = 'large_qubit/registrations/sections_1_9_operations_v2/manifest.json'
ARCHIVE = 'large_qubit/archives/sections_1_9_operations_v2/original'
HISTORICAL_MANIFESTS = {
    'large_qubit/registrations/sections_1_9_v1/manifest.json':
        '7ea4123cd837d9db69159be132a5bff94d53923b16e32999957151eb20745091',
    'large_qubit/registrations/sections_1_9_parallel_v1/manifest.json':
        'fa162771755f41bec3a76438c1f23207fb6bdb519968834bdd97e09f5800ff45',
    'large_qubit/registrations/sections_1_9_recovery_v1/manifest.json':
        '9509abe5569c71ea72bfeb7102488ab570b3d9b3bb5ec1217fbf1cfd2133ae72',
    'large_qubit/registrations/sections_1_9_gpu_control_v1/manifest.json':
        '64d0f7589147d4f8f44a6442a712b1711716c2aa5d1dabd987d4820356b23a91',
}
ARCHIVED_SOURCES = tuple('large_qubit/' + name for name in (
    'section9.md', 'section9_parallel.md', 'section9_recovery.md', 'section9_gpu_control.md',
    'section9_registration.py', 'section9_parallel_registration.py',
    'section9_recovery_registration.py', 'section9_control_registration.py',
    'section9_pause.py', 'scripts/run_section9_control_tmux.sh', 'tests/test_section9_pause.py'))
SOURCES = tuple('large_qubit/' + name for name in (
    'section9_operations_registration.py', 'section9_registration.py',
    'section9_parallel_registration.py', 'section9_recovery_registration.py',
    'section9_control_registration.py', 'section9_pause.py',
    'scripts/run_section9_control_tmux.sh', 'tests/test_section9_pause.py',
    'tests/test_section9_operations_registration.py', 'tests/test_section9_operations_launcher.py',
    'section9_update.py', 'tests/test_section9_update.py', 'scripts/install_section9_update_tmux.sh'))


def policy():
    return {
        'scope': 'correct safe pause waiting and consolidate the active Section 9 runbook',
        'historical_manifests': 'retain all four original manifest bytes and hashes',
        'historical_bindings': 'resolve only the explicit archived source allowlist',
        'unchanged_sources': 'authenticate all other historical sources at their live paths',
        'replacement_sources': 'authenticate active executable replacements under this amendment',
        'active_runbook': 'large_qubit/section9.md',
        'active_runbook_hash_bound': False,
        'scientific_protocol': 'unchanged; no new campaign, quota refill or result replacement',
        'runtime_identities': 'preserve original registration, preparation, allocation epoch and ledger identities',
        'operation_provenance': 'new pause and launcher receipts identify this amendment hash',
        'archived_code_execution': False,
    }


def _historical_sources(root):
    """Check archives and unchanged live sources directly, without verifier recursion."""
    archived = {}
    live = {}
    plan_sha = sha256_file(root / 'research_plan_revised.md')
    for name, expected_manifest_sha in HISTORICAL_MANIFESTS.items():
        path = workspace_path(root, name, namespace='large_qubit/registrations')
        require(sha256_file(path) == expected_manifest_sha,
                'Historical Section9 registration changed: ' + name)
        require(path.with_suffix('.sha256').read_text().strip() == expected_manifest_sha,
                'Historical Section9 registration checksum changed: ' + name)
        saved = read_json(path)
        require(saved['research_plan_sha256'] == plan_sha, 'Registered research plan changed')
        for source, expected_source_sha in saved['bindings_sha256'].items():
            if source in ARCHIVED_SOURCES:
                require(source not in archived, 'Historical archive source has multiple owners: ' + source)
                archive_name = ARCHIVE + '/' + source
                archive_path = workspace_path(root, archive_name, namespace=ARCHIVE)
                require(sha256_file(archive_path) == expected_source_sha,
                        'Historical source archive changed: ' + source)
                archived[source] = {'path': archive_name, 'sha256': expected_source_sha,
                                    'registration': name}
            else:
                require(source not in live or live[source] == expected_source_sha,
                        'Historical live source has conflicting registrations: ' + source)
                require(sha256_file(workspace_path(root, source, namespace='large_qubit')) == expected_source_sha,
                        'Unchanged historical source changed at its live path: ' + source)
                live[source] = expected_source_sha
    require(set(archived) == set(ARCHIVED_SOURCES), 'Historical source archive allowlist is incomplete')
    return archived, live, plan_sha


def specification(workspace):
    root = Path(workspace).resolve(strict=True)
    archived, live, plan_sha = _historical_sources(root)
    return {
        'schema_version': 'section9-operations-amendment-v2',
        'historical_manifest_sha256': dict(HISTORICAL_MANIFESTS),
        'archived_bindings': archived, 'unchanged_live_bindings_sha256': live,
        'bindings_sha256': {name: sha256_file(workspace_path(root, name, namespace='large_qubit'))
                           for name in SOURCES},
        'research_plan_sha256': plan_sha, 'policy': policy(),
        'implementation_prepared': True, 'training_executed': False,
        'scientific_evaluation_executed': False, 'execution_ready': False,
    }


def register(workspace):
    """Create only the new amendment, after independent archive and source checks."""
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
    require(path.exists(), 'Section9 operations amendment is missing; sync its registration and archived sources')
    actual = sha256_file(path)
    require(path.with_suffix('.sha256').read_text().strip() == actual,
            'Section9 operations amendment checksum changed')
    current = specification(root)
    require(digest(read_json(path)) == digest(current),
            'Section9 operations implementation or historical evidence changed')
    return {
        'all_passed': True, 'registration_sha256': actual,
        'historical_registrations_preserved': len(HISTORICAL_MANIFESTS),
        'archived_source_sha256': {name: entry['sha256'] for name, entry in current['archived_bindings'].items()},
        'training_executed': False, 'scientific_evaluation_executed': False,
        'execution_ready': False, 'scope': 'read_only_operations_and_historical_source_authentication',
    }


def amendment_sha(workspace):
    """Return an authenticated amendment hash for new operation receipts."""
    return verify(workspace)['registration_sha256']


def source_hash(workspace, name):
    """Resolve a historical binding; never substitute an archive for live science."""
    root = Path(workspace).resolve(strict=True)
    if name in ARCHIVED_SOURCES:
        # A checkout that has not adopted archival still uses its original live
        # bytes. The caller's immutable manifest comparison remains the gate.
        # Once either the amendment or archive exists, substitution requires the
        # complete amendment; partial synchronization cannot skip its checks.
        if not (root / REGISTRATION).exists() and not (root / ARCHIVE / name).exists():
            return sha256_file(workspace_path(root, name))
        return verify(root)['archived_source_sha256'][name]
    return sha256_file(workspace_path(root, name))
