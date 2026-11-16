"""Append-only Section 3 physical realization and read-only authentication.

Scientific values and integral arrays stay in scoring_only.  The public
manifest contains identities, status and hashes; it supplies no reference or
target-energy labels.  Importing this module does not import scientific code.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

from .common import (canonical_bytes, digest, read_json, require, sha256_file,
                     workspace_path)
from .definitions import GEOMETRY_IDS, build_definition, geometry
from .protocol import BENCHMARK_ID


DEFAULT_PHYSICS = 'large_qubit/physics/section3_v3'
SCHEMA = 'large-qubit-section3-physical-realization-v1'
SOURCE_PATHS = ('large_qubit/chemistry.py', 'large_qubit/orbitals.py')
ARRAY_NAMES = ('ao_overlap', 'cross_ao_overlap_to_anchor',
               'ao_one_electron_integrals', 'ao_eri_chemists',
               'canonical_mo_coefficients', 'canonical_mo_energies',
               'mo_coefficients', 'mo_rotation', 'one_electron_integrals', 'eri_chemists')


def _paths(root, output):
    folder = workspace_path(root, output, namespace='large_qubit/physics')
    prefix = 'large_qubit/physics/'
    require(output.startswith(prefix), 'Choose a versioned physics subdirectory')
    suffix = output[len(prefix):]
    private = f'large_qubit/scoring_only/{BENCHMARK_ID}/physical_validation/{suffix}'
    restricted = workspace_path(root, private, namespace=f'large_qubit/scoring_only/{BENCHMARK_ID}/physical_validation')
    return folder, private, restricted


def _write_json(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n')


def _write_sidecar(path):
    sidecar = path.with_suffix('.sha256')
    with sidecar.open('x', encoding='ascii', newline='\n') as handle:
        handle.write(sha256_file(path) + '\n')


def _policy():
    # Deferred to explicit build/verify, never foundation metadata imports.
    from .chemistry import scf_policy
    from .orbitals import orbital_conventions
    return {'scf_policy': scf_policy(), 'orbital_conventions': orbital_conventions()}


def _array_identity(array):
    import numpy as np
    array = np.asarray(array)
    require(array.dtype == np.dtype('float64') and np.all(np.isfinite(array)),
            'Physical arrays must be finite real float64')
    contiguous = np.ascontiguousarray(array.astype('<f8', copy=False))
    header = {'dtype': '<f8', 'shape': list(contiguous.shape), 'order': 'C'}
    h = hashlib.sha256(canonical_bytes(header) + b'\n')
    h.update(contiguous.tobytes(order='C'))
    return {**header, 'sha256': h.hexdigest()}


def _source_bindings(root):
    return {relative: sha256_file(workspace_path(root, relative, namespace='large_qubit'))
            for relative in SOURCE_PATHS}


def _validate_metadata(metadata, alias, policy):
    require(metadata.get('geometry') == geometry(alias, purpose='physical_validation'),
            f'Physical metadata geometry mismatch: {alias}')
    require(metadata.get('physical_validation_passed') is True,
            f'Physical validation did not pass: {alias}')
    require(metadata.get('correlated_reference_computed') is False,
            'Section 3 cannot supply correlated reference results')
    require(metadata.get('scf_policy') == policy['scf_policy']
            and metadata.get('orbital_conventions') == policy['orbital_conventions'],
            'Realization used a different SCF/orbital policy')
    require(metadata.get('ao_count') == 15 and metadata.get('spatial_orbital_count') == 15
            and metadata.get('electron_population') == [11, 11], 'Physical counts differ')
    require(isinstance(metadata.get('ao_labels'), list) and len(metadata['ao_labels']) == 15,
            'Exact AO labels are missing')
    scf = metadata.get('scf', {})
    require(scf.get('converged') is True, f'SCF did not converge: {alias}')
    require(scf.get('gradient_norm', float('inf')) <= 1e-8, 'SCF gradient exceeds tolerance')
    stability = metadata.get('stability', {})
    require(stability.get('all_channels_stable') is True, f'SCF stability failed: {alias}')
    channels = stability.get('channels', {})
    require(set(channels) == {'internal_RHF', 'external_RHF_to_UHF', 'external_real_to_complex'},
            'Complete SCF stability channels are missing')
    for channel in channels.values():
        require(channel.get('stable') is True and channel.get('dimension') == 44
                and len(channel.get('eigenvalues', [])) == 44,
                'Incomplete bounded stability certificate')
        require(channel.get('minimum_eigenvalue', -float('inf')) >= -1e-7
                and channel.get('maximum_symmetry_residual', float('inf')) <= 1e-8
                and channel.get('maximum_eigenpair_residual', float('inf')) <= 1e-8,
                'Invalid SCF stability certificate')
        require(channel['eigenvalues'] == sorted(channel['eigenvalues'])
                and abs(channel['minimum_eigenvalue'] - min(channel['eigenvalues'])) <= 1e-12,
                'SCF stability spectrum and minimum differ')
    consistency = metadata.get('integral_consistency', {})
    require(consistency.get('scf_energy_absolute_difference_ha', float('inf')) <= 1e-8,
            'HF integral consistency check did not pass')
    for field in ('one_electron_hermiticity_max_abs', 'eri_pair_exchange_max_abs',
                  'eri_within_pair_max_abs'):
        require(consistency.get(field, float('inf')) <= 1e-8, f'Integral symmetry check failed: {field}')
    require(metadata.get('alignment', {}).get('anchor_geometry_id') == 'CO2_T02',
            'Wrong orbital anchor')
    require(metadata['alignment'].get('occupation_order') == [2] * 11 + [0] * 4,
            'Wrong aligned MO occupation order')


def _fingerprints(metadata, arrays, definition_sha, policy_sha):
    identities = {name: _array_identity(arrays[name]) for name in ARRAY_NAMES}
    hamiltonian = digest({
        'definition_sha256': definition_sha, 'policy_sha256': policy_sha,
        'geometry': metadata['geometry'], 'backend': metadata['backend'],
        'one_electron_integrals': identities['one_electron_integrals'],
        'eri_chemists': identities['eri_chemists'],
        'nuclear_repulsion_ha': metadata['nuclear_repulsion_ha'],
        'sector': [11, 11], 'nuclear_constant_included': True,
    })
    orbital = digest({'geometry': metadata['geometry'], 'ao_labels': metadata['ao_labels'],
                     'basis_definition': metadata['basis_definition'],
                     'ao_overlap': identities['ao_overlap'],
                     'cross_ao_overlap_to_anchor': identities['cross_ao_overlap_to_anchor'],
                     'mo_coefficients': identities['mo_coefficients'],
                     'conventions': metadata['orbital_conventions']})
    return identities, hamiltonian, orbital


def build_physics(root, output=DEFAULT_PHYSICS, *, threads=1):
    """Run six bounded SCF validations, preserving a failed attempt if necessary.

    This command never overwrites either public or restricted output.  A failure
    leaves failure.json plus its private diagnostics, and no passed manifest.
    """
    from .chemistry import PhysicalValidationError, prepare_geometry
    import numpy as np
    root = Path(root).resolve(strict=True)
    require(type(threads) is int and threads > 0, 'threads must be a positive integer')
    folder, private, restricted = _paths(root, output)
    require(not folder.exists() and not restricted.exists(),
            'Physics output already exists; preserve it and choose a new version')
    source = _source_bindings(root)
    policy = _policy()
    definition_sha = digest(build_definition())
    policy_sha = digest(policy)
    started = time.perf_counter()
    folder.mkdir(parents=True)
    restricted.mkdir(parents=True)
    files, entries = {}, {}
    anchor, active_alias = None, None
    try:
        for alias in ('CO2_T02',) + tuple(alias for alias in GEOMETRY_IDS if alias != 'CO2_T02'):
            active_alias = alias
            realized = prepare_geometry(geometry(alias, purpose='physical_validation'), anchor, threads=threads)
            metadata, arrays = realized['metadata'], realized['arrays']
            _validate_metadata(metadata, alias, policy)
            if anchor is not None:
                require(metadata['backend'] == anchor['metadata']['backend'],
                        'Mixed numerical backends in one physical realization')
            require(set(arrays) == set(ARRAY_NAMES), 'Unexpected or missing physical arrays')
            identities, hamiltonian, orbital = _fingerprints(metadata, arrays, definition_sha, policy_sha)
            metadata = {**metadata, 'array_identities': identities,
                        'definition_sha256': definition_sha, 'policy_sha256': policy_sha,
                        'hamiltonian_fingerprint': hamiltonian, 'orbital_fingerprint': orbital}
            metadata_relative = f'{private}/{alias}.json'
            array_relative = f'{private}/{alias}.npz'
            metadata_path = workspace_path(root, metadata_relative, namespace=private)
            array_path = workspace_path(root, array_relative, namespace=private)
            _write_json(metadata_path, metadata)
            with array_path.open('xb') as handle:
                np.savez(handle, **arrays)
            files[metadata_relative] = sha256_file(metadata_path)
            files[array_relative] = sha256_file(array_path)
            entries[alias] = {
                'geometry_id': alias, 'partition': geometry(alias)['partition'],
                'physical_validation_passed': True,
                'scf_converged': True, 'all_stability_channels_stable': True,
                'metadata_path': metadata_relative, 'arrays_path': array_relative,
                'hamiltonian_fingerprint': hamiltonian, 'orbital_fingerprint': orbital,
                'array_identities': identities,
            }
            if alias == 'CO2_T02':
                anchor = realized
        manifest = {
            'schema_version': SCHEMA, 'benchmark_id': BENCHMARK_ID,
            'output': output, 'restricted_output': private,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'definition_sha256': definition_sha, 'policy_sha256': policy_sha, 'policy': policy,
            'source_bindings': source, 'backend': anchor['metadata']['backend'],
            'geometry_order': list(GEOMETRY_IDS), 'geometries': entries,
            'all_passed': True, 'physical_validation_passed': True,
            'correlated_reference_computed': False, 'execution_ready': False,
            'scope': 'Section 3 only: full-space CO2 SCF stability, gauge and integral consistency; no FCI, circuit evaluator, VQE or training',
            'files': files, 'wall_seconds': time.perf_counter() - started,
        }
        manifest_path = folder / 'manifest.json'
        _write_json(manifest_path, manifest)
        _write_sidecar(manifest_path)
        # Read every persisted payload and independently recheck the arithmetic
        # before returning a successful build to the caller.
        verify_physics(root, output, numerical_integrity=True)
        return manifest
    except Exception as error:
        diagnostic = (error.diagnostics if isinstance(error, PhysicalValidationError)
                      else {'error_type': type(error).__name__, 'error': str(error)})
        diagnostic_path = restricted / 'failure_diagnostics.json'
        _write_json(diagnostic_path, {'geometry_id': active_alias, 'diagnostics': diagnostic,
                                      'all_passed': False, 'physical_validation_passed': False})
        failure = {'schema_version': SCHEMA, 'benchmark_id': BENCHMARK_ID,
                   'all_passed': False, 'physical_validation_passed': False,
                   'failed_geometry_id': active_alias, 'error_type': type(error).__name__,
                   'diagnostics_path': f'{private}/failure_diagnostics.json',
                   'diagnostics_sha256': sha256_file(diagnostic_path),
                   'completed_geometries': list(entries),
                   'wall_seconds': time.perf_counter() - started}
        _write_json(folder / 'failure.json', failure)
        _write_sidecar(folder / 'failure.json')
        raise


def _require_alignment_equal(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                'Saved alignment metadata fields differ')
        for key in expected:
            _require_alignment_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                'Saved alignment metadata dimensions differ')
        for a, b in zip(actual, expected):
            _require_alignment_equal(a, b)
    elif type(expected) is float:
        require(type(actual) in (float, int) and abs(actual - expected) <= 1e-8,
                'Saved alignment numerical metadata differs')
    else:
        require(type(actual) is type(expected) and actual == expected,
                'Saved alignment convention metadata differs')


def _numerical_integrity(metadata, arrays, entry, definition_sha, policy_sha,
                         anchor_coeff=None):
    import numpy as np
    from .orbitals import (align_orbitals, canonicalize_anchor,
                           closed_shell_hf_energy, transform_integrals)
    require(set(arrays) == set(ARRAY_NAMES), 'Physical array payload differs from the declaration')
    for name in ARRAY_NAMES:
        expected = (15,) if name == 'canonical_mo_energies' else ((15,) * 4 if name in ('ao_eri_chemists', 'eri_chemists') else (15, 15))
        require(arrays[name].shape == expected, f'Physical array shape differs: {name}')
    identities, hamiltonian, orbital = _fingerprints(metadata, arrays, definition_sha, policy_sha)
    require(identities == metadata['array_identities'] == entry['array_identities'],
            'Physical tensor identity mismatch')
    require(hamiltonian == metadata['hamiltonian_fingerprint'] == entry['hamiltonian_fingerprint'],
            'Hamiltonian fingerprint mismatch')
    require(orbital == metadata['orbital_fingerprint'] == entry['orbital_fingerprint'],
            'Orbital fingerprint mismatch')
    c, s, u = arrays['mo_coefficients'], arrays['ao_overlap'], arrays['mo_rotation']
    require(np.max(np.abs(c.T @ s @ c - np.eye(15))) <= 1e-8,
            'Saved orbitals are not S-orthonormal')
    require(np.max(np.abs(u.T @ u - np.eye(15))) <= 1e-8,
            'Saved MO rotation is not orthogonal')
    require(np.max(np.abs(arrays['canonical_mo_coefficients'] @ u - c)) <= 1e-8,
            'Saved canonical-to-aligned MO transformation differs')
    canonical_occ = np.asarray(metadata['scf']['canonical_mo_occupations'])
    require(canonical_occ.shape == (15,) and np.all((canonical_occ == 0) | (canonical_occ == 2))
            and np.sum(canonical_occ) == 22, 'Invalid recorded canonical RHF occupations')
    require(np.all(np.abs(u[np.ix_(canonical_occ == 2, np.arange(15) >= 11)]) < 1e-8)
            and np.all(np.abs(u[np.ix_(canonical_occ == 0, np.arange(15) < 11)]) < 1e-8),
            'Saved MO rotation mixes occupied and virtual sectors')
    if metadata['geometry']['geometry_id'] == 'CO2_T02':
        require(np.max(np.abs(arrays['cross_ao_overlap_to_anchor'] - s)) <= 1e-12,
                'Anchor cross-AO overlap must equal its AO overlap')
        replay_c, replay_u, replay_metadata = canonicalize_anchor(
            arrays['canonical_mo_coefficients'], arrays['canonical_mo_energies'],
            s, canonical_occ)
    else:
        require(anchor_coeff is not None, 'Saved orbital alignment anchor is missing')
        replay_c, replay_u, replay_metadata = align_orbitals(
            anchor_coeff, arrays['canonical_mo_coefficients'], arrays['cross_ao_overlap_to_anchor'],
            s, arrays['canonical_mo_energies'], canonical_occ)
    require(np.max(np.abs(replay_c - c)) <= 1e-8 and np.max(np.abs(replay_u - u)) <= 1e-8,
            'Saved orbitals differ from the declared cross-AO gauge recipe')
    _require_alignment_equal(metadata['alignment'], replay_metadata)
    h, eri = transform_integrals(arrays['ao_one_electron_integrals'], arrays['ao_eri_chemists'], c)
    require(np.max(np.abs(h - arrays['one_electron_integrals'])) <= 1e-8
            and np.max(np.abs(eri - arrays['eri_chemists'])) <= 1e-8,
            'Saved integrals do not use the final MO gauge')
    hf = closed_shell_hf_energy(h, eri, 11, metadata['nuclear_repulsion_ha'])
    require(abs(hf - metadata['scf']['total_energy_ha']) <= 1e-8
            and abs(hf - metadata['integral_consistency']['hf_contraction_total_energy_ha']) <= 1e-8,
            'Saved Hamiltonian does not reproduce the recorded HF determinant energy')


def verify_physics(root, physics_root=DEFAULT_PHYSICS, *, numerical_integrity=False):
    """Authenticate all six persisted physical payloads without changing files.

    The default verifies saved certificates and every byte hash without loading
    numerical arrays.  Explicit numerical_integrity redoes MO/integral/HF
    arithmetic; it does not rerun SCF or claim independent FCI correctness.
    """
    root = Path(root).resolve(strict=True)
    folder, private, restricted = _paths(root, physics_root)
    require(not (folder / 'failure.json').exists(), 'Physics attempt contains a recorded failure')
    manifest_path = folder / 'manifest.json'
    sidecar_path = folder / 'manifest.sha256'
    require(manifest_path.is_file() and sidecar_path.is_file(), 'Complete physical manifest is missing')
    require(sidecar_path.read_text(encoding='ascii').strip() == sha256_file(manifest_path),
            'Physical manifest sidecar mismatch')
    saved = read_json(manifest_path)
    require(saved.get('schema_version') == SCHEMA and saved.get('benchmark_id') == BENCHMARK_ID,
            'Wrong physical manifest identity')
    require(saved.get('output') == physics_root and saved.get('restricted_output') == private,
            'Physical manifest output identity differs')
    require(saved.get('all_passed') is True and saved.get('physical_validation_passed') is True
            and saved.get('correlated_reference_computed') is False
            and saved.get('execution_ready') is False, 'Physical manifest scope/status differs')
    require(saved.get('definition_sha256') == digest(build_definition()), 'Physical definition binding changed')
    require(saved.get('policy_sha256') == digest(saved.get('policy')),
            'SCF/orbital policy binding changed')
    require(saved.get('source_bindings') == _source_bindings(root), 'Chemistry/orbital source binding changed')
    if numerical_integrity:
        require(saved['policy'] == _policy(), 'Runtime SCF/orbital policy differs')
    require(saved.get('geometry_order') == list(GEOMETRY_IDS)
            and set(saved.get('geometries', {})) == set(GEOMETRY_IDS), 'Incomplete geometry grid')
    expected_files = {f'{private}/{alias}.{suffix}' for alias in GEOMETRY_IDS for suffix in ('json', 'npz')}
    require(set(saved.get('files', {})) == expected_files, 'Physical payload set differs')
    actual_public = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual_public == {'manifest.json', 'manifest.sha256'}, 'Unexpected public physics files')
    actual_private = {p.relative_to(root).as_posix() for p in restricted.rglob('*') if p.is_file()}
    require(actual_private == expected_files, 'Unexpected/missing restricted physics files')
    for relative, expected_hash in saved['files'].items():
        path = workspace_path(root, relative, namespace=private)
        require(path.is_file() and sha256_file(path) == expected_hash, f'Physical payload changed: {relative}')
    anchor_coeff = None
    if numerical_integrity:
        import numpy as np
        anchor_entry = saved['geometries']['CO2_T02']
        with np.load(workspace_path(root, anchor_entry['arrays_path'], namespace=private), allow_pickle=False) as payload:
            anchor_coeff = payload['mo_coefficients']
    for alias in GEOMETRY_IDS:
        entry = saved['geometries'][alias]
        require(entry.get('geometry_id') == alias and entry.get('partition') == geometry(alias)['partition']
                and entry.get('physical_validation_passed') is True
                and entry.get('scf_converged') is True
                and entry.get('all_stability_channels_stable') is True,
                f'Geometry manifest certificate differs: {alias}')
        require(entry.get('metadata_path') == f'{private}/{alias}.json'
                and entry.get('arrays_path') == f'{private}/{alias}.npz', 'Cross-geometry payload binding')
        metadata = read_json(workspace_path(root, entry['metadata_path'], namespace=private))
        _validate_metadata(metadata, alias, saved['policy'])
        require(metadata['backend'] == saved['backend'], 'Geometry numerical backend differs')
        require(metadata.get('definition_sha256') == saved['definition_sha256']
                and metadata.get('policy_sha256') == saved['policy_sha256'], 'Metadata semantic binding differs')
        require(metadata.get('hamiltonian_fingerprint') == entry.get('hamiltonian_fingerprint')
                and metadata.get('orbital_fingerprint') == entry.get('orbital_fingerprint')
                and metadata.get('array_identities') == entry.get('array_identities'),
                'Public/restricted physical identities differ')
        if numerical_integrity:
            import numpy as np
            with np.load(workspace_path(root, entry['arrays_path'], namespace=private), allow_pickle=False) as payload:
                arrays = {name: payload[name] for name in payload.files}
            _numerical_integrity(metadata, arrays, entry, saved['definition_sha256'],
                                 saved['policy_sha256'], anchor_coeff)
    return {'all_passed': True, 'physical_validation_passed': True,
            'geometries_verified': len(GEOMETRY_IDS), 'payload_files_verified': len(expected_files),
            'manifest_sha256': sha256_file(manifest_path),
            'numerical_integrity_checked': bool(numerical_integrity),
            'scf_recomputed': False, 'correlated_reference_computed': False,
            'execution_ready': False}
