"""Documentation successors inherit semantic proofs only for unchanged bytes.

This authenticates the complete saved scientific payload closure and archives.
It does not replay samplers, optimizers, numerical arrays, model loads or GPUs.
The original full semantic builder remains registration.make_registration.
"""

from copy import deepcopy
from pathlib import Path
import re

from .common import digest, read_json, require, sha256_file, workspace_path
from . import registration as r


AMENDMENT_SHA256 = 'a7d0f94216ef09968076d932c18f39d61aa19cfd87a290a377215a5c8be80354'
CHANGED = {'large_qubit/README.md', 'large_qubit/registration.py', 'large_qubit/section7.md',
           'large_qubit/tests/test_section7_gpu_docs.py', 'large_qubit/tests/test_section7_uuid_registration.py'}
ADDED = {'large_qubit/documentation_registration.py', 'large_qubit/tests/test_section7_guide_registration.py',
         'large_qubit/amendments/section7_guide_v1.json'}
STAGES = (('ORIGINAL', 17), ('SECTION3', 36), ('SECTION4', 54), ('SECTION5', 74),
          ('SECTION6', 89), ('SECTION6_V2', 109), ('SECTION7_INITIAL', 128),
          ('SECTION7_GPU_DOCS', 130), ('SECTION7_UUID', 134))
SHA = re.compile(r'^[0-9a-f]{64}$')


def authenticate_predecessor(workspace):
    """Hash each source/payload once; semantic conclusions come from the pin."""
    root = Path(workspace).resolve(strict=True)
    checked = {}

    def check(relative, expected, size=None):
        require(type(expected) is str and SHA.fullmatch(expected), 'Invalid documentary proof hash')
        path = workspace_path(root, relative)
        if relative in checked:
            require(checked[relative] == expected, 'Conflicting documentary proof identities')
        else:
            require(path.is_file() and sha256_file(path) == expected, 'Documentary proof bytes changed: ' + relative)
            checked[relative] = expected
        require(size is None or path.stat().st_size == size, 'Documentary payload size changed: ' + relative)
        return path

    saved = None
    for prefix, count in STAGES:
        folder = getattr(r, prefix + '_REGISTRATION')
        anchor = getattr(r, prefix + '_MANIFEST_SHA256')
        archive = getattr(r, prefix + '_ARCHIVE')
        manifest, actual = r._registration_files(root, folder)
        require(actual == anchor, 'Documentary predecessor registration anchor changed')
        check(folder + '/manifest.json', anchor)
        old = read_json(manifest)
        require(len(old['bindings_sha256']) == count, 'Documentary archive inventory differs')
        for relative, expected in old['bindings_sha256'].items():
            workspace_path(root, relative)
            check(archive + '/' + relative, expected)
        if prefix == 'SECTION7_UUID':
            saved = old
    require(saved is not None and saved['registration_id'] == 'sections_1_7_v2'
            and saved['execution_ready'] is False, 'Missing anchored UUID implementation')
    bindings = saved['bindings_sha256']
    for relative, expected in bindings.items():
        if relative not in CHANGED:
            check(relative, expected)

    def references(value, historical_sources=False):
        if isinstance(value, list):
            for item in value: references(item, historical_sources)
        elif isinstance(value, dict):
            for path_key, hash_key in (('path', 'sha256'), ('path', 'file_sha256'),
                ('path', 'manifest_sha256'), ('receipt', 'sha256'), ('log', 'log_sha256'),
                ('manifest_path', 'manifest_sha256'), ('sidecar_path', 'sidecar_sha256'),
                ('policy_path', 'policy_sha256'), ('reservations_path', 'reservations_sha256')):
                if path_key in value and hash_key in value:
                    check(value[path_key], value[hash_key], value.get('bytes') if path_key == 'path' else None)
            for key, item in value.items():
                if key == 'archived_bindings_sha256': continue  # Authenticated via all nine inventories above.
                if isinstance(item, str) and SHA.fullmatch(item) and '/' in key:
                    target = key
                    if historical_sources and bindings.get(key) == item:
                        target = r.SECTION7_UUID_ARCHIVE + '/' + key
                    check(target, item)
                elif key == 'source_bindings' and historical_sources:
                    references(item, True)
                elif isinstance(item, (dict, list)):
                    references(item, historical_sources)

    # All baseline descriptors, original source bindings and selected saved config records.
    references(saved['reuse_inventory'])
    references(saved['grid_amendment_verification'])
    references(saved['base_asset_metadata_verification'])
    gate_names = ('realized_physics_manifest', 'section4_gate_manifest', 'section5_gate_manifest',
                  'section6_v1_gate_manifest', 'section6_gate_manifest')
    for name in gate_names:
        binding = saved[name]
        require(binding is not None, 'Missing inherited scientific gate')
        references(binding)
        certificate = read_json(root / binding['path'])
        references(certificate)
        private = workspace_path(root, certificate['restricted_output'], namespace='large_qubit/scoring_only')
        paths = private.iterdir() if name.startswith('section6') else private.rglob('*')
        actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()
                  and (not name.startswith('section6') or not p.name.startswith('.'))}
        require(actual == set(certificate['files']), 'Scientific raw payload inventory changed: ' + name)
        if name == 'realized_physics_manifest':
            public = workspace_path(root, certificate['output'], namespace='large_qubit/physics')
            require({p.name for p in public.iterdir()} == {'manifest.json', 'manifest.sha256'},
                    'Physical public manifest inventory changed')
        if name == 'section5_gate_manifest':
            prep_path = 'large_qubit/evidence/sections_1_5_v1/preparation.json'
            prep = read_json(check(prep_path, certificate['preparation_file_sha256']))
            references(prep)
            check('large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/section5_preparation_v1/screening.json', prep['screening_file_sha256'])
            check('phase6/grammar.py', prep['reused_grammar_sha256'])
            check('large_qubit/catalogues/co2_screened_v1.json', prep['catalogue_file_sha256'])
        if name.startswith('section6'):
            version = 'v1' if name == 'section6_v1_gate_manifest' else 'v2'
            prep = read_json(check(f'large_qubit/evidence/sections_1_6_{version}/preparation.json',
                                   certificate['preparation_file_sha256']))
            references(prep)
            check('phase6/objective.py', prep['legacy_weight_mathematics_sha256'])
    require(not (root / r.SECTION6_RELEASE_MANIFEST_PATH).exists(), 'The inherited v1 NO-GO cannot gain a release')
    release = read_json(root / r.SECTION6_V2_RELEASE_MANIFEST_PATH)
    references(release)
    for folder_key, expected in (('corpus_root', {Path(d['path']).name for d in release['files'].values()}),
                                 ('release_root', {'manifest.json', 'manifest.sha256'})):
        folder = workspace_path(root, release[folder_key])
        require({p.name for p in folder.iterdir()} == expected, 'Released corpus inventory changed')

    # Same 21-source preparation, cache bytes, metadata and zero-production certificate.
    implementation = r.section7_implementation_binding(root)
    require(digest(implementation) == digest(saved['section7_implementation']), 'Prepared workflow differs from inherited proof')
    prep = read_json(root / implementation['preparation_path'])
    references(prep)
    cert = read_json(root / implementation['implementation_certificate_path'])
    for key in ('unit_tests', 'historical_preservation'):
        references(cert[key])
    unit = read_json(root / cert['unit_tests']['receipt'])
    references(unit, True)
    return saved, {'all_passed': True, 'predecessor_registration': r.SECTION7_UUID_REGISTRATION,
        'manifest_sha256': r.SECTION7_UUID_MANIFEST_SHA256, 'archive_root': r.SECTION7_UUID_ARCHIVE,
        'bound_files_verified': len(bindings), 'registration_archives_verified': len(STAGES),
        'proof_files_verified': len(checked), 'proof_bindings_sha256': checked,
        'inherited_saved_scientific_validation': True, 'semantic_scientific_replay_performed': False,
        'model_weights_loaded': False, 'training_executed': False, 'execution_ready': False}


def make_documentation_registration(workspace):
    root = Path(workspace).resolve(strict=True)
    saved, proof = authenticate_predecessor(root)
    path = workspace_path(root, r.SECTION7_GUIDE_AMENDMENT)
    require(sha256_file(path) == AMENDMENT_SHA256, 'Guide amendment anchor changed')
    amendment = read_json(path)
    require(amendment['schema_version'] == 'section7-guide-documentation-amendment-v1'
        and amendment['same_implementation_id'] == saved['registration_id']
        and amendment['successor_registration'] == r.DEFAULT_REGISTRATION
        and set(amendment['allowed_changed_bindings']) == CHANGED
        and set(amendment['allowed_added_bindings']) == ADDED
        and amendment['predecessor_registration'] == {'path': r.SECTION7_UUID_REGISTRATION+'/manifest.json',
            'sha256': r.SECTION7_UUID_MANIFEST_SHA256, 'bound_files': 134, 'archive_root': r.SECTION7_UUID_ARCHIVE}
        and amendment['training_preparation'] == {'path': r.SECTION7_PREPARATION_PATH,
            'sha256': saved['section7_implementation']['preparation_sha256']}
        and all(amendment[key] is False for key in ('scientific_settings_changed',
            'training_code_or_policy_changed', 'data_or_token_caches_changed', 'runtime_execution_authorized')),
        'Guide amendment exceeds documentary scope')
    current = r.current_bindings(root)
    previous = saved['bindings_sha256']
    require(set(current) == set(previous) | ADDED, 'Unexpected documentary source inventory')
    changes = {p: {'old_sha256': previous[p], 'new_sha256': current[p]}
               for p in previous if current[p] != previous[p]}
    require(set(changes) <= CHANGED, 'A scientific or runtime binding changed during documentation revision')
    document = deepcopy(saved)
    document['bindings_sha256'] = current
    document['section7_guide_documentation_revision'] = {'amendment_path': r.SECTION7_GUIDE_AMENDMENT,
        'amendment_sha256': AMENDMENT_SHA256, 'predecessor_proof': proof,
        'changed_bindings': changes, 'added_bindings': {p: current[p] for p in sorted(ADDED)},
        'inherited_saved_scientific_validation': True, 'semantic_scientific_replay_performed': False}
    return document


def _report(saved, registration, manifest_sha, proof):
    s4 = saved['section4_gate_manifest']['saved_certificate_and_payload_verification']
    s5 = saved['section5_gate_manifest']['saved_certificate_and_payload_verification']
    s6 = saved['section6_gate_manifest']['saved_certificate_and_payload_verification']
    report = {'all_passed': True, 'registration': registration, 'registration_id': saved['registration_id'],
        'manifest_sha256': manifest_sha, 'bound_files_verified': len(saved['bindings_sha256']),
        'source_verification_mode': 'documentation_snapshot_with_authenticated_inherited_scientific_proofs',
        'inherited_saved_scientific_validation': True, 'semantic_scientific_replay_performed': False,
        'scientific_proof_authentication': proof, 'section_3_physical_definition_verified': True,
        'section_3_complete': True, 'saved_section_3_physical_certificates_verified': True,
        'numerical_arrays_recomputed_by_registration': False, 'section_4_saved_gate_verified': True,
        'section_4_complete': s4['section_4_complete'], 'section_4_feasibility_gate': s4['feasibility_gate'],
        'reference_and_circuit_evaluator_validated': s4['correctness_passed'] and s4['reference_converged'],
        'section_5_saved_gate_verified': True, 'section_5_complete': s5['section_5_complete'],
        'section_5_feasibility_gate': s5['feasibility_gate'], 'section_6_saved_gate_verified': True,
        'section_6_complete': s6['section_6_complete'], 'section_6_feasibility_gate': s6['feasibility_gate'],
        'section_6_release_sha256': s6['release_sha256'], 'section_6_production_ordinary_requests': s6['production_ordinary_requests'],
        'section_6_cumulative_ordinary_requests': s6['cumulative_ordinary_requests'],
        'section_6_prior_failed_attempt_ordinary_requests': 6538,
        'section_7_implementation_ready': True, 'section_7_status': 'IMPLEMENTATION_READY',
        'section_7_training_complete': False, 'section_7_training_not_run': True,
        'section_7_training_not_run_scope': 'as_of_saved_implementation_certificate',
        'section_7_production_gpu_preflight_verified': False, 'section_7_trained_adapters': 0,
        'historical_reuse_verification': saved['section6_v2_registration_lineage']['historical_reuse_verification'],
        'model_weight_bytes_verified': False,
        'execution_ready': False, 'scope': 'Scientific conclusions inherited only for exact authenticated bytes; no fresh semantic or GPU execution'}
    for key in ('original_registration_lineage', 'section3_registration_lineage', 'section4_registration_lineage',
        'section5_registration_lineage', 'section6_v1_registration_lineage', 'section6_v2_registration_lineage',
        'section6_acquisition_amendment', 'section7_documentation_revision', 'section7_uuid_compatibility_amendment',
        'section7_original_registration_lineage', 'section7_gpu_docs_registration_lineage',
        'section7_guide_documentation_revision'):
        report[key] = saved.get(key)
    return report


def verify_predecessor_registration(workspace):
    saved, proof = authenticate_predecessor(workspace)
    return _report(saved, r.SECTION7_UUID_REGISTRATION, r.SECTION7_UUID_MANIFEST_SHA256, proof)


def verify_documentation_registration(workspace, registration):
    root = Path(workspace).resolve(strict=True)
    path, actual = r._registration_files(root, registration)
    saved = read_json(path)
    expected = make_documentation_registration(root)
    require(digest(saved) == digest(expected), 'Documentation registration or inherited evidence changed')
    return _report(saved, registration, actual,
                   expected['section7_guide_documentation_revision']['predecessor_proof'])
