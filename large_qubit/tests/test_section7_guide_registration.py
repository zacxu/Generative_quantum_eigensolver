"""A guide edit inherits scientific decisions only for exact saved payloads."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from large_qubit import documentation_registration as d, registration as r
from large_qubit.common import read_json, sha256_file


class GuideRegistrationTests(unittest.TestCase):
    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value).encode())
        return path

    def descriptor(self, relative, value):
        path = self.write(relative, value)
        return {'path': relative, 'sha256': sha256_file(path), 'bytes': path.stat().st_size}

    def patch(self, owner, name, value):
        context = patch.object(owner, name, value)
        context.start(); self.addCleanup(context.stop)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.write('large_qubit/runtime.py', '# frozen scientific and training source')
        source_sha = sha256_file(source)
        for relative in d.CHANGED: self.write(relative, 'Old registered documentation/bookkeeping')
        catalogue = self.write('large_qubit/catalogues/co2_screened_v1.json', '{}')
        grammar = self.write('phase6/grammar.py', '# historical grammar')
        objective = self.write('phase6/objective.py', '# historical weights')
        gates = {}
        self.payloads = {}
        paths = [('realized_physics_manifest', r.PHYSICS_MANIFEST_PATH),
                 ('section4_gate_manifest', r.SECTION4_MANIFEST_PATH),
                 ('section5_gate_manifest', r.SECTION5_MANIFEST_PATH),
                 ('section6_v1_gate_manifest', r.SECTION6_MANIFEST_PATH),
                 ('section6_gate_manifest', r.SECTION6_V2_MANIFEST_PATH)]
        for name, relative in paths:
            private = 'large_qubit/scoring_only/fixture/'+name
            payload = private + ('/array.npz' if name == 'realized_physics_manifest' else '/journal.jsonl')
            self.payloads[name] = payload
            descriptor = self.descriptor(payload, 'pinned raw scientific bytes\n')
            manifest = {'files': {payload: descriptor['sha256']}, 'source_bindings': {'large_qubit/runtime.py': source_sha},
                'restricted_output': private, 'output': str(Path(relative).parent).replace('\\','/')}
            if name == 'section5_gate_manifest':
                self.screening = 'large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/section5_preparation_v1/screening.json'
                screening = self.write(self.screening, '{"screened":true}')
                prep = self.write('large_qubit/evidence/sections_1_5_v1/preparation.json', {
                    'screening_file_sha256': sha256_file(screening), 'reused_grammar_sha256': sha256_file(grammar),
                    'catalogue_file_sha256': sha256_file(catalogue)})
                manifest['preparation_file_sha256'] = sha256_file(prep)
            if name.startswith('section6'):
                version = 'v1' if name == 'section6_v1_gate_manifest' else 'v2'
                reservations = self.descriptor(private+'/reservations.json', {'reserved': 384})
                manifest['files'][reservations['path']] = reservations['sha256']
                prep = self.write(f'large_qubit/evidence/sections_1_6_{version}/preparation.json', {
                    'legacy_weight_mathematics_sha256': sha256_file(objective),
                    'reservations_path': reservations['path'], 'reservations_sha256': reservations['sha256']})
                manifest['preparation_file_sha256'] = sha256_file(prep)
            path = self.write(relative, manifest)
            sidecar = self.write(relative.removesuffix('.json')+'.sha256', sha256_file(path))
            saved_report = {'all_passed': True, 'section_4_complete': True, 'section_5_complete': True,
                'section_6_complete': name != 'section6_v1_gate_manifest',
                'feasibility_gate': 'no_go' if name == 'section6_v1_gate_manifest' else 'go',
                'correctness_passed': True, 'reference_converged': True, 'release_sha256': 'a'*64,
                'production_ordinary_requests': 6552, 'cumulative_ordinary_requests': 13090}
            gates[name] = {'path': relative, 'manifest_sha256': sha256_file(path),
                'sidecar_path': relative.removesuffix('.json')+'.sha256', 'sidecar_sha256': sha256_file(sidecar),
                'saved_certificate_and_payload_verification': saved_report}
        self.rows = self.descriptor('large_qubit/corpora/co2_corpus_v2/fitting.jsonl', 'fitting rows\n')
        diagnostic = self.descriptor('large_qubit/corpora/co2_corpus_v2/diagnostic.jsonl', 'isolated diagnostics\n')
        release = self.write(r.SECTION6_V2_RELEASE_MANIFEST_PATH, {
            'files': {'fitting': self.rows, 'diagnostic': diagnostic},
            'raw_artifacts_sha256': {self.payloads['section6_gate_manifest']: sha256_file(self.root/self.payloads['section6_gate_manifest'])},
            'corpus_root': 'large_qubit/corpora/co2_corpus_v2', 'release_root': str(Path(r.SECTION6_V2_RELEASE_MANIFEST_PATH).parent).replace('\\','/')})
        self.write(r.SECTION6_V2_RELEASE_MANIFEST_PATH.removesuffix('.json')+'.sha256', sha256_file(release))
        self.cache_payload = self.descriptor('large_qubit/training_data/section7_v1/gpt2_xl/rows.jsonl', 'compiled tokens\n')
        cache = self.descriptor(r.SECTION7_CACHE_MANIFEST_PATHS[0], {'files': {'rows': self.cache_payload}})
        self.unit_log = self.descriptor('large_qubit/evidence/sections_1_7_gpu_uuid_v1/unit_validation.log', 'offline tests OK\n')
        unit = self.descriptor('large_qubit/evidence/sections_1_7_gpu_uuid_v1/unit_validation.json', {
            'all_passed': True, 'log': self.unit_log['path'], 'log_sha256': self.unit_log['sha256'],
            'source_bindings': {'large_qubit/registration.py': sha256_file(self.root/'large_qubit/registration.py')}})
        prep = self.write(r.SECTION7_PREPARATION_PATH, {'source_bindings': {'large_qubit/runtime.py': source_sha},
            'token_caches': {'gpt2_xl': {'manifest_path': cache['path'], 'manifest_sha256': cache['sha256']}}})
        cert = self.write(r.SECTION7_IMPLEMENTATION_PATH, {'unit_tests': {'receipt': unit['path'], 'sha256': unit['sha256']},
            'historical_preservation': self.descriptor('large_qubit/evidence/sections_1_7_v1/preservation.json', {'all_passed': True})})
        self.implementation = {'preparation_path': r.SECTION7_PREPARATION_PATH, 'preparation_sha256': sha256_file(prep),
            'implementation_certificate_path': r.SECTION7_IMPLEMENTATION_PATH,
            'implementation_certificate_sha256': sha256_file(cert), 'implementation_ready': True}
        initial = {p.relative_to(self.root).as_posix(): sha256_file(p) for p in self.root.rglob('*') if p.is_file()
                   and not p.relative_to(self.root).as_posix().startswith(('large_qubit/scoring_only/', 'large_qubit/corpora/'))}
        for index in range(134-len(initial)):
            path = f'large_qubit/static_{index:03d}.txt'
            initial[path] = sha256_file(self.write(path, 'old static artifact'))
        self.saved = {**gates, 'schema_version': 'large-qubit-section7-registration-v2',
            'registration_id': 'sections_1_7_v2', 'execution_ready': False, 'bindings_sha256': initial,
            'section7_implementation': self.implementation, 'reuse_inventory': {},
            'grid_amendment_verification': {}, 'base_asset_metadata_verification': {},
            'section6_v2_registration_lineage': {'historical_reuse_verification': {'all_passed': True}}}
        old = self.write(r.SECTION7_UUID_REGISTRATION+'/manifest.json', self.saved)
        self.write(r.SECTION7_UUID_REGISTRATION+'/manifest.sha256', sha256_file(old))
        self.patch(r, 'SECTION7_UUID_MANIFEST_SHA256', sha256_file(old))
        self.patch(d, 'STAGES', (('SECTION7_UUID', 134),))
        for relative in initial:
            self.write(r.SECTION7_UUID_ARCHIVE+'/'+relative, (self.root/relative).read_bytes())
        for relative in d.ADDED: self.write(relative, 'new documentary bookkeeping')
        amendment = {'schema_version': 'section7-guide-documentation-amendment-v1',
            'same_implementation_id': 'sections_1_7_v2', 'successor_registration': r.DEFAULT_REGISTRATION,
            'allowed_changed_bindings': sorted(d.CHANGED), 'allowed_added_bindings': sorted(d.ADDED),
            'predecessor_registration': {'path': r.SECTION7_UUID_REGISTRATION+'/manifest.json',
                'sha256': r.SECTION7_UUID_MANIFEST_SHA256, 'bound_files': 134, 'archive_root': r.SECTION7_UUID_ARCHIVE},
            'training_preparation': {'path': r.SECTION7_PREPARATION_PATH, 'sha256': sha256_file(prep)},
            'scientific_settings_changed': False, 'training_code_or_policy_changed': False,
            'data_or_token_caches_changed': False, 'runtime_execution_authorized': False}
        self.patch(d, 'AMENDMENT_SHA256', sha256_file(self.write(r.SECTION7_GUIDE_AMENDMENT, amendment)))
        self.write('large_qubit/section7.md', 'Short guide with CUDA_VISIBLE_DEVICES=5,6,7')
        self.write('large_qubit/registration.py', '# current documentary registration bookkeeping')
        def binding(root):
            if sha256_file(Path(root)/self.cache_payload['path']) != self.cache_payload['sha256']:
                raise ValueError('Compiled cache payload changed')
            return deepcopy(self.implementation)
        self.patch(r, 'section7_implementation_binding', binding)
        self.patch(r, 'current_bindings', lambda root: {p: sha256_file(Path(root)/p)
            for p in set(initial) | d.ADDED | {p.relative_to(root).as_posix() for p in Path(root).glob('large_qubit/*.py')}})

    def assert_bad(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            d.make_documentation_registration(self.root)

    def test_creation_preserves_every_scientific_field_and_avoids_semantic_builder(self):
        with patch.object(r, 'make_registration', side_effect=AssertionError('Scientific replay')):
            r.register(self.root)
            report = r.verify_registration(self.root)
        current = read_json(self.root/r.DEFAULT_REGISTRATION/'manifest.json')
        for key, value in self.saved.items():
            if key != 'bindings_sha256': self.assertEqual(current[key], value)
        self.assertTrue(report['inherited_saved_scientific_validation'])
        self.assertFalse(report['semantic_scientific_replay_performed'])
        self.assertFalse(report['section_7_training_complete'])

    def test_guide_tampering_after_registration_is_rejected(self):
        r.register(self.root)
        self.write('large_qubit/section7.md', 'Unregistered edit')
        with self.assertRaisesRegex(ValueError, 'registration or inherited evidence changed'): r.verify_registration(self.root)

    def test_registry_tampering_after_registration_is_rejected(self):
        r.register(self.root)
        self.write('large_qubit/registration.py', 'Unregistered bookkeeping edit')
        with self.assertRaises(ValueError): r.verify_registration(self.root)

    def test_unexpected_new_source_is_rejected(self):
        self.write('large_qubit/unregistered.py', '# new runtime source')
        self.assert_bad()

    def test_raw_journal_tampering_is_rejected(self):
        self.write(self.payloads['section6_gate_manifest'], 'altered journal\n'); self.assert_bad()

    def test_physical_array_tampering_is_rejected(self):
        self.write(self.payloads['realized_physics_manifest'], b'altered NPZ bytes'); self.assert_bad()

    def test_corpus_row_tampering_is_rejected(self):
        self.write(self.rows['path'], 'altered corpus row\n'); self.assert_bad()

    def test_rewritten_scientific_manifest_and_sidecar_cannot_replace_pin(self):
        p = self.write(r.SECTION5_MANIFEST_PATH, '{}')
        self.write(r.SECTION5_MANIFEST_PATH.removesuffix('.json')+'.sha256', sha256_file(p)); self.assert_bad()

    def test_preparation_tampering_is_rejected(self):
        self.write(r.SECTION7_PREPARATION_PATH, '{}'); self.assert_bad()

    def test_added_journal_segment_is_rejected(self):
        self.write(str(Path(self.payloads['section6_gate_manifest']).parent/'journal_extra.jsonl').replace('\\','/'), 'extra\n'); self.assert_bad()

    def test_v1_negative_attempt_cannot_gain_a_release(self):
        self.write(r.SECTION6_RELEASE_MANIFEST_PATH, '{}'); self.assert_bad()

    def test_missing_archived_binding_is_rejected(self):
        (self.root/r.SECTION7_UUID_ARCHIVE/'large_qubit/section7.md').unlink(); self.assert_bad()

    def test_historical_unit_log_tampering_is_rejected(self):
        self.write(self.unit_log['path'], 'different test output'); self.assert_bad()

    def test_implicit_screening_artifact_tampering_is_rejected(self):
        self.write(self.screening, '{}'); self.assert_bad()

    def test_compiled_cache_tampering_is_rejected(self):
        self.write(self.cache_payload['path'], 'changed compiled rows'); self.assert_bad()

    def test_non_documentary_runtime_change_is_rejected(self):
        self.write('large_qubit/runtime.py', '# changed training policy'); self.assert_bad()


if __name__ == '__main__':
    unittest.main()
