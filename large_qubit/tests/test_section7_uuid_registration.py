"""Legacy implementation proofs survive a technical GPU-UUID successor."""

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit import registration as r
from large_qubit.common import sha256_file


class Section7UUIDRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache_calls = []
        prior = self.write(r.SECTION6_V2_REGISTRATION + '/manifest.json', {'prior': 'completed'})
        self.pin('SECTION6_V2_MANIFEST_SHA256', sha256_file(prior))
        self.write(r.SECTION6_V2_REGISTRATION + '/manifest.sha256', sha256_file(prior))
        source_names = ('large_qubit/section7.py', 'large_qubit/training_data.py',
                        'large_qubit/training_cache_access.py', 'phase6/models.py')
        self.source_hashes = {name: sha256_file(self.write(name, '# original ' + name)) for name in source_names}
        policy = self.write('large_qubit/section7_policy.json', {'unchanged': 'optimizer and objectives'})
        audit = self.write('large_qubit/evidence/sections_1_7_v1/data_validation.json', {'all_passed': True})
        self.caches, self.checks = {}, {}
        for key in r.MODEL_REVISIONS:
            relative = f'large_qubit/training_data/section7_v1/{key}/manifest.json'
            manifest = self.write(relative, {'native': key})
            fingerprint = sha256_file(manifest)
            self.write(relative.removesuffix('.json') + '.sha256', fingerprint)
            payload = self.write(f'large_qubit/training_data/section7_v1/{key}/rows.jsonl', 'native rows\n')
            self.caches[key] = {'manifest_path': relative, 'manifest_sha256': fingerprint}
            self.checks[key] = {'all_passed': True, 'model_key': key, 'rows': 6144, 'payload_sha256': sha256_file(payload)}
        self.prep = {'schema_version': 'section7-preparation-v1', 'source_bindings': self.source_hashes,
            'production_model_weights_loaded': False, 'production_training_executed': False,
            'production_gpu_preflight_executed': False, 'energy_requests': 0,
            'prior_registration_sha256': r.SECTION6_V2_MANIFEST_SHA256,
            'policy_path': 'large_qubit/section7_policy.json', 'policy_sha256': sha256_file(policy),
            'deep_native_cache_audit': {'path': audit.relative_to(self.root).as_posix(), 'sha256': sha256_file(audit)},
            'release_sha256': 'a'*64, 'token_caches': self.caches}
        prep = self.write(r.SECTION7_INITIAL_PREPARATION_PATH, self.prep)
        self.pin('SECTION7_INITIAL_PREPARATION_SHA256', sha256_file(prep))
        self.verification = {'all_passed': True, 'implementation_ready': True,
            'scope': 'prepared_fixed_fit_workflow_not_completed_training_or_GPU_preflight',
            'preparation_path': r.SECTION7_INITIAL_PREPARATION_PATH, 'preparation_sha256': sha256_file(prep),
            'policy_sha256': self.prep['policy_sha256'], 'token_caches': self.caches,
            'cache_verification': self.checks, 'source_bindings': self.source_hashes,
            'production_training_executed': False, 'production_gpu_preflight_executed': False,
            'production_optimizer_steps_executed': 0, 'model_weights_loaded': False,
            'energy_requests': 0, 'execution_ready': False}
        unit_path = 'large_qubit/evidence/sections_1_7_v1/unit_validation_ready.json'
        unit = self.write(unit_path, {'all_passed': True, 'exit_code': 0, 'tests_run': 67, 'skipped': 0,
            'production_training_executed': False, 'production_gpu_preflight_executed': False,
            'production_optimizer_steps_executed': 0, 'model_weights_loaded': False, 'server_connected': False})
        certificate = self.write(r.SECTION7_INITIAL_IMPLEMENTATION_PATH, {
            'schema_version': 'section7-implementation-certificate-v1', 'all_passed': True,
            'implementation_ready': True, 'production_training_executed': False,
            'production_gpu_preflight_executed': False, 'production_optimizer_steps_executed': 0,
            'model_weights_loaded': False, 'server_connected': False, 'trained_adapters': 0,
            'execution_ready': False, 'preparation': {'path': r.SECTION7_INITIAL_PREPARATION_PATH,
                'sha256': sha256_file(prep), 'verification': self.verification},
            'unit_tests': {'all_passed': True, 'tests_run': 67, 'skipped': 0,
                           'receipt': unit_path, 'sha256': sha256_file(unit)}})
        self.binding = {'preparation_path': r.SECTION7_INITIAL_PREPARATION_PATH,
            'preparation_sha256': sha256_file(prep), 'implementation_certificate_path': r.SECTION7_INITIAL_IMPLEMENTATION_PATH,
            'implementation_certificate_sha256': sha256_file(certificate),
            'saved_preparation_and_cache_verification': self.verification, 'implementation_ready': True,
            'production_training_executed': False, 'production_gpu_preflight_executed': False,
            'production_optimizer_steps_executed': 0, 'trained_adapters': 0,
            'training_not_run_scope': 'as_of_saved_implementation_certificate',
            'source_or_data_prepared_by_registration': False, 'execution_ready': False}
        bound_names = [name for name in source_names if name.startswith('large_qubit/')]
        bound_names += [r.SECTION7_INITIAL_PREPARATION_PATH, r.SECTION7_INITIAL_IMPLEMENTATION_PATH,
                        'large_qubit/section7_policy.json', *(x['manifest_path'] for x in self.caches.values())]
        bindings = {name: sha256_file(self.root/name) for name in bound_names}
        for index in range(128-len(bindings)):
            name = f'large_qubit/legacy_{index:03d}.txt'
            bindings[name] = sha256_file(self.write(name, f'preserved {index}'))
        self.initial = {'schema_version': 'large-qubit-section7-registration-v1',
            'registration_id': 'sections_1_7_v1', 'execution_ready': False,
            'bindings_sha256': bindings, 'section7_implementation': self.binding}
        self.snapshot(r.SECTION7_INITIAL_REGISTRATION, r.SECTION7_INITIAL_ARCHIVE, self.initial,
                      'SECTION7_INITIAL_MANIFEST_SHA256')

        def verify_cache(root, relative, fingerprint, **kwargs):
            key = relative.split('/')[-2]
            if sha256_file(Path(root)/relative) != fingerprint:
                raise ValueError('Live cache manifest changed')
            payload = Path(root)/f'large_qubit/training_data/section7_v1/{key}/rows.jsonl'
            if sha256_file(payload) != self.checks[key]['payload_sha256']:
                raise ValueError('Live cache payload changed')
            self.cache_calls.append(key)
            return copy.deepcopy(self.checks[key])
        self.pin_module('large_qubit.training_cache_access', SimpleNamespace(verify_training_cache=verify_cache))
        self.pin_module('large_qubit.section7', SimpleNamespace(
            verify_section7=lambda *_: (_ for _ in ()).throw(AssertionError('Current verifier called for legacy sources'))))
        amendment = {'schema_version': 'section7-gpu-selection-documentation-amendment-v1',
            'same_implementation_id': 'sections_1_7_v1', 'successor_registration': r.SECTION7_GPU_DOCS_REGISTRATION,
            'scientific_settings_changed': False, 'training_code_or_policy_changed': False,
            'gpu_execution_authorized_by_this_documentation_request': False,
            'original_registration': {'path': r.SECTION7_INITIAL_REGISTRATION+'/manifest.json',
                'sha256': r.SECTION7_INITIAL_MANIFEST_SHA256, 'bound_files': 128, 'archive_root': r.SECTION7_INITIAL_ARCHIVE},
            'training_preparation': {'path': r.SECTION7_INITIAL_PREPARATION_PATH,
                                     'sha256': r.SECTION7_INITIAL_PREPARATION_SHA256}}
        self.pin('SECTION7_GPU_DOCS_AMENDMENT_SHA256', sha256_file(self.write(r.SECTION7_GPU_DOCS_AMENDMENT, amendment)))
        self.documentation = r.section7_documentation_binding(self.root)
        self.docs = copy.deepcopy(self.initial)
        self.docs['section7_documentation_revision'] = self.documentation
        for name in [r.SECTION7_GPU_DOCS_AMENDMENT, 'large_qubit/gpu_selection.md']:
            if not (self.root/name).exists(): self.write(name, 'GPU selection instructions')
            self.docs['bindings_sha256'][name] = sha256_file(self.root/name)
        self.snapshot(r.SECTION7_GPU_DOCS_REGISTRATION, r.SECTION7_GPU_DOCS_ARCHIVE, self.docs,
                      'SECTION7_GPU_DOCS_MANIFEST_SHA256')

    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value).encode())
        return path

    def pin(self, name, value):
        context = patch.object(r, name, value)
        context.start(); self.addCleanup(context.stop)

    def pin_module(self, name, module):
        context = patch.dict(sys.modules, {name: module})
        context.start(); self.addCleanup(context.stop)

    def snapshot(self, relative, archive, saved, anchor):
        path = self.write(relative+'/manifest.json', saved)
        fingerprint = sha256_file(path)
        self.write(relative+'/manifest.sha256', fingerprint)
        self.pin(anchor, fingerprint)
        for name in saved['bindings_sha256']:
            self.write(archive+'/'+name, (self.root/name).read_bytes())

    def technical_amendment(self):
        value = {'schema_version': 'section7-h200-uuid-compatibility-amendment-v1',
            'successor_registration': r.SECTION7_UUID_REGISTRATION, 'successor_preparation': r.SECTION7_PREPARATION_PATH,
            'successor_implementation_id': r.REGISTRATION_ID, 'scientific_settings_changed': False,
            'data_or_token_caches_changed': False, 'training_objective_changed': False, 'policy_changed': False,
            'new_preflight_and_run_namespace': 'section7_gpu_uuid_v1',
            'production_gpu_execution_authorized_by_this_repair': False, 'model_downloads_required': False,
            'original_registration': {'path': r.SECTION7_GPU_DOCS_REGISTRATION+'/manifest.json',
                'sha256': r.SECTION7_GPU_DOCS_MANIFEST_SHA256, 'bound_files': 130, 'archive': r.SECTION7_GPU_DOCS_ARCHIVE},
            'original_implementation_registration': {'path': r.SECTION7_INITIAL_REGISTRATION+'/manifest.json',
                'sha256': r.SECTION7_INITIAL_MANIFEST_SHA256, 'bound_files': 128, 'archive': r.SECTION7_INITIAL_ARCHIVE},
            'original_preparation': {'path': r.SECTION7_INITIAL_PREPARATION_PATH,
                                     'sha256': r.SECTION7_INITIAL_PREPARATION_SHA256}}
        self.pin('SECTION7_UUID_AMENDMENT_SHA256', sha256_file(self.write(r.SECTION7_UUID_AMENDMENT, value)))
        return value

    def test_legacy_proofs_use_archived_sources_without_current_or_archived_execution(self):
        self.write('large_qubit/section7.py', 'raise AssertionError("new orchestration")')
        for verify, count in [(r.verify_section7_initial_registration, 128), (r.verify_section7_gpu_docs_registration, 130)]:
            result = verify(self.root)
            self.assertTrue(result['all_passed'])
            self.assertEqual(result['bound_files_verified'], count)
            self.assertFalse(result['active_source_verified_as_original'])
            self.assertFalse(result['section_7_training_complete'])
        self.assertTrue(self.cache_calls)

    def test_original_report_remains_identical_to_anchored_documentation_lineage(self):
        self.assertEqual(r.verify_section7_initial_registration(self.root), self.documentation['original_registration'])
        self.assertEqual(r.section7_documentation_binding(self.root), self.documentation)

    def test_both_legacy_dispatches_bypass_current_registration_rebuild(self):
        with patch.object(r, 'make_registration', side_effect=AssertionError('Current registration rebuilt')):
            for relative in [r.SECTION7_INITIAL_REGISTRATION, r.SECTION7_GPU_DOCS_REGISTRATION]:
                self.assertTrue(r.verify_registration(self.root, relative)['all_passed'])

    def test_changed_original_archived_source_is_rejected(self):
        self.write(r.SECTION7_INITIAL_ARCHIVE+'/large_qubit/section7.py', '# overwritten')
        with self.assertRaisesRegex(ValueError, 'archived binding changed'): r.verify_section7_initial_registration(self.root)

    def test_missing_documentation_archive_is_rejected(self):
        (self.root/r.SECTION7_GPU_DOCS_ARCHIVE/'large_qubit/gpu_selection.md').unlink()
        with self.assertRaisesRegex(ValueError, 'archived binding is missing'): r.verify_section7_gpu_docs_registration(self.root)

    def test_original_count_is_exact_even_under_a_trusted_fixture_anchor(self):
        changed = copy.deepcopy(self.initial)
        changed['bindings_sha256'].pop(next(iter(changed['bindings_sha256'])))
        self.snapshot(r.SECTION7_INITIAL_REGISTRATION, r.SECTION7_INITIAL_ARCHIVE, changed, 'SECTION7_INITIAL_MANIFEST_SHA256')
        with self.assertRaisesRegex(ValueError, 'binding count differs'): r.verify_section7_initial_registration(self.root)

    def test_rewritten_documentation_manifest_and_sidecar_do_not_replace_anchor(self):
        path = self.write(r.SECTION7_GPU_DOCS_REGISTRATION+'/manifest.json', {})
        self.write(r.SECTION7_GPU_DOCS_REGISTRATION+'/manifest.sha256', sha256_file(path))
        with self.assertRaisesRegex(ValueError, 'immutable anchor'): r.verify_section7_gpu_docs_registration(self.root)

    def test_unarchived_historical_preparation_source_must_stay_unchanged_live(self):
        self.write('phase6/models.py', '# changed historical source')
        with self.assertRaisesRegex(ValueError, 'prepared source or evidence changed'): r.verify_section7_initial_registration(self.root)

    def test_frozen_live_cache_reader_cannot_be_substituted_with_archived_copy(self):
        self.write('large_qubit/training_cache_access.py', '# changed cache reader')
        with self.assertRaisesRegex(ValueError, 'unchanged live reader'): r.verify_section7_initial_registration(self.root)

    def test_unchanged_archived_manifest_still_requires_unchanged_live_cache_payload(self):
        self.write('large_qubit/training_data/section7_v1/gpt2_xl/rows.jsonl', 'changed rows\n')
        with self.assertRaisesRegex(ValueError, 'cache payload changed'): r.verify_section7_initial_registration(self.root)

    def test_prior_unit_receipt_cannot_change_after_anchored_certificate(self):
        self.write('large_qubit/evidence/sections_1_7_v1/unit_validation_ready.json', {})
        with self.assertRaisesRegex(ValueError, 'prepared source or evidence changed'): r.verify_section7_initial_registration(self.root)

    def test_saved_cache_proof_is_reconstructed_instead_of_trusted(self):
        self.checks['gpt2_xl']['rows'] = 6143
        with self.assertRaisesRegex(ValueError, 'saved preparation/cache proof differs'): r.verify_section7_initial_registration(self.root)

    def test_technical_amendment_binds_both_legacy_snapshots_and_new_namespace(self):
        self.technical_amendment()
        report = r.section7_uuid_compatibility_binding(self.root)
        self.assertTrue(report['all_passed'])
        self.assertEqual(report['original_registration']['bound_files_verified'], 130)
        self.assertEqual(report['original_implementation_registration']['bound_files_verified'], 128)
        self.assertEqual(report['successor_implementation_id'], 'sections_1_7_v2')
        self.assertFalse(report['production_gpu_execution_authorized_by_this_repair'])

    def test_technical_amendment_cannot_change_science_even_with_fixture_pin(self):
        value = self.technical_amendment()
        value['scientific_settings_changed'] = True
        self.pin('SECTION7_UUID_AMENDMENT_SHA256', sha256_file(self.write(r.SECTION7_UUID_AMENDMENT, value)))
        with self.assertRaisesRegex(ValueError, 'technical-only scope'): r.section7_uuid_compatibility_binding(self.root)


if __name__ == '__main__':
    unittest.main()
