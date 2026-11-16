"""The GPU-selection docs snapshot stays historical after a UUID repair."""

import unittest
from unittest.mock import patch

from large_qubit import registration as r
from large_qubit.common import read_json, sha256_file
from large_qubit.tests import test_section7_uuid_registration as fixtures


class Section7GPUDocumentationTests(unittest.TestCase):
    setUp = fixtures.Section7UUIDRegistrationTests.setUp
    write = fixtures.Section7UUIDRegistrationTests.write
    pin = fixtures.Section7UUIDRegistrationTests.pin
    pin_module = fixtures.Section7UUIDRegistrationTests.pin_module
    snapshot = fixtures.Section7UUIDRegistrationTests.snapshot

    def test_documentation_snapshot_retains_its_original_id_under_new_default(self):
        self.assertEqual(r.DEFAULT_REGISTRATION, 'large_qubit/registrations/sections_1_7_v2_guide_v1')
        self.assertEqual(r.REGISTRATION_ID, 'sections_1_7_v2')
        report = r.verify_registration(self.root, r.SECTION7_GPU_DOCS_REGISTRATION)
        self.assertEqual(report['registration_id'], 'sections_1_7_v1')
        self.assertFalse(report['section7_documentation_revision']['training_code_or_policy_changed'])
        self.assertFalse(report['execution_ready'])

    def test_original_registration_verifies_archive_after_current_guide_edit(self):
        self.write('large_qubit/section7.md', 'Current UUID guidance')
        report = r.verify_registration(self.root, r.SECTION7_INITIAL_REGISTRATION)
        self.assertTrue(report['all_passed'])
        self.assertFalse(report['active_source_verified_as_original'])
        self.assertFalse(report['section_7_training_complete'])

    def test_current_guide_and_both_amendment_paths_are_static_bindings(self):
        self.write('large_qubit/protocol.py', '# current protocol')
        for relative in (*r.BOUND_DOCUMENTS, *r.SECTION3_DOCUMENTS, *r.SECTION4_DOCUMENTS, *r.SECTION5_DOCUMENTS):
            if not (self.root/relative).exists():
                self.write(relative, 'Synthetic static declaration')
        guide = self.write('large_qubit/section7.md', 'Current UUID instructions')
        self.write(r.SECTION7_UUID_AMENDMENT, '{}')
        bindings = r.current_bindings(self.root)
        self.assertIn(r.SECTION7_GPU_DOCS_AMENDMENT, bindings)
        self.assertIn(r.SECTION7_UUID_AMENDMENT, bindings)
        self.assertEqual(bindings['large_qubit/section7.md'], sha256_file(guide))

    def test_missing_original_archive_blocks_documentation_lineage(self):
        (self.root/r.SECTION7_INITIAL_ARCHIVE/'large_qubit/section7.py').unlink()
        with self.assertRaisesRegex(ValueError, 'archived binding is missing'):
            r.section7_documentation_binding(self.root)

    def test_modified_original_archive_is_rejected(self):
        self.write(r.SECTION7_INITIAL_ARCHIVE+'/large_qubit/section7.py', 'changed')
        with self.assertRaisesRegex(ValueError, 'archived binding changed'):
            r.verify_registration(self.root, r.SECTION7_INITIAL_REGISTRATION)

    def test_rewritten_original_manifest_and_sidecar_do_not_move_anchor(self):
        path = self.write(r.SECTION7_INITIAL_REGISTRATION+'/manifest.json', '{}')
        self.write(r.SECTION7_INITIAL_REGISTRATION+'/manifest.sha256', sha256_file(path))
        with self.assertRaisesRegex(ValueError, 'immutable anchor'):
            r.verify_section7_initial_registration(self.root)

    def test_successor_preparation_is_independent_of_legacy_saved_preparation(self):
        self.write(r.SECTION7_PREPARATION_PATH, {'new_technical_source': 'UUID normalizer'})
        self.assertTrue(r.verify_section7_initial_registration(self.root)['all_passed'])

    def test_current_source_verifier_is_not_used_for_historical_snapshot(self):
        # The fixture installs a current verifier that raises if invoked.
        self.write('large_qubit/section7.py', '# changed current technical implementation')
        self.assertEqual(r.section7_documentation_binding(self.root), self.documentation)

    def test_modified_amendment_cannot_replace_its_anchor(self):
        self.write(r.SECTION7_GPU_DOCS_AMENDMENT, '{}')
        with self.assertRaisesRegex(ValueError, 'immutable anchor'):
            r.section7_documentation_binding(self.root)

    def test_amendment_stays_documentation_only_even_with_trusted_fixture_pin(self):
        original = read_json(self.root/r.SECTION7_GPU_DOCS_AMENDMENT)
        for key in ('scientific_settings_changed', 'training_code_or_policy_changed',
                    'gpu_execution_authorized_by_this_documentation_request'):
            with self.subTest(key=key):
                path = self.write(r.SECTION7_GPU_DOCS_AMENDMENT, dict(original, **{key: True}))
                with patch.object(r, 'SECTION7_GPU_DOCS_AMENDMENT_SHA256', sha256_file(path)):
                    with self.assertRaisesRegex(ValueError, 'declared scope'):
                        r.section7_documentation_binding(self.root)

    def test_absent_documentation_amendment_is_not_silently_fabricated(self):
        (self.root/r.SECTION7_GPU_DOCS_AMENDMENT).unlink()
        self.assertIsNone(r.section7_documentation_binding(self.root))


if __name__ == '__main__':
    unittest.main()
