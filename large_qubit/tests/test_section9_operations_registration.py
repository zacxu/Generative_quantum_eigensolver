"""CPU-only checks for exact archives and the separate operational amendment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_operations_registration as registration
from large_qubit.common import canonical_bytes, read_json, sha256_file


class OperationsRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guide = 'large_qubit/section9.md'
        self.replacement = 'large_qubit/section9_pause.py'
        self.science = 'large_qubit/section9_ledger.py'
        self.helper = 'large_qubit/section9_operations_registration.py'
        for name, value in ((self.guide, b'original guide\n'),
                            (self.replacement, b'original pause implementation\n'),
                            (self.science, b'unchanged scientific implementation\n')):
            self.write(name, value)
        self.write('research_plan_revised.md', b'fixed research plan\n')
        historical = 'large_qubit/registrations/fixture_original/manifest.json'
        value = {'research_plan_sha256': sha256_file(self.root / 'research_plan_revised.md'),
                 'bindings_sha256': {name: sha256_file(self.root / name)
                                     for name in (self.guide, self.replacement, self.science)}}
        self.write(historical, canonical_bytes(value) + b'\n')
        path = self.root / historical
        path.with_suffix('.sha256').write_text(sha256_file(path) + '\n', encoding='ascii')
        self.historical_path = path
        self.historical_bytes = path.read_bytes()
        self.historical_sha = sha256_file(path)
        self.original_sources = value['bindings_sha256']
        for name in (self.guide, self.replacement):
            self.write(registration.ARCHIVE + '/' + name, (self.root / name).read_bytes())
        self.write(self.guide, b'one consolidated human runbook\n')
        self.write(self.replacement, b'amended pause implementation\n')
        self.write(self.helper, b'archive aware verifier\n')
        for patcher in (
            patch.object(registration, 'HISTORICAL_MANIFESTS', {historical: self.historical_sha}),
            patch.object(registration, 'ARCHIVED_SOURCES', (self.guide, self.replacement)),
            patch.object(registration, 'SOURCES', (self.replacement, self.helper)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def rewrite_amendment(self, mutation):
        path = self.root / registration.REGISTRATION
        value = read_json(path)
        mutation(value)
        path.write_bytes(canonical_bytes(value) + b'\n')
        path.with_suffix('.sha256').write_text(sha256_file(path) + '\n', encoding='ascii')

    def test_additive_registration_preserves_original_bytes_and_uses_no_old_verifier(self):
        with patch('large_qubit.section9_control_registration.verify',
                   side_effect=AssertionError('recursive verifier invoked')):
            first = registration.register(self.root)
            path = self.root / registration.REGISTRATION
            saved = path.read_bytes()
            self.assertEqual(first, registration.register(self.root))
            self.assertEqual(saved, path.read_bytes())
        self.assertEqual(self.historical_bytes, self.historical_path.read_bytes())
        self.assertTrue(first['all_passed'])
        self.assertFalse(first['training_executed'])
        self.assertFalse(first['scientific_evaluation_executed'])
        self.assertFalse((self.root / 'large_qubit/main').exists())
        self.assertEqual(registration.amendment_sha(self.root), sha256_file(path))

    def test_only_explicit_archives_resolve_historical_hashes(self):
        registration.register(self.root)
        for name in (self.guide, self.replacement):
            self.assertEqual(registration.source_hash(self.root, name), self.original_sources[name])
            self.assertNotEqual(registration.source_hash(self.root, name), sha256_file(self.root / name))
        self.assertEqual(registration.source_hash(self.root, self.science), sha256_file(self.root / self.science))

    def test_current_human_runbook_can_be_edited_without_changing_execution_registration(self):
        registration.register(self.root)
        original_sha = registration.amendment_sha(self.root)
        self.write(self.guide, b'clearer instructions; no executable protocol change\n')
        self.assertEqual(registration.amendment_sha(self.root), original_sha)
        (self.root / self.guide).unlink()
        self.assertEqual(registration.source_hash(self.root, self.guide), self.original_sources[self.guide])

    def test_changed_archive_is_refused(self):
        registration.register(self.root)
        self.write(registration.ARCHIVE + '/' + self.guide, b'altered old instructions\n')
        with self.assertRaisesRegex(ValueError, 'Historical source archive changed'):
            registration.verify(self.root)

    def test_missing_archive_is_refused(self):
        registration.register(self.root)
        (self.root / registration.ARCHIVE / self.replacement).unlink()
        with self.assertRaises(FileNotFoundError):
            registration.source_hash(self.root, self.replacement)

    def test_unchanged_science_must_match_its_live_path_even_with_an_exact_archive(self):
        registration.register(self.root)
        self.write(registration.ARCHIVE + '/' + self.science, (self.root / self.science).read_bytes())
        self.write(self.science, b'altered science\n')
        with self.assertRaisesRegex(ValueError, 'Unchanged historical source changed at its live path'):
            registration.verify(self.root)

    def test_active_replacement_and_verifier_are_both_bound(self):
        registration.register(self.root)
        for name in (self.replacement, self.helper):
            with self.subTest(name=name):
                original = (self.root / name).read_bytes()
                self.write(name, original + b'alteration\n')
                with self.assertRaisesRegex(ValueError, 'operations implementation or historical evidence changed'):
                    registration.source_hash(self.root, self.guide)
                self.write(name, original)

    def test_historical_manifest_anchor_cannot_be_redefined_with_its_checksum(self):
        registration.register(self.root)
        self.historical_path.write_bytes(self.historical_bytes + b' ')
        self.historical_path.with_suffix('.sha256').write_text(
            sha256_file(self.historical_path) + '\n', encoding='ascii')
        with self.assertRaisesRegex(ValueError, 'Historical Section9 registration changed'):
            registration.verify(self.root)

    def test_amendment_checksum_and_policy_are_checked(self):
        registration.register(self.root)
        path = self.root / registration.REGISTRATION
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'amendment checksum changed'):
            registration.verify(self.root)
        path.write_bytes(original)
        self.rewrite_amendment(lambda value: value['policy'].update({'scientific_protocol': 'reset budgets'}))
        with self.assertRaisesRegex(ValueError, 'operations implementation or historical evidence changed'):
            registration.verify(self.root)

    def test_manifest_cannot_expand_archive_substitutions(self):
        registration.register(self.root)
        self.rewrite_amendment(lambda value: value['archived_bindings'].update({
            self.science: {'path': registration.ARCHIVE + '/' + self.science,
                          'sha256': self.original_sources[self.science],
                          'registration': self.historical_path.relative_to(self.root).as_posix()}}))
        with self.assertRaisesRegex(ValueError, 'operations implementation or historical evidence changed'):
            registration.verify(self.root)

    def test_missing_amendment_gives_sync_instruction(self):
        with self.assertRaisesRegex(ValueError, 'sync its registration and archived sources'):
            registration.source_hash(self.root, self.guide)

    def test_unamended_checkout_hashes_live_bytes_without_substitution(self):
        (self.root / registration.ARCHIVE / self.guide).unlink()
        self.assertEqual(registration.source_hash(self.root, self.guide), sha256_file(self.root / self.guide))
        self.assertNotEqual(registration.source_hash(self.root, self.guide), self.original_sources[self.guide])

    def test_partial_sync_does_not_create_an_amendment(self):
        (self.root / self.helper).unlink()
        with self.assertRaises(FileNotFoundError):
            registration.register(self.root)
        self.assertFalse((self.root / registration.REGISTRATION).exists())

    def test_original_research_plan_remains_bound(self):
        registration.register(self.root)
        self.write('research_plan_revised.md', b'changed science plan\n')
        with self.assertRaisesRegex(ValueError, 'Registered research plan changed'):
            registration.verify(self.root)


class ExistingRegistrationCompatibilityTests(unittest.TestCase):
    def test_original_manifest_still_verifies_after_guide_and_verifier_archival(self):
        from large_qubit import section9_registration as original
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ('large_qubit/section9.md', 'large_qubit/section9_registration.py',
                     'large_qubit/scientific_fixture.py')
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original ' + name + '\n', encoding='utf-8')
            (root / 'research_plan_revised.md').write_text('fixed plan\n', encoding='utf-8')
            with patch.object(original, 'SOURCES', names), \
                 patch.object(original, 'predecessor', return_value={'authenticated_fixture': True}):
                with patch.object(original, 'source_hash', side_effect=lambda folder, name: sha256_file(folder / name)):
                    original.register(root)
                original_path = root / original.REGISTRATION
                original_bytes, original_sha = original_path.read_bytes(), sha256_file(original_path)
                for name in names[:2]:
                    archive = root / registration.ARCHIVE / name
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes((root / name).read_bytes())
                    (root / name).write_text('replacement ' + name + '\n', encoding='utf-8')
                with patch.object(registration, 'HISTORICAL_MANIFESTS', {original.REGISTRATION: original_sha}), \
                     patch.object(registration, 'ARCHIVED_SOURCES', names[:2]), \
                     patch.object(registration, 'SOURCES', (names[1],)):
                    registration.register(root)
                    result = original.verify(root)
                    self.assertTrue(result['all_passed'])
                    self.assertEqual(result['registration_sha256'], original_sha)
                    self.assertEqual(original_path.read_bytes(), original_bytes)
                    (root / names[2]).write_text('changed live science\n', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Unchanged historical source changed at its live path'):
                        original.verify(root)


if __name__ == '__main__':
    unittest.main()
