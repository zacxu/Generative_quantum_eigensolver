"""CPU fixtures for staged pause-before-install, strict scope and rollback."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit import section9_update as update
from large_qubit import section9_operations_registration as registration
from large_qubit.common import canonical_bytes, sha256_file
from large_qubit.section8 import save


class StagedUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / update.STAGE
        self.stage.mkdir(parents=True)
        self.events = []
        self.write(self.root, 'research_plan_revised.md', b'fixed plan\n')
        self.original = {}
        for name in update.ARCHIVED_SOURCES + ('large_qubit/section9_ledger.py',):
            self.original[name] = ('original ' + name + '\n').encode()
            self.write(self.root, name, self.original[name])
        historical = 'large_qubit/registrations/original_fixture/manifest.json'
        history = {'research_plan_sha256': sha256_file(self.root / 'research_plan_revised.md'),
                   'bindings_sha256': {name: sha256_file(self.root / name) for name in self.original}}
        old_path = self.write(self.root, historical, canonical_bytes(history) + b'\n')
        old_sha = sha256_file(old_path)
        old_path.with_suffix('.sha256').write_text(old_sha + '\n', encoding='ascii')
        self.anchors = {historical: old_sha}
        patcher = patch.object(update, 'HISTORICAL_MANIFESTS', self.anchors)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in update.SOURCES:
            self.write(self.stage, name, ('amended ' + name + '\n').encode())
        archives = {}
        for name in update.ARCHIVED_SOURCES:
            archive_name = update.ARCHIVE + '/' + name
            self.write(self.stage, archive_name, self.original[name])
            archives[name] = {'path': archive_name, 'sha256': history['bindings_sha256'][name],
                              'registration': historical}
        self.write(self.stage, update.GUIDE, b'one clear guide\n')
        manifest = {'schema_version': 'section9-operations-amendment-v2',
                    'historical_manifest_sha256': self.anchors,
                    'bindings_sha256': {name: sha256_file(self.stage / name) for name in update.SOURCES},
                    'archived_bindings': archives,
                    'unchanged_live_bindings_sha256': {
                        'large_qubit/section9_ledger.py': history['bindings_sha256']['large_qubit/section9_ledger.py']},
                    'research_plan_sha256': history['research_plan_sha256'],
                    'training_executed': False, 'scientific_evaluation_executed': False}
        path = self.write(self.stage, update.REGISTRATION, canonical_bytes(manifest) + b'\n')
        path.with_suffix('.sha256').write_text(sha256_file(path) + '\n', encoding='ascii')

    def write(self, base, name, value):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def assert_original(self):
        for name, expected in self.original.items():
            self.assertEqual((self.root / name).read_bytes(), expected, name)
        self.assertFalse((self.root / update.REGISTRATION).exists())

    def fixture_pause(self, result=None):
        coordinator = SimpleNamespace(pid=123, start_ticks=456,
                                      module='large_qubit.section9_recovery', argv=('fixture',))
        def pause(*args):
            self.assert_original()
            self.events.append('paused')
            return {'paused': True} if result is None else result
        return SimpleNamespace(
            _pause_interruptions=nullcontext, _already_paused=lambda root: None,
            discover=lambda root: ('co2-s9-recovery-matrix_0001', coordinator,
                                   root / update.MATRIX / 'execution_attempts/attempt_0001'),
            _process=lambda pid: SimpleNamespace(pid=pid, start_ticks=789),
            _legacy_pause=pause, _cooperative_pause=pause,
            _processes=lambda: {}, _is_work_process=lambda process, root: False)

    @contextmanager
    def launch_fixture(self, pause=None):
        pause = pause or self.fixture_pause()
        def authenticate(root):
            self.assert_original()
            self.events.append('authenticated')
        @contextmanager
        def leases(root):
            self.assertTrue('paused' in self.events or (root / update.MATRIX / 'completed.json').exists())
            self.events.append('leases')
            yield
        def verified(root):
            self.assertIn('leases', self.events)
            self.assertTrue(update.validate_stage(root, self.stage)['installed'])
            self.events.append('verified')
            return {'all_passed': True}
        with patch.object(update.platform, 'system', return_value='Linux'), \
             patch('large_qubit.training._run_lock', side_effect=lambda path: nullcontext()), \
             patch('large_qubit.section9_control.verify_binding', side_effect=authenticate), \
             patch('large_qubit.section9_control._leases', side_effect=leases), \
             patch.object(update, '_load_staged_pause', return_value=pause), \
             patch.object(update, '_fresh_verification', side_effect=verified):
            yield pause

    def test_installer_and_registration_allowlists_are_identical(self):
        self.assertEqual(update.SOURCES, registration.SOURCES)
        self.assertEqual(update.ARCHIVED_SOURCES, registration.ARCHIVED_SOURCES)
        self.assertEqual(update.ARCHIVE, registration.ARCHIVE)
        self.assertEqual(update.REGISTRATION, registration.REGISTRATION)

    def test_stage_validation_preserves_all_live_files(self):
        result = update.validate_stage(self.root, self.stage)
        self.assertTrue(result['all_passed'])
        self.assertFalse(result['installed'])
        self.assert_original()

    def test_tampered_stage_refuses_before_import_or_pause(self):
        self.write(self.stage, 'large_qubit/section9_pause.py', b'changed staged code\n')
        with self.launch_fixture(), self.assertRaisesRegex(ValueError, 'Staged source or archive changed'):
            update.update(self.root, self.stage, execute=True)
        self.assertEqual(self.events, [])
        self.assert_original()

    def test_changed_live_bound_file_refuses_before_pause(self):
        self.write(self.root, update.GUIDE, b'already overwritten guide\n')
        with self.launch_fixture(), self.assertRaisesRegex(ValueError, 'Live bound file changed before safe update'):
            update.update(self.root, self.stage, execute=True)
        self.assertEqual(self.events, [])

    def test_stage_path_cannot_redirect_outside_update_namespace(self):
        with self.assertRaisesRegex(ValueError, 'Stage the update only'):
            update.validate_stage(self.root, self.root)

    def test_no_execution_without_explicit_opt_in(self):
        with patch.object(update, 'validate_stage') as validate, self.assertRaisesRegex(ValueError, '--execute'):
            update.update(self.root, self.stage)
        validate.assert_not_called()

    def test_pause_precedes_every_live_replacement_and_preserves_science(self):
        with self.launch_fixture():
            result = update.update(self.root, self.stage, execute=True)
        self.assertEqual(self.events, ['authenticated', 'paused', 'leases', 'verified'])
        self.assertTrue(result['installed'])
        self.assertTrue(result['paused'])
        for name in update.SOURCES:
            self.assertEqual((self.root / name).read_bytes(), (self.stage / name).read_bytes())
        for name in update.OLD_GUIDES:
            self.assertFalse((self.root / name).exists())
            self.assertEqual((self.root / update.ARCHIVE / name).read_bytes(), self.original[name])
        self.assertEqual((self.root / update.GUIDE).read_bytes(), b'one clear guide\n')
        self.assertEqual((self.root / 'large_qubit/section9_ledger.py').read_bytes(),
                         self.original['large_qubit/section9_ledger.py'])
        self.assertFalse((self.root / update.MATRIX).exists())

    def test_waiting_response_does_not_install(self):
        with self.launch_fixture(self.fixture_pause({'paused': False, 'pause_requested': True})), \
             self.assertRaisesRegex(ValueError, 'confirmed clean pause'):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()

    def test_pause_interruption_does_not_install(self):
        pause = self.fixture_pause()
        pause._legacy_pause = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.launch_fixture(pause), self.assertRaises(KeyboardInterrupt):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()

    def test_owned_process_after_pause_refuses_installation(self):
        pause = self.fixture_pause()
        pause._processes = lambda: {1: object()}
        pause._is_work_process = lambda process, root: True
        with self.launch_fixture(pause), self.assertRaisesRegex(ValueError, 'processes are still active'):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()

    def test_copy_failure_rolls_back_every_changed_file(self):
        actual = update._atomic_bytes
        calls = [0]
        def fail_once(path, value, mode):
            calls[0] += 1
            if calls[0] == 5:
                raise OSError('synthetic disk failure')
            return actual(path, value, mode)
        with self.launch_fixture(), patch.object(update, '_atomic_bytes', side_effect=fail_once), \
             self.assertRaisesRegex(OSError, 'synthetic disk failure'):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()
        self.assertFalse((self.root / 'large_qubit/section9_operations_registration.py').exists())

    def test_fresh_verification_failure_rolls_back_removed_guides_and_sources(self):
        with self.launch_fixture(), patch.object(update, '_fresh_verification', side_effect=ValueError('fresh verify failed')), \
             self.assertRaisesRegex(ValueError, 'fresh verify failed'):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()
        self.assertFalse((self.root / 'large_qubit/section9_operations_registration.py').exists())

    def test_unlisted_installation_path_is_refused(self):
        checked = update.validate_stage(self.root, self.stage)
        checked['files']['large_qubit/main/forbidden.json'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fixed operational allowlist'):
            update.install_files(self.root, self.stage, checked, self.root / update.CONTROL / 'updates/fixture')
        self.assert_original()

    def test_completed_matrix_is_authenticated_before_install_and_never_resumed(self):
        save(self.root / update.MATRIX / 'completed.json', {'all_trajectories_successful': True})
        pause = self.fixture_pause()
        with self.launch_fixture(pause), \
             patch('large_qubit.section9_control.verify_matrix',
                   return_value={'all_passed': True, 'all_trajectories_successful': True}) as verify:
            result = update.update(self.root, self.stage, execute=True)
        verify.assert_called_once_with(self.root.resolve())
        self.assertTrue(result['matrix_completed'])
        self.assertTrue(result['installed'])
        self.assertFalse(result['paused'])
        self.assertNotIn('paused', self.events)

    def test_false_completed_authentication_keeps_original_live_files(self):
        save(self.root / update.MATRIX / 'completed.json', {'all_trajectories_successful': True})
        with self.launch_fixture(), \
             patch('large_qubit.section9_control.verify_matrix',
                   return_value={'all_passed': True, 'all_trajectories_successful': False}), \
             self.assertRaisesRegex(ValueError, 'did not authenticate successfully'):
            update.update(self.root, self.stage, execute=True)
        self.assert_original()


if __name__ == '__main__':
    unittest.main()
