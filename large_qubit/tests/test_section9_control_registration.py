"""CPU-only closure checks for the additive execution-control registration."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from large_qubit import section9_control_registration as registration
from large_qubit.common import sha256_file


class ControlRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'large_qubit/control_fixture.py'
        source.parent.mkdir(parents=True)
        source.write_text('# fixed fixture\n')
        (root / 'research_plan_revised.md').write_text('fixed research plan\n')
        return source

    def test_round_trip_does_not_rewrite_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with mock.patch.object(registration, 'SOURCES', (source.relative_to(root).as_posix(),)), \
                 mock.patch.object(registration, 'predecessor', return_value={'authenticated': True}):
                result = registration.register(root)
                path = root / registration.REGISTRATION
                before = path.read_bytes()
                self.assertTrue(result['all_passed'])
                self.assertFalse(result['training_executed'])
                self.assertFalse(result['scientific_evaluation_executed'])
                self.assertEqual(registration.register(root)['registration_sha256'], sha256_file(path))
                self.assertEqual(before, path.read_bytes())

    def test_changed_source_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with mock.patch.object(registration, 'SOURCES', (source.relative_to(root).as_posix(),)), \
                 mock.patch.object(registration, 'predecessor', return_value={'authenticated': True}):
                registration.register(root)
                source.write_text('# altered fixture\n')
                with self.assertRaisesRegex(ValueError, 'implementation or original recovery evidence changed'):
                    registration.verify(root)

    def test_changed_predecessor_rejected_before_downstream_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predecessor = root / registration.PREDECESSOR
            predecessor.parent.mkdir(parents=True)
            predecessor.write_text('{}\n')
            with mock.patch('large_qubit.section9_recovery_registration.verify') as check:
                with self.assertRaisesRegex(ValueError, 'recovery registration changed'):
                    registration.predecessor(root)
                check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
