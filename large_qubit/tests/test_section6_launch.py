"""Execute the wrapper against a fake CLI and tmux; never start acquisition."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash') if os.name != 'nt' else None


def shell_path(path):
    text = Path(path).resolve().as_posix()
    return '/' + text[0].lower() + text[2:] if os.name == 'nt' and text[1:2] == ':' else text


@unittest.skipUnless(BASH, 'A local standalone Bash is required; WSL is never started by this test')
class AcquisitionWrapperTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'co2 project with spaces'
        self.root.mkdir()
        self.script = self.root / 'large_qubit/scripts/run_section6_tmux.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / 'large_qubit/scripts/run_section6_tmux.sh', self.script)
        shutil.copyfile(PROJECT / 'large_qubit/common.py', self.root / 'large_qubit/common.py')
        (self.root / 'large_qubit/__init__.py').write_text('', encoding='utf-8')
        (self.root / 'large_qubit/__main__.py').write_text(
            'import json,sys\nprint(json.dumps({"fake_cli_arguments":sys.argv[1:]}))\n', encoding='utf-8')
        self.private = self.root / 'large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/section6/section6_v1'
        self.private.mkdir(parents=True)
        (self.private / 'reservations.json').write_text('{}\n', encoding='ascii')
        self.base = self.root / 'large_qubit/evidence/section6_runs/section6_v1'
        self.public = self.root / 'large_qubit/validation/section6_v1'
        fakebin = self.root / 'fakebin'
        fakebin.mkdir()
        fake = fakebin / 'tmux'
        fake.write_text('''#!/usr/bin/env bash
set -eu
case "$1" in
    has-session) [[ "${TMUX_TEST_ACTIVE:-0}" == 1 ]] ;;
    -V) printf 'tmux synthetic-test-only\\n' ;;
    new-session)
        printf '%s\\0' "$@" >> "$TMUX_TEST_LOG"
        [[ "${TMUX_TEST_FAIL:-0}" == 0 ]]
        ;;
    list-panes) printf 'synthetic-pane\\n' ;;
    *) exit 2 ;;
esac
''', encoding='utf-8', newline='\n')
        fake.chmod(0o755)
        self.mock_log = self.root / 'tmux_calls.bin'
        self.environment = {**os.environ, 'CHEMISTRY_PYTHON': shell_path(sys.executable),
            'TMUX_TEST_LOG': shell_path(self.mock_log),
            'PATH': shell_path(fakebin) + ':/usr/bin:/bin'}

    def call(self, *arguments, **extra_env):
        return subprocess.run([BASH, str(self.script), *arguments], cwd=self.root,
                              env={**self.environment, **extra_env}, capture_output=True, text=True, timeout=30)

    def test_shell_syntax_and_default_preview_perform_no_detached_execution(self):
        syntax = subprocess.run([BASH, '-n', str(self.script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        preview = self.call()
        self.assertEqual(preview.returncode, 0, preview.stderr)
        arguments = json.loads(preview.stdout)['fake_cli_arguments']
        self.assertEqual(arguments, ['section6', '--output', 'large_qubit/validation/section6_v1'])
        self.assertNotIn('--execute', arguments)
        self.assertFalse(self.base.exists())
        self.assertFalse(self.mock_log.exists())

    def test_launch_and_explicit_resume_create_new_attempt_without_overwrite(self):
        first = self.call('launch')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('attempt=attempt_0001', first.stdout)
        saved_launch = (self.base / 'attempt_0001/launch.txt').read_bytes()
        second_launch = self.call('launch')
        self.assertNotEqual(second_launch.returncode, 0)
        self.public.mkdir(parents=True)
        (self.private / 'journal_0001.jsonl').write_text('', encoding='ascii')
        resumed = self.call('resume')
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn('attempt=attempt_0002', resumed.stdout)
        self.assertEqual((self.base / 'attempt_0001/launch.txt').read_bytes(), saved_launch)
        self.assertIn('mode=resume', (self.base / 'attempt_0002/launch.txt').read_text())
        argv = self.mock_log.read_bytes().split(b'\0')
        self.assertIn(b'attempt_0002', argv)
        self.assertIn(b'resume', argv)

    def test_resume_worker_passes_flag_and_preserves_its_completion_markers(self):
        self.assertEqual(self.call('launch').returncode, 0)
        self.public.mkdir(parents=True)
        (self.private / 'journal_0001.jsonl').write_text('', encoding='ascii')
        resumed = self.call('resume')
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        result = self.call('__worker', 'section6_v1', 'attempt_0002', 'resume')
        self.assertEqual(result.returncode, 0, result.stderr)
        attempt = self.base / 'attempt_0002'
        self.assertEqual((attempt / 'worker.exit_code').read_text().strip(), '0')
        self.assertTrue((attempt / 'worker.success').exists())
        log = (attempt / 'worker.log').read_bytes()
        arguments = json.loads(log.decode().splitlines()[0])['fake_cli_arguments']
        self.assertIn('--execute', arguments)
        self.assertIn('--resume', arguments)
        repeated = self.call('__worker', 'section6_v1', 'attempt_0002', 'resume')
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual((attempt / 'worker.log').read_bytes(), log)
        self.assertEqual((attempt / 'worker.exit_code').read_text().strip(), '0')

    def test_live_session_missing_resume_and_unsafe_versions_refuse_before_new_attempt(self):
        live = self.call('launch', TMUX_TEST_ACTIVE='1')
        self.assertNotEqual(live.returncode, 0)
        self.assertFalse(self.base.exists())
        self.assertNotEqual(self.call('resume').returncode, 0)
        for version in ('../other', 'section6_v1/escape', 'section6_bad;touch-payload'):
            with self.subTest(version=version):
                self.assertNotEqual(self.call('launch', version).returncode, 0)
        self.assertFalse(self.base.exists())

    def test_failed_tmux_creation_is_preserved_and_retry_uses_new_attempt(self):
        failed = self.call('launch', TMUX_TEST_FAIL='1')
        self.assertNotEqual(failed.returncode, 0)
        first = self.base / 'attempt_0001'
        self.assertTrue((first / 'launch.failed').exists())
        log = (first / 'worker.log').read_bytes()
        retried = self.call('launch')
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertTrue((self.base / 'attempt_0002/launch.txt').exists())
        self.assertEqual((first / 'worker.log').read_bytes(), log)
        self.assertTrue((first / 'launch.failed').exists())


if __name__ == '__main__':
    unittest.main()
