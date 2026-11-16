"""Run the operations launcher with synthetic Python modules and fake tmux only."""

import hashlib
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
BASH = (str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file()
        else shutil.which('bash') if os.name != 'nt' else None)
PAUSE_SESSION = 'co2-s9-pause-matrix_0001'
RUN_SESSION = 'co2-s9-control-matrix_0001'
UPDATE_SESSION = 'co2-s9-update-matrix_0001'


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return '/' + value[0].lower() + value[2:] if os.name == 'nt' and value[1:2] == ':' else value


@unittest.skipUnless(BASH, 'A standalone Bash is required; this test never starts WSL')
class OperationsLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'section9 fixture with spaces'
        self.script = self.root / 'large_qubit/scripts/run_section9_control_tmux.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / 'large_qubit/scripts/run_section9_control_tmux.sh', self.script)
        self.shell_root = subprocess.check_output([BASH, '-c', 'pwd -P'], cwd=self.root,
                                                  text=True).strip()
        package = self.root / 'large_qubit'
        (package / '__init__.py').write_text('', encoding='utf-8')
        common = '''import json, os, sys
from pathlib import Path
def record(module):
    value = {'kind':'python', 'module':module, 'arguments':sys.argv[1:],
             'original_arguments':getattr(sys, 'orig_argv', []),
             'gpu_mask':os.environ.get('CUDA_VISIBLE_DEVICES')}
    with Path(os.environ['FIXTURE_EVENTS']).open('a') as handle:
        handle.write(json.dumps(value)+'\\n')
'''
        (package / 'fixture_common.py').write_text(common, encoding='utf-8')
        (package / 'section9_operations_registration.py').write_text(
            'from .fixture_common import *\n'
            'record("operations_registration")\n'
            'print("fixture amendment verification")\n'
            'raise SystemExit(int(os.environ.get("FIXTURE_VERIFY_CODE", "0")))\n', encoding='utf-8')
        (package / 'section9_control.py').write_text(
            'from .fixture_common import *\n'
            'record("control")\n'
            'print("fixture control " + sys.argv[1])\n'
            'print(json.dumps({"paused":sys.argv[1] == "pause"}))\n'
            'raise SystemExit(int(os.environ.get("FIXTURE_CONTROL_CODE", "0")))\n', encoding='utf-8')
        (package / 'section9_pause.py').write_text(
            'def pause_status(root):\n'
            '    return {"fixture_pause_status":True,"paused":True}\n', encoding='utf-8')
        self.manifest = package / 'registrations/sections_1_9_operations_v2/manifest.json'
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"synthetic_fixture":true}\n', encoding='utf-8')
        self.pause_jobs = package / 'operations/section9_recovery_v1/control/pause_jobs'
        self.run_jobs = package / 'evidence/section9_control_runs/matrix_0001'
        self.events = self.root / 'events.jsonl'
        self.sessions = self.root / 'fake_tmux_sessions'
        self.sessions.mkdir()
        fakebin = self.root / 'fakebin'
        fakebin.mkdir()
        tmux_python = fakebin / 'fake_tmux.py'
        tmux_python.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['FIXTURE_EVENTS']).open('a') as handle:
    handle.write(json.dumps({'kind':'tmux','arguments':args})+'\\n')
sessions = Path(os.environ['FIXTURE_SESSIONS'])
if args[0] == 'has-session':
    raise SystemExit(0 if (sessions / args[args.index('-t')+1].lstrip('=')).exists() else 1)
if args[0] == 'new-session':
    if os.environ.get('FIXTURE_TMUX_FAIL') == '1':
        raise SystemExit(7)
    (sessions / args[args.index('-s')+1]).write_text('synthetic active session')
    raise SystemExit(0)
raise SystemExit('Unsupported fake tmux operation: ' + repr(args))
''', encoding='utf-8')
        fake_tmux = fakebin / 'tmux'
        fake_tmux.write_text('#!/usr/bin/env bash\nexec "$FIXTURE_PYTHON" "$FIXTURE_TMUX" "$@"\n',
                             encoding='utf-8', newline='\n')
        fake_tmux.chmod(0o755)
        self.environment = {
            **os.environ,
            'TRAINING_PYTHON': shell_path(sys.executable),
            'CHEMISTRY_PYTHON': shell_path(sys.executable),
            'FIXTURE_PYTHON': shell_path(sys.executable),
            'FIXTURE_TMUX': str(tmux_python),
            'FIXTURE_EVENTS': str(self.events),
            'FIXTURE_SESSIONS': str(self.sessions),
            'PATH': shell_path(fakebin) + ':/usr/bin:/bin',
            'CUDA_VISIBLE_DEVICES': '0,1,2,3',
        }
        self.stage = package / 'updates/section9_operations_v2'
        self.install_script = self.stage / 'large_qubit/scripts/install_section9_update_tmux.sh'
        self.install_script.parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / 'large_qubit/scripts/install_section9_update_tmux.sh', self.install_script)
        (self.stage / 'large_qubit/section9_update.py').write_text(
            common + '\nrecord("update")\n'
            'print("fixture update: installed: true; paused: true")\n'
            'raise SystemExit(int(os.environ.get("FIXTURE_UPDATE_CODE", "0")))\n', encoding='utf-8')
        self.update_jobs = package / 'operations/section9_recovery_v1/control/update_jobs'

    def call(self, *arguments, **extra_env):
        return subprocess.run([BASH, str(self.script), *arguments], cwd=self.root,
                              env={**self.environment, **extra_env}, capture_output=True,
                              text=True, timeout=30)

    def install(self, *arguments, script=None, cwd=None, **extra_env):
        return subprocess.run([BASH, str(script or self.install_script), *arguments],
                              cwd=cwd or self.root, env={**self.environment, **extra_env},
                              capture_output=True, text=True, timeout=30)

    def recorded(self, kind=None, module=None):
        rows = [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []
        return [row for row in rows if (kind is None or row['kind'] == kind)
                and (module is None or row.get('module') == module)]

    def launches(self):
        return [row['arguments'] for row in self.recorded('tmux') if row['arguments'][0] == 'new-session']

    def internal_path(self, path):
        # Git Bash's /tmp mount need not equal the /c path for the same file.
        return self.shell_root + '/' + path.relative_to(self.root).as_posix()

    def create_pause(self):
        result = self.call('pause')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        jobs = list(self.pause_jobs.glob('job_*'))
        self.assertEqual(len(jobs), 1)
        return jobs[0], result

    def assert_amendment_receipt(self, job):
        self.assertEqual((job / 'operations_amendment.sha256').read_text().strip(),
                         hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_syntax_and_read_only_commands_do_not_start_jobs(self):
        checked = subprocess.run([BASH, '-n', str(self.script)], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        for action in ('verify', 'matrix-verify'):
            result = self.call(action)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('fixture control ' + action, result.stdout)
        self.assertEqual([row['arguments'] for row in self.recorded('python', 'control')],
                         [['verify'], ['matrix-verify']])
        self.assertFalse(self.pause_jobs.exists())
        self.assertFalse(self.run_jobs.exists())
        self.assertEqual(self.launches(), [])

    def test_pause_detaches_and_worker_uses_unlimited_wait_with_exit_receipts(self):
        job, result = self.create_pause()
        self.assertIn('Checking pause launcher files', result.stdout)
        self.assertIn('Pause REQUESTED', result.stdout)
        self.assertIn('not yet paused', result.stdout)
        self.assertEqual(self.recorded('python', 'control'), [])
        self.assert_amendment_receipt(job)
        command = self.launches()[0]
        self.assertIn('-d', command)
        self.assertEqual(command[command.index('-s') + 1], PAUSE_SESSION)
        self.assertEqual(command[-2], '__pause_worker')
        self.assertEqual(Path(command[-1]).resolve(), job.resolve())
        events = self.recorded()
        verify_index = next(i for i, row in enumerate(events) if row.get('module') == 'operations_registration')
        launch_index = next(i for i, row in enumerate(events) if row['kind'] == 'tmux' and row['arguments'][0] == 'new-session')
        self.assertLess(verify_index, launch_index)
        finished = self.call('__pause_worker', self.internal_path(job))
        self.assertEqual(finished.returncode, 0, finished.stderr)
        pause = self.recorded('python', 'control')[-1]
        self.assertEqual(pause['arguments'], ['pause', '--execute', '--timeout-seconds', '0'])
        self.assertIn('-u', pause['original_arguments'])
        self.assertIn('-B', pause['original_arguments'])
        self.assertEqual((job / 'exit_code').read_text().strip(), '0')
        self.assertTrue((job / 'started_utc').is_file())
        self.assertTrue((job / 'finished_utc').is_file())
        self.assertIn('"paused": true', (job / 'worker.log').read_text())

    def test_failed_pause_worker_keeps_error_exit_and_log(self):
        job, _ = self.create_pause()
        result = self.call('__pause_worker', self.internal_path(job), FIXTURE_CONTROL_CODE='9')
        self.assertEqual(result.returncode, 9)
        self.assertEqual((job / 'exit_code').read_text().strip(), '9')
        self.assertTrue((job / 'finished_utc').is_file())
        self.assertIn('fixture control pause', (job / 'worker.log').read_text())
        self.assert_amendment_receipt(job)

    def test_repeated_pause_uses_existing_helper_without_new_attempt(self):
        job, _ = self.create_pause()
        before = (job / 'operations_amendment.sha256').read_bytes()
        repeated = self.call('pause')
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertIn('already pending', repeated.stdout)
        self.assertEqual(list(self.pause_jobs.glob('job_*')), [job])
        self.assertEqual(len(self.launches()), 1)
        self.assertEqual((job / 'operations_amendment.sha256').read_bytes(), before)

    def test_status_shows_pause_log_and_control_status_without_execution(self):
        job, _ = self.create_pause()
        (job / 'worker.log').write_text('fixture waiting for safe lane boundary\n', encoding='utf-8')
        result = self.call('status')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Pause helper: RUNNING', result.stdout)
        self.assertIn('fixture waiting for safe lane boundary', result.stdout)
        self.assertIn('fixture_pause_status', result.stdout)
        self.assertIn('fixture control status', result.stdout)
        self.assertEqual(self.recorded('python', 'control')[-1]['arguments'], ['status'])
        self.assertEqual(len(self.launches()), 1)
        (self.sessions / PAUSE_SESSION).unlink()
        (job / 'exit_code').write_text('0\n', encoding='utf-8')
        finished = self.call('pause-status')
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertIn('Pause helper: not running', finished.stdout)
        self.assertIn('Pause helper exit code: 0', finished.stdout)

    def test_pause_status_reads_authenticated_receipt_even_without_launcher_jobs(self):
        result = self.call('pause-status')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('No persistent pause job', result.stdout)
        self.assertIn('"fixture_pause_status": true', result.stdout)
        self.assertIn('"paused": true', result.stdout)
        self.assertFalse(self.pause_jobs.exists())
        self.assertEqual(self.launches(), [])

    def test_pending_pause_blocks_reassignment_and_resume(self):
        self.create_pause()
        for action in ('reassign', 'resume'):
            with self.subTest(action=action):
                result = self.call(action)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('Pause is still pending', result.stderr)
        self.assertEqual(self.recorded('python', 'control'), [])
        self.assertFalse(self.run_jobs.exists())
        self.assertEqual(len(self.launches()), 1)

    def test_failed_amendment_verification_refuses_before_operation_side_effects(self):
        for action in ('pause', 'reassign', 'resume'):
            with self.subTest(action=action):
                result = self.call(action, FIXTURE_VERIFY_CODE='11')
                self.assertEqual(result.returncode, 11, result.stdout + result.stderr)
        self.assertFalse(self.pause_jobs.exists())
        self.assertFalse(self.run_jobs.exists())
        self.assertEqual(self.launches(), [])
        self.assertEqual(self.recorded('python', 'control'), [])

    def test_public_extra_arguments_are_refused_before_any_python_or_tmux(self):
        for action in ('verify', 'pause', 'pause-status', 'reassign', 'resume', 'status', 'matrix-verify'):
            with self.subTest(action=action):
                result = self.call(action, '--timeout-seconds', '300')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('Unexpected extra arguments', result.stderr)
        self.assertEqual(self.recorded(), [])
        self.assertFalse(self.pause_jobs.exists())
        self.assertFalse(self.run_jobs.exists())

    def test_resume_preserves_control_session_gpu_mask_and_worker_arguments(self):
        result = self.call('resume', CUDA_VISIBLE_DEVICES='4,5,6,7')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        jobs = list(self.run_jobs.glob('launch_*'))
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assert_amendment_receipt(job)
        self.assertEqual((job / 'gpu_mask.txt').read_text().strip(), '4,5,6,7')
        command = self.launches()[0]
        self.assertEqual(command[command.index('-s') + 1], RUN_SESSION)
        self.assertIn('-d', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=4,5,6,7', command)
        self.assertEqual(command[-2], '__worker')
        self.assertEqual(Path(command[-1]).resolve(), job.resolve())
        finished = self.call('__worker', self.internal_path(job), CUDA_VISIBLE_DEVICES='4,5,6,7')
        self.assertEqual(finished.returncode, 0, finished.stderr)
        execution = self.recorded('python', 'control')[-1]
        self.assertEqual(execution['arguments'][:-1], ['resume', '--execute', '--chemistry-python'])
        self.assertEqual(Path(execution['arguments'][-1]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(execution['gpu_mask'], '4,5,6,7')
        self.assertEqual((job / 'exit_code').read_text().strip(), '0')

    def test_installer_requires_exact_staging_root_and_rejects_extra_arguments(self):
        checked = subprocess.run([BASH, '-n', str(self.install_script)], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        wrong_script = self.root / 'wrong_stage/large_qubit/scripts/install_section9_update_tmux.sh'
        wrong_script.parent.mkdir(parents=True)
        shutil.copyfile(self.install_script, wrong_script)
        wrong_stage = self.install('start', script=wrong_script)
        self.assertEqual(wrong_stage.returncode, 2, wrong_stage.stdout + wrong_stage.stderr)
        self.assertIn('Extract the update into', wrong_stage.stderr)
        wrong_working_directory = self.install('start', cwd=self.root.parent)
        self.assertEqual(wrong_working_directory.returncode, 2)
        for action in ('start', 'status'):
            with self.subTest(action=action):
                result = self.install(action, 'unexpected')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(self.update_jobs.exists())
        self.assertEqual(self.recorded(), [])

    def test_installer_detaches_then_worker_passes_exact_explicit_arguments(self):
        result = self.install('start')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Update/pause REQUESTED', result.stdout)
        self.assertIn('Existing evaluation files stay in place', result.stdout)
        self.assertEqual(self.recorded('python'), [])
        jobs = list(self.update_jobs.glob('job_*'))
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        command = self.launches()[0]
        self.assertIn('-d', command)
        self.assertEqual(command[command.index('-s') + 1], UPDATE_SESSION)
        self.assertEqual(command[-2], '__worker')
        self.assertEqual(Path(command[-1]).resolve(), job.resolve())
        completed = self.install('__worker', self.internal_path(job))
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        executions = self.recorded('python')
        self.assertEqual(len(executions), 1)
        execution = executions[0]
        self.assertEqual(execution['module'], 'update')
        self.assertEqual(execution['arguments'][::2], ['--root', '--stage-path', '--execute'])
        self.assertEqual(len(execution['arguments']), 5)
        self.assertEqual(Path(execution['arguments'][1]).resolve(), self.root.resolve())
        self.assertEqual(Path(execution['arguments'][3]).resolve(), self.stage.resolve())
        self.assertIn('-u', execution['original_arguments'])
        self.assertIn('-B', execution['original_arguments'])
        self.assertEqual((job / 'exit_code').read_text().strip(), '0')
        self.assertTrue((job / 'started_utc').is_file())
        self.assertTrue((job / 'finished_utc').is_file())
        self.assertIn('installed: true; paused: true', (job / 'worker.log').read_text())

    def test_installer_duplicate_keeps_one_job_and_status_shows_result(self):
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        job, = self.update_jobs.glob('job_*')
        (job / 'worker.log').write_text('fixture waiting for safe shutdown\n', encoding='utf-8')
        duplicate = self.install('start')
        self.assertEqual(duplicate.returncode, 0, duplicate.stdout + duplicate.stderr)
        self.assertIn('already running', duplicate.stdout)
        self.assertEqual(list(self.update_jobs.glob('job_*')), [job])
        self.assertEqual(len(self.launches()), 1)
        status = self.install('status')
        self.assertEqual(status.returncode, 0, status.stdout + status.stderr)
        self.assertIn('Update/pause helper: RUNNING', status.stdout)
        self.assertIn('fixture waiting for safe shutdown', status.stdout)
        (self.sessions / UPDATE_SESSION).unlink()
        (job / 'exit_code').write_text('0\n', encoding='utf-8')
        (job / 'worker.log').write_text('fixture installed: true; paused: true\n', encoding='utf-8')
        completed = self.install('status')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('Update/pause helper: not running', completed.stdout)
        self.assertIn('installed: true; paused: true', completed.stdout)
        self.assertIn('Update helper exit code: 0', completed.stdout)
        self.assertEqual(self.recorded('python'), [])

    def test_installer_worker_preserves_failed_exit_and_log(self):
        first = self.install('start')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        job, = self.update_jobs.glob('job_*')
        failed = self.install('__worker', self.internal_path(job), FIXTURE_UPDATE_CODE='13')
        self.assertEqual(failed.returncode, 13, failed.stdout + failed.stderr)
        self.assertEqual((job / 'exit_code').read_text().strip(), '13')
        self.assertTrue((job / 'finished_utc').is_file())
        self.assertIn('fixture update', (job / 'worker.log').read_text())
        self.assertEqual(len(self.recorded('python', 'update')), 1)


if __name__ == '__main__':
    unittest.main()
