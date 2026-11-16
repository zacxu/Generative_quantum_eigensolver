"""Pause using staged code before replacing any file used by the live runner."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
import uuid

# This file is deliberately executable directly from the staged update tree.
# Imports resolve against the untouched project until it has safely paused.
if __name__ == '__main__' and not __package__:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--root', default='.')
    bootstrap_args, _ = bootstrap.parse_known_args()
    sys.path.insert(0, str(Path(bootstrap_args.root).resolve(strict=True)))

from large_qubit.common import read_json, require, sha256_file, workspace_path
from large_qubit.section8 import load, save, utc

STAGE = 'large_qubit/updates/section9_operations_v2'
CONTROL = 'large_qubit/operations/section9_recovery_v1/control'
MATRIX = 'large_qubit/main/section9_recovery_v1/matrix_0001'
REGISTRATION = 'large_qubit/registrations/sections_1_9_operations_v2/manifest.json'
ARCHIVE = 'large_qubit/archives/sections_1_9_operations_v2/original'
GUIDE = 'large_qubit/section9.md'
OLD_GUIDES = ('large_qubit/section9_parallel.md', 'large_qubit/section9_recovery.md',
              'large_qubit/section9_gpu_control.md')
HISTORICAL_MANIFESTS = {
    'large_qubit/registrations/sections_1_9_v1/manifest.json':
        '7ea4123cd837d9db69159be132a5bff94d53923b16e32999957151eb20745091',
    'large_qubit/registrations/sections_1_9_parallel_v1/manifest.json':
        'fa162771755f41bec3a76438c1f23207fb6bdb519968834bdd97e09f5800ff45',
    'large_qubit/registrations/sections_1_9_recovery_v1/manifest.json':
        '9509abe5569c71ea72bfeb7102488ab570b3d9b3bb5ec1217fbf1cfd2133ae72',
    'large_qubit/registrations/sections_1_9_gpu_control_v1/manifest.json':
        '64d0f7589147d4f8f44a6442a712b1711716c2aa5d1dabd987d4820356b23a91',
}
ARCHIVED_SOURCES = tuple('large_qubit/' + name for name in (
    'section9.md', 'section9_parallel.md', 'section9_recovery.md', 'section9_gpu_control.md',
    'section9_registration.py', 'section9_parallel_registration.py',
    'section9_recovery_registration.py', 'section9_control_registration.py',
    'section9_pause.py', 'scripts/run_section9_control_tmux.sh', 'tests/test_section9_pause.py'))
SOURCES = tuple('large_qubit/' + name for name in (
    'section9_operations_registration.py', 'section9_registration.py',
    'section9_parallel_registration.py', 'section9_recovery_registration.py',
    'section9_control_registration.py', 'section9_pause.py',
    'scripts/run_section9_control_tmux.sh', 'tests/test_section9_pause.py',
    'tests/test_section9_operations_registration.py', 'tests/test_section9_operations_launcher.py',
    'section9_update.py', 'tests/test_section9_update.py', 'scripts/install_section9_update_tmux.sh'))


def _stage_path(root, stage_path):
    expected = workspace_path(root, STAGE, namespace=STAGE)
    require(Path(stage_path).resolve(strict=True) == expected,
            'Stage the update only in ' + STAGE)
    return expected


def validate_stage(workspace, stage_path):
    """Authenticate staged changes and original live bytes without importing staged code."""
    root = Path(workspace).resolve(strict=True)
    stage = _stage_path(root, stage_path)
    manifest_path = workspace_path(stage, REGISTRATION, namespace='large_qubit/registrations')
    amendment = sha256_file(manifest_path)
    require(manifest_path.with_suffix('.sha256').read_text().strip() == amendment,
            'Staged Section9 operations amendment checksum changed')
    value = read_json(manifest_path)
    require(value['schema_version'] == 'section9-operations-amendment-v2'
            and value['historical_manifest_sha256'] == HISTORICAL_MANIFESTS
            and set(value['bindings_sha256']) == set(SOURCES)
            and set(value['archived_bindings']) == set(ARCHIVED_SOURCES)
            and value['training_executed'] is False
            and value['scientific_evaluation_executed'] is False,
            'Staged update scope or historical registration anchors differ')
    installed = (root / REGISTRATION).exists()
    if installed:
        require(sha256_file(root / REGISTRATION) == amendment
                and (root / REGISTRATION).with_suffix('.sha256').read_text().strip() == amendment,
                'Another operations amendment is already installed')
    expected_files = dict(value['bindings_sha256'])
    original_bindings, archived, unchanged = {}, {}, {}
    plan_sha = sha256_file(root / 'research_plan_revised.md')
    require(value['research_plan_sha256'] == plan_sha, 'Staged research plan binding changed')
    for name, expected_manifest in HISTORICAL_MANIFESTS.items():
        path = workspace_path(root, name, namespace='large_qubit/registrations')
        require(sha256_file(path) == expected_manifest
                and path.with_suffix('.sha256').read_text().strip() == expected_manifest,
                'Original live registration changed: ' + name)
        old = read_json(path)
        require(old['research_plan_sha256'] == plan_sha, 'Original registered research plan changed')
        for source, expected in old['bindings_sha256'].items():
            original_bindings[source] = expected
            if source in ARCHIVED_SOURCES:
                archive_name = ARCHIVE + '/' + source
                archived[source] = {'path': archive_name, 'sha256': expected, 'registration': name}
                expected_files[archive_name] = expected
                if not installed:
                    require(sha256_file(workspace_path(root, source, namespace='large_qubit')) == expected,
                            'Live bound file changed before safe update: ' + source)
            else:
                unchanged[source] = expected
                require(sha256_file(workspace_path(root, source, namespace='large_qubit')) == expected,
                        'Unchanged scientific source changed: ' + source)
    require(archived == value['archived_bindings']
            and unchanged == value['unchanged_live_bindings_sha256'],
            'Staged archive mapping or unchanged scientific closure differs')
    for name, expected in expected_files.items():
        require(sha256_file(workspace_path(stage, name, namespace='large_qubit')) == expected,
                'Staged source or archive changed: ' + name)
        if installed:
            require(sha256_file(workspace_path(root, name, namespace='large_qubit')) == expected,
                    'Installed source or archive changed: ' + name)
    expected_files[REGISTRATION] = amendment
    expected_files[str(Path(REGISTRATION).with_suffix('.sha256')).replace('\\', '/')] = sha256_file(
        manifest_path.with_suffix('.sha256'))
    expected_files[GUIDE] = sha256_file(workspace_path(stage, GUIDE, namespace='large_qubit'))
    return {'all_passed': True, 'installed': installed, 'amendment_sha256': amendment,
            'files': expected_files, 'original_bindings': original_bindings}


def _load_staged_pause(stage):
    name = 'large_qubit._section9_staged_pause'
    spec = importlib.util.spec_from_file_location(name, stage / 'large_qubit/section9_pause.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _atomic_bytes(path, value, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.section9-update-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fresh_verification(root):
    result = subprocess.run([sys.executable, '-B', '-m', 'large_qubit.section9_control', 'verify'],
                            cwd=root, capture_output=True, text=True)
    require(result.returncode == 0, 'Installed verification failed: ' + result.stdout + result.stderr)
    value = json.loads(result.stdout)
    require(value['all_passed'] is True, 'Installed verification did not authenticate the project')
    return value


def install_files(root, stage, checked, receipt):
    """Replace only fixed operational files, restoring exact backups on any failure."""
    files = checked['files']
    allowed = set(SOURCES) | {ARCHIVE + '/' + name for name in ARCHIVED_SOURCES} | {
        REGISTRATION, str(Path(REGISTRATION).with_suffix('.sha256')).replace('\\', '/'), GUIDE}
    require(set(files) == allowed, 'Update file inventory exceeds the fixed operational allowlist')
    backups = {}
    for name in sorted(allowed | set(OLD_GUIDES)):
        path = workspace_path(root, name, namespace='large_qubit')
        if path.exists():
            require(path.is_file(), 'Update destination is not a file: ' + name)
            backup = receipt / 'backups' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            with backup.open('xb') as handle:
                handle.write(path.read_bytes())
            expected = sha256_file(path)
            require(sha256_file(backup) == expected, 'Update backup did not preserve exact bytes: ' + name)
            backups[name] = {'path': backup.relative_to(root).as_posix(), 'sha256': expected,
                             'mode': stat.S_IMODE(path.stat().st_mode)}
        else:
            backups[name] = None
    save(receipt / 'backups.json', {'utc': utc(), 'files': backups})
    changed = []
    try:
        for name in sorted(files):
            source = workspace_path(stage, name, namespace='large_qubit')
            require(sha256_file(source) == files[name], 'Staged update changed during installation: ' + name)
            destination = workspace_path(root, name, namespace='large_qubit')
            changed.append(name)  # Include the currently attempted replacement in rollback.
            mode = backups[name]['mode'] if backups[name] is not None else 0o644
            _atomic_bytes(destination, source.read_bytes(), mode)
            require(sha256_file(destination) == files[name], 'Installed bytes differ: ' + name)
        for name in OLD_GUIDES:
            path = workspace_path(root, name, namespace='large_qubit')
            if path.exists():
                expected = checked['original_bindings'][name]
                require(sha256_file(path) == expected
                        and sha256_file(root / ARCHIVE / name) == expected,
                        'Historical guide cannot be removed until its exact archive is verified: ' + name)
                changed.append(name)
                path.unlink()
        print(json.dumps({'update_stage': 'verifying_installed_project', 'matrix_idle': True}), flush=True)
        verification = _fresh_verification(root)
        save(receipt / 'verification.json', verification)
        return verification
    except BaseException as error:
        failures = []
        for name in reversed(changed):
            path = workspace_path(root, name, namespace='large_qubit')
            old = backups[name]
            try:
                if old is None:
                    if path.exists():
                        path.unlink()
                else:
                    backup = workspace_path(root, old['path'], namespace=CONTROL)
                    require(sha256_file(backup) == old['sha256'], 'Rollback backup changed: ' + name)
                    _atomic_bytes(path, backup.read_bytes(), old['mode'])
                    require(sha256_file(path) == old['sha256'], 'Rollback bytes differ: ' + name)
            except BaseException as rollback_error:
                failures.append({'path': name, 'error': str(rollback_error)})
        save(receipt / 'failed.json', {'utc': utc(), 'error': str(error), 'paused': True,
             'rollback_complete': not failures, 'rollback_errors': failures})
        if failures:
            raise ValueError('Update failed and rollback needs inspection; run remains paused: ' + str(failures)) from error
        raise


def update(workspace, stage_path, *, execute=False):
    require(execute is True, 'Installing the staged pause update requires --execute')
    require(platform.system() == 'Linux', 'Install the staged pause update on the Linux server')
    root = Path(workspace).resolve(strict=True)
    stage = _stage_path(root, stage_path)
    from large_qubit.training import _run_lock
    from large_qubit import section9_control as control
    with _run_lock(root / CONTROL / 'update_command'):
        print(json.dumps({'update_stage': 'authenticating_untouched_project', 'paused': False}), flush=True)
        checked = validate_stage(root, stage)
        control.verify_binding(root)
        pause = _load_staged_pause(stage)
        with pause._pause_interruptions(), _run_lock(root / CONTROL / 'pause_command'):
            previous = pause._already_paused(root)
            completed = False
            if previous is None and (root / MATRIX / 'completed.json').exists():
                completed = load(root / MATRIX / 'completed.json').get('all_trajectories_successful') is True
                require(completed, 'The matrix is sealed unsuccessfully; inspect results before updating')
                previous = {'paused': False, 'matrix_completed': True}
            if previous is None:
                session, coordinator, attempt = pause.discover(root)
                own = pause._process(os.getpid())
                pause_receipt = root / CONTROL / 'pauses' / ('pause_' + uuid.uuid4().hex)
                save(pause_receipt / 'started.json', {
                    'utc': utc(), 'session': session, 'coordinator_pid': coordinator.pid,
                    'coordinator_start_ticks': coordinator.start_ticks,
                    'coordinator_argv': list(coordinator.argv), 'timeout_seconds': 0,
                    'matrix_path': MATRIX, 'execution_attempt': attempt.relative_to(root).as_posix(),
                    'pause_pid': own.pid, 'pause_start_ticks': own.start_ticks,
                    'operations_amendment_sha256': checked['amendment_sha256'],
                    'bootstrap_staged_update': True})
                print(json.dumps({'update_stage': 'waiting_for_safe_pause', 'paused': False,
                                  'receipt_path': pause_receipt.relative_to(root).as_posix()}), flush=True)
                pause_function = (pause._cooperative_pause if coordinator.module == 'large_qubit.section9_control'
                                  else pause._legacy_pause)
                previous = pause_function(root, coordinator, attempt, pause_receipt, 0)
                completed = previous.get('matrix_completed') is True
            require(previous.get('paused') is True or completed,
                    'Update waits for a confirmed clean pause before changing live files')
            if completed:
                last_notice = float('-inf')
                while any(pause._is_work_process(process, root) for process in pause._processes().values()):
                    if time.monotonic() - last_notice >= 10:
                        print(json.dumps({'update_stage': 'waiting_for_completed_coordinator_exit'}), flush=True)
                        last_notice = time.monotonic()
                    time.sleep(.2)
            with control._leases(root):
                require(not any(pause._is_work_process(process, root) for process in pause._processes().values()),
                        'Project work processes are still active; live sources were not changed')
                if (root / MATRIX / 'completed.json').exists():
                    authenticated = control.verify_matrix(root)
                    require(authenticated['all_passed'] is True
                            and authenticated['all_trajectories_successful'] is True,
                            'Completed matrix did not authenticate successfully; live sources were not changed')
                    completed = True
                checked = validate_stage(root, stage)
                receipt = root / CONTROL / 'updates' / ('update_' + uuid.uuid4().hex)
                save(receipt / 'started.json', {'utc': utc(), 'amendment_sha256': checked['amendment_sha256'],
                     'stage_path': STAGE, 'pause': previous, 'scientific_work_started': False})
                if not checked['installed']:
                    install_files(root, stage, checked, receipt)
                result = {'installed': True, 'already_installed': checked['installed'],
                          'paused': previous.get('paused') is True and not completed, 'matrix_completed': completed,
                          'operations_amendment_sha256': checked['amendment_sha256'],
                          'receipt_path': receipt.relative_to(root).as_posix(), 'scientific_work_started': False}
                save(receipt / 'completed.json', dict(result, utc=utc()))
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--stage-path', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(update(args.root, args.stage_path, execute=args.execute), indent=2))
        return 0
    except KeyboardInterrupt:
        print('Update interrupted; temporary stops were released. Inspect its receipts before resuming.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
