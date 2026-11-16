"""Read one live Section 6 snapshot without changing or starting any work.

Only counts, costs and fitting improvements relative to the fresh HF baseline
are printed.  Raw energies, circuits, row labels, logs and error messages never
leave this helper.  A trailing fragment in any immutable journal segment is
preserved and ignored; malformed committed lines or hash chains are errors.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from large_qubit.common import digest, read_json, require, workspace_path
from large_qubit.protocol import BENCHMARK_ID


ALIASES = ('CO2_T01', 'CO2_T02', 'CO2_T03')
ROLES = ('HF', 'diagnostic', 'fitting')
QUOTAS = {'CO2_T01': 2731, 'CO2_T02': 2731, 'CO2_T03': 2730}
GENESIS = '0' * 64
SHA = re.compile('[0-9a-f]{64}')
SEGMENT = re.compile(r'journal_([0-9]{4})\.jsonl')
ATTEMPT = re.compile(r'attempt_([0-9]{4})')


def _finite(value, name, minimum=None):
    require(type(value) in (int, float), f'Invalid {name}')
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'Invalid {name}') from error
    require(math.isfinite(value) and (minimum is None or value >= minimum), f'Invalid {name}')
    return value


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'Invalid {name}')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate committed journal field')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('Nonfinite committed journal data')


def _records(root, private_relative):
    private = workspace_path(root, private_relative, namespace='large_qubit/scoring_only')
    records, fragments, previous = [], [], GENESIS
    paths = sorted(private.glob('journal_*.jsonl')) if private.exists() else []
    for number, path in enumerate(paths, 1):
        require(SEGMENT.fullmatch(path.name) and path.name == f'journal_{number:04d}.jsonl',
                'Journal segments must be contiguous and canonical')
        path = workspace_path(root, path.relative_to(root).as_posix(), namespace=private_relative)
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b'\n'):
                require(index == len(lines) - 1, 'Uncommitted fragment is not at a segment end')
                fragments.append({'segment': path.name, 'bytes': len(line)})
                continue
            try:
                value = json.loads(line.decode('ascii'), object_pairs_hook=_pairs, parse_constant=_nonfinite)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError('Malformed committed journal record') from error
            require(type(value) is dict and set(value) == {'record_index', 'previous_sha256', 'payload', 'sha256'},
                    'Journal envelope fields differ')
            require(type(value['record_index']) is int and value['record_index'] == len(records) + 1
                    and value['previous_sha256'] == previous and type(value['payload']) is dict,
                    'Journal index or previous hash differs')
            body = {key: value[key] for key in ('record_index', 'previous_sha256', 'payload')}
            require(type(value['sha256']) is str and SHA.fullmatch(value['sha256'])
                    and value['sha256'] == digest(body), 'Journal checksum differs')
            records.append(value['payload'])
            previous = value['sha256']
    return records, fragments, previous, len(paths)


def _timestamp(value):
    require(type(value) is str, 'Timestamp must be text')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
            'Timestamp must include a UTC offset')
    return parsed


def _worker_status(root, run_relative):
    base = workspace_path(root, run_relative, namespace='large_qubit/evidence/section6_runs')
    attempts = []
    if base.exists():
        for child in sorted(base.iterdir()):
            require(ATTEMPT.fullmatch(child.name) and child.is_dir(), 'Unexpected worker-attempt evidence entry')
            attempts.append(workspace_path(root, child.relative_to(root).as_posix(), namespace=run_relative))
    if not attempts:
        return {'state': 'not_started', 'attempt_count': 0, 'latest_attempt': None,
                'exit_code': None, 'previous_failed_attempts': 0,
                'elapsed_since_latest_worker_start_seconds': None}
    latest = attempts[-1]
    for number, attempt in enumerate(attempts, 1):
        require(attempt.name == f'attempt_{number:04d}', 'Worker attempts must retain a contiguous history')
    def marker(name):
        return workspace_path(root, f'{latest.relative_to(root).as_posix()}/{name}', namespace=run_relative)
    exit_code = None
    if marker('worker.exit_code').exists():
        text = marker('worker.exit_code').read_text(encoding='ascii').strip()
        require(re.fullmatch('-?[0-9]+', text), 'Invalid worker exit code')
        exit_code = int(text)
    failed = marker('worker.failed').exists() or marker('launch.failed').exists()
    success = marker('worker.success').exists()
    if failed or (exit_code is not None and exit_code != 0):
        state = 'failed'
    elif success and exit_code == 0:
        state = 'finished_success'
    elif exit_code is not None:
        state = 'exit_recorded_pending_complete_markers'
    elif marker('worker.started_utc').exists():
        state = 'started_no_exit_marker'
    else:
        state = 'launch_recorded_without_worker_start'
    elapsed = None
    if marker('worker.started_utc').exists():
        elapsed = max(0., (datetime.now(timezone.utc) - _timestamp(
            marker('worker.started_utc').read_text(encoding='ascii').strip())).total_seconds())
    previous_failed = sum((attempt / 'worker.failed').exists() or (attempt / 'launch.failed').exists()
                          for attempt in attempts[:-1])
    return {'state': state, 'attempt_count': len(attempts), 'latest_attempt': latest.name,
            'exit_code': exit_code, 'success_marker': success, 'failure_marker': failed,
            'previous_failed_attempts': previous_failed,
            'elapsed_since_latest_worker_start_seconds': elapsed}


def progress(workspace, *, section6_root='large_qubit/validation/section6_v1', run_directory=None):
    root = Path(workspace).resolve(strict=True)
    public = workspace_path(root, section6_root, namespace='large_qubit/validation')
    require(public != root / 'large_qubit/validation', 'Use a versioned acquisition output')
    version = section6_root.removeprefix('large_qubit/validation/')
    require(re.fullmatch('section6_[A-Za-z0-9][A-Za-z0-9_-]*', version), 'Use a simple Section6 version')
    policy_path = workspace_path(root, 'large_qubit/section6_policy.json', namespace='large_qubit')
    policy = read_json(policy_path)
    acquisition = policy['acquisition']
    require(acquisition['training_aliases'] == list(ALIASES)
            and acquisition['geometry_request_quotas'] == QUOTAS
            and type(acquisition['maximum_logical_requests']) is int
            and acquisition['maximum_logical_requests'] == 8192,
            'Unexpected acquisition partition or request budget')
    catalogue_sha = acquisition.get('catalogue_sha256')
    require(type(catalogue_sha) is str and SHA.fullmatch(catalogue_sha), 'Acquisition catalogue identity differs')
    length = _integer(acquisition['length'], 'complete circuit length', 1)
    require(length in (32, 64), 'Unsupported acquisition circuit length')
    fitting_goal = _integer(acquisition['fitting_goal_per_geometry'], 'fitting goal', 1)
    diagnostic_goal = _integer(acquisition['diagnostics_per_geometry'], 'diagnostic goal', 1)
    require(fitting_goal == 2048 and diagnostic_goal == 128, 'Acquisition retention goals differ')
    private_relative = f'large_qubit/scoring_only/{BENCHMARK_ID}/section6/{version}'
    records, fragments, last_hash, segment_count = _records(root, private_relative)
    groups = {alias: {'issued': 0, 'completed': 0, 'failed': 0, 'abandoned': 0,
        'by_role': {role: {'issued': 0, 'completed': 0} for role in ROLES},
        'retained_fitting': 0, 'retained_diagnostics': 0} for alias in ALIASES}
    hf, best, pending, row_ids, circuits = {}, {}, {}, set(), {alias: set() for alias in ALIASES}
    issued = completed = failed = abandoned = 0
    wall, peak = 0., 0
    first_session, last_stop, preflight = None, None, None
    integrity = {'issued': 0, 'completed': 0, 'abandoned': 0}
    for item in records:
        event = item.get('event')
        if event == 'session_started':
            timestamp = _timestamp(item['utc'])
            if first_session is None:
                first_session = timestamp
            continue
        if event == 'session_finished':
            last_stop = item.get('stop_reason') if item.get('stop_reason') in ('target', 'schedule', 'budget', 'failure') else 'other_recorded_status'
            continue
        if event == 'session_interrupted':
            continue
        if event == 'preflight':
            preflight = {'measured_seconds_per_request': _finite(item['seconds_per_request'], 'preflight request seconds', 0.)}
            continue
        if event in ('integrity_started', 'integrity_completed', 'integrity_abandoned',
                     'integrity_request_started', 'integrity_request_completed', 'integrity_request_abandoned'):
            stage = event.rsplit('_', 1)[1]
            integrity[{'started': 'issued', 'completed': 'completed', 'abandoned': 'abandoned'}[stage]] += 1
            require(integrity['completed'] + integrity['abandoned'] <= integrity['issued'], 'Integrity overhead lacks an issued action')
            continue
        ordinal = _integer(item.get('request_ordinal'), 'global request ordinal', 1)
        if event == 'request_started':
            require(ordinal == issued + 1 <= 8192, 'Acquisition request order or cap differs')
            alias, role = item.get('geometry_id'), item.get('role')
            require(alias in ALIASES and role in ROLES, 'Acquisition request is outside training scope')
            require(groups[alias]['issued'] < QUOTAS[alias], 'Geometry request quota exceeded')
            require(role == 'HF' or alias in hf, 'HF must complete before non-HF requests')
            key = item.get('query_sha256')
            require(type(key) is str and SHA.fullmatch(key)
                    and item.get('catalogue_sha256') == catalogue_sha, 'Request identity is missing')
            issued += 1
            groups[alias]['issued'] += 1; groups[alias]['by_role'][role]['issued'] += 1
            pending[ordinal] = (alias, role, key)
            continue
        require(event in ('request_completed', 'request_failed', 'request_abandoned')
                and ordinal in pending, 'Response or failure lacks an issued intent')
        alias, role, key = pending.pop(ordinal)
        require(item.get('geometry_id', alias) == alias and item.get('role', role) == role,
                'Response aliases or roles differ from its intent')
        if event in ('request_failed', 'request_abandoned'):
            label = 'failed' if event == 'request_failed' else 'abandoned'
            groups[alias][label] += 1
            if label == 'failed':
                failed += 1
            else:
                abandoned += 1
            continue
        require(item.get('query_sha256') == key, 'Response identity differs from its intent')
        energy = _finite(item.get('energy_hartree'), 'completed query value')
        wall += _finite(item.get('wall_seconds'), 'completed query seconds', 0.)
        memory = _integer(item.get('peak_rss_bytes'), 'query peak RSS', 1)
        peak = max(peak, memory)
        completed += 1; groups[alias]['completed'] += 1; groups[alias]['by_role'][role]['completed'] += 1
        row = item.get('eligible_row')
        if role == 'HF':
            require(row is None, 'HF is an auxiliary request, not a corpus row')
            require(alias not in hf or abs(hf[alias] - energy) <= 1e-8, 'Repeated HF responses disagree')
            hf[alias] = energy
        elif row is not None:
            require(type(row) is dict and set(row) == {'row_id', 'geometry_id', 'role', 'ops',
                'energy_hartree', 'catalogue_sha256', 'request_ordinal'}, 'Eligible row fields differ')
            require(row['geometry_id'] == alias and row['role'] == role
                    and type(row['request_ordinal']) is int and row['request_ordinal'] == ordinal,
                    'Eligible row identity differs')
            require(row['catalogue_sha256'] == catalogue_sha, 'Retained row catalogue differs')
            require(type(row['row_id']) is str and row['row_id'] and row['row_id'] not in row_ids,
                    'Retained row identity is duplicated')
            require(type(row['ops']) is list and len(row['ops']) == length
                    and all(type(operator) is int and 0 <= operator <= 512 for operator in row['ops']),
                    'Retained circuit is incomplete or invalid')
            require(abs(_finite(row['energy_hartree'], 'retained row value') - energy) <= 1e-12,
                    'Retained row value differs from its charged response')
            circuit = tuple(row['ops'])
            require(circuit not in circuits[alias], 'Retained circuit duplicates fitting or diagnostics')
            row_ids.add(row['row_id']); circuits[alias].add(circuit)
            if role == 'fitting':
                groups[alias]['retained_fitting'] += 1
                best[alias] = energy if alias not in best else min(best[alias], energy)
            else:
                groups[alias]['retained_diagnostics'] += 1
                require(groups[alias]['retained_diagnostics'] <= diagnostic_goal, 'Diagnostic retention goal exceeded')
    require(completed + failed + abandoned + len(pending) == issued, 'Acquisition accounting frontier differs')
    elapsed = None if first_session is None else max(0., (datetime.now(timezone.utc) - first_session).total_seconds())
    max_wall = _finite(policy['max_wall_seconds'], 'campaign wall allowance', 0.)
    for alias, group in groups.items():
        group.update({'geometry_id': alias, 'request_cap': QUOTAS[alias],
            'remaining_request_allowance': QUOTAS[alias] - group['issued'],
            'fitting_goal': fitting_goal, 'diagnostic_goal': diagnostic_goal,
            'HF_baseline_available': alias in hf,
            'best_retained_fitting_HF_improvement_ha': None if alias not in best or alias not in hf else
                _finite(hf[alias] - best[alias], 'HF-relative fitting improvement')})
    return {'schema': 'section6-live-progress-v1', 'section6_output': section6_root,
        'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'snapshot_valid': True,
        'worker': _worker_status(root, run_directory or f'large_qubit/evidence/section6_runs/{version}'),
        'requests': {'cap': 8192, 'issued': issued, 'completed': completed, 'failed': failed,
                     'abandoned': abandoned, 'pending': len(pending), 'remaining_after_issued': 8192 - issued},
        'retained': {'fitting': sum(group['retained_fitting'] for group in groups.values()),
                     'fitting_goal': fitting_goal * 3,
                     'diagnostic': sum(group['retained_diagnostics'] for group in groups.values()),
                     'diagnostic_goal': diagnostic_goal * 3},
        'completed_query_wall_seconds': wall, 'peak_query_rss_bytes': peak,
        'measured_completed_query_seconds_mean': None if completed == 0 else wall / completed,
        'elapsed_campaign_seconds_including_downtime': elapsed,
        'remaining_campaign_wall_seconds': None if elapsed is None else max(0., max_wall - elapsed),
        'preflight': preflight, 'last_session_stop_reason': last_stop,
        'journal': {'segments': segment_count, 'complete_records': len(records),
                    'local_hash_chain_valid': True, 'last_sha256': last_hash,
                    'uncommitted_fragments_ignored': fragments},
        'separate_integrity_overhead': integrity, 'geometries': groups,
        'public_completion_manifest_present': (public / 'manifest.json').is_file(),
        'public_failure_report_present': (public / 'failure.json').is_file(),
        'snapshot_is_release_verification': False, 'model_training_executed_by_helper': False,
        'diagnostic_values_used_for_selection_or_improvement': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--section6-root', default='large_qubit/validation/section6_v1')
    parser.add_argument('--run-directory')
    args = parser.parse_args(argv)
    try:
        report = progress(args.root, section6_root=args.section6_root, run_directory=args.run_directory)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        # Exception messages can contain private circuit labels; never echo them.
        print(json.dumps({'snapshot_valid': False, 'error_type': type(error).__name__}))
        return 1
    print(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
