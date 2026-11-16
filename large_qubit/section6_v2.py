"""Approved v2 confirmation: fresh corpus, unchanged gates, cumulative costs.

Every physical request has a durable intent before scoring. Segmented hash-
chained journals preserve crashes, partial tails, failed requests and resumes.
Model fitting, held-out scoring and correlated references are never invoked.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid

from .common import _pairs, canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .protocol import BENCHMARK_ID


DEFAULT_SECTION6 = 'large_qubit/validation/section6_v2'
POLICY_PATH = 'large_qubit/section6_v2_policy.json'
PREPARATION_PATH = 'large_qubit/evidence/sections_1_6_v2/preparation.json'
PRIVATE_ROOT = f'large_qubit/scoring_only/{BENCHMARK_ID}/section6/section6_v2'
RELEASE_ROOT = 'large_qubit/releases/co2_corpus_v2'
SOURCES = tuple('large_qubit/' + name + '.py' for name in
                ('acquisition_v2', 'corpus_v2', 'section6_v2', 'acquisition', 'corpus', 'section6'))
AMENDMENT_PATH = 'large_qubit/amendments/section6_acquisition_v2.json'
AMENDMENT_SHA256 = '611592dad5d900a536939ce4435f5a1982174d2e1a94df1f947c6a95357cc130'
PRIOR_GATE_PATH = 'large_qubit/validation/section6_v1/manifest.json'
PRIOR_GATE_SHA256 = 'cdf23f3493d1c583cbf4e46126412416ea0afe295247f699f1a4305845693737'
PRIOR_REQUESTS = 6538
MAXIMUM_CUMULATIVE_REQUESTS = 14730
TRAINING = ('CO2_T01', 'CO2_T02', 'CO2_T03')
GENESIS = '0' * 64


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def _sources(root):
    return {name: sha256_file(root / name) for name in SOURCES}


def _approved_amendment(root):
    """Authenticate the single user-approved prospective budget/design amendment."""
    path = workspace_path(root, AMENDMENT_PATH, namespace='large_qubit/amendments')
    require(sha256_file(path) == AMENDMENT_SHA256, 'Approved confirmation amendment changed')
    amendment = read_json(path)
    require(amendment['approved'] is True
            and amendment['authorization']['verbatim_user_answer'] == 'Approve one fresh v2 CPU campaign'
            and amendment['stopping_and_scope']['authorized_fresh_campaigns'] == 1
            and amendment['stopping_and_scope']['automatic_repeated_campaigns'] is False
            and amendment['prospective_campaign']['fresh_ordinary_request_ceiling'] == 8192
            and amendment['prospective_campaign']['maximum_cumulative_ordinary_requests'] == MAXIMUM_CUMULATIVE_REQUESTS,
            'The confirmation exceeds the recorded user authorization')
    return amendment


def _finite(value, *, minimum=0.):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            'Expected a finite nonnegative scalar')
    return value


def _norm_check(norm):
    require(type(norm) is dict and abs(_finite(norm['state_norm']) - 1) <= 1e-10
            and _finite(norm['norm_error']) <= 1e-10
            and all(type(norm[k]) is int for k in ('n_alpha', 'n_beta', 'sector_dimension'))
            and (norm['n_alpha'], norm['n_beta'], norm['sector_dimension']) == (11, 11, 1863225)
            and _finite(norm['sector_leakage_probability']) == 0,
            'Population or normalization differs')


def _commit_bytes(path, data):
    """No-clobber atomic commit; retain every staged candidate for recovery."""
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, 'Existing committed artifact differs; preserve it')
        return
    staging = path.parent / 'commit_attempts'
    staging.mkdir(parents=True, exist_ok=True)
    candidate = staging / f'{path.name}.{uuid.uuid4().hex}.pending'
    with candidate.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    try:
        os.link(candidate, path)
    except FileExistsError:
        require(path.read_bytes() == data, 'Concurrent committed artifact differs')


def build_section6_policy(catalogue):
    from .acquisition_v2 import acquisition_policy
    return {'schema_version': 'section6-policy-v2', 'acquisition': acquisition_policy(catalogue, length=32),
        'amendment': {'path': AMENDMENT_PATH, 'file_sha256': AMENDMENT_SHA256},
        'prior_failed_attempt_gate': {'path': PRIOR_GATE_PATH, 'sha256': PRIOR_GATE_SHA256},
        'prior_ordinary_requests': PRIOR_REQUESTS,
        'maximum_cumulative_ordinary_requests': MAXIMUM_CUMULATIVE_REQUESTS,
        'fresh_labels_only': True, 'authorized_confirmation_campaigns': 1,
        'cpu_threads': 4, 'worker_concurrency': 1, 'max_wall_seconds': 24 * 3600.,
        'memory_limit_bytes': 3 * 1024 ** 3, 'normalization_tolerance': 1e-10,
        'energy_tolerance_ha': 1e-8, 'preflight_completed_requests': 18,
        'schedule_projection_retry_factor': 1.25, 'schedule_clock': 'first session start, including downtime',
        'request_logging': 'fsynced intent then fsynced response; every intent charged including failure or crash',
        'resume': 'new immutable journal segment; pending intents abandoned and charged; no silent energy replay',
        'row_retention': 'all eligible novel fitting rows, including poor and rejected local proposals',
        'HF_duplicate': 'auxiliary HF is charged separately and is not a fitting row',
        'diagnostic_reservations_excluded_even_before_labels': True,
        'final_label_integrity': 'first,middle,last selected fitting row plus first diagnostic at each geometry',
        'separate_label_integrity_hamiltonian_actions': 12,
        'integrity_checks_admitted_to_search': False,
        'heldout_energy_or_reference_access': False, 'model_training_executed': False,
        'execution_ready': False}


def prepare_section6(workspace):
    from .section5 import verify_section5, verify_preparation
    from .acquisition_v2 import AcquisitionSampler
    from .corpus_v2 import prior_attempt_cost
    root = Path(workspace).resolve(strict=True)
    amendment = _approved_amendment(root)
    require(verify_section5(root)['section_5_complete'], 'Section5 GO is required before acquisition')
    _, catalogue, screening, _ = verify_preparation(root)
    policy = build_section6_policy(catalogue)
    require(canonical_bytes(policy['acquisition']['quality_policy']) == canonical_bytes(amendment['unchanged_quality_policy'])
            and policy['acquisition']['fitting_component_cycle'] == amendment['prospective_campaign']['fitting_component_cycle'],
            'Confirmation policy differs from the approved mixture or unchanged quality criteria')
    prior = prior_attempt_cost(root)
    private = workspace_path(root, PRIVATE_ROOT, namespace='large_qubit/scoring_only')
    reservations_path = private / 'reservations.json'
    for path in (root / POLICY_PATH, root / PREPARATION_PATH, private):
        require(not path.exists(), 'Preserve existing Section6 preparation; verify it instead')
    reservations = {alias: AcquisitionSampler(alias, catalogue, screening).reserve_diagnostics() for alias in TRAINING}
    _write(root / POLICY_PATH, policy)
    _write(reservations_path, reservations)
    prep = {'schema_version': 'section6-preparation-v2', 'created_utc': _utc(), 'all_passed': True,
        'policy_sha256': digest(policy), 'policy_file_sha256': sha256_file(root / POLICY_PATH),
        'source_bindings': _sources(root), 'reservations_path': reservations_path.relative_to(root).as_posix(),
        'approved_amendment': policy['amendment'], 'prior_acquisition_cost': prior,
        'reservations_sha256': sha256_file(reservations_path),
        'catalogue_sha256': catalogue['sha256'],
        'section5_manifest_sha256': sha256_file(root / 'large_qubit/validation/section5_v1/manifest.json'),
        'section3_manifest_sha256': sha256_file(root / 'large_qubit/physics/section3_v3/manifest.json'),
        'legacy_weight_mathematics_sha256': sha256_file(root / 'phase6/objective.py'),
        'candidate_energy_requests': 0, 'model_training_executed': False,
        'heldout_energy_or_reference_access': False}
    _write(root / PREPARATION_PATH, prep)
    return prep


def verify_preparation(workspace):
    from .section5 import verify_preparation as verify_previous
    from .acquisition_v2 import AcquisitionSampler
    from .section4 import verify_section4
    from .corpus_v2 import prior_attempt_cost
    root = Path(workspace).resolve(strict=True)
    amendment = _approved_amendment(root)
    verify_section4(root)
    _, catalogue, screening, _ = verify_previous(root)
    policy, prep = read_json(root / POLICY_PATH), read_json(root / PREPARATION_PATH)
    require(canonical_bytes(policy) == canonical_bytes(build_section6_policy(catalogue)), 'Section6 policy differs')
    require(canonical_bytes(policy['acquisition']['quality_policy']) == canonical_bytes(amendment['unchanged_quality_policy'])
            and policy['acquisition']['fitting_component_cycle'] == amendment['prospective_campaign']['fitting_component_cycle']
            and prep['approved_amendment'] == policy['amendment']
            and canonical_bytes(prep['prior_acquisition_cost']) == canonical_bytes(prior_attempt_cost(root)),
            'Approved amendment or retained prior-attempt cost differs')
    require(prep['source_bindings'] == _sources(root) and prep['all_passed'] is True
            and prep['policy_sha256'] == digest(policy)
            and prep['policy_file_sha256'] == sha256_file(root / POLICY_PATH)
            and prep['candidate_energy_requests'] == 0
            and prep['model_training_executed'] is False
            and prep['heldout_energy_or_reference_access'] is False,
            'Section6 pre-acquisition source, policy or scope differs')
    require(prep['section5_manifest_sha256'] == sha256_file(root / 'large_qubit/validation/section5_v1/manifest.json')
            and prep['section3_manifest_sha256'] == sha256_file(root / 'large_qubit/physics/section3_v3/manifest.json')
            and prep['legacy_weight_mathematics_sha256'] == sha256_file(root / 'phase6/objective.py'),
            'Prerequisite or weighting source changed')
    path = workspace_path(root, prep['reservations_path'], namespace='large_qubit/scoring_only')
    require(sha256_file(path) == prep['reservations_sha256'], 'Reservation bytes changed')
    reservations = read_json(path)
    rebuilt = {alias: AcquisitionSampler(alias, catalogue, screening).reserve_diagnostics() for alias in TRAINING}
    require(canonical_bytes(reservations) == canonical_bytes(rebuilt)
            and prep['catalogue_sha256'] == catalogue['sha256'], 'Diagnostic reservation is not the frozen seeded rule')
    return policy, catalogue, screening, reservations, prep


def read_journal(folder):
    """Return authenticated complete records, preserving any uncommitted tails."""
    records, tails, previous = [], [], GENESIS
    paths = sorted(Path(folder).glob('journal_*.jsonl'))
    for number, path in enumerate(paths, 1):
        require(path.name == f'journal_{number:04d}.jsonl', 'Journal segment sequence has a gap')
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b'\n'):
                require(index == len(lines) - 1, 'Partial journal record is not a trailing fragment')
                tails.append({'segment': path.name, 'uncommitted_tail_sha256': digest(list(line)), 'bytes': len(line)})
                continue
            def reject(_):
                raise ValueError('Nonfinite journal JSON')
            value = json.loads(line, object_pairs_hook=_pairs, parse_constant=reject)
            require(type(value) is dict and set(value) == {'record_index', 'previous_sha256', 'payload', 'sha256'},
                    'Malformed journal envelope')
            payload = {key: value[key] for key in ('record_index', 'previous_sha256', 'payload')}
            require(type(value['record_index']) is int and value['record_index'] == len(records) + 1
                    and value['previous_sha256'] == previous
                    and value['sha256'] == digest(payload), 'Journal index or hash chain differs')
            records.append(value)
            previous = value['sha256']
    return records, tails


class Journal:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        records, self.tails = read_journal(folder)
        self.index = len(records)
        self.previous = records[-1]['sha256'] if records else GENESIS
        number = len(list(self.folder.glob('journal_*.jsonl'))) + 1
        self.path = self.folder / f'journal_{number:04d}.jsonl'
        self.handle = self.path.open('xb')
        self.records = records

    def append(self, payload):
        envelope = {'record_index': self.index + 1, 'previous_sha256': self.previous, 'payload': payload}
        value = {**envelope, 'sha256': digest(envelope)}
        self.handle.write(canonical_bytes(value) + b'\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.index += 1
        self.previous = value['sha256']
        self.records.append(value)
        return value

    def close(self):
        self.handle.close()


def _fresh_state():
    return {'issued': 0, 'completed': 0, 'failed': 0, 'abandoned': 0, 'pending': {},
        'counts': {a: 0 for a in TRAINING}, 'fit_requests': {a: 0 for a in TRAINING},
        'hf': {}, 'diagnostics': {a: {} for a in TRAINING}, 'fitting': {a: [] for a in TRAINING},
        'eligible': [], 'seen': {}, 'successful_energies': {}, 'retained': {a: set() for a in TRAINING},
        'sessions': {}, 'session_start': None, 'peak_rss_bytes': 0, 'completed_wall_seconds': 0.,
        'preflight': None, 'integrity_issued': 0, 'integrity_pending': {},
        'integrity_checks': {}, 'integrity_wall_seconds': 0., 'last_stop_reason': None, 'last_stop_utc': None}


def next_request(state, samplers, reservations, policy):
    """Choose the deterministic frontier, with balanced acquisition on every pass."""
    quotas = policy['acquisition']['geometry_request_quotas']
    for alias in TRAINING:
        if alias not in state['hf'] and state['counts'][alias] < quotas[alias]:
            return alias, 'HF', {'ops': [0] * 32, 'recipe': {'HF': True}}
    if len(state['hf']) != 3:
        return None
    for index in range(1, 129):
        for alias in TRAINING:
            if index not in state['diagnostics'][alias] and state['counts'][alias] < quotas[alias]:
                return alias, 'diagnostic', reservations[alias][index - 1]
    if any(len(state['diagnostics'][a]) != 128 for a in TRAINING):
        return None
    available = [a for a in TRAINING if len(state['fitting'][a]) < 2048 and state['counts'][a] < quotas[a]]
    if not available:
        return None
    alias = min(available, key=lambda a: (len(state['fitting'][a]), state['fit_requests'][a], TRAINING.index(a)))
    plan = samplers[alias].propose(state['fit_requests'][alias] + 1, state['fitting'][alias])
    return alias, 'fitting', plan


def _eligible_row(state, started, energy, reserved):
    alias, role, ops = started['geometry_id'], started['role'], started['proposal']['ops']
    circuit = tuple(ops)
    if role == 'HF':
        return None
    if role == 'fitting' and (circuit in reserved[alias] or circuit in state['retained'][alias]
                             or not any(ops)):
        return None
    index = started['proposal']['diagnostic_ordinal'] if role == 'diagnostic' else len(state['fitting'][alias]) + 1
    return {'row_id': f'{alias}_v2_{"D" if role == "diagnostic" else "F"}{index:06d}',
        'geometry_id': alias, 'role': role, 'ops': ops, 'energy_hartree': float(energy),
        'catalogue_sha256': started['catalogue_sha256'], 'request_ordinal': started['request_ordinal']}


def replay_records(records, policy, catalogue, screening, reservations, *, state=None,
                   samplers=None, expected_sources=None, expected_hf=None):
    """Reconstruct every proposal/parent/retained row without new energy calls."""
    from .acquisition_v2 import AcquisitionSampler
    state = _fresh_state() if state is None else state
    samplers = ({a: AcquisitionSampler(a, catalogue, screening) for a in TRAINING}
                if samplers is None else samplers)
    reserved = {a: {tuple(x['ops']) for x in reservations[a]} for a in TRAINING}
    for envelope in records:
        item = envelope['payload']; event = item['event']
        if 'session_id' in item and 'elapsed_seconds' in item:
            value = item['elapsed_seconds']
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid elapsed time')
            state['sessions'][item['session_id']] = max(value, state['sessions'].get(item['session_id'], 0.))
        if event == 'session_started':
            if state['session_start'] is None:
                state['session_start'] = item['utc']
            if expected_sources is not None:
                require(item['source_bindings'] == expected_sources, 'Session source identity differs')
        elif event == 'request_started':
            require(not state['pending'] and not state['integrity_pending'],
                    'Resolve a previous pending request before another request')
            require(state['last_stop_reason'] is None and (state['preflight'] is None or
                    state['preflight']['projected_remaining_seconds_with_margin'] <= state['preflight']['remaining_wall_seconds']),
                    'A terminal acquisition decision cannot be resumed with more requests')
            require(type(item['request_ordinal']) is int and state['issued'] < 8192
                    and PRIOR_REQUESTS + state['issued'] < MAXIMUM_CUMULATIVE_REQUESTS
                    and item['request_ordinal'] == state['issued'] + 1,
                    'Request cap or global ordinal differs')
            alias, role, plan = next_request(state, samplers, reservations, policy) or (None, None, None)
            require(item['geometry_id'] == alias and item['role'] == role
                    and canonical_bytes(item['proposal']) == canonical_bytes(plan), 'Proposal/parent/frontier differs')
            key = digest({'geometry_id': alias, 'ops': plan['ops'], 'catalogue_sha256': catalogue['sha256']})
            require(item['query_sha256'] == key and item['duplicate_of'] == state['seen'].get(key)
                    and item['catalogue_sha256'] == catalogue['sha256']
                    and item['geometry_request_ordinal'] == state['counts'][alias] + 1,
                    'Request recipe, duplicate or catalogue identity differs')
            state['issued'] += 1; state['counts'][alias] += 1
            if role == 'fitting':
                state['fit_requests'][alias] += 1
            state['pending'][item['request_ordinal']] = item
            state['seen'].setdefault(key, item['request_ordinal'])
        elif event in ('request_failed', 'request_abandoned'):
            ordinal = item['request_ordinal']
            require(ordinal in state['pending'], 'Missing failed/abandoned intent')
            pending = state['pending'][ordinal]
            require(all(item.get(k, pending[k]) == pending[k] for k in ('geometry_id', 'role')),
                    'Failed/abandoned request identity differs')
            state['pending'].pop(ordinal)
            state['failed' if event == 'request_failed' else 'abandoned'] += 1
        elif event == 'request_completed':
            ordinal = item['request_ordinal']
            require(ordinal in state['pending'], 'Response lacks a charged intent')
            started = state['pending'].pop(ordinal)
            energy = item['energy_hartree']; norm = item['norm']
            require(type(energy) in (int, float) and math.isfinite(energy)
                    and item['query_sha256'] == started['query_sha256']
                    and item['geometry_id'] == started['geometry_id'] and item['role'] == started['role'],
                    'Response identity or energy differs')
            key = started['query_sha256']
            require(key not in state['successful_energies'] or abs(energy - state['successful_energies'][key]) <= 1e-8,
                    'Repeated physical circuit has inconsistent energies')
            state['successful_energies'].setdefault(key, energy)
            _norm_check(norm)
            require(type(item['hamiltonian_actions']) is int and item['hamiltonian_actions'] == 1,
                    'Hamiltonian accounting differs')
            require(_finite(item['state_preparation_seconds']) <= _finite(item['wall_seconds'])
                    and type(item['peak_rss_bytes']) is int
                    and 0 < item['peak_rss_bytes'] <= policy['memory_limit_bytes'],
                    'Invalid timing or peak memory')
            row = _eligible_row(state, started, energy, reserved)
            require(canonical_bytes(row) == canonical_bytes(item['eligible_row']), 'Retention or eligible row differs')
            alias, role = started['geometry_id'], started['role']
            if role == 'HF':
                require(_finite(item['HF_energy_discrepancy_ha']) <= 1e-8, 'HF differs from saved physical realization')
                if expected_hf is not None:
                    require(abs(energy - expected_hf[alias]) <= 1e-8
                            and abs(item['HF_energy_discrepancy_ha'] - abs(energy - expected_hf[alias])) <= 1e-12,
                            'HF response does not match the authenticated SCF scalar')
                state['hf'][alias] = float(energy)
            elif row is not None:
                state['retained'][alias].add(tuple(row['ops'])); state['eligible'].append(row)
                if role == 'diagnostic':
                    state['diagnostics'][alias][started['proposal']['diagnostic_ordinal']] = row
                else:
                    state['fitting'][alias].append(row)
            state['completed'] += 1
            state['completed_wall_seconds'] += item['wall_seconds']
            state['peak_rss_bytes'] = max(state['peak_rss_bytes'], item['peak_rss_bytes'])
        elif event == 'preflight':
            require(state['completed'] == policy['preflight_completed_requests'] and state['preflight'] is None,
                    'Preflight frontier differs')
            seconds = state['completed_wall_seconds'] / state['completed']
            projection = seconds * (8192 - state['issued']) * policy['schedule_projection_retry_factor']
            deadline = datetime.fromisoformat(state['session_start']).timestamp() + policy['max_wall_seconds']
            remaining = max(0., deadline - datetime.fromisoformat(item['utc']).timestamp())
            require(abs(_finite(item['seconds_per_request']) - seconds) < 1e-8
                    and abs(_finite(item['projected_remaining_seconds_with_margin']) - projection) < 1e-8
                    and abs(_finite(item['remaining_wall_seconds']) - remaining) < 1e-5,
                    'Measured preflight projection/window differs')
            state['preflight'] = item
        elif event == 'session_finished':
            require(item['stop_reason'] in ('target', 'budget', 'schedule'), 'Unknown stop reason')
            state['last_stop_reason'], state['last_stop_utc'] = item['stop_reason'], item['utc']
        elif event == 'session_interrupted':
            pass
        elif event == 'integrity_started':
            require(not state['integrity_pending'] and not state['pending'], 'Resolve pending integrity intent')
            require(type(item['integrity_ordinal']) is int and item['integrity_ordinal'] == state['integrity_issued'] + 1,
                    'Integrity request ordinal differs')
            candidates = {r['row_id']: r for r in state['eligible']}
            require(item['row_id'] in candidates and candidates[item['row_id']]['request_ordinal'] == item['request_ordinal']
                    and candidates[item['row_id']]['geometry_id'] == item['geometry_id']
                    and item['row_id'] not in state['integrity_checks'], 'Integrity row differs')
            state['integrity_issued'] += 1
            state['integrity_pending'][item['integrity_ordinal']] = item
        elif event == 'integrity_completed':
            pending = state['integrity_pending'].pop(item['integrity_ordinal'])
            require(all(pending[k] == item[k] for k in ('row_id', 'request_ordinal', 'geometry_id'))
                    and _finite(item['absolute_energy_discrepancy_ha']) <= 1e-8
                    and _finite(item['wall_seconds']) >= 0,
                    'Integrity response differs')
            _norm_check(item['norm'])
            state['integrity_checks'][item['row_id']] = item
            state['integrity_wall_seconds'] += item['wall_seconds']
        elif event == 'integrity_abandoned':
            require(item['integrity_ordinal'] in state['integrity_pending'], 'Missing abandoned integrity intent')
            state['integrity_pending'].pop(item['integrity_ordinal'])
        else:
            raise ValueError('Unknown acquisition journal event')
    return state


def _state_from_journal(journal, policy, catalogue, screening, reservations):
    return replay_records(journal.records, policy, catalogue, screening, reservations,
                          expected_sources=journal.expected_sources, expected_hf=journal.expected_hf)


def _expected_hf(root):
    manifest = read_json(root / 'large_qubit/physics/section3_v3/manifest.json')
    return {a: read_json(workspace_path(root, manifest['geometries'][a]['metadata_path'],
                                      namespace='large_qubit/scoring_only'))['scf']['total_energy_ha']
            for a in TRAINING}


def _integrity_checks(selected, evaluators, catalogue, state, append, consume):
    from .catalogue import circuit_gates
    expected_ids = []
    for alias in TRAINING:
        rows = selected['fitting_by_geometry'][alias]
        chosen = [rows[0], rows[len(rows) // 2], rows[-1], selected['diagnostic_by_geometry'][alias][0]]
        evaluator = evaluators[alias][0]
        for row in chosen:
            expected_ids.append(row['row_id'])
            if row['row_id'] in state['integrity_checks']:
                continue
            ordinal = state['integrity_issued'] + 1
            append({'event': 'integrity_started', 'integrity_ordinal': ordinal,
                    'row_id': row['row_id'], 'request_ordinal': row['request_ordinal'], 'geometry_id': alias})
            consume()
            began = time.perf_counter()
            vector = evaluator.basis.apply_circuit(evaluator.basis.hf_state(), circuit_gates(row['ops'], catalogue),
                                                  family='whole_qubit')
            measured = evaluator.energy_sector(vector)
            error = abs(measured - row['energy_hartree'])
            require(error <= 1e-8, 'Independent corpus label check differs')
            append({'event': 'integrity_completed', 'integrity_ordinal': ordinal,
                'row_id': row['row_id'], 'request_ordinal': row['request_ordinal'],
                'geometry_id': alias, 'absolute_energy_discrepancy_ha': error,
                'norm': evaluator.basis.diagnostics(vector), 'wall_seconds': time.perf_counter() - began})
            del vector
            state = consume()
    require(set(expected_ids) == set(state['integrity_checks']), 'Integrity sample identity differs')
    return {'all_passed': True, 'checks': [state['integrity_checks'][key] for key in expected_ids],
        'hamiltonian_actions': state['integrity_issued'], 'successful_label_checks': 12,
        'wall_seconds': state['integrity_wall_seconds'], 'admitted_to_search': False}


def _finish_frozen(root, public, inputs):
    """Idempotent finalization from immutable inputs, without launching physics."""
    from .corpus_v2 import build_release
    from .section4 import _reject_public_energy_values
    require(inputs['source_bindings'] == _sources(root)
            and inputs['preparation_file_sha256'] == sha256_file(root / PREPARATION_PATH),
            'Frozen finalization inputs refer to different source or preparation')
    private = workspace_path(root, PRIVATE_ROOT, namespace='large_qubit/scoring_only')
    actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(private.iterdir())
              if p.is_file() and not p.name.startswith('.')}
    require(actual == inputs['manifest_base']['files'], 'Frozen finalization raw evidence changed')
    arguments = inputs['release_arguments']
    release = None if arguments is None else build_release(root, **arguments)
    manifest = {**inputs['manifest_base'],
        'release': None if release is None else {'path': release['manifest_path'], 'manifest_sha256': release['manifest_sha256']},
        'section_6_complete': release is not None, 'feasibility_gate': 'go' if release else 'no_go'}
    _reject_public_energy_values(manifest)
    _commit_bytes(public / 'manifest.json', canonical_bytes(manifest) + b'\n')
    _commit_bytes(public / 'manifest.sha256', (sha256_file(public / 'manifest.json') + '\n').encode('ascii'))
    return verify_section6(root)


def run_section6(workspace, output=DEFAULT_SECTION6, *, resume=False):
    """Execute or explicitly resume this campaign; a new name cannot reset its cap."""
    import fcntl
    from .acquisition_v2 import AcquisitionSampler, quality_report
    from .catalogue import circuit_gates
    from .section4 import load_co2_evaluator, _reject_public_energy_values
    from .benchmark import _peak_rss_bytes
    from .corpus_v2 import select_balanced_rows, build_release
    root = Path(workspace).resolve(strict=True)
    require(output == DEFAULT_SECTION6, 'This frozen campaign has one output identity; do not reset its budget')
    public = workspace_path(root, output, namespace='large_qubit/validation')
    policy, catalogue, screening, reservations, prep = verify_preparation(root)
    private = workspace_path(root, PRIVATE_ROOT, namespace='large_qubit/scoring_only')
    require(not (public / 'manifest.json').exists() or (resume and (public / 'finalization_input.json').exists()),
            'Preserve the completed or NO-GO campaign')
    require(resume == public.exists(), 'Use a new launch once, or explicit --resume for the existing campaign')
    public.mkdir(parents=True, exist_ok=True)
    lock = (private / '.controller.lock').open('a+b')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('Another Section6 controller holds the campaign lock')
    if (public / 'finalization_input.json').exists():
        try:
            return _finish_frozen(root, public, read_json(public / 'finalization_input.json'))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    journal = Journal(private)
    journal.expected_sources = _sources(root)
    journal.expected_hf = _expected_hf(root)
    session, began = uuid.uuid4().hex, time.perf_counter()
    def append(item):
        return journal.append({**item, 'session_id': session, 'elapsed_seconds': time.perf_counter() - began})
    try:
        state = _state_from_journal(journal, policy, catalogue, screening, reservations)
        append({'event': 'session_started', 'utc': _utc(), 'resume': resume,
                'source_bindings': _sources(root), 'preserved_partial_tails': journal.tails})
        for ordinal in list(state['pending']):
            append({'event': 'request_abandoned', 'request_ordinal': ordinal,
                    'reason': 'durable intent without a durable response; charge retained'})
        for ordinal in list(state['integrity_pending']):
            append({'event': 'integrity_abandoned', 'integrity_ordinal': ordinal,
                    'reason': 'durable integrity intent without a response; overhead charge retained'})
        state = _state_from_journal(journal, policy, catalogue, screening, reservations)
        start_date = datetime.fromisoformat(state['session_start'])
        deadline = start_date.timestamp() + policy['max_wall_seconds']
        evaluators = {a: load_co2_evaluator(root, a, threads=policy['cpu_threads']) for a in TRAINING}
        samplers = {a: AcquisitionSampler(a, catalogue, screening) for a in TRAINING}
        reserved = {a: {tuple(r['ops']) for r in reservations[a]} for a in TRAINING}
        cursor = len(journal.records)
        def consume():
            nonlocal cursor, state
            state = replay_records(journal.records[cursor:], policy, catalogue, screening, reservations,
                state=state, samplers=samplers, expected_sources=journal.expected_sources,
                expected_hf=journal.expected_hf)
            cursor = len(journal.records)
            return state
        stop_reason = 'target'
        while True:
            if state['last_stop_reason'] is not None:
                stop_reason = state['last_stop_reason']; break
            if state['preflight'] is not None and (state['preflight']['projected_remaining_seconds_with_margin']
                                                  > state['preflight']['remaining_wall_seconds']):
                stop_reason = 'schedule'; break
            if all(len(state['fitting'][a]) >= 2048 for a in TRAINING):
                break
            if time.time() + 30 >= deadline:
                stop_reason = 'schedule'; break
            if state['completed'] == policy['preflight_completed_requests'] and state['preflight'] is None:
                seconds = state['completed_wall_seconds'] / state['completed']
                projection = seconds * (8192 - state['issued']) * policy['schedule_projection_retry_factor']
                preflight_utc = _utc()
                remaining = max(0., deadline - datetime.fromisoformat(preflight_utc).timestamp())
                append({'event': 'preflight', 'utc': preflight_utc, 'seconds_per_request': seconds,
                    'projected_remaining_seconds_with_margin': projection,
                    'remaining_wall_seconds': remaining,
                    'ordinary_requests_included_in_campaign': True})
                consume()
                if projection > remaining:
                    stop_reason = 'schedule'; break
            choice = next_request(state, samplers, reservations, policy)
            if choice is None or state['issued'] >= 8192:
                stop_reason = 'budget'; break
            alias, role, plan = choice
            ordinal = state['issued'] + 1
            key = digest({'geometry_id': alias, 'ops': plan['ops'], 'catalogue_sha256': catalogue['sha256']})
            started = {'event': 'request_started', 'request_ordinal': ordinal,
                'geometry_request_ordinal': state['counts'][alias] + 1, 'geometry_id': alias, 'role': role,
                'proposal': plan, 'catalogue_sha256': catalogue['sha256'], 'query_sha256': key,
                'duplicate_of': state['seen'].get(key), 'utc': _utc()}
            append(started)
            request_start = time.perf_counter()
            try:
                evaluator, metadata = evaluators[alias]
                vector = evaluator.basis.apply_circuit(evaluator.basis.hf_state(), circuit_gates(plan['ops'], catalogue),
                                                       family='whole_qubit')
                preparation_seconds = time.perf_counter() - request_start
                before = evaluator.statistics()['apply_h_calls']
                energy = float(evaluator.energy_sector(vector))
                norm = evaluator.basis.diagnostics(vector)
                peak = _peak_rss_bytes()
                require(peak <= policy['memory_limit_bytes'], 'Acquisition exceeds its process RAM allowance')
                row = _eligible_row(state, started, energy, reserved)
                response = {'event': 'request_completed', 'request_ordinal': ordinal,
                    'geometry_id': alias, 'role': role, 'query_sha256': key, 'energy_hartree': energy,
                    'eligible_row': row, 'norm': norm, 'state_preparation_seconds': preparation_seconds,
                    'wall_seconds': time.perf_counter() - request_start, 'peak_rss_bytes': peak,
                    'hamiltonian_actions': evaluator.statistics()['apply_h_calls'] - before, 'utc': _utc()}
                if role == 'HF':
                    response['HF_energy_discrepancy_ha'] = abs(energy - metadata['scf']['total_energy_ha'])
                    require(response['HF_energy_discrepancy_ha'] <= 1e-8, 'Fresh HF differs from saved SCF')
                append(response)
                del vector
            except Exception as error:
                append({'event': 'request_failed', 'request_ordinal': ordinal, 'geometry_id': alias,
                    'role': role, 'error_type': type(error).__name__, 'error': str(error)})
                raise
            # Apply only new records; complete replay is reserved for resume/final verification.
            consume()
            if state['issued'] % 30 == 0:
                print('Section6: ' + json.dumps({'requests': state['issued'], 'cap': 8192,
                    'fitting': {a: len(state['fitting'][a]) for a in TRAINING},
                    'diagnostics': {a: len(state['diagnostics'][a]) for a in TRAINING}}), flush=True)
        append({'event': 'session_finished', 'stop_reason': stop_reason, 'utc': _utc()})
        consume()
        rows_path = private / f'eligible_rows_{session}.jsonl'
        with rows_path.open('xb') as handle:
            for row in state['eligible']:
                handle.write(canonical_bytes(row) + b'\n')
        quality, integrity, release_arguments = None, None, None
        selection_error = None
        try:
            selected = select_balanced_rows(state['eligible'], catalogue=catalogue, stop_reason=stop_reason)
        except ValueError as error:
            selection_error = str(error)
        else:
            quality = quality_report(selected['fitting_by_geometry'], selected['diagnostic_by_geometry'],
                                     state['hf'], catalogue)
            if quality['all_passed']:
                integrity = _integrity_checks(selected, evaluators, catalogue, state, append, consume)
                _write(private / f'label_integrity_{session}.json', integrity)
                journal.close()
                raw = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(private.iterdir())
                       if p.is_file() and not p.name.startswith('.')}
                release_arguments = {'eligible_rows_path': rows_path.relative_to(root).as_posix(),
                    'catalogue': catalogue, 'quality_report': quality, 'stop_reason': stop_reason,
                    'prior_acquisition_cost': prep['prior_acquisition_cost'],
                    'acquisition_cost': {'logical_energy_requests': state['issued'],
                                      'wall_seconds': sum(state['sessions'].values()) - integrity['wall_seconds']},
                    'verification_cost': {'hamiltonian_actions': integrity['hamiltonian_actions'],
                                       'wall_seconds': integrity['wall_seconds']},
                    'raw_artifacts': raw}
        if not journal.handle.closed:
            journal.close()
        files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(private.iterdir())
                 if p.is_file() and not p.name.startswith('.')}
        manifest = {'schema_version': 'section6-validation-v2', 'created_utc': _utc(),
            'benchmark_id': BENCHMARK_ID, 'output': output, 'restricted_output': PRIVATE_ROOT,
            'preparation_file_sha256': sha256_file(root / PREPARATION_PATH),
            'eligible_rows_path': rows_path.relative_to(root).as_posix(),
            'label_integrity_path': None if integrity is None else (private / f'label_integrity_{session}.json').relative_to(root).as_posix(),
            'source_bindings': _sources(root), 'policy': policy, 'policy_sha256': digest(policy),
            'approved_amendment': policy['amendment'],
            'prior_acquisition_cost': prep['prior_acquisition_cost'],
            'cumulative_ordinary_requests': PRIOR_REQUESTS + state['issued'],
            'ordinary_requests': state['issued'], 'completed_requests': state['completed'],
            'failed_requests': state['failed'], 'abandoned_requests': state['abandoned'],
            'requests_by_geometry': state['counts'],
            'retained_fitting_by_geometry': {a: len(state['fitting'][a]) for a in TRAINING},
            'retained_diagnostics_by_geometry': {a: len(state['diagnostics'][a]) for a in TRAINING},
            'stop_reason': stop_reason, 'quality': quality, 'selection_error': selection_error,
            'release': None, 'section_6_complete': False, 'feasibility_gate': 'pending_finalization',
            'separate_integrity_hamiltonian_actions': 0 if integrity is None else integrity['hamiltonian_actions'],
            'peak_rss_bytes': state['peak_rss_bytes'],
            'recorded_active_wall_seconds': sum(state['sessions'].values()),
            'preflight': state['preflight'], 'files': files,
            'pending_or_failed_attempts_preserved': True, 'heldout_energy_or_reference_access': False,
            'model_training_executed': False, 'execution_ready': False}
        inputs = {'source_bindings': _sources(root), 'preparation_file_sha256': sha256_file(root / PREPARATION_PATH),
                  'manifest_base': manifest, 'release_arguments': release_arguments}
        _commit_bytes(public / 'finalization_input.json', canonical_bytes(inputs) + b'\n')
        return _finish_frozen(root, public, inputs)
    except BaseException as error:
        if not journal.handle.closed:
            try:
                append({'event': 'session_interrupted', 'utc': _utc(), 'error_type': type(error).__name__})
            except OSError:
                pass
            journal.close()
        _write(public / 'attempt_errors' / f'{session}.json', {'created_utc': _utc(),
            'session_id': session, 'observed_elapsed_seconds': time.perf_counter() - began,
            'error_type': type(error).__name__, 'error': str(error),
            'partial_evidence_preserved': True, 'resume_requires_explicit_flag': True,
            'model_training_executed': False})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def verify_section6(workspace, section6_root=DEFAULT_SECTION6, *, numerical_integrity=False):
    """Authenticate/replay source, all requests and the release; default uses no H action."""
    from .corpus_v2 import select_balanced_rows, verify_release
    from .acquisition_v2 import quality_report
    root = Path(workspace).resolve(strict=True)
    policy, catalogue, screening, reservations, prep = verify_preparation(root)
    folder = workspace_path(root, section6_root, namespace='large_qubit/validation')
    path = folder / 'manifest.json'; m = read_json(path)
    require((folder / 'manifest.sha256').read_text(encoding='ascii').strip() == sha256_file(path),
            'Section6 manifest sidecar differs')
    require(m['schema_version'] == 'section6-validation-v2' and m['source_bindings'] == _sources(root)
            and m['policy_sha256'] == digest(policy) and canonical_bytes(m['policy']) == canonical_bytes(policy)
            and m['preparation_file_sha256'] == sha256_file(root / PREPARATION_PATH)
            and m['output'] == section6_root and m['restricted_output'] == PRIVATE_ROOT,
            'Section6 source, preparation or scope differs')
    require(m['approved_amendment'] == policy['amendment']
            and canonical_bytes(m['prior_acquisition_cost']) == canonical_bytes(prep['prior_acquisition_cost'])
            and type(m['cumulative_ordinary_requests']) is int
            and m['cumulative_ordinary_requests'] == PRIOR_REQUESTS + m['ordinary_requests'] <= MAXIMUM_CUMULATIVE_REQUESTS,
            'Prior failed-attempt cost or cumulative authorized budget differs')
    private = workspace_path(root, PRIVATE_ROOT, namespace='large_qubit/scoring_only')
    actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(private.iterdir())
              if p.is_file() and not p.name.startswith('.')}
    require(actual == m['files'], 'Section6 raw payload inventory or bytes differ')
    records, tails = read_journal(private)
    state = replay_records(records, policy, catalogue, screening, reservations,
                           expected_sources=_sources(root), expected_hf=_expected_hf(root))
    require(not state['pending'] and state['issued'] == m['ordinary_requests']
            and state['completed'] == m['completed_requests'] and state['failed'] == m['failed_requests']
            and state['abandoned'] == m['abandoned_requests'] and state['counts'] == m['requests_by_geometry'],
            'Request counts or unfinished frontier differs')
    require(m['stop_reason'] == state['last_stop_reason'], 'Saved stop reason differs from journal')
    if m['stop_reason'] == 'target':
        require(all(len(state['fitting'][a]) >= 2048 for a in TRAINING), 'Target stop lacks the complete corpus')
    elif m['stop_reason'] == 'budget':
        from .acquisition_v2 import AcquisitionSampler
        samplers = {a: AcquisitionSampler(a, catalogue, screening) for a in TRAINING}
        require(state['issued'] == 8192 or next_request(state, samplers, reservations, policy) is None,
                'Budget stop has an admissible uncharged frontier')
    else:
        cutoff = datetime.fromisoformat(state['session_start']).timestamp() + policy['max_wall_seconds'] - 30
        stopped = datetime.fromisoformat(state['last_stop_utc']).timestamp()
        require(stopped >= cutoff or (state['preflight'] is not None and
                state['preflight']['projected_remaining_seconds_with_margin'] > state['preflight']['remaining_wall_seconds']),
                'Schedule stop lacks measured or elapsed schedule exhaustion')
    rows_path = workspace_path(root, m['eligible_rows_path'], namespace=PRIVATE_ROOT)
    saved_rows = [json.loads(line) for line in rows_path.read_text(encoding='ascii').splitlines()]
    require(canonical_bytes(saved_rows) == canonical_bytes(state['eligible']), 'Eligible row retention differs')
    require(m['retained_fitting_by_geometry'] == {a: len(state['fitting'][a]) for a in TRAINING}
            and m['retained_diagnostics_by_geometry'] == {a: len(state['diagnostics'][a]) for a in TRAINING}
            and m['model_training_executed'] is False and m['heldout_energy_or_reference_access'] is False
            and m['execution_ready'] is False, 'Population or execution scope differs')
    require(type(m['peak_rss_bytes']) is int and m['peak_rss_bytes'] == state['peak_rss_bytes']
            and abs(_finite(m['recorded_active_wall_seconds']) - sum(state['sessions'].values())) <= 1e-8
            and canonical_bytes(m['preflight']) == canonical_bytes(state['preflight']),
            'Saved timing, peak memory or preflight summary differs')
    selection_error = None
    try:
        selected = select_balanced_rows(state['eligible'], catalogue=catalogue, stop_reason=m['stop_reason'])
    except ValueError as error:
        selection_error = str(error)
        require(m['quality'] is None and m['release'] is None, 'Insufficient rows cannot supply a release')
    else:
        quality = quality_report(selected['fitting_by_geometry'], selected['diagnostic_by_geometry'], state['hf'], catalogue)
        require(canonical_bytes(quality) == canonical_bytes(m['quality'])
                and (m['release'] is not None) == quality['all_passed'],
                'Released-prefix quality or decision differs')
    require(m['selection_error'] == selection_error, 'Prefix selection outcome differs')
    if m['release'] is not None:
        verify_release(root, expected_release_sha256=m['release']['manifest_sha256'], release_path=m['release']['path'])
        released = read_json(workspace_path(root, m['release']['path'], namespace='large_qubit/releases'))
        require(canonical_bytes(released['prior_acquisition_cost']) == canonical_bytes(prep['prior_acquisition_cost']),
                'Release dropped or changed the failed first acquisition cost')
        require(released['cumulative_acquisition_cost']['logical_energy_requests'] == PRIOR_REQUESTS + state['issued']
                and abs(released['cumulative_acquisition_cost']['wall_seconds'] -
                        (prep['prior_acquisition_cost']['wall_seconds'] + sum(state['sessions'].values())
                         - state['integrity_wall_seconds'])) <= 1e-8
                and released['campaign_production_request_ceiling'] == 8192
                and released['cumulative_production_request_ceiling'] == MAXIMUM_CUMULATIVE_REQUESTS
                and released['prior_attempt_labels_imported'] is False,
                'Released cumulative source cost or campaign provenance differs')
        require(released['acquisition_cost']['logical_energy_requests'] == state['issued']
                and abs(released['acquisition_cost']['wall_seconds'] -
                        (sum(state['sessions'].values()) - state['integrity_wall_seconds'])) <= 1e-8
                and released['separate_verification_cost']['hamiltonian_actions'] == state['integrity_issued']
                and abs(released['separate_verification_cost']['wall_seconds'] - state['integrity_wall_seconds']) <= 1e-8,
                'Released source acquisition/verification costs differ from the charged journal')
        integrity = read_json(workspace_path(root, m['label_integrity_path'], namespace=PRIVATE_ROOT))
        require(integrity['all_passed'] and len(integrity['checks']) == integrity['successful_label_checks'] == 12
                and integrity['hamiltonian_actions'] == state['integrity_issued'] >= 12
                and not state['integrity_pending']
                and all(c['absolute_energy_discrepancy_ha'] <= 1e-8 for c in integrity['checks'])
                and integrity['admitted_to_search'] is False, 'Saved label verification differs')
        expected_ids = []
        for alias in TRAINING:
            rows = selected['fitting_by_geometry'][alias]
            expected_ids.extend([rows[0]['row_id'], rows[len(rows) // 2]['row_id'], rows[-1]['row_id'],
                                 selected['diagnostic_by_geometry'][alias][0]['row_id']])
        require([check['row_id'] for check in integrity['checks']] == expected_ids
                and canonical_bytes(integrity['checks']) == canonical_bytes([state['integrity_checks'][key] for key in expected_ids])
                and m['separate_integrity_hamiltonian_actions'] == state['integrity_issued'],
                'Independent label sample or overhead differs')
    complete = m['release'] is not None
    require(m['section_6_complete'] is complete and m['feasibility_gate'] == ('go' if complete else 'no_go'),
            'Scientific decision differs')
    replays = 0
    if numerical_integrity and complete:
        from .sector import SectorBasis
        from .catalogue import circuit_gates
        basis = SectorBasis(15, 11, 11, cache_budget_bytes=16 * 1024 ** 2)
        by_id = {r['row_id']: r for r in state['eligible']}
        for check in integrity['checks']:
            row = by_id[check['row_id']]
            vector = basis.apply_circuit(basis.hf_state(), circuit_gates(row['ops'], catalogue), family='whole_qubit')
            norm = basis.diagnostics(vector)
            require(norm['norm_error'] <= 1e-10, 'Saved label state replay lost normalization')
            replays += 1
    return {'all_passed': True, 'manifest_sha256': sha256_file(path), 'section_6_complete': complete,
        'amended_confirmation': True, 'prior_ordinary_requests': PRIOR_REQUESTS,
        'cumulative_ordinary_requests': PRIOR_REQUESTS + state['issued'],
        'maximum_cumulative_ordinary_requests': MAXIMUM_CUMULATIVE_REQUESTS,
        'feasibility_gate': m['feasibility_gate'], 'ordinary_requests': state['issued'],
        'production_ordinary_requests': state['issued'],
        'production_request_ceiling': 8192, 'remaining_production_allowance': 8192 - state['issued'],
        'released_rows_per_geometry': None if not complete else selected['rows_per_geometry'],
        'release_sha256': None if not complete else m['release']['manifest_sha256'],
        'payload_files_verified': len(actual), 'journal_records_verified': len(records),
        'uncommitted_tail_fragments_preserved': len(tails), 'sampler_and_parent_trajectories_replayed': True,
        'additional_hamiltonian_actions': 0, 'numerical_states_replayed': replays,
        'separate_saved_integrity_hamiltonian_actions': m['separate_integrity_hamiltonian_actions'],
        'model_training_executed': False, 'execution_ready': False}


def main(argv=None):
    """Explicit v2 entry point; previews never start physical work."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('prepare', help='Preview, freeze or verify the approved v2 policy')
    preparation_mode = prepare.add_mutually_exclusive_group()
    preparation_mode.add_argument('--execute', action='store_true')
    preparation_mode.add_argument('--verify', action='store_true')
    acquire = commands.add_parser('acquire', help='Preview or explicitly execute this single CPU campaign')
    acquire.add_argument('--execute', action='store_true')
    acquire.add_argument('--resume', action='store_true')
    acquire.add_argument('--output', default=DEFAULT_SECTION6)
    verify = commands.add_parser('verify', help='Authenticate the saved v2 decision; no new H action')
    verify.add_argument('--section6-root', default=DEFAULT_SECTION6)
    verify.add_argument('--numerical-integrity', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'verify':
        result = verify_section6(args.root, args.section6_root, numerical_integrity=args.numerical_integrity)
    elif args.action == 'prepare' and args.verify:
        policy, catalogue, screening, reservations, preparation = verify_preparation(args.root)
        result = {'all_passed': True, 'mode': 'prepared_metadata_authenticated',
                  'preparation': preparation, 'new_hamiltonian_actions': 0}
    elif args.action == 'prepare' and args.execute:
        result = prepare_section6(args.root)
    elif args.action == 'acquire' and args.execute:
        result = run_section6(args.root, args.output, resume=args.resume)
    else:
        require(not getattr(args, 'resume', False), 'Resume requires explicit --execute')
        from .section5 import verify_preparation as verify_previous
        _approved_amendment(args.root)
        _, catalogue, _, _ = verify_previous(args.root)
        result = {'mode': 'preview_only', 'policy': build_section6_policy(catalogue),
                  'new_hamiltonian_actions': 0, 'model_training_executed': False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
