"""Deterministic final-matrix reduction, preserving failures and early incumbents.

The caller authenticates ledger heads, references, verification and identities
against its sealed artifact inventory. This module checks their numerical and
semantic agreement; an embedded success flag does not authenticate an artifact.
No state, Hamiltonian, numerical backend or language model is loaded here.
"""

from copy import deepcopy
import math

from .common import digest, require
from .protocol import METHODS
from .section8_policy import attempt_seed
from .section9_scoring import normalize_query


SCHEMA = 'section9-trajectory-report-v1'
TARGETS = ('CO2_I01', 'CO2_I02')
BUDGET = 1024
ATTEMPTS = 2048
ENERGY_TOLERANCE = 1e-8
NORM_TOLERANCE = 1e-10
PAIRS = {
    'gpt_weighted_minus_uniform': ('gpt2_xl_weighted_lora', 'gpt2_xl_uniform_lora'),
    'gpt_weighted_minus_frozen': ('gpt2_xl_weighted_lora', 'gpt2_xl_frozen'),
    'qwen_weighted_minus_frozen': ('qwen3_4b_weighted_lora', 'qwen3_4b_frozen'),
    'qwen_weighted_minus_gpt_weighted': ('qwen3_4b_weighted_lora', 'gpt2_xl_weighted_lora'),
}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _number(value, label):
    require(_finite(value), label + ' must be finite numeric data')
    return float(value)


def verify_reference(reference, cache_identity):
    """Validate the saved two-start target reference without recomputing it."""
    require(type(reference) is dict and type(cache_identity) is dict,
            'A target reference and exact physics identity are required')
    alias = cache_identity.get('geometry_id')
    require(alias in TARGETS and cache_identity.get('partition') == 'final_test'
            and reference.get('geometry_id') == alias
            and reference.get('partition') == 'final_test'
            and reference.get('cache_identity') == cache_identity,
            'Reference target, partition or Hamiltonian identity differs')
    report = reference.get('report', {})
    require(type(report) is dict, 'Reference convergence report must be a mapping')
    runs = report.get('runs', [])
    require(report.get('all_passed') is True and report.get('converged') is True
            and type(report.get('independent_starts')) is int and report['independent_starts'] == 2
            and type(runs) is list and len(runs) == 2,
            'Reference requires two converged independent starts')
    energies = []
    for run in runs:
        require(type(run) is dict and run.get('converged') is True,
                'A reference start did not converge')
        energies.append(_number(run.get('energy_ha'), 'Reference start energy'))
        residual = _number(run.get('complex128_residual_norm_ha'), 'Reference residual')
        norm = _number(run.get('norm_squared'), 'Reference norm')
        require(0. <= residual <= ENERGY_TOLERANCE and abs(norm - 1.) <= NORM_TOLERANCE,
                'Reference residual or normalization gate failed')
    energy = _number(reference.get('energy_hartree'), 'Reference energy')
    require(abs(energy - min(energies)) <= 1e-12 and abs(energies[0] - energies[1]) <= ENERGY_TOLERANCE,
            'Reference energies or independent-start agreement differ')
    return {'all_passed': True, 'geometry_id': alias, 'energy_hartree': energy,
            'maximum_residual_hartree': max(r['complex128_residual_norm_ha'] for r in runs)}


def _evidence_issues(evidence, query, cache_identity, energy, *, independent=False):
    issues = []
    if type(evidence) is not dict:
        return ['missing physical evaluation evidence']
    if (evidence.get('query_sha256') != digest(query)
            or evidence.get('cache_identity') != cache_identity
            or evidence.get('geometry_id') != cache_identity['geometry_id']
            or evidence.get('partition') != 'final_test'):
        issues.append('physical query, target or Hamiltonian identity differs')
    norm = evidence.get('norm')
    if not _finite(norm) or abs(norm - 1.) > NORM_TOLERANCE or evidence.get('populations') != [11, 11]:
        issues.append('physical normalization or fixed populations failed')
    measured = evidence.get('energy_hartree')
    if not _finite(energy) or not _finite(measured) or abs(measured - energy) > ENERGY_TOLERANCE:
        issues.append('physical energy is nonfinite or disagrees')
    if independent and evidence.get('fresh_basis_and_evaluator') is not True:
        issues.append('winning circuit lacks fresh independent state/evaluator reconstruction')
    return issues


def _cost(rows, *, generation=False):
    """Recorded marginal work only; unavailable receipts make a lower bound."""
    result = {'recorded_wall_seconds': 0., 'records': len(rows), 'missing_cost_records': 0,
              'wall_is_lower_bound': False, 'numerical_work_is_lower_bound': False}
    fields = ('input_tokens', 'output_tokens') if generation else (
        'evaluator_calls', 'kernel_contractions', 'hamiltonian_action_seconds')
    result.update({name: 0 for name in fields})
    for row in rows:
        if not generation and row.get('cache_hit') is True:
            # Origin evaluation proves the scalar but its producer paid that
            # numerical work. Reuse is charged logically, with zero new kernel work.
            continue
        receipt = row.get('output' if generation else 'evaluation')
        if type(receipt) is not dict or not _finite(receipt.get('wall_seconds')) or receipt['wall_seconds'] < 0:
            result['missing_cost_records'] += 1
            result['wall_is_lower_bound'] = True
        else:
            result['recorded_wall_seconds'] += receipt['wall_seconds']
        if type(receipt) is dict:
            result['wall_is_lower_bound'] |= bool(receipt.get('wall_is_lower_bound') or receipt.get('partial'))
            for name in fields:
                value = receipt.get(name)
                if _finite(value) and value >= 0:
                    result[name] += value
                else:
                    result['numerical_work_is_lower_bound'] = True
        else:
            result['numerical_work_is_lower_bound'] = True
        result['wall_is_lower_bound'] |= row.get('status') in ('pending', 'abandoned')
        result['numerical_work_is_lower_bound'] |= row.get('status') in ('pending', 'abandoned')
    return result


def trajectory_report(snapshot, *, method, geometry_id, verification=None, reference=None, failure=None):
    """Reduce one authenticated trajectory; scientific failures stay reportable.

    Failed early trajectories retain their verified incumbent at B1024, with
    their failed status and unused allowance explicit. They do not enter a
    primary two-target mean: both target trajectories must complete and verify.
    A physical integrity mismatch withholds even the per-target verified
    endpoint rather than hiding the raw incumbent.
    """
    require(method in METHODS and geometry_id in TARGETS, 'Unknown fixed main-matrix job')
    require(type(snapshot) is dict and type(snapshot.get('identity')) is dict, 'Missing trajectory identity')
    identity = snapshot['identity']
    require(identity.get('method') == method and identity.get('geometry_id') == geometry_id,
            'Trajectory method or target identity differs')
    cache_identity = identity.get('cache_identity')
    require(type(cache_identity) is dict and cache_identity.get('geometry_id') == geometry_id
            and cache_identity.get('partition') == 'final_test', 'Trajectory physics partition differs')
    require(failure is None or type(failure) is dict, 'Failure outcome must be an explicit mapping')
    requests, attempts = snapshot.get('requests'), snapshot.get('attempts')
    require(type(requests) is list and type(attempts) is list
            and len(requests) <= BUDGET and len(attempts) <= ATTEMPTS,
            'Trajectory exceeds the frozen request or attempt budget')
    for records in (requests, attempts):
        require(all(type(r) is dict and type(r.get('ordinal')) is int for r in records)
                and [r.get('ordinal') for r in records] == list(range(1, len(records) + 1)),
                'Trajectory ordinal frontier is not contiguous')
    require(all(r.get('status') in ('completed', 'failed', 'abandoned', 'pending') for r in requests)
            and all(a.get('status') in ('completed', 'abandoned', 'pending') for a in attempts),
            'Unknown durable request or completion status')
    if 'logical_requests' in snapshot:
        require(snapshot['logical_requests'] == len(requests), 'Logical-request counter differs')
    if 'completion_attempts' in snapshot:
        require(snapshot['completion_attempts'] == len(attempts), 'Completion-attempt counter differs')
    require(method in METHODS[:5] or not attempts, 'A non-generator method cannot have model completion attempts')
    issues = []
    if requests and (requests[0].get('kind') != 'hf' or requests[0].get('call_id') != 'hf'
                     or type(requests[0].get('query')) is not dict or requests[0]['query'].get('kind') != 'hf'):
        issues.append('first logical request is not the unique initial HF request')
    for attempt in attempts:
        require(attempt.get('input', {}).get('seed') == attempt_seed(method, geometry_id, attempt['ordinal']),
                'Recorded generation seed differs from its frozen method/target/ordinal stream')
    completed = [r for r in requests if r.get('status') == 'completed']
    finite_completed = [r for r in completed if _finite(r.get('energy_hartree'))]
    for record in completed:
        ordinal, query = record['ordinal'], record.get('query')
        try:
            normalized = normalize_query(query, geometry_id)
            if normalized != query:
                issues.append(f'request {ordinal}: target query is not canonical')
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            issues.append(f'request {ordinal}: invalid physical query: {error}')
        evaluation = record.get('evaluation')
        evidence = evaluation.get('origin_evaluation', evaluation) if type(evaluation) is dict else evaluation
        for issue in _evidence_issues(evidence, query, cache_identity, record.get('energy_hartree')):
            issues.append(f'request {ordinal}: {issue}')
        if record.get('cache_hit') is True and type(evaluation) is dict:
            if any(evaluation.get(k, 0) != 0 for k in ('evaluator_calls', 'kernel_contractions', 'wall_seconds')):
                issues.append(f'request {ordinal}: cache hit claims new physical numerical work')
        linked = record.get('attempt_ordinal')
        if linked is not None:
            valid_link = (type(linked) is int and 1 <= linked <= len(attempts)
                          and attempts[linked - 1].get('status') == 'completed'
                          and attempts[linked - 1].get('valid') is True)
            output = attempts[linked - 1].get('output') if valid_link else None
            if not (valid_link and type(output) is dict and type(query) is dict and output.get('ops') == query.get('ops')
                    and output.get('condition_id', method) == method
                    and output.get('geometry_id', geometry_id) == geometry_id):
                issues.append(f'request {ordinal}: scored circuit differs from its valid generation attempt')
        elif method in METHODS[:5] and ordinal > 1:
            issues.append(f'request {ordinal}: generator candidate lacks a valid attempt link')
    for record in requests:
        partial = record.get('evaluation')
        if record.get('status') == 'completed' or type(partial) is not dict:
            continue
        label = f'request {record["ordinal"]} partial failure: '
        if 'norm' in partial and (not _finite(partial['norm']) or abs(partial['norm'] - 1.) > NORM_TOLERANCE):
            issues.append(label + 'normalization failed')
        if 'populations' in partial and partial['populations'] != [11, 11]:
            issues.append(label + 'fixed populations failed')
        if 'cache_identity' in partial and partial['cache_identity'] != cache_identity:
            issues.append(label + 'Hamiltonian identity differs')
        if 'query_sha256' in partial and partial['query_sha256'] != digest(record.get('query')):
            issues.append(label + 'physical query identity differs')
    best = min(finite_completed, key=lambda r: (r['energy_hartree'], r['ordinal'])) if finite_completed else None
    resolved, reference_status = None, 'missing'
    if reference is not None:
        try:
            resolved = verify_reference(reference, cache_identity)
            reference_status = 'resolved'
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            reference_status = 'invalid'
            issues.append('reference: ' + str(error))
    verified, verification_status = False, 'missing'
    if verification is not None:
        if best is None:
            verification_status = 'invalid'
            issues.append('verification supplied without a finite scored incumbent')
        else:
            check = _evidence_issues(verification, best['query'], cache_identity,
                                     best['energy_hartree'], independent=True)
            verification_status = 'invalid' if check else 'verified'
            issues.extend('verification: ' + issue for issue in check)
            verified = not check
    error = None
    if best is not None and resolved is not None:
        raw_error = best['energy_hartree'] - resolved['energy_hartree']
        verified_error = (verification['energy_hartree'] - resolved['energy_hartree']) if verified else None
        if not _finite(raw_error) or raw_error < -ENERGY_TOLERANCE:
            issues.append('incumbent lies below the resolved same-Hamiltonian reference beyond tolerance')
        elif verified_error is not None:
            if not _finite(verified_error) or verified_error < -ENERGY_TOLERANCE:
                issues.append('verified incumbent lies below reference beyond tolerance')
            else:
                error = verified_error  # Signed; never absolute-valued or clipped.
    counts = {
        'logical_requests': len(requests), 'completion_attempts': len(attempts),
        'completed_requests': len(completed),
        'failed_requests': sum(r.get('status') == 'failed' for r in requests),
        'abandoned_requests': sum(r.get('status') == 'abandoned' for r in requests),
        'pending_requests': sum(r.get('status') == 'pending' for r in requests),
        'valid_completion_attempts': sum(a.get('status') == 'completed' and a.get('valid') is True for a in attempts),
        'invalid_completion_attempts': sum(a.get('status') == 'completed' and a.get('valid') is False for a in attempts),
        'abandoned_completion_attempts': sum(a.get('status') == 'abandoned' for a in attempts),
        'pending_completion_attempts': sum(a.get('status') == 'pending' for a in attempts),
        'infrastructure_completion_failures': sum(type(a.get('output')) is dict
            and a['output'].get('error_kind') == 'infrastructure' for a in attempts),
        'cache_hits': sum(r.get('cache_hit') is True for r in requests),
        'duplicate_requests': sum(r.get('duplicate_of') is not None for r in requests),
    }
    pending = counts['pending_requests'] or counts['pending_completion_attempts']
    budget_complete = (len(requests) == 1023 if method == 'conventional_vqe'
        else len(requests) == BUDGET if method in METHODS[5:7]
        else len(requests) == BUDGET or len(attempts) == ATTEMPTS)
    failed = bool(failure is not None or issues or counts['failed_requests'] or counts['abandoned_requests']
                  or counts['infrastructure_completion_failures'] or snapshot.get('halted'))
    status = 'failed' if failed else 'complete' if budget_complete and not pending else 'incomplete'
    endpoint_available = bool(best is not None and verified and resolved is not None and not issues
                              and status in ('complete', 'failed') and not pending)
    incumbent, curve = None, []
    for ordinal in range(1, BUDGET + 1):
        record = requests[ordinal - 1] if ordinal <= len(requests) else None
        if record is not None and record.get('status') == 'completed' and _finite(record.get('energy_hartree')):
            incumbent = record['energy_hartree'] if incumbent is None else min(incumbent, record['energy_hartree'])
        curve.append({'logical_requests': ordinal, 'best_energy_hartree': incumbent,
                      'actual_request': record is not None,
                      'carried_after_stop': record is None and incumbent is not None})
    hf = requests[0]['energy_hartree'] if requests and requests[0].get('status') == 'completed' \
        and _finite(requests[0].get('energy_hartree')) else None
    verified_energy = verification['energy_hartree'] if verified else None
    return {'schema_version': SCHEMA, 'job_id': method + '__' + geometry_id,
        'method': method, 'geometry_id': geometry_id, 'identity': deepcopy(identity),
        'ledger_head': snapshot.get('head_sha256'), 'status': status,
        'failure': deepcopy(failure), 'integrity_failures': issues,
        'recorded_failure_reasons': [r.get('failure_reason') for r in requests + attempts if r.get('failure_reason')],
        'counts': counts, **{k: counts[k] for k in ('logical_requests', 'completion_attempts', 'cache_hits', 'duplicate_requests')},
        'endpoint_budget': BUDGET, 'unused_allowance': BUDGET - len(requests),
        'budget_condition_complete': bool(budget_complete), 'best_request': best['ordinal'] if best else None,
        'best_query': deepcopy(best['query']) if best else None,
        'best_energy_hartree': best['energy_hartree'] if best else None,
        'best_verified_energy_hartree': verified_energy,
        'best_verified_error_hartree': error if not issues else None,
        'hf_energy_hartree': hf, 'hf_improvement_hartree': hf - verified_energy if hf is not None and verified else None,
        'reference_energy_hartree': resolved['energy_hartree'] if resolved else None,
        'reference_status': reference_status, 'verification_status': verification_status,
        'incumbent_verified': bool(verified), 'primary_endpoint_available': endpoint_available,
        'incumbent_carried_to_budget_1024': best is not None and len(requests) < BUDGET,
        'budget_curve': curve, 'negative_outcomes_retained': True, 'performance_threshold_applied': False,
        'numerical_cost': {'search': _cost(requests), 'generation': _cost(attempts, generation=True),
            'winner_verification': _cost([{'status': 'completed', 'evaluation': verification}]) if verification else None,
            'shared_target_reference_overhead': _cost([{'status': 'completed', 'evaluation': reference}]) if reference else None,
            'scope': 'recorded marginal request/generation costs; reference is shared target overhead, not per-method marginal cost; model setup/controller costs supplied separately'},
        'artifact_authentication': 'caller_pins_ledger_reference_verification_and_final_freeze_bytes'}


def aggregate_reports(reports, *, require_complete=True, resources=None, costs=None):
    """Materialize all fixed jobs; a failed target never creates a partial mean."""
    require(type(reports) is dict, 'Reports must be keyed by fixed job ID')
    expected = [method + '__' + alias for method in METHODS for alias in TARGETS]
    require(set(reports) <= set(expected), 'Unexpected main-matrix job ID')
    require(not require_complete or set(reports) == set(expected), 'Sealing requires exactly all 16 job reports')
    rows, missing = {}, []
    for method in METHODS:
        for alias in TARGETS:
            job = method + '__' + alias
            report = reports.get(job)
            if report is None:
                require(not require_complete, 'Sealing requires 16 actual reports, not missing placeholders')
                missing.append(job)
                rows[job] = {'job_id': job, 'method': method, 'geometry_id': alias,
                             'status': 'missing', 'primary_endpoint_available': False,
                             'best_verified_error_hartree': None}
                continue
            require(type(report) is dict and report.get('schema_version') == SCHEMA
                    and report.get('job_id') == job and report.get('method') == method
                    and report.get('geometry_id') == alias and report.get('endpoint_budget') == BUDGET,
                    'Report identity or fixed endpoint differs')
            require(report.get('status') in ('complete', 'failed', 'incomplete'), 'Missing trajectory outcome status')
            require(not require_complete or report['status'] != 'incomplete',
                    'Sealing requires terminal outcomes for all 16 trajectories')
            if report.get('primary_endpoint_available') is True:
                difference = (_number(report.get('best_verified_energy_hartree'), 'Verified incumbent')
                              - _number(report.get('reference_energy_hartree'), 'Resolved target reference'))
                require(report.get('incumbent_verified') is True and not report.get('integrity_failures')
                        and report['status'] in ('complete', 'failed')
                        and difference == _number(report.get('best_verified_error_hartree'), 'Verified error')
                        and difference >= -ENERGY_TOLERANCE,
                        'Primary endpoint does not match its verified signed error')
            rows[job] = deepcopy(report)
    methods = {}
    for method in METHODS:
        selected = {alias: rows[method + '__' + alias] for alias in TARGETS}
        available = all(r.get('primary_endpoint_available') is True and r['status'] == 'complete'
                        for r in selected.values())
        values = {alias: row.get('best_verified_error_hartree') if row.get('primary_endpoint_available') else None
                  for alias, row in selected.items()}
        mean = _number(values[TARGETS[0]] / 2 + values[TARGETS[1]] / 2, 'Two-target mean') if available else None
        methods[method] = {'target_errors_hartree': values, 'primary_mean_available': available,
            'mean_verified_error_hartree': mean,
            'target_status': {alias: row['status'] for alias, row in selected.items()},
            'failed_targets': [alias for alias, row in selected.items() if row['status'] == 'failed'],
            'missing_endpoint_targets': [alias for alias, value in values.items() if value is None]}
    comparisons = {}
    for name, (left, right) in PAIRS.items():
        differences, statuses = {}, {}
        for alias in TARGETS:
            a, b = methods[left]['target_errors_hartree'][alias], methods[right]['target_errors_hartree'][alias]
            differences[alias] = _number(a - b, 'Paired error difference') if a is not None and b is not None else None
            left_status = rows[left + '__' + alias]['status']
            right_status = rows[right + '__' + alias]['status']
            statuses[alias] = {'left_status': left_status, 'right_status': right_status,
                'includes_failed_incumbent': differences[alias] is not None
                    and 'failed' in (left_status, right_status)}
        mean = _number(differences[TARGETS[0]] / 2 + differences[TARGETS[1]] / 2, 'Paired two-target mean') \
            if methods[left]['primary_mean_available'] and methods[right]['primary_mean_available'] else None
        comparisons[name] = {'left_method': left, 'right_method': right,
            'target_error_differences_hartree': differences, 'target_status': statuses,
            'mean_difference_hartree': mean,
            'failed_target_note': 'failed verified incumbents retained per target; excluded from primary paired mean',
            'sign': 'negative means the left method has lower verified energy error'}
    return {'schema_version': 'section9-fixed-matrix-report-v1', 'jobs': rows,
        'all_jobs_accounted': not missing, 'missing_jobs': missing,
        'terminal_jobs': sum(row['status'] in ('complete', 'failed') for row in rows.values()),
        'failed_jobs': [job for job, row in rows.items() if row['status'] == 'failed'],
        'all_endpoints_verified': all(row.get('primary_endpoint_available') is True for row in rows.values()),
        'methods': methods, 'paired_comparisons': comparisons,
        'resources': deepcopy(resources), 'costs': deepcopy(costs),
        'failure_exclusion_applied': False,
        'primary_mean_rule': 'both target trajectories complete and independently verified; no success-only partial mean',
        'interpretation': 'descriptive fixed-two-target one-seed study; no seed confidence intervals or significance tests'}

