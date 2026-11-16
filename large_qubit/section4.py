"""Section 4 scientific gate. Importing this module executes no physics.

Reference values and circuit energies are written only to scoring_only. A saved
gate is engineering evidence, not the final experimental freeze or a VQE result.
"""

from datetime import datetime, timezone
import math
from pathlib import Path
import os
import time

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .protocol import BENCHMARK_ID


DEFAULT_SECTION4 = 'large_qubit/validation/section4_v1'
POLICY_PATH = 'large_qubit/section4_policy.json'
SOURCES = ('large_qubit/sector.py', 'large_qubit/evaluator.py', 'large_qubit/reference.py',
           'large_qubit/regression.py', 'large_qubit/benchmark.py', 'large_qubit/section4.py')
HISTORICAL_INPUTS = ('phase3/benchmark_amendment.json', 'phase3/references_development.json',
                     'phase3/validation_fixtures.json', 'phase3/catalogues/h2.json',
                     'phase3/catalogues/lih.json', 'phase3/hamiltonians/H2_DEV01.json',
                     'phase3/hamiltonians/LIH_T02.json')


def build_section4_policy():
    return {
        'schema_version': 'section4-policy-v1', 'benchmark_id': BENCHMARK_ID,
        'geometry': 'CO2_T02', 'physics_manifest': 'large_qubit/physics/section3_v3/manifest.json',
        'representation': 'fixed_Nalpha_11_Nbeta_11_sector; explicit interleaved_to_blocked_CI_phase',
        'hamiltonian': 'PySCF_2.11.0_direct_spin1_integral_contraction_no_dense_sector_matrix',
        'precision': 'complex128_candidate_states; real_float64_reference_action_cross_checked',
        'energy_agreement_ha': 1e-8, 'reference_residual_tolerance': 1e-8,
        'norm_and_population_tolerance': 1e-10,
        'reference': {'starts': 2, 'ncv': 24, 'maxiter': 500, 'tol': 1e-11, 'cpu_threads': 4,
                      'max_wall_seconds': 600.0, 'seed': 22,
                      'target': 'lowest_numerically_resolved_root_in_fixed_population_sector',
                      'no_total_spin_projection': True, 'formal_spectral_lower_bound_claim': False},
        'benchmark': {'threads_per_worker': 1, 'concurrency': [1, 2], 'samples': 2,
                      'warmups_per_family_length': 1, 'lengths': [32, 64],
                      'families': ['whole_qubit', 'fermionic'], 'gate_cache_budget_bytes': 67108864,
                      'charged_pilot_queries': 36,
                      'circuits': 'deterministic_representative_family_circuits_not_LLM_outputs_or_optimized_VQE',
                      'all_warmups_and_duplicates_charged': True},
        'schedule': {'planning_window_days': 7.0, 'protected_analysis_days': 2.0,
                     'reserved_other_implementation_training_days': 2.0,
                     'ordinary_requests': 26624, 'retry_factor': 1.25,
                     'deadline_and_full_pipeline_timing_confirmed': False,
                     'scope': 'measured_local_CPU_evaluator_allowance_under_explicit_planning_window'},
        'memory_fraction_limit': 0.75,
        'heldout_references': 'not_computed_or_used_for_this_training_geometry_gate',
        'no_model_training': True, 'no_server_connection': True, 'execution_ready': False,
    }


def validate_section4_policy(value):
    require(canonical_bytes(value) == canonical_bytes(build_section4_policy()),
            'Section 4 policy changed; create a reviewed versioned amendment')
    return value


def _write_json(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n')


def _source_bindings(root):
    return {p: sha256_file(workspace_path(root, p, namespace='large_qubit')) for p in SOURCES}


def load_co2_evaluator(workspace, alias='CO2_T02', *, threads=1):
    """Load authenticated Section 3 integrals, without references or a model.

This development loader deliberately permits training geometries only. Future
validation/test execution needs its own frozen scoring authorization.
"""
    import numpy as np
    from .definitions import geometry
    from .physics import verify_physics
    from .evaluator import IntegralEvaluator

    root = Path(workspace).resolve(strict=True)
    geometry(alias, purpose='representation')
    verification = verify_physics(root)
    manifest = read_json(root / 'large_qubit/physics/section3_v3/manifest.json')
    entry = manifest['geometries'][alias]
    metadata = read_json(workspace_path(root, entry['metadata_path'], namespace='large_qubit/scoring_only'))
    with np.load(workspace_path(root, entry['arrays_path'], namespace='large_qubit/scoring_only'),
                 allow_pickle=False) as payload:
        evaluator = IntegralEvaluator(payload['one_electron_integrals'], payload['eri_chemists'],
                                      metadata['nuclear_repulsion_ha'], 15, (11, 11), threads=threads)
    metadata['section3_manifest_sha256'] = verification['manifest_sha256']
    metadata['hamiltonian_fingerprint'] = entry['hamiltonian_fingerprint']
    metadata['orbital_fingerprint'] = entry['orbital_fingerprint']
    return evaluator, metadata


def schedule_gate(benchmark, reference_seconds, *, total_memory_bytes):
    """Use actual measured worker capacity, with an explicit retry/time reserve."""
    policy = build_section4_policy()
    scheduling = policy['schedule']
    require(type(total_memory_bytes) is int and total_memory_bytes > 0,
            'A positive measured memory byte count is required')
    _finite(reference_seconds, 'reference wall time', minimum=0)
    require(type(benchmark.get('runs')) is list
            and [run.get('workers') for run in benchmark['runs']] == [1, 2],
            'Actual serial and two-worker measurements are required in that order')
    for run in benchmark['runs']:
        require(type(run['measured_requests']) is int and run['measured_requests'] > 0,
                'Measured request counts must be positive integers')
        require(type(run['aggregate_peak_rss_bytes']) is int and run['aggregate_peak_rss_bytes'] > 0,
                'Measured RSS must be positive integer bytes')
        _finite(run['measured_wall_seconds'], 'measured wall time', minimum=0, strict=True)
    eligible = [run for run in benchmark['runs']
                if 0 < run['aggregate_peak_rss_bytes'] <= total_memory_bytes * policy['memory_fraction_limit']
                and run['measured_requests'] > 0 and run['measured_wall_seconds'] > 0]
    serial = benchmark['runs'][0]
    selected = min(eligible or [serial], key=lambda run: run['measured_wall_seconds'] / run['measured_requests'])
    seconds_per_request = selected['measured_wall_seconds'] / selected['measured_requests']
    estimate = (seconds_per_request * scheduling['ordinary_requests'] + reference_seconds) * scheduling['retry_factor']
    _finite(estimate, 'capacity schedule estimate', minimum=0)
    serial_rate = serial['measured_wall_seconds'] / serial['measured_requests']
    serial_estimate = _finite((serial_rate * scheduling['ordinary_requests'] + reference_seconds)
                              * scheduling['retry_factor'], 'serial schedule estimate', minimum=0)
    allowance = (scheduling['planning_window_days'] - scheduling['protected_analysis_days']
                 - scheduling['reserved_other_implementation_training_days']) * 86400
    peak = max(run['aggregate_peak_rss_bytes'] for run in benchmark['runs'])
    memory_passed = bool(eligible)
    passed = memory_passed and estimate <= allowance
    return {'status': 'go_local_cpu_under_declared_window' if passed else 'no_go',
            'serial_seconds_per_request': serial_rate,
            'measured_capacity_seconds_per_request': seconds_per_request,
            'selected_concurrency': selected.get('workers', selected.get('concurrency')),
            'actual_serial_seconds_per_request': serial_rate,
            'estimated_measured_capacity_evaluator_seconds_with_retry_margin': estimate,
            'estimated_serial_evaluator_seconds_with_retry_margin': serial_estimate,
            'evaluator_allowance_seconds': allowance, 'ordinary_requests': scheduling['ordinary_requests'],
            'retry_factor': scheduling['retry_factor'], 'memory_passed': memory_passed,
            'peak_measured_aggregate_rss_bytes': peak, 'total_memory_bytes': total_memory_bytes,
            'protected_analysis_days': scheduling['protected_analysis_days'],
            'reserved_other_implementation_training_days': scheduling['reserved_other_implementation_training_days'],
            'deadline_confirmed': False, 'full_pipeline_schedule_validated': False,
            'server_throughput_claimed': False,
            'scope': 'Local measured evaluator projection; future training/implementation and actual deadline remain separate'}


def _finite(value, name, *, minimum=None, strict=False):
    require(type(value) in (int, float), f'{name} must be finite numeric data')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and (minimum is None or (value > minimum if strict else value >= minimum)),
            f'{name} is nonfinite or outside its physical range')
    return value


def _verify_benchmark_evidence(root, private, benchmark, physical_entry, section3_digest):
    from .benchmark import THREAD_VARIABLES, representative_circuits

    summary = read_json(private / 'benchmark/benchmark_summary.json')
    require(canonical_bytes(summary) == canonical_bytes(benchmark), 'Saved benchmark summary differs from public report')
    policy = build_section4_policy()['benchmark']
    require(benchmark['geometry'] == 'CO2_T02'
            and benchmark['logical_requests'] == policy['charged_pilot_queries']
            and benchmark['warmup_requests'] == 12 and benchmark['measured_requests'] == 24
            and benchmark['reference_solving_included'] is False
            and benchmark['model_training_started'] is False
            and benchmark['scalar_energy_cache_used'] is False,
            'Benchmark scope or charged requests differ')
    plans = representative_circuits(samples=policy['samples'])
    raw_paths = {}
    for run in benchmark['runs']:
        workers = run['workers']
        require(run['logical_requests'] == 12 * workers and run['warmup_requests'] == 4 * workers
                and run['measured_requests'] == 8 * workers
                and run['threads_per_worker'] == 1, 'Measured worker workload differs')
        require(len(run['worker_reports']) == workers, 'Missing worker timing reports')
        for worker_id in range(workers):
            path = private / f'benchmark/concurrency_{workers}/worker_{worker_id:02d}.jsonl'
            raw_paths[path.relative_to(root).as_posix()] = sha256_file(path)
            rows = [read_json_line(line) for line in path.read_text(encoding='ascii').splitlines()]
            initialized = [r for r in rows if r.get('event') == 'worker_initialized']
            finished = [r for r in rows if r.get('event') == 'worker_finished']
            issued = [r for r in rows if r.get('event') == 'request_started']
            completed = [r for r in rows if r.get('event') == 'request_completed']
            require(len(initialized) == len(finished) == 1 and len(issued) == len(completed) == 12
                    and len(rows) == 26, 'Raw worker ledger is missing or contains unrecorded events')
            require([r['event'] for r in rows] == ['worker_initialized']
                    + ['request_started', 'request_completed'] * 12 + ['worker_finished'],
                    'Raw request frontier was reordered')
            header = initialized[0]
            require(header['worker_id'] == worker_id and header['geometry'] == 'CO2_T02' and header['threads'] == 1
                    and header['metadata']['hamiltonian_fingerprint'] == physical_entry['hamiltonian_fingerprint']
                    and header['metadata']['orbital_fingerprint'] == physical_entry['orbital_fingerprint']
                    and header['metadata']['section3_manifest_sha256'] == section3_digest,
                    'Raw worker physical or thread identity differs')
            require(header['native_thread_environment'] == {name: '1' for name in THREAD_VARIABLES},
                    'Raw native thread limits differ')
            require(finished[0]['logical_requests'] == 12, 'Raw worker request count differs')
            report = run['worker_reports'][worker_id]
            require(report['worker_id'] == worker_id and report['logical_requests'] == 12
                    and report['measured_requests'] == 8 and len(report['timings']) == 12,
                    'Worker public timing counts differ')
            seen = {}
            for ordinal, (start, end, plan, timing) in enumerate(zip(issued, completed, plans, report['timings']), 1):
                require(start['request_ordinal'] == end['request_ordinal'] == timing['request_ordinal'] == ordinal,
                        'Raw request ordinals are incomplete or repeated')
                require(all(type(r['request_ordinal']) is int for r in (start, end, timing)),
                        'Request ordinal types differ')
                for key in ('phase', 'family', 'length', 'sample', 'circuit_sha256'):
                    require(start[key] == end[key] == timing[key] == plan[key], 'Raw workload recipe differs')
                require(start['gates'] == plan['gates'] and start['energy_cache_hit'] is False
                        and start['duplicate_of'] == seen.get(plan['circuit_sha256']),
                        'Raw gate recipe or duplicate accounting differs')
                seen.setdefault(plan['circuit_sha256'], ordinal)
                _finite(end['energy_hartree'], 'restricted benchmark energy')
                _finite(end['normalization']['norm_error'], 'state norm error', minimum=0)
                _finite(end['normalization']['sector_leakage_probability'], 'population leakage', minimum=0)
                require(end['normalization']['norm_error'] <= 1e-10
                        and end['normalization']['sector_leakage_probability'] <= 1e-10,
                        'Benchmark state norm or populations failed')
                for key in ('state_preparation_seconds', 'score_seconds', 'hamiltonian_action_seconds',
                            'gate_cache_bytes', 'peak_rss_bytes'):
                    require(timing[key] == end[key], 'Public timing differs from raw request record')
                    _finite(timing[key], key, minimum=0)
                _finite(timing['logging_seconds'], 'logging time', minimum=0)
    require(raw_paths == benchmark['private_raw_log_sha256'], 'Benchmark raw-log hash bindings differ')


def read_json_line(line):
    # Reuse strict duplicate-key/nonfinite JSON validation without a temporary file.
    import json
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, 'Duplicate raw-ledger JSON key')
            value[key] = child
        return value
    def nonfinite(value):
        raise ValueError('Nonfinite raw-ledger value')
    return json.loads(line, object_pairs_hook=pairs, parse_constant=nonfinite)


def _reject_public_energy_values(value):
    if isinstance(value, dict):
        forbidden = {'energy_hartree', 'energy_ha', 'reference_energy_ha', 'hf_energy_ha',
                     'saved_reference_energy_ha', 'independent_ground_energy_ha'}
        require(not (set(value) & forbidden), 'Absolute energy values belong only in scoring_only')
        for child in value.values():
            _reject_public_energy_values(child)
    elif isinstance(value, list):
        for child in value:
            _reject_public_energy_values(child)


def _memory_total():
    path = Path('/proc/meminfo')
    if path.exists():
        return int(next(line.split()[1] for line in path.read_text().splitlines()
                        if line.startswith('MemTotal:'))) * 1024
    raise RuntimeError('Run scientific timing in Linux/WSL to record native process RAM')


def _precision_checks(evaluator, metadata):
    import numpy as np
    basis = evaluator.basis
    hf = basis.hf_state()
    hf_energy = evaluator.energy_sector(hf)
    error = abs(hf_energy - metadata['scf']['total_energy_ha'])
    require(error <= 1e-8, 'Matrix-free HF differs from the realized Hamiltonian')
    real = basis.apply_gate(hf.real.copy(), (0, 22), .13)
    ci_real = basis.to_ci(real)
    evaluator.validate_real_path(seed=22, samples=1, tolerance=1e-10)
    action_real = evaluator.apply_h_real(ci_real)
    action_complex = evaluator.apply_h(ci_real.astype(np.complex128))
    action_error = float(np.linalg.norm(action_complex - action_real))
    require(action_error <= 1e-8, 'Real reference action differs from complex128')
    candidate = real.astype(np.complex128) + 0.3j * hf
    candidate /= np.linalg.norm(candidate)
    ci = basis.to_ci(candidate)
    action = evaluator.apply_h(ci)
    value = np.vdot(ci, action)
    require(abs(value.imag) <= 1e-10, 'Complex expectation has an imaginary residual')
    energy = evaluator.energy(ci)
    require(abs(energy - value.real) <= 1e-8, 'Evaluator does not use the conjugating inner product')
    return {'all_passed': True, 'hf_energy_absolute_error_ha': error,
            'real_complex_action_norm_difference': action_error,
            'complex_expectation_imaginary_residual': abs(float(value.imag)),
            'state_diagnostics': basis.diagnostics(candidate),
            'absolute_energies_excluded_from_public_report': True,
            'correctness_hamiltonian_actions': evaluator.statistics()}


def run_section4(workspace, output=DEFAULT_SECTION4):
    """Execute bounded local correctness, reference and timing work; never train."""
    import numpy as np
    from .reference import solve_reference
    from .regression import run_historical_regressions
    from .benchmark import benchmark_evaluator

    root = Path(workspace).resolve(strict=True)
    folder = workspace_path(root, output, namespace='large_qubit/validation')
    require(folder != root / 'large_qubit/validation', 'Choose a versioned validation directory')
    suffix = output.removeprefix('large_qubit/validation/')
    private_relative = f'large_qubit/scoring_only/{BENCHMARK_ID}/section4/{suffix}'
    private = workspace_path(root, private_relative, namespace='large_qubit/scoring_only')
    require(not folder.exists() and not private.exists(), 'Preserve the existing Section 4 attempt; choose a new version')
    policy = validate_section4_policy(read_json(root / POLICY_PATH))
    sources = _source_bindings(root)
    folder.mkdir(parents=True)
    private.mkdir(parents=True)
    started = time.perf_counter()
    try:
        print('Section 4: historical H2/LiH regressions', flush=True)
        historical = run_historical_regressions(root, threads=1)
        _write_json(private / 'historical_regressions.json', historical['private'])
        require(historical['public']['all_passed'], 'Historical physics regression failed')
        print('Section 4: CO2 precision and unrestricted reference checks', flush=True)
        evaluator, metadata = load_co2_evaluator(root, threads=policy['reference']['cpu_threads'])
        correctness = _precision_checks(evaluator, metadata)
        reference_parameters = {key: policy['reference'][key] for key in
                                ('seed', 'starts', 'ncv', 'maxiter', 'tol', 'max_wall_seconds')}
        reference_parameters['residual_tolerance'] = policy['reference_residual_tolerance']
        def reference_progress(progress):
            if progress['cycle'] % 5 == 0:
                print('Reference start {start}, cycle {cycle}, residual {residual_norm_ha:.3g} Ha, elapsed {wall_seconds:.1f}s'.format(**progress), flush=True)
        reference = solve_reference(evaluator, **reference_parameters, progress_callback=reference_progress)
        reference_report = reference['report']
        require(reference_report['converged'], 'CO2 iterative reference did not converge')
        with (private / 'reference.npz').open('xb') as handle:
            np.savez(handle, state=reference['state'])
        _write_json(private / 'reference.json', {
            'geometry_id': 'CO2_T02', 'energy_hartree': reference['energy'],
            'report': reference_report, 'hamiltonian_fingerprint': metadata['hamiltonian_fingerprint'],
            'orbital_fingerprint': metadata['orbital_fingerprint'], 'populations': [11, 11],
            'nuclear_repulsion_included': True, 'reference_is_not_model_input': True})
        # Release Krylov/reference arrays before independent timing workers.
        del reference, evaluator
        print('Section 4: measured one-worker and two-worker circuit benchmark', flush=True)
        benchmark = benchmark_evaluator(root, output_directory=private / 'benchmark',
                                        workers=(1, 2), threads=1, samples=2)
        feasibility = schedule_gate(benchmark, reference_report['wall_seconds'], total_memory_bytes=_memory_total())
        files = {p.relative_to(root).as_posix(): sha256_file(p)
                 for p in sorted(private.rglob('*')) if p.is_file()}
        # Absolute reference energies remain restricted; public metadata records residuals/cost only.
        public_reference = {'converged': reference_report['converged'],
                            'residual_norm': reference_report['maximum_complex128_residual_norm_ha'],
                            'wall_seconds': reference_report['wall_seconds'],
                            'starts': reference_report['independent_starts'],
                            'hamiltonian_actions': reference_report['hamiltonian_actions']}
        manifest = {'schema_version': 'section4-validation-v1', 'benchmark_id': BENCHMARK_ID,
                    'created_utc': datetime.now(timezone.utc).isoformat(), 'output': output,
                    'restricted_output': private_relative, 'policy': policy,
                    'policy_sha256': digest(policy), 'source_bindings': sources,
                    'section3_manifest_sha256': metadata['section3_manifest_sha256'],
                    'historical_regressions': historical['public'], 'co2_correctness': correctness,
                    'reference': public_reference, 'benchmark': benchmark, 'feasibility': feasibility,
                    'correctness_passed': True, 'reference_converged': True, 'section_4_complete': True,
                    'execution_ready': False, 'training_executed': False,
                    'heldout_references_computed': False, 'files': files,
                    'total_wall_seconds': time.perf_counter() - started}
        _write_json(folder / 'manifest.json', manifest)
        with (folder / 'manifest.sha256').open('x', encoding='ascii', newline='\n') as handle:
            handle.write(sha256_file(folder / 'manifest.json') + '\n')
        return verify_section4(root, section4_root=output, numerical_integrity=True)
    except Exception as error:
        if hasattr(error, 'report'):
            _write_json(private / 'reference_failure.json', error.report)
        if getattr(error, 'state', None) is not None:
            with (private / 'partial_reference.npz').open('xb') as handle:
                np.savez(handle, state=error.state)
        _write_json(folder / 'failure.json', {'all_passed': False, 'error_type': type(error).__name__,
                    'error': str(error), 'wall_seconds': time.perf_counter() - started,
                    'training_executed': False, 'execution_ready': False})
        raise


def verify_section4(workspace, section4_root=DEFAULT_SECTION4, *, numerical_integrity=False):
    """Authenticate a saved gate; optionally recompute the reference residual."""
    from .physics import verify_physics
    root = Path(workspace).resolve(strict=True)
    folder = workspace_path(root, section4_root, namespace='large_qubit/validation')
    require(not (folder / 'failure.json').exists(), 'Section 4 attempt contains a preserved failure')
    path = folder / 'manifest.json'
    require((folder / 'manifest.sha256').read_text(encoding='ascii').strip() == sha256_file(path),
            'Section 4 manifest checksum differs')
    saved = read_json(path)
    _reject_public_energy_values(saved)
    require(saved['schema_version'] == 'section4-validation-v1' and saved['benchmark_id'] == BENCHMARK_ID,
            'Wrong Section 4 manifest identity')
    require(saved['output'] == section4_root and saved['execution_ready'] is False
            and saved['training_executed'] is False and saved['heldout_references_computed'] is False,
            'Section 4 scope differs')
    require({p.name for p in folder.iterdir() if p.is_file()} == {'manifest.json', 'manifest.sha256'},
            'Section 4 public output inventory differs')
    require(saved['policy_sha256'] == digest(validate_section4_policy(saved['policy']))
            and saved['policy'] == validate_section4_policy(read_json(root / POLICY_PATH)),
            'Section 4 policy binding differs')
    require(saved['source_bindings'] == _source_bindings(root), 'Section 4 numerical source changed')
    require(saved['section3_manifest_sha256'] == verify_physics(root)['manifest_sha256'],
            'Section 3 physical realization changed')
    suffix = section4_root.removeprefix('large_qubit/validation/')
    expected_private = f'large_qubit/scoring_only/{BENCHMARK_ID}/section4/{suffix}'
    require(saved['restricted_output'] == expected_private, 'Section 4 restricted output identity differs')
    private = workspace_path(root, saved['restricted_output'], namespace='large_qubit/scoring_only')
    actual = {p.relative_to(root).as_posix() for p in private.rglob('*') if p.is_file()}
    require(actual == set(saved['files']), 'Section 4 payload inventory differs')
    required = {f'{expected_private}/{name}' for name in ('historical_regressions.json', 'reference.json',
        'reference.npz', 'benchmark/benchmark_summary.json', 'benchmark/concurrency_1/worker_00.jsonl',
        'benchmark/concurrency_2/worker_00.jsonl', 'benchmark/concurrency_2/worker_01.jsonl')}
    require(actual == required, 'Section 4 requires all seven passed-attempt payloads')
    for relative, expected in saved['files'].items():
        candidate = workspace_path(root, relative, namespace=saved['restricted_output'])
        require(sha256_file(candidate) == expected, f'Section 4 payload changed: {relative}')
    require(saved['correctness_passed'] is True and saved['reference_converged'] is True
            and saved['reference']['residual_norm'] <= 1e-8, 'Section 4 correctness/reference gate failed')
    require(saved['section_4_complete'] is True and saved['historical_regressions']['all_passed'] is True
            and saved['co2_correctness']['all_passed'] is True, 'Section 4 aggregate pass status differs')
    historical_sources = saved['historical_regressions']['source_files_sha256']
    require(set(historical_sources) == set(HISTORICAL_INPUTS), 'Historical regression input inventory differs')
    for relative, expected in historical_sources.items():
        require(sha256_file(workspace_path(root, relative)) == expected,
                f'Historical regression input changed: {relative}')
    ref = read_json(private / 'reference.json')
    require(ref['geometry_id'] == 'CO2_T02' and ref['populations'] == [11, 11]
            and ref['nuclear_repulsion_included'] is True and ref['reference_is_not_model_input'] is True,
            'Reference access/sector metadata differs')
    _finite(ref['energy_hartree'], 'restricted reference energy')
    report = ref['report']
    require(report['all_passed'] is True and report['converged'] is True, 'Restricted reference did not pass')
    expected_reference = {'converged': report['converged'],
        'residual_norm': report['maximum_complex128_residual_norm_ha'], 'wall_seconds': report['wall_seconds'],
        'starts': report['independent_starts'], 'hamiltonian_actions': report['hamiltonian_actions']}
    require(canonical_bytes(saved['reference']) == canonical_bytes(expected_reference),
            'Public and restricted reference reports differ')
    _finite(report['maximum_complex128_residual_norm_ha'], 'reference residual', minimum=0)
    require(type(report['independent_starts']) is int and report['independent_starts'] >= 2,
            'Reference lacks independent full-sector starts')
    physical_entry = read_json(root / saved['policy']['physics_manifest'])['geometries']['CO2_T02']
    require(ref['hamiltonian_fingerprint'] == physical_entry['hamiltonian_fingerprint']
            and ref['orbital_fingerprint'] == physical_entry['orbital_fingerprint'], 'Reference physical identity differs')
    _verify_benchmark_evidence(root, private, saved['benchmark'], physical_entry, saved['section3_manifest_sha256'])
    feasibility = schedule_gate(saved['benchmark'], report['wall_seconds'],
                                total_memory_bytes=saved['benchmark']['total_memory_bytes'])
    require(canonical_bytes(saved['feasibility']) == canonical_bytes(feasibility),
            'Saved feasibility decision differs from measured capacity and memory')
    reference_residual = None
    verification_overhead = None
    if numerical_integrity:
        verification_started = time.perf_counter()
        import numpy as np
        evaluator, metadata = load_co2_evaluator(root, threads=saved['policy']['reference']['cpu_threads'])
        ref = read_json(private / 'reference.json')
        require(ref['hamiltonian_fingerprint'] == metadata['hamiltonian_fingerprint']
                and ref['orbital_fingerprint'] == metadata['orbital_fingerprint'], 'Reference physical identity differs')
        with np.load(private / 'reference.npz', allow_pickle=False) as payload:
            state = payload['state']
        require(abs(float(np.vdot(state, state).real) - 1.0) <= 1e-10, 'Reference norm differs')
        reference_residual = float(np.linalg.norm(evaluator.apply_h(state) - ref['energy_hartree'] * state))
        require(reference_residual <= 1e-8, 'Saved CO2 reference residual exceeds tolerance')
        verification_overhead = {'wall_seconds': time.perf_counter() - verification_started,
                                 'hamiltonian_actions': evaluator.statistics()['apply_h_calls'],
                                 'ordinary_circuit_energy_requests': 0,
                                 'category': 'saved_reference_integrity_verification'}
    return {'all_passed': True, 'correctness_passed': True, 'reference_converged': True,
            'section_4_complete': True, 'manifest_sha256': sha256_file(path),
            'feasibility_gate': saved['feasibility']['status'],
            'numerical_integrity_checked': bool(numerical_integrity),
            'recomputed_reference_residual': reference_residual,
            'verification_overhead': verification_overhead,
            'payload_files_verified': len(saved['files']), 'training_executed': False,
            'execution_ready': False}
