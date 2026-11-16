"""Training-only representation and compact VQE gate; never model training.

Preparation authenticates integrals and tokenizers without candidate energies.
The immutable policy and preparation certificate precede all pilot requests.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .protocol import BENCHMARK_ID


DEFAULT_SECTION5 = 'large_qubit/validation/section5_v1'
POLICY_PATH = 'large_qubit/section5_policy.json'
CATALOGUE_PATH = 'large_qubit/catalogues/co2_screened_v1.json'
PREPARATION_PATH = 'large_qubit/evidence/sections_1_5_v1/preparation.json'
SCREENING_PATH = f'large_qubit/scoring_only/{BENCHMARK_ID}/section5_preparation_v1/screening.json'
TRAINING = ('CO2_T01', 'CO2_T02', 'CO2_T03')
VQE_PARAMETERS = 24
RESTART_REQUESTS = 1 + 6 * VQE_PARAMETERS
GEOMETRY_REQUESTS = 1 + 12 + 2 * RESTART_REQUESTS
ORDINARY_REQUESTS = len(TRAINING) * GEOMETRY_REQUESTS
TOTAL_PILOT_REQUESTS = 36 + ORDINARY_REQUESTS
REMAINING_PILOT_REQUESTS = 1024 - TOTAL_PILOT_REQUESTS
SOURCES = tuple('large_qubit/' + name + '.py' for name in
                ('catalogue', 'prompts', 'grammar', 'baselines', 'vqe', 'section5'))


def build_section5_policy():
    return {
        'schema_version': 'section5-policy-v1', 'benchmark_id': BENCHMARK_ID,
        'training_geometries': list(TRAINING), 'pipeline_seed': 22,
        'catalogue_templates': 64, 'compact_vqe_parameters': VQE_PARAMETERS,
        'screening': 'training_integrals_only_mean_abs_HF_coupling_over_diagonal_Fock_gap_plus_all_wire_cover',
        'catalogue_path': CATALOGUE_PATH, 'lengths': [32, 64],
        'angles': [-.1, -.05, -.025, -.0125, .0125, .025, .05, .1],
        'context_limit': 1024, 'complete_answer_truncation': False,
        'expressivity': {
            'samples_per_geometry_length': 6, 'scales': [.5, .75, 1., 1.25, 1.5, 2.],
            'template_order': 'mandatory cover then descending aggregate training score; sample-seeded order shuffle after selection',
            'angle_recipe': '-2*scale*whole_qubit_HF_derivative/gap; nearest existing nonzero discrete angle; ties smaller angle index',
            'minimum_best_improvement_ha': .001,
            'minimum_distinct_improving_circuits': 2, 'improving_threshold_ha': .0001,
            'minimum_dynamically_active_spatial_orbitals': 8,
            'occupation_change_threshold': 1e-6,
            'structural_coverage': 'all30wires_in_catalogue_and_union_of_each_length_pilot',
            'selection': 'smallest length passing every training geometry; neither passing means NO-GO before corpus',
            'claim': 'bounded_engineering_reachability_not_chemical_accuracy_or_uniform_tail_coverage',
        },
        'vqe': {
            'name': 'screened_compact_fermionic_excitation_VQE', 'full_UCCSD': False,
            'optimizer': 'five_point_fourier_coordinate_minimization_with_sixth_independent_energy_check',
            'generator_eigenfrequencies': [0., .5, 1.],
            'coordinate_order': 'ascending compact template index, repeated deterministic sweeps',
            'proposal_angle_bounds': [-.25, .25],
            'sampling_offsets_radians': [-math.pi, -math.pi / 2, 0., math.pi / 2, math.pi],
            'sample_angles_may_exceed_proposal_bounds': True,
            'incumbent': 'all charged actual energies, including coordinate probes; no discarded search opportunities',
            'restarts': ['HF_zero', 'fixed_training_mean_perturbative_clipped_0.1'],
            'pilot_requests_per_restart': RESTART_REQUESTS, 'pilot_sweeps_per_restart': 1,
            'main_post_HF_restart_allocation': [511, 512],
            'main_post_HF_budget': 1023, 'unused_partial_coordinate_requests_are_not_invented': True,
            'minimum_HF_improvement_ha': .001,
            'minimum_fraction_of_best_discrete_improvement': .5,
            'maximum_fourier_identity_error_ha': 1e-8,
            'initialization_scope': 'same fixed training-derived vector at all geometries; no source-state warm start',
            'convergence_claim': False,
            'failure': 'NO-GO; any larger conventional allowance must be explicitly amended before final target execution',
        },
        'budget': {
            'pilot_ceiling': 1024, 'section4_already_charged': 36,
            'section5_ordinary_requests': ORDINARY_REQUESTS, 'HF_requests': 3,
            'discrete_requests': 36, 'VQE_requests': 2 * RESTART_REQUESTS * len(TRAINING),
            'remaining_after_completed_section5': REMAINING_PILOT_REQUESTS,
            'duplicates_cache_hits_initializations_and_all_five_samples_charged': True,
            'separate_final_integrity_actions': 9,
        },
        'cpu_threads': 4, 'max_wall_seconds': 14400.,
        'memory_limit_bytes': 5 * 1024 ** 3,
        'normalization_tolerance': 1e-10, 'energy_tolerance_ha': 1e-8,
        'random': 'independent uniform draws over identical screened IDs including identity, fixed selected length',
        'retrieval': 'nearest fitting geometry; equal distance lower R; source energy,circuit tuple,row_id; max1023 after HF; no target ordering',
        'reference_access': False, 'heldout_energy_access': False,
        'model_training_executed': False, 'execution_ready': False,
        'claim_limits': ['different_whole_qubit_and_fermionic_families',
                         'no_optimizer_specific_advantage_claim', 'no_amortized_transfer_advantage_claim'],
    }


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n')


def _source_bindings(root):
    return {name: sha256_file(workspace_path(root, name, namespace='large_qubit')) for name in SOURCES}


def load_training_integrals(workspace):
    import numpy as np
    from .physics import verify_physics
    root = Path(workspace).resolve(strict=True)
    verify_physics(root)
    saved = read_json(root / 'large_qubit/physics/section3_v3/manifest.json')
    result = {}
    for alias in TRAINING:
        entry = saved['geometries'][alias]
        require(entry['partition'] == 'training', 'Screening cannot use held-out physics')
        with np.load(workspace_path(root, entry['arrays_path'], namespace='large_qubit/scoring_only'),
                     allow_pickle=False) as arrays:
            result[alias] = {'geometry_id': alias, 'partition': 'training',
                'hamiltonian_fingerprint': entry['hamiltonian_fingerprint'],
                'h1': arrays['one_electron_integrals'].copy(), 'eri': arrays['eri_chemists'].copy()}
    return result


def prepare_section5(workspace):
    """Write pre-pilot analytic catalogue and native-tokenizer certificates once."""
    from .catalogue import build_screened_catalogue
    from .prompts import load_prompt_tokenizer, build_prompt
    root = Path(workspace).resolve(strict=True)
    for name in (CATALOGUE_PATH, SCREENING_PATH, PREPARATION_PATH, POLICY_PATH):
        require(not workspace_path(root, name).exists(), 'Preserve existing Section5 preparation; do not overwrite')
    policy = build_section5_policy()
    catalogue, screening = build_screened_catalogue(load_training_integrals(root))
    _write(root / POLICY_PATH, policy)
    _write(root / CATALOGUE_PATH, catalogue)
    _write(root / SCREENING_PATH, screening)
    certificates = []
    for model in ('gpt2_xl', 'qwen3_4b'):
        tokenizer = load_prompt_tokenizer(root, model, execute=True)
        for alias in ('CO2_T01', 'CO2_T02', 'CO2_T03', 'CO2_V01', 'CO2_I01', 'CO2_I02'):
            for length in (32, 64):
                certificates.append(build_prompt(tokenizer, model, alias, catalogue, length)['context_certificate'])
    report = {'schema_version': 'section5-preparation-v1', 'all_passed': True,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'catalogue_sha256': catalogue['sha256'],
        'catalogue_file_sha256': sha256_file(root / CATALOGUE_PATH),
        'screening_file_sha256': sha256_file(root / SCREENING_PATH),
        'policy_sha256': digest(policy), 'context_certificates': certificates,
        'reused_grammar_sha256': sha256_file(root / 'phase6/grammar.py'),
        'source_bindings': _source_bindings(root), 'candidate_energy_requests': 0,
        'model_weights_loaded': False, 'model_training_executed': False, 'downloads_executed': 0,
        'heldout_geometry_structure_only_for_token_lengths': True,
        'heldout_energy_or_reference_access': False}
    _write(root / PREPARATION_PATH, report)
    return report


def verify_preparation(workspace, *, numerical_integrity=False):
    from .catalogue import build_screened_catalogue, validate_catalogue
    root = Path(workspace).resolve(strict=True)
    policy = read_json(root / POLICY_PATH)
    require(canonical_bytes(policy) == canonical_bytes(build_section5_policy()), 'Section5 policy differs')
    prepared = read_json(root / PREPARATION_PATH)
    catalogue = validate_catalogue(read_json(root / CATALOGUE_PATH))
    screening = read_json(root / SCREENING_PATH)
    require(prepared['source_bindings'] == _source_bindings(root), 'Section5 prepared source differs')
    require(prepared['all_passed'] is True and prepared['policy_sha256'] == digest(policy)
            and prepared['catalogue_sha256'] == catalogue['sha256']
            and prepared['catalogue_file_sha256'] == sha256_file(root / CATALOGUE_PATH)
            and prepared['screening_file_sha256'] == sha256_file(root / SCREENING_PATH), 'Section5 preparation identities differ')
    from .models import MODEL_SPECS, PINNED_LOCK_SHA256, inspect_assets
    from .prompts import PROMPT_VERSION, BOUNDARY_VERSION
    require(prepared['reused_grammar_sha256'] == sha256_file(root / 'phase6/grammar.py')
            and prepared['candidate_energy_requests'] == 0
            and prepared['model_weights_loaded'] is False and prepared['model_training_executed'] is False
            and prepared['downloads_executed'] == 0 and prepared['heldout_energy_or_reference_access'] is False,
            'Preparation scope or reused grammar source differs')
    expected = {(model, alias, length) for model in MODEL_SPECS
                for alias in (*TRAINING, 'CO2_V01', 'CO2_I01', 'CO2_I02') for length in (32, 64)}
    rows = prepared['context_certificates']
    require(type(rows) is list and len(rows) == 24, 'Exactly24 context certificates required')
    seen = set()
    metadata = {model: inspect_assets(root, model, full_hash=False) for model in MODEL_SPECS}
    for row in rows:
        identity = (row['model_key'], row['geometry_id'], row['length'])
        require(identity in expected and identity not in seen, 'Context identities missing, duplicated or outside declared grid')
        seen.add(identity)
        model, _, length = identity
        reserve = 138 if length == 32 else 266
        require(type(row['prompt_tokens']) is int and row['prompt_tokens'] > 0
                and type(row['complete_answer_bound_including_EOS']) is int
                and row['complete_answer_bound_including_EOS'] == reserve
                and type(row['total_upper_bound']) is int
                and row['total_upper_bound'] == row['prompt_tokens'] + reserve <= 1024
                and row['context_limit'] == 1024
                and row['all_legal_answer_token_paths_covered'] is True
                and row['longest_character_fallback_path_verified'] is True
                and row['catalogue_sha256'] == catalogue['sha256']
                and row['model_revision'] == MODEL_SPECS[model]['revision']
                and row['model_lock_sha256'] == PINNED_LOCK_SHA256
                and row['native_chat_template_sha256'] == metadata[model]['native_chat_template_sha256']
                and row['template_kwargs'] == MODEL_SPECS[model]['template_kwargs']
                and row['prompt_version'] == PROMPT_VERSION and row['boundary'] == BOUNDARY_VERSION
                and row['energies_or_examples_in_prompt'] is False and row['model_weights_loaded'] is False
                and row['downloads_executed'] == 0, 'Complete worst-case context or native tokenizer identity failed')
    if numerical_integrity:
        # Deterministically rederive screening from authenticated training integrals.
        rebuilt, private = build_screened_catalogue(load_training_integrals(root))
        require(canonical_bytes(rebuilt) == canonical_bytes(catalogue)
                and canonical_bytes(private) == canonical_bytes(screening), 'Screening is not the declared analytic rule')
    return policy, catalogue, screening, prepared


def discrete_proposals(catalogue, screening, alias, length):
    """Fixed local-guided engineering proposals; no candidate energies consumed."""
    import random
    from .protocol import seed_for
    require(alias in TRAINING and length in (32, 64), 'Pilot uses training geometries and lengths32/64 only')
    main = set(screening['main_source_indices'])
    source_to_template = {row['source_index']: row['template_id'] for row in catalogue['templates']}
    cover = screening['coverage_selection_order']
    sources = cover + [source for source in screening['ranking_source_indices'] if source in main and source not in cover]
    sources = sources[:length]
    require(len(sources) == length, 'Insufficient distinct screened templates')
    metrics = screening['metrics_by_geometry'][alias]
    result = []
    for sample, scale in enumerate(build_section5_policy()['expressivity']['scales']):
        ordered = list(sources)
        seed = seed_for('generation', group=f'section5:{alias}:{length}:{sample}')
        random.Random(seed).shuffle(ordered)
        ops = []
        for source in ordered:
            row = metrics[source]
            target = -2 * scale * row['whole_qubit_hf_derivative'] / row['gap']
            angle = min(range(8), key=lambda k: (abs(catalogue['angles'][k] - target), k))
            ops.append(1 + 8 * source_to_template[source] + angle)
        result.append({'sample': sample, 'scale': scale, 'seed': seed, 'ops': ops})
    return result


def state_participation(state, basis, threshold=1e-6):
    import numpy as np
    probability = np.abs(np.asarray(state).reshape(basis.shape)) ** 2
    alpha, beta = probability.sum(axis=1), probability.sum(axis=0)
    occupations = []
    for orbital in range(basis.norb):
        occupations.extend([float(alpha @ ((basis.alpha_strings >> orbital) & 1)),
                            float(beta @ ((basis.beta_strings >> orbital) & 1))])
    delta = [abs(value - (1. if wire < 22 else 0.)) for wire, value in enumerate(occupations)]
    active = [wire for wire, value in enumerate(delta) if value > threshold]
    return {'wire_occupations': occupations, 'HF_occupation_changes': delta,
        'active_wires': active, 'active_spatial_orbitals': sorted({wire // 2 for wire in active}),
        'effectively_fixed_wires': sorted(set(range(30)) - set(active)), 'threshold': threshold,
        'participation_is_not_a_chemical_accuracy_claim': True}


class PilotLedger:
    def __init__(self, root, path, policy, catalogue, evaluator, alias, *, budget, deadline):
        self.root, self.policy, self.catalogue, self.evaluator, self.alias = root, policy, catalogue, evaluator, alias
        self.budget, self.deadline, self.count = budget, deadline, 0
        self.handle = Path(path).open('x', encoding='ascii', newline='\n')
        self.first_seen = {}
        self.peak_bytes = 0

    def _append(self, value):
        self.handle.write(canonical_bytes(value).decode('ascii') + '\n')
        self.handle.flush()

    def energy(self, *, category, gates, family, recipe):
        import numpy as np
        from .benchmark import _peak_rss_bytes
        require(self.count < self.budget and time.perf_counter() < self.deadline,
                'Declared pilot request/wall budget exhausted; preserve this attempt')
        self.count += 1
        key = digest({'geometry': self.alias, 'family': family, 'gates': gates})
        started = time.perf_counter()
        self._append({'event': 'request_started', 'ordinal': self.count, 'geometry_id': self.alias,
            'category': category, 'family': family, 'recipe': recipe, 'gates': gates,
            'query_sha256': key, 'duplicate_of': self.first_seen.get(key)})
        self.first_seen.setdefault(key, self.count)
        basis = self.evaluator.basis
        state = basis.apply_circuit(basis.hf_state(), [(row['wires'], row['theta']) for row in gates], family=family)
        preparation = time.perf_counter() - started
        energy = self.evaluator.energy_sector(state)
        peak = _peak_rss_bytes(); self.peak_bytes = max(self.peak_bytes, peak)
        require(peak <= self.policy['memory_limit_bytes'], 'Pilot process exceeds memory allowance')
        diagnostics = basis.diagnostics(state)
        self._append({'event': 'request_completed', 'ordinal': self.count, 'geometry_id': self.alias,
            'query_sha256': key, 'energy_hartree': energy, 'norm': diagnostics,
            'state_preparation_seconds': preparation, 'wall_seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak})
        if self.count % 20 == 0:
            print(f'{self.alias}: {self.count}/{self.budget} charged pilot requests', flush=True)
        del state
        return float(energy)

    def close(self):
        self.handle.close()


def _discrete_gates(ops, catalogue):
    from .catalogue import circuit_gates
    return [{'wires': list(wires), 'theta': theta} for wires, theta in circuit_gates(ops, catalogue)]


def _vqe_gates(parameters, catalogue):
    return [{'wires': catalogue['templates'][index]['wires'], 'theta': float(theta)}
            for index, theta in zip(catalogue['vqe_template_indices'], parameters)]


def _geometry_pilot(root, alias, folder, policy, catalogue, screening, deadline):
    import numpy as np
    from .section4 import load_co2_evaluator
    from .vqe import optimize_compact
    evaluator, metadata = load_co2_evaluator(root, alias, threads=policy['cpu_threads'])
    ledger = PilotLedger(root, folder / f'{alias}.jsonl', policy, catalogue, evaluator, alias,
                         budget=GEOMETRY_REQUESTS, deadline=deadline)
    try:
        hf = ledger.energy(category='HF', gates=[], family='whole_qubit', recipe={'HF': True})
        hf_discrepancy = abs(hf - metadata['scf']['total_energy_ha'])
        require(hf_discrepancy <= 1e-8, 'Training-geometry evaluator HF differs from its saved SCF realization')
        discrete, private_discrete = {}, {}
        for length in (32, 64):
            rows = []
            for plan in discrete_proposals(catalogue, screening, alias, length):
                value = ledger.energy(category='expressivity', gates=_discrete_gates(plan['ops'], catalogue),
                    family='whole_qubit', recipe={'length': length, **plan})
                rows.append({**plan, 'energy_hartree': value})
            best = min(rows, key=lambda row: (row['energy_hartree'], tuple(row['ops']), row['sample']))
            state = evaluator.basis.apply_circuit(evaluator.basis.hf_state(),
                [(g['wires'], g['theta']) for g in _discrete_gates(best['ops'], catalogue)], family='whole_qubit')
            participation = state_participation(state, evaluator.basis)
            covered = sorted({wire for row in rows for g in _discrete_gates(row['ops'], catalogue) for wire in g['wires']})
            improving = {tuple(row['ops']) for row in rows if hf - row['energy_hartree'] >= .0001}
            passed = (hf - best['energy_hartree'] >= .001 and len(improving) >= 2
                      and len(participation['active_spatial_orbitals']) >= 8 and covered == list(range(30)))
            discrete[str(length)] = {'passed': passed, 'best_HF_improvement_ha': hf - best['energy_hartree'],
                'distinct_improving_circuits': len(improving), 'circuits_evaluated': len(rows),
                'covered_wires_union': covered, 'best_state_participation': participation}
            private_discrete[str(length)] = {'rows': rows, 'winner': best}
            del state
        vqe_runs = []
        for restart, initial in zip(policy['vqe']['restarts'],
                                    ([0.] * VQE_PARAMETERS, screening['vqe_initial_angles'])):
            def evaluate(parameters, step):
                return ledger.energy(category='VQE', gates=_vqe_gates(parameters, catalogue), family='fermionic',
                    recipe={'restart': restart, 'parameters': parameters, 'optimizer_step': step})
            run = optimize_compact(evaluate, initial, max_requests=RESTART_REQUESTS,
                bounds=tuple(policy['vqe']['proposal_angle_bounds']))
            run['restart'] = restart
            vqe_runs.append(run)
        best_vqe = min(vqe_runs, key=lambda row: (row['energy'], tuple(row['parameters']), row['restart']))
        state = evaluator.basis.apply_circuit(evaluator.basis.hf_state(),
            [(g['wires'], g['theta']) for g in _vqe_gates(best_vqe['parameters'], catalogue)], family='fermionic')
        vqe_participation = state_participation(state, evaluator.basis)
        discrete_best_improvement = max(row['best_HF_improvement_ha'] for row in discrete.values())
        vqe_improvement = hf - best_vqe['energy']
        criterion = max(.001, .5 * max(0., discrete_best_improvement))
        vqe = {'passed': vqe_improvement >= criterion, 'best_HF_improvement_ha': vqe_improvement,
            'required_improvement_ha': criterion, 'restarts': 2, 'parameters': VQE_PARAMETERS,
            'ordinary_queries': sum(run['requests'] for run in vqe_runs),
            'maximum_Fourier_identity_error_ha': max(row['interpolation_error'] for run in vqe_runs for row in run['history']),
            'state_participation': vqe_participation, 'convergence_claimed': False}
        require(ledger.count == GEOMETRY_REQUESTS, 'Geometry pilot did not retain every declared request')
        private = {'geometry_id': alias, 'hf_energy_hartree': hf, 'discrete': private_discrete,
            'vqe_runs': vqe_runs, 'vqe_winner': best_vqe, 'hamiltonian_fingerprint': metadata['hamiltonian_fingerprint'],
            'HF_energy_discrepancy_ha': hf_discrepancy}
        private['participation'] = {'discrete_32': discrete['32']['best_state_participation'],
                                    'discrete_64': discrete['64']['best_state_participation'],
                                    'VQE': vqe_participation}
        _write(folder / f'{alias}_outcomes.json', private)
        # Integrity checks are not admitted to search or used for choosing a new candidate.
        checks = []
        before = evaluator.statistics()['apply_h_calls']
        for label, gates, family, expected in [
            *[(f'discrete_{length}', _discrete_gates(private_discrete[str(length)]['winner']['ops'], catalogue),
               'whole_qubit', private_discrete[str(length)]['winner']['energy_hartree']) for length in (32, 64)],
            ('VQE', _vqe_gates(best_vqe['parameters'], catalogue), 'fermionic', best_vqe['energy'])]:
            vector = evaluator.basis.apply_circuit(evaluator.basis.hf_state(),
                [(g['wires'], g['theta']) for g in gates], family=family)
            measured = evaluator.energy_sector(vector)
            error = abs(measured - expected)
            require(error <= 1e-8, 'Saved winner energy integrity failed')
            checks.append({'winner': label, 'absolute_energy_discrepancy_ha': error,
                           'normalization': evaluator.basis.diagnostics(vector)})
        return {'geometry_id': alias, 'discrete': discrete, 'vqe': vqe,
            'HF_energy_discrepancy_ha': hf_discrepancy,
            'ordinary_requests': ledger.count, 'peak_rss_bytes': ledger.peak_bytes,
            'hamiltonian_fingerprint': metadata['hamiltonian_fingerprint'],
            'saved_winner_integrity': checks,
            'separate_integrity_hamiltonian_actions': evaluator.statistics()['apply_h_calls'] - before,
            'total_evaluator_statistics': evaluator.statistics()}
    finally:
        ledger.close()


def decision(results):
    require(set(results) == set(TRAINING), 'Every training geometry must be represented')
    viable = [length for length in (32, 64) if all(results[alias]['discrete'][str(length)]['passed'] for alias in TRAINING)]
    vqe_passed = all(results[alias]['vqe']['passed'] for alias in TRAINING)
    return {'status': 'go' if viable and vqe_passed else 'no_go',
        'selected_length': min(viable) if viable else None, 'viable_lengths': viable,
        'compact_vqe_passed': vqe_passed, 'representation_passed': bool(viable),
        'corpus_acquisition_allowed_by_this_engineering_gate': bool(viable and vqe_passed),
        'final_experiment_execution_ready': False, 'chemical_accuracy_claimed': False}


def run_section5(workspace, output=DEFAULT_SECTION5):
    from .section4 import verify_section4, _reject_public_energy_values
    root = Path(workspace).resolve(strict=True)
    public = workspace_path(root, output, namespace='large_qubit/validation')
    require(public != root / 'large_qubit/validation', 'Use a versioned Section5 directory')
    suffix = output.removeprefix('large_qubit/validation/')
    private_relative = f'large_qubit/scoring_only/{BENCHMARK_ID}/section5/{suffix}'
    private = workspace_path(root, private_relative, namespace='large_qubit/scoring_only')
    require(not public.exists() and not private.exists(), 'Preserve existing Section5 attempt')
    prior = root / f'large_qubit/scoring_only/{BENCHMARK_ID}/section5'
    prior_charges = 0
    for ledger_path in prior.glob('*/*.jsonl'):
        for line in ledger_path.read_text(encoding='ascii').splitlines():
            if json.loads(line).get('event') == 'request_started':
                prior_charges += 1
    require(prior_charges == 0,
            'Prior Section5 requests require a reviewed budget amendment; a new output name does not reset the pilot allowance')
    verify_section4(root)
    policy, catalogue, screening, preparation = verify_preparation(root, numerical_integrity=True)
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    started = time.perf_counter()
    try:
        results = {}
        for alias in TRAINING:
            print(f'Section5 training-only pilot: {alias}', flush=True)
            results[alias] = _geometry_pilot(root, alias, private, policy, catalogue, screening,
                                           started + policy['max_wall_seconds'])
        gate = decision(results)
        require(sum(row['ordinary_requests'] for row in results.values()) == ORDINARY_REQUESTS,
                'Section5 request totals differ')
        files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(private.rglob('*')) if p.is_file()}
        manifest = {'schema_version': 'section5-validation-v1', 'benchmark_id': BENCHMARK_ID,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'output': output,
            'restricted_output': private_relative, 'policy': policy, 'policy_sha256': digest(policy),
            'catalogue_sha256': catalogue['sha256'], 'catalogue_file_sha256': sha256_file(root / CATALOGUE_PATH),
            'preparation_file_sha256': sha256_file(root / PREPARATION_PATH),
            'source_bindings': _source_bindings(root),
            'section4_manifest_sha256': sha256_file(root / 'large_qubit/validation/section4_v1/manifest.json'),
            'results': results, 'decision': gate, 'ordinary_requests': ORDINARY_REQUESTS,
            'pilot_total_including_section4': TOTAL_PILOT_REQUESTS,
            'remaining_pilot_allowance': REMAINING_PILOT_REQUESTS,
            'separate_integrity_hamiltonian_actions': 9, 'correctness_passed': True,
            'section_5_complete': gate['status'] == 'go', 'feasibility_gate': gate['status'],
            'model_training_executed': False, 'heldout_energy_or_reference_access': False,
            'execution_ready': False, 'wall_seconds': time.perf_counter() - started, 'files': files}
        _reject_public_energy_values(manifest)
        _write(public / 'manifest.json', manifest)
        with (public / 'manifest.sha256').open('x', encoding='ascii', newline='\n') as handle:
            handle.write(sha256_file(public / 'manifest.json') + '\n')
        return verify_section5(root, section5_root=output)
    except Exception as error:
        _write(public / 'failure.json', {'all_passed': False, 'error_type': type(error).__name__,
            'error': str(error), 'wall_seconds': time.perf_counter() - started,
            'partial_files_are_retained': True, 'model_training_executed': False})
        raise


def verify_section5(workspace, section5_root=DEFAULT_SECTION5, *, numerical_integrity=False):
    """Derive the scientific decision from charged requests and saved responses.

    Optimizer replay consumes the saved ledger without Hamiltonian calls.  It
    uses NumPy for the five-point fit; default verification loads no PySCF,
    model, state vector or reference.  Optional numerical integrity prepares
    the nine winning states to recheck occupations, again without H actions.
    """
    from .section4 import verify_section4, _reject_public_energy_values, read_json_line
    from .vqe import optimize_compact

    def finite(value, label, minimum=None, maximum=None):
        require(type(value) in (int, float) and math.isfinite(value), f'Invalid finite {label}')
        require(minimum is None or value >= minimum, f'{label} is below its bound')
        require(maximum is None or value <= maximum, f'{label} exceeds its bound')
        return float(value)

    def integer(value, expected, label):
        require(type(value) is int and value == expected, f'{label} count/type differs')

    def same(actual, expected, label, tolerance=1e-8):
        if type(expected) is dict:
            require(type(actual) is dict and set(actual) == set(expected), f'{label} fields differ')
            for key in expected:
                same(actual[key], expected[key], f'{label}.{key}', tolerance)
        elif type(expected) is list:
            require(type(actual) is list and len(actual) == len(expected), f'{label} length differs')
            for index, (left, right) in enumerate(zip(actual, expected)):
                same(left, right, f'{label}[{index}]', tolerance)
        elif type(expected) is float:
            finite(actual, label)
            require(abs(actual - expected) <= tolerance, f'{label} numerical value differs')
        else:
            require(type(actual) is type(expected) and actual == expected, f'{label} value/type differs')

    def all_finite(value):
        if type(value) is dict:
            for child in value.values():
                all_finite(child)
        elif type(value) is list:
            for child in value:
                all_finite(child)
        elif type(value) is float:
            require(math.isfinite(value), 'Nonfinite saved numeric field')

    def normalization(value):
        require(type(value) is dict and set(value) == {
            'state_norm', 'norm_error', 'n_alpha', 'n_beta', 'sector_dimension',
            'sector_leakage_probability', 'population_conservation'}, 'Normalization receipt fields differ')
        norm = finite(value['state_norm'], 'state norm', 0.)
        error = finite(value['norm_error'], 'normalization error', 0., 1e-10)
        require(abs(abs(norm - 1.) - error) <= 1e-12, 'Normalization error is not derived from the saved norm')
        finite(value['sector_leakage_probability'], 'sector leakage', 0., 1e-10)
        for key, expected in (('n_alpha', 11), ('n_beta', 11), ('sector_dimension', 1863225)):
            integer(value[key], expected, key)
        require(value['population_conservation'] == 'by_fixed_basis_construction', 'Population certificate differs')

    def participation(value):
        require(type(value) is dict and set(value) == {
            'wire_occupations', 'HF_occupation_changes', 'active_wires', 'active_spatial_orbitals',
            'effectively_fixed_wires', 'threshold', 'participation_is_not_a_chemical_accuracy_claim'},
            'Private participation receipt fields differ')
        occupations = value['wire_occupations']
        require(type(occupations) is list and len(occupations) == 30, 'Participation must describe all 30 wires')
        for occupation in occupations:
            finite(occupation, 'wire occupation', -1e-10, 1. + 1e-10)
        require(abs(sum(occupations[::2]) - 11) <= 1e-8
                and abs(sum(occupations[1::2]) - 11) <= 1e-8, 'Participation populations differ')
        threshold = policy['expressivity']['occupation_change_threshold']
        same(value['threshold'], threshold, 'participation threshold', 0.)
        delta = [abs(occupation - (1. if wire < 22 else 0.)) for wire, occupation in enumerate(occupations)]
        active = [wire for wire, change in enumerate(delta) if change > threshold]
        expected = {'wire_occupations': occupations, 'HF_occupation_changes': delta,
            'active_wires': active, 'active_spatial_orbitals': sorted({wire // 2 for wire in active}),
            'effectively_fixed_wires': sorted(set(range(30)) - set(active)), 'threshold': threshold,
            'participation_is_not_a_chemical_accuracy_claim': True}
        same(value, expected, 'derived participation', 1e-10)
        return expected

    require(type(numerical_integrity) is bool, 'numerical_integrity must be a boolean')
    root = Path(workspace).resolve(strict=True)
    folder = workspace_path(root, section5_root, namespace='large_qubit/validation')
    require(folder != root / 'large_qubit/validation', 'Use a versioned Section5 directory')
    require(not (folder / 'failure.json').exists(), 'Preserved Section5 failure is not a passed gate')
    path = workspace_path(root, f'{section5_root}/manifest.json', namespace='large_qubit/validation')
    sidecar = workspace_path(root, f'{section5_root}/manifest.sha256', namespace='large_qubit/validation')
    require(sidecar.read_text(encoding='ascii').strip() == sha256_file(path), 'Section5 sidecar differs')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
            == {'manifest.json', 'manifest.sha256'}, 'Unexpected public Section5 files')
    saved = read_json(path)
    all_finite(saved)
    _reject_public_energy_values(saved)
    policy, catalogue, screening, preparation = verify_preparation(root, numerical_integrity=numerical_integrity)
    verify_section4(root)
    parameter_count = policy['compact_vqe_parameters']
    restart_requests = policy['vqe']['pilot_requests_per_restart']
    integer(len(catalogue['vqe_template_indices']), parameter_count, 'compact catalogue parameters')
    integer(len(screening['vqe_initial_angles']), parameter_count, 'fixed training initial angles')
    geometry_requests = 1 + 2 * policy['expressivity']['samples_per_geometry_length'] + 2 * restart_requests
    total_requests = len(TRAINING) * geometry_requests
    pilot_total = policy['budget']['section4_already_charged'] + total_requests
    pilot_remaining = policy['budget']['pilot_ceiling'] - pilot_total
    require(set(saved) == {'schema_version', 'benchmark_id', 'created_utc', 'output',
        'restricted_output', 'policy', 'policy_sha256', 'catalogue_sha256', 'catalogue_file_sha256',
        'preparation_file_sha256', 'source_bindings', 'section4_manifest_sha256', 'results', 'decision',
        'ordinary_requests', 'pilot_total_including_section4', 'remaining_pilot_allowance',
        'separate_integrity_hamiltonian_actions', 'correctness_passed', 'section_5_complete',
        'feasibility_gate', 'model_training_executed', 'heldout_energy_or_reference_access',
        'execution_ready', 'wall_seconds', 'files'}, 'Section5 manifest fields differ')
    require(saved['schema_version'] == 'section5-validation-v1' and saved['benchmark_id'] == BENCHMARK_ID
            and saved['output'] == section5_root and saved['execution_ready'] is False
            and saved['correctness_passed'] is True and saved['model_training_executed'] is False
            and saved['heldout_energy_or_reference_access'] is False, 'Section5 scope differs')
    require(type(saved['created_utc']) is str, 'Section5 creation timestamp must be UTC text')
    created = datetime.fromisoformat(saved['created_utc'])
    require(created.utcoffset() is not None and created.utcoffset().total_seconds() == 0,
            'Section5 creation timestamp must include UTC offset')
    require(saved['policy_sha256'] == digest(policy) and canonical_bytes(saved['policy']) == canonical_bytes(policy)
            and saved['source_bindings'] == _source_bindings(root)
            and saved['catalogue_sha256'] == catalogue['sha256']
            and saved['catalogue_file_sha256'] == sha256_file(root / CATALOGUE_PATH)
            and saved['preparation_file_sha256'] == sha256_file(root / PREPARATION_PATH)
            and saved['section4_manifest_sha256'] == sha256_file(root / 'large_qubit/validation/section4_v1/manifest.json'),
            'Section5 source/preparation/physics identity differs')
    private_relative = f'large_qubit/scoring_only/{BENCHMARK_ID}/section5/' + section5_root.removeprefix('large_qubit/validation/')
    require(saved['restricted_output'] == private_relative, 'Section5 private namespace differs')
    private = workspace_path(root, private_relative, namespace='large_qubit/scoring_only')
    expected_files = {f'{private_relative}/{alias}{suffix}' for alias in TRAINING
                      for suffix in ('.jsonl', '_outcomes.json')}
    actual = {p.relative_to(root).as_posix(): sha256_file(workspace_path(
        root, p.relative_to(root).as_posix(), namespace=private_relative))
        for p in sorted(private.rglob('*')) if p.is_file()}
    require(set(actual) == expected_files and saved['files'] == actual,
            'Section5 private file inventory/checksums differ')
    require(type(saved['results']) is dict and set(saved['results']) == set(TRAINING), 'Training result set differs')
    derived_results, preparation_replays, total_recorded_wall = {}, 0, 0.
    for alias in TRAINING:
        rows = [read_json_line(line) for line in (private / f'{alias}.jsonl').read_text(encoding='ascii').splitlines()]
        require(len(rows) == 2 * geometry_requests, 'Every issued and completed request must be retained')
        seen, previous_energy, starts, values, peaks = {}, {}, [], [], []
        for ordinal in range(1, geometry_requests + 1):
            start, end = rows[2 * (ordinal - 1):2 * ordinal]
            all_finite(start); all_finite(end)
            require(set(start) == {'event', 'ordinal', 'geometry_id', 'category', 'family',
                'recipe', 'gates', 'query_sha256', 'duplicate_of'} and set(end) == {
                'event', 'ordinal', 'geometry_id', 'query_sha256', 'energy_hartree', 'norm',
                'state_preparation_seconds', 'wall_seconds', 'peak_rss_bytes'}, 'Raw request fields differ')
            integer(start['ordinal'], ordinal, 'issued ordinal'); integer(end['ordinal'], ordinal, 'completed ordinal')
            require(start['event'] == 'request_started' and end['event'] == 'request_completed'
                    and start['geometry_id'] == end['geometry_id'] == alias, 'Raw request order differs')
            key = digest({'geometry': alias, 'family': start['family'], 'gates': start['gates']})
            require(start['query_sha256'] == end['query_sha256'] == key,
                    'Raw circuit identity differs')
            same(start['duplicate_of'], seen.get(key), 'duplicate ordinal')
            seen.setdefault(key, ordinal)
            value = finite(end['energy_hartree'], 'saved energy')
            if key in previous_energy:
                require(abs(value - previous_energy[key]) <= policy['energy_tolerance_ha'],
                        'Identical physical queries have inconsistent saved energies')
            previous_energy.setdefault(key, value)
            normalization(end['norm'])
            preparation_seconds = finite(end['state_preparation_seconds'], 'state preparation seconds', 0.)
            wall = finite(end['wall_seconds'], 'raw request wall seconds', 0.)
            require(preparation_seconds <= wall + 1e-9, 'Preparation time exceeds request wall time')
            require(type(end['peak_rss_bytes']) is int and 0 < end['peak_rss_bytes'] <= policy['memory_limit_bytes'],
                    'Raw memory certificate exceeds allowance')
            total_recorded_wall += wall
            peaks.append(end['peak_rss_bytes']); starts.append(start); values.append(value)

        def recipe(index, category, family, gates, expected_recipe):
            start = starts[index]
            require(start['category'] == category and start['family'] == family, 'Raw query family/category differs')
            same(start['gates'], gates, 'charged gate recipe')
            same(start['recipe'], expected_recipe, 'charged request recipe')

        recipe(0, 'HF', 'whole_qubit', [], {'HF': True})
        hf = values[0]
        outcome = read_json(private / f'{alias}_outcomes.json')
        all_finite(outcome)
        require(set(outcome) == {'geometry_id', 'hf_energy_hartree', 'discrete', 'vqe_runs',
                'vqe_winner', 'hamiltonian_fingerprint', 'participation',
                'HF_energy_discrepancy_ha'}, 'Private outcome fields differ')
        require(outcome['geometry_id'] == alias
                and outcome['hamiltonian_fingerprint'] == catalogue['training_hamiltonian_bindings'][alias],
                'Outcome physical identity differs')
        same(outcome['hf_energy_hartree'], hf, 'HF charged energy', 1e-12)
        finite(outcome['HF_energy_discrepancy_ha'], 'HF reference determinant discrepancy', 0., 1e-8)
        require(set(outcome['discrete']) == {'32', '64'}
                and set(outcome['participation']) == {'discrete_32', 'discrete_64', 'VQE'},
                'Private representation/participation set differs')
        public = saved['results'][alias]
        require(set(public) == {'geometry_id', 'discrete', 'vqe', 'ordinary_requests', 'peak_rss_bytes',
                'hamiltonian_fingerprint', 'saved_winner_integrity',
                'separate_integrity_hamiltonian_actions', 'total_evaluator_statistics',
                'HF_energy_discrepancy_ha'}, 'Public geometry fields differ')
        require(public['geometry_id'] == alias and public['hamiltonian_fingerprint'] == outcome['hamiltonian_fingerprint'],
                'Public geometry/physical identity differs')
        same(public['HF_energy_discrepancy_ha'], outcome['HF_energy_discrepancy_ha'], 'public HF discrepancy', 1e-12)
        derived_discrete, winners, winner_gates = {}, {}, {}
        for length, offset in ((32, 1), (64, 7)):
            planned = discrete_proposals(catalogue, screening, alias, length)
            recorded = []
            for index, plan in enumerate(planned):
                gates = _discrete_gates(plan['ops'], catalogue)
                recipe(offset + index, 'expressivity', 'whole_qubit', gates, {'length': length, **plan})
                recorded.append({**plan, 'energy_hartree': values[offset + index]})
            winner = min(recorded, key=lambda row: (row['energy_hartree'], tuple(row['ops']), row['sample']))
            same(outcome['discrete'][str(length)], {'rows': recorded, 'winner': winner}, 'private discrete outcomes', 1e-12)
            part = participation(outcome['participation'][f'discrete_{length}'])
            covered = sorted({wire for row in recorded for gate in _discrete_gates(row['ops'], catalogue)
                              for wire in gate['wires']})
            improving = {tuple(row['ops']) for row in recorded
                         if hf - row['energy_hartree'] >= policy['expressivity']['improving_threshold_ha']}
            improvement = hf - winner['energy_hartree']
            passed = (improvement >= policy['expressivity']['minimum_best_improvement_ha']
                and len(improving) >= policy['expressivity']['minimum_distinct_improving_circuits']
                and len(part['active_spatial_orbitals']) >= policy['expressivity']['minimum_dynamically_active_spatial_orbitals']
                and covered == list(range(30)))
            derived_discrete[str(length)] = {'passed': passed, 'best_HF_improvement_ha': improvement,
                'distinct_improving_circuits': len(improving), 'circuits_evaluated': len(recorded),
                'covered_wires_union': covered, 'best_state_participation': part}
            winners[f'discrete_{length}'] = winner
            winner_gates[f'discrete_{length}'] = ('whole_qubit', _discrete_gates(winner['ops'], catalogue))

        require(type(outcome['vqe_runs']) is list and len(outcome['vqe_runs']) == 2, 'Two VQE restarts are required')
        replayed_runs, cursor = [], 13
        for restart, initial in zip(policy['vqe']['restarts'], ([0.] * parameter_count, screening['vqe_initial_angles'])):
            beginning = cursor

            def consume(parameters, step):
                nonlocal cursor
                require(cursor < beginning + restart_requests and cursor < len(values), 'Optimizer exceeds its charged saved trajectory')
                recipe(cursor, 'VQE', 'fermionic', _vqe_gates(parameters, catalogue),
                       {'restart': restart, 'parameters': parameters, 'optimizer_step': step})
                value = values[cursor]
                if step['stage'] == 'initial' and all(parameter == 0. for parameter in parameters):
                    require(abs(value - hf) <= policy['energy_tolerance_ha'], 'Zero-angle VQE must reproduce its charged HF energy')
                cursor += 1
                return value

            run = optimize_compact(consume, initial, max_requests=restart_requests,
                bounds=tuple(policy['vqe']['proposal_angle_bounds']),
                interpolation_tolerance=policy['vqe']['maximum_fourier_identity_error_ha'])
            run['restart'] = restart
            integer(cursor - beginning, restart_requests, 'replayed VQE restart requests')
            integer(run['completed_sweeps'], 1, 'pilot VQE sweeps')
            integer(len(run['history']), parameter_count, 'pilot VQE coordinate updates')
            replayed_runs.append(run)
        integer(cursor, geometry_requests, 'replayed geometry requests')
        same(outcome['vqe_runs'], replayed_runs, 'deterministically replayed VQE', 1e-8)
        best = min(replayed_runs, key=lambda row: (row['energy'], tuple(row['parameters']), row['restart']))
        same(outcome['vqe_winner'], best, 'VQE incumbent from every charged probe', 1e-8)
        vqe_part = participation(outcome['participation']['VQE'])
        improvement = hf - best['energy']
        required = max(policy['vqe']['minimum_HF_improvement_ha'],
                       policy['vqe']['minimum_fraction_of_best_discrete_improvement']
                       * max(0., max(row['best_HF_improvement_ha'] for row in derived_discrete.values())))
        derived_vqe = {'passed': improvement >= required, 'best_HF_improvement_ha': improvement,
            'required_improvement_ha': required, 'restarts': 2, 'parameters': parameter_count,
            'ordinary_queries': 2 * restart_requests,
            'maximum_Fourier_identity_error_ha': max(row['interpolation_error'] for run in replayed_runs for row in run['history']),
            'state_participation': vqe_part, 'convergence_claimed': False}
        same(public['discrete'], derived_discrete, 'public discrete gate derived from private requests', 1e-10)
        same(public['vqe'], derived_vqe, 'public VQE gate derived from private requests', 1e-8)
        integer(public['ordinary_requests'], geometry_requests, 'geometry requests')
        integer(public['peak_rss_bytes'], max(peaks), 'geometry peak RSS')
        integer(public['separate_integrity_hamiltonian_actions'], 3, 'geometry integrity actions')
        checks = public['saved_winner_integrity']
        require(type(checks) is list and len(checks) == 3, 'Three independent saved-winner integrity receipts required')
        for check, label in zip(checks, ('discrete_32', 'discrete_64', 'VQE')):
            require(set(check) == {'winner', 'absolute_energy_discrepancy_ha', 'normalization'}
                    and check['winner'] == label, 'Saved winner integrity order differs')
            finite(check['absolute_energy_discrepancy_ha'], 'winner integrity discrepancy', 0., 1e-8)
            normalization(check['normalization'])
        stats = public['total_evaluator_statistics']
        require(type(stats) is dict and set(stats) == {'apply_h_calls', 'complex_actions',
            'real_actions', 'kernel_contractions', 'hamiltonian_action_seconds',
            'initialization_seconds', 'integral_and_link_bytes', 'reference_diagonal_bytes',
            'complex128_state_bytes', 'float64_state_bytes', 'dense_sector_matrix_bytes_allocated',
            'real_path_validation'}, 'Evaluator accounting receipt fields differ')
        for key, expected in (('apply_h_calls', geometry_requests + 3),
                              ('complex_actions', geometry_requests + 3), ('real_actions', 0),
                              ('kernel_contractions', geometry_requests + 3),
                              ('reference_diagonal_bytes', 0), ('dense_sector_matrix_bytes_allocated', 0)):
            integer(stats.get(key), expected, f'evaluator {key}')
        require(stats['real_path_validation'] is None, 'Pilot must retain its complex128 scoring path')
        for key in ('hamiltonian_action_seconds', 'initialization_seconds'):
            finite(stats.get(key), f'evaluator {key}', 0.)
        for key in ('integral_and_link_bytes', 'complex128_state_bytes', 'float64_state_bytes'):
            require(type(stats.get(key)) is int and stats[key] > 0, f'Invalid evaluator {key}')
        integer(stats['complex128_state_bytes'], 1863225 * 16, 'complex128 sector state bytes')
        integer(stats['float64_state_bytes'], 1863225 * 8, 'float64 sector state bytes')
        if numerical_integrity:
            from .sector import SectorBasis
            basis = SectorBasis(15, 11, 11, cache_budget_bytes=16 * 1024 * 1024)
            winner_gates['VQE'] = ('fermionic', _vqe_gates(best['parameters'], catalogue))
            for label, (family, gates) in winner_gates.items():
                state = basis.apply_circuit(basis.hf_state(), [(gate['wires'], gate['theta']) for gate in gates], family=family)
                measured = state_participation(state, basis, policy['expressivity']['occupation_change_threshold'])
                same(outcome['participation'][label], measured, 'replayed winning-state participation', 1e-8)
                preparation_replays += 1
                del state
                basis.clear_cache()
            del basis
        derived_results[alias] = {'discrete': derived_discrete, 'vqe': derived_vqe}

    for key, expected in (('ordinary_requests', total_requests), ('pilot_total_including_section4', pilot_total),
                          ('remaining_pilot_allowance', pilot_remaining), ('separate_integrity_hamiltonian_actions', 9)):
        integer(saved[key], expected, key)
    finite(saved['wall_seconds'], 'pilot wall seconds', total_recorded_wall - 1e-8,
           policy['max_wall_seconds'])
    gate = decision(derived_results)
    same(saved['decision'], gate, 'scientific decision derived from private ledger')
    require(saved['section_5_complete'] is (gate['status'] == 'go')
            and saved['feasibility_gate'] == gate['status'], 'Saved scientific decision differs')
    return {'all_passed': True, 'correctness_passed': True,
        'section_5_complete': gate['status'] == 'go', 'feasibility_gate': gate['status'],
        'manifest_sha256': sha256_file(path), 'ordinary_requests': total_requests,
        'pilot_total_including_section4': pilot_total, 'remaining_pilot_allowance': pilot_remaining,
        'selected_length': gate['selected_length'], 'payload_files_verified': len(actual),
        'VQE_replayed_from_charged_responses': True,
        'winning_state_preparations_replayed': preparation_replays,
        'additional_hamiltonian_actions': 0, 'model_training_executed': False, 'execution_ready': False}
