"""Prospective Section 8 protocol; importing it performs no experiments."""
from .common import digest
from .protocol import BENCHMARK_ID, METHODS, seed_for

REGISTRATION = 'large_qubit/registrations/sections_1_8_v1/manifest.json'
PREDECESSOR = 'large_qubit/registrations/sections_1_7_v2_guide_v1/manifest.json'
PREDECESSOR_SHA = 'd8f0460a13b088e7676bf6da9121f26cc13a594e9f61ae9839e6e8f36ea8ae77'
ARCHIVE = 'large_qubit/archives/sections_1_7_v2_guide_v1'
PREPARATION = 'large_qubit/preflights/section8_v1/preparation.json'
OUTPUT_ROOT = 'large_qubit/validation/section8_runs'
FINAL_FREEZE = 'large_qubit/freezes/section8_v1/freeze.json'
GEOMETRY = 'CO2_V01'
CATALOGUE = 'large_qubit/catalogues/co2_screened_v1.json'
SOURCES = tuple('large_qubit/' + name for name in (
    'section8.py', 'section8_policy.py', 'section8_registration.py',
    'section8_ledger.py', 'section8_inference.py', 'section8_scoring.py',
    'section8.md', 'scripts/run_section8_tmux.sh',
    'tests/test_section8.py', 'tests/test_section8_ledger.py',
    'tests/test_section8_inference.py', 'tests/test_section8_scoring.py'))


def policy():
    return {
        'schema_version': 'section8-validation-policy-v1', 'benchmark_id': BENCHMARK_ID,
        'geometry_id': GEOMETRY, 'partition': 'validation', 'methods': list(METHODS),
        'requests_per_method': 128, 'hf_included': True, 'max_completion_attempts': 256,
        'maximum_panel_requests': 1024, 'pipeline_seed': 22,
        'circuit_length': 32, 'checkpoint': 'final_after_five_passes',
        'temperature': .7, 'top_k': 0, 'top_p': 1., 'thinking': False,
        'inference_batch_size': 1, 'kv_cache': True, 'precision': 'bf16',
        'attention_implementation': 'eager', 'cpu_threads': 1, 'maximum_pending_scores': 2,
        'rng': 'existing method:geometry stream, SHA256 per completion ordinal; independent of queue timing',
        'validation_vqe_restart_caps': [63, 64],
        'validation_vqe_actual_requests_without_failures': 123,
        'validation_vqe_unused_requests': 5,
        'vqe_partial_coordinate': 'stop before six-query update; no invented requests or reallocation',
        'vqe_initializations': ['HF_zero', 'fixed_training_mean_perturbative_vector'],
        'vqe_convergence_required': False,
        'generation_attempt_failures': 'retain invalid outputs; no hidden replacements beyond attempt ceiling',
        'score_failure': 'charged; technical NO-GO; documented fresh panel retains every earlier cost',
        'pending_physical_request_on_resume': 'abandon and charge; refuse implicit rescore or final freeze',
        'pending_generation_on_resume': 'abandon charged attempt; continue next deterministic ordinal',
        'cache': 'exact scalar energies only; partition/physics/orbitals/catalogue/backend/precision/query key',
        'cache_hits_and_duplicates_charged': True,
        'technical_gate': ['all eight conditions accounted', 'final adapters authenticated',
            'no failed or abandoned physical requests', 'finite physical scores',
            'norm/population checks', 'independent reconstruction of eight winners',
            'validation reference converged', 'ledger integrity and bounded budgets',
            'no target access or energy feedback to generators'],
        'performance_gate': None, 'poor_performance_and_invalid_outputs_retained': True,
        'selection': 'no checkpoint, temperature, seed, method or budget selection from panel',
        'repeat_after_technical_pass': 'refused; freeze the completed panel or register a scientific redesign',
        'energy_tolerance_ha': 1e-8, 'norm_tolerance': 1e-10,
        'main': {'targets': ['CO2_I01', 'CO2_I02'], 'methods': list(METHODS),
            'requests_per_trajectory': 1024, 'max_completion_attempts': 2048,
            'trajectories': 16, 'hf_included': True, 'vqe_restart_caps': [511, 512],
            'primary_endpoint': 'best verified energy error at B=1024; each target and unweighted mean',
            'secondary': ['budget curves', 'HF improvement', 'validity', 'duplicates',
                'failure counts', 'circuit resources', 'marginal and total cost'],
            'exclusions': 'none based on performance; preserve failures and unused request allowances',
            'uncertainty': 'descriptive one-seed case study; no seed confidence intervals or significance tests',
            'target_reference_access': 'scoring only after final freeze; never generator or retrieval inputs'},
        'scientific_validation_executed_by_implementation': False,
    }


def attempt_seed(method, alias, ordinal):
    base = seed_for('generation', group=f'{method}:{alias}')
    return int(digest(['section8-attempt-v1', base, ordinal])[:16], 16) & ((1 << 63) - 1)
