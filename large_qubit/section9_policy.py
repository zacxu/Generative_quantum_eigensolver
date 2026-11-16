"""Additive main-matrix execution policy; the scientific choices remain frozen."""
from .section8_policy import policy as validation_policy, attempt_seed
from .protocol import METHODS

REGISTRATION = 'large_qubit/registrations/sections_1_9_v1/manifest.json'
PREDECESSOR = 'large_qubit/registrations/sections_1_8_v1/manifest.json'
PREDECESSOR_SHA = '645007aae6dbdb502faa976f76112b0080a208677cec181d27ad42f1fef1ce7a'
PREPARATION = 'large_qubit/preflights/section9_v1/preparation.json'
OUTPUT_ROOT = 'large_qubit/main/section9_v1'
MATRIX = 'matrix_0001'
TARGETS = ('CO2_I01', 'CO2_I02')
SOURCES = tuple('large_qubit/' + name for name in (
    'section9.py', 'section9_policy.py', 'section9_registration.py',
    'section9_ledger.py', 'section9_scoring.py', 'section9_resources.py',
    'section9_reporting.py', 'section9.md', 'scripts/run_section9_tmux.sh',
    'tests/test_section9.py', 'tests/test_section9_ledger.py',
    'tests/test_section9_scoring.py', 'tests/test_section9_resources.py',
    'tests/test_section9_reporting.py', 'tests/test_section9_controller.py'))


def job_id(method, alias):
    if method not in METHODS or alias not in TARGETS:
        raise ValueError('Unknown main-matrix method or target')
    return method + '__' + alias


def jobs():
    return [(method, alias) for method in METHODS for alias in TARGETS]


def policy():
    frozen = validation_policy()
    return {
        'schema_version': 'section9-main-policy-v1',
        'scientific_protocol': frozen['main'], 'pipeline_seed': 22,
        'inherited_inference': {k: frozen[k] for k in (
            'circuit_length', 'checkpoint', 'temperature', 'top_k', 'top_p', 'thinking',
            'inference_batch_size', 'kv_cache', 'precision', 'attention_implementation')},
        'matrix': MATRIX, 'job_order': [job_id(m, g) for m, g in jobs()],
        'cpu_threads_per_worker': 1, 'resident_target_workers': 2,
        'maximum_active_physical_queues': 1, 'maximum_pending_scores': 2,
        'maximum_physics_worker_rss_bytes': 4 * 1024 ** 3,
        'combined_physics_rss_allowance_bytes': 8 * 1024 ** 3,
        'parent_model_loading_memory': 'additional; not included in physics allowance',
        'gpu_count': 1, 'backbone_reuse': 'method outer, target inner; one resident backbone',
        'gpu_identity': 'H200; frozen numerical stack; first main UUID pinned for every resume',
        'rng': 'unchanged Section8 per-method/target/completion-ordinal seed function',
        'cache_partition': 'final_test', 'logical_budget_includes_cache_hits_and_duplicates': True,
        'vqe_actual_requests_without_failures': 1023, 'vqe_unused_allowance': 1,
        'vqe_work': 'all energies charged; 170 algebraic coordinate fits plus root solves recorded separately',
        'vqe_algebraic_allowance': 'at most 170 coordinate fits per VQE trajectory per explicit controller execution; replay work separately logged',
        'failure_policy': 'retain charged failed/uncertain physical calls; terminate that trajectory without replacement',
        'generation_interruption': 'charge abandoned attempt; continue next ordinal on explicit resume',
        'infrastructure_interruption': 'halt controller; explicit resume, no silent relaunch or changed GPU',
        'reference_policy': 'once per target, same bounded solver; record failures and block affected target',
        'winner_verification': 'once per incumbent; fresh basis/evaluator, separate integrity overhead',
        'uncertain_overhead': 'retain intent and unknown cost; no implicit reference or verification retry',
        'completed_jobs': 'authenticate and skip', 'performance_selection': False,
        'failed_target_analysis': 'retain verified incumbent and curve; preserve Section3 rule: primary two-target mean requires both trajectories complete and verified, never a success-only mean',
        'reruns': 'no new campaign or extra seed; redesign needs a separate prospective registration',
        'circuit_resources': 'exact all-to-all h/rx/rz/cx/x decomposition; no hardware-routing claim',
        'execution_authorization': 'explicit user launch after authenticated Section8 freeze',
    }
