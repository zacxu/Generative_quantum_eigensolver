"""Prospective four-H200 scheduling amendment; all scientific settings inherited."""
from .section9_policy import policy as serial_policy, jobs, job_id, attempt_seed, TARGETS

REGISTRATION = 'large_qubit/registrations/sections_1_9_parallel_v1/manifest.json'
PREDECESSOR = 'large_qubit/registrations/sections_1_9_v1/manifest.json'
PREDECESSOR_SHA = '7ea4123cd837d9db69159be132a5bff94d53923b16e32999957151eb20745091'
PREPARATION = 'large_qubit/preflights/section9_parallel_v1/preparation.json'
OUTPUT_ROOT = 'large_qubit/main/section9_parallel_v1'
MATRIX = 'matrix_0001'
SOURCES = tuple('large_qubit/' + name for name in (
    'section9_parallel.py', 'section9_parallel_worker.py',
    'section9_parallel_policy.py', 'section9_parallel_registration.py',
    'section9_parallel.md', 'scripts/run_section9_parallel_tmux.sh',
    'tests/test_section9_parallel.py', 'tests/test_section9_parallel_registration.py',
    'tests/test_section9_parallel_worker.py', 'tests/test_section9_parallel_integration.py',
    'tests/test_section9_parallel_lifecycle.py'))


def lane_specs():
    """One target and one backbone per lane, with exclusive file/cache ownership."""
    gpt = ('gpt2_xl_frozen', 'gpt2_xl_weighted_lora', 'gpt2_xl_uniform_lora',
           'uniform_random', 'training_circuit_retrieval')
    qwen = ('qwen3_4b_frozen', 'qwen3_4b_weighted_lora', 'conventional_vqe')
    return tuple({'lane_id': f'lane_{slot}', 'gpu_slot': slot, 'model_key': model,
                  'target': target, 'methods': list(methods),
                  'job_ids': [job_id(method, target) for method in methods]}
                 for slot, (model, target, methods) in enumerate((
                     ('gpt2_xl', TARGETS[0], gpt), ('gpt2_xl', TARGETS[1], gpt),
                     ('qwen3_4b', TARGETS[0], qwen), ('qwen3_4b', TARGETS[1], qwen))))


def lane_spec(lane_id):
    for spec in lane_specs():
        if spec['lane_id'] == lane_id:
            return spec
    raise ValueError('Unknown four-GPU lane')


def policy():
    result = serial_policy()
    result.update({
        'schema_version': 'section9-four-gpu-policy-v1',
        'matrix': MATRIX, 'gpu_count': 4, 'lanes': list(lane_specs()),
        'job_order_scope': 'canonical reporting order; concurrent execution follows each lane methods order',
        'resident_target_workers': 4, 'cpu_threads_per_worker': 1,
        'maximum_active_physical_queues': 4, 'maximum_pending_scores': 2,
        'maximum_pending_scores_scope': 'per lane; eight globally',
        'maximum_physics_worker_rss_bytes': 4 * 1024 ** 3,
        'combined_physics_rss_allowance_bytes': 16 * 1024 ** 3,
        'parent_model_loading_memory': 'additional for four independent backbone/controller processes',
        'backbone_reuse': 'one resident backbone per lane; authenticated adapter switches',
        'gpu_identity': 'four distinct H200 UUIDs; ordered lane allocation pinned before model loading and unchanged on resume',
        'cache_partition': 'final_test',
        'scalar_cache_ownership': 'one independent cache per lane; no cross-lane scalar cache sharing',
        'cache_accounting': 'lane isolation may increase physical work; every logical request remains charged',
        'reference_policy': 'coordinator computes at most one bounded reference per target before launching lanes; completed reference is authenticated and shared',
        'winner_verification': 'once per trajectory incumbent in its owning lane; fresh evaluator and separate integrity overhead',
        'concurrency': 'four independent lane subprocesses; one CPU physics worker per lane',
        'prospective_gate': 'refuse if the original Section9 matrix has started; no second campaign or seed',
        'campaign_choice': 'use this amended executor or original Section9, never both',
        'infrastructure_interruption': 'retain all lane artifacts; healthy siblings may finish after a lane failure; withhold completion seal and require explicit resume on the same ordered four GPU UUIDs',
        'coordinator_interruption': 'terminate lane process groups and physics children; retain all accounting and require explicit resume',
        'execution_authorization': 'explicit researcher launch after authenticated Section8 freeze',
    })
    return result
