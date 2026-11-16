"""One registered infrastructure recovery of the sealed HF-only failed matrix."""
from .section9_parallel_policy import policy as parallel_policy, lane_specs, lane_spec
from .section9_policy import jobs, job_id, attempt_seed, TARGETS

REGISTRATION = 'large_qubit/registrations/sections_1_9_recovery_v1/manifest.json'
PREDECESSOR = 'large_qubit/registrations/sections_1_9_parallel_v1/manifest.json'
PREDECESSOR_SHA = 'fa162771755f41bec3a76438c1f23207fb6bdb519968834bdd97e09f5800ff45'
PREPARATION = 'large_qubit/preflights/section9_recovery_v1/preparation.json'
OUTPUT_ROOT = 'large_qubit/main/section9_recovery_v1'
MATRIX = 'matrix_0001'
FAILED_ROOT = 'large_qubit/main/section9_parallel_v1/matrix_0001'
FAILED_COMPLETION_SHA = '3fda903ac1d47352d581a3140e85d2efd693a88810edd19ac58397f40fd01bf4'
SOURCES = tuple('large_qubit/' + name for name in (
    'section9_recovery.py', 'section9_recovery_worker.py',
    'section9_recovery_policy.py', 'section9_recovery_registration.py',
    'section9_memory.py', 'section9_recovery_scoring.py',
    'section9_recovery.md', 'scripts/run_section9_recovery_tmux.sh',
    'tests/test_section9_recovery.py', 'tests/test_section9_recovery_worker.py',
    'tests/test_section9_recovery_registration.py', 'tests/test_section9_memory.py',
    'tests/test_section9_recovery_integration.py'))


def policy():
    result = parallel_policy()
    result.update({
        'schema_version': 'section9-four-gpu-infrastructure-recovery-policy-v1',
        'recovery_scope': 'one infrastructure correction before any candidate generation; no performance-based retry',
        'failed_campaign': {'path': FAILED_ROOT, 'completed_sha256': FAILED_COMPLETION_SHA,
                            'logical_requests': 16, 'completion_attempts': 0,
                            'successful_ordinary_requests': 0, 'all_first_requests_hf': True},
        'memory_guard': {'source': 'Linux /proc/self/status VmHWM and VmRSS',
                         'scope': 'current executable physics process; excludes inherited pre-exec parent high-water mark',
                         'limit_bytes': 4 * 1024 ** 3,
                         'legacy_ru_maxrss': 'record diagnostically; never apply the physics ceiling to inherited parent usage'},
        'reference_policy': 'authenticate and reuse the two completed failed-campaign references byte for byte; never solve them again',
        'gpu_identity': 'reuse the exact failed-campaign four H200 UUIDs in the original lane order; unchanged numerical environment',
        'prospective_gate': 'only the fixed sealed matrix with 16 failed initial HF requests and zero generated candidates is eligible',
        'campaign_choice': 'one additive correction of the authenticated HF-only failure; original output and registration remain immutable',
        'recovery_accounting': {
            'failed_requests_retained_as_infrastructure_overhead': 16,
            'new_requests_per_trajectory_including_hf': 1024,
            'new_maximum_ordinary_requests': 16384,
            'new_normal_ordinary_requests': 16382,
            'cumulative_maximum_ordinary_requests': 16400,
            'cumulative_normal_ordinary_requests': 16398,
            'references_charged_once': True,
            'failed_campaign_model_and_physics_setup_retained': True},
        'execution_authorization': 'explicit researcher launch after completed Section8 freeze and fixed failed-campaign authentication',
    })
    return result
