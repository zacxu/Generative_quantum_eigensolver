"""Three fixed-budget CO2 fits; production actions require explicit --execute.

Preparation loads local tokenizers only. GPU preflight and fitting are separate,
researcher-run operations. This module never evaluates a circuit energy.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .protocol import FITS, MODEL_REVISIONS, training_conditions


PREPARATION = 'large_qubit/evidence/sections_1_7_gpu_uuid_v1/preparation.json'
POLICY = 'large_qubit/section7_policy.json'
CACHE_ROOT = 'large_qubit/training_data/section7_v1'
CACHE_AUDIT = 'large_qubit/evidence/sections_1_7_v1/data_validation.json'
CACHE_AUDIT_SHA256 = '4e058a5a27b45fa0369fe35c130325e06a788a710f1d7f8ec71b87758027e2ea'
TOKEN_CACHE_SHA256 = {
    'gpt2_xl': 'f249b19110fb88c4d8170e8709b5e43e03a3def972615318b6e68d6d5bff8ca7',
    'qwen3_4b': '753516ba33f211247b4ee0ed2405a2a08c94004dfc7ec16f496af38aee84d365',
}
PREFLIGHT_ROOT = 'large_qubit/preflights/section7_gpu_uuid_v1'
RUN_ROOT = 'large_qubit/training_runs/section7_gpu_uuid_v1'
RELEASE_SHA256 = '323fe5fa6d4830339aa9b14dc546364fe5d24765745f08c421e3ca562e2d3e2a'
PRIOR_REGISTRATION = 'large_qubit/registrations/sections_1_6_v2/manifest.json'
PRIOR_REGISTRATION_SHA256 = '1007e51094a01537482041ac57e07a180ece667730fbaa40d778cd9ea64a6a23'
SOURCES = (
    'large_qubit/section7.py', 'large_qubit/training.py', 'large_qubit/training_data.py',
    'large_qubit/training_cache_access.py', 'large_qubit/gpu_identity.py',
    'large_qubit/amendments/section7_h200_uuid_v1.json',
    'large_qubit/section7_policy.json', 'large_qubit/requirements-training.txt',
    'large_qubit/common.py', 'large_qubit/protocol.py', 'large_qubit/models.py',
    'large_qubit/prompts.py', 'large_qubit/grammar.py', 'large_qubit/corpus_v2.py',
    'large_qubit/catalogues/co2_screened_v1.json', 'large_qubit/physical_definition.json',
    'phase6/models.py', 'phase6/grammar.py', 'phase6/model_registry.py',
    'phase7/development_protocol.json', 'large_qubit/scripts/run_section7_tmux.sh',
)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def build_section7_policy():
    """Fixed scientific settings; only the prespecified GPU implementation is measured."""
    return {
        'schema_version': 'section7-fixed-fit-policy-v1',
        'conditions': training_conditions(),
        'release_manifest': 'large_qubit/releases/co2_corpus_v2/manifest.json',
        'release_sha256': RELEASE_SHA256,
        'fitting_rows': 6144, 'diagnostic_rows': 384,
        'passes': 5, 'effective_batch_size': 16,
        'example_presentations_per_fit': 30720, 'optimizer_steps_per_fit': 1920,
        'checkpoint_selection': 'final', 'energy_based_selection': False,
        'optimizer': {'name': 'AdamW', 'learning_rate': 0.0002,
                      'betas': [0.9, 0.999], 'epsilon': 1e-8, 'weight_decay': 0.0,
                      'schedule': 'constant', 'warmup_steps': 0,
                      'gradient_clip_norm': 1.0, 'error_if_nonfinite': True},
        'lora': {'rank': 16, 'alpha': 32, 'dropout': 0.05, 'bias': 'none',
                 'initialization': 'fresh_PEFT_default_zero_B',
                 'trainable': 'only_lora_A_and_lora_B',
                 'gpt2_target_count': 192, 'qwen3_target_count': 252,
                 'target_source': 'phase6/models.py: native all-block attention and MLP projections'},
        'precision': 'bf16', 'attention_implementation': 'eager',
        'optimizer_fused': False,
        'batch_candidates': [[4, 4], [8, 2], [16, 1], [1, 16]],
        'gradient_checkpointing_candidates': [False, True],
        'benchmark_warmup_steps_per_candidate': 1,
        'benchmark_measured_steps_per_candidate': 3,
        'benchmark_memory_probe': 'warmup_on_16_longest_complete_fitting_contexts; peaks_include_warmup',
        'maximum_available_memory_fraction': 0.8,
        'batch_selection': 'fastest_minimum_throughput_across_paired_GPT_reports; Qwen_fastest; declared_order_breaks_ties',
        'gpt_pair': 'identical_initial_adapter_row_order_batch_optimizer_exposure_except_objective_and_lambda',
        'loss': 'complete_answer_grammar_masked_sequence_NLL_including_EOS_without_length_divisor',
        'weighting': 'fitting_only_within_geometry_exponential_IQR; n_times_w_once; uniform_multiplier_one',
        'logical_window_divisor': 'actual_examples_in_effective_window; never_sum_of_weights',
        'generation_temperature_for_later_execution': 0.7,
        'qwen_enable_thinking': False,
        'accelerators': {'requested': 'three_H200', 'fits_per_gpu': 1, 'distributed_training': False,
                         'visible_gpus_per_worker': 1, 'worker_device': 'cuda:0',
                         'allocation_identity': 'distinct_GPU_UUID_masks'},
        'cache': {'root': CACHE_ROOT, 'gpt_pair_shared': True,
                  'manifest_sha256': dict(TOKEN_CACHE_SHA256),
                  'deep_native_validation': {'path': CACHE_AUDIT, 'sha256': CACHE_AUDIT_SHA256},
                  'routine_loading': 'authenticate_prevalidated_manifest_source_metadata_and_payload_hashes',
                  'stores': 'prompts_answer_token_ids_deduplicated_grammar_choices',
                  'diagnostics_accessible_to_training': False},
        'resume': {'effective_step_boundaries': True, 'checkpoint_every_steps': 96,
                   'rolling_resume_checkpoints': 2, 'final_adapter_required': True,
                   'states': ['adapter', 'optimizer', 'python_rng', 'numpy_rng',
                              'torch_cpu_rng', 'torch_cuda_rng', 'row_frontier', 'counters']},
        'preflight_costs': 'recorded_separately_with_disposable_adapters_not_hidden_in_final_fit_exposure',
        'training_control': 'researcher_runs_preflight_freeze_and_training',
        'heldout_or_reference_access': False, 'execution_ready': False,
    }


def _write_new_json(root, relative, value, namespace):
    path = workspace_path(root, relative, namespace=namespace)
    require(not path.exists(), 'Preserve existing output: ' + relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _sources(root):
    return {name: sha256_file(workspace_path(root, name)) for name in SOURCES}


def prepare_section7(workspace):
    """Freeze actual local-tokenizer caches, without loading pretrained model weights."""
    root = Path(workspace).resolve(strict=True)
    require(canonical_bytes(read_json(root / POLICY)) == canonical_bytes(build_section7_policy()),
            'Section7 policy differs from its fixed implementation')
    require(sha256_file(root / PRIOR_REGISTRATION) == PRIOR_REGISTRATION_SHA256,
            'Preserve the completed Section6 v2 registration')
    if (root / PREPARATION).exists():
        return verify_section7(root)
    from .training_data import prepare_training_data
    from .training_cache_access import verify_training_cache
    require(sha256_file(root / CACHE_AUDIT) == CACHE_AUDIT_SHA256,
            'Preserve the deep native fitting-cache audit')
    caches = {}
    for model_key in MODEL_REVISIONS:
        existing = CACHE_ROOT + '/' + model_key + '/manifest.json'
        path = workspace_path(root, existing, namespace=CACHE_ROOT)
        if path.is_file():
            fingerprint = sha256_file(path)
            require(verify_training_cache(root, existing, fingerprint,
                    expected_release_sha256=RELEASE_SHA256)['all_passed'] is True,
                    'Existing fitting cache failed authentication')
            result = {'manifest_path': existing, 'manifest_sha256': fingerprint}
        else:
            result = prepare_training_data(root, model_key=model_key, execute=True, cache_root=CACHE_ROOT,
                                           expected_release_sha256=RELEASE_SHA256)
        require(result['manifest_sha256'] == TOKEN_CACHE_SHA256[model_key],
                'Use the exact fully audited native token cache')
        caches[model_key] = {'manifest_path': result['manifest_path'],
                             'manifest_sha256': result['manifest_sha256']}
    record = {'schema_version': 'section7-preparation-v1', 'created_utc': _utc(),
              'policy_path': POLICY, 'policy_sha256': sha256_file(root / POLICY),
              'source_bindings': _sources(root), 'token_caches': caches,
              'release_sha256': RELEASE_SHA256,
              'deep_native_cache_audit': {'path': CACHE_AUDIT, 'sha256': CACHE_AUDIT_SHA256},
              'production_model_weights_loaded': False, 'production_training_executed': False,
              'production_gpu_preflight_executed': False, 'energy_requests': 0,
              'prior_registration_sha256': PRIOR_REGISTRATION_SHA256}
    _write_new_json(root, PREPARATION, record, 'large_qubit/evidence/sections_1_7_gpu_uuid_v1')
    return verify_section7(root)


def verify_section7(workspace):
    """Authenticate immutable preparation; production progress is a separate status."""
    root = Path(workspace).resolve(strict=True)
    record = read_json(workspace_path(root, PREPARATION, namespace='large_qubit/evidence/sections_1_7_gpu_uuid_v1'))
    require(record['schema_version'] == 'section7-preparation-v1', 'Unknown Section7 preparation')
    require(record['production_model_weights_loaded'] is False
            and record['production_training_executed'] is False
            and record['production_gpu_preflight_executed'] is False
            and type(record['energy_requests']) is int and record['energy_requests'] == 0
            and record['prior_registration_sha256'] == PRIOR_REGISTRATION_SHA256,
            'Prepared execution scope or historical registration identity changed')
    require(record['release_sha256'] == RELEASE_SHA256, 'Wrong shared release')
    require(sha256_file(root / CACHE_AUDIT) == CACHE_AUDIT_SHA256
            and record['deep_native_cache_audit'] == {'path': CACHE_AUDIT, 'sha256': CACHE_AUDIT_SHA256},
            'Deep native cache validation changed')
    require(record['source_bindings'] == _sources(root), 'Section7 prepared source changed')
    require(record['policy_path'] == POLICY and record['policy_sha256'] == sha256_file(root / POLICY),
            'Section7 prepared policy changed')
    require(canonical_bytes(read_json(root / POLICY)) == canonical_bytes(build_section7_policy()),
            'Section7 fixed policy changed')
    require(sha256_file(root / PRIOR_REGISTRATION) == PRIOR_REGISTRATION_SHA256,
            'Historical Section6 registration changed')
    require(set(record['token_caches']) == set(MODEL_REVISIONS), 'Both model caches are required')
    from .training_cache_access import verify_training_cache
    checks = {}
    for model_key, binding in record['token_caches'].items():
        require(binding['manifest_path'] == CACHE_ROOT + '/' + model_key + '/manifest.json',
                'Unexpected model cache path')
        require(binding['manifest_sha256'] == TOKEN_CACHE_SHA256[model_key],
                'Token cache differs from its audited preparation pin')
        checks[model_key] = verify_training_cache(root, binding['manifest_path'], binding['manifest_sha256'],
                                                 expected_release_sha256=RELEASE_SHA256)
        require(checks[model_key]['all_passed'] is True, 'A fitting cache failed authentication')
    return {'all_passed': True, 'implementation_ready': True,
            'scope': 'prepared_fixed_fit_workflow_not_completed_training_or_GPU_preflight',
            'preparation_path': PREPARATION, 'preparation_sha256': sha256_file(root / PREPARATION),
            'policy_sha256': record['policy_sha256'], 'token_caches': record['token_caches'],
            'cache_verification': checks, 'source_bindings': record['source_bindings'],
            'production_training_executed': False, 'production_gpu_preflight_executed': False,
            'production_optimizer_steps_executed': 0, 'model_weights_loaded': False,
            'energy_requests': 0, 'execution_ready': False}


def _dataset(root, condition_id, preparation):
    require(condition_id in FITS, 'Unknown fit; the revised branch has exactly three')
    condition = training_conditions()[condition_id]
    from .training_cache_access import load_training_data
    binding = preparation['token_caches'][condition['model_key']]
    dataset = load_training_data(root, condition_id=condition_id,
        manifest_path=binding['manifest_path'], expected_manifest_sha256=binding['manifest_sha256'],
        expected_release_sha256=RELEASE_SHA256)
    return condition, dataset


def _preflight_path(preflight_root, condition_id):
    require(condition_id in FITS, 'Unknown fit')
    return preflight_root + '/' + condition_id + '.json'


@contextmanager
def _operation_lock(root, identity):
    from .training import _run_lock
    require(identity and all(character.isalnum() or character in '_-' for character in identity),
            'Invalid operation lock identity')
    directory = workspace_path(root, 'large_qubit/operations/section7_gpu_uuid_v1/' + identity,
                               namespace='large_qubit/operations/section7_gpu_uuid_v1')
    with _run_lock(directory):
        yield


@contextmanager
def _gpu_operation_lock(root, device, expected_environment=None):
    from .training import environment
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    snapshot = environment(device)
    if expected_environment is not None:
        require(_stable_environment(snapshot) == expected_environment,
                'Training GPU or environment differs from measured freeze; no model has been loaded')
    with _operation_lock(root, 'gpu_' + snapshot['gpu_uuid']):
        yield


def run_preflight(workspace, condition_id, *, device='cuda:0', preflight_root=PREFLIGHT_ROOT):
    """Researcher-run disposable-adapter benchmark; never called by local verification."""
    root = Path(workspace).resolve(strict=True)
    require(condition_id in FITS, 'Unknown fit')
    with _operation_lock(root, 'fit_' + condition_id):
        return _run_preflight_locked(root, condition_id, device=device, preflight_root=preflight_root)


def _run_preflight_locked(workspace, condition_id, *, device, preflight_root):
    root = Path(workspace).resolve(strict=True)
    preparation = verify_section7(root)
    output = _preflight_path(preflight_root, condition_id)
    path = workspace_path(root, output, namespace='large_qubit/preflights')
    if path.exists():
        old = read_json(path)
        _verify_preflight(old, condition_id, preparation)
        require(old['requested_device'] == device, 'Existing preflight used another device')
        return {'state': 'already_completed', 'path': output, 'sha256': sha256_file(path)}
    condition, dataset = _dataset(root, condition_id, preparation)
    from .training import benchmark, load_runtime, numerical_configuration
    policy = build_section7_policy()
    with _gpu_operation_lock(root, device):
        runtime = load_runtime(root, condition, numerical_configuration(), device=device, execute=True)
        result = benchmark(runtime, dataset, candidates=tuple(map(tuple, policy['batch_candidates'])),
            gradient_checkpointing_options=tuple(policy['gradient_checkpointing_candidates']),
            warmup_steps=policy['benchmark_warmup_steps_per_candidate'],
            measured_steps=policy['benchmark_measured_steps_per_candidate'], execute=True)
    record = {'schema_version': 'section7-production-preflight-v1', 'created_utc': _utc(),
              'condition_id': condition_id, 'requested_device': device,
              'preparation_sha256': preparation['preparation_sha256'],
              'cache_binding': preparation['token_caches'][condition['model_key']],
              'production_gpu_preflight_executed': True, 'final_fit_training_executed': False,
              'benchmark': result}
    sha = _write_new_json(root, output, record, 'large_qubit/preflights')
    return {'state': 'completed', 'path': output, 'sha256': sha, 'benchmark': result}


def _verify_preflight(record, condition_id, preparation):
    require(record['schema_version'] == 'section7-production-preflight-v1'
            and record['condition_id'] == condition_id
            and record['preparation_sha256'] == preparation['preparation_sha256'],
            'Preflight belongs to another condition/preparation')
    model_key = training_conditions()[condition_id]['model_key']
    require(record['cache_binding'] == preparation['token_caches'][model_key], 'Wrong preflight fitting cache')
    require(record['production_gpu_preflight_executed'] is True
            and record['final_fit_training_executed'] is False, 'Preflight scope differs')
    benchmark = record['benchmark']
    require(benchmark['schema_version'] == 'section7-resident-benchmark-v1'
            and benchmark['condition_id'] == condition_id and benchmark['model_key'] == model_key
            and benchmark['base_model_resident_between_candidates'] is True
            and benchmark['adapter_optimizer_rng_restored'] is True
            and benchmark['additional_energy_requests'] == 0,
            'Measured benchmark identity or restoration differs')
    policy = build_section7_policy()
    require(benchmark['warmup_steps_per_candidate'] == policy['benchmark_warmup_steps_per_candidate']
            and benchmark['measured_steps_per_candidate'] == policy['benchmark_measured_steps_per_candidate'],
            'Preflight used another measurement exposure')
    from .training import numerical_configuration
    expected = [numerical_configuration(batch, accumulation, gradient_checkpointing=checkpointing)
                for checkpointing in policy['gradient_checkpointing_candidates']
                for batch, accumulation in policy['batch_candidates']]
    require(canonical_bytes([row['numerical_configuration'] for row in benchmark['candidates']])
            == canonical_bytes(expected), 'Preserve all prescribed candidates in their declared order')
    environment = benchmark['runtime']
    require(environment['fixture_only'] is False and environment['visible_gpu_count'] == 1
            and environment['native_bf16'] is True and 'H200' in environment['device_name'].upper()
            and environment['device'] == record['requested_device'] == 'cuda:0'
            and isinstance(environment['gpu_uuid'], str) and environment['gpu_uuid'].startswith('GPU-'),
            'Preflight requires exactly one UUID-identified visible BF16 H200')


def _stable_environment(environment):
    return {key: environment[key] for key in (
        'python', 'packages', 'torch_cuda_version', 'cudnn_version', 'compute_capability',
        'device_name', 'gpu_uuid', 'visible_gpu_count', 'native_bf16', 'CUBLAS_WORKSPACE_CONFIG')}


def _validate_candidate_measurement(row):
    require(type(row['success']) is bool and type(row['memory_headroom_passed']) is bool,
            'Invalid measured candidate status')
    if not row['success']:
        require(row['memory_headroom_passed'] is False, 'A failed candidate cannot satisfy memory headroom')
        return
    for key in ('initial_free_bytes', 'initial_reserved_bytes', 'usable_memory_budget_bytes', 'peak_reserved_bytes'):
        require(type(row[key]) is int and row[key] >= 0, 'Invalid measured memory counter')
    budget = row['initial_free_bytes'] + row['initial_reserved_bytes']
    require(budget > 0 and row['usable_memory_budget_bytes'] == budget, 'Measured memory budget differs')
    require(row['required_memory_headroom_fraction'] == 0.2
            and row['memory_headroom_passed'] is (row['peak_reserved_bytes'] <= 0.8 * budget),
            'Memory headroom flag differs from the measured allocation')
    seconds = row['measured_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            'Nonfinite measured duration')
    counts = row['measured_counts']
    require(type(counts['examples']) is int and counts['examples'] == 48
            and type(counts['input_tokens']) is int and counts['input_tokens'] > 0
            and type(counts['answer_tokens']) is int and 0 < counts['answer_tokens'] <= counts['input_tokens']
            and type(counts['forward_calls']) is int
            and counts['forward_calls'] == 48 // row['numerical_configuration']['microbatch_size'],
            'Measured work differs from three full effective windows')
    speed = row['examples_per_second']
    require(type(speed) in (int, float) and math.isfinite(speed) and speed > 0
            and math.isclose(speed, counts['examples'] / seconds, rel_tol=1e-12, abs_tol=0),
            'Measured throughput differs from actual work and time')


def select_numerical_configurations(reports):
    """Select implementation speed only, jointly for the matched GPT pair."""
    require(set(reports) == set(FITS), 'All three measured preflights are required')
    choices = {}
    for fit in FITS:
        candidates = reports[fit]['benchmark']['candidates']
        acceptable, seen = {}, set()
        for ordinal, row in enumerate(candidates):
            config = row['numerical_configuration']
            require(type(config['microbatch_size']) is int and type(config['gradient_accumulation']) is int
                    and config['microbatch_size'] > 0 and config['gradient_accumulation'] > 0
                    and config['microbatch_size'] * config['gradient_accumulation'] == 16,
                    'Preflight changed the effective batch')
            require([config['microbatch_size'], config['gradient_accumulation']]
                    in build_section7_policy()['batch_candidates'], 'Undeclared preflight batch')
            from .training import numerical_configuration
            expected = numerical_configuration(config['microbatch_size'], config['gradient_accumulation'],
                                                 gradient_checkpointing=config['gradient_checkpointing'])
            require(canonical_bytes(config) == canonical_bytes(expected), 'Unmeasured numerical settings')
            identity = canonical_bytes(config).decode()
            require(identity not in seen, 'Duplicate measured numerical configuration')
            seen.add(identity)
            _validate_candidate_measurement(row)
            if row['success'] is not True:
                continue
            speed = row['examples_per_second']
            require(isinstance(speed, (int, float)) and not isinstance(speed, bool)
                    and math.isfinite(speed) and speed > 0, 'Nonfinite preflight throughput')
            if row.get('memory_headroom_passed') is not True:
                continue
            acceptable[identity] = (config, float(speed), ordinal)
        require(acceptable, 'No measured candidate satisfies the memory headroom for ' + fit)
        choices[fit] = acceptable
    shared = set(choices[FITS[0]]) & set(choices[FITS[1]])
    require(shared, 'The GPT pair has no common measured numerical configuration')
    selected_pair = min(shared, key=lambda key: (
        -min(choices[FITS[0]][key][1], choices[FITS[1]][key][1]),
        choices[FITS[0]][key][2], key))
    selected_qwen = min(choices[FITS[2]], key=lambda key: (
        -choices[FITS[2]][key][1], choices[FITS[2]][key][2], key))
    return {FITS[0]: choices[FITS[0]][selected_pair][0],
            FITS[1]: choices[FITS[1]][selected_pair][0],
            FITS[2]: choices[FITS[2]][selected_qwen][0]}


def freeze_section7(workspace, *, preflight_root=PREFLIGHT_ROOT):
    root = Path(workspace).resolve(strict=True)
    preparation = verify_section7(root)
    output = preflight_root + '/freeze.json'
    path = workspace_path(root, output, namespace='large_qubit/preflights')
    if path.exists():
        return verify_freeze(root, preflight_root=preflight_root)
    reports, bindings = {}, {}
    for fit in FITS:
        relative = _preflight_path(preflight_root, fit)
        report_path = workspace_path(root, relative, namespace='large_qubit/preflights')
        reports[fit] = read_json(report_path)
        _verify_preflight(reports[fit], fit, preparation)
        bindings[fit] = {'path': relative, 'sha256': sha256_file(report_path)}
    devices = [reports[fit]['requested_device'] for fit in FITS]
    gpu_uuids = {fit: reports[fit]['benchmark']['runtime']['gpu_uuid'] for fit in FITS}
    require(len(set(gpu_uuids.values())) == 3, 'Assign one distinct GPU UUID to each fit')
    environments = {fit: _stable_environment(reports[fit]['benchmark']['runtime']) for fit in FITS}
    require(all(environments[fit]['packages'] == environments[FITS[0]]['packages']
                and environments[fit]['torch_cuda_version'] == environments[FITS[0]]['torch_cuda_version']
                for fit in FITS), 'Preflights used different package or CUDA environments')
    require(reports[FITS[0]]['benchmark']['restored_state_sha256'][0]
            == reports[FITS[1]]['benchmark']['restored_state_sha256'][0],
            'The GPT pair must begin from identical fresh LoRA parameters')
    selected = select_numerical_configurations(reports)
    record = {'schema_version': 'section7-numerical-freeze-v1', 'created_utc': _utc(),
              'preparation_sha256': preparation['preparation_sha256'],
              'preflight_reports': bindings, 'conditions': training_conditions(),
              'numerical_configurations': selected,
              'devices': dict(zip(FITS, devices)),
              'gpu_uuids': gpu_uuids, 'runtime_environments': environments,
              'passes': 5, 'effective_batch_size': 16,
              'optimizer_steps_per_fit': 1920, 'checkpoint_selection': 'final',
              'final_experiment_protocol_frozen': False}
    _write_new_json(root, output, record, 'large_qubit/preflights')
    return verify_freeze(root, preflight_root=preflight_root)


def verify_freeze(workspace, *, preflight_root=PREFLIGHT_ROOT):
    root = Path(workspace).resolve(strict=True)
    preparation = verify_section7(root)
    relative = preflight_root + '/freeze.json'
    path = workspace_path(root, relative, namespace='large_qubit/preflights')
    record = read_json(path)
    require(record['schema_version'] == 'section7-numerical-freeze-v1'
            and record['preparation_sha256'] == preparation['preparation_sha256']
            and record['conditions'] == training_conditions(), 'Numerical freeze changed its scope')
    reports = {}
    for fit in FITS:
        binding = record['preflight_reports'][fit]
        require(binding['path'] == _preflight_path(preflight_root, fit), 'Unexpected preflight binding')
        report_path = workspace_path(root, binding['path'], namespace='large_qubit/preflights')
        require(sha256_file(report_path) == binding['sha256'], 'Measured preflight changed after freezing')
        reports[fit] = read_json(report_path)
        _verify_preflight(reports[fit], fit, preparation)
        require(record['devices'][fit] == reports[fit]['requested_device'], 'Frozen device changed')
        environment = reports[fit]['benchmark']['runtime']
        require(record['gpu_uuids'][fit] == environment['gpu_uuid']
                and record['runtime_environments'][fit] == _stable_environment(environment),
                'Frozen GPU or numerical environment changed')
    require(len(set(record['gpu_uuids'].values())) == 3, 'Frozen GPU UUIDs must be distinct')
    require(reports[FITS[0]]['benchmark']['restored_state_sha256'][0]
            == reports[FITS[1]]['benchmark']['restored_state_sha256'][0],
            'Frozen GPT initialization differs')
    require(record['numerical_configurations'] == select_numerical_configurations(reports),
            'Frozen numerical choice differs from the prespecified measured selection')
    require(record['passes'] == 5 and record['effective_batch_size'] == 16
            and record['optimizer_steps_per_fit'] == 1920 and record['checkpoint_selection'] == 'final'
            and record['final_experiment_protocol_frozen'] is False, 'Frozen fit exposure changed')
    return {'all_passed': True, 'freeze_path': relative, 'freeze_sha256': sha256_file(path), 'freeze': record}


def run_fit(workspace, condition_id, *, device='cuda:0', preflight_root=PREFLIGHT_ROOT,
            run_root=RUN_ROOT, resume=False):
    root = Path(workspace).resolve(strict=True)
    require(condition_id in FITS, 'Unknown fit')
    with _operation_lock(root, 'fit_' + condition_id):
        return _run_fit_locked(root, condition_id, device=device, preflight_root=preflight_root,
                               run_root=run_root, resume=resume)


def _run_fit_locked(workspace, condition_id, *, device, preflight_root, run_root, resume):
    root = Path(workspace).resolve(strict=True)
    preparation = verify_section7(root)
    frozen = verify_freeze(root, preflight_root=preflight_root)
    require(condition_id in FITS, 'Unknown fit')
    freeze = frozen['freeze']
    require(device == freeze['devices'][condition_id], 'Use the device frozen for this fit')
    condition, dataset = _dataset(root, condition_id, preparation)
    directory = workspace_path(root, run_root + '/' + condition_id, namespace='large_qubit/training_runs')
    identity = {'schema_version': 'section7-training-identity-v1', 'condition': condition,
                'preparation_sha256': preparation['preparation_sha256'],
                'freeze_sha256': frozen['freeze_sha256'],
                'cache_binding': preparation['token_caches'][condition['model_key']],
                'numerical_configuration': freeze['numerical_configurations'][condition_id],
                'release_sha256': RELEASE_SHA256}
    from .training import load_runtime, train, verify_training_run
    if (directory / 'completed.json').exists():
        result = verify_training_run(directory, expected_identity=identity)
        return {'state': 'already_completed', 'verification': result}
    require(resume or not directory.exists(), 'Existing partial training requires explicit --resume')
    with _gpu_operation_lock(root, device, freeze['runtime_environments'][condition_id]):
        runtime = load_runtime(root, condition, identity['numerical_configuration'], device=device, execute=True)
        require(_stable_environment(runtime.metadata['environment_before_load'])
                == freeze['runtime_environments'][condition_id], 'Training GPU or environment differs from measured freeze')
        report = read_json(root / freeze['preflight_reports'][condition_id]['path'])
        require(runtime.metadata['initial_adapter_sha256'] == report['benchmark']['restored_state_sha256'][0],
                'Fresh training adapter differs from the measured initialization')
        return train(runtime, dataset, run_directory=directory, identity=identity, resume=resume,
                     checkpoint_every_steps=build_section7_policy()['resume']['checkpoint_every_steps'])


def status_section7(workspace, *, preflight_root=PREFLIGHT_ROOT, run_root=RUN_ROOT):
    root = Path(workspace).resolve(strict=True)
    result = {'scope': 'read_only_no_model_loading', 'implementation_prepared': (root / PREPARATION).is_file(),
              'preflights': {}, 'fits': {}, 'training_complete': False, 'execution_ready': False}
    preparation = verify_section7(root) if result['implementation_prepared'] else None
    frozen = None
    for fit in FITS:
        path = workspace_path(root, _preflight_path(preflight_root, fit), namespace='large_qubit/preflights')
        if path.is_file():
            require(preparation is not None, 'A preflight requires authenticated preparation')
            _verify_preflight(read_json(path), fit, preparation)
            result['preflights'][fit] = 'authenticated'
        else:
            result['preflights'][fit] = 'not_run'
        directory = workspace_path(root, run_root + '/' + fit, namespace='large_qubit/training_runs')
        if (directory / 'completed.json').is_file():
            from .training import verify_training_run
            require(preparation is not None, 'A completed fit requires authenticated preparation')
            frozen = frozen or verify_freeze(root, preflight_root=preflight_root)
            condition = training_conditions()[fit]
            identity = {'schema_version': 'section7-training-identity-v1', 'condition': condition,
                        'preparation_sha256': preparation['preparation_sha256'],
                        'freeze_sha256': frozen['freeze_sha256'],
                        'cache_binding': preparation['token_caches'][condition['model_key']],
                        'numerical_configuration': frozen['freeze']['numerical_configurations'][fit],
                        'release_sha256': RELEASE_SHA256}
            verification = verify_training_run(directory, expected_identity=identity)
            require(verification['counters']['optimizer_steps'] == 1920
                    and verification['counters']['examples_seen'] == 30720,
                    'A completed production fit must cover the entire registered corpus five times')
            result['fits'][fit] = {'state': 'complete', 'verification': verification}
        else:
            result['fits'][fit] = {'state': 'partial' if directory.exists() else 'not_run'}
            if (directory / 'run_binding.json').is_file():
                from .training import training_cost_summary
                result['fits'][fit]['progress_and_cost'] = training_cost_summary(directory)
    result['training_complete'] = all(row['state'] == 'complete' for row in result['fits'].values())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('plan')
    for action in ('prepare', 'preflight', 'freeze', 'train'):
        item = sub.add_parser(action)
        item.add_argument('--execute', action='store_true', help='Explicitly perform this action')
        if action in ('preflight', 'freeze', 'train'):
            item.add_argument('--preflight-root', default=PREFLIGHT_ROOT)
        if action in ('preflight', 'train'):
            item.add_argument('--condition', choices=FITS, required=True)
            item.add_argument('--device', default='cuda:0')
        if action == 'train':
            item.add_argument('--run-root', default=RUN_ROOT)
            item.add_argument('--resume', action='store_true')
    sub.add_parser('verify')
    item = sub.add_parser('status')
    item.add_argument('--preflight-root', default=PREFLIGHT_ROOT)
    item.add_argument('--run-root', default=RUN_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.action == 'verify':
            result = verify_section7(args.root)
        elif args.action == 'status':
            result = status_section7(args.root, preflight_root=args.preflight_root, run_root=args.run_root)
        elif args.action == 'plan' or not args.execute:
            require(not getattr(args, 'resume', False), 'Resume requires explicit --execute')
            result = {'mode': 'preview_only', 'requested_action': args.action,
                      'policy': build_section7_policy(), 'production_training_executed': False,
                      'production_gpu_preflight_executed': False, 'energy_requests': 0}
        elif args.action == 'prepare':
            result = prepare_section7(args.root)
        elif args.action == 'preflight':
            result = run_preflight(args.root, args.condition, device=args.device, preflight_root=args.preflight_root)
        elif args.action == 'freeze':
            result = freeze_section7(args.root, preflight_root=args.preflight_root)
        else:
            result = run_fit(args.root, args.condition, device=args.device, preflight_root=args.preflight_root,
                             run_root=args.run_root, resume=args.resume)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
