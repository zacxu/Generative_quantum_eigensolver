"""Explicit user-run fresh LoRA fitting and disposable resident-model timing.

Imports are lightweight. No model, optimizer, network, GPU or energy evaluator
is touched by importing this module. Production execution is BF16 on one H200;
small CPU runtime fixtures exercise the same optimizer/checkpoint machinery.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid

from .common import canonical_bytes, digest, read_json, require, sha256_file
from .gpu_identity import normalize_gpu_uuid
from .protocol import training_conditions, seed_for


OPTIMIZER = {'name': 'AdamW', 'learning_rate': .0002, 'betas': [.9, .999],
             'epsilon': 1e-8, 'weight_decay': 0., 'scheduler': None,
             'warmup_steps': 0, 'max_gradient_norm': 1., 'error_if_nonfinite': True}
LORA = {'rank': 16, 'alpha': 32, 'dropout': .05, 'bias': 'none',
        'initialization': 'PEFT default zero-B no-op', 'use_rslora': False, 'use_dora': False}


def numerical_configuration(microbatch_size=4, gradient_accumulation=4, *,
                            gradient_checkpointing=False, attention_implementation='eager', optimizer_fused=False):
    require(type(microbatch_size) is int and type(gradient_accumulation) is int
            and microbatch_size > 0 and gradient_accumulation > 0
            and microbatch_size * gradient_accumulation == 16, 'Effective batch must be exactly16')
    require(type(gradient_checkpointing) is bool and attention_implementation == 'eager'
            and optimizer_fused is False, 'Use the inspected eager backend and unfused AdamW')
    return {'microbatch_size': microbatch_size, 'gradient_accumulation': gradient_accumulation,
        'effective_batch_size': 16, 'gradient_checkpointing': gradient_checkpointing,
        'attention_implementation': 'eager', 'optimizer_fused': False,
        'precision': 'bf16', 'allow_tf32': False, 'deterministic_algorithms': True}


def _validated_numeric(value):
    expected = numerical_configuration(value['microbatch_size'], value['gradient_accumulation'],
        gradient_checkpointing=value['gradient_checkpointing'], attention_implementation=value['attention_implementation'],
        optimizer_fused=value['optimizer_fused'])
    require(value == expected, 'Resolved numerical configuration differs')
    return deepcopy(expected)


def _validated_condition(condition):
    expected = training_conditions().get(condition.get('condition_id'))
    require(expected is not None, 'Select one of the three registered fresh fits')
    for key in ('model_key', 'model_revision', 'objective', 'lambda_scale', 'passes', 'learning_rate',
                'lora_rank', 'lora_alpha', 'lora_dropout', 'precision', 'effective_batch_size',
                'adapter_initialization_seed', 'row_order_seed', 'resume_from_historical_adapter'):
        require(condition.get(key) == expected[key], 'Fresh-fit declaration differs: ' + key)
    return deepcopy(condition)


def _cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


def state_fingerprint(value):
    """Stable tensor-aware diagnostic digest, without exposing tensor contents."""
    import torch
    def descriptor(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                    'bytes_sha256': sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(item, dict):
            return [[str(key), descriptor(val)] for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))]
        if isinstance(item, (tuple, list)):
            return [descriptor(val) for val in item]
        return item
    return digest(descriptor(value))


def capture_rng():
    import numpy as np
    import torch
    kind, keys, position, gaussian, cached = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [kind, torch.tensor(keys.astype('int64')), position, gaussian, cached],
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    kind, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((kind, keys.cpu().numpy().astype('uint32'), position, gaussian, cached))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        require(torch.cuda.is_available() and len(state['torch_cuda']) == torch.cuda.device_count(), 'CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment(device='cuda:0', *, fixture=False):
    import torch
    selected = torch.device(device)
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'packages': {name: metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')},
        'torch_cuda_version': torch.version.cuda, 'device': str(selected), 'fixture_only': fixture,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if selected.type == 'cpu':
        require(fixture, 'Production fitting requires one BF16 H200')
        return result
    require(selected.type == 'cuda' and torch.cuda.is_available(), 'CUDA is unavailable')
    require(torch.cuda.device_count() == 1, 'Expose exactly one assigned H200 through CUDA_VISIBLE_DEVICES per fit')
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        properties = torch.cuda.get_device_properties(index)
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        free, total = torch.cuda.mem_get_info(index)
    require(bf16 and 'H200' in properties.name.upper(), 'Production fitting requires native BF16 on an H200')
    raw_gpu_uuid = getattr(properties, 'uuid', 'unavailable')
    try:
        gpu_uuid = normalize_gpu_uuid(raw_gpu_uuid)
    except ValueError as error:
        raise ValueError(f'An authenticated H200 GPU UUID is required for cuda:{index}; '
                         f'name={properties.name!r}, uuid={str(raw_gpu_uuid)!r}') from error
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    processes = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        require(len(fields) == 3 and fields[0].isdigit(), 'Unexpected GPU sharing snapshot format')
        if fields[1] == gpu_uuid:
            processes.append({'pid': int(fields[0]), 'gpu_uuid': fields[1],
                              'used_memory_mib': int(fields[2]) if fields[2].isdigit() else None})
    processes.sort(key=lambda row: row['pid'])
    result.update(device=f'cuda:{index}', device_name=properties.name, gpu_uuid=gpu_uuid,
        native_bf16=bf16, compute_capability=[properties.major, properties.minor], free_vram_bytes=free,
        total_vram_bytes=total, total_memory_bytes=total, visible_gpu_count=torch.cuda.device_count(), cudnn_version=torch.backends.cudnn.version(),
        process_pid=os.getpid(), gpu_compute_processes=processes,
        other_compute_processes_visible=[row for row in processes if row['pid'] != os.getpid()],
        gpu_sharing_snapshot_source='nvidia-smi query-compute-apps; PID namespace is the host report')
    return result


def _configure_model(model, enabled):
    model.config.use_cache = False
    if hasattr(model, '_require_grads_hook'):
        model.disable_input_require_grads()
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()


def _trainable(model):
    result = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(result and all('lora_A' in name or 'lora_B' in name for name in result), 'Only LoRA A/B parameters may train')
    return result


def _adapter_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in _trainable(model).items()}


def _restore_adapter(model, state):
    import torch
    parameters = _trainable(model)
    require(set(parameters) == set(state), 'Adapter parameter names changed')
    with torch.no_grad():
        for name, parameter in parameters.items():
            require(parameter.shape == state[name].shape and parameter.dtype == state[name].dtype, 'Adapter tensor shape/dtype changed')
            parameter.copy_(state[name].to(parameter.device))


def _optimizer(model):
    import torch
    return torch.optim.AdamW(list(_trainable(model).values()), lr=.0002, betas=(.9, .999), eps=1e-8,
                             weight_decay=0., foreach=False, fused=False)


@dataclass
class TrainingRuntime:
    model: object
    optimizer: object
    device: str
    condition: dict
    numerical: dict
    metadata: dict


def load_runtime(workspace, condition, numerical_config, *, device='cuda:0', execute=False):
    require(execute is True, 'Loading a training runtime requires explicit user execution')
    condition, numerical = _validated_condition(condition), _validated_numeric(numerical_config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from .models import load_base_model
    from phase6.models import attach_adapters, adapter_manifest
    require(metadata.version('peft') == '0.18.1', 'Use the pinned inspected PEFT implementation')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    before = environment(device)
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model = load_base_model(workspace, condition['model_key'], execute=True, device=device,
                            precision='bf16', local_files_only=True)
    binding = deepcopy(model._large_qubit_binding)
    _seed_all(condition['adapter_initialization_seed'])
    model = attach_adapters(model, rank=16, alpha=32, dropout=.05)
    initial = state_fingerprint(_adapter_state(model))
    adapter = adapter_manifest(model)
    _configure_model(model, numerical['gradient_checkpointing'])
    group = 'gpt2_pair' if condition['model_key'] == 'gpt2_xl' else 'qwen3_weighted'
    dropout_seed = seed_for('adapter_initialization', group=group + ':training_dropout')
    _seed_all(dropout_seed)
    optimizer = _optimizer(model)
    return TrainingRuntime(model, optimizer, before['device'], condition, numerical,
        {'execute_authorized': True, 'fixture_only': False, 'base_binding': binding,
         'adapter_manifest': adapter, 'initial_adapter_sha256': initial, 'training_rng_seed': dropout_seed,
         'optimizer': deepcopy(OPTIMIZER), 'lora': deepcopy(LORA), 'environment_before_load': before,
         'environment_after_load': environment(device), 'model_load_seconds': time.perf_counter() - began,
         'model_load_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
         'model_load_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
         'historical_adapters_loaded': False, 'downloads_executed': 0, 'energy_requests': 0})


def _sync(device):
    import torch
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _step(runtime, dataset, indices, *, optimizer=None, microbatch=None, progress_callback=None):
    import torch
    optimizer = runtime.optimizer if optimizer is None else optimizer
    microbatch = runtime.numerical['microbatch_size'] if microbatch is None else microbatch
    optimizer.zero_grad(set_to_none=True)
    objective, unweighted, input_tokens, answer_tokens, forwards = 0., 0., 0, 0, 0
    for offset in range(0, len(indices), microbatch):
        selected = indices[offset:offset + microbatch]
        batch = dataset.collate(selected, device=runtime.device)
        counts = {'microbatch_index': offset // microbatch, 'row_indices': list(selected),
                  'examples': len(selected), 'input_tokens': int(batch['attention_mask'].sum().item()),
                  'answer_tokens': sum(map(int, batch['answer_lengths']))}
        if progress_callback is not None:
            progress_callback('microbatch_started', **counts)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(runtime.device).type == 'cuda' else nullcontext()
        with context:
            logits = runtime.model(**{name: batch[name] for name in ('input_ids', 'attention_mask', 'position_ids')}).logits
            nll = dataset.sequence_nll(logits, batch)
            require(nll.ndim == 1 and len(nll) == len(selected) and torch.isfinite(nll).all().item(), 'Invalid complete-sequence NLL')
            multipliers = batch['loss_multipliers'].to(device=nll.device, dtype=torch.float32)
            require(multipliers.shape == nll.shape and torch.isfinite(multipliers).all().item()
                    and torch.all(multipliers >= 0).item(), 'Invalid fitting-only loss multipliers')
            loss = (nll.float() * multipliers).sum() / len(indices)
        loss.backward()
        if progress_callback is not None:
            progress_callback('microbatch_completed', **counts)
        objective += float(loss.detach()); unweighted += float(nll.detach().sum())
        input_tokens += counts['input_tokens']
        answer_tokens += counts['answer_tokens']; forwards += 1
    norm = torch.nn.utils.clip_grad_norm_(list(_trainable(runtime.model).values()), max_norm=1., error_if_nonfinite=True)
    if progress_callback is not None:
        progress_callback('optimizer_started')
    optimizer.step()
    if progress_callback is not None:
        progress_callback('optimizer_completed')
    return {'objective': objective, 'unweighted_sequence_nll_sum': unweighted,
            'input_tokens': input_tokens, 'answer_tokens': answer_tokens, 'forward_calls': forwards,
            'unclipped_gradient_norm': float(norm), 'examples': len(indices)}


def _atomic_json(path, value):
    path = Path(path)
    candidate = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with candidate.open('xb') as handle:
        handle.write(canonical_bytes(value) + b'\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(candidate, path)


def _inside(path, boundary):
    path, boundary = Path(path), Path(boundary).resolve(strict=True)
    require(path.resolve() == path and path.resolve().is_relative_to(boundary) and path.resolve() != boundary,
            'Checkpoint path escaped or redirected its run namespace')
    return path


def _checkpoint_manifest(folder, identity_sha):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    require((folder / 'manifest.sha256').read_text(encoding='ascii').strip() == sha256_file(folder / 'manifest.json')
            and manifest['identity_sha256'] == identity_sha, 'Checkpoint identity or sidecar changed')
    actual = {path.relative_to(folder).as_posix(): sha256_file(path) for path in folder.rglob('*')
              if path.is_file() and path.relative_to(folder).as_posix() not in ('manifest.json', 'manifest.sha256')}
    require(actual == manifest['files'], 'Checkpoint payload inventory or bytes changed')
    for relative in actual:
        _inside(folder / relative, folder)
    return manifest


def _latest_checkpoint(directory, identity_sha):
    """Recover a completed rename even if its mutable latest pointer was interrupted."""
    root = Path(directory) / 'checkpoints'
    candidates = sorted(root.glob('resume_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
    require(candidates, 'No committed resume checkpoint exists')
    for folder in candidates:
        _inside(folder, root)
        manifest = _checkpoint_manifest(folder, identity_sha)
        require(manifest['kind'] == 'resume'
                and folder.name == f"resume_{manifest['counters']['optimizer_steps']:08d}", 'Resume checkpoint name or kind differs')
    folder = candidates[-1]
    descriptor = {'path': folder.name, 'manifest_sha256': sha256_file(folder / 'manifest.json')}
    pointer = Path(directory) / 'latest.json'
    if pointer.exists():
        saved = read_json(pointer)
        previous = _inside(root / saved['path'], root)
        require(previous in candidates and sha256_file(previous / 'manifest.json') == saved['manifest_sha256'],
                'Latest checkpoint pointer changed')
    # The pointer can lag after a crash; it cannot select a noncommitted file.
    if not pointer.exists() or read_json(pointer) != descriptor:
        _atomic_json(pointer, descriptor)
    return descriptor, folder


def _save_checkpoint(runtime, directory, identity_sha, counters, *, final=False):
    import torch
    directory = Path(directory)
    root = directory / 'checkpoints'; root.mkdir(exist_ok=True)
    name = 'final' if final else f"resume_{counters['optimizer_steps']:08d}"
    target = _inside(root / name, root)
    if target.exists():
        existing = _checkpoint_manifest(target, identity_sha)
        require(existing['counters'] == counters and existing['adapter_sha256'] == state_fingerprint(_adapter_state(runtime.model)),
                'Existing checkpoint differs; preserve it')
        return {'path': name, 'manifest_sha256': sha256_file(target / 'manifest.json')}
    stage_root = root / 'incomplete'; stage_root.mkdir(exist_ok=True)
    stage = _inside(stage_root / (name + '.' + uuid.uuid4().hex + '.partial'), stage_root)
    stage.mkdir()
    rng = capture_rng()
    try:
        runtime.model.save_pretrained(stage / 'adapter', safe_serialization=True)
        state = {'identity_sha256': identity_sha, 'counters': counters,
                 'adapter': _adapter_state(runtime.model), 'rng': rng}
        if not final:
            state['optimizer'] = _cpu_tree(runtime.optimizer.state_dict())
        torch.save(state, stage / 'training_state.pt')
        for path in stage.rglob('*'):
            if path.is_file():
                with path.open('r+b') as handle:
                    os.fsync(handle.fileno())
        files = {path.relative_to(stage).as_posix(): sha256_file(path) for path in stage.rglob('*') if path.is_file()}
        manifest = {'schema_version': 'section7-training-checkpoint-v1', 'identity_sha256': identity_sha,
            'kind': 'final' if final else 'resume', 'counters': counters, 'files': files,
            'adapter_sha256': state_fingerprint(state['adapter']), 'contains_pretrained_base_weights': False}
        _atomic_json(stage / 'manifest.json', manifest)
        with (stage / 'manifest.sha256').open('x', encoding='ascii') as handle:
            handle.write(sha256_file(stage / 'manifest.json') + '\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(stage, target)
        descriptor = {'path': name, 'manifest_sha256': sha256_file(target / 'manifest.json')}
        if not final:
            _atomic_json(directory / 'latest.json', descriptor)
            complete = sorted(path for path in root.glob('resume_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]') if path.is_dir())
            for previous in complete[:-2]:
                _inside(previous, root); _checkpoint_manifest(previous, identity_sha)
                shutil.rmtree(previous)
        return descriptor
    finally:
        restore_rng(rng)


def _order(dataset, pass_index):
    order = tuple(dataset.row_order(pass_index))
    require(len(order) == len(dataset.rows) and sorted(order) == list(range(len(dataset.rows)))
            and len(order) % 16 == 0, 'Require one complete row permutation and whole effective batches')
    return order


def _validate_counters(counters, row_count, passes):
    steps, examples = counters['optimizer_steps'], counters['examples_seen']
    require(all(type(value) is int and value >= 0 for value in counters.values())
            and examples == steps * 16 and examples <= row_count * passes
            and (counters['next_pass'], counters['next_offset']) == divmod(examples, row_count),
            'Checkpoint step, exposure or next-row cursor disagrees')


_WORK_KEYS = ('window_attempts', 'forward_attempts', 'backwards_completed', 'optimizer_attempts',
              'optimizer_steps_completed', 'examples_attempted', 'input_tokens_attempted',
              'answer_tokens_attempted', 'examples_backpropagated', 'input_tokens_backpropagated',
              'answer_tokens_backpropagated')
_COST_SCOPE = ('Per-attempt wall starts at train entry and includes setup, checkpointing and durable '
               'event logging through the finish event; receipt/completed JSON publication is excluded. '
               'Authenticated base/adapter load wall is reported separately and added once per invocation. '
               'A process killed without a finish event has only a recorded wall lower bound; '
               'forward/optimizer attempts count durable intents before each call.')


def _event_cost(path, identity_sha, *, data=None, allow_uncommitted_tail=False):
    """Derive attempted work from durable intents, including an interrupted tail."""
    data = Path(path).read_bytes() if data is None else data
    complete = data
    tail_bytes = 0
    if data and not data.endswith(b'\n'):
        boundary = data.rfind(b'\n') + 1
        complete, tail_bytes = data[:boundary], len(data) - boundary
        require(allow_uncommitted_tail, 'Attempt ledger has an uncommitted trailing write')
    records = [json.loads(line) for line in complete.decode('utf-8').splitlines()]
    if not records:
        require(allow_uncommitted_tail, 'Attempt ledger has no complete start event')
        return {'work': {key: 0 for key in _WORK_KEYS}, 'finished': False, 'status': 'uncommitted_start_record',
                'wall_seconds': 0., 'wall_is_lower_bound': True, 'model_load_seconds': 0.,
                'gpu_assigned': False, 'gpu_uuid': None, 'peak_allocated_bytes': None,
                'peak_reserved_bytes': None, 'uncommitted_tail_bytes': tail_bytes,
                'peak_memory_is_lower_bound': True}
    require(records and records[0]['event'] == 'attempt_started'
            and records[0]['identity_sha256'] == identity_sha, 'Attempt event identity differs')
    start = records[0]
    work = {key: 0 for key in _WORK_KEYS}
    previous_elapsed, active, pending, optimization, finished = -1., None, None, None, False
    step = start['committed_counters']['optimizer_steps']
    for index, record in enumerate(records):
        elapsed = record['elapsed_seconds']
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= previous_elapsed,
                'Attempt wall receipt is invalid')
        previous_elapsed = elapsed
        event = record['event']
        require(not finished, 'Events follow an immutable attempt finish')
        if index == 0:
            continue
        if event == 'window_started':
            require(active is None and record['optimizer_step'] == step + 1
                    and len(record['row_indices']) == 16 and len(set(record['row_indices'])) == 16,
                    'Attempt optimizer window order differs')
            active = {'step': step + 1, 'rows': record['row_indices'], 'microbatches': 0,
                      'examples': 0, 'input_tokens': 0, 'answer_tokens': 0}
            pending, optimization = None, None
            work['window_attempts'] += 1
        elif event == 'microbatch_started':
            require(active is not None and pending is None and optimization is None
                    and record['optimizer_step'] == active['step']
                    and record['microbatch_index'] == active['microbatches'], 'Microbatch intent order differs')
            counts = {key: record[key] for key in ('examples', 'input_tokens', 'answer_tokens')}
            require(all(type(value) is int and value > 0 for value in counts.values())
                    and counts['input_tokens'] >= counts['answer_tokens'] >= counts['examples']
                    and record['row_indices'] == active['rows'][active['examples']:active['examples'] + counts['examples']]
                    and counts['examples'] == len(record['row_indices']), 'Microbatch token or row receipt differs')
            pending = {key: record[key] for key in ('microbatch_index', 'row_indices', 'examples', 'input_tokens', 'answer_tokens')}
            work['forward_attempts'] += 1
            for key, value in counts.items():
                work[key + '_attempted'] += value
        elif event == 'microbatch_completed':
            require(pending is not None and record['optimizer_step'] == active['step']
                    and pending == {key: record[key] for key in pending}, 'Microbatch completion lacks its intent')
            work['backwards_completed'] += 1
            for key in ('examples', 'input_tokens', 'answer_tokens'):
                work[key + '_backpropagated'] += pending[key]
                active[key] += pending[key]
            active['microbatches'] += 1
            pending = None
        elif event == 'optimizer_started':
            require(active is not None and pending is None and active['examples'] == 16 and optimization is None
                    and record['optimizer_step'] == active['step'], 'Optimizer intent lacks a complete accumulation window')
            optimization = 'started'; work['optimizer_attempts'] += 1
        elif event == 'optimizer_completed':
            require(active is not None and optimization == 'started' and record['optimizer_step'] == active['step'],
                    'Optimizer completion lacks its intent')
            optimization = 'completed'; work['optimizer_steps_completed'] += 1
        elif event == 'window_completed':
            require(active is not None and optimization == 'completed'
                    and all(record[key] == active[key] for key in ('examples', 'input_tokens', 'answer_tokens'))
                    and record['forward_calls'] == active['microbatches']
                    and record['counters']['optimizer_steps'] == active['step'], 'Completed window receipt differs')
            step, active, pending = active['step'], None, None
        elif event in ('bounded_stop', 'completed'):
            require(active is None, 'Terminal success has an unfinished optimizer window')
        elif event == 'attempt_interrupted':
            require(type(record['error_type']) is str, 'Interruption type is absent')
        elif event == 'attempt_finished':
            require(record['status'] in ('complete', 'bounded_stop', 'interrupted', 'already_complete'),
                    'Attempt finish status differs')
            finished = True
        else:
            raise ValueError('Unknown training cost event: ' + str(event))
    load_seconds = start['model_load_seconds']
    require(type(load_seconds) in (int, float) and math.isfinite(load_seconds) and load_seconds >= 0,
            'Model load wall receipt differs')
    peak_allocated = records[-1].get('peak_allocated_bytes') if finished else None
    peak_reserved = records[-1].get('peak_reserved_bytes') if finished else None
    return {'work': work, 'finished': finished and not tail_bytes, 'status': records[-1].get('status', 'unclosed_process_attempt'),
            'wall_seconds': previous_elapsed, 'wall_is_lower_bound': not finished or bool(tail_bytes),
            'model_load_seconds': load_seconds, 'gpu_assigned': start['gpu_assigned'],
            'gpu_uuid': start.get('gpu_uuid'),
            'peak_allocated_bytes': max(peak_allocated or 0, start.get('model_load_peak_allocated_bytes') or 0),
            'peak_reserved_bytes': max(peak_reserved or 0, start.get('model_load_peak_reserved_bytes') or 0),
            'peak_memory_is_lower_bound': not finished or bool(tail_bytes) or (start['gpu_assigned'] and peak_reserved is None),
            'uncommitted_tail_bytes': tail_bytes}


class _Attempt:
    def __init__(self, runtime, directory):
        self.runtime, self.directory = runtime, Path(directory).resolve()
        self.began, self.identifier = time.perf_counter(), uuid.uuid4().hex
        self.path = self.directory / ('attempt_' + self.identifier + '.jsonl')
        self.identity_sha, self.counters, self.started = None, None, False

    def event(self, kind, **fields):
        with self.path.open('ab') as handle:
            handle.write(canonical_bytes({'event': kind, 'elapsed_seconds': time.perf_counter() - self.began, **fields}) + b'\n')
            handle.flush(); os.fsync(handle.fileno())

    def start(self, identity_sha, counters, resume):
        import torch
        self.identity_sha, self.counters, self.started = identity_sha, deepcopy(counters), True
        cuda = torch.device(self.runtime.device).type == 'cuda'
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.runtime.device)
        env = self.runtime.metadata.get('environment_after_load', {})
        self.event('attempt_started', identity_sha256=identity_sha, resume=resume,
                   committed_counters=counters, model_load_seconds=self.runtime.metadata.get('model_load_seconds', 0.),
                   gpu_assigned=cuda, gpu_uuid=env.get('gpu_uuid'),
                   model_load_peak_allocated_bytes=self.runtime.metadata.get('model_load_peak_allocated_bytes'),
                   model_load_peak_reserved_bytes=self.runtime.metadata.get('model_load_peak_reserved_bytes'))

    def finish(self, status):
        if not self.started:
            return
        import torch
        cuda = torch.device(self.runtime.device).type == 'cuda'
        try:
            allocated = torch.cuda.max_memory_allocated(self.runtime.device) if cuda else None
            reserved = torch.cuda.max_memory_reserved(self.runtime.device) if cuda else None
        except Exception:
            # A lost CUDA device must not erase its durable attempted work.
            allocated, reserved = None, None
        self.event('attempt_finished', status=status, last_in_memory_counters=self.counters,
                   peak_allocated_bytes=allocated, peak_reserved_bytes=reserved)
        cost = _event_cost(self.path, self.identity_sha)
        receipt = {'schema_version': 'section7-attempt-cost-v1', 'identity_sha256': self.identity_sha,
                   'event_file': self.path.name, 'events_sha256': sha256_file(self.path), 'cost': cost,
                   'clock_scope': _COST_SCOPE}
        target = self.path.with_suffix('.receipt.json')
        require(not target.exists(), 'An immutable attempt receipt already exists')
        _atomic_json(target, receipt)


def training_cost_summary(run_directory, *, committed_counters=None, allow_uncommitted_tail=True):
    """Read-only cost reduction, including failed/replayed work; no tensor imports."""
    directory = Path(run_directory).resolve(strict=True)
    saved = read_json(directory / 'run_binding.json')
    identity_sha = saved['identity_sha256']
    require(digest(saved['binding']) == identity_sha, 'Cost run binding checksum differs')
    if committed_counters is None:
        checkpoints = list((directory / 'checkpoints').glob('resume_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
        require(checkpoints, 'Cost summary requires a committed checkpoint')
        committed_counters = _checkpoint_manifest(sorted(checkpoints)[-1], identity_sha)['counters']
    _validate_counters(committed_counters, saved['binding']['rows_per_pass'], 5)
    attempts, totals = [], {key: 0 for key in _WORK_KEYS}
    for path in sorted(directory.glob('attempt_*.jsonl')):
        # One byte snapshot keeps the cost/hash pair consistent while a live
        # process is appending its next intent. Incomplete lines are uncharged
        # until the writer finishes and fsyncs them before its model call.
        data = path.read_bytes()
        cost = _event_cost(path, identity_sha, data=data, allow_uncommitted_tail=allow_uncommitted_tail)
        descriptor = {'event_file': path.name, 'events_sha256': sha256(data).hexdigest(), 'cost': cost}
        receipt_path = path.with_suffix('.receipt.json')
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['events_sha256'] != descriptor['events_sha256']:
                # A live writer may publish its finish/receipt between our
                # snapshot and this exists check. Finished streams are immutable.
                data = path.read_bytes()
                cost = _event_cost(path, identity_sha, data=data, allow_uncommitted_tail=allow_uncommitted_tail)
                descriptor.update(events_sha256=sha256(data).hexdigest(), cost=cost)
            require(receipt == {'schema_version': 'section7-attempt-cost-v1', 'identity_sha256': identity_sha,
                    'event_file': path.name, 'events_sha256': descriptor['events_sha256'], 'cost': cost,
                    'clock_scope': _COST_SCOPE}, 'Immutable attempt receipt differs')
            descriptor.update(receipt_file=receipt_path.name, receipt_sha256=sha256_file(receipt_path))
        else:
            descriptor.update(receipt_file=None, receipt_sha256=None)
        attempts.append(descriptor)
        for key in totals:
            totals[key] += cost['work'][key]
    load_wall = sum(row['cost']['model_load_seconds'] for row in attempts)
    train_wall = sum(row['cost']['wall_seconds'] for row in attempts)
    gpu_wall = sum(row['cost']['wall_seconds'] + row['cost']['model_load_seconds']
                   for row in attempts if row['cost']['gpu_assigned'])
    excess = {key: totals[key + '_attempted'] - committed_counters[counter]
              for key, counter in (('examples', 'examples_seen'), ('input_tokens', 'input_tokens_seen'), ('answer_tokens', 'answer_tokens_seen'))}
    excess['optimizer_steps_completed'] = totals['optimizer_steps_completed'] - committed_counters['optimizer_steps']
    require(all(value >= 0 for value in excess.values()), 'Actual work is below committed exposure')
    return {'schema_version': 'section7-training-cost-v1', 'identity_sha256': identity_sha,
        'attempt_count': len(attempts), 'attempts': attempts, 'actual_totals': totals,
        'committed_exposure': deepcopy(committed_counters), 'uncommitted_or_replayed_work': excess,
        'training_wall_seconds': train_wall, 'model_load_seconds': load_wall,
        'attributable_wall_seconds': train_wall + load_wall, 'assigned_device_hours': gpu_wall / 3600.,
        'wall_is_lower_bound': any(row['cost']['wall_is_lower_bound'] for row in attempts),
        'peak_memory_is_lower_bound': any(row['cost']['peak_memory_is_lower_bound'] for row in attempts),
        'peak_allocated_bytes': max((row['cost']['peak_allocated_bytes'] or 0 for row in attempts), default=0),
        'peak_reserved_bytes': max((row['cost']['peak_reserved_bytes'] or 0 for row in attempts), default=0),
        'clock_scope': _COST_SCOPE, 'additional_energy_requests': 0}


@contextmanager
def _run_lock(directory):
    """OS-owned lock releases on process death; its file never resets a run."""
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    handle = (directory.parent / ('.' + directory.name + '.training.lock')).open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError as error:
                raise ValueError('Another trainer holds this run lock') from error
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError as error:
                raise ValueError('Another trainer holds this run lock') from error
        yield
    finally:
        if locked:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def train(runtime, dataset, *, run_directory, identity, resume=False, checkpoint_every_steps=96, stop_after_steps=None):
    require(runtime.metadata.get('execute_authorized') is True, 'Runtime lacks explicit execution authorization')
    with _run_lock(run_directory):
        attempt = _Attempt(runtime, run_directory)
        try:
            result = _train_unlocked(runtime, dataset, run_directory=run_directory, identity=identity,
                resume=resume, checkpoint_every_steps=checkpoint_every_steps, stop_after_steps=stop_after_steps,
                attempt=attempt)
        except BaseException as error:
            if attempt.started:
                attempt.event('attempt_interrupted', error_type=type(error).__name__, last_in_memory_counters=attempt.counters)
                attempt.finish('interrupted')
            raise
        attempt.finish('complete' if result['complete'] else 'bounded_stop')
        result['cost_summary'] = training_cost_summary(run_directory, committed_counters=result['counters'])
        if result['complete']:
            _atomic_json(Path(run_directory) / 'completed.json', result)
        return result


def _train_unlocked(runtime, dataset, *, run_directory, identity, resume=False, checkpoint_every_steps=96,
                    stop_after_steps=None, attempt):
    """Five fixed passes; interruption resumes the exact committed optimizer window."""
    import torch
    require(runtime.metadata.get('execute_authorized') is True, 'Runtime lacks explicit execution authorization')
    require(type(checkpoint_every_steps) is int and checkpoint_every_steps > 0, 'Invalid checkpoint interval')
    require(stop_after_steps is None or type(stop_after_steps) is int and stop_after_steps > 0, 'Invalid bounded stop')
    _validated_numeric(runtime.numerical)
    require(runtime.condition['passes'] == 5 and dataset.condition_id == runtime.condition['condition_id']
            and dataset.model_key == runtime.condition['model_key'], 'Dataset and fixed fit identity differ')
    directory = Path(run_directory).resolve()
    binding = {'identity': identity, 'condition': runtime.condition, 'numerical': runtime.numerical,
        'dataset_summary_sha256': digest(dataset.summary), 'base_binding': runtime.metadata['base_binding'],
        'initial_adapter_sha256': runtime.metadata['initial_adapter_sha256'], 'adapter_manifest': runtime.metadata.get('adapter_manifest'),
        'rows_per_pass': len(dataset.rows), 'optimizer': deepcopy(OPTIMIZER), 'lora': deepcopy(LORA)}
    identity_sha = digest(binding)
    rows = len(dataset.rows); require(rows > 0 and rows % 16 == 0, 'Require complete balanced effective batches')
    counters = {'next_pass': 0, 'next_offset': 0, 'optimizer_steps': 0, 'examples_seen': 0,
                'input_tokens_seen': 0, 'answer_tokens_seen': 0}
    if not resume:
        require(not directory.exists(), 'Use a new run or explicit authenticated resume')
        directory.mkdir(parents=True)
        _atomic_json(directory / 'run_binding.json', {'binding': binding, 'identity_sha256': identity_sha})
        _atomic_json(directory / 'runtime.json', runtime.metadata)
        attempt.start(identity_sha, counters, resume)
        _save_checkpoint(runtime, directory, identity_sha, counters)
    else:
        saved = read_json(directory / 'run_binding.json')
        require(saved == {'binding': binding, 'identity_sha256': identity_sha}, 'Resume data/base/configuration/code identity changed')
        if (directory / 'completed.json').exists():
            verify_training_run(directory, expected_identity=identity)
            counters = read_json(directory / 'completed.json')['counters']
            attempt.start(identity_sha, counters, resume)
            return read_json(directory / 'completed.json')
        checkpoints = directory / 'checkpoints'
        if not list(checkpoints.glob('resume_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]')):
            require(not any(_event_cost(path, identity_sha, allow_uncommitted_tail=True)['work']['forward_attempts']
                            for path in directory.glob('attempt_*.jsonl')),
                    'A run with attempted updates lost all committed checkpoints')
            # Initial serialization failed before any forward intent.
            # Fresh initialization is bound to the identical adapter fingerprint.
            _save_checkpoint(runtime, directory, identity_sha, counters)
        latest, folder = _latest_checkpoint(directory, identity_sha)
        manifest = _checkpoint_manifest(folder, identity_sha)
        require(sha256_file(folder / 'manifest.json') == latest['manifest_sha256'], 'Latest checkpoint pointer changed')
        state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=True)
        require(state['identity_sha256'] == identity_sha and state['counters'] == manifest['counters'], 'Trainer state differs from manifest')
        _restore_adapter(runtime.model, state['adapter'])
        runtime.optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        counters = state['counters']
        _validate_counters(counters, rows, 5)
        attempt.start(identity_sha, counters, resume)
    initial_step = counters['optimizer_steps']
    event = attempt.event
    runtime.model.train(); _configure_model(runtime.model, runtime.numerical['gradient_checkpointing'])
    ordered_pass, order = None, None
    while counters['next_pass'] < 5:
        if ordered_pass != counters['next_pass']:
            ordered_pass = counters['next_pass']
            order = _order(dataset, ordered_pass)
        window = order[counters['next_offset']:counters['next_offset'] + 16]
        event('window_started', optimizer_step=counters['optimizer_steps'] + 1, row_indices=list(window))
        measured = _step(runtime, dataset, window, progress_callback=lambda kind, **fields:
                         event(kind, optimizer_step=counters['optimizer_steps'] + 1, **fields))
        examples = counters['examples_seen'] + len(window)
        next_pass, offset = divmod(examples, rows)
        counters = {'next_pass': next_pass, 'next_offset': offset,
            'optimizer_steps': counters['optimizer_steps'] + 1, 'examples_seen': examples,
            'input_tokens_seen': counters['input_tokens_seen'] + measured['input_tokens'],
            'answer_tokens_seen': counters['answer_tokens_seen'] + measured['answer_tokens']}
        attempt.counters = deepcopy(counters)
        event('window_completed', counters=counters, **measured)
        bounded_stop = stop_after_steps is not None and counters['optimizer_steps'] - initial_step >= stop_after_steps
        if counters['optimizer_steps'] % checkpoint_every_steps == 0 or offset == 0 or bounded_stop:
            _save_checkpoint(runtime, directory, identity_sha, counters)
        if bounded_stop and next_pass < 5:
            event('bounded_stop', counters=counters)
            return {'complete': False, 'identity_sha256': identity_sha, 'counters': counters, 'run_directory': str(directory)}
    _validate_counters(counters, rows, 5)
    final = _save_checkpoint(runtime, directory, identity_sha, counters, final=True)
    result = {'complete': True, 'identity_sha256': identity_sha, 'counters': counters, 'final_checkpoint': final,
              'checkpoint_selection': 'final_after_five_passes', 'run_directory': str(directory),
              'additional_energy_requests': 0, 'training_executed_by': 'explicit user execution'}
    event('completed', counters=counters)
    return result


def verify_training_run(run_directory, *, expected_identity=None):
    """Read-only final-run authentication using stdlib hashes; loads no tensors.

    expected_identity may be the caller's original identity mapping or the full
    effective identity SHA256 returned by train. This never repairs pointers.
    """
    directory = Path(run_directory).resolve(strict=True)
    saved = read_json(directory / 'run_binding.json')
    binding, identity_sha = saved['binding'], saved['identity_sha256']
    require(digest(binding) == identity_sha, 'Run binding checksum differs')
    if expected_identity is not None:
        require(expected_identity == identity_sha if isinstance(expected_identity, str)
                else expected_identity == binding['identity'], 'Expected training identity differs')
    completed = read_json(directory / 'completed.json')
    require(completed['complete'] is True and completed['identity_sha256'] == identity_sha
            and completed['checkpoint_selection'] == 'final_after_five_passes', 'Final fit completion identity differs')
    counters = completed['counters']
    _validate_counters(counters, binding['rows_per_pass'], 5)
    require(counters['examples_seen'] == 5 * binding['rows_per_pass']
            and counters['next_pass'] == 5 and counters['next_offset'] == 0,
            'Final checkpoint does not cover exactly five full passes')
    final = _inside(directory / 'checkpoints' / 'final', directory / 'checkpoints')
    manifest = _checkpoint_manifest(final, identity_sha)
    require(manifest['kind'] == 'final' and manifest['counters'] == counters
            and completed['final_checkpoint'] == {'path': 'final', 'manifest_sha256': sha256_file(final / 'manifest.json')}
            and manifest['contains_pretrained_base_weights'] is False, 'Final adapter or counter binding differs')
    resumes = sorted((directory / 'checkpoints').glob('resume_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
    require(1 <= len(resumes) <= 2, 'Complete resume checkpoint retention differs')
    for folder in resumes:
        _inside(folder, directory / 'checkpoints')
        previous = _checkpoint_manifest(folder, identity_sha)
        _validate_counters(previous['counters'], binding['rows_per_pass'], 5)
        require(previous['counters']['optimizer_steps'] <= counters['optimizer_steps'], 'Resume checkpoint exceeds final exposure')
    cost = training_cost_summary(directory, committed_counters=counters)
    require(completed.get('cost_summary') == cost, 'Actual attempt cost or event authentication differs')
    return {'all_passed': True, 'complete': True, 'identity_sha256': identity_sha,
        'final_checkpoint': str(final), 'final_checkpoint_sha256': sha256_file(final / 'manifest.json'),
        'counters': counters, 'cost_summary': cost, 'resume_checkpoints_verified': len(resumes), 'model_loads': 0,
        'optimizer_steps_executed': 0, 'additional_energy_requests': 0}
def benchmark(runtime, dataset, *, candidates=((4, 4), (8, 2), (16, 1), (1, 16)),
              gradient_checkpointing_options=(False, True), warmup_steps=1, measured_steps=3, execute=False):
    """User-run disposable updates; restore resident adapter, optimizer and all RNG."""
    require(execute is True and runtime.metadata.get('execute_authorized') is True, 'Benchmark requires explicit user execution')
    require(type(warmup_steps) is int and warmup_steps >= 1 and type(measured_steps) is int and measured_steps >= 1,
            'Require real warmup and measured optimizer steps')
    import torch
    adapter, rng = _adapter_state(runtime.model), capture_rng()
    optimizer_state = _cpu_tree(runtime.optimizer.state_dict())
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in _trainable(runtime.model).items()}
    was_training, original_numeric = runtime.model.training, deepcopy(runtime.numerical)
    cache_setting = runtime.model.config.use_cache
    expected = (state_fingerprint(adapter), state_fingerprint(optimizer_state), state_fingerprint(rng))
    order = _order(dataset, 0)
    if hasattr(dataset, 'memory_stress_indices'):
        stress = tuple(dataset.memory_stress_indices(16))
        stress_policy = '16 longest cached complete prompt+answer sequences; row index breaks ties'
    else:
        require(runtime.metadata.get('fixture_only') is True, 'Production data must expose the authenticated longest-sequence stress window')
        stress, stress_policy = tuple(order[:16]), 'CPU fixture only: first16 row-order indices'
    require(len(stress) == 16 and len(set(stress)) == 16
            and all(type(index) is int and 0 <= index < len(order) for index in stress), 'Memory stress window differs')
    records = []
    benchmark_began = time.perf_counter()
    try:
        for enabled in gradient_checkpointing_options:
            for microbatch, accumulation in candidates:
                numerical = numerical_configuration(microbatch, accumulation, gradient_checkpointing=enabled)
                _restore_adapter(runtime.model, adapter); restore_rng(rng)
                _configure_model(runtime.model, enabled); runtime.model.train()
                optimizer = _optimizer(runtime.model)
                initial_free = torch.cuda.mem_get_info(runtime.device)[0] if torch.device(runtime.device).type == 'cuda' else None
                initial_reserved = torch.cuda.memory_reserved(runtime.device) if initial_free is not None else None
                budget = None if initial_free is None else initial_free + initial_reserved
                record = {'numerical_configuration': numerical, 'success': False, 'initial_free_bytes': initial_free,
                    'initial_reserved_bytes': initial_reserved, 'usable_memory_budget_bytes': budget,
                    'required_memory_headroom_fraction': .2, 'memory_headroom_passed': False,
                    'optimizer_step_attempts': 0, 'optimizer_steps_completed': 0,
                    'warmup_optimizer_steps_completed': 0, 'measured_optimizer_steps_completed': 0,
                    'actual_work': {key: 0 for key in _WORK_KEYS}}
                candidate_began = time.perf_counter()
                if initial_free is not None:
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                def observed(kind, **counts):
                    work = record['actual_work']
                    if kind in ('microbatch_started', 'microbatch_completed'):
                        suffix = '_attempted' if kind == 'microbatch_started' else '_backpropagated'
                        work['forward_attempts' if kind == 'microbatch_started' else 'backwards_completed'] += 1
                        for key in ('examples', 'input_tokens', 'answer_tokens'):
                            work[key + suffix] += counts[key]
                    elif kind == 'optimizer_started':
                        work['optimizer_attempts'] += 1
                    elif kind == 'optimizer_completed':
                        work['optimizer_steps_completed'] += 1
                try:
                    totals = {'examples': 0, 'input_tokens': 0, 'answer_tokens': 0, 'forward_calls': 0}
                    for step in range(warmup_steps + measured_steps):
                        if step == warmup_steps:
                            _sync(runtime.device)
                            cuda = torch.device(runtime.device).type == 'cuda'
                            record.update(stress_warmup_peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device) if cuda else None,
                                          stress_warmup_peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device) if cuda else None)
                            started = time.perf_counter()
                        indices = stress if step < warmup_steps else tuple(
                            order[((step - warmup_steps) * 16 + index) % len(order)] for index in range(16))
                        record['optimizer_step_attempts'] += 1
                        record['actual_work']['window_attempts'] += 1
                        result = _step(runtime, dataset, indices, optimizer=optimizer, microbatch=microbatch,
                                       progress_callback=observed)
                        record['optimizer_steps_completed'] += 1
                        record['warmup_optimizer_steps_completed' if step < warmup_steps else 'measured_optimizer_steps_completed'] += 1
                        if step >= warmup_steps:
                            for key in totals:
                                totals[key] += result[key]
                    _sync(runtime.device)
                    elapsed = time.perf_counter() - started
                    cuda = torch.device(runtime.device).type == 'cuda'
                    free, total = torch.cuda.mem_get_info(runtime.device) if cuda else (None, None)
                    record.update(success=True, measured_seconds=elapsed, examples_per_second=totals['examples'] / elapsed,
                        input_tokens_per_second=totals['input_tokens'] / elapsed, measured_counts=totals,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device) if cuda else None,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device) if cuda else None,
                        free_vram_bytes_after_measurement=free, total_vram_bytes=total)
                    record['memory_headroom_passed'] = cuda and record['peak_reserved_bytes'] <= .8 * budget
                except Exception as error:
                    record.update(error_type=type(error).__name__, out_of_memory=isinstance(error, torch.cuda.OutOfMemoryError))
                finally:
                    record['wall_seconds_including_warmup_and_failed_work'] = time.perf_counter() - candidate_began
                    cuda = torch.device(runtime.device).type == 'cuda'
                    record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device) if cuda else None,
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device) if cuda else None)
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    if torch.device(runtime.device).type == 'cuda':
                        torch.cuda.empty_cache()
                records.append(record)
    finally:
        _restore_adapter(runtime.model, adapter); runtime.optimizer.load_state_dict(optimizer_state)
        _configure_model(runtime.model, original_numeric['gradient_checkpointing'])
        runtime.model.config.use_cache = cache_setting; runtime.model.train(was_training)
        for name, parameter in _trainable(runtime.model).items():
            parameter.grad = gradients[name]
        restore_rng(rng)
    actual = (state_fingerprint(_adapter_state(runtime.model)), state_fingerprint(runtime.optimizer.state_dict()), state_fingerprint(capture_rng()))
    require(actual == expected, 'Disposable benchmark failed to restore adapter/optimizer/RNG')
    runtime_environment = environment(runtime.device, fixture=runtime.metadata.get('fixture_only', False))
    return {'schema_version': 'section7-resident-benchmark-v1', 'candidates': records,
        'model_key': runtime.condition['model_key'], 'condition_id': runtime.condition['condition_id'],
        'warmup_steps_per_candidate': warmup_steps, 'measured_steps_per_candidate': measured_steps,
        'memory_stress_row_indices': list(stress), 'memory_stress_policy': stress_policy,
        'runtime': runtime_environment,
        'benchmark_wall_seconds_including_warmup_and_failed_candidates': time.perf_counter() - benchmark_began,
        'model_load_seconds': runtime.metadata.get('model_load_seconds', 0.),
        'disposable_optimizer_step_attempts': sum(row['optimizer_step_attempts'] for row in records),
        'disposable_optimizer_steps_completed': sum(row['optimizer_steps_completed'] for row in records),
        'disposable_actual_work': {key: sum(row['actual_work'][key] for row in records) for key in _WORK_KEYS},
        'memory_headroom_rule': 'peak reserved <= 0.8 * (free before candidate + allocator reserved before candidate)',
        'base_model_resident_between_candidates': True, 'adapter_optimizer_rng_restored': True,
        'restored_state_sha256': list(actual), 'candidate_selection_executed': False,
        'production_optimizer_steps': 0, 'additional_energy_requests': 0,
        'scope': 'disposable training-row throughput preflight; does not fit or select a final adapter'}
