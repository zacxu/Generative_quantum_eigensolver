"""Explicit, local-only Section 8 inference with one resident native backbone.

Importing this module never imports Torch, queries CUDA or loads weights. The
controller authenticates the complete Section 7 release before constructing a
generator; this module also authenticates each selected final adapter payload.
No optimizer or energy evaluator is constructed here.
"""

from contextlib import nullcontext
from copy import deepcopy
import gc
from importlib import metadata as package_metadata
import os
from pathlib import Path
import platform
import time

from .common import digest, read_json, require, sha256_file, workspace_path
from .gpu_identity import normalize_gpu_uuid
from .grammar import CircuitGrammar, parse_answer
from .models import MODEL_SPECS, load_base_model
from .prompts import build_prompt, load_prompt_tokenizer


RUN_ROOT = 'large_qubit/training_runs/section7_gpu_uuid_v1'
CONDITIONS = {
    'gpt2_xl': ('gpt2_xl_frozen', 'gpt2_xl_weighted_lora', 'gpt2_xl_uniform_lora'),
    'qwen3_4b': ('qwen3_4b_frozen', 'qwen3_4b_weighted_lora'),
}
DECODING = {
    'do_sample': True, 'temperature': .7, 'top_k': 0, 'top_p': 1.,
    'typical_p': 1., 'repetition_penalty': 1., 'num_beams': 1,
    'num_return_sequences': 1, 'min_new_tokens': 0, 'use_cache': True,
    'return_dict_in_generate': True, 'output_scores': False,
}


def _runtime_identity(torch, index):
    with torch.cuda.device(index):
        properties = torch.cuda.get_device_properties(index)
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    require(native_bf16, 'Production inference requires native BF16 CUDA')
    return {
        'python': platform.python_version(),
        'packages': {name: package_metadata.version(name) for name in (
            'torch', 'transformers', 'peft', 'accelerate', 'tokenizers',
            'safetensors', 'huggingface-hub', 'numpy')},
        'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'device': f'cuda:{index}', 'gpu_uuid': normalize_gpu_uuid(properties.uuid),
        'gpu_name': properties.name,
        'gpu_compute_capability': list(torch.cuda.get_device_capability(index)),
        'native_bf16': bool(native_bf16), 'precision': 'bf16',
        'attention_implementation': 'eager', 'batch_size': 1,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }


class ResidentGenerator:
    """A CUDA backbone reused across frozen and named final-adapter conditions.

    ``execute=True`` is deliberately required even for construction. GPU locking,
    campaign admission and durable attempt reservation belong to the controller.
    ``generate`` always samples one answer at a time, so changing queue grouping
    cannot change the RNG stream belonging to another reserved attempt.
    """

    def __init__(self, workspace, model_key, device='cuda:0', *, execute=False):
        require(execute is True, 'Model inference requires explicit execute=True')
        require(model_key in MODEL_SPECS, 'Unknown registered inference backbone')
        require(package_metadata.version('peft') == '0.18.1',
                'Use the pinned inspected PEFT 0.18.1 inference implementation')
        self.workspace = Path(workspace).resolve(strict=True)
        self.model_key = model_key
        self._closed = False
        self._adapters = {}
        self._prompts = {}
        self._grammars = {}
        self.condition_id = None
        self._stats = {'base_loads': 0, 'adapter_loads': 0, 'attempts': 0,
                       'valid_attempts': 0, 'generation_errors': 0,
                       'input_tokens': 0, 'output_tokens': 0,
                       'model_load_seconds': 0., 'adapter_load_seconds': 0.,
                       'generation_wall_seconds': 0.}
        # This must precede CUDA initialization. An incompatible existing value
        # is refused rather than silently changing the numerical environment.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        require(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'),
                'A deterministic CUBLAS workspace configuration is required')
        import torch
        from transformers import GenerationConfig
        selected = torch.device(device)
        require(selected.type == 'cuda' and torch.cuda.is_available(),
                'Production inference requires a CUDA device')
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        require(0 <= index < torch.cuda.device_count(), 'Selected CUDA device is unavailable')
        self._torch, self.device = torch, torch.device(f'cuda:{index}')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = _runtime_identity(torch, index)
        started = time.perf_counter()
        # Tokenizer-only metadata authentication avoids hashing all base tensors
        # twice. The native base loader below performs the full byte check once.
        self.tokenizer = load_prompt_tokenizer(self.workspace, model_key, execute=True)
        self.tokenizer.padding_side = 'left'
        self.model = load_base_model(self.workspace, model_key, execute=True,
                                     device=str(self.device), precision='bf16')
        self.model.config.use_cache = True
        self._generation_config = GenerationConfig(**DECODING,
            max_new_tokens=138, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=getattr(self.tokenizer, 'bos_token_id', None))
        torch.cuda.synchronize(self.device)
        self._stats['base_loads'] = 1
        self._stats['model_load_seconds'] = time.perf_counter() - started
        self._metadata = {
            'schema': 'section8-resident-inference-v1', 'model_key': model_key,
            'runtime_identity': runtime,
            'model_binding': deepcopy(self.model._large_qubit_binding),
            'generation_configuration': self._generation_config.to_dict(),
            'length': 32, 'batch_size': 1, 'padding_side': 'left',
            'template_kwargs': dict(MODEL_SPECS[model_key]['template_kwargs']),
            'rng': 'fork_CPU_and_selected_CUDA_only_per_attempt_seed',
            'adapter_autocast_dtype': True, 'optimizer_constructed': False,
            'downloads_executed': 0,
        }

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def statistics(self):
        return deepcopy(self._stats)

    def _require_open(self):
        require(not self._closed, 'The resident generator is closed')

    def _adapter(self, condition_id, adapter_path):
        relative = f'{RUN_ROOT}/{condition_id}/checkpoints/final/adapter'
        expected = workspace_path(self.workspace, relative, namespace=RUN_ROOT)
        provided = expected if adapter_path is None else Path(adapter_path)
        if not provided.is_absolute():
            provided = self.workspace / provided
        require(provided.resolve(strict=True) == expected and expected.is_dir(),
                'Only this CO2 condition final adapter is permitted')
        folder = expected.parent
        manifest_path = folder / 'manifest.json'
        manifest_hash = sha256_file(manifest_path)
        require((folder / 'manifest.sha256').read_text(encoding='ascii').strip() == manifest_hash,
                'Final adapter checkpoint checksum differs')
        manifest = read_json(manifest_path)
        require(manifest['schema_version'] == 'section7-training-checkpoint-v1'
                and manifest['kind'] == 'final'
                and manifest['counters']['optimizer_steps'] == 1920
                and manifest['counters']['examples_seen'] == 30720
                and manifest['contains_pretrained_base_weights'] is False,
                'Only the complete fixed-budget Section 7 final adapter is permitted')
        recorded = {key[8:]: value for key, value in manifest['files'].items()
                    if key.startswith('adapter/')}
        paths = [p for p in expected.rglob('*') if p.is_file()]
        for path in paths:
            workspace_path(self.workspace, path.relative_to(self.workspace).as_posix(),
                           namespace=RUN_ROOT)
        actual = {p.relative_to(expected).as_posix(): sha256_file(p) for p in paths}
        require(actual == recorded and {'adapter_config.json', 'adapter_model.safetensors'} <= set(actual),
                'Final adapter payload inventory or bytes changed')
        return expected, {'path': relative, 'manifest_sha256': manifest_hash,
                          'files': actual, 'adapter_state_sha256': manifest['adapter_sha256']}

    def switch(self, condition_id, adapter_path=None):
        self._require_open()
        require(condition_id in CONDITIONS[self.model_key],
                'Condition does not belong to the resident backbone')
        if condition_id.endswith('_frozen'):
            require(adapter_path is None, 'Frozen controls cannot receive an adapter')
        elif condition_id not in self._adapters:
            path, binding = self._adapter(condition_id, adapter_path)
            from peft import PeftModel
            started = time.perf_counter()
            kwargs = {'adapter_name': condition_id, 'is_trainable': False,
                      'local_files_only': True, 'autocast_adapter_dtype': True}
            if self._adapters:
                self.model.load_adapter(str(path), **kwargs, torch_device=str(self.device))
            else:
                self.model = PeftModel.from_pretrained(self.model, str(path), **kwargs)
            self._torch.cuda.synchronize(self.device)
            self._stats['adapter_loads'] += 1
            self._stats['adapter_load_seconds'] += time.perf_counter() - started
            self._adapters[condition_id] = binding
        elif adapter_path is not None:
            expected = self.workspace / self._adapters[condition_id]['path']
            supplied = Path(adapter_path)
            supplied = supplied if supplied.is_absolute() else self.workspace / supplied
            require(supplied.resolve(strict=True) == expected,
                    'A resident named adapter cannot change its path')
        if not condition_id.endswith('_frozen'):
            self.model.set_adapter(condition_id)
        # PEFT set_adapter can mark parameters trainable: inference disables all
        # gradients again after every switch, including a frozen-control switch.
        self.model.requires_grad_(False)
        self.model.eval()
        self.condition_id = condition_id
        return deepcopy(self._adapters.get(condition_id))

    def generate(self, alias, catalogue, attempt_seed):
        self._require_open()
        require(self.condition_id is not None, 'Select an inference condition first')
        require(type(attempt_seed) is int and 0 <= attempt_seed < 2**63,
                'Attempt seed must be a nonnegative signed 64-bit integer')
        catalogue_key = digest(catalogue)
        key = (alias, catalogue_key)
        if key not in self._prompts:
            self._prompts[key] = build_prompt(self.tokenizer, self.model_key, alias, catalogue, 32)
        if catalogue_key not in self._grammars:
            self._grammars[catalogue_key] = CircuitGrammar(self.tokenizer, catalogue, 32)
        prompt, grammar = self._prompts[key], self._grammars[catalogue_key]
        require(grammar.max_completion_tokens == self._generation_config.max_new_tokens,
                'The frozen complete-answer generation bound changed')
        torch = self._torch
        started = time.perf_counter()
        result = {'condition_id': self.condition_id, 'geometry_id': alias,
                  'attempt_seed': attempt_seed, 'valid': False, 'ops': None,
                  'raw_text': '', 'token_ids': [], 'input_tokens': len(prompt['input_ids']),
                  'output_tokens': 0, 'wall_seconds': 0., 'batch_size': 1,
                  'prompt_certificate': deepcopy(prompt['context_certificate']),
                  'runtime_identity_sha256': digest(self._metadata['runtime_identity']),
                  'adapter_binding': deepcopy(self._adapters.get(self.condition_id))}
        self._stats['attempts'] += 1
        self._stats['input_tokens'] += result['input_tokens']
        error_kind = 'infrastructure'
        try:
            inputs = torch.tensor([prompt['input_ids']], dtype=torch.long, device=self.device)
            attention = torch.ones_like(inputs)
            width = inputs.shape[1]

            def allowed(batch_id, ids):
                require(batch_id == 0, 'Inference batch size is frozen at one')
                return list(grammar.allowed_token_ids(ids[width:].tolist()))

            disabled = (self.model.disable_adapter() if self._adapters
                        and self.condition_id.endswith('_frozen') else nullcontext())
            devices = [self.device.index] if self.device.type == 'cuda' else []
            torch.cuda.synchronize(self.device)
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), disabled:
                # torch.manual_seed also seeds every visible CUDA device. Seed
                # the CPU and selected CUDA generators independently instead.
                torch.random.default_generator.manual_seed(attempt_seed)
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(attempt_seed)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model.generate(input_ids=inputs, attention_mask=attention,
                        generation_config=self._generation_config,
                        prefix_allowed_tokens_fn=allowed)
            # Surface asynchronous device faults before classifying completed
            # native generation output as an invalid scientific completion.
            torch.cuda.synchronize(self.device)
            sequences = output.sequences
            sequence = sequences[0].tolist() if len(sequences) == 1 else None
            error_kind = 'invalid_output'
            require(len(sequences) == 1, 'Generation returned an unexpected batch size')
            result['token_ids'] = sequence[width:]
            result['output_tokens'] = len(result['token_ids'])
            # Keep every generated token. The validator rejects duplicate EOS,
            # trailing tokens and truncation instead of silently cropping them.
            text_ids = result['token_ids'][:-1] if result['token_ids'][-1:] == [grammar.eos_token_id] else result['token_ids']
            result['raw_text'] = self.tokenizer.decode(text_ids, skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)
            require(sequence[:width] == prompt['input_ids'], 'Generated prompt prefix changed')
            ops = grammar.validate_answer_ids(result['token_ids'])
            require(parse_answer(result['raw_text'], catalogue=catalogue, length=32) == ops,
                    'Token DFA and canonical answer parser disagree')
            result['ops'], result['valid'] = ops, True
        except Exception as error:
            result['error'] = {'type': type(error).__name__, 'message': str(error)}
            result['error_kind'] = error_kind
        finally:
            # CUDA faults may make synchronization fail too; retain the original
            # attempt and account for its measured wall lower bound in that case.
            try:
                torch.cuda.synchronize(self.device)
            except Exception as error:
                result['valid'], result['ops'] = False, None
                if 'error' in result:
                    result['prior_error'] = result['error']
                result['error'] = {'type': type(error).__name__, 'message': str(error)}
                result['error_kind'] = 'infrastructure'
                result['wall_is_lower_bound'] = True
            result['wall_seconds'] = time.perf_counter() - started
            self._stats['generation_wall_seconds'] += result['wall_seconds']
            self._stats['output_tokens'] += result['output_tokens']
            self._stats['valid_attempts'] += int(result['valid'])
            self._stats['generation_errors'] += int('error' in result)
        return result

    def close(self):
        if not self._closed:
            self.model, self.tokenizer = None, None
            self._prompts.clear()
            self._grammars.clear()
            self._closed = True
            gc.collect()
            self._torch.cuda.empty_cache()

