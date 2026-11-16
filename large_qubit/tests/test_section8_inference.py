"""CPU/fake-model inference mechanics; no pretrained weights or CUDA calls."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from phase6.test_grammar import FakeByteLevel
from large_qubit.common import canonical_bytes, sha256_file
from large_qubit.section8_inference import ResidentGenerator, DECODING, RUN_ROOT
from large_qubit.tests._section5_fixtures import catalogue_fixture


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=False)
        self._large_qubit_binding = {'model_key': 'gpt2_xl', 'weights_authenticated': True}
        self.names = []
        self.active = None
        self.disabled = False
        self.calls = []
        self.mode = 'valid'

    def set_adapter(self, name):
        self.active = name
        self.weight.requires_grad_(True)  # PEFT may do this; switch must undo it.

    def load_adapter(self, path, **kwargs):
        self.names.append((path, kwargs))

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def generate(self, input_ids, attention_mask, generation_config, prefix_allowed_tokens_fn):
        self.calls.append({'disabled': self.disabled, 'active': self.active,
                           'inference': torch.is_inference_mode_enabled(),
                           'grad': self.weight.requires_grad,
                           'attention': attention_mask.tolist(),
                           'configuration': deepcopy(generation_config)})
        if self.mode == 'raise':
            torch.rand(3)
            raise RuntimeError('fixture generation failure')
        output = input_ids.clone()
        for _ in range(generation_config.max_new_tokens):
            allowed = prefix_allowed_tokens_fn(0, output[0])
            token = allowed[int(torch.randint(len(allowed), (1,)).item())]
            output = torch.cat((output, torch.tensor([[token]], dtype=torch.long)), dim=1)
            if token == generation_config.eos_token_id:
                break
        if self.mode == 'truncated':
            output = output[:, :-1]
        elif self.mode == 'duplicate_eos':
            output = torch.cat((output, torch.tensor([[generation_config.eos_token_id]])), dim=1)
        elif self.mode == 'changed_prompt':
            output[0, 0] += 1
        return SimpleNamespace(sequences=output)


class FakeConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.model = FakeModel()
        self.generator = ResidentGenerator.__new__(ResidentGenerator)
        self.generator.workspace = self.root
        self.generator.model_key = 'gpt2_xl'
        self.generator._closed = False
        self.generator._adapters = {}
        self.generator._prompts = {}
        self.generator._grammars = {}
        self.generator.condition_id = None
        self.generator.model = self.model
        self.generator.tokenizer = FakeByteLevel()
        self.generator.tokenizer.pieces.update({2000: 'tiny', 2001: '\nAnswer:\n'})
        self.generator.tokenizer.chat_template = None
        from large_qubit.models import MODEL_SPECS
        self.generator.tokenizer._large_qubit_binding = {
            'model_key': 'gpt2_xl', 'revision': MODEL_SPECS['gpt2_xl']['revision'],
            'metadata_authenticated': True}
        self.generator._torch = torch
        self.generator.device = torch.device('cpu')
        self.generator._generation_config = FakeConfig(**DECODING, max_new_tokens=138,
            eos_token_id=1000, pad_token_id=1000)
        self.generator._metadata = {'runtime_identity': {'fixture': True}}
        self.generator._stats = dict(base_loads=1, adapter_loads=0, attempts=0,
            valid_attempts=0, generation_errors=0, input_tokens=0, output_tokens=0,
            model_load_seconds=0., adapter_load_seconds=0., generation_wall_seconds=0.)
        self.patches = [
            patch('large_qubit.prompts.compact_task', return_value='tiny'),
            patch.object(torch.cuda, 'synchronize'),
            patch.object(torch.cuda, 'device', side_effect=lambda *a: nullcontext()),
            patch.object(torch.cuda, 'manual_seed'),
            patch.object(torch, 'autocast', side_effect=lambda *a, **kw: nullcontext()),
        ]
        for mocked in self.patches:
            mocked.start()

    def tearDown(self):
        for mocked in reversed(self.patches):
            mocked.stop()
        self.temporary.cleanup()

    def adapter(self, condition):
        folder = self.root / RUN_ROOT / condition / 'checkpoints/final'
        adapter = folder / 'adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'not real tensors')
        manifest = {'schema_version': 'section7-training-checkpoint-v1', 'kind': 'final',
            'counters': {'optimizer_steps': 1920, 'examples_seen': 30720},
            'contains_pretrained_base_weights': False, 'adapter_sha256': 'state-fixture',
            'files': {p.relative_to(folder).as_posix(): sha256_file(p) for p in adapter.iterdir()}}
        (folder / 'manifest.json').write_bytes(canonical_bytes(manifest))
        (folder / 'manifest.sha256').write_text(sha256_file(folder / 'manifest.json') + '\n')
        return adapter

    def peft(self):
        def attach(model, path, **kwargs):
            model.names.append((path, kwargs))
            return model
        return patch.dict(sys.modules, {'peft': SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=Mock(side_effect=attach)))})

    def test_no_implicit_load_or_cuda_query(self):
        with patch('large_qubit.section8_inference.load_base_model') as base, \
                patch('large_qubit.section8_inference.load_prompt_tokenizer') as token, \
                patch.object(torch.cuda, 'is_available') as cuda:
            with self.assertRaisesRegex(ValueError, 'execute=True'):
                ResidentGenerator(self.root, 'gpt2_xl')
        base.assert_not_called()
        token.assert_not_called()
        cuda.assert_not_called()

    def test_constructor_authenticates_base_once_and_uses_metadata_tokenizer(self):
        device = SimpleNamespace(type='cuda', index=0)
        fake_torch = SimpleNamespace(device=lambda d: device,
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 synchronize=Mock()),
            use_deterministic_algorithms=Mock(),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=9)
        with patch.dict(sys.modules, {'torch': fake_torch,
                'transformers': SimpleNamespace(GenerationConfig=FakeConfig)}), \
                patch('large_qubit.section8_inference._runtime_identity', return_value={'stable': True}), \
                patch('large_qubit.section8_inference.load_base_model', return_value=self.model) as base, \
                patch('large_qubit.section8_inference.load_prompt_tokenizer', return_value=tokenizer) as token, \
                patch.dict('os.environ', {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            generator = ResidentGenerator(self.root, 'gpt2_xl', execute=True)
        base.assert_called_once_with(self.root, 'gpt2_xl', execute=True,
            device=str(device), precision='bf16')
        token.assert_called_once_with(self.root, 'gpt2_xl', execute=True)
        self.assertEqual(generator.statistics()['base_loads'], 1)
        self.assertEqual(generator.metadata['runtime_identity'], {'stable': True})
        self.assertEqual(generator.metadata['generation_configuration']['top_k'], 0)
        self.assertTrue(self.model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, 'left')

    def test_fixed_native_masked_sampling_and_canonical_eos(self):
        self.generator.switch('gpt2_xl_frozen')
        result = self.generator.generate('CO2_V01', self.catalogue, 123)
        self.assertTrue(result['valid'], result)
        self.assertEqual(len(result['ops']), 32)
        self.assertEqual(result['token_ids'][-1], 1000)
        self.assertEqual(result['input_tokens'], 2)
        self.assertEqual(result['output_tokens'], len(result['token_ids']))
        self.assertNotIn('<eos>', result['raw_text'])
        call = self.model.calls[0]
        self.assertTrue(call['inference'])
        self.assertFalse(call['grad'])
        self.assertEqual(call['attention'], [[1, 1]])
        for key, value in DECODING.items():
            self.assertEqual(getattr(call['configuration'], key), value)
        self.assertEqual(self.generator.statistics()['valid_attempts'], 1)
        self.assertGreaterEqual(result['wall_seconds'], 0.)

    def test_attempt_seed_replay_restores_rng_and_ignores_interleaving(self):
        self.generator.switch('gpt2_xl_frozen')
        before = torch.random.get_rng_state().clone()
        first = self.generator.generate('CO2_V01', self.catalogue, 42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.generator.generate('CO2_V01', self.catalogue, 8)
        repeated = self.generator.generate('CO2_V01', self.catalogue, 42)
        self.assertEqual(first['token_ids'], repeated['token_ids'])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(len(self.generator._prompts), 1)
        self.assertEqual(len(self.generator._grammars), 1)

    def test_two_named_adapters_reuse_backbone_and_frozen_disables_them(self):
        weighted = self.adapter('gpt2_xl_weighted_lora')
        uniform = self.adapter('gpt2_xl_uniform_lora')
        with self.peft():
            binding = self.generator.switch('gpt2_xl_weighted_lora', weighted)
            self.generator.generate('CO2_V01', self.catalogue, 1)
            self.generator.switch('gpt2_xl_uniform_lora', uniform)
            self.generator.generate('CO2_V01', self.catalogue, 1)
            self.generator.switch('gpt2_xl_frozen')
            self.generator.generate('CO2_V01', self.catalogue, 1)
            self.generator.switch('gpt2_xl_weighted_lora', weighted)
            self.generator.generate('CO2_V01', self.catalogue, 1)
        self.assertIs(self.generator.model, self.model)
        self.assertEqual(self.generator.statistics()['base_loads'], 1)
        self.assertEqual(self.generator.statistics()['adapter_loads'], 2)
        self.assertEqual([row['disabled'] for row in self.model.calls], [False, False, True, False])
        self.assertEqual(self.model.active, 'gpt2_xl_weighted_lora')
        self.assertFalse(self.model.disabled)
        self.assertFalse(self.model.weight.requires_grad)
        self.assertEqual(binding['path'], weighted.relative_to(self.root).as_posix())
        for _, kwargs in self.model.names:
            self.assertFalse(kwargs['is_trainable'])
            self.assertTrue(kwargs['local_files_only'])

    def test_wrong_backbone_frozen_adapter_and_historical_adapter_refused(self):
        with self.assertRaisesRegex(ValueError, 'backbone'):
            self.generator.switch('qwen3_4b_frozen')
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            self.generator.switch('gpt2_xl_frozen', self.root)
        self.adapter('gpt2_xl_weighted_lora')
        with self.assertRaisesRegex(ValueError, 'final adapter'):
            self.generator.switch('gpt2_xl_weighted_lora', self.root)

    def test_adapter_payload_mutation_refused_before_peft(self):
        adapter = self.adapter('gpt2_xl_weighted_lora')
        (adapter / 'adapter_model.safetensors').write_bytes(b'altered')
        with self.peft(), self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.generator.switch('gpt2_xl_weighted_lora', adapter)
        self.assertEqual(self.generator.statistics()['adapter_loads'], 0)

    def test_invalid_and_truncated_tokens_preserved(self):
        self.generator.switch('gpt2_xl_frozen')
        for mode in ('truncated', 'duplicate_eos'):
            self.model.mode = mode
            result = self.generator.generate('CO2_V01', self.catalogue, 42)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['ops'])
            self.assertTrue(result['token_ids'])
            self.assertTrue(result['raw_text'])
            self.assertIn('error', result)
            self.assertEqual(result['error_kind'], 'invalid_output')
        self.assertEqual(self.generator.statistics()['attempts'], 2)
        self.assertEqual(self.generator.statistics()['valid_attempts'], 0)

    def test_generation_exception_recorded_and_rng_restored(self):
        self.generator.switch('gpt2_xl_frozen')
        self.model.mode = 'raise'
        before = torch.random.get_rng_state().clone()
        result = self.generator.generate('CO2_V01', self.catalogue, 1)
        self.assertEqual(result['error']['message'], 'fixture generation failure')
        self.assertEqual(result['error_kind'], 'infrastructure')
        self.assertFalse(result['valid'])
        self.assertEqual(result['token_ids'], [])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_synchronization_fault_is_infrastructure_even_after_valid_generation(self):
        self.generator.switch('gpt2_xl_frozen')
        # Before generation and immediately after generation succeed; the final
        # synchronization then exposes a device/runtime failure.
        with patch.object(torch.cuda, 'synchronize',
                          side_effect=[None, None, RuntimeError('fixture CUDA fault')]):
            result = self.generator.generate('CO2_V01', self.catalogue, 1)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['ops'])
        self.assertEqual(result['error_kind'], 'infrastructure')
        self.assertEqual(result['error']['message'], 'fixture CUDA fault')
        self.assertTrue(result['token_ids'])
        self.assertTrue(result['raw_text'])
        self.assertTrue(result['wall_is_lower_bound'])
        self.assertEqual(self.generator.statistics()['generation_errors'], 1)

    def test_prompt_prefix_change_rejected(self):
        self.generator.switch('gpt2_xl_frozen')
        self.model.mode = 'changed_prompt'
        result = self.generator.generate('CO2_V01', self.catalogue, 1)
        self.assertFalse(result['valid'])
        self.assertIn('prefix changed', result['error']['message'])
        self.assertTrue(result['token_ids'])
        self.assertTrue(result['raw_text'])

    def test_native_qwen_template_explicitly_disables_thinking(self):
        from large_qubit.models import MODEL_SPECS
        self.generator.model_key = 'qwen3_4b'
        tokenizer = self.generator.tokenizer
        template = 'fixture-native-chat-template'
        tokenizer.get_chat_template = lambda: template
        calls = []

        def apply(messages, *, tokenize, add_generation_prompt, **kwargs):
            calls.append((messages, tokenize, add_generation_prompt, kwargs))
            text = 'tiny\nAnswer:\n'
            return tokenizer.encode(text, add_special_tokens=False) if tokenize else text

        tokenizer.apply_chat_template = apply
        tokenizer._large_qubit_binding = {
            'model_key': 'qwen3_4b', 'revision': MODEL_SPECS['qwen3_4b']['revision'],
            'metadata_authenticated': True,
            'native_chat_template_sha256': sha256(template.encode()).hexdigest()}
        self.generator.switch('qwen3_4b_frozen')
        result = self.generator.generate('CO2_V01', self.catalogue, 5)
        self.assertTrue(result['valid'], result)
        self.assertEqual(len(calls), 2)
        for messages, _, add_generation_prompt, kwargs in calls:
            self.assertEqual(messages, [{'role': 'user', 'content': 'tiny'}])
            self.assertTrue(add_generation_prompt)
            self.assertEqual(kwargs, {'enable_thinking': False})
        self.assertEqual(result['prompt_certificate']['template_kwargs'], {'enable_thinking': False})

    def test_missing_condition_and_invalid_seeds_refused(self):
        with self.assertRaisesRegex(ValueError, 'Select'):
            self.generator.generate('CO2_V01', self.catalogue, 1)
        self.generator.switch('gpt2_xl_frozen')
        for seed in (True, -1, 2**63, 1.0):
            with self.assertRaisesRegex(ValueError, 'seed'):
                self.generator.generate('CO2_V01', self.catalogue, seed)
        self.assertEqual(self.generator.statistics()['attempts'], 0)

    def test_metadata_copies_and_close_release_resident_models(self):
        result = self.generator.metadata
        result['runtime_identity']['fixture'] = False
        self.assertTrue(self.generator.metadata['runtime_identity']['fixture'])
        with patch.object(torch.cuda, 'empty_cache') as empty:
            self.generator.close()
            self.generator.close()
        empty.assert_called_once()
        self.assertIsNone(self.generator.model)
        self.assertIsNone(self.generator.tokenizer)
        with self.assertRaisesRegex(ValueError, 'closed'):
            self.generator.switch('gpt2_xl_frozen')


if __name__ == '__main__':
    unittest.main()

