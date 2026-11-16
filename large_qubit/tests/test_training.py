"""Tiny synthetic tensor fixtures only; no pretrained load, GPU or server job."""

from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from large_qubit.common import canonical_bytes, read_json, sha256_file
from large_qubit.protocol import FITS, training_conditions
from large_qubit import training


class ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Embedding(4, 4)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(4, 2) * .1)
        self.lora_B = nn.Parameter(torch.zeros(2, 4))
        self.dropout = nn.Dropout(.05)
        self.config = SimpleNamespace(use_cache=True)
        self.is_gradient_checkpointing = False

    def forward(self, input_ids, attention_mask, position_ids):
        hidden = self.base(input_ids)
        return SimpleNamespace(logits=hidden + self.dropout(hidden) @ self.lora_A @ self.lora_B)

    def gradient_checkpointing_enable(self, **kwargs):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def enable_input_require_grads(self):
        self._require_grads_hook = True

    def disable_input_require_grads(self):
        del self._require_grads_hook

    def save_pretrained(self, directory, safe_serialization=True):
        from safetensors.torch import save_file
        directory = Path(directory); directory.mkdir()
        (directory / 'adapter_config.json').write_bytes(canonical_bytes({'fixture_only': True, 'r': 16}) + b'\n')
        save_file(training._adapter_state(self), directory / 'adapter_model.safetensors')


class ToyDataset:
    def __init__(self, condition_id=FITS[0], count=32):
        self.condition_id = condition_id
        self.model_key = 'gpt2_xl'
        self.rows = tuple({'row_id': f'fixture_v2_F{index + 1:06d}', 'geometry_id': f'CO2_T0{index % 3 + 1}'} for index in range(count))
        self.summary = {'fixture_only': True, 'rows': count, 'condition_id': condition_id,
                        'release_sha256': 'a' * 64, 'cache_sha256': 'b' * 64}

    def row_order(self, pass_index):
        result = list(range(len(self.rows)))
        random.Random(719 + pass_index).shuffle(result)
        return tuple(result)

    def collate(self, indices, device='cpu'):
        tokens = torch.tensor([[index % 4, (index + 1) % 4] for index in indices], device=device)
        labels = tokens.clone(); labels[:, 0] = -100
        return {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens),
                'position_ids': torch.arange(2, device=device).expand(len(indices), 2),
                'labels': labels, 'loss_multipliers': torch.tensor([.7 if index % 2 else 1.3 for index in indices], device=device),
                'row_ids': [self.rows[index]['row_id'] for index in indices], 'answer_lengths': [1] * len(indices)}

    def sequence_nll(self, logits, batch):
        return torch.nn.functional.cross_entropy(logits[:, 0, :].float(), batch['labels'][:, 1], reduction='none')


def runtime(*, microbatch=4, condition_id=FITS[0]):
    training._seed_all(20)
    model = ToyAdapter()
    condition = training_conditions()[condition_id]
    numeric = training.numerical_configuration(microbatch, 16 // microbatch)
    metadata = {'execute_authorized': True, 'fixture_only': True, 'base_binding': {'fixture_only': True},
                'adapter_manifest': {'fixture_only': True, 'rank': 16, 'targets': ['lora_A', 'lora_B']},
                'initial_adapter_sha256': training.state_fingerprint(training._adapter_state(model)),
                'historical_adapters_loaded': False, 'downloads_executed': 0, 'energy_requests': 0}
    training._seed_all(91)
    return training.TrainingRuntime(model, training._optimizer(model), 'cpu', condition, numeric, metadata)


class TrainingRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_import_and_nonexecute_load_start_no_model_optimizer_or_scoring(self):
        script = ('import sys; import large_qubit.training; '
                  'assert not ({"torch","transformers","peft","pyscf"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', script], check=True,
                       cwd=Path(__file__).resolve().parents[2])
        with patch('large_qubit.models.load_base_model', side_effect=AssertionError('no model load')):
            with self.assertRaisesRegex(ValueError, 'explicit'):
                training.load_runtime('.', training_conditions()[FITS[0]], training.numerical_configuration())

    def test_optimizer_and_numerical_controls_are_fixed_and_pair_initialization_matches(self):
        left, right = runtime(), runtime(condition_id=FITS[1])
        self.assertEqual(left.metadata['initial_adapter_sha256'], right.metadata['initial_adapter_sha256'])
        group = left.optimizer.param_groups[0]
        self.assertEqual((group['lr'], group['betas'], group['eps'], group['weight_decay']), (.0002, (.9, .999), 1e-8, 0.))
        self.assertFalse(group['fused']); self.assertFalse(group['foreach'])
        for arguments in ((3, 5), (True, 16), (4, 8)):
            with self.assertRaises(ValueError):
                training.numerical_configuration(*arguments)
        with self.assertRaises(ValueError):
            training.numerical_configuration(attention_implementation='sdpa')
        with self.assertRaises(ValueError):
            training.numerical_configuration(optimizer_fused=True)

    def test_production_environment_requires_one_uuid_and_records_sharing_without_commands(self):
        gpu_uuid = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
        properties = SimpleNamespace(name='NVIDIA H200', uuid=gpu_uuid, major=9, minor=0)
        output = SimpleNamespace(stdout=f'{os.getpid()}, {gpu_uuid}, 32\n43210, {gpu_uuid}, 64\n999, GPU-other, 12\n')
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'device', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_device_properties', return_value=properties), \
             patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
             patch.object(torch.cuda, 'mem_get_info', return_value=(1000, 2000)), \
             patch.object(training.subprocess, 'run', return_value=output) as query:
            record = training.environment('cuda:0')
            self.assertEqual(record['gpu_uuid'], gpu_uuid)
            self.assertEqual(record['visible_gpu_count'], 1)
            self.assertEqual(record['other_compute_processes_visible'], [
                {'pid': 43210, 'gpu_uuid': gpu_uuid, 'used_memory_mib': 64}])
            self.assertNotIn('command', json.dumps(record))
            self.assertEqual(query.call_args.args[0][0], 'nvidia-smi')
            properties.uuid = None
            with self.assertRaisesRegex(ValueError, 'UUID'):
                training.environment('cuda:0')
            with patch.object(torch.cuda, 'device_count', return_value=2):
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    training.environment('cuda:0')

    def test_sequence_multiplier_is_applied_once_with_full_window_denominator(self):
        item, data = runtime(), ToyDataset()
        item.model.eval()
        indices = tuple(range(16))
        batch = data.collate(indices)
        with torch.no_grad():
            nll = data.sequence_nll(item.model(**{key: batch[key] for key in ('input_ids', 'attention_mask', 'position_ids')}).logits, batch)
            expected = float((nll * batch['loss_multipliers']).sum() / 16)
        base = item.model.base.weight.detach().clone()
        result = training._step(item, data, indices)
        self.assertAlmostEqual(result['objective'], expected, places=6)
        self.assertEqual((result['examples'], result['input_tokens'], result['answer_tokens'], result['forward_calls']), (16, 32, 16, 4))
        self.assertTrue(torch.equal(base, item.model.base.weight))

    def test_duplicate_controller_is_rejected_and_lock_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'exclusive'
            with training._run_lock(directory):
                with self.assertRaisesRegex(ValueError, 'trainer holds'):
                    training.train(runtime(), ToyDataset(), run_directory=directory, identity={'fixture': 'exclusive'})
                self.assertFalse(directory.exists())
            result = training.train(runtime(), ToyDataset(), run_directory=directory,
                                    identity={'fixture': 'exclusive'}, stop_after_steps=1)
            self.assertFalse(result['complete'])

    def test_midpass_resume_is_bit_exact_for_dropout_adam_rng_and_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = ToyDataset(); identity = {'fixture': 'matched-resume'}
            whole = runtime()
            expected = training.train(whole, data, run_directory=root / 'whole', identity=identity, checkpoint_every_steps=1)
            split = runtime()
            pause = training.train(split, data, run_directory=root / 'split', identity=identity,
                                   checkpoint_every_steps=1, stop_after_steps=3)
            self.assertFalse(pause['complete'])
            self.assertEqual((pause['counters']['next_pass'], pause['counters']['next_offset']), (1, 16))
            resumed = runtime()
            final = training.train(resumed, data, run_directory=root / 'split', identity=identity, resume=True, checkpoint_every_steps=1)
            self.assertTrue(final['complete'])
            self.assertEqual(final['counters'], expected['counters'])
            self.assertEqual((final['counters']['optimizer_steps'], final['counters']['examples_seen']), (10, 160))
            for name, tensor in training._adapter_state(whole.model).items():
                self.assertTrue(torch.equal(tensor, training._adapter_state(resumed.model)[name]), name)
            self.assertEqual(training.state_fingerprint(whole.optimizer.state_dict()),
                             training.state_fingerprint(resumed.optimizer.state_dict()))
            self.assertEqual(len(list((root / 'split').glob('attempt_*.jsonl'))), 2)
            self.assertEqual(len(list((root / 'split/checkpoints').glob('resume_*'))), 2)
            verified = training.verify_training_run(root / 'split', expected_identity=identity)
            self.assertTrue(verified['all_passed'])
            cost = verified['cost_summary']
            self.assertEqual(cost['attempt_count'], 2)
            self.assertEqual(cost['actual_totals']['examples_attempted'], 160)
            self.assertEqual(cost['actual_totals']['input_tokens_attempted'], 320)
            self.assertEqual(cost['uncommitted_or_replayed_work']['examples'], 0)
            self.assertFalse(cost['wall_is_lower_bound'])

    def test_failed_microbatch_is_durable_and_resume_charges_replayed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'failed-forward'; data = ToyDataset(); item = runtime()
            original = data.sequence_nll
            calls = 0
            def failed(logits, batch):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt('fixture after second forward')
                return original(logits, batch)
            with patch.object(data, 'sequence_nll', side_effect=failed):
                with self.assertRaises(KeyboardInterrupt):
                    training.train(item, data, run_directory=directory, identity={'fixture': 'failed-forward'})
            cost = training.training_cost_summary(directory)
            self.assertEqual(cost['committed_exposure']['optimizer_steps'], 0)
            self.assertEqual(cost['actual_totals']['forward_attempts'], 2)
            self.assertEqual(cost['actual_totals']['backwards_completed'], 1)
            self.assertEqual(cost['actual_totals']['optimizer_attempts'], 0)
            self.assertEqual(cost['actual_totals']['examples_attempted'], 8)
            self.assertEqual(cost['actual_totals']['input_tokens_attempted'], 16)
            self.assertEqual(cost['uncommitted_or_replayed_work']['examples'], 8)
            self.assertFalse(cost['wall_is_lower_bound'])
            receipt = next(directory.glob('attempt_*.receipt.json'))
            original_bytes = receipt.read_bytes()
            result = training.train(runtime(), data, run_directory=directory,
                                    identity={'fixture': 'failed-forward'}, resume=True)
            self.assertEqual(receipt.read_bytes(), original_bytes)
            cost = result['cost_summary']
            self.assertEqual(cost['actual_totals']['examples_attempted'], 168)
            self.assertEqual(cost['uncommitted_or_replayed_work']['examples'], 8)
            self.assertEqual(cost['actual_totals']['optimizer_steps_completed'], 10)
            training.verify_training_run(directory)

    def test_cost_authentication_rejects_changed_receipt_and_preserves_unclosed_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'cost'; data = ToyDataset()
            training.train(runtime(), data, run_directory=directory, identity={'fixture': 'cost'})
            receipt = next(directory.glob('attempt_*.receipt.json'))
            saved = receipt.read_bytes()
            changed = read_json(receipt); changed['cost']['work']['examples_attempted'] += 1
            receipt.write_bytes(canonical_bytes(changed) + b'\n')
            with self.assertRaisesRegex(ValueError, 'receipt differs'):
                training.verify_training_run(directory)
            receipt.write_bytes(saved)
            events = next(directory.glob('attempt_*.jsonl'))
            lines = events.read_bytes().splitlines(keepends=True)
            # Simulate a hard kill after a successful forward intent: durable
            # token counts survive, and elapsed wall is explicitly only a bound.
            unclosed = directory / 'attempt_zz_hardkill.jsonl'
            unclosed.write_bytes(b''.join(lines[:3]))
            summary = training.training_cost_summary(directory)
            self.assertTrue(summary['wall_is_lower_bound'])
            self.assertEqual(summary['uncommitted_or_replayed_work']['examples'], 4)
            self.assertEqual(summary['actual_totals']['forward_attempts'], 41)
            with self.assertRaisesRegex(ValueError, 'cost or event'):
                training.verify_training_run(directory)
            with unclosed.open('ab') as handle:
                handle.write(b'{"event":"microbatch_completed"')
            snapshot = training.training_cost_summary(directory)
            self.assertTrue(snapshot['wall_is_lower_bound'])
            self.assertEqual(snapshot['actual_totals']['forward_attempts'], 41)
            partial = next(row for row in snapshot['attempts'] if row['event_file'] == unclosed.name)
            self.assertEqual(partial['cost']['uncommitted_tail_bytes'], 31)
            with self.assertRaisesRegex(ValueError, 'trailing write'):
                training.training_cost_summary(directory, allow_uncommitted_tail=False)

    def test_checkpoint_hash_corruption_identity_and_cursor_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'; data = ToyDataset(); item = runtime()
            training.train(item, data, run_directory=directory, identity={'fixture': 'bound'}, stop_after_steps=1)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.train(runtime(), data, run_directory=directory, identity={'fixture': 'other'}, resume=True)
            latest = read_json(directory / 'latest.json')
            state = directory / 'checkpoints' / latest['path'] / 'training_state.pt'
            with state.open('ab') as handle:
                handle.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'payload'):
                training.train(runtime(), data, run_directory=directory, identity={'fixture': 'bound'}, resume=True)
            counters = {'next_pass': 0, 'next_offset': 0, 'optimizer_steps': 1, 'examples_seen': 16,
                        'input_tokens_seen': 32, 'answer_tokens_seen': 16}
            with self.assertRaisesRegex(ValueError, 'cursor'):
                training._validate_counters(counters, 32, 5)

    def test_checkpoint_publication_crash_preserves_staging_and_recovers_missing_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = ToyDataset(); directory = root / 'run'
            original = training._atomic_json
            def interrupted(path, value):
                if Path(path).name == 'latest.json':
                    raise KeyboardInterrupt('after checkpoint publication')
                return original(path, value)
            with patch.object(training, '_atomic_json', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    training.train(runtime(), data, run_directory=directory, identity={'fixture': 'pointer'})
            self.assertTrue((directory / 'checkpoints/resume_00000000/manifest.json').is_file())
            self.assertFalse((directory / 'latest.json').exists())
            result = training.train(runtime(), data, run_directory=directory, identity={'fixture': 'pointer'}, resume=True)
            self.assertTrue(result['complete'])
            self.assertTrue((directory / 'latest.json').is_file())
            failed = root / 'failed'
            item = runtime()
            with patch.object(item.model, 'save_pretrained', side_effect=OSError('synthetic serialization failure')):
                with self.assertRaises(OSError):
                    training.train(item, data, run_directory=failed, identity={'fixture': 'partial'})
            self.assertEqual(len(list((failed / 'checkpoints/incomplete').glob('*.partial'))), 1)
            self.assertFalse((failed / 'latest.json').exists())
            recovered = training.train(runtime(), data, run_directory=failed, identity={'fixture': 'partial'}, resume=True)
            self.assertTrue(recovered['complete'])
            self.assertEqual(len(list((failed / 'checkpoints/incomplete').glob('*.partial'))), 1)

    def test_completed_verification_loads_no_torch_and_rejects_final_payload_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            training.train(runtime(), ToyDataset(), run_directory=directory, identity={'fixture': 'stdlib'})
            script = ('import sys; from large_qubit.training import verify_training_run; '
                      f'verify_training_run({str(directory)!r}, expected_identity={{"fixture":"stdlib"}}); '
                      'assert not ({"torch","transformers","peft","pyscf"} & set(sys.modules))')
            subprocess.run([sys.executable, '-B', '-c', script], check=True, cwd=Path(__file__).resolve().parents[2])
            final = directory / 'checkpoints/final/adapter/adapter_model.safetensors'
            with final.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload'):
                training.verify_training_run(directory)

    def test_resident_benchmark_restores_adapter_optimizer_all_rng_and_model_mode(self):
        item, data = runtime(), ToyDataset()
        item.model.train()
        training._step(item, data, tuple(range(16)))  # Populate AdamW state before snapshot.
        before = (training.state_fingerprint(training._adapter_state(item.model)),
                  training.state_fingerprint(item.optimizer.state_dict()), training.state_fingerprint(training.capture_rng()))
        report = training.benchmark(item, data, candidates=((4, 4), (8, 2)),
                                   gradient_checkpointing_options=(False, True), warmup_steps=1, measured_steps=1, execute=True)
        after = (training.state_fingerprint(training._adapter_state(item.model)),
                 training.state_fingerprint(item.optimizer.state_dict()), training.state_fingerprint(training.capture_rng()))
        self.assertEqual(before, after)
        self.assertTrue(item.model.training)
        self.assertFalse(item.model.is_gradient_checkpointing)
        self.assertTrue(report['adapter_optimizer_rng_restored'])
        self.assertEqual(len(report['candidates']), 4)
        self.assertTrue(all(row['success'] for row in report['candidates']))
        self.assertEqual({row['numerical_configuration']['gradient_checkpointing'] for row in report['candidates']}, {False, True})
        self.assertTrue(all(row['measured_counts']['examples'] == 16 for row in report['candidates']))
        self.assertEqual(report['production_optimizer_steps'], 0)
        self.assertEqual(report['disposable_optimizer_step_attempts'], 8)
        self.assertEqual(report['disposable_optimizer_steps_completed'], 8)

    def test_benchmark_stress_warmup_precedes_unskipped_balanced_measurements(self):
        item, data = runtime(), ToyDataset()
        stress, observed = tuple(range(31, 15, -1)), []
        original = training._step
        def counted(runtime, dataset, indices, **kwargs):
            observed.append(tuple(indices))
            return original(runtime, dataset, indices, **kwargs)
        with patch.object(data, 'memory_stress_indices', return_value=stress, create=True), \
             patch.object(training, '_step', side_effect=counted):
            report = training.benchmark(item, data, candidates=((4, 4),),
                gradient_checkpointing_options=(False,), warmup_steps=1, measured_steps=2, execute=True)
        self.assertEqual(observed, [stress, data.row_order(0)[:16], data.row_order(0)[16:]])
        self.assertEqual(report['memory_stress_row_indices'], list(stress))
        self.assertNotIn('fixture', report['memory_stress_policy'])

    def test_memory_headroom_includes_larger_stress_warmup_peak(self):
        item, data = runtime(), ToyDataset(); item.device = 'cuda:0'
        peak = {'bytes': 100, 'steps': 0}
        def reset(*args):
            peak['bytes'] = 100
        def step(runtime, dataset, indices, **kwargs):
            peak['steps'] += 1
            peak['bytes'] = max(peak['bytes'], 900 if peak['steps'] == 1 else 500)
            return {'examples': 16, 'input_tokens': 32, 'answer_tokens': 16, 'forward_calls': 4}
        with patch.object(training, '_step', side_effect=step), \
             patch.object(training, '_sync'), \
             patch.object(training, 'environment', return_value={'fixture_only': True}), \
             patch.object(torch.cuda, 'mem_get_info', return_value=(1000, 2000)), \
             patch.object(torch.cuda, 'memory_reserved', return_value=100), \
             patch.object(torch.cuda, 'reset_peak_memory_stats', side_effect=reset), \
             patch.object(torch.cuda, 'max_memory_allocated', side_effect=lambda *args: peak['bytes']), \
             patch.object(torch.cuda, 'max_memory_reserved', side_effect=lambda *args: peak['bytes']), \
             patch.object(torch.cuda, 'empty_cache'):
            report = training.benchmark(item, data, candidates=((4, 4),),
                gradient_checkpointing_options=(False,), warmup_steps=1, measured_steps=1, execute=True)
        candidate = report['candidates'][0]
        self.assertTrue(candidate['success'])
        self.assertEqual(candidate['stress_warmup_peak_reserved_bytes'], 900)
        self.assertEqual(candidate['peak_reserved_bytes'], 900)
        self.assertEqual(candidate['usable_memory_budget_bytes'], 1100)
        self.assertFalse(candidate['memory_headroom_passed'])

    def test_failed_benchmark_candidate_also_restores_state_and_requires_explicit_execute(self):
        item, data = runtime(), ToyDataset()
        before = training.state_fingerprint(training.capture_rng())
        with self.assertRaisesRegex(ValueError, 'explicit'):
            training.benchmark(item, data)
        with patch.object(data, 'sequence_nll', side_effect=RuntimeError('fixture bad batch')):
            report = training.benchmark(item, data, candidates=((4, 4),),
                gradient_checkpointing_options=(False,), warmup_steps=1, measured_steps=1, execute=True)
        self.assertFalse(report['candidates'][0]['success'])
        self.assertEqual(report['disposable_optimizer_step_attempts'], 1)
        self.assertEqual(report['disposable_optimizer_steps_completed'], 0)
        self.assertEqual(report['disposable_actual_work']['forward_attempts'], 1)
        self.assertEqual(report['disposable_actual_work']['input_tokens_attempted'], 8)
        self.assertEqual(report['disposable_actual_work']['answer_tokens_attempted'], 4)
        self.assertEqual(report['disposable_actual_work']['backwards_completed'], 0)
        self.assertGreater(report['candidates'][0]['wall_seconds_including_warmup_and_failed_work'], 0)
        self.assertEqual(training.state_fingerprint(training.capture_rng()), before)
        self.assertTrue(report['adapter_optimizer_rng_restored'])


if __name__ == '__main__':
    unittest.main()
