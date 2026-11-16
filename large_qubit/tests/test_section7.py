"""Section7 control/selection guards using synthetic reports, never GPU work."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from large_qubit import section7
from large_qubit.common import canonical_bytes, read_json, sha256_file
from large_qubit.training import numerical_configuration


WORKSPACE = Path(__file__).resolve().parents[2]
FITS = section7.FITS


def candidate(batch=4, accumulation=4, *, checkpointing=False, rate=100., success=True, headroom=True):
    numeric = numerical_configuration(batch, accumulation, gradient_checkpointing=checkpointing)
    base = {'numerical_configuration': numeric, 'success': success,
        'memory_headroom_passed': headroom if success else False,
        'initial_free_bytes': 800, 'initial_reserved_bytes': 200,
        'usable_memory_budget_bytes': 1000, 'required_memory_headroom_fraction': .2,
        'optimizer_step_attempts': 4 if success else 1,
        'optimizer_steps_completed': 4 if success else 0,
        'warmup_optimizer_steps_completed': 1 if success else 0,
        'measured_optimizer_steps_completed': 3 if success else 0}
    if not success:
        return {**base, 'out_of_memory': True, 'error_type': 'OutOfMemoryError'}
    seconds = 48/rate
    return {**base, 'examples_per_second': rate, 'measured_seconds': seconds,
        'input_tokens_per_second': 480/seconds,
        'measured_counts': {'examples': 48, 'input_tokens': 480, 'answer_tokens': 192,
                            'forward_calls': 3*accumulation},
        'peak_allocated_bytes': 450 if headroom else 850,
        'peak_reserved_bytes': 500 if headroom else 900,
        'free_vram_bytes_after_measurement': 500, 'total_vram_bytes': 2000}


def selection_reports(rows=None):
    rows = [candidate(), candidate(8, 2, rate=110.)] if rows is None else rows
    return {fit: {'benchmark': {'candidates': deepcopy(rows)}} for fit in FITS}


class SelectionTests(unittest.TestCase):
    def test_GPT_pair_chooses_fastest_slower_paired_rate_and_Qwen_independently(self):
        reports = selection_reports()
        for fit, rates in zip(FITS, ((200., 100.), (10., 90.), (70., 50.))):
            reports[fit]['benchmark']['candidates'] = [candidate(rate=rates[0]), candidate(8, 2, rate=rates[1])]
        selected = section7.select_numerical_configurations(reports)
        self.assertEqual(selected[FITS[0]], selected[FITS[1]])
        self.assertEqual(selected[FITS[0]]['microbatch_size'], 8)
        self.assertEqual(selected[FITS[2]]['microbatch_size'], 4)

    def test_full_gradient_checkpointing_configuration_must_match_across_GPT_pair(self):
        reports = selection_reports()
        reports[FITS[0]]['benchmark']['candidates'] = [candidate(checkpointing=False)]
        reports[FITS[1]]['benchmark']['candidates'] = [candidate(checkpointing=True)]
        with self.assertRaisesRegex(ValueError, 'common'):
            section7.select_numerical_configurations(reports)

    def test_declared_order_breaks_throughput_ties(self):
        selected = section7.select_numerical_configurations(selection_reports([candidate(), candidate(8, 2)]))
        self.assertTrue(all(value['microbatch_size'] == 4 for value in selected.values()))

    def test_OOM_and_insufficient_headroom_are_not_selectable(self):
        rows = [candidate(rate=1000., headroom=False), candidate(8, 2, success=False), candidate(1, 16, rate=20.)]
        selected = section7.select_numerical_configurations(selection_reports(rows))
        self.assertTrue(all(value['microbatch_size'] == 1 for value in selected.values()))
        with self.assertRaisesRegex(ValueError, 'headroom'):
            section7.select_numerical_configurations(selection_reports([candidate(headroom=False)]))

    def test_invalid_nonfinite_and_undeclared_candidates_are_rejected(self):
        changes = [lambda row: row.update(examples_per_second=float('nan')),
            lambda row: row.update(examples_per_second=float('inf')),
            lambda row: row.update(examples_per_second=True),
            lambda row: row.update(examples_per_second=0.),
            lambda row: row['numerical_configuration'].update(microbatch_size=-4, gradient_accumulation=-4),
            lambda row: row['numerical_configuration'].update(microbatch_size=2, gradient_accumulation=8),
            lambda row: row['numerical_configuration'].update(gradient_checkpointing=0),
            lambda row: row['numerical_configuration'].update(precision='float32'),
            lambda row: row['numerical_configuration'].update(optimizer_fused=True),
            lambda row: row.update(success=1), lambda row: row.update(memory_headroom_passed=1)]
        for change in changes:
            reports = selection_reports(); change(reports[FITS[0]]['benchmark']['candidates'][0])
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                section7.select_numerical_configurations(reports)

    def test_duplicate_configs_including_failed_candidates_are_rejected(self):
        for duplicate in (candidate(), candidate(success=False)):
            with self.subTest(success=duplicate['success']), self.assertRaisesRegex(ValueError, 'Duplicate'):
                section7.select_numerical_configurations(selection_reports([candidate(), duplicate]))

    def test_reported_headroom_and_throughput_must_match_raw_measurement(self):
        changes = [lambda row: row.update(peak_reserved_bytes=900),
            lambda row: row.update(usable_memory_budget_bytes=2000),
            lambda row: row.update(measured_seconds=float('nan')),
            lambda row: row.update(measured_seconds=0.),
            lambda row: row.update(measured_seconds=row['measured_seconds']*2),
            lambda row: row['measured_counts'].update(examples=32)]
        for index, change in enumerate(changes):
            reports = selection_reports(); change(reports[FITS[0]]['benchmark']['candidates'][0])
            with self.subTest(change=index), self.assertRaises(ValueError):
                section7.select_numerical_configurations(reports)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write('source.py', b'fixed source\n')
        self.write(section7.PRIOR_REGISTRATION, b'{"synthetic":"prior registration"}\n')
        self.write(section7.CACHE_AUDIT, b'{"synthetic":"deep native cache audit"}\n')
        pin = sha256_file(self.root / section7.PRIOR_REGISTRATION)
        audit_pin = sha256_file(self.root / section7.CACHE_AUDIT)
        self.cache_pins = {key:('a' if key == 'gpt2_xl' else 'b')*64 for key in section7.MODEL_REVISIONS}
        for context in (patch.object(section7, 'SOURCES', ('source.py', section7.POLICY)),
                        patch.object(section7, 'PRIOR_REGISTRATION_SHA256', pin),
                        patch.object(section7, 'CACHE_AUDIT_SHA256', audit_pin),
                        patch.object(section7, 'TOKEN_CACHE_SHA256', self.cache_pins)):
            context.start(); self.addCleanup(context.stop)
        self.write(section7.POLICY, canonical_bytes(section7.build_section7_policy()))
        self.cache_verifier = Mock(return_value={'all_passed': True})
        context = patch('large_qubit.training_cache_access.verify_training_cache', self.cache_verifier)
        context.start(); self.addCleanup(context.stop)
        self.record = {'schema_version': 'section7-preparation-v1', 'created_utc': 'synthetic',
            'policy_path': section7.POLICY, 'policy_sha256': sha256_file(self.root / section7.POLICY),
            'source_bindings': section7._sources(self.root), 'release_sha256': section7.RELEASE_SHA256,
            'deep_native_cache_audit': {'path': section7.CACHE_AUDIT, 'sha256': audit_pin},
            'token_caches': {key: {'manifest_path': f'{section7.CACHE_ROOT}/{key}/manifest.json',
                                  'manifest_sha256': ('a' if key == 'gpt2_xl' else 'b')*64}
                             for key in section7.MODEL_REVISIONS},
            'production_model_weights_loaded': False, 'production_training_executed': False,
            'production_gpu_preflight_executed': False, 'energy_requests': 0,
            'prior_registration_sha256': pin}
        self.save()

    def write(self, relative, data):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

    def save(self): self.write(section7.PREPARATION, canonical_bytes(self.record))

    def test_verify_preparation_authenticates_two_caches_without_production_work(self):
        report = section7.verify_section7(self.root)
        self.assertTrue(report['implementation_ready'])
        self.assertFalse(report['production_training_executed'])
        self.assertFalse(report['production_gpu_preflight_executed'])
        self.assertEqual(report['energy_requests'], 0)
        self.assertEqual(self.cache_verifier.call_count, 2)

    def test_changed_prepared_source_or_policy_is_rejected(self):
        self.write('source.py', b'changed source\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            section7.verify_section7(self.root)
        self.write('source.py', b'fixed source\n')
        self.write(section7.POLICY, b'{}')
        with self.assertRaisesRegex(ValueError, 'source changed|policy changed'):
            section7.verify_section7(self.root)

    def test_scope_cache_and_prior_identity_tampering_are_rejected(self):
        for key, value in (('production_training_executed', True), ('production_model_weights_loaded', True),
                           ('production_gpu_preflight_executed', True), ('energy_requests', True),
                           ('prior_registration_sha256', '0'*64), ('release_sha256', '0'*64)):
            original = deepcopy(self.record); self.record[key] = value; self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): section7.verify_section7(self.root)
            self.record = original
        self.record['token_caches']['gpt2_xl']['manifest_path'] = f'{section7.CACHE_ROOT}/qwen3_4b/manifest.json'; self.save()
        with self.assertRaisesRegex(ValueError, 'cache path'): section7.verify_section7(self.root)

    def test_prepared_cache_reuse_verifies_each_cache_without_tokenizer_preparation(self):
        (self.root/section7.PREPARATION).unlink()  # Synthetic preparation has not yet been committed.
        paths = []
        for key in section7.MODEL_REVISIONS:
            relative = f'{section7.CACHE_ROOT}/{key}/manifest.json'
            self.write(relative, canonical_bytes({'synthetic_model_cache':key})); paths.append(relative)
            self.cache_pins[key] = sha256_file(self.root/relative)
        self.write(section7.POLICY, canonical_bytes(section7.build_section7_policy()))
        before = {name:(sha256_file(self.root/name),(self.root/name).stat().st_mtime_ns) for name in paths}
        with patch('large_qubit.training_data.prepare_training_data', side_effect=AssertionError('Retokenized fitting data')) as prepare, \
             patch('large_qubit.training_data.load_prompt_tokenizer', side_effect=AssertionError('Loaded tokenizer')) as tokenizer:
            result = section7.prepare_section7(self.root)
        self.assertTrue(result['all_passed'])
        self.assertEqual({call.args[1] for call in self.cache_verifier.call_args_list}, set(paths))
        self.assertGreaterEqual(self.cache_verifier.call_count, 2)
        prepare.assert_not_called(); tokenizer.assert_not_called()
        self.assertEqual(before, {name:(sha256_file(self.root/name),(self.root/name).stat().st_mtime_ns) for name in paths})

    def test_existing_cache_verification_failure_cannot_trigger_retokenization(self):
        (self.root/section7.PREPARATION).unlink()
        self.write(f'{section7.CACHE_ROOT}/gpt2_xl/manifest.json', b'{"synthetic":"cache"}')
        self.cache_verifier.return_value = {'all_passed':False}
        with patch('large_qubit.training_data.prepare_training_data') as prepare:
            with self.assertRaisesRegex(ValueError, 'cache failed authentication'):
                section7.prepare_section7(self.root)
        prepare.assert_not_called()
        self.assertFalse((self.root/section7.PREPARATION).exists())


class FreezeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.preparation = {'preparation_sha256': 'c'*64,
            'token_caches': {key: {'manifest_path': f'{section7.CACHE_ROOT}/{key}/manifest.json',
                                  'manifest_sha256': ('a' if key == 'gpt2_xl' else 'b')*64}
                             for key in section7.MODEL_REVISIONS}}
        context = patch.object(section7, 'verify_section7', return_value=self.preparation)
        context.start(); self.addCleanup(context.stop)
        self.reports = {}
        for index, fit in enumerate(FITS):
            model = section7.training_conditions()[fit]['model_key']
            records = [candidate(batch, accumulation, checkpointing=checkpointing, rate=100.+ordinal)
                for checkpointing in (False, True)
                for ordinal, (batch, accumulation) in enumerate(((4,4),(8,2),(16,1),(1,16)))]
            runtime = {'python': 'synthetic-python', 'packages': {'torch': '2.10.0', 'peft': '0.18.1'},
                'torch_cuda_version': 'synthetic-cuda', 'cudnn_version': 1, 'compute_capability': [9,0],
                'device': 'cuda:0', 'fixture_only': False, 'visible_gpu_count': 1, 'native_bf16': True,
                'device_name': 'NVIDIA H200', 'gpu_uuid': f'GPU-synthetic-{index}', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            benchmark = {'schema_version': 'section7-resident-benchmark-v1', 'model_key': model,
                'condition_id': fit, 'warmup_steps_per_candidate': 1, 'measured_steps_per_candidate': 3,
                'runtime': runtime, 'candidates': records, 'base_model_resident_between_candidates': True,
                'adapter_optimizer_rng_restored': True, 'restored_state_sha256': [('e' if index < 2 else 'f')*64, '1'*64, '2'*64],
                'additional_energy_requests': 0, 'production_optimizer_steps': 0}
            self.reports[fit] = {'schema_version': 'section7-production-preflight-v1', 'condition_id': fit,
                'requested_device': 'cuda:0', 'preparation_sha256': self.preparation['preparation_sha256'],
                'cache_binding': self.preparation['token_caches'][model],
                'production_gpu_preflight_executed': True, 'final_fit_training_executed': False, 'benchmark': benchmark}
        self.save_reports()

    def write(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical_bytes(value))

    def save_reports(self):
        for fit, report in self.reports.items(): self.write(section7._preflight_path(section7.PREFLIGHT_ROOT, fit), report)

    def test_freeze_uses_distinct_UUIDs_with_same_visible_cuda0_and_is_idempotent(self):
        result = section7.freeze_section7(self.root)
        self.assertTrue(result['all_passed'])
        self.assertEqual(set(result['freeze']['devices'].values()), {'cuda:0'})
        self.assertEqual(len(set(result['freeze']['gpu_uuids'].values())), 3)
        path = self.root / result['freeze_path']; before = (sha256_file(path), path.stat().st_mtime_ns)
        self.assertEqual(section7.freeze_section7(self.root), result)
        self.assertEqual((sha256_file(path), path.stat().st_mtime_ns), before)

    def test_reused_GPU_UUID_or_mismatched_GPT_adapter_or_packages_refuse_freeze(self):
        changes = [lambda reports: reports[FITS[1]]['benchmark']['runtime'].update(gpu_uuid='GPU-synthetic-0'),
            lambda reports: reports[FITS[1]]['benchmark']['restored_state_sha256'].__setitem__(0, '0'*64),
            lambda reports: reports[FITS[2]]['benchmark']['runtime']['packages'].update(torch='different')]
        originals = deepcopy(self.reports)
        for index, change in enumerate(changes):
            self.reports = deepcopy(originals); change(self.reports); self.save_reports()
            with self.subTest(change=index), self.assertRaises(ValueError): section7.freeze_section7(self.root)

    def test_report_bytes_and_frozen_exposure_tampering_are_detected(self):
        saved = section7.freeze_section7(self.root)
        path = self.root / section7._preflight_path(section7.PREFLIGHT_ROOT, FITS[0]); original = path.read_bytes()
        path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'changed after freezing'): section7.verify_freeze(self.root)
        path.write_bytes(original)
        frozen_path = self.root / saved['freeze_path']; frozen = read_json(frozen_path)
        frozen['passes'] = 4; self.write(saved['freeze_path'], frozen)
        with self.assertRaisesRegex(ValueError, 'exposure'): section7.verify_freeze(self.root)

    def test_missing_or_reordered_prescribed_candidates_and_CPU_reports_are_rejected(self):
        for change in (lambda record: record['benchmark']['candidates'].pop(),
                       lambda record: record['benchmark']['candidates'].reverse(),
                       lambda record: record['benchmark']['runtime'].update(fixture_only=True),
                       lambda record: record['benchmark']['runtime'].update(gpu_uuid='unavailable')):
            record = deepcopy(self.reports[FITS[0]]); change(record)
            with self.assertRaises(ValueError): section7._verify_preflight(record, FITS[0], self.preparation)

    def test_training_rejects_other_UUID_environment_before_any_optimizer_work(self):
        section7.freeze_section7(self.root)
        with patch.object(section7, '_dataset', return_value=(section7.training_conditions()[FITS[0]], object())), \
             patch('large_qubit.training.environment', return_value={
                 **self.reports[FITS[0]]['benchmark']['runtime'], 'gpu_uuid': 'GPU-wrong'}), \
             patch('large_qubit.training.load_runtime') as loader, \
             patch('large_qubit.training.train') as train:
            with self.assertRaisesRegex(ValueError, 'environment differs'): section7.run_fit(self.root, FITS[0])
        loader.assert_not_called()
        train.assert_not_called()

    def test_GPU_UUID_lock_refuses_preflight_before_model_loading_or_benchmark(self):
        environment = self.reports[FITS[0]]['benchmark']['runtime']
        with section7._operation_lock(self.root, 'gpu_'+environment['gpu_uuid']), \
             patch.object(section7, '_dataset', return_value=(section7.training_conditions()[FITS[0]], object())), \
             patch('large_qubit.training.environment', return_value=environment), \
             patch('large_qubit.training.load_runtime') as loader, \
             patch('large_qubit.training.benchmark') as benchmark:
            with self.assertRaisesRegex(ValueError, 'lock'):
                section7.run_preflight(self.root, FITS[0], preflight_root=section7.PREFLIGHT_ROOT+'/fresh')
        loader.assert_not_called(); benchmark.assert_not_called()

    def test_GPU_UUID_lock_refuses_training_before_model_loading(self):
        section7.freeze_section7(self.root)
        environment = self.reports[FITS[0]]['benchmark']['runtime']
        with section7._operation_lock(self.root, 'gpu_'+environment['gpu_uuid']), \
             patch.object(section7, '_dataset', return_value=(section7.training_conditions()[FITS[0]], object())), \
             patch('large_qubit.training.environment', return_value=environment), \
             patch('large_qubit.training.load_runtime') as loader, \
             patch('large_qubit.training.train') as train:
            with self.assertRaisesRegex(ValueError, 'lock'): section7.run_fit(self.root,FITS[0])
        loader.assert_not_called(); train.assert_not_called()


class OperationLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_shared_fit_lock_refuses_both_public_actions_before_internal_callbacks(self):
        with section7._operation_lock(self.root,'fit_'+FITS[0]), \
             patch.object(section7,'_run_preflight_locked') as preflight, \
             patch.object(section7,'_run_fit_locked') as fit, \
             patch('large_qubit.training.load_runtime') as loader:
            for action in (section7.run_preflight,section7.run_fit):
                with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError,'lock'):
                    action(self.root,FITS[0])
        preflight.assert_not_called();fit.assert_not_called();loader.assert_not_called()

    def test_preflight_holds_same_fit_lock_as_training_and_releases_on_exception(self):
        def callback(*args,**kwargs):
            with self.assertRaisesRegex(ValueError,'lock'): section7.run_fit(self.root,FITS[0])
            raise RuntimeError('synthetic callback failure')
        with patch.object(section7,'_run_preflight_locked',side_effect=callback), \
             patch.object(section7,'_run_fit_locked') as fit:
            with self.assertRaisesRegex(RuntimeError,'synthetic callback failure'):
                section7.run_preflight(self.root,FITS[0])
            fit.assert_not_called()
        with patch.object(section7,'_run_fit_locked',return_value={'synthetic':'released'}) as fit:
            self.assertEqual(section7.run_fit(self.root,FITS[0]),{'synthetic':'released'})
            fit.assert_called_once()

    def test_another_process_holding_fit_lock_blocks_before_any_callback(self):
        script = """from pathlib import Path
import sys
from large_qubit.section7 import _operation_lock
root=Path(sys.argv[1])
with _operation_lock(root,'fit_'+sys.argv[2]):
    (root/'lock_ready').write_text('ready',encoding='ascii')
    sys.stdin.readline()
"""
        child = subprocess.Popen([sys.executable,'-B','-c',script,str(self.root),FITS[0]],
                                 cwd=WORKSPACE,text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+10
            while not (self.root/'lock_ready').exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue((self.root/'lock_ready').is_file(),'Synthetic lock worker did not become ready')
            with patch.object(section7,'_run_preflight_locked') as preflight, \
                 patch.object(section7,'_run_fit_locked') as fit, \
                 patch('large_qubit.training.load_runtime') as loader:
                for action in (section7.run_preflight,section7.run_fit):
                    with self.assertRaisesRegex(ValueError,'lock'): action(self.root,FITS[0])
            preflight.assert_not_called();fit.assert_not_called();loader.assert_not_called()
        finally:
            if child.poll() is None:
                try: child.communicate(input='\n',timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill();child.communicate(timeout=5)
            else: child.communicate(timeout=5)
        self.assertEqual(child.returncode,0)
        with patch.object(section7,'_run_fit_locked',return_value={'synthetic':'released'}):
            self.assertEqual(section7.run_fit(self.root,FITS[0]),{'synthetic':'released'})


class PreviewTests(unittest.TestCase):
    def test_all_mutating_commands_preview_without_execute_and_resume_is_explicit(self):
        calls = ('prepare_section7', 'run_preflight', 'freeze_section7', 'run_fit')
        contexts = [patch.object(section7, name, side_effect=AssertionError('Executed '+name)) for name in calls]
        for context in contexts: context.start(); self.addCleanup(context.stop)
        for arguments in (['plan'], ['prepare'], ['preflight','--condition',FITS[0]], ['freeze'], ['train','--condition',FITS[0]]):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(section7.main(arguments), 0)
            result = json.loads(output.getvalue()); self.assertEqual(result['mode'], 'preview_only')
        with redirect_stderr(io.StringIO()):
            self.assertEqual(section7.main(['train','--condition',FITS[0],'--resume']), 1)

    def test_fresh_plan_import_does_not_import_Torch_numpy_or_model_libraries(self):
        script = """import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy','transformers','peft'):
        raise AssertionError('Forbidden execution dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from large_qubit.section7 import main
assert main(['plan'])==0
print('IMPORT_GUARD_PASSED')
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=WORKSPACE,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('IMPORT_GUARD_PASSED', result.stdout)


if __name__ == '__main__': unittest.main()
