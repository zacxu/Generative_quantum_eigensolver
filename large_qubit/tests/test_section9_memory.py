"""Synthetic memory and fake scorer checks; no molecular evaluation or models."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from large_qubit import section9_memory as memory
from large_qubit import section9_recovery_scoring as recovery
from large_qubit import section9_scoring as original
from large_qubit.tests.test_section9_scoring import ROOT, FREEZE_SHA, identity, scorer


def snapshot(peak=123456789, raw=8 * 1024**3):
    return {'schema_version': memory.SCHEMA, 'source': memory.POLICY['guard_source'],
            'peak_rss_bytes': peak, 'current_rss_bytes': min(peak, 45678912),
            'raw_getrusage_peak_rss_bytes': raw}


def fake_scorer():
    value = recovery.TargetScorer.__new__(recovery.TargetScorer)
    value.__dict__.update(scorer().__dict__)
    return value


class MemoryTests(unittest.TestCase):
    def test_units_and_order(self):
        self.assertEqual(memory.parse_status('Name: fixture\nVmRSS: 2048 kB\nVmHWM: 4096 kB\n'),
                         {'current_rss_bytes': 2097152, 'peak_rss_bytes': 4194304})

    def test_missing_malformed_and_inconsistent_measurements_refused(self):
        for text in ('', 'VmHWM: 10 kB', 'VmHWM: 10 MB\nVmRSS: 2 kB',
                     'VmHWM: -10 kB\nVmRSS: 2 kB', 'VmHWM: 0 kB\nVmRSS: 0 kB',
                     'VmHWM: 1 kB\nVmRSS: 2 kB', 'VmHWM: 10 kB\nVmHWM: 10 kB\nVmRSS: 2 kB'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                memory.parse_status(text)

    def test_linux_required_not_zero_fallback(self):
        with patch.object(memory.sys, 'platform', 'win32'), self.assertRaisesRegex(RuntimeError, 'Linux'):
            memory.memory_snapshot()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux /proc fixture')
    def test_missing_proc_is_fatal(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError('fixture')), \
             self.assertRaises(FileNotFoundError):
            memory.memory_snapshot()

    def test_high_inherited_peak_does_not_change_own_peak(self):
        with patch.object(memory, 'memory_snapshot', return_value=snapshot()):
            fields = memory.enforce_memory_budget(original.MAX_RSS_BYTES)
        self.assertEqual(fields['peak_rss_bytes'], 123456789)
        self.assertEqual(fields['raw_getrusage_peak_rss_bytes'], 8 * 1024**3)
        self.assertTrue(fields['memory_accounting']['guard_passed'])
        self.assertFalse(memory.POLICY['baseline_subtraction'])

    def test_own_peak_still_enforces_exact_four_gib(self):
        with patch.object(memory, 'memory_snapshot', return_value=snapshot(peak=original.MAX_RSS_BYTES)):
            self.assertTrue(memory.enforce_memory_budget(original.MAX_RSS_BYTES)['memory_accounting']['guard_passed'])
        with patch.object(memory, 'memory_snapshot', return_value=snapshot(peak=original.MAX_RSS_BYTES + 1)), \
             self.assertRaisesRegex(ValueError, '4 GiB') as caught:
            memory.enforce_memory_budget(original.MAX_RSS_BYTES)
        self.assertEqual(caught.exception.report['peak_rss_bytes'], original.MAX_RSS_BYTES + 1)
        self.assertFalse(caught.exception.report['memory_accounting']['guard_passed'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux fork/exec highwater regression')
    def test_real_exec_child_ignores_inherited_parent_peak_but_retains_own_peak(self):
        # The outer process keeps test runners small. Its inner Popen uses exactly
        # the production PhysicsClient arguments; only 192 MiB + 16 MiB allocated.
        child = '''import gc, json
from large_qubit.section9_memory import memory_snapshot
before = memory_snapshot()
payload = bytearray(16 * 1024**2)
during = memory_snapshot()
del payload
gc.collect()
after = memory_snapshot()
print(json.dumps({'before': before, 'during': during, 'after': after}))
'''
        parent = '''import json, os, resource, subprocess, sys, tempfile
payload = bytearray(192 * 1024**2)
with tempfile.TemporaryFile(mode='a+t', encoding='utf-8', errors='replace') as err:
    proc = subprocess.Popen([sys.executable, '-B', '-c', CHILD], cwd=os.getcwd(), env=dict(os.environ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, encoding='utf-8', bufsize=1)
    stdout, _ = proc.communicate(timeout=30)
    err.seek(0)
    assert proc.returncode == 0, err.read()
    print(json.dumps({'parent_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'child': json.loads(stdout)}))
'''.replace('CHILD', repr(child))
        run = subprocess.run([sys.executable, '-B', '-c', parent], cwd=ROOT, text=True,
                             capture_output=True, timeout=40, check=True)
        value = json.loads(run.stdout)
        initial, during, after = (value['child'][key] for key in ('before', 'during', 'after'))
        self.assertGreater(value['parent_ru_maxrss_bytes'], 192 * 1024**2)
        self.assertGreater(initial['raw_getrusage_peak_rss_bytes'], 192 * 1024**2)
        self.assertLess(initial['peak_rss_bytes'], 96 * 1024**2)
        self.assertGreater(during['peak_rss_bytes'], initial['peak_rss_bytes'] + 8 * 1024**2)
        self.assertGreaterEqual(after['peak_rss_bytes'], during['peak_rss_bytes'])
        self.assertLess(after['current_rss_bytes'], after['peak_rss_bytes'])


class ScorerTests(unittest.TestCase):
    def test_reference_identity_only_allows_additive_memory_instrumentation(self):
        import copy
        saved = identity()
        saved['source_bindings'] = {'original.py': 'a' * 64}
        current = copy.deepcopy(saved)
        current['memory_accounting_policy'] = dict(memory.POLICY)
        current['memory_accounting_at_startup'] = memory.memory_fields(original.MAX_RSS_BYTES, snapshot=snapshot())
        current['source_bindings'].update({path: 'b' * 64 for path in recovery.SOURCES})
        self.assertTrue(recovery.compatible_reference_identity(current, saved))
        self.assertIn(recovery.SOURCES[0], current['source_bindings'])
        for field, value in (('partition', 'training'), ('geometry_id', 'CO2_I02'),
                             ('final_freeze_sha256', '0' * 64)):
            changed = copy.deepcopy(current)
            changed[field] = value
            self.assertFalse(recovery.compatible_reference_identity(changed, saved), field)
        changed = copy.deepcopy(current)
        changed['source_bindings']['original.py'] = 'c' * 64
        self.assertFalse(recovery.compatible_reference_identity(changed, saved))
        changed = copy.deepcopy(current)
        changed['memory_accounting_at_startup']['memory_accounting']['guard_passed'] = False
        self.assertFalse(recovery.compatible_reference_identity(changed, saved))

    def test_hf_passes_with_large_inherited_peak(self):
        with patch.object(memory, 'memory_snapshot', return_value=snapshot()):
            result = fake_scorer().score({'kind': 'hf'})
        self.assertTrue(result['physical_evaluation_performed'])
        self.assertEqual(result['evaluator_calls'], 1)
        self.assertEqual(result['peak_rss_bytes'], 123456789)
        self.assertEqual(result['raw_getrusage_peak_rss_bytes'], 8 * 1024**3)

    def test_scientific_fields_match_original_fake_scorer(self):
        with patch('large_qubit.benchmark._peak_rss_bytes', return_value=123456789):
            old = scorer().score({'kind': 'hf'})
        with patch.object(memory, 'memory_snapshot', return_value=snapshot()):
            new = fake_scorer().score({'kind': 'hf'})
        ignored = {'state_preparation_seconds', 'wall_seconds'}
        for key in set(old) - ignored:
            self.assertEqual(old[key], new[key], key)

    def test_over_limit_failure_preserves_numeric_measurements_and_work(self):
        with patch.object(memory, 'memory_snapshot', return_value=snapshot(peak=original.MAX_RSS_BYTES + 1)), \
             self.assertRaises(recovery.ScoringError) as caught:
            fake_scorer().score({'kind': 'hf'})
        report = caught.exception.report
        self.assertEqual(report['evaluator_calls'], 1)
        self.assertEqual(report['peak_rss_bytes'], original.MAX_RSS_BYTES + 1)
        self.assertEqual(report['raw_getrusage_peak_rss_bytes'], 8 * 1024**3)
        self.assertFalse(report['memory_accounting']['guard_passed'])
        json.dumps(report, allow_nan=False)

    def test_other_scoring_failures_also_capture_memory(self):
        with patch.object(memory, 'memory_snapshot', return_value=snapshot()), \
             self.assertRaises(recovery.ScoringError) as caught:
            fake_scorer().score({'kind': 'bad'})
        self.assertEqual(caught.exception.report['peak_rss_bytes'], 123456789)
        self.assertEqual(caught.exception.report['evaluator_calls'], 0)

    def test_missing_measurement_fails_closed_with_partial_work(self):
        with patch.object(memory, 'memory_snapshot', side_effect=OSError('fixture proc unavailable')), \
             self.assertRaises(recovery.ScoringError) as caught:
            fake_scorer().score({'kind': 'hf'})
        self.assertTrue(caught.exception.report['memory_measurement_unavailable'])
        self.assertEqual(caught.exception.report['evaluator_calls'], 1)

    def test_reference_fake_uses_same_settings_and_memory_policy(self):
        report = {'all_passed': True, 'converged': True, 'independent_starts': 2}
        value = fake_scorer()
        with patch('large_qubit.reference.solve_reference', return_value={'energy': -3., 'report': report}) as solve, \
             patch.object(memory, 'memory_snapshot', return_value=snapshot()):
            result = value.reference()
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                value.reference()
        self.assertEqual(solve.call_args.kwargs, original.REFERENCE_SETTINGS)
        self.assertEqual(result['ordinary_energy_requests'], 0)
        self.assertEqual(result['raw_getrusage_peak_rss_bytes'], 8 * 1024**3)

    def test_reference_memory_failure_keeps_convergence_and_rss(self):
        report = {'all_passed': True, 'converged': True, 'independent_starts': 2}
        with patch('large_qubit.reference.solve_reference', return_value={'energy': -3., 'report': report}), \
             patch.object(memory, 'memory_snapshot', return_value=snapshot(peak=original.MAX_RSS_BYTES + 1)), \
             self.assertRaises(recovery.ScoringError) as caught:
            fake_scorer().reference()
        self.assertTrue(caught.exception.report['converged'])
        self.assertFalse(caught.exception.report['memory_accounting']['guard_passed'])

    def test_probe_and_preview_do_not_run_chemistry(self):
        with patch.object(recovery.subprocess, 'Popen') as popen, \
             patch.object(recovery, 'TargetScorer') as factory, redirect_stdout(io.StringIO()) as out:
            recovery.main(['--probe'])
            client = recovery.PhysicsClient(ROOT, sys.executable, 'CO2_I01')
            client.close()
        popen.assert_not_called()
        factory.assert_not_called()
        self.assertFalse(json.loads(out.getvalue())['scientific_computation_performed'])

    def test_client_runs_additive_worker_and_preserves_transport(self):
        actual_popen = subprocess.Popen
        def spawn(command, **kwargs):
            self.assertIn('large_qubit.section9_recovery_scoring', command)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            saved = identity()
            saved['memory_accounting_policy'] = dict(memory.POLICY)
            ready = {'event': 'ready', 'schema_version': recovery.SCHEMA, 'identity': saved}
            code = '''import json, sys
print(json.dumps(READY), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    print(json.dumps({'id': r['id'], 'ok': True, 'result': {'fixture_only': True, 'query': r.get('query')}}), flush=True)
    if r['action'] == 'close': break
'''.replace('READY', repr(ready))
            return actual_popen([sys.executable, '-B', '-c', code], **kwargs)
        with patch.object(recovery.subprocess, 'Popen', side_effect=spawn):
            with recovery.PhysicsClient(ROOT, sys.executable, 'CO2_I01', execute=True,
                                        expected_freeze_sha256=FREEZE_SHA) as client:
                result = client.score({'kind': 'hf'})
                self.assertEqual(result['query'], {'kind': 'hf', 'geometry_id': 'CO2_I01'})
                self.assertTrue(result['fixture_only'])
            self.assertIsNone(client._process)


if __name__ == '__main__':
    unittest.main()

