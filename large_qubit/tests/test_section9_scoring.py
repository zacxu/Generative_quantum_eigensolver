"""Small fake chemistry/transport fixtures; never score a real target."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section9_scoring as scoring
from large_qubit import section8_scoring as inherited
from large_qubit.common import digest, read_json, sha256_file
from large_qubit.tests.test_section8_scoring import FakeBasis, FakeEvaluator


ROOT = Path(__file__).resolve().parents[2]
FREEZE_SHA = 'f' * 64


def identity(alias='CO2_I01'):
    return {'geometry_id': alias, 'partition': 'final_test', 'final_freeze_sha256': FREEZE_SHA,
        'physics_manifest_sha256': inherited.PHYSICS_SHA256,
        'cache_identity': {'partition': 'final_test', 'geometry_id': alias,
            'hamiltonian': 'h' * 64, 'orbitals': 'o' * 64,
            'catalogue': inherited.CATALOGUE_SHA256, 'backend': 'b' * 64,
            'precision': {'state': 'complex128', 'integrals': 'float64'}}}


def scorer(alias='CO2_I01'):
    value = scoring.TargetScorer.__new__(scoring.TargetScorer)
    value.geometry_id = alias
    value.evaluator, value.basis = FakeEvaluator(), FakeBasis()
    value.catalogue = read_json(ROOT / inherited.CATALOGUE)
    value.identity = identity(alias)
    value._reference_computed = False
    return value


def validation_identity():
    current = inherited.probe()
    return {'reference_settings': dict(scoring.REFERENCE_SETTINGS),
            'cache_identity': {'precision': {'state': 'complex128', 'integrals': 'float64'},
                               'catalogue': inherited.CATALOGUE_SHA256},
            'runtime': {'cpu_threads': 1, 'packages': current['packages'], 'python': current['python']}}


class AdmissionTests(unittest.TestCase):
    def test_target_is_always_in_canonical_query(self):
        self.assertEqual(scoring.normalize_query({'kind': 'hf'}, 'CO2_I02'),
                         {'kind': 'hf', 'geometry_id': 'CO2_I02'})
        a = scoring.normalize_query({'kind': 'hf'}, 'CO2_I01')
        b = scoring.normalize_query({'kind': 'hf'}, 'CO2_I02')
        self.assertNotEqual(digest(a), digest(b))

    def test_validation_training_and_cross_target_queries_refused(self):
        for alias in ('CO2_V01', 'CO2_T01', 'CO2_I02'):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                scoring.normalize_query({'kind': 'hf', 'geometry_id': alias}, 'CO2_I01')

    def test_strict_recipes_and_no_parameter_rounding(self):
        values = [.12345678901234567, -0.] + [0.] * 22
        self.assertEqual(scoring.normalize_query({'kind': 'vqe', 'parameters': values}, 'CO2_I01')['parameters'], values)
        queries = [{'kind': 'discrete', 'ops': [0] * 64}, {'kind': 'discrete', 'ops': [True] * 32},
                   {'kind': 'discrete', 'ops': [513] * 32}, {'kind': 'vqe', 'parameters': [float('nan')] * 24},
                   {'kind': 'vqe', 'parameters': [10**400] * 24}, {'kind': 'hf', 'target_energy': -100}]
        for query in queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                scoring.normalize_query(query, 'CO2_I01')

    def test_preview_client_never_spawns_or_authenticates(self):
        with patch.object(scoring.subprocess, 'Popen') as popen, patch.object(scoring, 'authenticate_freeze') as gate:
            client = scoring.PhysicsClient(ROOT, sys.executable, 'CO2_I01')
            with self.assertRaisesRegex(ValueError, 'execute=True'):
                client.score({'kind': 'hf'})
            client.close()
        popen.assert_not_called()
        gate.assert_not_called()

    def test_probe_is_scientifically_inert(self):
        with patch.object(scoring, 'authenticate_freeze') as gate, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(scoring.main(['--probe']), 0)
        self.assertFalse(json.loads(out.getvalue())['scientific_computation_performed'])
        gate.assert_not_called()

    def test_freeze_failure_precedes_any_array_load(self):
        import numpy as np
        with patch.object(scoring, 'authenticate_freeze', side_effect=ValueError('freeze incomplete')), \
             patch.object(np, 'load') as load, patch('large_qubit.evaluator.IntegralEvaluator') as evaluator, \
             self.assertRaisesRegex(ValueError, 'freeze incomplete'):
            scoring.TargetScorer(ROOT, 'CO2_I01', execute=True, expected_freeze_sha256=FREEZE_SHA)
        load.assert_not_called()
        evaluator.assert_not_called()

    def test_changed_chemistry_packages_refused_before_array_load(self):
        import numpy as np
        validation = validation_identity()
        validation['runtime']['packages'] = {'fixture_wrong_version': '0'}
        with patch.object(scoring, 'authenticate_freeze', return_value=({}, validation, 'f' * 64)), \
             patch.object(np, 'load') as load, self.assertRaisesRegex(ValueError, 'Python/packages differ'):
            scoring.TargetScorer(ROOT, 'CO2_I01', execute=True, expected_freeze_sha256=FREEZE_SHA)
        load.assert_not_called()

    def test_thread_change_refused(self):
        with self.assertRaisesRegex(ValueError, 'frozen at one'):
            scoring.PhysicsClient(ROOT, sys.executable, 'CO2_I01', threads=4)

    def test_freeze_authentication_checks_expected_hash_and_saved_numerics(self):
        from large_qubit.section8 import save
        from large_qubit.section8_policy import FINAL_FREEZE, OUTPUT_ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            freeze = {'protocol': {'main': {'targets': list(scoring.TARGETS)}, 'cpu_threads': 1},
                      'selected_panel': 'panel_0001'}
            save(root / FINAL_FREEZE, freeze)
            frozen_sha = sha256_file(root / FINAL_FREEZE)
            saved_identity = validation_identity()
            save(root / OUTPUT_ROOT / 'panel_0001/physics_identity.json', saved_identity)
            proof = {'all_passed': True, 'section_8_complete': True, 'final_protocol_frozen': True,
                     'freeze_sha256': frozen_sha}
            with patch('large_qubit.section8.verify_final_freeze', return_value=proof) as verify:
                actual, saved, _ = scoring.authenticate_freeze(root, frozen_sha)
                self.assertEqual(actual, freeze)
                self.assertEqual(saved, saved_identity)
                verify.assert_called_once_with(root.resolve())
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    scoring.authenticate_freeze(root, '0' * 64)

    def test_constructor_reads_only_the_selected_target_fixture(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            private = 'large_qubit/scoring_only/target_fixture'
            (root / private).mkdir(parents=True)
            entries, files = {}, {}
            for alias in scoring.TARGETS:
                arrays, meta = root / private / (alias + '.npz'), root / private / (alias + '.json')
                np.savez(arrays, one_electron_integrals=np.zeros((1, 1)), eri_chemists=np.zeros((1, 1, 1, 1)))
                meta.write_text(json.dumps({'nuclear_repulsion_ha': 0.}), encoding='utf-8')
                entries[alias] = {'geometry_id': alias, 'partition': 'final_test',
                    'arrays_path': arrays.relative_to(root).as_posix(),
                    'metadata_path': meta.relative_to(root).as_posix(),
                    'hamiltonian_fingerprint': 'h' * 64, 'orbital_fingerprint': 'o' * 64}
                files[entries[alias]['arrays_path']] = sha256_file(arrays)
                files[entries[alias]['metadata_path']] = sha256_file(meta)
            manifest_path = root / inherited.PHYSICS / 'manifest.json'
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({'restricted_output': private,
                'geometries': entries, 'files': files}), encoding='utf-8')
            catalogue = root / inherited.CATALOGUE
            catalogue.parent.mkdir(parents=True)
            catalogue.write_bytes((ROOT / inherited.CATALOGUE).read_bytes())
            for name in scoring.SOURCES:
                (root / name).write_bytes(b'fixture source only')
            original_load = np.load
            for alias in scoring.TARGETS:
                with self.subTest(alias=alias), patch.object(scoring, 'authenticate_freeze', return_value=({}, validation_identity(), 'f' * 64)), \
                     patch.object(inherited, 'PHYSICS_SHA256', sha256_file(manifest_path)), \
                     patch('large_qubit.physics.verify_physics', return_value={'manifest_sha256': sha256_file(manifest_path)}), \
                     patch('large_qubit.evaluator.IntegralEvaluator', FakeEvaluator), \
                     patch.object(scoring.TargetScorer, '_new_basis', return_value=FakeBasis()), \
                     patch.object(np, 'load', wraps=original_load) as load, patch.dict(os.environ):
                    value = scoring.TargetScorer(root, alias, execute=True, expected_freeze_sha256=FREEZE_SHA)
                    self.assertEqual(load.call_count, 1)
                    self.assertEqual(Path(load.call_args.args[0]).name, alias + '.npz')
                    self.assertEqual(value.identity['cache_identity']['partition'], 'final_test')
                    self.assertEqual(value.identity['cache_identity']['geometry_id'], alias)
                    self.assertEqual(value.identity['runtime']['CUDA_VISIBLE_DEVICES'], '')


class MeasurementTests(unittest.TestCase):
    def test_ordinary_score_is_one_fake_action_and_target_keyed(self):
        value = scorer('CO2_I02')
        result = value.score({'kind': 'hf'})
        self.assertEqual(result['geometry_id'], 'CO2_I02')
        self.assertEqual(result['evaluator_calls'], 1)
        self.assertEqual(result['query_sha256'], digest({'geometry_id': 'CO2_I02', 'kind': 'hf'}))
        self.assertNotIn('state', result)

    def test_score_failure_retains_executed_action_cost(self):
        value = scorer()
        with patch.object(value.basis, 'diagnostics', return_value={'state_norm': .9, 'n_alpha': 11, 'n_beta': 11}):
            with self.assertRaises(scoring.ScoringError) as raised:
                value.score({'kind': 'hf'})
        report = raised.exception.report
        self.assertEqual(report['evaluator_calls'], 1)
        self.assertEqual(report['stage'], 'score')
        self.assertEqual(report['geometry_id'], 'CO2_I01')
        self.assertTrue(report['partial'])

    def test_fresh_verify_does_not_use_resident_evaluator_or_gate_cache(self):
        value, fresh, basis = scorer(), FakeEvaluator(), FakeBasis()
        with patch.object(value, '_new_evaluator', return_value=fresh), \
             patch.object(value, '_new_basis', return_value=basis) as new_basis:
            result = value.verify({'kind': 'hf'})
        new_basis.assert_called_once_with(cache_bytes=0)
        self.assertEqual(value.evaluator.calls, 0)
        self.assertEqual(fresh.calls, 1)
        self.assertTrue(result['fresh_basis_and_evaluator'])
        self.assertFalse(result['independent_backend_claimed'])

    def test_failed_fresh_verify_retains_its_action_count(self):
        value, fresh, basis = scorer(), FakeEvaluator(), FakeBasis()
        with patch.object(value, '_new_evaluator', return_value=fresh), \
             patch.object(value, '_new_basis', return_value=basis), \
             patch.object(basis, 'diagnostics', return_value={'state_norm': .9, 'n_alpha': 11, 'n_beta': 11}), \
             self.assertRaises(scoring.ScoringError) as raised:
            value.verify({'kind': 'hf'})
        self.assertEqual(raised.exception.report['stage'], 'verify')
        self.assertEqual(raised.exception.report['evaluator_calls'], 1)
        self.assertEqual(value.evaluator.calls, 0)

    def test_reference_is_bounded_and_returns_no_state(self):
        value = scorer()
        solved = {'energy': -3., 'state': object(),
                  'report': {'all_passed': True, 'converged': True, 'independent_starts': 2}}
        with patch('large_qubit.reference.solve_reference', return_value=solved) as solve:
            result = value.reference()
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                value.reference()
        self.assertEqual(solve.call_args.kwargs, inherited.REFERENCE_SETTINGS)
        self.assertNotIn('state', result)
        self.assertFalse(result['state_arrays_returned'])
        self.assertEqual(result['ordinary_energy_requests'], 0)

    def test_reference_failure_retains_partial_scientific_report(self):
        value = scorer()
        original = {'hamiltonian_actions': 3, 'wall_seconds': 1., 'converged': False,
                    'failure_reason': 'fixture_budget_exhausted'}
        def fail(*args, **kwargs):
            value.evaluator.calls += 3
            error = RuntimeError('fixture only')
            error.report = original
            raise error
        with patch('large_qubit.reference.solve_reference', side_effect=fail), self.assertRaises(scoring.ScoringError) as raised:
            value.reference()
        self.assertEqual(raised.exception.report['original_scientific_report'], original)
        self.assertEqual(raised.exception.report['evaluator_calls'], 3)
        self.assertEqual(raised.exception.report['hamiltonian_actions'], 3)

    def test_reference_memory_gate_keeps_completed_solver_diagnostics(self):
        value = scorer()
        report = {'all_passed': True, 'converged': True, 'independent_starts': 2,
                  'hamiltonian_actions': 7, 'fixture_only': True}
        with patch('large_qubit.reference.solve_reference', return_value={'energy': -3., 'state': object(), 'report': report}), \
             patch('large_qubit.benchmark._peak_rss_bytes', return_value=scoring.MAX_RSS_BYTES + 1), \
             self.assertRaises(scoring.ScoringError) as raised:
            value.reference()
        self.assertEqual(raised.exception.report['original_scientific_report'], report)
        self.assertEqual(raised.exception.report['hamiltonian_actions'], 7)

    def test_nonfinite_failure_diagnostics_remain_serializable(self):
        result = scoring._safe({'bad': float('nan'), 'partial_actions': 2})
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['partial_actions'], 2)


class FakeWireScorer:
    def __init__(self, root, alias, **kwargs):
        self.identity = identity(alias)

    def score(self, query):
        return {'energy_hartree': -2., 'query': query}

    verify = score

    def reference(self):
        return {'energy_hartree': -3., 'report': {'fixture_only': True}}


class TransportTests(unittest.TestCase):
    def test_dead_plain_child_records_unknown_work_and_stderr(self):
        actual_popen = subprocess.Popen
        ready = {'event': 'ready', 'schema_version': scoring.SCHEMA, 'identity': identity()}
        code = '''import json, sys
print(json.dumps(READY), flush=True)
sys.stdin.readline()
sys.stderr.write('fixture native worker stopped'); sys.stderr.flush()
'''.replace('READY', repr(ready))
        def spawn(command, **kwargs):
            return actual_popen([sys.executable, '-B', '-u', '-c', code], **kwargs)
        with patch.object(scoring.subprocess, 'Popen', side_effect=spawn):
            with scoring.PhysicsClient(ROOT, sys.executable, 'CO2_I01', execute=True, expected_freeze_sha256=FREEZE_SHA) as client:
                with self.assertRaises(scoring.ScoringError) as raised:
                    client.score({'kind': 'hf'})
                report = raised.exception.report
                self.assertTrue(report['transport_failure'])
                self.assertTrue(report['possibly_unreported_physical_work'])
                self.assertFalse(report['evaluator_calls_known'])
                self.assertIn('fixture native worker stopped', report['stderr_tail'])

    def test_worker_rejects_cross_target_before_fake_scoring(self):
        line = json.dumps({'id': 1, 'action': 'score', 'query': {'kind': 'hf', 'geometry_id': 'CO2_I02'}}) + '\n'
        with patch.object(FakeWireScorer, 'score') as fake:
            self.assertEqual(scoring.worker(ROOT, 'CO2_I01', execute=True, expected_freeze_sha256=FREEZE_SHA,
                input_stream=io.StringIO(line), output_stream=io.StringIO(), factory=FakeWireScorer), 1)
        fake.assert_not_called()

    def test_worker_requires_explicit_execution(self):
        with patch.object(scoring, 'TargetScorer') as factory:
            self.assertEqual(scoring.worker(ROOT, 'CO2_I01', expected_freeze_sha256=FREEZE_SHA,
                                             output_stream=io.StringIO()), 1)
        factory.assert_not_called()

    def test_two_plain_fake_children_with_bounded_stderr_and_canonical_queries(self):
        actual_popen = subprocess.Popen
        def spawn(command, **kwargs):
            self.assertIn('large_qubit.section9_scoring', command)
            alias = command[command.index('--geometry-id') + 1]
            ready = {'event': 'ready', 'schema_version': scoring.SCHEMA, 'identity': identity(alias)}
            code = '''import json, os, sys
sys.stderr.write('x' * 300000); sys.stderr.flush()
print(json.dumps(READY), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    v = {'energy_hartree': -2., 'query': r.get('query'), 'cuda': os.environ['CUDA_VISIBLE_DEVICES']}
    print(json.dumps({'id': r['id'], 'ok': True, 'result': v}), flush=True)
    if r['action'] == 'close': break
'''.replace('READY', repr(ready))
            return actual_popen([sys.executable, '-B', '-u', '-c', code], **kwargs)
        with patch.object(scoring.subprocess, 'Popen', side_effect=spawn):
            with scoring.PhysicsClient(ROOT, sys.executable, 'CO2_I01', execute=True, expected_freeze_sha256=FREEZE_SHA) as first, \
                 scoring.PhysicsClient(ROOT, sys.executable, 'CO2_I02', execute=True, expected_freeze_sha256=FREEZE_SHA) as second:
                for client in (first, second):
                    result = client.score({'kind': 'hf'})
                    self.assertEqual(result['query']['geometry_id'], client.geometry_id)
                    self.assertEqual(result['cuda'], '')
                    self.assertEqual(len(client.stderr_tail(limit=100)), 100)
                self.assertNotEqual(first.identity['cache_identity'], second.identity['cache_identity'])
            self.assertIsNone(first._process)
            self.assertIsNone(second._process)


if __name__ == '__main__':
    unittest.main()
