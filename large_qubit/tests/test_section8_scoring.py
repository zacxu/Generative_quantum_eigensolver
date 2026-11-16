"""Fake chemistry and small transport fixtures; no real molecule/model scoring."""

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

from large_qubit import section8_scoring as scoring
from large_qubit.common import read_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def identity():
    return {'geometry_id': scoring.GEOMETRY, 'partition': 'validation',
            'physics_manifest_sha256': scoring.PHYSICS_SHA256,
            'cache_identity': {'partition': 'validation', 'geometry_id': scoring.GEOMETRY,
                'hamiltonian': 'h' * 64, 'orbitals': 'o' * 64,
                'catalogue': scoring.CATALOGUE_SHA256, 'backend': 'b' * 64,
                'precision': {'state': 'complex128', 'integrals': 'float64'}}}


class FakeBasis:
    def __init__(self):
        self.gates = None
        self.family = None

    def hf_state(self):
        return [1.0]

    def apply_circuit(self, state, gates, *, family):
        self.gates, self.family = gates, family
        return state

    def diagnostics(self, state):
        return {'state_norm': 1.0, 'n_alpha': 11, 'n_beta': 11,
                'sector_leakage_probability': 0.0}


class FakeEvaluator:
    def __init__(self, *_, **kwargs):
        self.kwargs = kwargs
        self.hamiltonian_identity = 'h' * 64
        self.backend_metadata = {'backend': 'fixture_not_physics', 'threads': kwargs.get('threads', 1)}
        self.calls = 0

    def statistics(self):
        return {'apply_h_calls': self.calls, 'kernel_contractions': self.calls,
                'hamiltonian_action_seconds': self.calls * 0.001}

    def energy_sector(self, state, basis):
        self.calls += 1
        return -2.0


def fixture_scorer():
    scorer = scoring.ValidationScorer.__new__(scoring.ValidationScorer)
    scorer.evaluator = FakeEvaluator()
    scorer.basis = FakeBasis()
    scorer.catalogue = read_json(ROOT / scoring.CATALOGUE)
    scorer.identity = identity()
    scorer._reference_computed = False
    return scorer


class QueryTests(unittest.TestCase):
    def test_hf_and_discrete_recipes(self):
        self.assertEqual(scoring.normalize_query({'kind': 'hf'}), {'kind': 'hf'})
        self.assertEqual(scoring.normalize_query({'kind': 'discrete', 'ops': [0] * 32})['ops'], [0] * 32)

    def test_all_other_geometries_refused(self):
        for alias in ('CO2_I01', 'CO2_I02', 'CO2_T01'):
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, 'CO2_V01'):
                scoring.normalize_query({'kind': 'hf', 'geometry_id': alias})

    def test_wrong_length_bool_and_out_of_range_refused(self):
        for ops in ([0] * 64, [True] * 32, [513] * 32, [-1] * 32):
            with self.subTest(ops=ops[:1]), self.assertRaises(ValueError):
                scoring.normalize_query({'kind': 'discrete', 'ops': ops})

    def test_vqe_exact_float_values_preserved(self):
        values = [0.12345678901234567, -0.0] + [0.] * 22
        self.assertEqual(scoring.normalize_query({'kind': 'vqe', 'parameters': values})['parameters'], values)

    def test_nonfinite_bool_extra_fields_and_bad_kind_refused(self):
        queries = [{'kind': 'vqe', 'parameters': [float('nan')] * 24},
                   {'kind': 'vqe', 'parameters': [float('inf')] * 24},
                   {'kind': 'vqe', 'parameters': [True] * 24},
                   {'kind': 'hf', 'reference_energy': -100}, {'kind': ['hf']}]
        for query in queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                scoring.normalize_query(query)

    def test_default_client_never_spawns(self):
        with patch.object(scoring.subprocess, 'Popen') as popen:
            client = scoring.PhysicsClient(ROOT, sys.executable)
            with self.assertRaisesRegex(ValueError, 'execute=True'):
                client.score({'kind': 'hf'})
            client.close()
        popen.assert_not_called()

    def test_probe_and_default_cli_import_no_physics(self):
        before = set(sys.modules)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(scoring.main(['--probe']), 0)
        self.assertFalse(json.loads(output.getvalue())['scientific_computation_performed'])
        self.assertNotIn('pyscf', set(sys.modules) - before)


class MeasurementTests(unittest.TestCase):
    def test_score_calls_fake_evaluator_once_and_returns_no_state(self):
        scorer = fixture_scorer()
        result = scorer.score({'kind': 'hf'})
        self.assertEqual(result['energy_hartree'], -2.)
        self.assertEqual(result['evaluator_calls'], 1)
        self.assertEqual(result['populations'], [11, 11])
        self.assertNotIn('state', result)
        self.assertEqual(result['cost_category'], 'ordinary_energy_request')
        self.assertFalse(result['scalar_cache_used'])

    def test_discrete_and_vqe_families_are_distinct(self):
        scorer = fixture_scorer()
        scorer.score({'kind': 'discrete', 'ops': [1] + [0] * 31})
        self.assertEqual(scorer.basis.family, 'whole_qubit')
        self.assertEqual(len(scorer.basis.gates), 1)
        scorer.score({'kind': 'vqe', 'parameters': [0.1] * 24})
        self.assertEqual(scorer.basis.family, 'fermionic')
        self.assertEqual(len(scorer.basis.gates), 24)

    def test_verify_uses_fresh_evaluator_and_zero_cache_basis(self):
        scorer = fixture_scorer()
        fresh, basis = FakeEvaluator(), FakeBasis()
        with patch.object(scorer, '_new_evaluator', return_value=fresh) as evaluator_factory, \
             patch.object(scorer, '_new_basis', return_value=basis) as basis_factory:
            result = scorer.verify({'kind': 'hf'})
        evaluator_factory.assert_called_once_with()
        basis_factory.assert_called_once_with(cache_bytes=0)
        self.assertEqual(scorer.evaluator.calls, 0)
        self.assertEqual(fresh.calls, 1)
        self.assertEqual(result['cost_category'], 'integrity_overhead_not_search')
        self.assertFalse(result['independent_backend_claimed'])

    def test_reference_uses_two_bounded_starts_and_omits_state(self):
        scorer = fixture_scorer()
        solved = {'energy': -3., 'state': object(),
                  'report': {'all_passed': True, 'converged': True, 'independent_starts': 2}}
        with patch('large_qubit.reference.solve_reference', return_value=solved) as solve:
            result = scorer.reference()
            with self.assertRaisesRegex(ValueError, 'already solved'):
                scorer.reference()
        self.assertEqual(solve.call_args.kwargs, scoring.REFERENCE_SETTINGS)
        self.assertEqual(result['energy_hartree'], -3.)
        self.assertEqual(result['ordinary_energy_requests'], 0)
        self.assertFalse(result['state_arrays_returned'])
        self.assertNotIn('state', result)

    def test_invalid_population_or_norm_rejected(self):
        scorer = fixture_scorer()
        with patch.object(scorer.basis, 'diagnostics', return_value={'state_norm': .9, 'n_alpha': 11, 'n_beta': 11}), \
             self.assertRaisesRegex(ValueError, 'norm or populations'):
            scorer.score({'kind': 'hf'})

    def test_memory_guard_rejected_after_recorded_physical_call(self):
        scorer = fixture_scorer()
        with patch('large_qubit.benchmark._peak_rss_bytes', return_value=scoring.MAX_RSS_BYTES + 1), \
             self.assertRaisesRegex(ValueError, 'memory bound'):
            scorer.score({'kind': 'hf'})
        self.assertEqual(scorer.evaluator.calls, 1)

    def test_constructor_authenticates_only_validation_array_loading(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            private = 'large_qubit/scoring_only/fixture'
            (root / private).mkdir(parents=True)
            arrays = root / private / 'CO2_V01.npz'
            np.savez(arrays, one_electron_integrals=np.zeros((1, 1)), eri_chemists=np.zeros((1, 1, 1, 1)))
            meta = root / private / 'CO2_V01.json'
            meta.write_text(json.dumps({'nuclear_repulsion_ha': 0.}), encoding='utf-8')
            entry = {'geometry_id': 'CO2_V01', 'partition': 'validation',
                     'arrays_path': arrays.relative_to(root).as_posix(),
                     'metadata_path': meta.relative_to(root).as_posix(),
                     'hamiltonian_fingerprint': 'f' * 64, 'orbital_fingerprint': 'o' * 64}
            manifest = {'restricted_output': private, 'geometries': {'CO2_V01': entry},
                        'files': {entry['arrays_path']: sha256_file(arrays), entry['metadata_path']: sha256_file(meta)}}
            manifest_path = root / scoring.PHYSICS / 'manifest.json'
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            catalogue = root / scoring.CATALOGUE
            catalogue.parent.mkdir(parents=True)
            catalogue.write_bytes((ROOT / scoring.CATALOGUE).read_bytes())
            for relative in scoring.SOURCES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture source identity only')
            physical_hash = sha256_file(manifest_path)
            original_load = np.load
            with patch.object(scoring, 'PHYSICS_SHA256', physical_hash), \
                 patch('large_qubit.physics.verify_physics', return_value={'manifest_sha256': physical_hash}) as verification, \
                 patch('large_qubit.evaluator.IntegralEvaluator', FakeEvaluator), \
                 patch.object(scoring.ValidationScorer, '_new_basis', return_value=FakeBasis()), \
                 patch.object(np, 'load', wraps=original_load) as load, patch.dict(os.environ):
                scorer = scoring.ValidationScorer(root, execute=True, threads=2)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(Path(load.call_args.args[0]).name, 'CO2_V01.npz')
            self.assertEqual(scorer.identity['runtime']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(scorer.identity['runtime']['cpu_threads'], 2)
            self.assertFalse(scorer.identity['test_arrays_loaded'])
            verification.assert_called_once_with(root.resolve(), scoring.PHYSICS, numerical_integrity=False)


class FakeWireScorer:
    def __init__(self, *_, **__):
        self.identity = identity()

    def score(self, query):
        print('native diagnostic noise, not protocol JSON')
        return {'energy_hartree': -2., 'kind': query['kind']}

    verify = score

    def reference(self):
        return {'energy_hartree': -3., 'report': {'fixture': True}}


class ProtocolTests(unittest.TestCase):
    def test_worker_roundtrip_and_native_noise_separate(self):
        stream = io.StringIO('\n'.join(json.dumps(item) for item in (
            {'id': 1, 'action': 'score', 'query': {'kind': 'hf'}},
            {'id': 2, 'action': 'reference'}, {'id': 3, 'action': 'close'})) + '\n')
        output = io.StringIO()
        self.assertEqual(scoring.worker(ROOT, execute=True, input_stream=stream,
                                        output_stream=output, factory=FakeWireScorer), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[0]['event'], 'ready')
        self.assertEqual(rows[1]['result']['energy_hartree'], -2.)
        self.assertEqual(rows[2]['result']['energy_hartree'], -3.)

    def test_worker_requires_explicit_execute(self):
        output = io.StringIO()
        with patch.object(scoring, 'ValidationScorer') as factory:
            self.assertEqual(scoring.worker(ROOT, output_stream=output), 1)
        factory.assert_not_called()
        self.assertIn('requires --execute', output.getvalue())

    def test_worker_refuses_test_query_before_fake_energy(self):
        request = {'id': 1, 'action': 'score', 'query': {'kind': 'hf', 'geometry_id': 'CO2_I01'}}
        output = io.StringIO()
        with patch.object(FakeWireScorer, 'score') as score:
            self.assertEqual(scoring.worker(ROOT, execute=True, input_stream=io.StringIO(json.dumps(request) + '\n'),
                                            output_stream=output, factory=FakeWireScorer), 1)
        score.assert_not_called()
        self.assertIn('CO2_V01', output.getvalue())

    def test_reference_cannot_smuggle_test_query(self):
        output = io.StringIO()
        request = {'id': 1, 'action': 'reference', 'query': {'geometry_id': 'CO2_I02'}}
        with patch.object(FakeWireScorer, 'reference') as reference:
            self.assertEqual(scoring.worker(ROOT, execute=True, input_stream=io.StringIO(json.dumps(request) + '\n'),
                                            output_stream=output, factory=FakeWireScorer), 1)
        reference.assert_not_called()

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            scoring._loads('{"kind":"hf","kind":"vqe"}')

    def test_actual_plain_python_subprocess_transport_and_large_stderr(self):
        # A deliberately fake plain Python child; it imports no project/science modules.
        code = '''import json, os, sys
sys.stderr.write('x' * 300000); sys.stderr.flush()
print(json.dumps(READY), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    v = {'energy_hartree': -2., 'cuda': os.environ['CUDA_VISIBLE_DEVICES'], 'threads': os.environ['OMP_NUM_THREADS']}
    print(json.dumps({'id': r['id'], 'ok': True, 'result': v}), flush=True)
    if r['action'] == 'close': break
'''.replace('READY', repr({'event': 'ready', 'schema_version': scoring.SCHEMA, 'identity': identity()}))
        actual_popen = subprocess.Popen
        def fake_worker_command(command, **kwargs):
            self.assertIn('large_qubit.section8_scoring', command)
            self.assertIn('--execute', command)
            return actual_popen([sys.executable, '-B', '-u', '-c', code], **kwargs)
        with patch.object(scoring.subprocess, 'Popen', side_effect=fake_worker_command):
            with scoring.PhysicsClient(ROOT, sys.executable, execute=True, threads=2) as client:
                result = client.score({'kind': 'hf'})
                self.assertEqual(result['cuda'], '')
                self.assertEqual(result['threads'], '2')
                self.assertEqual(client.verify({'kind': 'hf'})['energy_hartree'], -2.)
                self.assertEqual(client.reference()['energy_hartree'], -2.)
                self.assertEqual(len(client.stderr_tail(limit=100)), 100)
            self.assertIsNone(client._process)


if __name__ == '__main__':
    unittest.main()
