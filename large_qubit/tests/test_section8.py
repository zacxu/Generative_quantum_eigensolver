"""Section 8 mechanics with fake generators and analytic scalar fixtures only."""
from contextlib import nullcontext
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section8 as s
from large_qubit.common import digest, read_json
from large_qubit.protocol import METHODS, FITS
from large_qubit.section8_ledger import TrajectoryLedger, ScalarCache, OrderedScorer
from large_qubit.section8_policy import policy, attempt_seed, PREPARATION, REGISTRATION, FINAL_FREEZE

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {'partition': 'validation', 'geometry_id': 'CO2_V01',
    'hamiltonian': 'fixture', 'orbitals': 'fixture', 'catalogue': 'fixture',
    'backend': 'analytic_fixture', 'precision': 'float64'}


def evaluate(query):
    if query['kind'] == 'hf':
        energy = -90.
    elif query['kind'] == 'discrete':
        energy = -90.1
    else:
        energy = -90. - .001 * sum(math.cos(x) for x in query['parameters'])
    return {'energy_hartree': energy, 'norm': 1., 'populations': [11, 11],
        'query_sha256': digest(query), 'cache_identity': IDENTITY,
        'evaluator_calls': 1, 'wall_seconds': .001, 'fixture_only': True}


class Generator:
    def __init__(self, root=None, key='gpt2_xl', **kwargs):
        self.model_key = key
        self.metadata = {'runtime_identity': {'fixture_only': True}}
        self.calls = 0
        self.invalid = False
        self.infrastructure = False

    def switch(self, *args, **kwargs):
        pass

    def generate(self, alias, catalogue, seed):
        self.calls += 1
        return {'valid': not self.invalid and not self.infrastructure,
            'ops': [0] * 32, 'raw_text': 'fixture', 'token_ids': [1],
            'wall_seconds': .001, 'input_tokens': 1, 'output_tokens': 1,
            'error_kind': 'infrastructure' if self.infrastructure else None}

    def statistics(self):
        return {'attempts': self.calls, 'fixture_only': True}

    def close(self):
        pass


class Physics:
    def __init__(self, *args, **kwargs):
        self.identity = {'cache_identity': IDENTITY, 'fixture_only': True}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    score = staticmethod(evaluate)

    def verify(self, query):
        return {**evaluate(query), 'fresh_basis_and_evaluator': True}

    def reference(self):
        return {'energy_hartree': -100., 'geometry_id': 'CO2_V01', 'partition': 'validation',
            'cache_identity': IDENTITY, 'report': {'all_passed': True, 'converged': True,
                'independent_starts': 2, 'runs': [
                    {'energy_ha': -100., 'converged': True, 'norm_squared': 1.,
                     'complex128_residual_norm_ha': 1e-12} for _ in range(2)]}, 'fixture_only': True}


class Section8Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalogue = read_json(ROOT / 'large_qubit/catalogues/co2_screened_v1.json')

    def tearDown(self):
        self.temp.cleanup()

    def components(self, method):
        ident = s.ledger_identity('fixture', method, IDENTITY)
        ledger = TrajectoryLedger(self.root / (method + '.sqlite'), ident)
        cache = ScalarCache(self.root / 'cache.sqlite')
        return ledger, OrderedScorer(ledger, cache, IDENTITY, evaluate)

    def test_prospective_original_budget_and_no_performance_gate(self):
        p = policy()
        self.assertEqual((p['maximum_panel_requests'], p['requests_per_method']), (1024, 128))
        self.assertEqual(p['validation_vqe_restart_caps'], [63, 64])
        self.assertIsNone(p['performance_gate'])
        self.assertNotEqual(attempt_seed(METHODS[0], 'CO2_V01', 1), attempt_seed(METHODS[0], 'CO2_V01', 2))

    def test_no_execution_without_explicit_permission(self):
        for call in (s.prepare, s.run_panel, s.freeze):
            with self.assertRaisesRegex(ValueError, 'explicit'):
                call(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unfinished_training_refuses_before_model_or_physics(self):
        with patch('large_qubit.section7.status_section7', return_value={'training_complete': False}), \
             self.assertRaisesRegex(ValueError, 'all three'):
            s.training_bindings(self.root)

    def test_no_clobber_atomic_record_and_tamper(self):
        path = self.root / 'proof.json'
        s.save(path, {'x': 1})
        s.save(path, {'x': 1})
        with self.assertRaises(ValueError):
            s.save(path, {'x': 2})
        path.write_text('{"payload":{"x":2},"sha256":"wrong"}')
        with self.assertRaises(ValueError):
            s.load(path)

    def test_reference_checks_independent_residuals_and_agreement(self):
        reference = Physics().reference()
        s.verify_reference(reference, IDENTITY)
        reference['report']['runs'][0]['complex128_residual_norm_ha'] = .01
        with self.assertRaisesRegex(ValueError, 'residual'):
            s.verify_reference(reference, IDENTITY)
        reference = Physics().reference()
        reference['report']['runs'][1]['energy_ha'] = -99.
        with self.assertRaisesRegex(ValueError, 'agreement'):
            s.verify_reference(reference, IDENTITY)

    def test_completed_vqe_replay_uses_no_new_evaluations(self):
        method = 'conventional_vqe'
        ledger, scorer = self.components(method)
        with scorer:
            first = s.execute_trajectory(ledger, scorer, method, catalogue=self.catalogue, initial_angles=[.01] * 24)
        cache = ScalarCache(self.root / 'cache.sqlite')
        with OrderedScorer(ledger, cache, IDENTITY, lambda q: self.fail('unexpected evaluation')) as scorer:
            second = s.execute_trajectory(ledger, scorer, method, catalogue=self.catalogue, initial_angles=[.01] * 24)
        self.assertEqual(first['head_sha256'], second['head_sha256'])

    def test_invalid_generation_exhaustion_is_reportable_negative_outcome(self):
        method = METHODS[0]
        ledger, scorer = self.components(method)
        gen = Generator()
        gen.invalid = True
        with scorer:
            snap = s.execute_trajectory(ledger, scorer, method, catalogue=self.catalogue, generator=gen)
        self.assertEqual((snap['logical_requests'], snap['completion_attempts']), (1, 256))
        checked = s.technical_checks(snap, method=method,
            verification=Physics().verify({'kind': 'hf'}), reference_energy=-100.)
        self.assertTrue(checked['all_passed'])

    def test_infrastructure_error_halts_instead_of_looping_to_256(self):
        ledger, scorer = self.components(METHODS[0])
        gen = Generator()
        gen.infrastructure = True
        with scorer, self.assertRaisesRegex(ValueError, 'infrastructure'):
            s.execute_trajectory(ledger, scorer, METHODS[0], catalogue=self.catalogue, generator=gen)
        self.assertEqual(gen.calls, 1)

    def test_uncertain_score_blocks_recovery_and_freeze(self):
        ledger, scorer = self.components(METHODS[0])
        scorer.close()
        ledger.reserve_request('hf', {'kind': 'hf'}, kind='hf')
        ledger.recover_pending()
        with self.assertRaisesRegex(ValueError, 'Abandoned'):
            OrderedScorer(ledger, ScalarCache(self.root / 'cache.sqlite'), IDENTITY, evaluate)

    def test_saved_generation_replays_without_repeating_the_attempt(self):
        method = METHODS[0]
        ledger, scorer = self.components(method)
        with scorer:
            scorer.submit('hf', {'kind': 'hf'}, kind='hf').result()
        seed = attempt_seed(method, 'CO2_V01', 1)
        ledger.reserve_attempt('attempt_0001', {'seed': seed})
        ledger.finish_attempt(1, Generator().generate('CO2_V01', self.catalogue, seed), valid=True)
        # The process ended after persisting the answer, before admitting it.
        cache = ScalarCache(self.root / 'cache.sqlite')
        with OrderedScorer(ledger, cache, IDENTITY, evaluate) as scorer:
            # Stop after replaying that one admission, without another generation.
            class StopGenerator:
                def generate(self, *args):
                    raise KeyboardInterrupt('fixture controlled stop')
            with self.assertRaises(KeyboardInterrupt):
                s.execute_trajectory(ledger, scorer, method, catalogue=self.catalogue,
                                     generator=StopGenerator())
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['logical_requests'], 2)
        self.assertEqual(snapshot['requests'][1]['attempt_ordinal'], 1)
        self.assertEqual(snapshot['attempts'][0]['output']['ops'], [0] * 32)
        self.assertEqual(snapshot['completion_attempts'], 2)
        self.assertEqual(snapshot['pending_attempts'], 1)
        ledger.recover_pending('fixture stopped before generation completed')
        self.assertEqual(ledger.snapshot()['attempts'][1]['status'], 'abandoned')

    def test_superseded_panel_cannot_resume(self):
        for panel in ('panel_0001', 'panel_0002'):
            (self.root / s.OUTPUT_ROOT / panel).mkdir(parents=True)
        with patch.object(s, 'verify_preparation', return_value={}), \
             patch.object(s.platform, 'system', return_value='Linux'), \
             patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, 'superseded'):
                s.run_panel(self.root, execute=True, resume=True, panel='panel_0001')

    def test_failed_reference_costs_are_retained_with_unknown_work_explicit(self):
        panel = self.root / s.OUTPUT_ROOT / 'panel_0001'
        s.save(panel / 'reference_started.json', {'cost_category': 'reference_overhead'})
        s.save(panel / 'execution_attempts/attempt_0001/failed.json', {
            'scientific_failure_report': {'cost_category': 'reference_solver_overhead',
                'hamiltonian_actions': 123, 'wall_seconds': 45.}})
        s.save(self.root / PREPARATION, {'training': {'fits': 'fixture'}})
        class Release:
            source_acquisition_cost = {'fixture': True}
        with patch('large_qubit.corpus_v2.load_release', return_value=Release()):
            costs = s.validation_costs(self.root)
        row = costs['panels']['panel_0001']
        self.assertEqual(row['reference_hamiltonian_actions'], 123)
        self.assertEqual(row['recorded_reference_seconds'], 45.)
        self.assertIn('reference_started.json', row['incomplete_overhead_records'])
        self.assertEqual(costs['total_validation_logical_requests'], 0)

    def test_backbone_load_failure_does_not_duplicate_previous_backbone_costs(self):
        cat = self.root / s.CATALOGUE
        cat.parent.mkdir(parents=True)
        cat.write_bytes((ROOT / s.CATALOGUE).read_bytes())
        prepared = {'training': {'fits': {}}}
        s.save(self.root / PREPARATION, prepared)
        def generator(root, key, **kwargs):
            if key == 'qwen3_4b':
                raise RuntimeError('fixture Qwen load failed')
            return Generator(root, key, **kwargs)
        def hf_only(ledger, scorer, *args, **kwargs):
            scorer.submit('hf', {'kind': 'hf'}, kind='hf').result()
        with patch.object(s, 'verify_preparation', return_value=prepared), \
             patch.object(s.platform, 'system', return_value='Linux'), \
             patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()), \
             patch('large_qubit.section8_scoring.PhysicsClient', Physics), \
             patch('large_qubit.section8_inference.ResidentGenerator', side_effect=generator), \
             patch.object(s, 'execute_trajectory', side_effect=hf_only), \
             patch.object(s, 'technical_checks', return_value={'fixture_only': True}), \
             self.assertRaisesRegex(RuntimeError, 'Qwen load failed'):
            s.run_panel(self.root, execute=True)
        attempt = self.root / s.OUTPUT_ROOT / 'panel_0001/execution_attempts/attempt_0001'
        self.assertTrue((attempt / 'gpt2_xl_inference_cost.json').is_file())
        self.assertFalse((attempt / 'gpt2_xl_failed_inference_cost.json').exists())
        self.assertTrue((attempt / 'failed.json').is_file())

    def test_query_changed_after_generation_fails_technical_gate(self):
        ledger, scorer = self.components(METHODS[0])
        with scorer:
            snap = s.execute_trajectory(ledger, scorer, METHODS[0], catalogue=self.catalogue, generator=Generator())
        snap['attempts'][0]['output']['ops'] = [1] * 32
        with self.assertRaisesRegex(ValueError, 'Scored circuit'):
            s.technical_checks(snap, method=METHODS[0],
                verification=Physics().verify({'kind': 'discrete', 'ops': [0] * 32}), reference_energy=-100.)

    def test_full_fake_panel_resume_freeze_and_mutation_refusal(self):
        from large_qubit.section5 import SCREENING_PATH
        cat = self.root / s.CATALOGUE
        cat.parent.mkdir(parents=True)
        cat.write_bytes((ROOT / s.CATALOGUE).read_bytes())
        screen = self.root / SCREENING_PATH
        screen.parent.mkdir(parents=True)
        screen.write_text('{"vqe_initial_angles":[' + ','.join(['0.01'] * 24) + ']}')
        prepared = {'release_sha256': 'fixture', 'contexts': {},
                    'training': {'fits': {fit: {'adapter_path': 'fixture'} for fit in FITS}}}
        s.save(self.root / PREPARATION, prepared)
        registration = self.root / REGISTRATION
        registration.parent.mkdir(parents=True)
        registration.write_text('{}')
        class Release:
            source_acquisition_cost = {'release_sha256': '0' * 64, 'logical_energy_requests': 10, 'wall_seconds': 1.}
            def retrieval_rows(self):
                return [{'ops': [0] * 32}] * 127
        with patch.object(s, 'verify_preparation', return_value=prepared), \
             patch.object(s.platform, 'system', return_value='Linux'), \
             patch('large_qubit.section7._gpu_operation_lock', return_value=nullcontext()), \
             patch('large_qubit.section8_scoring.PhysicsClient', Physics), \
             patch('large_qubit.section8_inference.ResidentGenerator', Generator), \
             patch('large_qubit.corpus_v2.load_release', return_value=Release()), \
             patch('large_qubit.baselines.retrieval_candidates', return_value={'candidates': [{'ops': [0] * 32}] * 127}):
            report = s.run_panel(self.root, execute=True)
            self.assertTrue(report['all_passed'])
            self.assertEqual(report['logical_requests'], 1019)
            for method, row in report['methods'].items():
                self.assertEqual(row['logical_requests'], 123 if method == 'conventional_vqe' else 128)
            resumed = s.run_panel(self.root, execute=True, resume=True)
            self.assertEqual(resumed, report)
            with self.assertRaisesRegex(ValueError, 'technically successful'):
                s.run_panel(self.root, panel='panel_0002', execute=True,
                            rerun_reason='try a second panel after technical success')
            self.assertFalse((self.root / s.OUTPUT_ROOT / 'panel_0002').exists())
            final = s.freeze(self.root, execute=True)
            self.assertTrue(final['section_8_complete'])
            self.assertFalse(final['execution_ready'])
            self.assertTrue((self.root / FINAL_FREEZE).is_file())
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                s.run_panel(self.root, execute=True, resume=True)
            log = self.root / s.OUTPUT_ROOT / 'panel_0001' / 'injected.txt'
            log.write_text('new bytes')
            with self.assertRaisesRegex(ValueError, 'history changed'):
                s.verify_final_freeze(self.root)


if __name__ == '__main__':
    unittest.main()
