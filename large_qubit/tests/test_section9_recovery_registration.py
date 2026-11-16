"""CPU-only recovery eligibility, accounting and immutable registration tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit.common import canonical_bytes, read_json, sha256_file
from large_qubit.section8 import save
from large_qubit.section9_parallel_policy import policy as old_policy
from large_qubit import section9_recovery_policy as p
from large_qubit import section9_recovery_registration as r


class PolicyTests(unittest.TestCase):
    def test_budgets_numerics_lanes_and_memory_ceiling_are_unchanged(self):
        old, new = old_policy(), p.policy()
        for key in ('scientific_protocol', 'pipeline_seed', 'inherited_inference', 'rng',
                    'logical_budget_includes_cache_hits_and_duplicates', 'lanes',
                    'vqe_actual_requests_without_failures', 'vqe_unused_allowance',
                    'performance_selection', 'failed_target_analysis', 'generation_interruption',
                    'failure_policy', 'circuit_resources', 'maximum_physics_worker_rss_bytes',
                    'combined_physics_rss_allowance_bytes'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['memory_guard']['limit_bytes'], 4 * 1024 ** 3)
        self.assertIn('VmHWM', new['memory_guard']['source'])

    def test_recovery_retains_failed_work_and_uses_one_fixed_case(self):
        value = p.policy()
        self.assertEqual(value['failed_campaign']['logical_requests'], 16)
        self.assertEqual(value['failed_campaign']['completion_attempts'], 0)
        counts = value['recovery_accounting']
        self.assertEqual(counts['new_normal_ordinary_requests'] + 16,
                         counts['cumulative_normal_ordinary_requests'])
        self.assertEqual(counts['new_maximum_ordinary_requests'] + 16,
                         counts['cumulative_maximum_ordinary_requests'])
        self.assertIn('never solve', value['reference_policy'])
        self.assertIn('exact failed-campaign', value['gpu_identity'])
        self.assertNotIn(p.OUTPUT_ROOT, p.FAILED_ROOT)


class EligibilityTests(unittest.TestCase):
    """Small synthetic receipts exercise eligibility without any energy evaluator."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / p.FAILED_ROOT
        save(self.base / 'completed.json', {'inventory': {'fixture': 'hash'},
             'matrix_execution_complete': True, 'all_trajectories_successful': False,
             'trajectories_accounted': 16, 'training_executed': False, 'pipeline_seeds': [22],
             'costs_path': p.FAILED_ROOT + '/costs.json'})
        save(self.base / 'costs.json', {'preserved': 'fixture'})
        save(self.base / 'gpu_allocation.json', {'fixture': True})
        save(self.base / 'binding.json', {'final_freeze_sha256': 'freeze-fixture'})
        for alias in p.TARGETS:
            folder = self.base / 'targets' / alias
            save(folder / 'physics_identity.json', {'cache_identity': {
                'geometry_id': alias, 'partition': 'final_test'}})
            save(folder / 'reference.json', {'status': 'completed', 'result': {'fixture': True}})
        for lane in p.lane_specs():
            for method in lane['methods']:
                job = self.base / 'lanes' / lane['lane_id'] / p.job_id(method, lane['target'])
                save(job / 'report.json', {'status': 'failed', 'ledger_head': 'head',
                    'logical_requests': 1, 'completion_attempts': 0, 'counts': {
                        'logical_requests': 1, 'failed_requests': 1, 'completed_requests': 0,
                        'completion_attempts': 0, 'pending_requests': 0, 'abandoned_requests': 0,
                        'cache_hits': 0, 'duplicate_requests': 0}})
                save(job / 'identity.json', {'method': method, 'geometry_id': lane['target'],
                                            'lane_id': lane['lane_id']})
        self.snapshot = {'logical_requests': 1, 'failed_requests': 1, 'completed_requests': 0,
            'pending_requests': 0, 'abandoned_requests': 0, 'pending_attempts': 0,
            'attempts': [], 'cache_hits': 0, 'duplicate_requests': 0,
            'completion_attempts': 0, 'incumbent': None, 'requests': [{
                'kind': 'hf', 'status': 'failed', 'query': {'kind': 'hf', 'geometry_id': 'CO2_I01'},
                'energy_hartree': None, 'evaluation': {
                    'error': 'Target scorer exceeds the 4 GiB process RSS ceiling',
                    'evaluator_calls': 1, 'kernel_contractions': 1,
                    'hamiltonian_action_seconds': 2.0, 'wall_seconds': 3.0}}]}
        patches = [patch.object(r, 'FAILED_COMPLETION_SHA', sha256_file(self.base / 'completed.json')),
                   patch('large_qubit.section9_parallel._inventory', return_value={'fixture': 'hash'}),
                   patch('large_qubit.section9_parallel.validate_allocation'),
                   patch('large_qubit.section9_reporting.verify_reference'),
                   patch('large_qubit.section9_ledger.verify_ledger', side_effect=self._snapshot)]
        for mock in patches:
            mock.start()
            self.addCleanup(mock.stop)

    def _snapshot(self, path, identity, **kwargs):
        result = deepcopy(self.snapshot)
        result['requests'][0]['query']['geometry_id'] = identity['geometry_id']
        return result

    def test_retained_costs_and_reference_binding(self):
        result = r.failed_campaign(self.root)
        self.assertEqual(result['completion_attempts'], 0)
        self.assertEqual(result['trajectories'], 16)
        self.assertEqual(result['physical_cost_retained']['kernel_contractions'], 16)
        self.assertEqual(result['physical_cost_retained']['recorded_wall_seconds'], 48.0)
        self.assertEqual(set(result['reference_sha256']), set(p.TARGETS))
        self.assertEqual(result['final_freeze_sha256'], 'freeze-fixture')

    def test_success_or_candidate_generation_refuses_recovery(self):
        for key, changed in (('completion_attempts', 1), ('completed_requests', 1),
                             ('logical_requests', 2), ('failed_requests', 0), ('incumbent', {'x': 1}),
                             ('pending_requests', 1), ('abandoned_requests', 1), ('pending_attempts', 1),
                             ('attempts', ['candidate']), ('cache_hits', 1), ('duplicate_requests', 1)):
            with self.subTest(key=key):
                old = self.snapshot[key]
                self.snapshot[key] = changed
                with self.assertRaisesRegex(ValueError, 'exactly one failed HF'):
                    r.failed_campaign(self.root)
                self.snapshot[key] = old

    def test_other_failure_or_non_hf_cannot_use_recovery(self):
        for key, value in (('kind', 'circuit'), ('status', 'completed'), ('energy_hartree', -1.0)):
            with self.subTest(key=key):
                old = self.snapshot['requests'][0][key]
                self.snapshot['requests'][0][key] = value
                with self.assertRaisesRegex(ValueError, 'specific first-HF'):
                    r.failed_campaign(self.root)
                self.snapshot['requests'][0][key] = old
        self.snapshot['requests'][0]['evaluation']['error'] = 'different failure'
        with self.assertRaisesRegex(ValueError, 'specific first-HF'):
            r.failed_campaign(self.root)

    def test_report_counts_must_match_verified_ledger(self):
        job = self.base / 'lanes/lane_0/gpt2_xl_frozen__CO2_I01/report.json'
        value = read_json(job)['payload']
        value['counts']['cache_hits'] = 1
        job.unlink()
        save(job, value)
        with self.assertRaisesRegex(ValueError, 'report counts differ'):
            r.failed_campaign(self.root)

    def test_sealed_failure_inventory_or_reference_mutation_refuses(self):
        with patch('large_qubit.section9_parallel._inventory', return_value={'changed': 'hash'}), \
             self.assertRaisesRegex(ValueError, 'inventory changed'):
            r.failed_campaign(self.root)
        reference = self.base / 'targets' / 'CO2_I01' / 'reference.json'
        reference.unlink()
        save(reference, {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'completed target references'):
            r.failed_campaign(self.root)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = ('large_qubit/section9_recovery.py', 'large_qubit/section9_memory.py')
        for name in self.sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic fixture\n', encoding='utf-8')
        (self.root / 'research_plan_revised.md').write_text('fixture\n', encoding='utf-8')
        self.previous = self.root / p.PREDECESSOR
        self.previous.parent.mkdir(parents=True, exist_ok=True)
        self.previous.write_bytes(b'preserve\n')
        for mock in (patch.object(r, 'SOURCES', self.sources),
                     patch.object(r, 'predecessor', return_value={'hash': 'prior'}),
                     patch.object(r, 'failed_campaign', return_value={'hash': 'failed'})):
            mock.start()
            self.addCleanup(mock.stop)

    def test_registration_is_additive_and_does_not_start_recovery(self):
        first = r.register(self.root)
        manifest = self.root / p.REGISTRATION
        before = manifest.read_bytes()
        self.assertEqual(first, r.register(self.root))
        self.assertEqual(before, manifest.read_bytes())
        self.assertEqual(self.previous.read_bytes(), b'preserve\n')
        value = read_json(manifest)
        for key in ('training_executed', 'scientific_evaluation_executed', 'main_matrix_complete', 'execution_ready'):
            self.assertIs(value[key], False)
        self.assertTrue(value['requires_exact_preserved_hf_only_failure'])
        self.assertFalse((self.root / p.OUTPUT_ROOT).exists())

    def test_source_plan_policy_and_prior_failure_drift_are_rejected(self):
        r.register(self.root)
        for name, value in (('failed_campaign', {'hash': 'changed'}), ('policy', {'changed': True})):
            with patch.object(r, name, return_value=value), \
                 self.assertRaisesRegex(ValueError, 'implementation or preserved evidence changed'):
                r.verify(self.root)
        (self.root / self.sources[0]).write_text('# changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'implementation or preserved evidence changed'):
            r.register(self.root)

    def test_manifest_rehash_cannot_redefine_the_recovery(self):
        r.register(self.root)
        path = self.root / p.REGISTRATION
        value = read_json(path)
        value['policy']['recovery_accounting']['failed_requests_retained_as_infrastructure_overhead'] = 0
        path.write_bytes(canonical_bytes(value) + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            r.verify(self.root)
        path.with_suffix('.sha256').write_text(sha256_file(path), encoding='ascii')
        with self.assertRaisesRegex(ValueError, 'implementation or preserved evidence changed'):
            r.verify(self.root)

    def test_ineligible_campaign_leaves_no_partial_registration(self):
        with patch.object(r, 'failed_campaign', side_effect=ValueError('ineligible')), \
             self.assertRaisesRegex(ValueError, 'ineligible'):
            r.register(self.root)
        self.assertFalse((self.root / p.REGISTRATION).exists())


class PredecessorTests(unittest.TestCase):
    def test_exact_predecessor_bytes_and_original_closure_are_required(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value=r.PREDECESSOR_SHA), \
             patch('large_qubit.section9_parallel_registration.verify', return_value={'all_passed': True}) as verify:
            value = r.predecessor(directory)
            verify.assert_called_once_with(Path(directory).resolve())
            self.assertTrue(value['scientific_closure_authenticated'])
            self.assertEqual(value['manifest_sha256'], r.PREDECESSOR_SHA)

    def test_changed_predecessor_bytes_refuse_before_verifier(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value='changed'), \
             patch('large_qubit.section9_parallel_registration.verify') as verify:
            with self.assertRaisesRegex(ValueError, 'registration changed'):
                r.predecessor(directory)
            verify.assert_not_called()

    def test_false_original_closure_cannot_authorize_recovery(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value=r.PREDECESSOR_SHA), \
             patch('large_qubit.section9_parallel_registration.verify', return_value={'all_passed': False}), \
             self.assertRaisesRegex(ValueError, 'implementation closure failed'):
            r.predecessor(directory)


if __name__ == '__main__':
    unittest.main()
