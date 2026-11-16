"""CPU-only prospective policy and immutable additive registration checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit.common import canonical_bytes, read_json, sha256_file
from large_qubit.section9_policy import policy as original_policy, jobs
from large_qubit import section9_parallel_policy as p
from large_qubit import section9_parallel_registration as r


class PolicyTests(unittest.TestCase):
    def test_complete_exclusive_lane_ownership(self):
        lanes = p.lane_specs()
        assigned = [(method, lane['target']) for lane in lanes for method in lane['methods']]
        self.assertCountEqual(assigned, jobs())
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual([lane['gpu_slot'] for lane in lanes], list(range(4)))
        self.assertEqual(len({lane['lane_id'] for lane in lanes}), 4)
        for lane in lanes:
            self.assertEqual(p.lane_spec(lane['lane_id']), lane)
            for method in lane['methods']:
                if method.startswith(('gpt2', 'qwen')):
                    self.assertTrue(method.startswith(lane['model_key']))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            p.lane_spec('lane_4')

    def test_scientific_budget_and_inference_are_unchanged(self):
        old, new = original_policy(), p.policy()
        for key in ('scientific_protocol', 'pipeline_seed', 'inherited_inference',
                    'rng', 'logical_budget_includes_cache_hits_and_duplicates',
                    'vqe_actual_requests_without_failures', 'vqe_unused_allowance',
                    'performance_selection', 'failed_target_analysis', 'reruns',
                    'generation_interruption', 'failure_policy', 'circuit_resources'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['inherited_inference']['inference_batch_size'], 1)
        self.assertEqual(new['job_order'], old['job_order'])

    def test_bounds_and_lane_isolation(self):
        value = p.policy()
        self.assertEqual(value['gpu_count'], 4)
        self.assertEqual(value['resident_target_workers'], 4)
        self.assertEqual(value['cpu_threads_per_worker'], 1)
        self.assertEqual(value['maximum_active_physical_queues'], 4)
        self.assertEqual(value['maximum_pending_scores'], 2)
        self.assertEqual(value['combined_physics_rss_allowance_bytes'],
                         4 * value['maximum_physics_worker_rss_bytes'])
        self.assertEqual(value['cache_partition'], 'final_test')
        self.assertIn('per lane', value['scalar_cache_ownership'])
        # Caller mutation cannot silently redefine the next registration's allocation.
        value['lanes'][0]['methods'].clear()
        self.assertEqual(len(p.policy()['lanes'][0]['methods']), 5)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = ('large_qubit/section9_parallel.py',
                        'large_qubit/tests/test_section9_parallel.py')
        for name in self.sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic fixture; no scientific execution\n', encoding='utf-8')
        (self.root / 'research_plan_revised.md').write_text('Fixture plan\n', encoding='utf-8')
        old = self.root / r.PREDECESSOR
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_bytes(b'preserved-original-registration\n')
        self.original_bytes = old.read_bytes()
        self.sources_patch = patch.object(r, 'SOURCES', self.sources)
        self.predecessor_patch = patch.object(r, 'predecessor', return_value={
            'manifest_sha256': r.PREDECESSOR_SHA, 'scientific_closure_authenticated': True,
            'original_section9_sources_unchanged': True})
        self.sources_patch.start()
        self.predecessor_patch.start()
        self.addCleanup(self.sources_patch.stop)
        self.addCleanup(self.predecessor_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def test_registration_is_additive_immutable_and_not_science(self):
        first = r.register(self.root)
        manifest = self.root / r.REGISTRATION
        bytes_before = manifest.read_bytes()
        second = r.register(self.root)
        self.assertEqual(first, second)
        self.assertEqual(manifest.read_bytes(), bytes_before)
        self.assertEqual((self.root / r.PREDECESSOR).read_bytes(), self.original_bytes)
        saved = read_json(manifest)
        self.assertEqual(set(saved['bindings_sha256']), set(self.sources))
        self.assertFalse(saved['training_executed'])
        self.assertFalse(saved['scientific_evaluation_executed'])
        self.assertFalse(saved['main_matrix_complete'])
        self.assertFalse(saved['execution_ready'])
        self.assertTrue(saved['requires_completed_section8_freeze'])
        self.assertTrue(saved['requires_no_previous_main_execution'])
        self.assertFalse((self.root / p.PREPARATION).exists())
        self.assertFalse((self.root / p.OUTPUT_ROOT).exists())

    def test_source_tampering_refuses_verify_and_reregistration(self):
        r.register(self.root)
        path = self.root / self.sources[0]
        path.write_text('# changed\n', encoding='utf-8')
        for check in (r.verify, r.register):
            with self.assertRaisesRegex(ValueError, 'implementation or original evidence changed'):
                check(self.root)

    def test_research_plan_and_policy_drift_are_bound(self):
        r.register(self.root)
        changed = p.policy()
        changed['gpu_count'] = 3
        with patch.object(r, 'policy', return_value=changed), \
             self.assertRaisesRegex(ValueError, 'implementation or original evidence changed'):
            r.verify(self.root)
        (self.root / 'research_plan_revised.md').write_text('Changed plan\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'implementation or original evidence changed'):
            r.verify(self.root)

    def test_manifest_and_checksum_cannot_redefine_policy(self):
        r.register(self.root)
        manifest = self.root / r.REGISTRATION
        value = read_json(manifest)
        value['policy']['scientific_protocol']['requests_per_trajectory'] = 2048
        manifest.write_bytes(canonical_bytes(value) + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            r.verify(self.root)
        manifest.with_suffix('.sha256').write_text(sha256_file(manifest) + '\n', encoding='ascii')
        with self.assertRaisesRegex(ValueError, 'implementation or original evidence changed'):
            r.verify(self.root)

    def test_predecessor_failure_prevents_partial_registration(self):
        with patch.object(r, 'predecessor', side_effect=ValueError('Original closure changed')), \
             self.assertRaisesRegex(ValueError, 'Original closure changed'):
            r.register(self.root)
        self.assertFalse((self.root / r.REGISTRATION).exists())


class PredecessorTests(unittest.TestCase):
    def test_predecessor_requires_expected_bytes_and_verifies_original_sources(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value=r.PREDECESSOR_SHA), \
             patch('large_qubit.section9_registration.verify', return_value={'all_passed': True}) as check:
            result = r.predecessor(directory)
            check.assert_called_once_with(Path(directory).resolve())
            self.assertTrue(result['scientific_closure_authenticated'])
            self.assertEqual(result['manifest_sha256'], r.PREDECESSOR_SHA)

    def test_predecessor_changed_bytes_refuse_before_loading_source_closure(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value='changed'), \
             patch('large_qubit.section9_registration.verify') as check:
            with self.assertRaisesRegex(ValueError, 'Original Section9 registration changed'):
                r.predecessor(directory)
            check.assert_not_called()

    def test_predecessor_failed_closure_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, 'sha256_file', return_value=r.PREDECESSOR_SHA), \
             patch('large_qubit.section9_registration.verify', return_value={'all_passed': False}), \
             self.assertRaisesRegex(ValueError, 'implementation closure failed'):
            r.predecessor(directory)


if __name__ == '__main__':
    unittest.main()
