"""Policy/provenance fixtures only; never runs CO2 physics, reference or training."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from large_qubit.benchmark import THREAD_VARIABLES, representative_circuits
from large_qubit.common import canonical_bytes, digest, read_json, sha256_file
from large_qubit.protocol import BENCHMARK_ID
from large_qubit import section4


def timing_fixture():
    return {'runs': [{'workers': 1, 'measured_requests': 8, 'measured_wall_seconds': 80.,
                      'aggregate_peak_rss_bytes': 1000},
                     {'workers': 2, 'measured_requests': 16, 'measured_wall_seconds': 50.,
                      'aggregate_peak_rss_bytes': 2000}]}


class PolicyScheduleTest(unittest.TestCase):
    def test_policy_changes_and_bool_budget_confusion_fail(self):
        self.assertEqual(section4.validate_section4_policy(section4.build_section4_policy()), section4.build_section4_policy())
        value = section4.build_section4_policy()
        value['reference']['starts'] = True
        with self.assertRaises(ValueError):
            section4.validate_section4_policy(value)

    def test_capacity_uses_measured_two_worker_rate_not_assumed_double_speed(self):
        result = section4.schedule_gate(timing_fixture(), 10., total_memory_bytes=10000)
        self.assertEqual(result['selected_concurrency'], 2)
        self.assertEqual(result['actual_serial_seconds_per_request'], 10.)
        self.assertEqual(result['serial_seconds_per_request'], 10.)
        self.assertEqual(result['measured_capacity_seconds_per_request'], 50. / 16)
        self.assertEqual(result['evaluator_allowance_seconds'], 3 * 86400)
        self.assertFalse(result['deadline_confirmed'])

    def test_memory_ineligible_faster_worker_pool_is_never_used(self):
        benchmark = timing_fixture()
        benchmark['runs'][1]['aggregate_peak_rss_bytes'] = 9000
        result = section4.schedule_gate(benchmark, 10., total_memory_bytes=10000)
        self.assertEqual(result['selected_concurrency'], 1)
        self.assertEqual(result['status'], 'no_go')
        benchmark['runs'][0]['aggregate_peak_rss_bytes'] = 8000
        self.assertFalse(section4.schedule_gate(benchmark, 10., total_memory_bytes=10000)['memory_passed'])

    def test_missing_zero_bool_and_nonfinite_measurements_fail_before_arithmetic(self):
        for field, value in [('measured_requests', 0), ('measured_requests', True),
                             ('measured_wall_seconds', float('inf')), ('measured_wall_seconds', float('nan')),
                             ('aggregate_peak_rss_bytes', False)]:
            benchmark = timing_fixture()
            benchmark['runs'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                section4.schedule_gate(benchmark, 10., total_memory_bytes=10000)
        for value in (float('inf'), -1., True):
            with self.subTest(reference=value), self.assertRaises(ValueError):
                section4.schedule_gate(timing_fixture(), value, total_memory_bytes=10000)


class Section4EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = 'large_qubit/validation/section4_fixture'
        self.private = f'large_qubit/scoring_only/{BENCHMARK_ID}/section4/section4_fixture'
        self.physics_digest = 'a' * 64
        self.physical_entry = {'hamiltonian_fingerprint': 'b' * 64, 'orbital_fingerprint': 'c' * 64}
        for source in section4.SOURCES:
            self.write(source, '# bound synthetic source; never executed\n')
        policy = section4.build_section4_policy()
        self.write(section4.POLICY_PATH, policy)
        self.write(policy['physics_manifest'], {'geometries': {'CO2_T02': self.physical_entry}})
        self.write(f'{self.private}/historical_regressions.json', {'synthetic_fixture': True})
        reference_report = {'all_passed': True, 'converged': True, 'maximum_complex128_residual_norm_ha': 1e-10,
                            'wall_seconds': 10., 'independent_starts': 2, 'hamiltonian_actions': 100}
        self.write(f'{self.private}/reference.json', {'geometry_id': 'CO2_T02', 'energy_hartree': -1.,
            'report': reference_report, **self.physical_entry, 'populations': [11, 11],
            'nuclear_repulsion_included': True, 'reference_is_not_model_input': True})
        self.write(f'{self.private}/reference.npz', b'synthetic bytes; default verification must never load this')
        benchmark = {**timing_fixture(), 'geometry': 'CO2_T02', 'logical_requests': 36, 'warmup_requests': 12,
                     'measured_requests': 24, 'reference_solving_included': False, 'model_training_started': False,
                     'scalar_energy_cache_used': False, 'total_memory_bytes': 10000, 'private_raw_log_sha256': {}}
        plans = representative_circuits()
        for run in benchmark['runs']:
            workers = run['workers']
            run.update(logical_requests=12 * workers, warmup_requests=4 * workers,
                       threads_per_worker=1, worker_reports=[])
            for worker in range(workers):
                rows = [{'event': 'worker_initialized', 'worker_id': worker, 'geometry': 'CO2_T02',
                         'threads': 1, 'native_thread_environment': {name: '1' for name in THREAD_VARIABLES},
                         'metadata': {**self.physical_entry, 'section3_manifest_sha256': self.physics_digest}}]
                timings, seen = [], {}
                for ordinal, plan in enumerate(plans, 1):
                    identity = {name: plan[name] for name in ('phase', 'family', 'length', 'sample', 'circuit_sha256')}
                    rows.append({'event': 'request_started', 'request_ordinal': ordinal, **identity,
                                 'gates': plan['gates'], 'energy_cache_hit': False,
                                 'duplicate_of': seen.get(plan['circuit_sha256'])})
                    seen.setdefault(plan['circuit_sha256'], ordinal)
                    timing = {**identity, 'request_ordinal': ordinal, 'state_preparation_seconds': .1,
                              'score_seconds': .2, 'hamiltonian_action_seconds': .15,
                              'gate_cache_bytes': 100, 'peak_rss_bytes': 500}
                    rows.append({'event': 'request_completed', **timing, 'energy_hartree': -1.,
                                 'normalization': {'norm_error': 0., 'sector_leakage_probability': 0.}})
                    timings.append({**timing, 'logging_seconds': .01})
                rows.append({'event': 'worker_finished', 'logical_requests': 12})
                relative = f'{self.private}/benchmark/concurrency_{workers}/worker_{worker:02d}.jsonl'
                self.write(relative, ''.join(canonical_bytes(row).decode('ascii') + '\n' for row in rows))
                benchmark['private_raw_log_sha256'][relative] = sha256_file(self.root / relative)
                run['worker_reports'].append({'worker_id': worker, 'logical_requests': 12,
                                              'measured_requests': 8, 'timings': timings})
        self.write(f'{self.private}/benchmark/benchmark_summary.json', benchmark)
        historical_sources = {}
        for relative in section4.HISTORICAL_INPUTS:
            self.write(relative, 'preserved historical fixture ' + relative)
            historical_sources[relative] = sha256_file(self.root / relative)
        self.manifest = {'schema_version': 'section4-validation-v1', 'benchmark_id': BENCHMARK_ID,
            'output': self.output, 'restricted_output': self.private, 'execution_ready': False,
            'training_executed': False, 'heldout_references_computed': False, 'section_4_complete': True,
            'policy': policy, 'policy_sha256': digest(policy), 'source_bindings': section4._source_bindings(self.root),
            'section3_manifest_sha256': self.physics_digest, 'correctness_passed': True, 'reference_converged': True,
            'historical_regressions': {'all_passed': True, 'source_files_sha256': historical_sources},
            'co2_correctness': {'all_passed': True},
            'reference': {'converged': True, 'residual_norm': 1e-10, 'wall_seconds': 10., 'starts': 2,
                          'hamiltonian_actions': 100}, 'benchmark': benchmark,
            'feasibility': section4.schedule_gate(benchmark, 10., total_memory_bytes=10000)}
        self.repin()
        context = patch('large_qubit.physics.verify_physics', return_value={'manifest_sha256': self.physics_digest})
        context.start()
        self.addCleanup(context.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         (value.encode('utf-8') if isinstance(value, str) else canonical_bytes(value) + b'\n'))

    def repin(self):
        self.manifest['files'] = {p.relative_to(self.root).as_posix(): sha256_file(p)
            for p in sorted((self.root / self.private).rglob('*')) if p.is_file()}
        self.write(f'{self.output}/manifest.json', self.manifest)
        self.write(f'{self.output}/manifest.sha256', sha256_file(self.root / self.output / 'manifest.json') + '\n')

    def verify(self):
        return section4.verify_section4(self.root, section4_root=self.output)

    def test_bound_fixture_verifies_without_loading_reference_arrays(self):
        self.assertTrue(self.verify()['all_passed'])
        script = ("import sys; from unittest.mock import patch; from large_qubit.section4 import verify_section4\n"
                  f"with patch('large_qubit.physics.verify_physics',return_value={{'manifest_sha256':'{self.physics_digest}'}}):\n"
                  " verify_section4(sys.argv[1],section4_root=sys.argv[2])\n"
                  "assert not ({'numpy','scipy','pyscf','torch','transformers'} & set(sys.modules))")
        subprocess.run([sys.executable, '-B', '-c', script, str(self.root), self.output], check=True)

    def test_sidecar_rewrite_does_not_authorize_changed_schedule(self):
        self.manifest['feasibility']['status'] = 'no_go'
        self.repin()
        with self.assertRaisesRegex(ValueError, 'feasibility decision'):
            self.verify()

    def test_changed_historical_input_is_rejected_even_when_new_payloads_match(self):
        self.write(section4.HISTORICAL_INPUTS[0], 'changed old scientific fixture')
        with self.assertRaisesRegex(ValueError, 'Historical regression input changed'):
            self.verify()

    def test_cross_namespace_missing_reference_and_public_energy_leak_are_refused(self):
        original = copy.deepcopy(self.manifest)
        self.manifest['restricted_output'] = f'large_qubit/scoring_only/{BENCHMARK_ID}/section4/other'
        self.repin()
        with self.assertRaisesRegex(ValueError, 'restricted output identity'):
            self.verify()
        self.manifest = copy.deepcopy(original)
        self.manifest['reference_energy_ha'] = -1.
        self.repin()
        with self.assertRaisesRegex(ValueError, 'only in scoring_only'):
            self.verify()
        self.manifest = copy.deepcopy(original)
        (self.root / self.private / 'reference.npz').unlink()
        self.repin()
        with self.assertRaisesRegex(ValueError, 'seven passed-attempt payloads'):
            self.verify()

    def test_bound_source_and_raw_payload_changes_fail(self):
        self.write('large_qubit/sector.py', '# modified source')
        with self.assertRaisesRegex(ValueError, 'numerical source changed'):
            self.verify()

    def test_rehashed_raw_ledger_cannot_omit_a_charged_request(self):
        relative = f'{self.private}/benchmark/concurrency_1/worker_00.jsonl'
        rows = (self.root / relative).read_text().splitlines()
        self.write(relative, '\n'.join(rows[:-2]) + '\n')
        self.repin()
        with self.assertRaisesRegex(ValueError, 'Raw worker ledger'):
            self.verify()

    def test_reference_metadata_must_match_fixed_training_hamiltonian(self):
        path = f'{self.private}/reference.json'
        ref = read_json(self.root / path)
        ref['hamiltonian_fingerprint'] = 'd' * 64
        self.write(path, ref)
        self.repin()
        with self.assertRaisesRegex(ValueError, 'Reference physical identity'):
            self.verify()

    def test_heldout_loader_access_and_existing_output_are_rejected_before_science(self):
        for alias in ('CO2_V01', 'CO2_I01', 'CO2_I02'):
            with self.subTest(alias=alias), patch('large_qubit.physics.verify_physics') as verifier:
                with self.assertRaises(ValueError):
                    section4.load_co2_evaluator(self.root, alias)
                verifier.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Preserve the existing'):
            section4.run_section4(self.root, self.output)


if __name__ == '__main__':
    unittest.main()
