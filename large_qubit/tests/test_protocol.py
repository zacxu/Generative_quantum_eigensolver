import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from large_qubit.common import digest, read_json, sha256_file, workspace_path
from large_qubit.definitions import build_definition
from large_qubit.protocol import (BENCHMARK_ID, FITS, METHODS, build_study,
                                 planned_jobs, require_execution_ready, seed_for, validate_study)
from large_qubit.registration import (DEFAULT_REGISTRATION, register, status,
                                      verify_registration)


class DecisionsTest(unittest.TestCase):
    def test_counts_and_hf_allowance(self):
        study = build_study()
        jobs = planned_jobs(study)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({j['job_id'] for j in jobs}), 16)
        self.assertEqual(sum(j['logical_request_cap'] for j in jobs), 16384)
        self.assertTrue(all(j['includes_hf'] and j['pipeline_seed'] == 22 for j in jobs))
        self.assertEqual(study['budget']['post_hf_requests_per_main_trajectory'], 1023)
        self.assertEqual(len(FITS), 3)
        self.assertEqual(len(METHODS), 8)

    def test_gpt_pair_has_only_objective_differences(self):
        fits = build_study()['training_conditions']
        pair = [copy.deepcopy(fits[n]) for n in FITS[:2]]
        for config in pair:
            for key in ('condition_id', 'objective', 'lambda_scale'):
                config.pop(key)
        self.assertEqual(pair[0], pair[1])
        self.assertEqual(pair[0]['learning_rate'], 0.0002)
        self.assertEqual(pair[0]['passes'], 5)
        self.assertEqual(pair[0]['checkpoint_selection'], 'final')

    def test_modified_scope_is_rejected(self):
        for field, value in [('pipeline_seeds', [22, 11]), ('targets', ['LIH_I01', 'LIH_I02']),
                             ('required_methods', list(METHODS) + ['qwen_uniform']),
                             ('final_protocol_frozen', True), ('benchmark_id', 'old-lih')]:
            with self.subTest(field=field):
                study = build_study()
                study[field] = value
                with self.assertRaises(ValueError):
                    validate_study(study)

    def test_scientific_and_boolean_type_mutations_rejected(self):
        for section, field, value in [('problem', 'electrons', 20), ('problem', 'qubits', 40),
                                     ('generation', 'qwen_enable_thinking', 0),
                                     ('generation', 'target_energy_feedback', True),
                                     ('budget', 'hf_included', 1),
                                     ('budget', 'requests_per_main_trajectory', 1023)]:
            with self.subTest(section=section, field=field):
                study = build_study()
                study[section][field] = value
                with self.assertRaises(ValueError):
                    validate_study(study)

    def test_old_adapters_and_fabricated_physics_do_not_enable_execution(self):
        study = build_study()
        with self.assertRaisesRegex(ValueError, 'not ready'):
            require_execution_ready(study)
        study['pending_bindings']['physics_and_orbital_alignment'] = 'old_lih_hash'
        with self.assertRaises(ValueError):
            require_execution_ready(study)
        study = build_study()
        study['training_conditions'][FITS[0]]['resume_from_historical_adapter'] = True
        with self.assertRaises(ValueError):
            validate_study(study)

    def test_seed_streams_deterministic_and_distinct(self):
        streams = ['acquisition', 'adapter_initialization', 'row_order', 'generation', 'vqe']
        values = [seed_for(x) for x in streams]
        self.assertEqual(values, [seed_for(x) for x in streams])
        self.assertEqual(len(set(values)), len(streams))
        self.assertNotEqual(seed_for('generation', group='a'), seed_for('generation', group='b'))
        with self.assertRaises(ValueError):
            seed_for('unknown')

    def test_imports_do_not_load_scientific_packages(self):
        script = ('import sys; import large_qubit.registration; '
                  'assert not ({"torch", "transformers", "numpy", "pennylane", "pyscf"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', script], check=True)


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, content in {
            'research_plan_revised.md': 'plan', 'requirements.txt': 'old==1',
            'phase6/requirements.txt': 'llm==2', 'large_qubit/README.md': 'readme',
            'large_qubit/claims.md': 'claims', 'large_qubit/protocol.py': '# fixture',
            'large_qubit/study.json': json.dumps(build_study()),
            'large_qubit/physical_definition.json': json.dumps(build_definition()),
            'large_qubit/section3_policies.json': '{}',
            'large_qubit/section3.md': 'Section 3 fixture',
            'large_qubit/requirements-chemistry.txt': 'numpy==2.1.2',
            'large_qubit/section4.md': 'Section 4 fixture',
            'large_qubit/section4_policy.json': '{}',
            'large_qubit/scripts/run_section4_tmux.sh': '# Section 4 wrapper fixture',
            'large_qubit/section5_policy.json': '{}',
            'large_qubit/catalogues/co2_screened_v1.json': '{}',
            'large_qubit/section5.md': 'Section 5 fixture',
            'large_qubit/scripts/run_section5_tmux.sh': '# Section 5 wrapper fixture',
            'large_qubit/amendments/section3_grid_v2.json': '{}',
            'large_qubit/amendments/section3_grid_v3.json': '{}',
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        for name, value in [('build_reuse_inventory', {'historical': True}),
                            ('verify_reuse_inventory', {'all_passed': True}),
                            ('verify_original_registration', {'all_passed': True,
                                                              'fixture_lineage': True}),
                            ('verify_section3_registration', {'all_passed': True,
                                                              'fixture_section3_lineage': True}),
                            ('verify_section4_registration', {'all_passed': True,
                                                              'fixture_section4_lineage': True}),
                            ('verify_section5_registration', {'all_passed': True,
                                                              'fixture_section5_lineage': True}),
                            ('verify_section6_registration', {'all_passed': True,
                                                              'fixture_section6_v1_lineage': True}),
                            ('verify_acquisition_amendment', {'all_passed': True,
                                                             'fixture_v2_authorization': True}),
                            ('verify_section6_v2_registration', {'all_passed': True,
                                                               'fixture_section6_v2_lineage': True}),
                            ('section7_implementation_binding', None),
                            ('verify_grid_amendment', {'all_passed': True,
                                                      'fixture_amendment': True}),
                            ('inspect_assets', {'metadata_authenticated': True})]:
            p = patch('large_qubit.registration.' + name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch('large_qubit.registration.REUSABLE_COMPONENTS', {})
        p.start()
        self.addCleanup(p.stop)

    def test_round_trip_and_status_explicitly_pending(self):
        register(self.root)
        self.assertTrue(verify_registration(self.root)['all_passed'])
        report = status(self.root)
        self.assertFalse(report['execution_ready'])
        self.assertFalse(report['final_protocol_frozen'])
        self.assertFalse(report['historical_lih_phase7_convergence_passed'])
        self.assertEqual(report['planned_target_trajectories'], 16)

    def test_existing_registration_cannot_be_overwritten(self):
        register(self.root)
        manifest = self.root / DEFAULT_REGISTRATION / 'manifest.json'
        original = manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            register(self.root)
        self.assertEqual(original, manifest.read_bytes())

    def test_changed_source_plan_dependencies_are_rejected(self):
        register(self.root)
        for name in ('research_plan_revised.md', 'large_qubit/protocol.py', 'requirements.txt'):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b' changed')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'changed'):
                verify_registration(self.root)
            path.write_bytes(original)

    def test_fake_final_freeze_rejected_even_with_rewritten_sidecar(self):
        register(self.root)
        folder = self.root / DEFAULT_REGISTRATION
        manifest = folder / 'manifest.json'
        doc = read_json(manifest)
        doc['execution_ready'] = True
        manifest.write_text(json.dumps(doc), encoding='utf-8')
        (folder / 'manifest.sha256').write_text(sha256_file(manifest), encoding='ascii')
        with self.assertRaises(ValueError):
            verify_registration(self.root)

    def test_output_escape_and_old_namespace_rejected_without_writes(self):
        for path in ('../outside', '/outside', 'C:/outside', 'phase7/new',
                     'large_qubit/registrations/../escape', 'large_qubit/registrations'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                register(self.root, path)
        self.assertFalse((self.root / 'large_qubit/registrations').exists())

    def test_resolved_output_escape_rejected(self):
        # Works on Windows without symlink creation privileges.
        real_resolve = Path.resolve
        target = self.root / 'large_qubit/registrations/escape'
        outside = self.root.parent / 'outside'
        def resolve(path, *args, **kwargs):
            return outside if path == target else real_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), self.assertRaises(ValueError):
            workspace_path(self.root, 'large_qubit/registrations/escape', namespace='large_qubit/registrations')

    def test_namespace_cannot_redirect_into_historical_runs(self):
        real_resolve = Path.resolve
        prefix = self.root / 'large_qubit/registrations'
        historical = self.root / 'phase7/runs'
        def resolve(path, *args, **kwargs):
            if path.is_relative_to(prefix):
                return historical / path.relative_to(prefix)
            return real_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), self.assertRaisesRegex(ValueError, 'redirect'):
            register(self.root, 'large_qubit/registrations/new')

    def test_sidecar_redirect_rejected(self):
        register(self.root)
        real_resolve = Path.resolve
        sidecar = self.root / DEFAULT_REGISTRATION / 'manifest.sha256'
        def resolve(path, *args, **kwargs):
            return self.root.parent / 'outside.sha256' if path == sidecar else real_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), self.assertRaises(ValueError):
            verify_registration(self.root)

    def test_strict_json(self):
        path = self.root / 'bad.json'
        for content in ('{"x": 1, "x": 2}', '{"x": NaN}'):
            path.write_text(content, encoding='utf-8')
            with self.assertRaises(ValueError):
                read_json(path)


if __name__ == '__main__':
    unittest.main()
