"""Artifact tests use synthetic physical payloads, never CO2 SCF energies."""

import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from large_qubit.common import canonical_bytes, read_json, sha256_file
from large_qubit.chemistry import PhysicalValidationError, scf_policy
from large_qubit.definitions import GEOMETRY_IDS, geometry
from large_qubit.orbitals import align_orbitals, canonicalize_anchor, orbital_conventions
from large_qubit.physics import DEFAULT_PHYSICS, build_physics, verify_physics


def synthetic_geometry(definition, anchor=None, *, threads=1):
    occupations = [2] * 11 + [0] * 4
    if definition['geometry_id'] == 'CO2_T02':
        _, _, alignment = canonicalize_anchor(np.eye(15), np.arange(15), np.eye(15), occupations)
    else:
        _, _, alignment = align_orbitals(anchor['arrays']['mo_coefficients'], np.eye(15),
                                         np.eye(15), np.eye(15), np.arange(15), occupations)
    stability_channel = {'dimension': 44, 'minimum_eigenvalue': 1., 'eigenvalues': [1.] * 44,
                         'maximum_symmetry_residual': 0., 'maximum_eigenpair_residual': 0.,
                         'stable': True}
    metadata = {
        'geometry': definition, 'physical_validation_passed': True,
        'correlated_reference_computed': False, 'scf_policy': scf_policy(),
        'orbital_conventions': orbital_conventions(),
        'backend': {'pyscf': 'synthetic-test-fixture', 'numpy': np.__version__, 'threads': threads},
        'ao_count': 15, 'spatial_orbital_count': 15, 'electron_population': [11, 11],
        'ao_labels': [f'fixture AO {i}' for i in range(15)],
        'basis_definition': {'fixture': 'synthetic_not_CO2'},
        'nuclear_repulsion_ha': 2., 'alignment': alignment,
        'scf': {'converged': True, 'gradient_norm': 0., 'total_energy_ha': 2.,
                'canonical_mo_occupations': occupations},
        'stability': {'all_channels_stable': True, 'channels': {
            name: copy.deepcopy(stability_channel) for name in scf_policy()['stability_channels']}},
        'integral_consistency': {'scf_energy_absolute_difference_ha': 0.,
                                'hf_contraction_total_energy_ha': 2.,
                                'one_electron_hermiticity_max_abs': 0.,
                                'eri_pair_exchange_max_abs': 0., 'eri_within_pair_max_abs': 0.},
    }
    arrays = {'ao_overlap': np.eye(15), 'cross_ao_overlap_to_anchor': np.eye(15),
              'ao_one_electron_integrals': np.zeros((15, 15)),
              'ao_eri_chemists': np.zeros((15,) * 4),
              'canonical_mo_coefficients': np.eye(15),
              'canonical_mo_energies': np.arange(15, dtype=float),
              'mo_coefficients': np.eye(15), 'mo_rotation': np.eye(15),
              'one_electron_integrals': np.zeros((15, 15)),
              'eri_chemists': np.zeros((15,) * 4)}
    return {'metadata': metadata, 'arrays': arrays, 'molecule': object()}


class PhysicsArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = Path(__file__).resolve().parents[2]
        (self.root / 'large_qubit').mkdir()
        for name in ('chemistry.py', 'orbitals.py'):
            shutil.copyfile(self.original / 'large_qubit' / name, self.root / 'large_qubit' / name)
        self.patch = patch('large_qubit.chemistry.prepare_geometry', side_effect=synthetic_geometry)
        self.prepare = self.patch.start()
        self.addCleanup(self.patch.stop)

    def build(self):
        return build_physics(self.root)

    def rewrite_manifest(self, manifest):
        path = self.root / DEFAULT_PHYSICS / 'manifest.json'
        path.write_bytes(canonical_bytes(manifest) + b'\n')
        path.with_suffix('.sha256').write_text(sha256_file(path) + '\n', encoding='ascii')

    def test_roundtrip_authenticates_all_payloads_and_rechecks_integrals(self):
        manifest = self.build()
        checked = verify_physics(self.root, numerical_integrity=True)
        self.assertTrue(checked['all_passed'])
        self.assertTrue(checked['numerical_integrity_checked'])
        self.assertEqual(checked['geometries_verified'], 6)
        self.assertEqual(checked['payload_files_verified'], 12)
        self.assertFalse(checked['scf_recomputed'])
        self.assertFalse(checked['correlated_reference_computed'])
        self.assertEqual([call.args[0]['geometry_id'] for call in self.prepare.call_args_list],
                         ['CO2_T02', 'CO2_T01', 'CO2_T03', 'CO2_V01', 'CO2_I01', 'CO2_I02'])
        self.assertEqual(set(manifest['geometries']), set(GEOMETRY_IDS))
        public = (self.root / DEFAULT_PHYSICS / 'manifest.json').read_text(encoding='utf-8')
        self.assertNotIn('total_energy_ha', public)
        self.assertNotIn('nuclear_repulsion_ha', public)
        self.assertNotIn('hf_contraction_total_energy_ha', public)

    def test_no_overwrite_preserves_prior_attempt(self):
        self.build()
        original_hash = sha256_file(self.root / DEFAULT_PHYSICS / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.build()
        self.assertEqual(original_hash, sha256_file(self.root / DEFAULT_PHYSICS / 'manifest.json'))

    def test_modified_and_missing_payloads_are_not_ignored(self):
        manifest = self.build()
        target = self.root / manifest['geometries']['CO2_I01']['arrays_path']
        target.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            verify_physics(self.root)
        target.unlink()
        with self.assertRaisesRegex(ValueError, 'missing restricted'):
            verify_physics(self.root)

    def test_rehashed_cross_geometry_metadata_cannot_pass(self):
        manifest = self.build()
        relative = manifest['geometries']['CO2_I01']['metadata_path']
        path = self.root / relative
        metadata = read_json(path)
        metadata['geometry'] = geometry('CO2_I02')
        path.write_bytes(canonical_bytes(metadata) + b'\n')
        manifest['files'][relative] = sha256_file(path)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            verify_physics(self.root)

    def test_rehashed_unstable_certificate_cannot_pass(self):
        manifest = self.build()
        relative = manifest['geometries']['CO2_T03']['metadata_path']
        path = self.root / relative
        metadata = read_json(path)
        metadata['stability']['channels']['external_RHF_to_UHF']['minimum_eigenvalue'] = -.2
        path.write_bytes(canonical_bytes(metadata) + b'\n')
        manifest['files'][relative] = sha256_file(path)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'stability certificate'):
            verify_physics(self.root)

    def test_rehashed_cross_overlap_alignment_metadata_cannot_pass_numerical_replay(self):
        manifest = self.build()
        relative = manifest['geometries']['CO2_I02']['metadata_path']
        path = self.root / relative
        metadata = read_json(path)
        metadata['alignment']['matched_overlap_diagonal'][0] = .7
        path.write_bytes(canonical_bytes(metadata) + b'\n')
        manifest['files'][relative] = sha256_file(path)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'alignment numerical metadata differs'):
            verify_physics(self.root, numerical_integrity=True)

    def test_different_integral_gauge_fails_even_when_payload_hashes_are_updated(self):
        manifest = self.build()
        entry = manifest['geometries']['CO2_T01']
        arrays_path = self.root / entry['arrays_path']
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays['one_electron_integrals'][0, 0] = 1.
        with arrays_path.open('wb') as handle:
            np.savez(handle, **arrays)
        manifest['files'][entry['arrays_path']] = sha256_file(arrays_path)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'tensor identity mismatch'):
            verify_physics(self.root, numerical_integrity=True)

    def test_failed_attempt_retains_diagnostics_without_passed_manifest(self):
        def fail_on_second(definition, anchor=None, *, threads=1):
            if definition['geometry_id'] == 'CO2_T01':
                raise PhysicalValidationError('unstable', {'unstable_eigenvalue': -.7})
            return synthetic_geometry(definition, anchor, threads=threads)
        self.prepare.side_effect = fail_on_second
        with self.assertRaises(PhysicalValidationError):
            self.build()
        folder = self.root / DEFAULT_PHYSICS
        self.assertFalse((folder / 'manifest.json').exists())
        failure = read_json(folder / 'failure.json')
        self.assertFalse(failure['all_passed'])
        self.assertEqual(failure['completed_geometries'], ['CO2_T02'])
        diagnostic = read_json(self.root / failure['diagnostics_path'])
        self.assertEqual(diagnostic['diagnostics']['unstable_eigenvalue'], -.7)
        with self.assertRaisesRegex(ValueError, 'recorded failure'):
            verify_physics(self.root)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.build()

    def test_paths_cannot_escape_or_collide_with_historical_namespaces(self):
        for output in ('phase7/physics', 'large_qubit/physics',
                       'large_qubit/physics/../archives/overwrite', 'large_qubit\\physics\\version'):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    build_physics(self.root, output)

    def test_hash_verification_does_not_import_scientific_packages(self):
        self.build()
        script = ('import sys; from large_qubit.physics import verify_physics; '
                  f'verify_physics({str(self.root)!r}); '
                  'assert not ({"numpy", "scipy", "pyscf", "torch"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', script], cwd=self.original, check=True)


if __name__ == '__main__':
    unittest.main()
