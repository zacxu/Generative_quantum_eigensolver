"""Bounded CO2 SCF/integral realization; no many-electron reference or training.

PySCF is imported only when an explicit realization is requested.  A converged
SCF is not automatically stable.  We form its small orbital-response Hessians
(44 by 44 here), diagonalize them, and fail closed for any unstable channel.
"""

import importlib
import platform
import time

import numpy as np

from .common import require
from .orbitals import (align_orbitals, canonicalize_anchor, closed_shell_hf_energy,
                       orbital_conventions, transform_integrals)


class PhysicalValidationError(ValueError):
    """Scientific failure with retained machine-readable diagnostics."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def scf_policy():
    return {
        'method': 'RHF', 'charge': 0, 'spin_2sz': 0,
        'required_pyscf_version': '2.11.0',
        'basis': 'sto-3g', 'cart': False, 'point_group_symmetry': False,
        'initial_guess': 'minao', 'maximum_cycles': 200,
        'energy_convergence_ha': 1e-12, 'gradient_convergence': 1e-9,
        'post_scf_gradient_norm_maximum': 1e-8,
        'stability_channels': ['internal_RHF', 'external_RHF_to_UHF', 'external_real_to_complex'],
        'stability_method': 'complete bounded orbital-response Hessians; symmetric eigensolver with residual checks',
        'stability_negative_eigenvalue_tolerance': 1e-7,
        'stability_hessian_symmetry_tolerance': 1e-8,
        'stability_eigenpair_residual_tolerance': 1e-8,
        'maximum_response_dimension': 44,
        'failure_policy': 'stop and retain diagnostics; no automatic orbital restart, UHF replacement or geometry change',
        'solution_choice': 'single deterministic minao-initialized converged stable closed-shell RHF; no energy-based multi-start selection',
        'hf_integral_energy_agreement_ha': 1e-8,
        'precision': 'float64 real AO/MO coefficients and integrals',
        'correlated_reference_computed': False,
    }


def inspect_response_hessian(action, dimension, *, scale=1.0):
    """Certify all eigenmodes of an explicitly bounded orbital-response kernel."""
    policy = scf_policy()
    require(type(dimension) is int and 0 < dimension <= policy['maximum_response_dimension'],
            'Orbital-response dimension exceeds the declared bounded stability analysis')
    basis = np.eye(dimension)
    raw = np.column_stack([np.asarray(action(basis[:, j])).reshape(-1)
                           for j in range(dimension)]) * scale
    require(raw.shape == (dimension, dimension), 'Orbital-response kernel dimensions differ')
    require(np.all(np.isfinite(raw)), 'Non-finite orbital-response Hessian')
    imaginary = float(np.max(np.abs(raw.imag))) if np.iscomplexobj(raw) else 0.0
    require(imaginary <= 1e-10, 'Real orbital-response Hessian has imaginary components')
    raw = raw.real
    symmetry = float(np.max(np.abs(raw - raw.T)))
    require(symmetry <= policy['stability_hessian_symmetry_tolerance'],
            'Orbital-response Hessian is not symmetric')
    symmetric = (raw + raw.T) * 0.5
    energies, vectors = np.linalg.eigh(symmetric)
    residual = float(np.max(np.linalg.norm(symmetric @ vectors - vectors * energies, axis=0)))
    require(residual <= policy['stability_eigenpair_residual_tolerance'],
            'Stability eigenpair residual exceeds tolerance')
    return {
        'dimension': dimension, 'minimum_eigenvalue': float(energies[0]),
        'eigenvalues': energies.tolist(), 'maximum_symmetry_residual': symmetry,
        'maximum_eigenpair_residual': residual,
        'stable': bool(energies[0] >= -policy['stability_negative_eigenvalue_tolerance']),
    }


def _stability(mf):
    # These are the kernels used in PySCF's documented stability module.  Its
    # public external status omits real->complex status, so inspect every mode.
    stability = importlib.import_module('pyscf.scf.stability')
    newton = importlib.import_module('pyscf.soscf.newton_ah')
    _, internal, diagonal = newton.gen_g_hop_rhf(
        mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    complex_action, _, unrestricted, _ = stability._gen_hop_rhf_external(
        mf, with_symmetry=False)
    dimension = len(diagonal)
    channels = {
        'internal_RHF': inspect_response_hessian(internal, dimension, scale=2.0),
        'external_RHF_to_UHF': inspect_response_hessian(unrestricted, dimension),
        'external_real_to_complex': inspect_response_hessian(complex_action, dimension),
    }
    return {'all_channels_stable': all(x['stable'] for x in channels.values()),
            'channels': channels,
            'backend_interfaces': ['pyscf.soscf.newton_ah.gen_g_hop_rhf',
                                   'pyscf.scf.stability._gen_hop_rhf_external'],
            'source': 'https://pyscf.org/_modules/pyscf/scf/stability.html'}


def prepare_geometry(geometry, anchor=None, *, threads=1):
    """Realize one declared geometry without writing files or solving FCI.

    ``geometry`` is the metadata returned by definitions.geometry().  For every
    geometry except CO2_T02, ``anchor`` is this function's CO2_T02 result.  Keep
    metadata and arrays in their declared physical-validation namespace, away
    from model inputs.  Returned ``molecule`` is a transient backend object.
    """
    require(type(threads) is int and threads > 0, 'threads must be a positive integer')
    from .definitions import geometry as defined_geometry
    require(isinstance(geometry, dict), 'A declared geometry dictionary is required')
    alias = geometry.get('geometry_id')
    declared = defined_geometry(alias, purpose='physical_validation')
    for field in ('geometry_id', 'partition', 'bond_length_angstrom', 'elements',
                  'coordinates_angstrom', 'coordinate_unit'):
        require(geometry.get(field) == declared[field], f'Undeclared geometry value: {field}')
    require(alias == 'CO2_T02' or anchor is not None,
            'Non-anchor geometries require the realized CO2_T02 anchor')
    if anchor is not None:
        require(anchor['metadata']['geometry']['geometry_id'] == 'CO2_T02',
                'Orbital gauge anchor must be CO2_T02')
        require(anchor['metadata']['physical_validation_passed'] is True,
                'Orbital gauge anchor has not passed physical validation')
    try:
        pyscf = importlib.import_module('pyscf')
    except ImportError as error:
        raise RuntimeError('CO2 realization requires the isolated Linux PySCF environment; no SCF or physical validation has run') from error
    require(pyscf.__version__ == scf_policy()['required_pyscf_version'],
            'PySCF version differs from the validated 2.11.0 response-kernel interfaces')
    from pyscf import gto, lib, scf
    started = time.perf_counter()
    policy = scf_policy()
    previous_threads = lib.num_threads()
    lib.num_threads(threads)
    metadata = {'geometry': declared, 'scf_policy': policy,
                'orbital_conventions': orbital_conventions(),
                'backend': {'pyscf': pyscf.__version__, 'numpy': np.__version__,
                            'python': platform.python_version(), 'platform': platform.platform(),
                            'threads': threads},
                'physical_validation_passed': False,
                'correlated_reference_computed': False,
                'scope': 'Section 3 geometry, SCF stability, AO/MO gauge and integral consistency only'}
    try:
        atoms = list(zip(declared['elements'], declared['coordinates_angstrom']))
        mol = gto.M(atom=atoms, unit='Angstrom', basis='sto-3g', charge=0,
                    spin=0, cart=False, symmetry=False, verbose=0)
        require(mol.nelec == (11, 11) and mol.nao_nr() == 15,
                'CO2 full-basis electron/orbital count differs from the declaration')
        mf = scf.RHF(mol)
        mf.init_guess = 'minao'
        mf.conv_tol = policy['energy_convergence_ha']
        mf.conv_tol_grad = policy['gradient_convergence']
        mf.max_cycle = policy['maximum_cycles']
        mf.chkfile = None
        cycles = []
        mf.callback = lambda env: cycles.append(int(env.get('cycle', -1)))
        mf.kernel()
        gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        metadata['scf'] = {'converged': bool(mf.converged),
                           'cycles': max(cycles) + 1 if cycles else None,
                           'total_energy_ha': float(mf.e_tot),
                           'gradient_norm': gradient,
                           'canonical_mo_energies_ha': mf.mo_energy.tolist(),
                           'canonical_mo_occupations': mf.mo_occ.astype(int).tolist()}
        if (not mf.converged or not np.isfinite(mf.e_tot) or not np.isfinite(gradient)
                or gradient > policy['post_scf_gradient_norm_maximum']):
            raise PhysicalValidationError('SCF did not satisfy the declared convergence policy', metadata)
        metadata['stability'] = _stability(mf)
        if not metadata['stability']['all_channels_stable']:
            raise PhysicalValidationError('SCF solution is unstable; stop before production and review retained diagnostics', metadata)
        overlap = mol.intor('int1e_ovlp')
        h_ao = mf.get_hcore()
        eri_ao = mol.intor('int2e', aosym='s1')
        if alias == 'CO2_T02':
            coeff, rotation, alignment = canonicalize_anchor(
                mf.mo_coeff, mf.mo_energy, overlap, mf.mo_occ)
            cross = overlap.copy()
        else:
            cross = gto.intor_cross('int1e_ovlp', anchor['molecule'], mol)
            coeff, rotation, alignment = align_orbitals(
                anchor['arrays']['mo_coefficients'], mf.mo_coeff, cross, overlap,
                mf.mo_energy, mf.mo_occ,
                anchor['metadata']['alignment']['occupation_order'])
        h_mo, eri_mo = transform_integrals(h_ao, eri_ao, coeff)
        nuclear = float(mol.energy_nuc())
        contracted = closed_shell_hf_energy(h_mo, eri_mo, 11, nuclear)
        difference = abs(contracted - float(mf.e_tot))
        metadata.update({
            'ao_labels': mol.ao_labels(), 'ao_count': int(mol.nao_nr()),
            'spatial_orbital_count': int(coeff.shape[1]), 'electron_population': list(mol.nelec),
            'basis_definition': {element: gto.basis.load('sto-3g', element) for element in ('C', 'O')},
            'coordinates_bohr': mol.atom_coords().tolist(),
            'nuclear_charges': mol.atom_charges().tolist(),
            'nuclear_repulsion_ha': nuclear, 'alignment': alignment,
            'integral_consistency': {'hf_contraction_total_energy_ha': contracted,
                                     'scf_energy_absolute_difference_ha': difference,
                                     'one_electron_hermiticity_max_abs': float(np.max(np.abs(h_mo - h_mo.T))),
                                     'eri_pair_exchange_max_abs': float(np.max(np.abs(eri_mo - eri_mo.transpose(2, 3, 0, 1)))),
                                     'eri_within_pair_max_abs': float(np.max(np.abs(eri_mo - eri_mo.swapaxes(0, 1))))},
        })
        consistency = metadata['integral_consistency']
        if difference > policy['hf_integral_energy_agreement_ha'] or any(
                consistency[key] > 1e-8 for key in ('one_electron_hermiticity_max_abs',
                                                   'eri_pair_exchange_max_abs',
                                                   'eri_within_pair_max_abs')):
            raise PhysicalValidationError('Aligned integrals failed independent HF/symmetry checks', metadata)
        metadata['physical_validation_passed'] = True
        return {'metadata': metadata, 'molecule': mol,
                'arrays': {'ao_overlap': overlap, 'cross_ao_overlap_to_anchor': cross,
                           'ao_one_electron_integrals': h_ao, 'ao_eri_chemists': eri_ao,
                           'canonical_mo_coefficients': np.asarray(mf.mo_coeff),
                           'canonical_mo_energies': np.asarray(mf.mo_energy),
                           'mo_coefficients': coeff, 'mo_rotation': rotation,
                           'one_electron_integrals': h_mo, 'eri_chemists': eri_mo}}
    except PhysicalValidationError:
        raise
    except Exception as error:
        metadata['error_type'] = type(error).__name__
        metadata['error'] = str(error)
        raise PhysicalValidationError('Physical validation failed; preserve diagnostics and do not produce Hamiltonians', metadata) from error
    finally:
        metadata['wall_seconds'] = time.perf_counter() - started
        lib.num_threads(previous_threads)
