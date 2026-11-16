"""Bounded iterative lowest-root evidence for the complete fixed-Sz sector.

Generic independent CI starts include every total-spin symmetry sector; the
diagonal determinant is augmented by unrestricted random CI, without spin
penalties or a singlet-only FCI solver. Residuals and
restart agreement are numerical evidence, not a rigorous spectral lower bound.
"""

import time

import numpy as np

from .common import require


class ReferenceFailure(RuntimeError):
    def __init__(self, message, report, state=None):
        super().__init__(message)
        self.report, self.state = report, state


def reference_policy():
    return {
        'solver': 'pyscf.lib.davidson1 with full-sector diagonal preconditioner; eigsh fallback for independent tiny fixtures',
        'operator': 'real symmetric matrix-free Hamiltonian action',
        'which': 'SA', 'roots_per_start': 1, 'default_independent_starts': 2,
        'starting_vectors': 'lowest-diagonal determinant plus independent dense standard-normal vector across the entire alpha/beta CI sector; no spin/point-group projection',
        'preconditioner': 'total Hamiltonian diagonal, level shift 1e-3 Ha; no dense pspace matrix',
        'default_ncv': 24, 'default_maxiter': 500, 'eigenvalue_relative_tolerance': 1e-11,
        'complex128_residual_tolerance_ha': 1e-8, 'independent_start_energy_agreement_ha': 1e-8,
        'default_max_wall_seconds': 600., 'default_max_matvecs': 2000,
        'cost_category': 'reference_solver_overhead_not_ordinary_circuit_energy_requests',
        'global_ground_claim': 'lowest converged full-fixed-sector Ritz root agreeing across independent unrestricted starts; not a rigorous lower-bound certificate',
        'spin_constraint': 'Nalpha,Nbeta only; no total-spin constraint',
        'dense_sector_matrix_constructed': False,
    }


def solve_reference(evaluator, *, seed=22, starts=2, ncv=24, maxiter=500,
                    tol=1e-11, residual_tolerance=1e-8,
                    agreement_tolerance=1e-8, max_wall_seconds=600.,
                    max_matvecs=2000, max_krylov_bytes=1_073_741_824,
                    progress_callback=None):
    """Return ``{'energy', 'state', 'report'}``, or retain diagnostics in failure.

    The state is complex128 CI in the same Hamiltonian/sector as the evaluator.
    A budget failure is never reported as a converged reference.  Time checks
    bracket native calls; an already-running native contraction is not killed.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
    import scipy
    require(type(seed) is int and seed >= 0, 'Reference seed must be a nonnegative integer')
    require(type(starts) is int and starts >= 2, 'Use at least two independent unrestricted reference starts')
    require(type(ncv) is int and ncv >= 3 and type(maxiter) is int and maxiter >= 1,
            'Invalid iterative solver bounds')
    require(0 < tol <= 1e-8 and 0 < residual_tolerance <= 1e-8
            and 0 < agreement_tolerance <= 1e-8, 'Reference tolerances cannot weaken the 1e-8 gate')
    require(np.isfinite(max_wall_seconds) and max_wall_seconds > 0
            and type(max_matvecs) is int and max_matvecs >= 4,
            'Invalid reference time/matvec budget')
    require(type(max_krylov_bytes) is int and max_krylov_bytes > 0, 'Invalid Krylov memory allowance')
    dimension = evaluator.dimension
    actual_ncv = min(ncv, dimension)
    # Davidson holds both vectors and their images, plus working vectors.
    basis_bytes = (2 * actual_ncv + 8) * dimension * 8
    require(basis_bytes <= max_krylov_bytes, 'Krylov basis exceeds the explicit reference memory allowance')
    started = time.perf_counter()
    before = evaluator.statistics()
    report = {
        'all_passed': False, 'converged': False, 'policy': reference_policy(),
        'hamiltonian_identity': evaluator.hamiltonian_identity,
        'backend': evaluator.backend_metadata, 'scipy': scipy.__version__,
        'dimension': dimension, 'shape': list(evaluator.shape),
        'seed': seed, 'independent_starts': starts,
        'requested_ncv': ncv, 'actual_ncv': actual_ncv,
        'maxiter': maxiter, 'tol': tol, 'residual_tolerance': residual_tolerance,
        'agreement_tolerance': agreement_tolerance,
        'max_wall_seconds': float(max_wall_seconds), 'max_matvecs': max_matvecs,
        'krylov_basis_bytes': basis_bytes, 'maximum_krylov_bytes': max_krylov_bytes,
        'memory_value_is_measured_peak': False,
        'time_budget_enforcement': 'before and after Hamiltonian actions; no interruption inside native contraction',
        'ordinary_circuit_energy_requests': 0, 'cost_category': 'reference_solver_overhead',
        'runs': [], 'stored_state_precision': 'complex128',
    }
    best_state, best_energy = None, float('inf')

    def update_cost():
        current = evaluator.statistics()
        report['wall_seconds'] = time.perf_counter() - started
        report['hamiltonian_actions'] = current['apply_h_calls'] - before['apply_h_calls']
        report['native_kernel_contractions'] = current['kernel_contractions'] - before['kernel_contractions']
        report['hamiltonian_action_seconds'] = (current['hamiltonian_action_seconds']
                                                - before['hamiltonian_action_seconds'])
        report['float64_actions'] = current['real_actions'] - before['real_actions']
        report['complex128_actions'] = current['complex_actions'] - before['complex_actions']
        return current

    def budget(*, before_action=False):
        update_cost()
        exhausted = (report['wall_seconds'] > max_wall_seconds
                     or report['hamiltonian_actions'] > max_matvecs
                     or (before_action and report['hamiltonian_actions'] >= max_matvecs))
        if exhausted:
            report['failure_reason'] = 'declared_reference_time_or_matvec_budget_exhausted'
            raise ReferenceFailure('Reference budget exhausted before convergence; preserve this attempt', report, best_state)

    try:
        budget(before_action=True)
        report['real_path_validation'] = evaluator.validate_real_path(seed=seed, samples=1, tolerance=1e-10)
        budget()

        def action(flat):
            budget(before_action=True)
            value = evaluator.apply_h_real(np.asarray(flat).reshape(evaluator.shape)).reshape(-1)
            budget()
            return value

        operator = LinearOperator((dimension, dimension), matvec=action, rmatvec=action, dtype=np.float64)
        use_davidson = callable(getattr(evaluator, 'diagonal', None))
        if use_davidson:
            from pyscf import lib
            diagonal_started = time.perf_counter()
            diagonal = evaluator.diagonal()
            report['diagonal_construction_seconds'] = time.perf_counter() - diagonal_started
            report['diagonal_bytes'] = int(diagonal.nbytes)
            report['solver_used'] = 'pyscf.lib.davidson1'
            precondition = lib.make_diag_precond(diagonal, level_shift=1e-3)
            budget()
        else:
            report['solver_used'] = 'scipy.sparse.linalg.eigsh_tiny_fixture_fallback'
        children = np.random.SeedSequence(seed).spawn(starts)
        for ordinal, child in enumerate(children):
            budget(before_action=True)
            rng = np.random.default_rng(child)
            initial = rng.standard_normal(dimension)
            initial /= np.linalg.norm(initial)
            run_started = time.perf_counter()
            run_action_start = evaluator.statistics()['apply_h_calls']
            run = {'ordinal': ordinal, 'seed_spawn_key': list(child.spawn_key),
                   'converged': False, 'starting_vector_symmetry_restricted': False}
            report['runs'].append(run)
            if dimension == 1:
                vector = np.ones(1)
                value = float(action(vector)[0])
            elif use_davidson:
                determinant = np.zeros(dimension)
                determinant[int(np.argmin(diagonal))] = 1.
                initial_block = [determinant, initial]
                def callback(environment):
                    if progress_callback is not None:
                        progress_callback({'start': ordinal, 'cycle': int(environment['icyc']),
                                           'residual_norm_ha': float(environment['max_dx_norm']),
                                           'wall_seconds': time.perf_counter() - started})
                with evaluator.native_threads():
                    converged, values, vectors = lib.davidson1(
                        lambda vectors: [action(vector) for vector in vectors],
                        initial_block, precondition, tol=tol, max_cycle=maxiter,
                        max_space=actual_ncv, max_memory=max_krylov_bytes / 1e6,
                        nroots=1, tol_residual=residual_tolerance * .1,
                        lindep=1e-24, verbose=0, callback=callback)
                run['native_solver_converged'] = bool(converged[0])
                value, vector = float(values[0]), vectors[0]
                del initial_block, determinant
                if not converged[0]:
                    report['failure_reason'] = 'Davidson_did_not_converge'
                    raise ReferenceFailure('Davidson did not converge within the declared bounds', report, best_state)
            else:
                values, vectors = eigsh(operator, k=1, which='SA', v0=initial,
                                         ncv=actual_ncv, maxiter=maxiter, tol=tol)
                value, vector = float(values[0]), vectors[:, 0]
            vector = np.asarray(vector, dtype=np.float64).reshape(evaluator.shape)
            vector /= np.linalg.norm(vector)
            # Fix a serialization phase without changing any physical claim.
            pivot = int(np.argmax(np.abs(vector)))
            if vector.reshape(-1)[pivot] < 0:
                vector *= -1
            budget(before_action=True)
            complex_state = vector.astype(np.complex128)
            h_state = evaluator.apply_h(complex_state)
            budget()
            rayleigh = np.vdot(complex_state, h_state)
            require(abs(rayleigh.imag) <= 1e-8, 'Reference energy has an imaginary component')
            value = float(rayleigh.real)
            residual = float(np.linalg.norm(h_state - value * complex_state))
            run.update({'energy_ha': value, 'complex128_residual_norm_ha': residual,
                        'norm_squared': float(np.vdot(complex_state, complex_state).real),
                        'hamiltonian_actions': evaluator.statistics()['apply_h_calls'] - run_action_start,
                        'wall_seconds': time.perf_counter() - run_started,
                        'converged': residual <= residual_tolerance})
            if residual > residual_tolerance:
                report['failure_reason'] = 'direct_complex128_reference_residual_exceeds_tolerance'
                raise ReferenceFailure('Iterative root failed the independent complex128 residual gate', report, complex_state)
            if value < best_energy:
                best_energy, best_state = value, complex_state
            del h_state, complex_state, vector, initial
        energies = [run['energy_ha'] for run in report['runs']]
        disagreement = max(energies) - min(energies)
        report['independent_start_energy_spread_ha'] = disagreement
        if disagreement > agreement_tolerance:
            report['failure_reason'] = 'independent_unrestricted_starts_disagree'
            raise ReferenceFailure('Independent reference starts did not agree on the lowest root', report, best_state)
        budget()
        report['spin_diagnostics'] = evaluator.spin_diagnostics(best_state)
        budget()
        report.update({'all_passed': True, 'converged': True,
                       'reference_energy_ha': best_energy,
                       'maximum_complex128_residual_norm_ha': max(run['complex128_residual_norm_ha'] for run in report['runs']),
                       'rigorous_global_ground_certificate': False,
                       'lowest_root_evidence': 'two_or_more_independent_full_sector_random_starts_agree_with_verified_residuals',
                       'spin_operator_diagnostic_actions': 1})
        return {'energy': best_energy, 'state': best_state, 'report': report}
    except ArpackNoConvergence as error:
        report['failure_reason'] = 'ARPACK_did_not_converge'
        report['partial_eigenvalues_ha'] = np.asarray(error.eigenvalues).tolist()
        raise ReferenceFailure('ARPACK did not converge within the declared bounds', report, best_state) from error
    except ReferenceFailure:
        raise
    except Exception as error:
        report['failure_reason'] = 'reference_solver_or_numeric_check_failed'
        report['error_type'], report['error'] = type(error).__name__, str(error)
        raise ReferenceFailure('Reference failed; retain diagnostics without a converged-reference claim', report, best_state) from error
    finally:
        update_cost()
