"""Matrix-free molecular Hamiltonian action in a fixed alpha/beta sector.

The public CI layout is PySCF alpha-string rows and beta-string columns.  A
flat interleaved Jordan-Wigner state must pass through SectorBasis.to_ci();
reshaping it directly would omit fermionic reordering signs.
"""

from contextlib import contextmanager
import importlib
import math
import threading
import time

import numpy as np

from .common import digest, require


REQUIRED_PYSCF_VERSION = '2.11.0'
_NATIVE_THREAD_LOCK = threading.RLock()


def evaluator_policy():
    return {
        'backend': 'pyscf.fci.direct_spin1', 'required_pyscf_version': REQUIRED_PYSCF_VERSION,
        'ci_layout': 'alpha-major/beta-minor ascending cistring integer occupations',
        'operator': 'absorb_h1e(h1,eri,norb,(nalpha,nbeta),fac=0.5); contract_2e',
        'nuclear_constant': 'added exactly once to apply_h, including nonnormalized inputs',
        'main_state_precision': 'complex128', 'integral_precision': 'float64',
        'complex_action': 'separate real and imaginary native contractions',
        'real_reference_path': 'float64; compare with complex128 action before iterative solving',
        'energy_inner_product': 'numpy.vdot, require unit norm and negligible imaginary part',
        'normalization_tolerance': 1e-10, 'energy_imaginary_tolerance_ha': 1e-8,
        'dense_sector_matrix_constructed': False, 'full_qubit_vector_constructed': False,
        'gate_cache': 'owned and bounded separately by SectorBasis',
    }


def _validated_integrals(h1, eri, nuclear_repulsion, norb, populations, max_dimension):
    require(type(norb) is int and 1 <= norb < 64, 'Invalid spatial orbital count')
    require(isinstance(populations, (tuple, list)) and len(populations) == 2
            and all(type(n) is int and 0 <= n <= norb for n in populations),
            'Use explicit integer (nalpha,nbeta) populations')
    require(sum(populations) > 0, 'The empty-electron sector is outside this evaluator')
    shape = tuple(math.comb(norb, n) for n in populations)
    require(type(max_dimension) is int and max_dimension > 0
            and math.prod(shape) <= max_dimension, 'Sector dimension exceeds the explicit allocation bound')
    require(not np.iscomplexobj(h1) and not np.iscomplexobj(eri), 'Molecular integrals must be real')
    h1 = np.array(h1, dtype=np.float64, order='C', copy=True)
    eri = np.array(eri, dtype=np.float64, order='C', copy=True)
    require(h1.shape == (norb, norb) and eri.shape == (norb,) * 4,
            'Full h1 and chemists-order four-index ERI shapes are required')
    require(np.all(np.isfinite(h1)) and np.all(np.isfinite(eri)), 'Non-finite molecular integrals')
    require(not isinstance(nuclear_repulsion, (bool, complex)) and np.isfinite(nuclear_repulsion),
            'Nuclear repulsion must be a finite real scalar')
    require(np.max(np.abs(h1 - h1.T)) <= 1e-10, 'h1 is not real symmetric')
    require(np.max(np.abs(eri - eri.swapaxes(0, 1))) <= 1e-10
            and np.max(np.abs(eri - eri.swapaxes(2, 3))) <= 1e-10
            and np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 1e-10,
            'ERI lacks real chemists-order permutation symmetry')
    return h1, eri, float(nuclear_repulsion), shape


def _backend():
    try:
        pyscf = importlib.import_module('pyscf')
    except ImportError as error:
        raise RuntimeError('Matrix-free action requires the isolated local PySCF 2.11.0 environment') from error
    require(pyscf.__version__ == REQUIRED_PYSCF_VERSION, 'Unsupported PySCF version for fixed-sector kernels')
    direct = importlib.import_module('pyscf.fci.direct_spin1')
    cistring = importlib.import_module('pyscf.fci.cistring')
    lib = importlib.import_module('pyscf.lib')
    return pyscf, direct, cistring, lib


class IntegralEvaluator:
    """Reusable integrals/links, with no full or dense-sector Hamiltonian.

    ``apply_h`` is linear and accepts nonnormalized CI vectors.  ``energy``
    validates normalization.  Neither operation assigns a total-spin quantum
    number from fixed spin populations.
    """

    def __init__(self, h1, eri, nuclear_repulsion, norb, populations, *, threads=1,
                 max_dimension=5_000_000):
        started = time.perf_counter()
        require(type(threads) is int and threads > 0, 'threads must be a positive integer')
        h1, eri, nuclear, shape = _validated_integrals(
            h1, eri, nuclear_repulsion, norb, populations, max_dimension)
        self.norb, self.populations = norb, tuple(populations)
        self.nalpha, self.nbeta = self.populations
        self.shape, self.dimension = shape, math.prod(shape)
        self.threads, self.nuclear_repulsion = threads, nuclear
        self.h1, self.eri = h1, eri
        self._pyscf, self._direct, self._cistring, self._lib = _backend()
        with self.native_threads():
            self._absorbed = np.ascontiguousarray(self._direct.absorb_h1e(
                self.h1, self.eri, norb, self.populations, .5), dtype=np.float64)
            alpha = self._cistring.gen_linkstr_index_trilidx(range(norb), self.nalpha)
            beta = (alpha if self.nalpha == self.nbeta else
                    self._cistring.gen_linkstr_index_trilidx(range(norb), self.nbeta))
        self._links = (np.ascontiguousarray(alpha), np.ascontiguousarray(beta))
        for value in (self.h1, self.eri, self._absorbed, *self._links):
            value.flags.writeable = False
        self._basis = None
        self._counts = {'apply_h_calls': 0, 'complex_actions': 0, 'real_actions': 0,
                        'kernel_contractions': 0, 'hamiltonian_action_seconds': 0.0}
        self._real_path_validation = None
        self.initialization_seconds = time.perf_counter() - started
        self.hamiltonian_identity = digest({
            'norb': norb, 'populations': list(self.populations), 'nuclear_repulsion': nuclear,
            'h1': h1.tolist(), 'eri': eri.tolist(), 'policy': evaluator_policy()})

    @contextmanager
    def native_threads(self):
        # PySCF's setting is process-global.  Same-process callers serialize;
        # independent worker processes retain their own explicit thread bound.
        with _NATIVE_THREAD_LOCK:
            previous = self._lib.num_threads()
            self._lib.num_threads(self.threads)
            try:
                yield
            finally:
                self._lib.num_threads(previous)

    @property
    def basis(self):
        if self._basis is None:
            from .sector import SectorBasis
            self._basis = SectorBasis(self.norb, *self.populations)
        return self._basis

    @property
    def backend_metadata(self):
        return {'backend': 'pyscf.fci.direct_spin1', 'pyscf': self._pyscf.__version__,
                'numpy': np.__version__, 'threads': self.threads,
                'norb': self.norb, 'populations': list(self.populations),
                'shape': list(self.shape), 'dimension': self.dimension,
                'hamiltonian_identity': self.hamiltonian_identity,
                'policy': evaluator_policy()}

    def statistics(self):
        unique_arrays = {id(a): a for a in (self.h1, self.eri, self._absorbed, *self._links)}
        return {**self._counts, 'initialization_seconds': self.initialization_seconds,
                'integral_and_link_bytes': sum(a.nbytes for a in unique_arrays.values()),
                'reference_diagonal_bytes': getattr(getattr(self, '_diagonal', None), 'nbytes', 0),
                'complex128_state_bytes': self.dimension * 16,
                'float64_state_bytes': self.dimension * 8,
                'dense_sector_matrix_bytes_allocated': 0,
                'real_path_validation': self._real_path_validation}

    def _ci(self, value, *, real=False):
        value = np.asarray(value)
        require(value.shape == self.shape, 'Expected explicit two-dimensional alpha/beta CI layout; convert interleaved states with SectorBasis.to_ci')
        require(not real or not np.iscomplexobj(value), 'The real path cannot discard imaginary amplitudes')
        value = np.asarray(value, dtype=np.float64 if real else np.complex128, order='C')
        require(np.all(np.isfinite(value)), 'Non-finite CI amplitudes')
        return value

    def _contract_real(self, vector):
        self._counts['kernel_contractions'] += 1
        return np.asarray(self._direct.contract_2e(
            self._absorbed, np.ascontiguousarray(vector, dtype=np.float64),
            self.norb, self.populations, link_index=self._links))

    def apply_h(self, state):
        """Return H|CI> in complex128, with the nuclear constant included once."""
        vector = self._ci(state)
        started = time.perf_counter()
        with self.native_threads():
            result = self._contract_real(vector.real).astype(np.complex128)
            # The imaginary contraction is needed for arbitrary complex states;
            # zero imaginary input has an exactly zero imaginary action.
            if np.any(vector.imag):
                result.imag = self._contract_real(vector.imag)
            result += self.nuclear_repulsion * vector
        self._counts['apply_h_calls'] += 1
        self._counts['complex_actions'] += 1
        self._counts['hamiltonian_action_seconds'] += time.perf_counter() - started
        require(np.all(np.isfinite(result)), 'Non-finite Hamiltonian action')
        return result

    def apply_h_real(self, state):
        """Real reference-only path, adopted only after explicit parity validation."""
        require(self._real_path_validation is not None,
                'Validate the float64 reference path against complex128 before adopting it')
        vector = self._ci(state, real=True)
        started = time.perf_counter()
        with self.native_threads():
            result = self._contract_real(vector) + self.nuclear_repulsion * vector
        self._counts['apply_h_calls'] += 1
        self._counts['real_actions'] += 1
        self._counts['hamiltonian_action_seconds'] += time.perf_counter() - started
        require(np.all(np.isfinite(result)), 'Non-finite real Hamiltonian action')
        return np.asarray(result, dtype=np.float64)

    def validate_real_path(self, *, seed=22, samples=2, tolerance=1e-10):
        require(type(samples) is int and samples >= 1 and tolerance > 0,
                'Invalid real-path validation settings')
        rng, errors = np.random.default_rng(seed), []
        # Temporarily permit the real implementation for this comparison only.
        prior = self._real_path_validation
        self._real_path_validation = {'in_progress': True}
        try:
            for _ in range(samples):
                vector = rng.normal(size=self.shape)
                vector /= np.linalg.norm(vector)
                baseline = self.apply_h(vector.astype(np.complex128))
                fast = self.apply_h_real(vector)
                errors.append(float(np.linalg.norm(baseline - fast)))
            require(max(errors) <= tolerance, 'float64 reference action disagrees with complex128')
            self._real_path_validation = {'all_passed': True, 'seed': int(seed),
                                          'samples': samples, 'maximum_vector_difference': max(errors),
                                          'tolerance': float(tolerance)}
        except Exception:
            self._real_path_validation = prior
            raise
        return dict(self._real_path_validation)

    def energy(self, state, *, norm_tolerance=1e-10):
        vector = self._ci(state)
        norm = np.vdot(vector, vector)
        require(abs(norm.real - 1.0) <= norm_tolerance and abs(norm.imag) <= 1e-12,
                'Energy requires a normalized CI state')
        value = np.vdot(vector, self.apply_h(vector))
        require(abs(value.imag) <= 1e-8, 'Hermitian energy has an imaginary component')
        return float(value.real)

    def hf_state(self, *, dtype=np.complex128):
        require(np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128')),
                'HF states use float64 or complex128')
        state = np.zeros(self.shape, dtype=dtype)
        state[0, 0] = 1
        return state

    def diagonal(self):
        """Full-sector diagonal for preconditioning; never a dense Hamiltonian."""
        if not hasattr(self, '_diagonal'):
            with self.native_threads():
                value = self._direct.make_hdiag(
                    self.h1, self.eri, self.norb, self.populations, compress=False)
            value = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
            require(value.shape == (self.dimension,) and np.all(np.isfinite(value)),
                    'Invalid fixed-sector Hamiltonian diagonal')
            value += self.nuclear_repulsion
            value.flags.writeable = False
            self._diagonal = value
        return self._diagonal

    def apply_sector(self, state, basis=None):
        basis = self.basis if basis is None else basis
        require((basis.norb, basis.nalpha, basis.nbeta) == (self.norb, *self.populations),
                'Sector basis and evaluator populations differ')
        return basis.from_ci(self.apply_h(basis.to_ci(state)))

    def energy_sector(self, state, basis=None, *, norm_tolerance=1e-10):
        basis = self.basis if basis is None else basis
        require((basis.norb, basis.nalpha, basis.nbeta) == (self.norb, *self.populations),
                'Sector basis and evaluator populations differ')
        return self.energy(basis.to_ci(state), norm_tolerance=norm_tolerance)

    def spin_diagnostics(self, state):
        """Report S^2 expectation/variance rather than infer purity from Sz."""
        vector = self._ci(state)
        require(abs(np.vdot(vector, vector).real - 1) <= 1e-10, 'Normalize spin diagnostic states')
        spin_op = importlib.import_module('pyscf.fci.spin_op')
        with self.native_threads():
            ss = np.asarray(spin_op.contract_ss(vector.real.copy(), self.norb, self.populations)).astype(np.complex128)
            if np.any(vector.imag):
                ss.imag = spin_op.contract_ss(vector.imag.copy(), self.norb, self.populations)
        value = np.vdot(vector, ss)
        require(abs(value.imag) <= 1e-8, 'S^2 expectation has an imaginary component')
        variance = max(0.0, float(np.vdot(ss, ss).real - value.real ** 2))
        return {'s_squared_expectation': float(value.real), 's_squared_variance': variance,
                'spin_projection': (self.nalpha - self.nbeta) / 2,
                'total_spin_enforced_by_solver': False}
