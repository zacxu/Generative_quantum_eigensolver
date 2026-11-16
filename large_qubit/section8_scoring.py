"""Isolated validation-only CPU scoring, with an explicit execution boundary.

Importing this module and --probe use only the standard library. The resident
chemistry worker loads only authenticated CO2_V01 arrays, never model weights,
test arrays, saved references, acquisition labels or candidate state caches.
The controller owns every logical request, durable receipt and scalar cache.
"""

import argparse
from contextlib import redirect_stdout
from importlib import metadata
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path


SCHEMA = 'section8-validation-scoring-v1'
GEOMETRY = 'CO2_V01'
PARTITION = 'validation'
PHYSICS = 'large_qubit/physics/section3_v3'
PHYSICS_SHA256 = '43b99fbde754bf61dd269046f1e664e70a8236ca9997908db0e2a3c32fbc78d0'
CATALOGUE = 'large_qubit/catalogues/co2_screened_v1.json'
CATALOGUE_SHA256 = '0a5fc7b1e21497e9fc4283a2e40829de8789f1e9f17e5f3726b878a549c4121c'
MAX_DIMENSION = 2_000_000
GATE_CACHE_BYTES = 64 * 1024 * 1024
MAX_RSS_BYTES = 4 * 1024**3
MAX_LINE_BYTES = 262144
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
REFERENCE_SETTINGS = {'seed': 22, 'starts': 2, 'ncv': 24, 'maxiter': 500,
                      'tol': 1e-11, 'residual_tolerance': 1e-8,
                      'agreement_tolerance': 1e-8, 'max_wall_seconds': 600.,
                      'max_matvecs': 2000, 'max_krylov_bytes': 1024**3}
SOURCES = tuple('large_qubit/' + name + '.py' for name in
                ('section8_scoring', 'common', 'physics', 'definitions', 'catalogue',
                 'sector', 'evaluator', 'reference'))


def probe():
    """Read package installation metadata; import no numerical/scoring code."""
    versions = {}
    for name in ('numpy', 'scipy', 'pyscf'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'schema_version': SCHEMA, 'executable': sys.executable,
            'python': sys.version, 'platform': sys.platform, 'packages': versions,
            'scientific_computation_performed': False, 'geometry_id': GEOMETRY}


def normalize_query(query):
    """Strict portable recipes; finite parameter values retain their exact floats."""
    require(type(query) is dict and query.get('geometry_id', GEOMETRY) == GEOMETRY,
            'Section 8 scoring permits CO2_V01 validation only')
    kind = query.get('kind')
    fields = {'hf': {'kind'}, 'discrete': {'kind', 'ops'}, 'vqe': {'kind', 'parameters'}}
    require(type(kind) is str and kind in fields and set(query) <= fields[kind] | {'geometry_id'},
            'Unknown scoring kind or unexpected query fields')
    if kind == 'hf':
        return {'kind': kind}
    name, count = ('ops', 32) if kind == 'discrete' else ('parameters', 24)
    values = query.get(name)
    require(type(values) is list and len(values) == count,
            f'{kind} requires exactly {count} values')
    if kind == 'discrete':
        require(all(type(x) is int and 0 <= x <= 512 for x in values),
                'Discrete IDs must be integers from 0 through 512')
        return {'kind': kind, name: list(values)}
    require(all(type(x) in (int, float) and math.isfinite(x) for x in values),
            'VQE parameters must be finite real numbers, excluding booleans')
    return {'kind': kind, name: [float(x) for x in values]}


def _thread_environment(threads):
    require(type(threads) is int and 1 <= threads <= 64, 'Use 1..64 explicit CPU threads')
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


class ValidationScorer:
    """A single serialized CPU evaluator; constructing it is scientific opt-in."""

    def __init__(self, workspace, *, execute=False, threads=1):
        require(execute is True, 'Validation scoring requires explicit execute=True')
        _thread_environment(threads)
        from .catalogue import validate_catalogue
        from .definitions import geometry
        from .evaluator import IntegralEvaluator
        from .physics import verify_physics
        import numpy as np

        self.root = Path(workspace).resolve(strict=True)
        geometry(GEOMETRY, purpose='validation')
        verification = verify_physics(self.root, PHYSICS, numerical_integrity=False)
        require(verification['manifest_sha256'] == PHYSICS_SHA256,
                'Section 8 physical manifest differs from its frozen predecessor')
        manifest = read_json(workspace_path(self.root, PHYSICS + '/manifest.json', namespace=PHYSICS))
        entry = manifest['geometries'][GEOMETRY]
        require(entry['geometry_id'] == GEOMETRY and entry['partition'] == PARTITION,
                'Validation geometry partition differs')
        self.catalogue = validate_catalogue(read_json(workspace_path(
            self.root, CATALOGUE, namespace='large_qubit/catalogues')))
        require(self.catalogue['sha256'] == CATALOGUE_SHA256, 'Section 8 catalogue differs')
        private = manifest['restricted_output']
        metadata_path = workspace_path(self.root, entry['metadata_path'], namespace=private)
        arrays_path = workspace_path(self.root, entry['arrays_path'], namespace=private)
        require(arrays_path.name == GEOMETRY + '.npz' and metadata_path.name == GEOMETRY + '.json',
                'Section 8 may load only validation arrays/metadata')
        physical = read_json(metadata_path)
        require(sha256_file(arrays_path) == manifest['files'][entry['arrays_path']],
                'Validation array payload changed')
        with np.load(arrays_path, allow_pickle=False) as arrays:
            self._h1 = np.array(arrays['one_electron_integrals'], dtype=np.float64, copy=True)
            self._eri = np.array(arrays['eri_chemists'], dtype=np.float64, copy=True)
        self._h1.flags.writeable = self._eri.flags.writeable = False
        self._nuclear = float(physical['nuclear_repulsion_ha'])
        self.threads = threads
        self._factory = IntegralEvaluator
        self.evaluator = self._new_evaluator()
        self.basis = self._new_basis()
        self._reference_computed = False
        runtime = {**probe(), 'cpu_threads': threads,
                   'thread_environment': {name: os.environ[name] for name in THREAD_VARIABLES},
                   'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}
        runtime.pop('scientific_computation_performed')
        self.identity = {
            'schema_version': SCHEMA, 'geometry_id': GEOMETRY, 'partition': PARTITION,
            'physics_manifest_sha256': PHYSICS_SHA256,
            'arrays_path': entry['arrays_path'], 'arrays_sha256': manifest['files'][entry['arrays_path']],
            'metadata_sha256': manifest['files'][entry['metadata_path']],
            'physical_hamiltonian_fingerprint': entry['hamiltonian_fingerprint'],
            'cache_identity': {'partition': PARTITION, 'geometry_id': GEOMETRY,
                'hamiltonian': self.evaluator.hamiltonian_identity,
                'orbitals': entry['orbital_fingerprint'], 'catalogue': self.catalogue['sha256'],
                'backend': digest(self.evaluator.backend_metadata),
                'precision': {'state': 'complex128', 'integrals': 'float64'}},
            'backend': self.evaluator.backend_metadata, 'runtime': runtime,
            'runtime_identity': digest(runtime), 'reference_settings': dict(REFERENCE_SETTINGS),
            'source_bindings': {p: sha256_file(workspace_path(self.root, p, namespace='large_qubit'))
                                for p in SOURCES},
            'memory_bounds': {'max_sector_dimension': MAX_DIMENSION,
                              'gate_cache_bytes': GATE_CACHE_BYTES, 'max_rss_bytes': MAX_RSS_BYTES,
                              'candidate_state_cache_entries': 0},
            'test_arrays_loaded': False, 'saved_references_loaded': False,
            'model_weights_loaded': False,
        }

    def _new_evaluator(self):
        return self._factory(self._h1, self._eri, self._nuclear, 15, (11, 11),
                             threads=self.threads, max_dimension=MAX_DIMENSION)

    def _new_basis(self, *, cache_bytes=GATE_CACHE_BYTES):
        from .sector import SectorBasis
        return SectorBasis(15, 11, 11, cache_budget_bytes=cache_bytes, max_dimension=MAX_DIMENSION)

    def _state(self, query, basis):
        from .catalogue import circuit_gates
        if query['kind'] == 'hf':
            return basis.hf_state()
        if query['kind'] == 'discrete':
            gates = circuit_gates(query['ops'], self.catalogue)
            family = 'whole_qubit'
        else:
            gates = [(self.catalogue['templates'][index]['wires'], theta)
                     for index, theta in zip(self.catalogue['vqe_template_indices'], query['parameters'])]
            family = 'fermionic'
        return basis.apply_circuit(basis.hf_state(), gates, family=family)

    def _measure(self, query, evaluator, basis, *, category):
        from .benchmark import _peak_rss_bytes
        query = normalize_query(query)
        started = time.perf_counter()
        before = evaluator.statistics()
        state = self._state(query, basis)
        prepared = time.perf_counter() - started
        energy = float(evaluator.energy_sector(state, basis))
        diagnostics = basis.diagnostics(state)
        del state
        after = evaluator.statistics()
        peak = _peak_rss_bytes()
        require(math.isfinite(energy) and diagnostics['n_alpha'] == 11 and diagnostics['n_beta'] == 11
                and abs(diagnostics['state_norm'] - 1.) <= 1e-10,
                'Validation energy, norm or populations failed')
        require(peak <= MAX_RSS_BYTES, 'Validation scoring exceeds the declared process memory bound')
        return {'energy_hartree': energy, 'geometry_id': GEOMETRY, 'partition': PARTITION,
                'query_sha256': digest(query), 'cache_identity': self.identity['cache_identity'],
                'norm': diagnostics['state_norm'], 'populations': [11, 11],
                'diagnostics': diagnostics, 'state_preparation_seconds': prepared,
                'wall_seconds': time.perf_counter() - started, 'peak_rss_bytes': peak,
                'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'kernel_contractions': after['kernel_contractions'] - before['kernel_contractions'],
                'hamiltonian_action_seconds': after['hamiltonian_action_seconds'] - before['hamiltonian_action_seconds'],
                'cost_category': category, 'physical_evaluation_performed': True,
                'scalar_cache_used': False, 'state_cache_used': False}

    def score(self, query):
        return self._measure(query, self.evaluator, self.basis, category='ordinary_energy_request')

    def verify(self, query):
        """Fresh basis/evaluator reconstruction, independent of search state/cache.

        This reuses the authenticated Hamiltonian implementation; it is not a
        claim of an independently implemented second Hamiltonian backend.
        """
        normalize_query(query)
        evaluator, basis = self._new_evaluator(), self._new_basis(cache_bytes=0)
        require(evaluator.hamiltonian_identity == self.evaluator.hamiltonian_identity,
                'Fresh integrity evaluator identity differs')
        result = self._measure(query, evaluator, basis, category='integrity_overhead_not_search')
        result.update({'fresh_basis_and_evaluator': True, 'independent_backend_claimed': False})
        return result

    def reference(self):
        from .reference import solve_reference
        from .benchmark import _peak_rss_bytes
        require(not self._reference_computed, 'Reference already solved by this worker; preserve its receipt')
        self._reference_computed = True
        started = time.perf_counter()
        before = self.evaluator.statistics()
        solved = solve_reference(self.evaluator, **REFERENCE_SETTINGS)
        report = solved['report']
        require(report['all_passed'] is True and report['converged'] is True
                and report['independent_starts'] == 2, 'Validation reference did not pass both starts')
        energy = float(solved['energy'])
        del solved
        peak = _peak_rss_bytes()
        require(math.isfinite(energy) and peak <= MAX_RSS_BYTES,
                'Validation reference energy or memory bound failed')
        after = self.evaluator.statistics()
        return {'energy_hartree': energy, 'report': report, 'geometry_id': GEOMETRY,
                'partition': PARTITION, 'cache_identity': self.identity['cache_identity'],
                'cost_category': 'validation_reference_overhead_not_search',
                'wall_seconds': time.perf_counter() - started, 'peak_rss_bytes': peak,
                'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'ordinary_energy_requests': 0, 'state_arrays_returned': False}


class ScoringError(RuntimeError):
    def __init__(self, message, *, report=None):
        super().__init__(message)
        self.report = report


def _loads(line):
    from .common import _pairs
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(line, object_pairs_hook=_pairs, parse_constant=reject)


def _emit(value, stream):
    stream.write(canonical_bytes(value).decode('ascii') + '\n')
    stream.flush()


def worker(workspace, *, execute=False, threads=1, input_stream=None, output_stream=None, factory=None):
    """JSON lines transport; native diagnostic output is directed to stderr."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    try:
        require(execute is True, 'Starting the chemistry worker requires --execute')
        with redirect_stdout(sys.stderr):
            scorer = (ValidationScorer if factory is None else factory)(workspace, execute=True, threads=threads)
        _emit({'event': 'ready', 'schema_version': SCHEMA, 'identity': scorer.identity}, output_stream)
        while True:
            line = input_stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                return 0
            require(len(line.encode('utf-8')) <= MAX_LINE_BYTES, 'Scoring protocol message exceeds bound')
            request = _loads(line)
            require(type(request) is dict and type(request.get('id')) is int
                    and set(request) <= {'id', 'action', 'query'}, 'Malformed scoring protocol request')
            ordinal = request['id']
            action = request.get('action')
            require(set(request) == ({'id', 'action'} if action in ('reference', 'close')
                                     else {'id', 'action', 'query'}),
                    'Unexpected or missing scoring action fields')
            if action == 'close':
                _emit({'id': ordinal, 'ok': True, 'result': {'closed': True}}, output_stream)
                return 0
            require(action in ('score', 'verify', 'reference'), 'Unknown scoring protocol action')
            try:
                with redirect_stdout(sys.stderr):
                    result = (scorer.reference() if action == 'reference'
                              else getattr(scorer, action)(normalize_query(request.get('query'))))
                _emit({'id': ordinal, 'ok': True, 'result': result}, output_stream)
            except Exception as error:
                _emit({'id': ordinal, 'ok': False, 'error': {'type': type(error).__name__,
                    'message': str(error), 'report': getattr(error, 'report', None)}}, output_stream)
                return 1  # Never silently retry an uncertain or failed scientific request.
    except Exception as error:
        _emit({'event': 'error', 'ok': False, 'error': {'type': type(error).__name__,
                'message': str(error), 'report': getattr(error, 'report', None)}}, output_stream)
        return 1


class PhysicsClient:
    """Serial model-free subprocess client using a separate chemistry interpreter."""

    def __init__(self, workspace, python_executable, *, execute=False, threads=1):
        require(type(execute) is bool and type(threads) is int and 1 <= threads <= 64,
                'Explicit execution flag and 1..64 CPU threads are required')
        self.root = Path(workspace).resolve(strict=True)
        self.python_executable = str(python_executable)
        self.execute, self.threads = execute, threads
        self.identity = None
        self._process = self._stderr = None
        self._messages, self._counter = queue.Queue(maxsize=4), 0
        self._lock = threading.Lock()
        if execute:
            self._start()

    def _start(self):
        env = dict(os.environ)
        env.update({name: str(self.threads) for name in THREAD_VARIABLES})
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        self._stderr = tempfile.TemporaryFile(mode='a+t', encoding='utf-8', errors='replace')
        try:
            self._process = subprocess.Popen([
                self.python_executable, '-B', '-m', 'large_qubit.section8_scoring',
                '--worker', '--root', str(self.root), '--threads', str(self.threads), '--execute'],
                cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding='utf-8', bufsize=1)
            threading.Thread(target=self._read_output, daemon=True).start()
            ready = self._receive(timeout=120)
            require(ready.get('event') == 'ready' and ready.get('schema_version') == SCHEMA,
                    'Chemistry worker failed its authenticated handshake: ' + str(ready))
            identity = ready['identity']
            require(identity['geometry_id'] == GEOMETRY and identity['partition'] == PARTITION
                    and identity['physics_manifest_sha256'] == PHYSICS_SHA256
                    and identity['cache_identity']['catalogue'] == CATALOGUE_SHA256
                    and identity['cache_identity']['partition'] == PARTITION
                    and identity['cache_identity']['geometry_id'] == GEOMETRY
                    and set(identity['cache_identity']) == {'partition', 'geometry_id', 'hamiltonian',
                                                          'orbitals', 'catalogue', 'backend', 'precision'},
                    'Chemistry worker returned a mismatched scoring identity')
            self.identity = identity
        except Exception:
            self.close(force=True)
            raise

    def _read_output(self):
        try:
            while True:
                line = self._process.stdout.readline(MAX_LINE_BYTES + 1)
                if not line:
                    self._messages.put(None)
                    return
                require(len(line.encode('utf-8')) <= MAX_LINE_BYTES, 'Worker response exceeds protocol bound')
                self._messages.put(_loads(line))
        except Exception as error:
            self._messages.put(error)

    def _receive(self, *, timeout=900):
        try:
            value = self._messages.get(timeout=timeout)
        except queue.Empty as error:
            raise ScoringError('Chemistry worker response timed out; preserve the pending request') from error
        if value is None or isinstance(value, Exception):
            raise ScoringError('Chemistry worker terminated or returned invalid protocol data; preserve the pending request')
        return value

    def _request(self, action, query=None, *, timeout=900):
        require(self.execute and self._process is not None, 'Scoring requires PhysicsClient(..., execute=True)')
        with self._lock:
            self._counter += 1
            request = {'id': self._counter, 'action': action}
            if query is not None:
                request['query'] = normalize_query(query)
            try:
                self._process.stdin.write(canonical_bytes(request).decode('ascii') + '\n')
                self._process.stdin.flush()
            except (OSError, ValueError) as error:
                raise ScoringError('Chemistry worker input failed; preserve the pending request') from error
            response = self._receive(timeout=timeout)
            require(response.get('id') == self._counter, 'Chemistry worker response/request ordinal differs')
            if response.get('ok') is not True:
                error = response.get('error', {})
                raise ScoringError(error.get('message', 'Chemistry scoring failed'), report=error.get('report'))
            return response['result']

    def score(self, query):
        return self._request('score', query)

    def verify(self, query):
        return self._request('verify', query)

    def reference(self):
        return self._request('reference')

    def stderr_tail(self, limit=16384):
        if self._stderr is None:
            return ''
        self._stderr.flush()
        self._stderr.seek(0, 2)
        size = self._stderr.tell()
        self._stderr.seek(max(0, size - limit))
        return self._stderr.read()

    def close(self, *, force=False):
        process = self._process
        if process is not None:
            try:
                if not force and process.poll() is None:
                    try:
                        self._request('close', timeout=5)
                    except Exception:
                        pass  # Cleanup must not replace the scientific failure/receipt.
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                for handle in (process.stdin, process.stdout):
                    if handle is not None:
                        handle.close()
                self._process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def __enter__(self):
        require(self.execute, 'A live scoring context requires explicit execute=True')
        return self

    def __exit__(self, error_type, *_):
        self.close(force=error_type is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.probe or not args.worker:
        print(json.dumps(probe(), indent=2, allow_nan=False))
        return 0
    return worker(args.root, execute=args.execute, threads=args.threads)


if __name__ == '__main__':
    raise SystemExit(main())
