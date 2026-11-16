"""Target-only CPU scoring behind the authenticated final Section 8 freeze.

No imports, previews or probes execute chemistry. An explicit worker owns one
resident target Hamiltonian; the coordinator may retain both target workers but
must serialize physical work globally. GPU models receive no reference values.
"""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from . import section8_scoring as inherited


SCHEMA = 'section9-final-target-scoring-v1'
TARGETS = ('CO2_I01', 'CO2_I02')
PARTITION = 'final_test'
REFERENCE_SETTINGS = dict(inherited.REFERENCE_SETTINGS)
MAX_RSS_BYTES = inherited.MAX_RSS_BYTES
MAX_LINE_BYTES = inherited.MAX_LINE_BYTES
SOURCES = tuple('large_qubit/' + name + '.py' for name in
                ('section9_scoring', 'section8_scoring', 'section8', 'section8_policy',
                 'common', 'physics', 'definitions', 'catalogue', 'sector', 'evaluator', 'reference'))
ScoringError = inherited.ScoringError


def target(value):
    require(type(value) is str and value in TARGETS,
            'Section 9 scoring permits only final-test CO2_I01 and CO2_I02')
    return value


def normalize_query(query, geometry_id=None):
    """Canonical queries always include their target alias, without rounding."""
    require(type(query) is dict, 'A plain target query mapping is required')
    alias = target(query.get('geometry_id') if geometry_id is None else geometry_id)
    require(query.get('geometry_id', alias) == alias, 'Cross-target query is forbidden')
    kind = query.get('kind')
    fields = {'hf': {'kind'}, 'discrete': {'kind', 'ops'}, 'vqe': {'kind', 'parameters'}}
    require(type(kind) is str and kind in fields and set(query) <= fields[kind] | {'geometry_id'},
            'Unknown scoring kind or unexpected query fields')
    result = {'geometry_id': alias, 'kind': kind}
    if kind == 'hf':
        return result
    name, count = ('ops', 32) if kind == 'discrete' else ('parameters', 24)
    values = query.get(name)
    require(type(values) is list and len(values) == count, f'{kind} requires exactly {count} values')
    if kind == 'discrete':
        require(all(type(x) is int and 0 <= x <= 512 for x in values),
                'Discrete IDs must be integers from 0 through 512')
        result[name] = list(values)
    else:
        try:
            valid = all(type(x) in (int, float) and math.isfinite(x) for x in values)
        except OverflowError:
            valid = False
        require(valid, 'VQE parameters must be finite real numbers, excluding booleans')
        result[name] = [float(x) for x in values]
    return result


def probe(geometry_id=None):
    value = inherited.probe()
    value.update(schema_version=SCHEMA, geometry_id=None if geometry_id is None else target(geometry_id),
                 allowed_targets=list(TARGETS), partition=PARTITION,
                 required_cpu_threads=1, maximum_rss_bytes_per_worker=MAX_RSS_BYTES,
                 maximum_two_worker_physics_rss_bytes=2 * MAX_RSS_BYTES)
    return value


def authenticate_freeze(workspace, expected_freeze_sha256):
    """Authenticate all prior evidence before admitting any target arrays."""
    from .section8 import load, verify_final_freeze
    from .section8_policy import FINAL_FREEZE, OUTPUT_ROOT
    root = Path(workspace).resolve(strict=True)
    require(type(expected_freeze_sha256) is str
            and re.fullmatch('[0-9a-f]{64}', expected_freeze_sha256) is not None,
            'Supply the exact expected final Section 8 freeze SHA256')
    path = workspace_path(root, FINAL_FREEZE, namespace='large_qubit/freezes')
    require(sha256_file(path) == expected_freeze_sha256, 'Final Section 8 freeze hash differs')
    checked = verify_final_freeze(root)
    require(checked.get('all_passed') is True and checked.get('section_8_complete') is True
            and checked.get('final_protocol_frozen') is True
            and checked.get('freeze_sha256') == expected_freeze_sha256,
            'Complete authenticated Section 8 freeze is required before final-test scoring')
    frozen = load(path)
    protocol = frozen['protocol']
    require(protocol['main']['targets'] == list(TARGETS) and protocol['cpu_threads'] == 1,
            'Frozen final targets or CPU thread policy differs')
    panel = frozen['selected_panel']
    require(type(panel) is str and re.fullmatch('panel_[0-9]{4}', panel) is not None,
            'Frozen validation panel identity differs')
    relative = OUTPUT_ROOT + '/' + panel + '/physics_identity.json'
    validation = load(workspace_path(root, relative, namespace=OUTPUT_ROOT))
    require(validation['reference_settings'] == REFERENCE_SETTINGS
            and validation['cache_identity']['precision'] == {'state': 'complex128', 'integrals': 'float64'}
            and validation['runtime']['cpu_threads'] == 1,
            'Final-test scoring must inherit frozen reference, precision and CPU settings')
    return frozen, validation, sha256_file(root / relative)


def _safe(value):
    """Keep failure diagnostics serializable, including nonfinite failure values."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return repr(value)


class TargetScorer(inherited.ValidationScorer):
    """Reuse immutable state/evaluator machinery with a new frozen target gate."""

    def __init__(self, workspace, geometry_id, *, execute=False, threads=1, expected_freeze_sha256=None):
        require(execute is True, 'Final-test scoring requires explicit execute=True')
        self.geometry_id = target(geometry_id)
        self.root = Path(workspace).resolve(strict=True)
        frozen, validation, validation_sha = authenticate_freeze(self.root, expected_freeze_sha256)
        require(type(threads) is int and threads == 1, 'Final-test CPU threads are frozen at one')
        current_probe = probe(self.geometry_id)
        require(current_probe['packages'] == validation['runtime']['packages']
                and current_probe['python'] == validation['runtime']['python'],
                'Final-test chemistry Python/packages differ from the frozen validation environment')
        inherited._thread_environment(threads)
        # All numerical imports and array reads follow the successful freeze gate.
        from .catalogue import validate_catalogue
        from .definitions import geometry
        from .evaluator import IntegralEvaluator
        from .physics import verify_physics
        import numpy as np

        geometry(self.geometry_id, purpose='final_test')
        verified = verify_physics(self.root, inherited.PHYSICS, numerical_integrity=False)
        require(verified['manifest_sha256'] == inherited.PHYSICS_SHA256, 'Frozen physical manifest differs')
        manifest = read_json(workspace_path(self.root, inherited.PHYSICS + '/manifest.json',
                                             namespace=inherited.PHYSICS))
        entry = manifest['geometries'][self.geometry_id]
        require(entry['geometry_id'] == self.geometry_id and entry['partition'] == PARTITION,
                'Final-test physical partition differs')
        self.catalogue = validate_catalogue(read_json(workspace_path(
            self.root, inherited.CATALOGUE, namespace='large_qubit/catalogues')))
        require(self.catalogue['sha256'] == inherited.CATALOGUE_SHA256
                == validation['cache_identity']['catalogue'], 'Frozen catalogue differs')
        private = manifest['restricted_output']
        arrays = workspace_path(self.root, entry['arrays_path'], namespace=private)
        meta = workspace_path(self.root, entry['metadata_path'], namespace=private)
        require(arrays.name == self.geometry_id + '.npz' and meta.name == self.geometry_id + '.json',
                'Cross-target or non-test array access is forbidden')
        require(sha256_file(arrays) == manifest['files'][entry['arrays_path']], 'Target array payload changed')
        physical = read_json(meta)
        with np.load(arrays, allow_pickle=False) as payload:
            self._h1 = np.array(payload['one_electron_integrals'], dtype=np.float64, copy=True)
            self._eri = np.array(payload['eri_chemists'], dtype=np.float64, copy=True)
        self._h1.flags.writeable = self._eri.flags.writeable = False
        self._nuclear = float(physical['nuclear_repulsion_ha'])
        self.threads, self._factory = threads, IntegralEvaluator
        self.evaluator, self.basis = self._new_evaluator(), self._new_basis()
        self._reference_computed = False
        runtime = {**probe(self.geometry_id), 'cpu_threads': threads,
                   'thread_environment': {key: os.environ[key] for key in inherited.THREAD_VARIABLES},
                   'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}
        runtime.pop('scientific_computation_performed')
        self.identity = {
            'schema_version': SCHEMA, 'geometry_id': self.geometry_id, 'partition': PARTITION,
            'final_freeze_sha256': expected_freeze_sha256,
            'frozen_validation_scoring_identity_sha256': validation_sha,
            'physics_manifest_sha256': inherited.PHYSICS_SHA256,
            'arrays_path': entry['arrays_path'], 'arrays_sha256': manifest['files'][entry['arrays_path']],
            'metadata_sha256': manifest['files'][entry['metadata_path']],
            'physical_hamiltonian_fingerprint': entry['hamiltonian_fingerprint'],
            'cache_identity': {'partition': PARTITION, 'geometry_id': self.geometry_id,
                'hamiltonian': self.evaluator.hamiltonian_identity, 'orbitals': entry['orbital_fingerprint'],
                'catalogue': self.catalogue['sha256'], 'backend': digest(self.evaluator.backend_metadata),
                'precision': dict(validation['cache_identity']['precision'])},
            'backend': self.evaluator.backend_metadata, 'runtime': runtime, 'runtime_identity': digest(runtime),
            'reference_settings': dict(REFERENCE_SETTINGS),
            'source_bindings': {p: sha256_file(workspace_path(self.root, p, namespace='large_qubit'))
                                for p in SOURCES},
            'memory_bounds': {'max_sector_dimension': inherited.MAX_DIMENSION,
                'gate_cache_bytes': inherited.GATE_CACHE_BYTES, 'max_rss_bytes': MAX_RSS_BYTES,
                'maximum_two_worker_physics_rss_bytes': 2 * MAX_RSS_BYTES,
                'candidate_state_cache_entries': 0},
            'saved_references_loaded': False, 'model_weights_loaded': False,
            'validation_arrays_loaded': False, 'other_target_arrays_loaded': False,
        }

    def _failure(self, stage, error, before, started, *, query=None, evaluator=None):
        evaluator = self.evaluator if evaluator is None else evaluator
        current = evaluator.statistics()
        original = getattr(error, 'report', None)
        report = dict(original) if isinstance(original, dict) else {}
        report.update({'partial': True, 'stage': stage, 'geometry_id': self.geometry_id,
            'partition': PARTITION, 'cache_identity': self.identity['cache_identity'],
            'cost_category': ('final_test_reference_overhead' if stage == 'reference'
                              else 'integrity_overhead_not_search' if stage == 'verify' else 'ordinary_energy_request'),
            'evaluator_calls': current['apply_h_calls'] - before['apply_h_calls'],
            'kernel_contractions': current['kernel_contractions'] - before['kernel_contractions'],
            'hamiltonian_action_seconds': current['hamiltonian_action_seconds'] - before['hamiltonian_action_seconds'],
            'wall_seconds': time.perf_counter() - started, 'error_type': type(error).__name__,
            'error': str(error), 'query': query, 'original_scientific_report': original})
        return ScoringError('Final-test ' + stage + ' failed: ' + str(error), report=_safe(report))

    def _measure(self, query, evaluator, basis, *, category):
        from .benchmark import _peak_rss_bytes
        started, before = time.perf_counter(), evaluator.statistics()
        try:
            query = normalize_query(query, self.geometry_id)
            state = self._state(query, basis)
            prepared = time.perf_counter() - started
            energy = float(evaluator.energy_sector(state, basis))
            diagnostics = basis.diagnostics(state)
            del state
            after, peak = evaluator.statistics(), _peak_rss_bytes()
            require(math.isfinite(energy) and diagnostics['n_alpha'] == 11 and diagnostics['n_beta'] == 11
                    and abs(diagnostics['state_norm'] - 1.) <= 1e-10,
                    'Final-test energy, norm or fixed populations failed')
            require(peak <= MAX_RSS_BYTES, 'Target scorer exceeds the 4 GiB process RSS ceiling')
            return {'energy_hartree': energy, 'geometry_id': self.geometry_id, 'partition': PARTITION,
                'query_sha256': digest(query), 'cache_identity': self.identity['cache_identity'],
                'norm': diagnostics['state_norm'], 'populations': [11, 11], 'diagnostics': diagnostics,
                'state_preparation_seconds': prepared, 'wall_seconds': time.perf_counter() - started,
                'peak_rss_bytes': peak, 'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'kernel_contractions': after['kernel_contractions'] - before['kernel_contractions'],
                'hamiltonian_action_seconds': after['hamiltonian_action_seconds'] - before['hamiltonian_action_seconds'],
                'cost_category': category, 'physical_evaluation_performed': True,
                'scalar_cache_used': False, 'state_cache_used': False}
        except Exception as error:
            raise self._failure('verify' if category.startswith('integrity') else 'score', error,
                                before, started, query=query, evaluator=evaluator) from error

    def score(self, query):
        return self._measure(query, self.evaluator, self.basis, category='ordinary_energy_request')

    def verify(self, query):
        query = normalize_query(query, self.geometry_id)
        started, before = time.perf_counter(), self.evaluator.statistics()
        try:
            evaluator, basis = self._new_evaluator(), self._new_basis(cache_bytes=0)
            require(evaluator.hamiltonian_identity == self.evaluator.hamiltonian_identity,
                    'Fresh integrity Hamiltonian differs')
            result = self._measure(query, evaluator, basis, category='integrity_overhead_not_search')
            result.update(fresh_basis_and_evaluator=True, independent_backend_claimed=False,
                          wall_seconds=time.perf_counter() - started)
            return result
        except ScoringError as error:
            if isinstance(error.report, dict):
                error.report['wall_seconds'] = time.perf_counter() - started
            raise
        except Exception as error:
            raise self._failure('verify', error, before, started, query=query) from error

    def reference(self):
        from .reference import solve_reference
        from .benchmark import _peak_rss_bytes
        require(not self._reference_computed, 'Reference already attempted; retain its completed or failure receipt')
        self._reference_computed = True
        started, before = time.perf_counter(), self.evaluator.statistics()
        report = None
        try:
            solved = solve_reference(self.evaluator, **REFERENCE_SETTINGS)
            report = solved['report']
            energy = float(solved['energy'])
            del solved
            peak, after = _peak_rss_bytes(), self.evaluator.statistics()
            require(report['all_passed'] is True and report['converged'] is True
                    and report['independent_starts'] == 2 and math.isfinite(energy),
                    'Target reference failed its independent-start convergence gate')
            require(peak <= MAX_RSS_BYTES, 'Target reference exceeds the 4 GiB process RSS ceiling')
            return {'energy_hartree': energy, 'report': report, 'geometry_id': self.geometry_id,
                'partition': PARTITION, 'cache_identity': self.identity['cache_identity'],
                'cost_category': 'final_test_reference_overhead_not_search',
                'wall_seconds': time.perf_counter() - started, 'peak_rss_bytes': peak,
                'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'ordinary_energy_requests': 0, 'state_arrays_returned': False}
        except Exception as error:
            if getattr(error, 'report', None) is None and report is not None:
                error.report = report
            raise self._failure('reference', error, before, started) from error


def worker(workspace, geometry_id, *, expected_freeze_sha256, execute=False, threads=1,
           input_stream=None, output_stream=None, factory=None):
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    try:
        require(execute is True, 'Starting a final-test worker requires --execute')
        alias = target(geometry_id)
        with redirect_stdout(sys.stderr):
            scorer = (TargetScorer if factory is None else factory)(workspace, alias, execute=True,
                threads=threads, expected_freeze_sha256=expected_freeze_sha256)
        inherited._emit({'event': 'ready', 'schema_version': SCHEMA, 'identity': scorer.identity}, output_stream)
        while True:
            line = input_stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                return 0
            require(len(line.encode('utf-8')) <= MAX_LINE_BYTES, 'Scoring message exceeds the protocol bound')
            request = inherited._loads(line)
            require(type(request) is dict and type(request.get('id')) is int, 'Malformed target scoring request')
            action = request.get('action')
            require(action in ('score', 'verify', 'reference', 'close')
                    and set(request) == ({'id', 'action'} if action in ('reference', 'close')
                                         else {'id', 'action', 'query'}), 'Unexpected scoring action fields')
            if action == 'close':
                inherited._emit({'id': request['id'], 'ok': True, 'result': {'closed': True}}, output_stream)
                return 0
            try:
                with redirect_stdout(sys.stderr):
                    result = (scorer.reference() if action == 'reference'
                              else getattr(scorer, action)(normalize_query(request['query'], alias)))
                inherited._emit({'id': request['id'], 'ok': True, 'result': result}, output_stream)
            except Exception as error:
                inherited._emit({'id': request['id'], 'ok': False, 'error': {'type': type(error).__name__,
                    'message': str(error), 'report': _safe(getattr(error, 'report', None))}}, output_stream)
                return 1
    except Exception as error:
        inherited._emit({'event': 'error', 'ok': False, 'error': {'type': type(error).__name__,
            'message': str(error), 'report': _safe(getattr(error, 'report', None))}}, output_stream)
        return 1


class PhysicsClient(inherited.PhysicsClient):
    """Reuse the frozen bounded transport; override only target admission/wire recipes."""

    def __init__(self, workspace, python_executable, geometry_id, *, execute=False, threads=1,
                 expected_freeze_sha256=None):
        self.geometry_id = target(geometry_id)
        self.expected_freeze_sha256 = expected_freeze_sha256
        require(type(execute) is bool and type(threads) is int and threads == 1,
                'Final-test execution is explicit and CPU threads are frozen at one')
        super().__init__(workspace, python_executable, execute=False, threads=threads)
        self.execute = execute
        if execute:
            self._start()

    def _start(self):
        require(type(self.expected_freeze_sha256) is str
                and re.fullmatch('[0-9a-f]{64}', self.expected_freeze_sha256) is not None,
                'Supply the exact expected final Section 8 freeze SHA256')
        env = dict(os.environ)
        env.update({name: str(self.threads) for name in inherited.THREAD_VARIABLES})
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        self._stderr = tempfile.TemporaryFile(mode='a+t', encoding='utf-8', errors='replace')
        try:
            self._process = subprocess.Popen([self.python_executable, '-B', '-m', 'large_qubit.section9_scoring',
                '--worker', '--root', str(self.root), '--geometry-id', self.geometry_id,
                '--freeze-sha256', self.expected_freeze_sha256, '--threads', '1', '--execute'],
                cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding='utf-8', bufsize=1)
            threading.Thread(target=self._read_output, daemon=True).start()
            ready = self._receive(timeout=180)
            require(ready.get('event') == 'ready' and ready.get('schema_version') == SCHEMA,
                    'Target worker failed its frozen handshake: ' + str(ready))
            identity = ready['identity']
            cache = identity['cache_identity']
            require(identity['geometry_id'] == self.geometry_id and identity['partition'] == PARTITION
                    and identity['final_freeze_sha256'] == self.expected_freeze_sha256
                    and identity['physics_manifest_sha256'] == inherited.PHYSICS_SHA256
                    and cache['geometry_id'] == self.geometry_id and cache['partition'] == PARTITION
                    and cache['catalogue'] == inherited.CATALOGUE_SHA256
                    and cache['precision'] == {'state': 'complex128', 'integrals': 'float64'}
                    and set(cache) == {'partition', 'geometry_id', 'hamiltonian', 'orbitals', 'catalogue', 'backend', 'precision'},
                    'Target worker returned an incompatible frozen scoring identity')
            self.identity = identity
        except Exception:
            self.close(force=True)
            raise

    def _request(self, action, query=None, *, timeout=900):
        require(self.execute and self._process is not None, 'Scoring requires PhysicsClient(..., execute=True)')
        with self._lock:
            self._counter += 1
            request = {'id': self._counter, 'action': action}
            if query is not None:
                request['query'] = normalize_query(query, self.geometry_id)
            try:
                self._process.stdin.write(canonical_bytes(request).decode('ascii') + '\n')
                self._process.stdin.flush()
            except (OSError, ValueError) as error:
                raise self._transport_error(action, 'Target worker input failed; preserve the pending request') from error
            try:
                response = self._receive(timeout=timeout)
            except ScoringError as error:
                raise self._transport_error(action, str(error)) from error
            require(response.get('id') == self._counter, 'Target response/request ordinal differs')
            if response.get('ok') is not True:
                error = response.get('error', {})
                raise ScoringError(error.get('message', 'Final-test scoring failed'), report=error.get('report'))
            return response['result']

    def _transport_error(self, action, message):
        return ScoringError(message, report={'stage': action, 'geometry_id': self.geometry_id,
            'partition': PARTITION, 'transport_failure': True, 'evaluator_calls_known': False,
            'possibly_unreported_physical_work': action != 'close',
            'worker_returncode': None if self._process is None else self._process.poll(),
            'stderr_tail': self.stderr_tail()})

    def close(self, *, force=False):
        """A dead worker's broken pipe must not replace its scientific failure."""
        process = self._process
        try:
            if process is not None:
                if not force and process.poll() is None:
                    try:
                        self._request('close', timeout=5)
                    except Exception:
                        pass
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            if process is not None:
                for handle in (process.stdin, process.stdout):
                    if handle is not None:
                        try:
                            handle.close()
                        except (OSError, ValueError):
                            pass
                self._process = None
            if self._stderr is not None:
                try:
                    self._stderr.close()
                except (OSError, ValueError):
                    pass
                self._stderr = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--geometry-id', choices=TARGETS)
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.probe or not args.worker:
        print(json.dumps(probe(args.geometry_id), indent=2, allow_nan=False))
        return 0
    return worker(args.root, args.geometry_id, expected_freeze_sha256=args.freeze_sha256,
                  execute=args.execute, threads=args.threads)


if __name__ == '__main__':
    raise SystemExit(main())
