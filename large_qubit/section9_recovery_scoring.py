"""Additive Section 9 scorer correcting inherited Linux peak-RSS accounting.

The old scorer, scientific routines, precision, threads and 4 GiB limit stay
unchanged. This entry point measures the exec'd chemistry address space rather
than the GPU parent's inherited getrusage peak, and records both measurements.
"""

import argparse
import copy
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

from .common import digest, require, sha256_file
from . import section9_scoring as original
from . import section9_memory as memory


SCHEMA = original.SCHEMA  # Same bounded wire protocol and scientific identity.
MAX_RSS_BYTES = original.MAX_RSS_BYTES
ScoringError = original.ScoringError
SOURCES = ('large_qubit/section9_memory.py', 'large_qubit/section9_recovery_scoring.py')


def compatible_reference_identity(current, saved):
    """Only the registered memory instrumentation may differ from old references."""
    if current.get('memory_accounting_policy') != memory.POLICY:
        return False
    normalized = copy.deepcopy(current)
    normalized.pop('memory_accounting_policy', None)
    startup = normalized.pop('memory_accounting_at_startup', None)
    if (not isinstance(startup, dict)
            or startup.get('memory_accounting', {}).get('guard_passed') is not True
            or startup['memory_accounting'].get('guard_limit_bytes') != MAX_RSS_BYTES):
        return False
    bindings = normalized.get('source_bindings', {})
    if any(path not in bindings for path in SOURCES):
        return False
    for path in SOURCES:
        bindings.pop(path)
    return normalized == saved


def probe(geometry_id=None):
    value = original.probe(geometry_id)
    value['memory_accounting_policy'] = dict(memory.POLICY)
    return value


class TargetScorer(original.TargetScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity['memory_accounting_policy'] = dict(memory.POLICY)
        self.identity['memory_accounting_at_startup'] = memory.enforce_memory_budget(MAX_RSS_BYTES)
        self.identity['source_bindings'].update({p: sha256_file(self.root / p) for p in SOURCES})

    def _failure(self, stage, error, before, started, *, query=None, evaluator=None):
        result = super()._failure(stage, error, before, started, query=query, evaluator=evaluator)
        if 'memory_accounting' not in result.report:
            try:
                result.report.update(memory.memory_fields(MAX_RSS_BYTES))
            except Exception as memory_error:
                result.report.update(memory_measurement_unavailable=True,
                                     memory_measurement_error=str(memory_error))
        return result

    def _measure(self, query, evaluator, basis, *, category):
        started, before = time.perf_counter(), evaluator.statistics()
        try:
            query = original.normalize_query(query, self.geometry_id)
            state = self._state(query, basis)
            prepared = time.perf_counter() - started
            energy = float(evaluator.energy_sector(state, basis))
            diagnostics = basis.diagnostics(state)
            del state
            after = evaluator.statistics()
            require(math.isfinite(energy) and diagnostics['n_alpha'] == 11 and diagnostics['n_beta'] == 11
                    and abs(diagnostics['state_norm'] - 1.) <= 1e-10,
                    'Final-test energy, norm or fixed populations failed')
            measured = memory.enforce_memory_budget(MAX_RSS_BYTES)
            return {'energy_hartree': energy, 'geometry_id': self.geometry_id, 'partition': original.PARTITION,
                'query_sha256': digest(query), 'cache_identity': self.identity['cache_identity'],
                'norm': diagnostics['state_norm'], 'populations': [11, 11], 'diagnostics': diagnostics,
                'state_preparation_seconds': prepared, 'wall_seconds': time.perf_counter() - started,
                **measured, 'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'kernel_contractions': after['kernel_contractions'] - before['kernel_contractions'],
                'hamiltonian_action_seconds': after['hamiltonian_action_seconds'] - before['hamiltonian_action_seconds'],
                'cost_category': category, 'physical_evaluation_performed': True,
                'scalar_cache_used': False, 'state_cache_used': False}
        except Exception as error:
            raise self._failure('verify' if category.startswith('integrity') else 'score', error,
                                before, started, query=query, evaluator=evaluator) from error

    def reference(self):
        from .reference import solve_reference
        require(not self._reference_computed, 'Reference already attempted; retain its completed or failure receipt')
        self._reference_computed = True
        started, before = time.perf_counter(), self.evaluator.statistics()
        report = None
        try:
            solved = solve_reference(self.evaluator, **original.REFERENCE_SETTINGS)
            report = solved['report']
            energy = float(solved['energy'])
            del solved
            after = self.evaluator.statistics()
            require(report['all_passed'] is True and report['converged'] is True
                    and report['independent_starts'] == 2 and math.isfinite(energy),
                    'Target reference failed its independent-start convergence gate')
            measured = memory.enforce_memory_budget(MAX_RSS_BYTES)
            return {'energy_hartree': energy, 'report': report, 'geometry_id': self.geometry_id,
                'partition': original.PARTITION, 'cache_identity': self.identity['cache_identity'],
                'cost_category': 'final_test_reference_overhead_not_search',
                'wall_seconds': time.perf_counter() - started, **measured,
                'evaluator_calls': after['apply_h_calls'] - before['apply_h_calls'],
                'ordinary_energy_requests': 0, 'state_arrays_returned': False}
        except Exception as error:
            if report is not None:
                partial = dict(getattr(error, 'report', None) or {})
                partial.update(report)
                error.report = partial
            raise self._failure('reference', error, before, started) from error


def worker(*args, factory=None, **kwargs):
    return original.worker(*args, factory=TargetScorer if factory is None else factory, **kwargs)


class PhysicsClient(original.PhysicsClient):
    """Same target transport, with an explicit additive worker module and check."""
    def _start(self):
        require(type(self.expected_freeze_sha256) is str
                and re.fullmatch('[0-9a-f]{64}', self.expected_freeze_sha256) is not None,
                'Supply the exact expected final Section 8 freeze SHA256')
        env = dict(os.environ)
        env.update({name: str(self.threads) for name in original.inherited.THREAD_VARIABLES})
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        self._stderr = tempfile.TemporaryFile(mode='a+t', encoding='utf-8', errors='replace')
        try:
            self._process = subprocess.Popen([self.python_executable, '-B', '-m',
                'large_qubit.section9_recovery_scoring', '--worker', '--root', str(self.root),
                '--geometry-id', self.geometry_id, '--freeze-sha256', self.expected_freeze_sha256,
                '--threads', '1', '--execute'], cwd=self.root, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding='utf-8', bufsize=1)
            threading.Thread(target=self._read_output, daemon=True).start()
            ready = self._receive(timeout=180)
            require(ready.get('event') == 'ready' and ready.get('schema_version') == SCHEMA,
                    'Recovery target worker failed its frozen handshake: ' + str(ready))
            identity = ready['identity']
            cache = identity['cache_identity']
            require(identity['geometry_id'] == self.geometry_id and identity['partition'] == original.PARTITION
                    and identity['final_freeze_sha256'] == self.expected_freeze_sha256
                    and identity['physics_manifest_sha256'] == original.inherited.PHYSICS_SHA256
                    and cache['geometry_id'] == self.geometry_id and cache['partition'] == original.PARTITION
                    and cache['catalogue'] == original.inherited.CATALOGUE_SHA256
                    and cache['precision'] == {'state': 'complex128', 'integrals': 'float64'}
                    and set(cache) == {'partition', 'geometry_id', 'hamiltonian', 'orbitals', 'catalogue', 'backend', 'precision'}
                    and identity.get('memory_accounting_policy') == memory.POLICY,
                    'Recovery target worker returned an incompatible frozen scoring identity')
            self.identity = identity
        except Exception:
            self.close(force=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--geometry-id', choices=original.TARGETS)
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
