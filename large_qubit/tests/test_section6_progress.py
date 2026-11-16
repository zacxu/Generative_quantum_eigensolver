"""Read-only acquisition status, journal replay, and raw-label privacy tests."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from large_qubit.common import canonical_bytes, digest, sha256_file
from large_qubit.protocol import BENCHMARK_ID
from tools import section6_progress as monitor


SECRET_HF = -9876.54321
SECRET_DIAGNOSTIC = -9999.7654321


class ProgressPrivacyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.private = f'large_qubit/scoring_only/{BENCHMARK_ID}/section6/section6_v1'
        self.run = 'large_qubit/evidence/section6_runs/section6_v1'
        self.catalogue_sha = 'c' * 64
        self.index, self.previous = 0, '0' * 64
        self.write('large_qubit/section6_policy.json', {'acquisition': {
            'training_aliases': list(monitor.ALIASES), 'geometry_request_quotas': monitor.QUOTAS,
            'maximum_logical_requests': 8192, 'fitting_goal_per_geometry': 2048,
            'diagnostics_per_geometry': 128, 'length': 32, 'catalogue_sha256': self.catalogue_sha},
            'max_wall_seconds': 86400.})
        self.append({'event': 'session_started', 'utc': '2026-11-05T00:00:00+00:00'})
        self.started(1, 'HF')
        self.completed(1, 'HF', SECRET_HF)
        self.started(2, 'diagnostic')
        self.completed(2, 'diagnostic', SECRET_DIAGNOSTIC, ops=[1] * 32)
        self.started(3, 'fitting')
        self.completed(3, 'fitting', SECRET_HF - .125, ops=[2] * 32)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         value.encode('utf-8') if isinstance(value, str) else canonical_bytes(value) + b'\n')

    def append(self, payload, segment=1):
        body = {'record_index': self.index + 1, 'previous_sha256': self.previous, 'payload': payload}
        value = {**body, 'sha256': digest(body)}
        path = self.root / self.private / f'journal_{segment:04d}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as handle:
            handle.write(canonical_bytes(value) + b'\n')
        self.index += 1; self.previous = value['sha256']

    def started(self, ordinal, role, segment=1, alias='CO2_T01'):
        self.append({'event': 'request_started', 'request_ordinal': ordinal, 'geometry_id': alias,
                     'role': role, 'query_sha256': digest({'ordinal': ordinal}),
                     'catalogue_sha256': self.catalogue_sha}, segment)

    def completed(self, ordinal, role, energy, ops=None, segment=1, alias='CO2_T01'):
        row = None if role == 'HF' else {'row_id': f'{alias}_{role}_{ordinal}',
            'geometry_id': alias, 'role': role, 'ops': ops, 'energy_hartree': energy,
            'catalogue_sha256': self.catalogue_sha, 'request_ordinal': ordinal}
        self.append({'event': 'request_completed', 'request_ordinal': ordinal, 'geometry_id': alias,
                     'role': role, 'query_sha256': digest({'ordinal': ordinal}),
                     'energy_hartree': energy, 'eligible_row': row, 'wall_seconds': .2,
                     'peak_rss_bytes': 1000}, segment)

    def test_snapshot_is_readonly_and_never_prints_absolute_or_diagnostic_values(self):
        before = {path.relative_to(self.root).as_posix(): sha256_file(path)
                  for path in self.root.rglob('*') if path.is_file()}
        report = monitor.progress(self.root)
        self.assertEqual(report['requests']['issued'], 3)
        self.assertEqual(report['retained']['fitting'], 1)
        self.assertEqual(report['retained']['diagnostic'], 1)
        self.assertAlmostEqual(report['geometries']['CO2_T01']['best_retained_fitting_HF_improvement_ha'], .125)
        self.assertFalse(report['diagnostic_values_used_for_selection_or_improvement'])
        encoded = json.dumps(report)
        self.assertNotIn(str(SECRET_HF), encoded)
        self.assertNotIn(str(SECRET_DIAGNOSTIC), encoded)
        self.assertNotIn('energy_hartree', encoded)
        self.assertNotIn('"ops"', encoded)
        after = {path.relative_to(self.root).as_posix(): sha256_file(path)
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_resumed_segment_preserves_abandoned_and_failed_charges_and_fragments(self):
        self.started(4, 'fitting')
        with (self.root / self.private / 'journal_0001.jsonl').open('ab') as handle:
            handle.write(b'{"PRIVATE_FRAGMENT_VALUE":-9876.54321')
        self.append({'event': 'session_started', 'utc': '2026-11-05T01:00:00+00:00'}, 2)
        self.append({'event': 'request_abandoned', 'request_ordinal': 4}, 2)
        self.started(5, 'HF', 2, alias='CO2_T02')
        self.append({'event': 'request_failed', 'request_ordinal': 5,
                     'error': 'PRIVATE_FAILED_VALUE=-9876.54321'}, 2)
        report = monitor.progress(self.root)
        self.assertEqual(report['requests'], {'cap': 8192, 'issued': 5, 'completed': 3,
            'failed': 1, 'abandoned': 1, 'pending': 0, 'remaining_after_issued': 8187})
        self.assertEqual(report['journal']['segments'], 2)
        self.assertEqual(len(report['journal']['uncommitted_fragments_ignored']), 1)
        self.assertNotIn('PRIVATE', json.dumps(report))

    def test_hash_chain_malformed_committed_lines_and_segment_gaps_fail_closed(self):
        path = self.root / self.private / 'journal_0001.jsonl'
        original = path.read_bytes()
        path.write_bytes(original.replace(b'-9876.54321', b'-9875.54321', 1))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            monitor.progress(self.root)
        path.write_bytes(original + b'{"private_error":-9876.54321}\n')
        with self.assertRaises(ValueError):
            monitor.progress(self.root)
        path.write_bytes(original)
        self.write(f'{self.private}/journal_0003.jsonl', b'')
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            monitor.progress(self.root)

    def test_cli_invalid_data_never_echoes_private_records_or_exception_text(self):
        self.append({'event': 'request_completed', 'request_ordinal': 999,
                     'energy_hartree': SECRET_DIAGNOSTIC, 'private_error': 'PRIVATE_SECRET_LABEL'})
        capture = StringIO()
        with redirect_stdout(capture):
            code = monitor.main(['--root', str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(capture.getvalue()), {'snapshot_valid': False, 'error_type': 'ValueError'})
        self.assertNotIn('PRIVATE', capture.getvalue())
        self.assertNotIn(str(SECRET_DIAGNOSTIC), capture.getvalue())

    def test_retained_duplicate_cross_role_and_boolean_ordinals_are_refused(self):
        self.started(4, 'fitting')
        self.completed(4, 'fitting', SECRET_HF - .125, ops=[1] * 32)
        with self.assertRaisesRegex(ValueError, 'duplicates fitting or diagnostics'):
            monitor.progress(self.root)

    def test_attempt_status_preserves_previous_failures_and_ignores_worker_logs(self):
        self.write(f'{self.run}/attempt_0001/worker.failed', b'')
        self.write(f'{self.run}/attempt_0001/worker.exit_code', '1\n')
        self.write(f'{self.run}/attempt_0001/worker.log', b'PRIVATE_LABEL=-9876.54321\xff')
        self.write(f'{self.run}/attempt_0002/worker.started_utc', '2026-11-05T01:00:00Z\n')
        self.write(f'{self.run}/attempt_0002/worker.log', b'PRIVATE_DIAGNOSTIC=-9999.7654321\xff')
        report = monitor.progress(self.root)
        self.assertEqual(report['worker']['attempt_count'], 2)
        self.assertEqual(report['worker']['latest_attempt'], 'attempt_0002')
        self.assertEqual(report['worker']['previous_failed_attempts'], 1)
        self.assertEqual(report['worker']['state'], 'started_no_exit_marker')
        self.assertNotIn('PRIVATE', json.dumps(report))

    def test_integrity_overhead_is_separate_from_ordinary_requests(self):
        self.append({'event': 'integrity_started', 'private_energy': SECRET_HF})
        self.append({'event': 'integrity_completed', 'energy_hartree': SECRET_HF})
        report = monitor.progress(self.root)
        self.assertEqual(report['requests']['issued'], 3)
        self.assertEqual(report['separate_integrity_overhead'], {'issued': 1, 'completed': 1, 'abandoned': 0})
        self.assertNotIn(str(SECRET_HF), json.dumps(report))

    def test_helper_import_and_snapshot_load_no_scientific_or_model_packages(self):
        script = ('import sys; from tools.section6_progress import progress; '
                  f'progress({str(self.root)!r}); '
                  'assert not ({"numpy","scipy","pyscf","torch","transformers"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', script], check=True,
                       cwd=Path(__file__).resolve().parents[2])


if __name__ == '__main__':
    unittest.main()
