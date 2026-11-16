"""V2-only, temporary awake receipts with mocked Windows calls; no power change."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import section6_v2_keep_awake as awake


class V2KeepAwakeTest(unittest.TestCase):
    def test_request_and_release_use_only_v2_receipts_and_do_not_touch_v1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1 = root / 'large_qubit/evidence/section6_runs/section6_v1/attempt_0001'
            v2 = root / 'large_qubit/evidence/section6_runs/section6_v2/attempt_0001'
            v1.mkdir(parents=True); v2.mkdir(parents=True)
            (v1 / 'preserved.txt').write_bytes(b'old evidence')
            (v2 / 'worker.started_utc').write_text('2026-11-06T00:00:00Z\n')
            (v2 / 'worker.exit_code').write_text('0\n')
            state = Mock(side_effect=[1, 1])
            kernel = SimpleNamespace(SetThreadExecutionState=state)
            with patch.object(awake, '__file__', str(root / 'tools/section6_v2_keep_awake.py')), \
                    patch.object(awake, 'os', SimpleNamespace(name='nt', getpid=lambda: 77)), \
                    patch.object(awake.ctypes, 'windll', SimpleNamespace(kernel32=kernel), create=True), \
                    patch('sys.argv', ['section6_v2_keep_awake.py', '--attempt', 'attempt_0001']):
                awake.main()
            self.assertEqual([call.args for call in state.call_args_list], [(0x80000001,), (0x80000000,)])
            started = json.loads((v2 / 'keep_awake.started.json').read_text())
            finished = json.loads((v2 / 'keep_awake.finished.json').read_text())
            self.assertEqual(started['maximum_seconds'], 25 * 3600)
            self.assertEqual(started['pid'], 77)
            self.assertTrue(finished['temporary_request_released'])
            self.assertTrue(finished['worker_exit_marker_present'])
            self.assertEqual(list(v1.iterdir()), [v1 / 'preserved.txt'])
            self.assertEqual((v1 / 'preserved.txt').read_bytes(), b'old evidence')

    def test_absent_v2_or_invalid_attempt_never_requests_a_system_awake_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'large_qubit/evidence/section6_runs/section6_v1/attempt_0001'
            old.mkdir(parents=True)
            (old / 'worker.started_utc').write_text('preserved old start')
            state = Mock()
            with patch.object(awake, '__file__', str(root / 'tools/section6_v2_keep_awake.py')), \
                    patch.object(awake, 'os', SimpleNamespace(name='nt', getpid=lambda: 77)), \
                    patch.object(awake.ctypes, 'windll', SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=state)), create=True):
                with patch('sys.argv', ['section6_v2_keep_awake.py']), self.assertRaises(FileNotFoundError):
                    awake.main()
                with patch('sys.argv', ['section6_v2_keep_awake.py', '--attempt', '../v1']), self.assertRaises(SystemExit):
                    awake.main()
            state.assert_not_called()
            self.assertEqual((old / 'worker.started_utc').read_text(), 'preserved old start')


if __name__ == '__main__':
    unittest.main()
