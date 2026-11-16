"""Linux subprocess fixtures: no GPUs, model weights or chemistry are accessed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from large_qubit import section9_parallel as main
from large_qubit.section8 import save, load
from large_qubit.tests.test_section9_parallel import allocation


def wait_for(predicate, timeout=8):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError('Subprocess fixture timed out')
        time.sleep(.02)


def running(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().split(') ')[1][0] != 'Z'
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform == 'linux', 'Requires Linux process groups and prctl')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exited_leader_live_child_is_killed_before_resume(self):
        marker = self.root / 'child.json'
        code = ('import json,subprocess,sys; from pathlib import Path; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                'Path(sys.argv[1]).write_text(json.dumps(p.pid))')
        parent = subprocess.Popen([sys.executable, '-B', '-c', code, str(marker)], start_new_session=True)
        try:
            wait_for(marker.exists)
            child = json.loads(marker.read_text())
            self.assertEqual(parent.wait(timeout=8), 0)
            self.assertTrue(running(child))
            main._stop_processes([parent])
            wait_for(lambda: not running(child))
        finally:
            main._signal_group(parent.pid, signal.SIGKILL)
            parent.wait(timeout=8)

    def test_parent_death_signal_kills_lane_without_coordinator_cleanup(self):
        marker = self.root / 'lane.json'
        child_code = ('import os,sys,time; from pathlib import Path; '
            'from large_qubit.section9_parallel_worker import _arm_parent_death_signal; '
            '_arm_parent_death_signal(os.getppid()); '
            'Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)')
        parent_code = ('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-B","-c",sys.argv[1],sys.argv[2]]); time.sleep(60)')
        parent = subprocess.Popen([sys.executable, '-B', '-c', parent_code, child_code, str(marker)],
                                  start_new_session=True)
        try:
            wait_for(marker.exists)
            child = int(marker.read_text())
            self.assertTrue(running(child))
            parent.kill()  # Abrupt death, no finally/normal parent supervision.
            parent.wait(timeout=8)
            wait_for(lambda: not running(child))
        finally:
            main._signal_group(parent.pid, signal.SIGKILL)
            parent.wait(timeout=8)

    def test_four_real_fixture_processes_overlap_and_use_isolated_gpu_masks(self):
        dest = self.root / 'matrix'
        save(dest / 'gpu_allocation.json', allocation())
        markers = self.root / 'markers'
        markers.mkdir()
        code = '''import json,os,sys,time
from pathlib import Path
folder,lane=Path(sys.argv[1]),sys.argv[2]
(folder/(lane+'.json')).write_text(json.dumps({'mask':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid()}))
deadline=time.monotonic()+8
while len(list(folder.glob('lane_*.json')))!=4:
    if time.monotonic()>deadline: raise RuntimeError('Workers were not concurrent')
    time.sleep(.02)
time.sleep(.05)
'''
        popen = subprocess.Popen
        def fixture_process(argv, **kwargs):
            lane = argv[argv.index('--lane') + 1]
            return popen([sys.executable, '-B', '-c', code, str(markers), lane], **kwargs)
        with patch.object(main.subprocess, 'Popen', side_effect=fixture_process):
            exits = main.launch_lanes(self.root, dest, dest / 'attempt', allocation(), 'unused')
        self.assertEqual(exits, {f'lane_{i}': 0 for i in range(4)})
        seen = [json.loads(p.read_text()) for p in markers.glob('lane_*.json')]
        self.assertEqual(len({v['pid'] for v in seen}), 4)
        self.assertEqual({v['mask'] for v in seen}, {v['gpu_uuid'] for v in allocation()['lanes'].values()})


if __name__ == '__main__':
    unittest.main()
