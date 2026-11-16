"""Hold a temporary Windows system-awake request for the approved v2 attempt."""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='attempt_0001')
    args = parser.parse_args()
    import re
    if os.name != 'nt' or not re.fullmatch(r'attempt_[0-9]{4}', args.attempt):
        raise SystemExit('Use Windows and one existing acquisition attempt')
    root = Path(__file__).resolve().parent.parent
    candidate = root / 'large_qubit/evidence/section6_runs/section6_v2' / args.attempt
    folder = candidate.resolve(strict=True)
    if folder != candidate or not folder.is_relative_to(root) or not (folder / 'worker.started_utc').is_file():
        raise SystemExit('The acquisition attempt has not started')
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    previous = kernel.SetThreadExecutionState(0x80000001)
    if not previous:
        raise SystemExit('Windows refused the temporary system-awake request')
    def write(name, value):
        with (folder / name).open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')
    started = time.monotonic()
    try:
        write('keep_awake.started.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'request': 'ES_CONTINUOUS|ES_SYSTEM_REQUIRED',
              'previous_thread_state': previous, 'maximum_seconds': 25 * 3600})
        while not (folder / 'worker.exit_code').is_file() and time.monotonic() - started < 25 * 3600:
            time.sleep(20)
    finally:
        restored = bool(kernel.SetThreadExecutionState(0x80000000))
        write('keep_awake.finished.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'temporary_request_released': restored, 'wall_seconds': time.monotonic() - started,
              'worker_exit_marker_present': (folder / 'worker.exit_code').is_file()})


if __name__ == '__main__':
    main()
