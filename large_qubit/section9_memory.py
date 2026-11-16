"""Linux memory accounting for a newly exec'd Section 9 chemistry worker.

getrusage(RUSAGE_SELF).ru_maxrss survives execve and can therefore contain the
GPU parent's larger pre-exec resident size. /proc/self/status VmHWM belongs to
the worker's current address space. Keep both measurements for audit; enforce
the unchanged process RSS limit against VmHWM, never by subtracting a baseline.
"""

from pathlib import Path
import sys


SCHEMA = 'section9-worker-address-space-rss-v1'
POLICY = {
    'schema_version': SCHEMA,
    'guard_source': '/proc/self/status:VmHWM',
    'guard_scope': 'peak resident size of the current worker address space since exec',
    'diagnostic_source': 'getrusage(RUSAGE_SELF).ru_maxrss; may include pre-exec parent peak',
    'baseline_subtraction': False,
    'missing_measurement': 'refuse; never assume zero or fall back to inherited peak',
}


def parse_status(text):
    """Require explicit, nonnegative Linux KiB measurements, not guessed units."""
    found = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('VmHWM:', 'VmRSS:'):
            if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit():
                raise RuntimeError('Invalid Linux process RSS field: ' + line)
            if fields[0] in found:
                raise RuntimeError('Duplicate Linux process RSS field: ' + fields[0])
            found[fields[0]] = int(fields[1]) * 1024
    if set(found) != {'VmHWM:', 'VmRSS:'} or found['VmHWM:'] <= 0 or found['VmRSS:'] <= 0:
        raise RuntimeError('Linux /proc/self/status must provide positive VmHWM and VmRSS')
    if found['VmHWM:'] < found['VmRSS:']:
        raise RuntimeError('Linux process VmHWM is below its current VmRSS')
    return {'peak_rss_bytes': found['VmHWM:'], 'current_rss_bytes': found['VmRSS:']}


def memory_snapshot():
    """Read the worker's peak and retain inherited getrusage only as diagnostic."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Recovery scoring requires Linux /proc process memory accounting')
    import resource
    value = parse_status(Path('/proc/self/status').read_text(encoding='ascii'))
    value.update(schema_version=SCHEMA, source=POLICY['guard_source'],
                 raw_getrusage_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    return value


def memory_fields(limit_bytes, *, snapshot=None):
    value = memory_snapshot() if snapshot is None else dict(snapshot)
    value.update(guard_limit_bytes=limit_bytes, guard_passed=value['peak_rss_bytes'] <= limit_bytes)
    return {'peak_rss_bytes': value['peak_rss_bytes'],
            'current_rss_bytes': value['current_rss_bytes'],
            'raw_getrusage_peak_rss_bytes': value['raw_getrusage_peak_rss_bytes'],
            'memory_accounting': value}


def enforce_memory_budget(limit_bytes):
    result = memory_fields(limit_bytes)
    if not result['memory_accounting']['guard_passed']:
        error = ValueError('Target scorer exceeds the 4 GiB process RSS ceiling')
        error.report = result
        raise error
    return result
