#!/usr/bin/env bash
# Detached local CPU acquisition. Default preview performs no energy requests.
set -euo pipefail
umask 077

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
script_path="$script_directory/run_section6_v2_tmux.sh"

usage() {
    printf '%s\n' \
        'Usage: bash large_qubit/scripts/run_section6_v2_tmux.sh [preview|launch|resume|status] [section6_v2]' \
        'Default preview. launch starts the first acquisition; resume creates a new immutable attempt.' \
        'CHEMISTRY_PYTHON: interpreter, default project/.venv-chemistry/bin/python.' \
        'CO2_SECTION6_V2_SESSION: tmux session, default co2-section6-<version>.'
}
fail() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }

action=${1:-preview}
version=${2:-section6_v2}
if [[ "$action" == __worker ]]; then
    (( $# == 4 )) || fail 'Internal worker requires version, attempt and launch mode.'
else
    (( $# <= 2 )) || { usage >&2; exit 2; }
fi
[[ "$version" == section6_v2 ]] || fail 'This approved confirmation has exactly one output identity: section6_v2.'
session=${CO2_SECTION6_V2_SESSION:-co2-section6-$version}
[[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || fail 'The tmux session name contains invalid characters.'
output_relative="large_qubit/validation/$version"
private_relative="large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/section6/$version"
evidence_relative="large_qubit/evidence/section6_runs/$version"
chemistry_python=${CHEMISTRY_PYTHON:-$project_root/.venv-chemistry/bin/python}
if [[ "$chemistry_python" != /* ]]; then chemistry_python="$project_root/$chemistry_python"; fi

require_python() { [[ -x "$chemistry_python" ]] || fail 'The local chemistry Python is missing or not executable.'; }
require_tmux() { command -v tmux >/dev/null 2>&1 || fail 'tmux is unavailable; no worker has been started.'; }

new_attempt() {
    "$chemistry_python" -B - "$project_root" "$output_relative" "$private_relative" "$evidence_relative" "$action" <<'PY'
from pathlib import Path
import re, sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
from large_qubit.common import require, workspace_path
public = workspace_path(root, sys.argv[2], namespace='large_qubit/validation')
private = workspace_path(root, sys.argv[3], namespace='large_qubit/scoring_only')
base = workspace_path(root, sys.argv[4], namespace='large_qubit/evidence/section6_runs')
action = sys.argv[5]
require(private.is_dir() and (private/'reservations.json').is_file(), 'Run and verify Section6 preparation first')
segments = list(private.glob('journal_*.jsonl'))
if action == 'launch':
    require(not public.exists() and not segments, 'Preserve prior output; use explicit resume for an interrupted campaign')
    if base.exists():
        old_attempts = sorted(base.iterdir())
        require(old_attempts and (old_attempts[-1]/'launch.failed').is_file(), 'Previous launch lacks a finished launch-failure marker; inspect it before continuing')
else:
    require(public.is_dir() and segments, 'Resume requires an existing campaign and charged journal')
    require(not (public/'manifest.json').exists() or
            ((public/'finalization_input.json').is_file() and not (public/'manifest.sha256').exists()),
            'A completed campaign must be verified, not resumed')
attempts = []
if base.exists():
    for child in sorted(base.iterdir()):
        require(re.fullmatch(r'attempt_[0-9]{4}', child.name) and child.is_dir(), 'Unexpected attempt evidence entry')
        workspace_path(root, child.relative_to(root).as_posix(), namespace=sys.argv[4])
        attempts.append(child.name)
require(attempts == [f'attempt_{index:04d}' for index in range(1,len(attempts)+1)], 'Attempt history contains a gap')
require(len(attempts) < 9999, 'Attempt counter exhausted')
attempt = f'attempt_{len(attempts)+1:04d}'
path = workspace_path(root, sys.argv[4]+'/'+attempt, namespace='large_qubit/evidence/section6_runs')
require(not path.exists(), 'Preserve an existing attempt directory')
print(attempt)
PY
}

case "$action" in
    preview)
        require_python
        cd -- "$project_root"
        exec "$chemistry_python" -B -m large_qubit.section6_v2 acquire --output "$output_relative"
        ;;
    launch|resume)
        require_python; require_tmux
        if tmux has-session -t "=$session" 2>/dev/null; then fail 'A controller tmux session already exists; inspect its status.'; fi
        cd -- "$project_root"
        "$chemistry_python" -B -m large_qubit.section6_v2 prepare --verify
        attempt=$(new_attempt)
        run_directory="$project_root/$evidence_relative/$attempt"
        log_path="$run_directory/worker.log"
        mkdir -p -- "$project_root/$evidence_relative"
        mkdir -- "$run_directory"
        set -o noclobber
        : > "$log_path"
        {
            printf 'requested_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'project_root=%s\noutput=%s\nsession=%s\npython=%s\nattempt=%s\nmode=%s\n' \
                "$project_root" "$output_relative" "$session" "$chemistry_python" "$attempt" "$action"
            tmux -V
        } > "$run_directory/launch.txt"
        if ! tmux new-session -d -s "$session" -c "$project_root" \
            env "CHEMISTRY_PYTHON=$chemistry_python" "CO2_SECTION6_V2_SESSION=$session" \
            bash "$script_path" __worker "$version" "$attempt" "$action"; then
            printf 'tmux failed to create the worker session\n' >> "$log_path"
            : > "$run_directory/launch.failed"
            fail 'Detached launch failed; preserve this attempt and investigate before an explicit resume.'
        fi
        (tmux list-panes -t "=$session" -F '#{session_name} #{pane_pid} #{pane_current_command}' || true) \
            > "$run_directory/tmux_pane.txt"
        printf 'state=launched\nmode=%s\nsession=%s\noutput=%s\nattempt=%s\nevidence=%s/%s\n' \
            "$action" "$session" "$output_relative" "$attempt" "$evidence_relative" "$attempt"
        ;;
    status)
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "=$session" 2>/dev/null; then
            printf 'tmux_session=present\n'
        else
            printf 'tmux_session=absent\n'
        fi
        require_python
        cd -- "$project_root"
        exec "$chemistry_python" -B "$project_root/tools/section6_v2_progress.py" \
            --section6-root "$output_relative" --run-directory "$evidence_relative"
        ;;
    __worker)
        require_python
        attempt=$3; mode=$4
        [[ "$attempt" =~ ^attempt_[0-9]{4}$ && "$attempt" != attempt_0000 ]] || fail 'Invalid worker attempt.'
        [[ "$mode" == launch || "$mode" == resume ]] || fail 'Invalid worker launch mode.'
        run_directory="$project_root/$evidence_relative/$attempt"
        log_path="$run_directory/worker.log"
        "$chemistry_python" -B - "$project_root" "$evidence_relative/$attempt" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
from large_qubit.common import workspace_path
for suffix in ('', '/worker.log', '/launch.txt', '/worker.pid', '/worker.exit_code',
               '/worker.started_utc', '/worker.finished_utc', '/worker.success', '/worker.failed'):
    workspace_path(root, sys.argv[2]+suffix, namespace='large_qubit/evidence/section6_runs')
PY
        [[ -d "$run_directory" && -f "$log_path" && -f "$run_directory/launch.txt" && ! -e "$run_directory/launch.failed" ]] \
            || fail 'The worker requires a newly created launch evidence directory.'
        set -o noclobber
        printf '%s\n' "$$" > "$run_directory/worker.pid"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$run_directory/worker.started_utc"
        exec >> "$log_path" 2>&1
        finish() {
            local code=$1
            trap - EXIT
            printf '%s\n' "$code" > "$run_directory/worker.exit_code"
            date -u +%Y-%m-%dT%H:%M:%SZ > "$run_directory/worker.finished_utc"
            if [[ "$code" == 0 ]]; then : > "$run_directory/worker.success";
            else : > "$run_directory/worker.failed"; fi
            printf 'Section6 worker finished with exit code %s\n' "$code"
            exit "$code"
        }
        trap 'finish "$?"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        trap 'exit 129' HUP
        cd -- "$project_root"
        export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
        if [[ "$mode" == resume ]]; then
            "$chemistry_python" -B -m large_qubit.section6_v2 acquire --execute --output "$output_relative" --resume
        else
            "$chemistry_python" -B -m large_qubit.section6_v2 acquire --execute --output "$output_relative"
        fi
        ;;
    -h|--help|help) usage ;;
    *) usage >&2; fail 'Unknown action.' ;;
esac
