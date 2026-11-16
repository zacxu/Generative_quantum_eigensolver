#!/usr/bin/env bash
# Explicit detached local CPU work. The default action previews without science.
set -euo pipefail
umask 077

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
script_path="$script_directory/run_section4_tmux.sh"

usage() {
    printf '%s\n' \
        'Usage: bash large_qubit/scripts/run_section4_tmux.sh [preview|launch|status] [section4_v1]' \
        'Default: preview. launch explicitly starts the detached local scientific gate.' \
        'CHEMISTRY_PYTHON: interpreter path, default project/.venv-chemistry/bin/python.' \
        'CO2_SECTION4_SESSION: tmux session name, default co2-section4-<version>.'
}

fail() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }

if (( $# > 2 )); then usage >&2; exit 2; fi
action=${1:-preview}
version=${2:-section4_v1}
[[ "$version" =~ ^section4_[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail 'Use a simple section4_<version> suffix; paths are not accepted.'
session=${CO2_SECTION4_SESSION:-co2-section4-$version}
[[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || fail 'The tmux session name must contain only letters, numbers, underscore or hyphen.'
output_relative="large_qubit/validation/$version"
evidence_relative="large_qubit/evidence/section4_runs/$version"
run_directory="$project_root/$evidence_relative"
log_path="$run_directory/worker.log"
chemistry_python=${CHEMISTRY_PYTHON:-$project_root/.venv-chemistry/bin/python}
if [[ "$chemistry_python" != /* ]]; then chemistry_python="$project_root/$chemistry_python"; fi

require_python() {
    [[ -x "$chemistry_python" ]] || fail "Numerical Python is missing or not executable: $chemistry_python"
}

require_tmux() {
    command -v tmux >/dev/null 2>&1 || fail 'tmux is unavailable; no worker has been started.'
}

validate_launch_paths() {
    "$chemistry_python" -B - "$project_root" "$output_relative" "$evidence_relative" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
from large_qubit.common import require, workspace_path
for relative, namespace in ((sys.argv[2], "large_qubit/validation"),
                            (sys.argv[3], "large_qubit/evidence")):
    path = workspace_path(root, relative, namespace=namespace)
    require(not path.exists(), f"Preserve the existing output/evidence directory: {relative}")
PY
}

case "$action" in
    preview)
        require_python
        cd -- "$project_root"
        exec "$chemistry_python" -B -m large_qubit section4 --output "$output_relative"
        ;;
    launch)
        require_python
        require_tmux
        if tmux has-session -t "=$session" 2>/dev/null; then fail "tmux session already exists: $session"; fi
        validate_launch_paths
        mkdir -p -- "$project_root/large_qubit/evidence/section4_runs"
        # Exclusive mkdir plus noclobber preserve every previous launch attempt.
        mkdir -- "$run_directory"
        set -o noclobber
        : > "$log_path"
        {
            printf 'requested_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'project_root=%s\noutput=%s\nsession=%s\npython=%s\n' \
                "$project_root" "$output_relative" "$session" "$chemistry_python"
            tmux -V
        } > "$run_directory/launch.txt"
        if ! tmux new-session -d -s "$session" -c "$project_root" \
            env "CHEMISTRY_PYTHON=$chemistry_python" "CO2_SECTION4_SESSION=$session" \
            bash "$script_path" __worker "$version"; then
            printf 'tmux failed to create the worker session\n' >> "$log_path"
            : > "$run_directory/launch.failed"
            fail "Detached launch failed; retain $evidence_relative and choose a new version after investigation."
        fi
        # Direct tmux argv execution preserves spaces without shell interpolation.
        (tmux list-panes -t "=$session" -F '#{session_name} #{pane_pid} #{pane_current_command}' || true) \
            > "$run_directory/tmux_pane.txt"
        printf 'state=launched\nsession=%s\noutput=%s\nevidence=%s\nlog=%s\n' \
            "$session" "$output_relative" "$evidence_relative" "$log_path"
        ;;
    status)
        printf 'session=%s\noutput=%s\nevidence=%s\n' "$session" "$output_relative" "$evidence_relative"
        if [[ -r "$run_directory/worker.pid" ]]; then
            printf 'worker_pid=%s\n' "$(cat -- "$run_directory/worker.pid")"
        fi
        if [[ -r "$run_directory/worker.exit_code" ]]; then
            code=$(cat -- "$run_directory/worker.exit_code")
            if [[ "$code" == 0 ]]; then printf 'state=completed\nexit_code=0\n';
            else printf 'state=failed\nexit_code=%s\n' "$code"; fi
        elif command -v tmux >/dev/null 2>&1 && tmux has-session -t "=$session" 2>/dev/null; then
            printf 'state=running\n'
        elif [[ -e "$run_directory/launch.failed" ]]; then
            printf 'state=launch_failed\n'
        elif [[ -d "$run_directory" ]]; then
            printf 'state=not_running_without_completion_marker\n'
        else
            printf 'state=not_started\n'
        fi
        ;;
    __worker)
        # Internal explicit launch target; never selected by the default action.
        require_python
        [[ -d "$run_directory" && -f "$log_path" && -f "$run_directory/launch.txt" ]] \
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
            printf 'Section 4 worker finished with exit code %s\n' "$code"
            exit "$code"
        }
        trap 'finish "$?"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        trap 'exit 129' HUP
        cd -- "$project_root"
        export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
        "$chemistry_python" -B -m large_qubit section4 --execute --output "$output_relative"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        fail "Unknown action: $action"
        ;;
esac
