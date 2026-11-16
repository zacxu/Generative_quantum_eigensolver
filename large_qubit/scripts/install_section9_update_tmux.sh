#!/usr/bin/env bash
# Run from the server project root after extracting the update into its staging folder.
set -euo pipefail
umask 077
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
stage_root=$(cd -- "$script_directory/../.." && pwd -P)
project_root=$(pwd -P)
training_python=${TRAINING_PYTHON:-$project_root/.venv/bin/python}
session=co2-s9-update-matrix_0001
logs="$project_root/large_qubit/operations/section9_recovery_v1/control/update_jobs"
action=${1:-start}
case "$action" in
  start)
    [[ $# -le 1 ]] || exit 2
    [[ -x "$training_python" ]] || { printf 'Run this command from the server project root.\n' >&2; exit 2; }
    [[ "$stage_root" == "$project_root/large_qubit/updates/section9_operations_v2" ]] || {
      printf 'Extract the update into large_qubit/updates/section9_operations_v2 first.\n' >&2; exit 2;
    }
    command -v tmux >/dev/null
    if tmux has-session -t "=$session" 2>/dev/null; then
      printf 'Update/pause helper is already running. Use this script with status.\n'
      exit 0
    fi
    mkdir -p -- "$logs"
    attempt=$(mktemp -d "$logs/job_$(date -u +%Y%m%dT%H%M%S%N)_XXXXXXXX")
    tmux new-session -d -s "$session" -c "$project_root" \
      env "TRAINING_PYTHON=$training_python" \
      bash "$script_directory/install_section9_update_tmux.sh" __worker "$attempt"
    printf 'Update/pause REQUESTED in %s. Existing evaluation files stay in place until safe shutdown.\n' "$session"
    printf 'SSH can disconnect. Check with this script followed by status.\nLog: %s/worker.log\n' "$attempt"
    ;;
  status)
    [[ $# -eq 1 ]] || exit 2
    if tmux has-session -t "=$session" 2>/dev/null; then
      printf 'Update/pause helper: RUNNING; wait for installed: true and paused: true.\n'
    else
      printf 'Update/pause helper: not running; check the final result below.\n'
    fi
    shopt -s nullglob
    attempts=("$logs"/job_*)
    (( ${#attempts[@]} )) || { printf 'No update helper has been started.\n'; exit 0; }
    attempt=${attempts[${#attempts[@]}-1]}
    [[ ! -f "$attempt/worker.log" ]] || tail -n 35 -- "$attempt/worker.log"
    if [[ -f "$attempt/exit_code" ]]; then
      printf 'Update helper exit code: '
      cat -- "$attempt/exit_code"
    fi
    ;;
  __worker)
    [[ $# -eq 2 ]] || exit 2
    attempt=$2
    [[ "$attempt" == "$logs/job_"* ]] || exit 2
    exec >> "$attempt/worker.log" 2>&1
    date -u +%Y-%m-%dT%H:%M:%SZ > "$attempt/started_utc"
    finish() {
      code=$?
      trap - EXIT
      printf '%s\n' "$code" > "$attempt/exit_code"
      date -u +%Y-%m-%dT%H:%M:%SZ > "$attempt/finished_utc"
      exit "$code"
    }
    trap finish EXIT
    "$training_python" -u -B "$stage_root/large_qubit/section9_update.py" \
      --root "$project_root" --stage-path "$stage_root" --execute
    ;;
  *) printf 'Use start or status.\n' >&2; exit 2 ;;
esac
