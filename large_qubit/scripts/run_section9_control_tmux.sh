#!/usr/bin/env bash
# One researcher-facing launcher for the existing Section 9 recovery matrix.
set -euo pipefail
umask 077
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
training_python=${TRAINING_PYTHON:-$project_root/.venv/bin/python}
chemistry_python=${CHEMISTRY_PYTHON:-$project_root/.venv-chemistry/bin/python}
action=${1:-status}
cd -- "$project_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
pause_session=co2-s9-pause-matrix_0001
pause_jobs="$project_root/large_qubit/operations/section9_recovery_v1/control/pause_jobs"
amendment="$project_root/large_qubit/registrations/sections_1_9_operations_v2/manifest.json"
if [[ "$action" != __worker && "$action" != __pause_worker && $# -gt 1 ]]; then
  printf 'Unexpected extra arguments. Use one action: verify, pause, pause-status, reassign, resume, status or matrix-verify.\n' >&2
  exit 2
fi
verify_operations() {
  "$training_python" -B -m large_qubit.section9_operations_registration verify >/dev/null
}
require_mask() {
  [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[^,[:space:]]+(,[^,[:space:]]+){3}$ ]] || {
    printf 'Set CUDA_VISIBLE_DEVICES to four allocated H200s in the desired lane order.\n' >&2
    exit 2
  }
}
require_pause_finished() {
  if tmux has-session -t "=$pause_session" 2>/dev/null; then
    printf 'Pause is still pending. Run pause-status and wait for paused: true before changing GPUs or resuming.\n' >&2
    exit 2
  fi
}
pause_status() {
  if tmux has-session -t "=$pause_session" 2>/dev/null; then
    printf 'Pause helper: RUNNING. The matrix is not yet confirmed paused.\n'
  else
    printf 'Pause helper: not running. Check the latest result below.\n'
  fi
  local folders=() latest
  shopt -s nullglob
  folders=("$pause_jobs"/job_*)
  shopt -u nullglob
  if (( ${#folders[@]} == 0 )); then
    printf 'No persistent pause job has been requested with this launcher.\n'
  else
    latest=${folders[${#folders[@]}-1]}
    printf 'Saved pause log (current state follows below): %s/worker.log\n' "$latest"
    [[ ! -f "$latest/worker.log" ]] || tail -n 35 -- "$latest/worker.log"
    if [[ -f "$latest/exit_code" ]]; then
      printf 'Pause helper exit code: '
      cat -- "$latest/exit_code"
    fi
  fi
  printf 'Current pause state:\n'
  "$training_python" -B -c 'import json; from large_qubit.section9_pause import pause_status; print(json.dumps(pause_status("."), indent=2))'
}
case "$action" in
  verify)
    printf 'Authenticating operation files and preserved registrations; this may take a few minutes.\n'
    verify_operations
    exec "$training_python" -u -B -m large_qubit.section9_control verify
    ;;
  pause)
    command -v tmux >/dev/null
    if tmux has-session -t "=$pause_session" 2>/dev/null; then
      printf 'A persistent pause is already pending. Use pause-status; do not start another helper.\n'
      exit 0
    fi
    printf 'Checking pause launcher files before starting the background helper...\n'
    verify_operations
    mkdir -p -- "$pause_jobs"
    attempt=$(mktemp -d "$pause_jobs/job_$(date -u +%Y%m%dT%H%M%S%N)_XXXXXXXX")
    sha256sum -- "$amendment" | cut -d ' ' -f 1 > "$attempt/operations_amendment.sha256"
    tmux new-session -d -s "$pause_session" -c "$project_root" \
      env "TRAINING_PYTHON=$training_python" \
      bash "$script_directory/run_section9_control_tmux.sh" __pause_worker "$attempt"
    printf 'Pause REQUESTED in %s. It will keep waiting for safe boundaries and survives SSH disconnection.\n' "$pause_session"
    printf 'This is not yet paused. Run: bash large_qubit/scripts/run_section9_control_tmux.sh pause-status\n'
    printf 'Log: %s/worker.log\n' "$attempt"
    ;;
  pause-status)
    pause_status
    ;;
  status)
    pause_status
    exec "$training_python" -B -m large_qubit.section9_control status
    ;;
  matrix-verify)
    verify_operations
    exec "$training_python" -B -m large_qubit.section9_control matrix-verify
    ;;
  reassign)
    command -v tmux >/dev/null
    require_pause_finished
    require_mask
    verify_operations
    exec "$training_python" -B -m large_qubit.section9_control reassign --execute
    ;;
  resume)
    command -v tmux >/dev/null
    require_pause_finished
    require_mask
    [[ -x "$training_python" && -x "$chemistry_python" ]] || { printf 'Both Python environments are required.\n' >&2; exit 2; }
    session=co2-s9-control-matrix_0001
    for existing in "$session" co2-s9-recovery-matrix_0001; do
      if tmux has-session -t "=$existing" 2>/dev/null; then
        printf 'Section9 session %s still exists. Finish the pause first.\n' "$existing" >&2
        exit 2
      fi
    done
    verify_operations
    "$training_python" -B -m large_qubit.section9_control verify >/dev/null
    logs="$project_root/large_qubit/evidence/section9_control_runs/matrix_0001"
    mkdir -p -- "$logs"
    attempt=$(mktemp -d "$logs/launch_XXXXXXXX")
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$attempt/gpu_mask.txt"
    sha256sum -- "$amendment" | cut -d ' ' -f 1 > "$attempt/operations_amendment.sha256"
    tmux new-session -d -s "$session" -c "$project_root" \
      env "TRAINING_PYTHON=$training_python" "CHEMISTRY_PYTHON=$chemistry_python" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      bash "$script_directory/run_section9_control_tmux.sh" __worker "$attempt"
    printf 'Started %s; log: %s/worker.log\n' "$session" "$attempt"
    ;;
  __worker|__pause_worker)
    attempt=${2:?Missing launch directory}
    [[ $# -eq 2 ]] || exit 2
    if [[ "$action" == __pause_worker ]]; then
      [[ "$attempt" == "$pause_jobs/job_"* ]] || exit 2
    else
      [[ "$attempt" == "$project_root/large_qubit/evidence/section9_control_runs/matrix_0001/launch_"* ]] || exit 2
    fi
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
    verify_operations
    if [[ "$action" == __pause_worker ]]; then
      "$training_python" -u -B -m large_qubit.section9_control pause --execute --timeout-seconds 0
    else
      "$training_python" -u -B -m large_qubit.section9_control resume --execute --chemistry-python "$chemistry_python"
    fi
    ;;
  *) printf 'Use verify, pause, pause-status, reassign, resume, status or matrix-verify.\n' >&2; exit 2 ;;
esac
