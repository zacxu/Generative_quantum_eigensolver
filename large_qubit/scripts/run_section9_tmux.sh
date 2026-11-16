#!/usr/bin/env bash
# Explicit user launch only. One H200, two isolated resident CPU target workers.
set -euo pipefail
umask 077
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
training_python=${TRAINING_PYTHON:-$project_root/.venv/bin/python}
chemistry_python=${CHEMISTRY_PYTHON:-$project_root/.venv-chemistry/bin/python}
action=${1:-plan}
cd -- "$project_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$action" in
  plan|verify|status|prepare|matrix-verify)
    args=("$training_python" -B -m large_qubit.section9 "$action")
    [[ "$action" != prepare ]] || args+=(--execute)
    exec "${args[@]}"
    ;;
  run|resume)
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
      printf 'Set CUDA_VISIBLE_DEVICES to exactly one allocated H200 index or full GPU UUID.\n' >&2; exit 2;
    }
    [[ -x "$training_python" && -x "$chemistry_python" ]] || { printf 'Both Python environments are required.\n' >&2; exit 2; }
    command -v tmux >/dev/null
    session=co2-s9-matrix_0001
    if tmux has-session -t "=$session" 2>/dev/null; then printf 'Main session already exists.\n' >&2; exit 2; fi
    # Read-only authentication and preparation precede the detached launch.
    # Missing/incomplete Section8 freeze refuses without loading models or target arrays.
    "$training_python" -B -m large_qubit.section9 prepare --execute >/dev/null
    logs="$project_root/large_qubit/evidence/section9_runs/matrix_0001"
    mkdir -p -- "$logs"
    attempt=$(mktemp -d "$logs/launch_XXXXXXXX")
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$attempt/gpu_mask.txt"
    tmux new-session -d -s "$session" -c "$project_root" \
      env "TRAINING_PYTHON=$training_python" "CHEMISTRY_PYTHON=$chemistry_python" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      bash "$script_directory/run_section9_tmux.sh" __worker "$action" "$attempt"
    printf 'Started %s; log: %s/worker.log\n' "$session" "$attempt"
    ;;
  __worker)
    stage=$2
    attempt=$3
    [[ "$stage" == run || "$stage" == resume ]] || exit 2
    [[ "$attempt" == "$project_root/large_qubit/evidence/section9_runs/matrix_0001/launch_"* ]] || exit 2
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
    args=("$training_python" -u -B -m large_qubit.section9 run --execute --chemistry-python "$chemistry_python")
    [[ "$stage" != resume ]] || args+=(--resume)
    "${args[@]}"
    ;;
  *) printf 'Use verify, run, resume, status, matrix-verify, prepare or plan.\n' >&2; exit 2 ;;
esac
