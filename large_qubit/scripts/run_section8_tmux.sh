#!/usr/bin/env bash
# One user-selected GPU, two resident backbones in sequence, isolated CPU scorer.
set -euo pipefail
umask 077
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
training_python=${TRAINING_PYTHON:-$project_root/.venv/bin/python}
chemistry_python=${CHEMISTRY_PYTHON:-$project_root/.venv-chemistry/bin/python}
action=${1:-plan}
panel=${2:-panel_0001}
[[ "$panel" =~ ^panel_[0-9]{4}$ ]] || { printf 'Invalid panel name\n' >&2; exit 2; }
cd -- "$project_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$action" in
  plan|verify|status|prepare|freeze|freeze-verify)
    args=("$training_python" -B -m large_qubit.section8 "$action" --panel "$panel")
    if [[ "$action" == prepare || "$action" == freeze ]]; then args+=(--execute); fi
    exec "${args[@]}"
    ;;
  run|resume)
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
      printf 'Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU index or full UUID.\n' >&2; exit 2;
    }
    [[ -x "$training_python" && -x "$chemistry_python" ]] || { printf 'Both Python environments are required.\n' >&2; exit 2; }
    command -v tmux >/dev/null
    session="co2-s8-$panel"
    if tmux has-session -t "=$session" 2>/dev/null; then printf 'Panel session already exists.\n' >&2; exit 2; fi
    "$training_python" -B -m large_qubit.section8 verify >/dev/null
    # Preparation cannot load a model or score a circuit, and refuses unfinished fits.
    "$training_python" -B -m large_qubit.section8 prepare --execute >/dev/null
    logs="$project_root/large_qubit/evidence/section8_runs/$panel"
    mkdir -p -- "$logs"
    attempt=$(mktemp -d "$logs/launch_XXXXXXXX")
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$attempt/gpu_mask.txt"
    tmux new-session -d -s "$session" -c "$project_root" \
      env "TRAINING_PYTHON=$training_python" "CHEMISTRY_PYTHON=$chemistry_python" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "SECTION8_RERUN_REASON=${SECTION8_RERUN_REASON:-}" \
      bash "$script_directory/run_section8_tmux.sh" __worker "$panel" "$action" "$attempt"
    printf 'Started %s; log: %s/worker.log\n' "$session" "$attempt"
    ;;
  __worker)
    stage=$3
    attempt=$4
    [[ "$stage" == run || "$stage" == resume ]] || exit 2
    [[ "$attempt" == "$project_root/large_qubit/evidence/section8_runs/$panel/launch_"* ]] || exit 2
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
    args=("$training_python" -u -B -m large_qubit.section8 run --execute --panel "$panel" --chemistry-python "$chemistry_python")
    [[ "$stage" != resume ]] || args+=(--resume)
    [[ -z "${SECTION8_RERUN_REASON:-}" ]] || args+=(--rerun-reason "$SECTION8_RERUN_REASON")
    "${args[@]}"
    ;;
  *) printf 'Use plan, verify, prepare, run, resume, status, freeze or freeze-verify.\n' >&2; exit 2 ;;
esac
