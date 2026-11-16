#!/usr/bin/env bash
# Three independent H200 workers; invoking this script without an action previews.
set -euo pipefail
umask 077
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_directory/../.." && pwd -P)
training_python=${TRAINING_PYTHON:-$project_root/.venv/bin/python}
[[ "$training_python" == /* ]] || training_python="$project_root/$training_python"
action=${1:-preview}
fits=(gpt2_xl_weighted_lora gpt2_xl_uniform_lora qwen3_4b_weighted_lora)
fail() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }
[[ -x "$training_python" ]] || fail 'Set TRAINING_PYTHON to the pinned model-environment interpreter.'
cd -- "$project_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "$action" == __worker ]]; then
    (( $# == 5 )) || fail 'Invalid internal worker arguments.'
    stage=$2; fit=$3; gpu_uuid=$4; attempt=$5
    [[ "$stage" == preflight || "$stage" == train || "$stage" == resume ]] || fail 'Invalid stage.'
    [[ " ${fits[*]} " == *" $fit "* && "$gpu_uuid" =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ && "$attempt" =~ ^attempt_[0-9]{4}$ ]] || fail 'Invalid worker identity.'
    [[ "${CUDA_VISIBLE_DEVICES:-}" == "$gpu_uuid" ]] || fail 'Worker GPU UUID mask changed.'
    directory="$project_root/large_qubit/evidence/section7_runs/$stage/$fit/$attempt"
    [[ -d "$directory" && -f "$directory/launch.json" && ! -e "$directory/worker.pid" ]] || fail 'A fresh launch record is required.'
    set -o noclobber
    printf '%s\n' "$$" > "$directory/worker.pid"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$directory/worker.started_utc"
    exec >> "$directory/worker.log" 2>&1
    finish() {
        local code=$1
        trap - EXIT
        printf '%s\n' "$code" > "$directory/worker.exit_code"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$directory/worker.finished_utc"
        if [[ "$code" == 0 ]]; then : > "$directory/worker.success"; else : > "$directory/worker.failed"; fi
        exit "$code"
    }
    trap 'finish "$?"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    arguments=("$training_python" -u -B -m large_qubit.section7)
    if [[ "$stage" == preflight ]]; then
        "${arguments[@]}" preflight --execute --condition "$fit" --device cuda:0
    elif [[ "$stage" == resume ]]; then
        "${arguments[@]}" train --execute --resume --condition "$fit" --device cuda:0
    else
        "${arguments[@]}" train --execute --condition "$fit" --device cuda:0
    fi
    exit 0
fi

case "$action" in
    preview)
        (( $# <= 1 )) || fail 'preview takes no GPU arguments.'
        exec "$training_python" -B -m large_qubit.section7 plan
        ;;
    status)
        (( $# == 1 )) || fail 'status takes no GPU arguments.'
        if command -v tmux >/dev/null 2>&1; then
            tmux list-sessions -F '#{session_name}' 2>/dev/null | sed -n '/^co2-s7-/p' || true
        fi
        exec "$training_python" -B -m large_qubit.section7 status
        ;;
    preflight|train|resume)
        (( $# == 1 || $# == 4 )) || fail 'Usage: run_section7_tmux.sh preflight|train|resume [GPU0 GPU1 GPU2]'
        gpus=("${2:-0}" "${3:-1}" "${4:-2}")
        for gpu in "${gpus[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || fail 'GPU ordinals must be nonnegative integers.'; done
        [[ "${gpus[0]}" != "${gpus[1]}" && "${gpus[0]}" != "${gpus[2]}" && "${gpus[1]}" != "${gpus[2]}" ]] || fail 'Use three distinct GPU ordinals.'
        command -v tmux >/dev/null 2>&1 || fail 'tmux is missing; no workers started.'
        # Validate all requested session identities before launching any worker.
        for fit in "${fits[@]}"; do
            for stage in preflight train resume; do
                if tmux has-session -t "=co2-s7-$stage-$fit" 2>/dev/null; then
                    fail "A worker already exists for $fit; inspect status and its log."
                fi
            done
        done
        "$training_python" -B -m large_qubit.section7 verify >/dev/null
        # Resolve allocation-relative ordinals before masking each child to one UUID.
        assignment_output=$("$training_python" -B - "${gpus[@]}" <<'PY'
import sys
import torch
from large_qubit.gpu_identity import normalize_gpu_uuid
if not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
identities=[]
for text in sys.argv[1:]:
    index=int(text)
    if not 0 <= index < torch.cuda.device_count(): raise ValueError('GPU ordinal is outside the visible allocation')
    properties=torch.cuda.get_device_properties(index)
    if 'H200' not in properties.name.upper():
        raise ValueError(f'Assigned cuda:{index} must be an H200; found name={properties.name!r}')
    raw_identity=getattr(properties,'uuid','unavailable')
    try:
        identity=normalize_gpu_uuid(raw_identity)
    except ValueError as error:
        raise ValueError(f'Assigned cuda:{index} ({properties.name}) has an invalid GPU UUID '
                         f'{str(raw_identity)!r}') from error
    identities.append(identity)
if len(set(identities)) != 3: raise ValueError('The three allocation slots must identify distinct GPUs')
print('\n'.join(identities))
PY
)
        mapfile -t gpu_uuids <<< "$assignment_output"
        (( ${#gpu_uuids[@]} == 3 )) || fail 'Expected three resolved GPU UUIDs.'
        for index in 0 1 2; do
            fit=${fits[$index]}; gpu=${gpus[$index]}; gpu_uuid=${gpu_uuids[$index]}
            attempt=$("$training_python" -B - "$project_root" "$action" "$fit" "$gpu_uuid" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(root))
from large_qubit.common import workspace_path,require
stage,fit,gpu_uuid=sys.argv[2:]
base=workspace_path(root,'large_qubit/evidence/section7_runs/'+stage+'/'+fit,namespace='large_qubit/evidence/section7_runs')
base.mkdir(parents=True,exist_ok=True)
for ordinal in range(1,10000):
    directory=base/f'attempt_{ordinal:04d}'
    try: directory.mkdir()
    except FileExistsError: continue
    with (directory/'launch.json').open('x',encoding='utf-8') as handle:
        json.dump({'requested_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'fit':fit,'device':'cuda:0','gpu_uuid':gpu_uuid},handle)
    (directory/'worker.log').touch(exist_ok=False)
    print(directory.name)
    break
else: raise ValueError('Attempt counter exhausted')
PY
)
            session="co2-s7-$action-$fit"
            if ! tmux new-session -d -s "$session" -c "$project_root" \
                env "TRAINING_PYTHON=$training_python" "CUDA_VISIBLE_DEVICES=$gpu_uuid" bash "$script_directory/run_section7_tmux.sh" __worker "$action" "$fit" "$gpu_uuid" "$attempt"; then
                : > "$project_root/large_qubit/evidence/section7_runs/$action/$fit/$attempt/launch.failed"
                fail "Launch failed for $fit; preserve its attempt and inspect the other sessions."
            fi
            printf 'launched=%s allocation_ordinal=%s gpu_uuid=%s worker_device=cuda:0 session=%s attempt=%s\n' "$fit" "$gpu" "$gpu_uuid" "$session" "$attempt"
        done
        ;;
    *) fail 'Actions: preview, preflight, train, resume, status. Default: preview.' ;;
esac
