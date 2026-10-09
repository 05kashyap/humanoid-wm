#!/usr/bin/env bash
# =============================================================================
# run_scripts/plan_eval.sh -- plan with several models on ONE fixed task set and compare.
#
#   bash run_scripts/plan_eval.sh ol aug random noreg both     # open loop  (GD, 25 actions at once)
#   bash run_scripts/plan_eval.sh cl aug random noreg both     # closed loop (GD-MPC, replans every 5)
#   bash run_scripts/plan_eval.sh full both                    # E3: my first state -> where my real T
#                                                              #     ended (phone2panda.make_fulltask_targets)
#   bash run_scripts/plan_eval.sh photo both                   # T poses filmed on the table
#                                                              #     (phone2panda.goal_from_photo)
#
# Closed loop (cl, full, photo) follows the repo's Push-T protocol: GD-MPC with
# objective.mode=staged. full and photo use every task in their file and a longer budget,
# since a whole demo is longer than 25 steps; score them with the report's "T only" column.
#
# Tasks: start/goal pairs cut from the 20 held-out human demos (the val/ split, which is the
# same un-augmented demos in every dataset, random included). The first run of all creates
# the task file $TASKS (N_OL tasks); every run after that, open or closed loop, any model,
# loads it, so all results are paired. Closed loop uses the first N_CL of the same tasks.
#
# Each run goes to plan_outputs/<ol|cl>/<model>/ (videos, logs.json, final_eval_results.npz),
# its output to plan_outputs/<ol|cl>/<model>.log. Finished runs are skipped, so the script
# can be rerun after an interruption or with more models. A report is printed at the end.
#
# Knobs (environment): CKPT=$HOME/wm_ckpts/checkpoints/test  TASKS=plan_targets/tasks.pkl
#   N_OL=200  N_CL=50  CHUNK=2  CL_ITERS=10 (MPC rounds of 5 actions: 50 steps, twice the
#   human's 25)  FULL_ITERS=40 (200 steps, for full and photo)  SEED=100  GOAL_H=25
#   FULL_TASKS=plan_targets/fulltask.pkl  PHOTO_TASKS=plan_targets/photo.pkl
#   EXTRA="<more plan.py overrides>"
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: bash run_scripts/plan_eval.sh ol|cl model [model ...]}"; shift
[[ "$MODE" =~ ^(ol|cl|full|photo)$ ]] || { echo "mode must be ol, cl, full or photo" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo "name at least one model (aug random noreg both branch ...)" >&2; exit 2; }
CKPT="${CKPT:-$HOME/wm_ckpts/checkpoints/test}"
TASKS="${TASKS:-plan_targets/tasks.pkl}"
N_OL="${N_OL:-200}"; N_CL="${N_CL:-50}"; CHUNK="${CHUNK:-2}"; CL_ITERS="${CL_ITERS:-10}"
SEED="${SEED:-100}"; GOAL_H="${GOAL_H:-25}"; FULL_ITERS="${FULL_ITERS:-40}"
FULL_TASKS="${FULL_TASKS:-plan_targets/fulltask.pkl}"; PHOTO_TASKS="${PHOTO_TASKS:-plan_targets/photo.pkl}"
case "$MODE" in
    full)  TASKS="$FULL_TASKS" ;;
    photo) TASKS="$PHOTO_TASKS" ;;
esac
if [[ "$MODE" == full || "$MODE" == photo ]]; then
    [[ -f "$TASKS" ]] || { echo "no task file $TASKS: make it with phone2panda.$([[ $MODE == full ]] && echo make_fulltask_targets || echo goal_from_photo)" >&2; exit 1; }
    N_FILE="$(python -c 'import pickle,sys;print(len(pickle.load(open(sys.argv[1],"rb"))["state_0"]))' "$TASKS")"
fi
mkdir -p "plan_outputs/$MODE" "$(dirname "$TASKS")"

run() {  # $1 model, $2.. extra overrides
    local m="$1"; shift
    local out="plan_outputs/$MODE/$m"
    if [[ -f "$out/final_eval_results.npz" ]]; then echo "[$MODE/$m] done already, skipping"; return; fi
    [[ -f "$CKPT/humanai_$m/hydra.yaml" ]] || { echo "[$MODE/$m] no model at $CKPT/humanai_$m" >&2; return 1; }
    case "$MODE" in
        ol) local cfg=plan_gd.yaml n="$N_OL" extra=() ;;
        cl) local cfg=plan_gd_mpc.yaml n="$N_CL" extra=("planner.max_iter=$CL_ITERS" objective.mode=staged decode_for_viz=false) ;;
        *)  local cfg=plan_gd_mpc.yaml n="$N_FILE" extra=("planner.max_iter=$FULL_ITERS" objective.mode=staged) ;;
    esac
    echo "[$MODE/$m] $n tasks, chunks of $CHUNK -> $out  ($(date +%H:%M))"
    : > "$out.log"
    if [[ -t 1 ]]; then progress "$out.log" "$MODE/$m" & MON=$!; fi
    local rc=0
    python plan.py --config-name "$cfg" \
        "ckpt_base_path=$CKPT/humanai_$m" "model_name=humanai_$m" \
        "n_evals=$n" "chunk_size=$CHUNK" "seed=$SEED" "goal_H=$GOAL_H" objective.alpha=1 \
        "hydra.run.dir=$out" ${extra[@]+"${extra[@]}"} ${EXTRA:-} "$@" > "$out.log" 2>&1 || rc=$?
    stop_progress
    if [[ $rc -ne 0 ]]; then echo "[$MODE/$m] FAILED, see $out.log" >&2; tail -n 5 "$out.log" >&2; return 1; fi
    echo "[$MODE/$m] $(grep 'Success rate' "$out.log" | tail -1)  ($(date +%H:%M))"
}

# progress bar (terminal only): chunks planned so far, from the "[progress]" lines plan.py prints
MON=""
progress() {  # $1 log, $2 label
    local log="$1" label="$2" t0=$SECONDS ts=0 k=0 n line bar el eta r mpc w=30
    while kill -0 $$ 2>/dev/null; do
        line="$(grep '^\[progress\]' "$log" 2>/dev/null | tail -1 || true)"
        el=$((SECONDS - t0))
        # closed loop: MPC rounds done in the chunk being planned now
        r="$(awk '/^\[progress\]/{c=0} /^MPC iter/{c++} END{print c+0}' "$log" 2>/dev/null || echo 0)"
        mpc=""; [[ "$r" -gt 0 ]] && mpc=", chunk $(( ${k:-0} + 1 )): MPC round $r"
        grep -q 'setup_workspace_s' "$log" 2>/dev/null || ts=$el
        if [[ "$line" =~ chunk\ ([0-9]+)/([0-9]+) ]]; then
            k=${BASH_REMATCH[1]}; n=${BASH_REMATCH[2]}
            bar="$(printf '%*s' $((k * w / n)) '' | tr ' ' '#')$(printf '%*s' $((w - k * w / n)) '' | tr ' ' '.')"
            eta=$(((el - ts) * (n - k) / k))  # time per chunk measured after the setup
            printf '\r\033[K  %s [%s] %d/%d chunks  %dm%02ds elapsed, ~%dm left%s' \
                "$label" "$bar" "$k" "$n" $((el / 60)) $((el % 60)) $(((eta + 59) / 60)) "$mpc"
            [[ "$line" == *final* ]] && printf ' (final evaluation)'
        elif [[ "$line" == *final* ]]; then
            printf '\r\033[K  %s final evaluation  %dm%02ds elapsed' "$label" $((el / 60)) $((el % 60))
        elif grep -q 'setup_workspace_s' "$log" 2>/dev/null; then
            printf '\r\033[K  %s planning%s  %dm%02ds' "$label" "${mpc:-, chunk 1}" $((el / 60)) $((el % 60))
        else
            printf '\r\033[K  %s loading model and preparing tasks  %dm%02ds' "$label" $((el / 60)) $((el % 60))
        fi
        sleep 5
    done
}
stop_progress() {
    if [[ -n "$MON" ]]; then kill "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true; MON=""; printf '\r\033[K'; fi
}
trap stop_progress EXIT

for m in "$@"; do
    if [[ ! -f "$TASKS" ]]; then
        # the very first run samples the tasks from the held-out human demos and saves them
        [[ "$MODE" == ol ]] || { echo "no task file $TASKS yet: run 'ol' first" >&2; exit 1; }  # (cl only)
        run "$m" goal_source=dset
        cp "plan_outputs/ol/$m/plan_targets.pkl" "$TASKS"
        echo "task file: $TASKS ($N_OL tasks)"
    else
        run "$m" goal_source=file "+goal_file_path=$PWD/$TASKS" || true
    fi
done

args=()
for m in "$@"; do [[ -f "plan_outputs/$MODE/$m/final_eval_results.npz" ]] && args+=("$m=plan_outputs/$MODE/$m"); done
if [[ ${#args[@]} -gt 0 ]]; then
    echo; python helpers/plan_report.py "plan_outputs/$MODE/${args[0]%%=*}/plan_targets.pkl" "${args[@]}"
fi
