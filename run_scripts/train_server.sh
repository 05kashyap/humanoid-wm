#!/usr/bin/env bash
# =============================================================================
# run_scripts/train_server.sh -- train the phone2panda world models (TRAINING ONLY).
#
# One RUN = one world model:
#
#   run        dataset              straighten   encoder lr
#   aug        pusht_human_aug      aggcos1e-1   1e-5   E1 main model
#   random     pusht_random         aggcos1e-1   1e-5   E1 baseline
#   noreg      pusht_human_aug      False        1e-6   E2 baseline
#   branch     pusht_human_branch   aggcos1e-1   1e-5   E5
#   raw        pusht_human_raw      aggcos1e-1   1e-5   E1 (no augmentation)
#
# The pacing-regularizer runs are not part of this release (code released with the paper).
# Encoder: frozen DINOv2 ViT-S/14 + trainable channel projector (14x14x8) + MLP aggregation
# head, decoder ON (for reconstructions only; it is stop-gradiented). Batch 32, num_hist 3,
# frameskip 5 -- the paper's settings, as in temporal-straightening-PL/run_scripts/train_server.sh.
#
# Usage (inside the container with the ts env active, from anywhere in the repo):
#   bash run_scripts/train_server.sh aug                 # one run
#   bash run_scripts/train_server.sh aug random noreg    # several, sequentially
#   bash run_scripts/train_server.sh arms                # the four arms CONCURRENTLY on one GPU
#   bash run_scripts/train_server.sh all                 # all eight, 4 at a time on one GPU
#   DRY_RUN=1 bash run_scripts/train_server.sh all       # print the commands, train nothing
# Usually submitted through run_scripts/train_server.slurm.
#
# ALL MODE: the runs share one GPU, MAX_PARALLEL at a time (default 4; a run needs about
# 20 GB, so 4 leave headroom on an H200), started in the priority order above; when one
# finishes the next starts. Torch caps a task at 16 CPUs, so NUM_WORKERS defaults to 3 per
# run in all mode (4 trainers + 12 loaders).
#
# RESUMING: train.py resumes from <run_dir>/checkpoints/model_latest.pth, and EPOCHS is the
# TARGET total (training.epochs_mode=target): rerunning a job finishes what is left.
# SKIP_FINISHED=1 (default) skips runs already at EPOCHS; FRESH=1 deletes a run dir first.
#
# WHERE THINGS GO
#   run dirs : $CKPT_ROOT/test/humanai_<run>/   (checkpoints/, hydra.yaml, rollout_plots/, wandb/)
#   logs     : $CKPT_ROOT/logs/humanai_<run>.log
#
# Knobs (env vars): DATA_ROOT CKPT_ROOT EPOCHS=20 BATCH_SIZE=32 NUM_WORKERS MAX_PARALLEL=4
#   STAGGER=30 FRESH=0 SKIP_FINISHED=1
#   DRY_RUN=0 WANDB_MODE=offline PYTHON
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../train.py" ]]; then
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/train.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$PWD"
fi
cd "$REPO"

SCRATCH="${SCRATCH:?set SCRATCH to your scratch space}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/worldmodeldata/pusht_human}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/datasets/worldmodelcheckpoints}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
STAGGER="${STAGGER:-30}"
FRESH="${FRESH:-0}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
DRY_RUN="${DRY_RUN:-0}"
PY="${PYTHON:-$(command -v python || echo python)}"
ALL_RUNS="noreg aug random branch raw"   # priority order
ARMS_RUNS="noreg aug"                     # the regularizer arms

[[ $# -ge 1 ]] || { sed -n '2,45p' "$0"; exit 2; }
if [[ "$1" == "all" || "$1" == "arms" ]]; then
    MODE=all; NUM_WORKERS="${NUM_WORKERS:-3}"
    if [[ "$1" == "all" ]]; then RUNS="$ALL_RUNS"; else RUNS="$ARMS_RUNS"; fi
else
    MODE=seq; RUNS="$*"; NUM_WORKERS="${NUM_WORKERS:-}"
fi

# run -> "dataset straighten encoder_lr"
spec() {
    case "$1" in
        aug)      echo "pusht_human_aug    aggcos1e-1 1e-5" ;;
        random)   echo "pusht_random       aggcos1e-1 1e-5" ;;
        noreg)    echo "pusht_human_aug    False      1e-6" ;;
        branch)   echo "pusht_human_branch aggcos1e-1 1e-5" ;;
        raw)      echo "pusht_human_raw    aggcos1e-1 1e-5" ;;
        *)        echo "" ;;
    esac
}

# ─── preflight ──────────────────────────────────────────────────────────────
problems=()
for r in $RUNS; do
    [[ -n "$(spec "$r")" ]] || problems+=("unknown run '$r' (choose from: $ALL_RUNS, or all)")
done
if ! py_info="$("$PY" -c 'import torch,sys;print(sys.version.split()[0],"torch",torch.__version__,"cuda",torch.cuda.is_available(),(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))' 2>&1)"; then
    problems+=("$PY cannot import torch (inside the container with 'conda activate ts'?): $py_info")
fi
conf_damping="$(grep -E '^\s+damping:' conf/env/pusht_human.yaml | head -1 | awk '{print $2}')"
for r in $RUNS; do
    read -r ds _ <<< "$(spec "$r")"
    [[ -n "${ds:-}" ]] || continue
    d="$DATA_ROOT/$ds"
    for f in train/states.pth val/states.pth stats.pth meta.json; do
        [[ -e "$d/$f" ]] || { problems+=("dataset incomplete: $d/$f missing"); break; }
    done
    if [[ -f "$d/meta.json" ]]; then
        data_damping="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["damping"])' "$d/meta.json" 2>/dev/null || echo "?")"
        if [[ "$data_damping" != "?" ]] && ! "$PY" -c "import sys; sys.exit(abs(float('$data_damping') - float('$conf_damping')) > 1e-9)" 2>/dev/null; then
            problems+=("$ds was built with damping $data_damping but conf/env/pusht_human.yaml says $conf_damping (commit + pull the yaml)")
        fi
    fi
done

echo "=========================================================================="
echo "train_server.sh  mode=$MODE  runs: $RUNS"
echo "  python    : ${py_info:-?}"
echo "  data      : $DATA_ROOT   (damping $conf_damping)"
echo "  ckpts     : $CKPT_ROOT/test/humanai_<run>"
echo "  optim     : batch=$BATCH_SIZE epochs=$EPOCHS (target total) num_workers=${NUM_WORKERS:-yaml}"
[[ "$MODE" == all ]] && echo "  parallel  : $MAX_PARALLEL runs at a time on one GPU, ${STAGGER}s apart"
if [[ ${#problems[@]} -gt 0 ]]; then
    echo "PREFLIGHT FAILURES:" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    [[ "$DRY_RUN" == 1 ]] || exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET_DIR="$DATA_ROOT"          # conf/env/pusht_human.yaml and plan.py read it
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$MODE" == all ]]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / MAX_PARALLEL > 0 ? $(nproc) / MAX_PARALLEL : 1 ))}"
fi
mkdir -p "$CKPT_ROOT/logs" "$CKPT_ROOT/test"

# ─── one run ────────────────────────────────────────────────────────────────
train_run() {
    local run="$1" ds st lr
    read -r ds st lr <<< "$(spec "$run")"
    local name="humanai_$run" run_dir="$CKPT_ROOT/test/humanai_$run"
    local latest="$run_dir/checkpoints/model_latest.pth"

    if [[ "$FRESH" == 1 && -d "$run_dir" ]]; then
        if [[ "$DRY_RUN" == 1 ]]; then echo "[$run] would rm -rf $run_dir"; else echo "[$run] FRESH=1: removing $run_dir"; rm -rf "$run_dir"; fi
    fi
    if [[ "$SKIP_FINISHED" == 1 && "$FRESH" != 1 && -f "$latest" ]]; then
        local saved
        saved="$("$PY" -c 'import sys,torch;ck=torch.load(sys.argv[1],map_location="cpu");print(int(ck.get("epoch") or 0), int(ck.get("current_iter") or 0))' "$latest" 2>/dev/null || echo "0 0")"
        local se si; read -r se si <<< "$saved"
        if [[ "$se" -ge "$EPOCHS" && "$si" -eq 0 ]]; then
            echo "[$run] already at epoch $se >= EPOCHS=$EPOCHS, skipping (raise EPOCHS or FRESH=1)"
            return 0
        fi
        echo "[$run] resuming from epoch $se (batch $si) -> target $EPOCHS"
    fi

    local cmd=( "$PY" -u train.py --config-name train.yaml
                env=pusht_human encoder=dino_channel
                "env.dataset.data_path=$DATA_ROOT/$ds"
                "env.save_name=$name"
                "training.straighten=$st"
                "training.encoder_lr=$lr"
                "training.batch_size=$BATCH_SIZE"
                "training.epochs=$EPOCHS"
                has_decoder=True model.train_decoder=True
                "hydra.run.dir=$run_dir" )
    [[ -n "$NUM_WORKERS" ]] && cmd+=( "env.num_workers=$NUM_WORKERS" )

    echo "[$run] $ds | straighten=$st lr=$lr -> $run_dir"
    printf '    $'; printf ' %q' "${cmd[@]}"; printf '\n'
    [[ "$DRY_RUN" == 1 ]] && return 0
    local rc=0
    "${cmd[@]}" >> "$CKPT_ROOT/logs/$name.log" 2>&1 || rc=$?
    echo "[$run] finished with exit code $rc ($(date '+%F %H:%M')); log: $CKPT_ROOT/logs/$name.log"
    return $rc
}

# ─── dispatch ───────────────────────────────────────────────────────────────
failed=()
if [[ "$MODE" == seq ]]; then
    for r in $RUNS; do train_run "$r" || failed+=("$r"); done
else
    declare -A pid_of=()
    for r in $RUNS; do
        while [[ $(jobs -rp | wc -l) -ge "$MAX_PARALLEL" ]]; do wait -n || true; done
        train_run "$r" &
        pid_of[$r]=$!
        [[ "$DRY_RUN" == 1 ]] || sleep "$STAGGER"
    done
    for r in $RUNS; do wait "${pid_of[$r]}" || failed+=("$r"); done
fi

echo "=========================================================================="
echo "done at $(date '+%F %H:%M')"
for r in $RUNS; do
    if [[ -f "$CKPT_ROOT/test/humanai_$r/checkpoints/model_latest.pth" ]]; then echo "  [ok]     humanai_$r"; else echo "  [no ckpt] humanai_$r"; fi
done
if [[ ${#failed[@]} -gt 0 ]]; then echo "FAILED: ${failed[*]} (see $CKPT_ROOT/logs/)"; exit 1; fi
