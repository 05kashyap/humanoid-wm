#!/usr/bin/env bash
# =============================================================================
# kaggle/kaggle_train.sh -- train the phone2panda world models in a Kaggle notebook.
# TRAINING ONLY (no MuJoCo, no planning); same runs as run_scripts/train_server.sh.
#
#   run        dataset              straighten   encoder lr
#   noreg      pusht_human_aug      False            1e-6
#   aug        pusht_human_aug      aggcos1e-1       1e-5
#   random     pusht_random         aggcos1e-1       1e-5
#   branch     pusht_human_branch   aggcos1e-1       1e-5
#   raw        pusht_human_raw      aggcos1e-1       1e-5
#
# Usage (notebook cell, after `!bash kaggle/kaggle_setup.sh`):
#   !DATASET_DIR=/kaggle/input/<your-dataset> bash kaggle/kaggle_train.sh aug random
#     -> on "GPU T4 x2" the two runs train IN PARALLEL, one per GPU; on one GPU, one after another.
#
# Kaggle-specific settings (as in temporal-straightening-PL/kaggle/kaggle_train.sh):
#   batch 16 (32 runs out of memory on a 16 GB T4) with the decoder ON; fp16 mixed precision
#   (T4/P100 have no bf16); 4 loader workers per run. If a run runs out of memory: DECODER=0.
#   The decoder is stop-gradient: it never changes the encoder or predictor, only adds images.
#   a checkpoint every 1000 iterations so a session cut at 12 h loses little.
#
# Environment variables:
#   DATASET_DIR  (required) the Kaggle input holding pusht_human_aug/, pusht_random/, ...
#                (found automatically up to two folders deep, since zips sometimes add a level)
#   EPOCHS       target total epochs per run (default 10). Runs resume, so a later session
#                with the same or a higher EPOCHS continues where the last one stopped.
#   RESUME_DIR   a Kaggle input holding a previous session's output (checkpoints/test/...);
#                each run's folder is copied into /kaggle/working before training
#   BATCH_SIZE=16  DECODER=1  NUM_WORKERS=4  CKPT_ROOT=/kaggle/working/checkpoints
#
# Output: $CKPT_ROOT/test/humanai_<run>/ (checkpoints/, hydra.yaml) and $CKPT_ROOT/logs/.
# Everything in /kaggle/working is kept as the notebook's output when you use
# "Save Version -> Save & Run All (Commit)"; turn that output into a Dataset to resume
# or to download the checkpoints.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

[[ $# -ge 1 ]] || { sed -n '2,40p' "$0"; exit 2; }
RUNS="$*"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DECODER="${DECODER:-1}"
if [[ "$DECODER" == 1 ]]; then DEC=True; else DEC=False; fi
NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_ROOT="${CKPT_ROOT:-/kaggle/working/checkpoints}"
: "${DATASET_DIR:?set DATASET_DIR=/kaggle/input/<your-dataset>}"

spec() {  # run -> "dataset straighten encoder_lr"
    case "$1" in
        noreg)    echo "pusht_human_aug    False 1e-6" ;;
        aug)      echo "pusht_human_aug    aggcos1e-1 1e-5" ;;
        random)   echo "pusht_random       aggcos1e-1 1e-5" ;;
        branch)   echo "pusht_human_branch aggcos1e-1 1e-5" ;;
        raw)      echo "pusht_human_raw    aggcos1e-1 1e-5" ;;
        *)        echo "" ;;
    esac
}

find_dataset() {  # the folder holding <name>/stats.pth, at most two levels below DATASET_DIR
    local hit
    hit="$(find -L "$DATASET_DIR" -maxdepth 3 -path "*/$1/stats.pth" -print -quit 2>/dev/null || true)"
    [[ -n "$hit" ]] && dirname "$hit"
}

# --- preflight ---------------------------------------------------------------
for r in $RUNS; do
    [[ -n "$(spec "$r")" ]] || { echo "unknown run '$r' (noreg aug random branch raw)" >&2; exit 2; }
    read -r ds _ <<< "$(spec "$r")"
    [[ -n "$(find_dataset "$ds")" ]] || { echo "dataset $ds not found under $DATASET_DIR" >&2; exit 1; }
done
conf_damping="$(grep -E '^\s+damping:' conf/env/pusht_human.yaml | head -1 | awk '{print $2}')"
NGPU="$(python -c 'import torch;print(torch.cuda.device_count())')"
echo "================================================================"
echo " runs: $RUNS   epochs=$EPOCHS batch=$BATCH_SIZE fp16, decoder=$DEC, GPUs=$NGPU"
echo " data: $DATASET_DIR   damping in conf: $conf_damping"
echo " ckpts: $CKPT_ROOT/test/humanai_<run>"
echo "================================================================"
[[ "$NGPU" -ge 1 ]] || { echo "no GPU: set the notebook accelerator to GPU T4 x2" >&2; exit 1; }

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$CKPT_ROOT/test" "$CKPT_ROOT/logs"

train_run() {  # $1 run, $2 GPU index
    local run="$1" gpu="$2" ds st lr data run_dir
    read -r ds st lr <<< "$(spec "$run")"
    data="$(find_dataset "$ds")"
    run_dir="$CKPT_ROOT/test/humanai_$run"

    # carry a previous session's checkpoints over
    if [[ -n "${RESUME_DIR:-}" && ! -f "$run_dir/checkpoints/model_latest.pth" ]]; then
        local prev
        prev="$(find -L "$RESUME_DIR" -maxdepth 5 -type d -name "humanai_$run" -print -quit 2>/dev/null || true)"
        if [[ -n "$prev" && -f "$prev/checkpoints/model_latest.pth" ]]; then
            echo "[$run] copying previous checkpoints from $prev"
            mkdir -p "$run_dir" && cp -r "$prev/." "$run_dir/"
        fi
    fi

    local meta_damping
    meta_damping="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["damping"])' "$data/meta.json" 2>/dev/null || echo "?")"
    if [[ "$meta_damping" != "?" ]] && ! python -c "import sys; sys.exit(abs(float('$meta_damping') - float('$conf_damping')) > 1e-9)"; then
        echo "[$run] WARNING: $ds was built with damping $meta_damping but conf/env/pusht_human.yaml says $conf_damping" >&2
    fi

    echo "[$run] GPU $gpu | $ds | straighten=$st lr=$lr | log: $CKPT_ROOT/logs/humanai_$run.log"
    local rc=0
    CUDA_VISIBLE_DEVICES="$gpu" python -u train.py --config-name train.yaml \
        env=pusht_human encoder=dino_channel \
        "env.dataset.data_path=$data" \
        "env.save_name=humanai_$run" \
        "training.straighten=$st" \
        "training.encoder_lr=$lr" \
        "training.batch_size=$BATCH_SIZE" \
        "training.epochs=$EPOCHS" \
        training.mixed_precision=fp16 \
        training.save_every_x_iterations=1000 \
        "has_decoder=$DEC" "model.train_decoder=$DEC" \
        "env.num_workers=$NUM_WORKERS" \
        "hydra.run.dir=$run_dir" \
        >> "$CKPT_ROOT/logs/humanai_$run.log" 2>&1 || rc=$?
    echo "[$run] finished with exit code $rc"
    return $rc
}

# --- dispatch: one run per GPU at a time ----------------------------------------
failed=()
set -- $RUNS
while [[ $# -gt 0 ]]; do
    pids=(); names=()
    for ((g = 0; g < NGPU && $# > 0; g++)); do
        train_run "$1" "$g" & pids+=($!); names+=("$1"); shift
        sleep 20
    done
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${names[$i]}"); done
done

echo "================================================================"
for r in $RUNS; do
    if [[ -f "$CKPT_ROOT/test/humanai_$r/checkpoints/model_latest.pth" ]]; then
        python - "$CKPT_ROOT/test/humanai_$r/checkpoints/model_latest.pth" "$r" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
print(f"  humanai_{sys.argv[2]}: epoch {ck.get('epoch')} (batch {ck.get('current_iter', 0)} into the next)")
PY
    else
        echo "  humanai_$r: no checkpoint"
    fi
done
[[ ${#failed[@]} -eq 0 ]] || { echo "FAILED: ${failed[*]} -- see $CKPT_ROOT/logs/"; exit 1; }
