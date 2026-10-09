#!/usr/bin/env bash
# Rebuild everything from the recorded clips, unattended:
#   1. re-track every clip (recovers demos that were skipped for tracking gaps)
#   2. fresh train/val split, damping fit (also written into conf/env/pusht_human.yaml)
#   3. the four datasets, each checked right after it is built
#
# Run from the repo root, inside the repo's conda env, with DATASET_DIR exported:
#   nohup bash phone2panda/build_all.sh "raw/clip*.mp4" > qa/build_all.log 2>&1 &
#   tail -f qa/build_all.log
# Optional overrides: SETUP=other_setup.json  WORKERS=8  D=0.1 (skip the automatic damping choice)
set -euo pipefail
CLIPS=${1:-raw/clip*.mp4}
SETUP=${SETUP:-setup.json}
WORKERS=${WORKERS:-16}
: "${DATASET_DIR:?DATASET_DIR is not set; export it first}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1
mkdir -p episodes qa

echo "== 1. tracking ($(date))"
n_clips=0
for v in $CLIPS; do
  [ -f "$v" ] || { echo "no clip matches $CLIPS"; exit 1; }
  n_clips=$((n_clips + 1))
  log="qa/$(basename "$v" .mp4)_track.log"
  echo "-- $v"
  python -m phone2panda.track_video --setup "$SETUP" --video "$v" --out episodes/ > "$log" 2>&1 \
    || echo "   track_video FAILED on $v, see $log"
  grep -h "saved\|split\|skip" "$log" || true
done
echo "$n_clips clips tracked; episodes/ now holds $(ls episodes/*.npz | wc -l) episodes"

echo "== 2. split, damping ($(date))"
if [ -f split.json ]; then
  mv split.json "qa/split_old_$(date +%Y%m%d_%H%M%S).json"
  echo "moved the old split.json into qa/"
fi
python -m phone2panda.fit_damping --episodes episodes/ --split split.json --val-frac 0.2 --out qa/damping.json
D=${D:-$(python -c "import json; print(json.load(open('qa/damping.json'))['best']['damping'])")}
sed -i -E "s/^(  damping: )[0-9.eE+-]+/\1$D/" conf/env/pusht_human.yaml
echo "using damping $D; conf/env/pusht_human.yaml now says: $(grep '^  damping:' conf/env/pusht_human.yaml)"

build() {  # build <name> <make_dataset flags...>, then check it
  local name=$1; shift
  echo "== 3. $name ($(date))"
  rm -rf "${DATASET_DIR:?}/$name"
  python -m phone2panda.make_dataset --episodes episodes/ --split split.json --damping "$D" \
    --renderer panda --workers "$WORKERS" --out "$DATASET_DIR/$name" "$@"
  python -m phone2panda.check_dataset --data "$DATASET_DIR/$name" --damping "$D" --renderer panda 2>&1 \
    | grep "^\[" || true
}
build pusht_human_raw    --augment 0
build pusht_human_aug    --augment 16
build pusht_human_branch --augment 8 --branches 24 --match "$DATASET_DIR/pusht_human_aug"
build pusht_random       --random --match "$DATASET_DIR/pusht_human_aug"

echo "== done ($(date)); training steps per dataset:"
for name in pusht_human_raw pusht_human_aug pusht_human_branch pusht_random; do
  python -c "import json; m = json.load(open('$DATASET_DIR/$name/meta.json')); print(f'  $name: {m[\"train_steps\"]} steps')"
done
