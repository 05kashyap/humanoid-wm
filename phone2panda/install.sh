#!/usr/bin/env bash
# Run from the temporal-straightening repo root after unzipping phone2panda.zip there.
set -e
grep -q 'id="pusht_human"' env/__init__.py || cat >> env/__init__.py << 'PY'

register(
    id="pusht_human",  # phone2panda: the repo's Push-T physics + damping fit + Panda or native renderer
    entry_point="env.pusht.pusht_human_wrapper:PushTHumanWrapper",
    max_episode_steps=1000,
    reward_threshold=1.0,
)
PY
pip install "opencv-contrib-python>=4.8" "mujoco>=3.1" imageio imageio-ffmpeg scipy matplotlib
[ -d third_party/mujoco_menagerie ] || git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
echo "done. Add to your shell:  export MENAGERIE_DIR=$PWD/third_party/mujoco_menagerie MUJOCO_GL=egl"
