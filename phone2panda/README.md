# phone2panda

Phone videos of my finger pushing a T  ->  physics replays in the repo's Push-T simulator
->  a Panda-rendered dataset in the repo's format  ->  train.py / plan.py unchanged.

Unzip at the repo root, then `bash phone2panda/install.sh`. The step-by-step runbook is in the
project doc; the commands in order:

```bash
python -m phone2panda.check_install                      # every dependency, 1 minute
python -m phone2panda.synth_test --out synth/            # whole video pipeline on a fake recording
python -m phone2panda.make_ipad_markers --width 2360 --height 1640 --out ipad_grid.png
python -m phone2panda.setup_frame --video raw/pilot.mp4 --frame 120 --out setup.json
python -m phone2panda.track_video --setup setup.json --video raw/s1_c01.mp4 --out episodes/ --preview qa/s1_c01.mp4
python -m phone2panda.fit_damping --episodes episodes/ --split split.json --out qa/damping.json
python -m phone2panda.render_compare --episodes episodes/ --out qa/render_compare.png
python -m phone2panda.make_dataset --episodes episodes/ --split split.json --out $DATASET_DIR/pusht_human_aug --augment 16 --damping 0 --renderer panda --workers 16
python -m phone2panda.check_dataset --data $DATASET_DIR/pusht_human_aug --damping 0 --renderer panda
python train.py --config-name train.yaml env=pusht_human encoder=dino_channel training.straighten=aggcos1e-1 training.encoder_lr=1e-5
python -m phone2panda.make_fulltask_targets --episodes episodes/ --split split.json --damping 0 --renderer panda --out plan_targets/fulltask.pkl
```

Environment variables: `MENAGERIE_DIR` (MuJoCo Menagerie clone), `MUJOCO_GL=egl` on GPU nodes
or `osmesa` on CPU, optional `PANDA_ARM_ALPHA=0.4` for a see-through arm.

Conventions (checked against env/pusht/pusht_env.py): arena units 0..512 with y down; the T's
state point is the midpoint of the bar's outer edge, stem along local +y; actions are relative
pusher targets in units, stored as-is in rel_actions.pth (the loader divides by 100). Demo
actions use a two-step lookahead, which cancels the PD controller's ~0.2 s lag.
