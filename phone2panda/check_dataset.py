"""Check a dataset written by make_dataset.py before training on it.

  1. loads it through the repo's own loader and slicer, exactly as train.py will
  2. replays the stored actions in the env from each episode's first state and checks that the
     stored states come back (plan.py builds its planning goals this way)
  3. checks every video has one frame per step, and that frames match a fresh render

Usage:
  python -m phone2panda.check_dataset --data $DATASET_DIR/pusht_human_aug --damping 0 --renderer panda
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .common import REPO_ROOT, make_env


def load_arr(path):
    import torch
    x = torch.load(path)
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--renderer", default="panda", choices=["panda", "native"])
    ap.add_argument("--n", type=int, default=5, help="episodes to replay per split")
    args = ap.parse_args()
    root = Path(args.data)

    # 1. the repo's loader
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from datasets.img_transforms import default_transform
        from datasets.pusht_human_dset import load_pusht_human_slice_train_val
        slices, trajs = load_pusht_human_slice_train_val(default_transform(224), data_path=str(root),
                                                         num_hist=3, num_pred=1, frameskip=5)
        obs, act, state = slices["train"][0]
        print(f"[loader] train slices {len(slices['train'])}, val slices {len(slices['valid'])}; one slice: "
              f"visual {tuple(obs['visual'].shape)}, proprio {tuple(obs['proprio'].shape)}, "
              f"action {tuple(act.shape)}, state {tuple(state.shape)}")
        print(f"[loader] normalised actions: mean {act.mean():.2f}, std {act.std():.2f} (expect about 0 and 1)")
    except Exception as e:  # noqa: BLE001
        print(f"[loader] skipped ({type(e).__name__}: {e}); run this inside the repo's environment")

    env = make_env(damping=args.damping, renderer=args.renderer)
    import imageio.v2 as imageio
    for split in ("train", "val"):
        d = root / split
        states, vels, acts = load_arr(d / "states.pth"), load_arr(d / "velocities.pth"), load_arr(d / "rel_actions.pth")
        lengths = pickle.load(open(d / "seq_lengths.pkl", "rb"))
        # 2. replay consistency
        rng = np.random.default_rng(0)
        worst = 0.0
        # branch rollouts start mid-push (the T's own velocity is not stored), so they cannot be
        # replayed from their first state exactly; test the episodes that start at rest
        cand = np.arange(len(lengths))
        meta_path = root / "meta.json"
        if split == "train" and meta_path.exists():
            kinds = [m.get("aug", "") for m in json.load(open(meta_path))["episodes"]]
            if len(kinds) == len(lengths):
                cand = np.array([i for i, k in enumerate(kinds) if not str(k).startswith("branch")])
                if len(cand) < len(lengths):
                    print(f"[{split}] {len(lengths) - len(cand)} branch rollouts (replay test runs on the others)")
        for i in rng.choice(cand, min(args.n, len(cand)), replace=False):
            L = lengths[i]
            env.prepare(0, np.r_[states[i, 0], vels[i, 0]])
            got = [states[i, 0]]
            for a in acts[i, :L - 1]:
                _, _, _, info = env.step(a / 100.0)
                got.append(info["state"][:5])
            worst = max(worst, float(np.abs(np.array(got) - states[i, :L]).max()))
        # 3. videos
        bad = 0
        diffs = []
        for i, L in enumerate(lengths):
            r = imageio.get_reader(str(d / "obses" / f"episode_{i:03d}.mp4"))
            n = r.count_frames()
            if n != L:
                bad += 1
            if i < 3:
                f0 = r.get_data(0)
                env.prepare(0, np.r_[states[i, 0], vels[i, 0]])
                diffs.append(np.abs(f0.astype(float) - env.render_state(states[i, 0]).astype(float)).mean())
            r.close()
        print(f"[{split}] {len(lengths)} episodes, {sum(lengths)} steps | replay max state error {worst:.4f} units "
              f"(expect ~0) | videos with wrong frame count: {bad} | first-frame vs fresh render: "
              f"{np.mean(diffs):.1f}/255 mean abs diff (mp4 compression, expect < 3)")


if __name__ == "__main__":
    main()
