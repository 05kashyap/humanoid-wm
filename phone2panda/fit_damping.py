"""System identification: pick the pymunk damping that best reproduces how my real T moved.

Each training demo is physics-replayed (no rendering) for every candidate damping, and the
simulated T is compared with the tracked real T, at the end of the demo and along the way.
The repo uses damping 0 (the T stops the instant contact ends); larger values let it coast.

Usage:
  python -m phone2panda.fit_damping --episodes episodes/ --split split.json --out qa/damping.json
  (creates split.json if it does not exist yet; every dataset then reuses it)
"""
import argparse

import numpy as np

from .common import UNITS_PER_CM, load_episode, make_env, save_json
from .make_dataset import load_split, replay


def pose_error(sim, real):
    pos = np.linalg.norm(sim[..., :2] - real[..., :2], axis=-1) / UNITS_PER_CM
    ang = np.rad2deg(np.abs((sim[..., 2] - real[..., 2] + np.pi) % (2 * np.pi) - np.pi))
    return pos, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--split", default="split.json", help="created on first use, then reused by every dataset")
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of demos held out when the split is created")
    ap.add_argument("--dampings", default="0,0.01,0.1,0.3")
    ap.add_argument("--out", default="qa/damping.json")
    args = ap.parse_args()

    names = load_split(args.episodes, args.split, args.val_frac, 0)["train"]
    eps = [load_episode(f"{args.episodes}/{n}.npz") for n in names]
    rows = []
    for d in [float(x) for x in args.dampings.split(",")]:
        env = make_env(damping=d, renderer="none")
        fin_pos, fin_ang, traj_pos = [], [], []
        for e in eps:
            _, states, _, _ = replay(env, e["finger"], e["tee"][0], render=False)
            p, a = pose_error(states[-1, 2:5], e["tee"][-1])
            tp, _ = pose_error(states[:, 2:5], e["tee"])
            fin_pos.append(p); fin_ang.append(a); traj_pos.append(tp.mean())
        row = {"damping": d, "final_pos_cm_median": float(np.median(fin_pos)),
               "final_angle_deg_median": float(np.median(fin_ang)), "traj_pos_cm_mean": float(np.mean(traj_pos))}
        rows.append(row)
        print(f"damping {d:<5}: final T error median {row['final_pos_cm_median']:.2f} cm, "
              f"{row['final_angle_deg_median']:.1f} deg; mean error along the path {row['traj_pos_cm_mean']:.2f} cm")
    best = min(rows, key=lambda r: r["final_pos_cm_median"])
    print(f"best damping: {best['damping']}  (put it in conf/env/pusht_human.yaml and pass --damping to make_dataset)")
    save_json({"results": rows, "best": best}, args.out)


if __name__ == "__main__":
    main()
