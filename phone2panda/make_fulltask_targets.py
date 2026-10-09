"""Full-task planning goals (E3) and the open-loop replay baseline.

For each validation demo the start is my first tracked state and the goal is where my REAL
T ended up (tracked pose, not the simulated replay's end state). Writes a goal file for
`plan.py goal_source=file`, and reports how often simply replaying my own fingertip path in
the simulator reaches that goal.

Usage:
  python -m phone2panda.make_fulltask_targets --episodes episodes/ --split split.json
  bash run_scripts/plan_eval.sh full both random        # closed-loop planning on these goals
(--damping defaults to the fitted value in conf/env/pusht_human.yaml, as plan.py uses.)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from .common import UNITS_PER_CM, conf_damping, load_episode, load_json, make_env, wrap_angle
from .make_dataset import replay


def t_only_success(goal, cur):
    pos = np.linalg.norm(goal[2:4] - cur[2:4])
    ang = np.abs((goal[4] - cur[4] + np.pi) % (2 * np.pi) - np.pi)
    return bool(pos < 20 and ang < np.pi / 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--damping", type=float, default=None, help="default: conf/env/pusht_human.yaml")
    ap.add_argument("--renderer", default="panda", choices=["panda", "native"])
    ap.add_argument("--goal-H", type=int, default=25, help="planner horizon in env steps (multiple of 5)")
    ap.add_argument("--out", default="plan_targets/fulltask.pkl")
    args = ap.parse_args()
    if args.damping is None:
        args.damping = conf_damping()
    print(f"damping {args.damping}")

    names = load_json(args.split)["val"]
    env = make_env(damping=args.damping, renderer=args.renderer)
    obs0, obsg, s0, sg = {"visual": [], "proprio": []}, {"visual": [], "proprio": []}, [], []
    rep_repo, rep_t, rep_err = [], [], []
    for n in names:
        e = load_episode(Path(args.episodes) / f"{n}.npz")
        start = np.r_[e["finger"][0], e["tee"][0, :2], wrap_angle(e["tee"][0, 2]), 0.0, 0.0]
        goal = np.r_[e["finger"][-1], e["tee"][-1, :2], wrap_angle(e["tee"][-1, 2]), 0.0, 0.0]
        for state, o, s in ((start, obs0, s0), (goal, obsg, sg)):
            ob, _ = env.prepare(0, state)
            o["visual"].append(ob["visual"][None]); o["proprio"].append(ob["proprio"][None]); s.append(state)
        # open-loop baseline: replay my own path, compare with where the real T ended up
        _, states, _, _ = replay(env, e["finger"], e["tee"][0], render=False)
        final = np.r_[states[-1], 0.0, 0.0]
        rep_repo.append(bool(env.eval_state(goal, final)["success"]))
        rep_t.append(t_only_success(goal, final))
        rep_err.append(np.linalg.norm(goal[2:4] - final[2:4]) / UNITS_PER_CM)

    data = {"obs_0": {k: np.stack(v) for k, v in obs0.items()}, "obs_g": {k: np.stack(v) for k, v in obsg.items()},
            "state_0": np.stack(s0), "state_g": np.stack(sg), "gt_actions": None, "goal_H": args.goal_H}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"wrote {args.out}: {len(names)} goals (use n_evals={len(names)})")
    lens = [len(load_episode(Path(args.episodes) / f"{n}.npz")["finger"]) for n in names]
    print(f"demo lengths: median {np.median(lens):.0f} steps, max {max(lens)} (MPC budget: FULL_ITERS x 5 steps)")
    from .common import save_json
    save_json({"repo_success": float(np.mean(rep_repo)), "t_only_success": float(np.mean(rep_t)),
               "median_t_error_cm": float(np.median(rep_err)), "n": len(names)},
              str(Path(args.out).with_name(Path(args.out).stem + "_baseline.json")))
    print(f"open-loop replay of my own path: repo success {np.mean(rep_repo):.0%}, T-only success {np.mean(rep_t):.0%}, "
          f"median final T error {np.median(rep_err):.2f} cm")


if __name__ == "__main__":
    main()
