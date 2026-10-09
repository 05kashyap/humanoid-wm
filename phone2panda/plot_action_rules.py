"""Figure 2: three ways to turn a fingertip path into the repo's pusher commands.

Replays one tracked path in the simulator (T removed, pusher only) with naive deltas,
track-the-next-point, and the two-step lookahead used for the dataset, and plots the paths.

Usage:  python -m phone2panda.plot_action_rules --episode episodes/s1_c01_ep03.npz --out qa/action_rules.png
"""
import argparse

import numpy as np

from .common import UNITS_PER_CM, load_episode, make_env


def run(env, path, rule):
    env.prepare(0, np.r_[path[0], 470.0, 40.0, 0.0, 0.0, 0.0])  # park the T in a corner
    xs = [path[0]]
    for t in range(len(path) - 1):
        ag = np.array(env.agent.position)
        if rule == "naive delta":
            d = path[t + 1] - path[t]
        elif rule == "track next":
            d = path[t + 1] - ag
        else:
            d = path[min(t + 2, len(path) - 1)] - ag
        env.step(d / 100.0)
        xs.append(np.array(env.agent.position))
    return np.array(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default="qa/action_rules.png")
    args = ap.parse_args()
    path = load_episode(args.episode)["finger"]
    env = make_env(renderer="none")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*path.T, "k-", lw=3, alpha=0.3, label="my fingertip")
    for rule in ["naive delta", "track next", "lookahead 2"]:
        xs = run(env, path, rule)
        err = np.linalg.norm(xs - path, axis=1).mean() / UNITS_PER_CM
        ax.plot(*xs.T, lw=1.5, label=f"{rule}: mean error {err:.1f} cm")
    ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.set_aspect("equal")
    ax.set_xlabel("arena x (units)"); ax.set_ylabel("arena y (units, down)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
