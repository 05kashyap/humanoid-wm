"""Render tracked states with the repo's native renderer and the Panda renderer, side by side.

Usage:  python -m phone2panda.render_compare --episodes episodes/ --out qa/render_compare.png
        (without --episodes it uses a few fixed test states)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .common import load_episode, make_env, wrap_angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="qa/render_compare.png")
    args = ap.parse_args()
    if args.episodes:
        files = sorted(Path(args.episodes).glob("*.npz"))[: args.n]
        states = []
        for f in files:
            e = load_episode(f)
            k = len(e["finger"]) // 2
            states.append(np.r_[e["finger"][k], e["tee"][k, :2], wrap_angle(e["tee"][k, 2]), 0, 0])
    else:
        states = [np.array(s, float) for s in [[256, 420, 256, 200, 0.0, 0, 0], [120, 120, 300, 330, 1.2, 0, 0],
                                               [400, 300, 180, 260, 2.6, 0, 0], [380, 256, 256, 256, 1.0, 0, 0]]]
    nat, pan = make_env(renderer="native"), make_env(renderer="panda")
    top, bot = [], []
    for s in states:
        nat.prepare(0, s); pan.prepare(0, s)
        top.append(nat.render_state(s)); bot.append(pan.render_state(s))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(np.vstack([np.hstack(top), np.hstack(bot)]), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out}: top row native renderer, bottom row Panda (same states)")


if __name__ == "__main__":
    main()
