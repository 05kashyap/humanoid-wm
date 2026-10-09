"""Diagnose a tracked episode whose physics replay does not match what really happened.

Prints three numbers and writes a side-by-side video:
  * how far the real T moved vs how far the replayed T moved
  * T size check: tracked T area vs the repo's T (a scale or corner problem shows here)
  * contact distance: while the real T is moving, how far the tracked fingertip is from the
    T's outline. The sim pusher has a 15-unit (1 cm) radius, so a correct fingertip track
    sits about +15 units from the outline when pushing. Much larger means the tracked point
    is behind the real contact point (tape on the nail, camera angle); negative means inside.
Video: left = what really happened (tracked states drawn by the repo renderer),
       right = physics replay of your fingertip path.

Usage:
  python -m phone2panda.replay_check --episode episodes_pilot/pilot_ep01.npz --damping 0.3 --out qa/replay_check.mp4
"""
import argparse

import numpy as np

from .common import PUSHER_RADIUS, UNITS_PER_CM, dist_to_t, load_episode, make_env, wrap_angle, write_mp4
from .make_dataset import replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--out", default="qa/replay_check.mp4")
    args = ap.parse_args()
    e = load_episode(args.episode)
    finger, tee = e["finger"], e["tee"]

    env = make_env(damping=args.damping, renderer="native")
    frames_sim, states, _, _ = replay(env, finger, tee[0], render=True)
    frames_real = []
    for f, t in zip(finger, tee):
        s = np.r_[f, t[:2], wrap_angle(t[2]), 0.0, 0.0]
        frames_real.append(env.render_state(s))

    def moved(p):
        return np.linalg.norm(p[-1, :2] - p[0, :2]) / UNITS_PER_CM, np.rad2deg(abs(p[-1, 2] - p[0, 2]))
    rd, ra = moved(tee)
    sd, sa = moved(np.c_[states[:, 2:4], np.unwrap(states[:, 4])])
    print(f"duration {len(finger) / 10:.1f} s, fingertip travelled "
          f"{np.linalg.norm(np.diff(finger, axis=0), axis=1).sum() / UNITS_PER_CM:.1f} cm")
    print(f"real T moved   {rd:5.1f} cm and turned {ra:5.1f} deg")
    print(f"replay T moved {sd:5.1f} cm and turned {sa:5.1f} deg")

    if "tee_iou" in e:
        print(f"T fit IoU {float(e['tee_iou']):.2f} (below ~0.8 suggests the real T and the repo's 8 x 8 cm T differ "
              f"in size, or a corner click is off)")

    speed = np.r_[0, np.linalg.norm(np.diff(tee[:, :2], axis=0), axis=1)] * 10 / UNITS_PER_CM  # cm/s
    pushing = speed > 1.0
    if pushing.sum() >= 3:
        d = np.array([dist_to_t(f, t) for f, t in zip(finger[pushing], tee[pushing])])
        print(f"contact distance while the T moves: median {np.median(d):.1f} units "
              f"({np.median(d) / UNITS_PER_CM:.2f} cm); the sim pusher radius is {PUSHER_RADIUS:.0f} units")
        if np.median(d) > PUSHER_RADIUS + 10:
            print("  -> the tracked point is behind the real contact point: the replayed pusher lags and pushes too late")
        elif np.median(d) < PUSHER_RADIUS - 10:
            print("  -> the tracked point is inside the T: the replayed pusher overlaps it and shoves it")
        else:
            print("  -> contact distance looks right")
    else:
        print("the real T barely moves in this episode: it is not a push, check its frame range in the preview")

    write_mp4([np.hstack([a, b]) for a, b in zip(frames_real, frames_sim)], args.out, fps=10)
    print(f"wrote {args.out}: left = what really happened, right = physics replay")


if __name__ == "__main__":
    main()
