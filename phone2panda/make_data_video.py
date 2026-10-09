"""The data-collection clip for the README: one stretch of a recording, three ways at once.

    [ phone video | top-down view with the tracker's fits | the tracked state on the Panda ]

  - phone video: the recording as it is
  - top-down: each frame warped with the iPad markers' homography, the fitted T outline (green)
    and the fingertip (red circle) drawn on, exactly what track_video extracts
  - Panda: the tracked fingertip and T pose placed in the simulator and rendered, frame by frame
    (no physics here: this shows the measurement, the replay comes later)

Usage:
  python -m phone2panda.make_data_video --setup setup.json --video raw/s1_c01.mp4 --start 12 --seconds 8 \\
      --out docs/media/data_collection
Writes <out>.mp4 and a smaller <out>.gif. Pick --start where a push is happening.
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .common import ARENA, GOAL_POSE, WALL_MARGIN, import_pusht_wrapper, load_json, write_mp4
from .make_demo_videos import fit_height, label
from .track_video import track_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="setup.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=15, help="output frame rate")
    ap.add_argument("--size", type=int, default=360, help="panel height (mp4)")
    ap.add_argument("--gif-size", type=int, default=200, help="panel height (gif)")
    ap.add_argument("--out", default="docs/media/data_collection")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    raw = []
    while len(raw) < int(args.seconds * fps):
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f)
    cap.release()
    if not raw:
        raise SystemExit(f"no frames at {args.start}s in {args.video}")

    with tempfile.TemporaryDirectory() as tmp:
        clip, prev = Path(tmp) / "clip.mp4", Path(tmp) / "preview.mp4"
        write_mp4([f[:, :, ::-1] for f in raw], clip, fps=int(round(fps)))
        _, finger, tee, _, _ = track_frames(clip, load_json(args.setup), preview=str(prev))
        cap = cv2.VideoCapture(str(prev))
        top = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            top.append(f[:, :, ::-1])
        cap.release()
    print(f"tracked {len(raw)} frames: fingertip in {np.mean(~np.isnan(finger[:, 0])):.0%}, "
          f"T in {np.mean(~np.isnan(tee[:, 0])):.0%}")

    import_pusht_wrapper()
    from env.pusht.panda_renderer import PandaPushTRenderer
    H = args.size
    rend = PandaPushTRenderer(size=H)
    step = max(1, int(round(fps / args.fps)))
    last_f, last_t = np.array([ARENA / 2, ARENA - WALL_MARGIN]), np.array([ARENA / 2, ARENA / 2, 0.0])
    frames = []
    for i in range(0, min(len(raw), len(top), len(finger)), step):
        if not np.isnan(finger[i, 0]):
            last_f = np.clip(finger[i], WALL_MARGIN, ARENA - WALL_MARGIN)
        if not np.isnan(tee[i, 0]):
            last_t = tee[i]
        sim = rend.render(last_f, last_t, GOAL_POSE)
        frames.append(np.hstack([
            label(fit_height(raw[i][:, :, ::-1], H), "phone video"),
            label(fit_height(top[i], H), "tracked (top-down)"),
            label(sim, "tracked state on the Panda")]))
    rend.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w16, h16 = -(-frames[0].shape[1] // 16) * 16, -(-frames[0].shape[0] // 16) * 16
    write_mp4([np.pad(f, ((0, h16 - f.shape[0]), (0, w16 - f.shape[1]), (0, 0)), constant_values=255) for f in frames],
              out.with_suffix(".mp4"), fps=args.fps)
    import imageio.v2 as imageio
    imageio.mimsave(out.with_suffix(".gif"), [fit_height(f, args.gif_size) for f in frames[::2]],
                    duration=2000 / args.fps, loop=0)
    print(f"wrote {out.with_suffix('.mp4')} and {out.with_suffix('.gif')} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
