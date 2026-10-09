"""Demo videos for the photo tasks: the real start and goal as the phone saw them, next to the
Panda carrying out the plan in the simulator.

For each task:
    [ phone: start | phone: goal  ]
    [ tracked: start | tracked: goal ]  [ Panda executing the plan ]  (+ [ my push ] with --push)
  - phone start / goal: the middle frame of start<k>.mp4 / goal<k>.mp4, as the phone saw it
  - tracked: the same frame warped to the top-down view, with the fitted T outline (what the
    robot is actually given)
  - Panda: the states the simulator actually went through (final_eval_results.npz of the
    planning run), re-rendered at --size with the GOAL pose drawn as the green ghost T
    (in the planning frames the ghost is the fixed Push-T target, which is not this task's goal)
  - "GOAL REACHED at step n" once the T is within 1.3 cm and 20 degrees of the goal (the clip
    stops there, like the planner does); otherwise how far off the T ended up

Writes <out>/photo_<k>.mp4 and a smaller <out>/photo_<k>.gif (GitHub READMEs play GIFs inline).

Usage (after `bash run_scripts/plan_eval.sh photo both`):
  python -m phone2panda.make_demo_videos --run plan_outputs/photo/both --out docs/media
  python -m phone2panda.make_demo_videos --run plan_outputs/photo/both --out docs/media --push
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from .common import UNITS_PER_CM, load_json, write_mp4


def middle_frame(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read {path}")
    return frame[:, :, ::-1]  # RGB


def all_frames(path, step=1):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(f[:, :, ::-1])
        i += 1
    cap.release()
    return out


def fit_height(img, h):
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def label(img, text, sub=None, color=(255, 255, 255)):
    """white text on a dark translucent box in the top-left corner (title, optional subtitle)"""
    img = np.array(img, dtype=np.uint8, copy=True)
    s = max(img.shape[0] / 448, 0.6)
    y = int(8 * s)
    for t, sc in ((text, 0.75), (sub, 0.55)):
        if not t:
            continue
        th = max(1, int(round(2 * s)))
        (tw, tht), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc * s, th)
        x0, y0, x1, y1 = int(8 * s), y, int(8 * s) + tw + int(12 * s), y + tht + base + int(10 * s)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        img[y0:y1, x0:x1] = (0.45 * img[y0:y1, x0:x1]).astype(np.uint8)
        cv2.putText(img, t, (x0 + int(6 * s), y1 - base - int(4 * s)), cv2.FONT_HERSHEY_SIMPLEX, sc * s, color, th,
                    cv2.LINE_AA)
        y = y1 + int(4 * s)
    return img


def trim_still_tail(states, hold=10, eps=0.05):
    """Cut the frames after the pusher and T stopped moving for good, keeping `hold` of them."""
    d = np.abs(np.diff(states[:, :5], axis=0)).max(axis=1)
    moving = np.nonzero(d > eps)[0]
    end = (moving[-1] + 2) if len(moving) else 1
    return states[: min(len(states), end + hold)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="planning output folder, e.g. plan_outputs/photo/both")
    ap.add_argument("--tasks", default="plan_targets/photo.pkl", help="the task file goal_from_photo wrote")
    ap.add_argument("--photo-dir", default="raw/photo", help="used when the task file predates its clip list")
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--size", type=int, default=448, help="panel height in pixels (mp4)")
    ap.add_argument("--gif-size", type=int, default=240, help="panel height in pixels (gif)")
    ap.add_argument("--fps", type=int, default=10, help="the simulator runs at 10 Hz")
    ap.add_argument("--push", action="store_true", help="add my own push (push<k>.mp4) as a fourth panel")
    ap.add_argument("--tasks-only", type=int, nargs="*", help="only these task numbers")
    ap.add_argument("--setup", default="setup.json", help="for the top-down tracked views of the clips")
    args = ap.parse_args()

    res = np.load(Path(args.run) / "final_eval_results.npz")
    with open(args.tasks, "rb") as f:
        tasks = pickle.load(f)
    if not np.allclose(np.asarray(tasks["state_g"])[: len(res["state_g"])], res["state_g"]):
        raise SystemExit(f"{args.tasks} does not hold the tasks of {args.run}")
    clips = tasks.get("clips")
    if clips is None:  # task file from before goal_from_photo stored its clips: pair them by name again
        from .goal_from_photo import clips_from_dir
        c = clips_from_dir(Path(args.photo_dir), "pair")
        clips = list(zip(c[0::2], c[1::2]))
    from .common import import_pusht_wrapper
    import_pusht_wrapper()  # makes the repo's env package importable without its gym registrations
    from env.pusht.panda_renderer import PandaPushTRenderer
    rend = PandaPushTRenderer(size=args.size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = args.size

    setup = load_json(args.setup) if Path(args.setup).exists() else None
    if setup is None:
        print(f"no {args.setup}: leaving out the top-down tracked views")
    h2 = H // 2

    def still(clip, name):
        """(phone frame, top-down tracked view) tiles for one clip, each h2 tall"""
        if clip is None:
            blank = np.full((h2, h2, 3), 235, np.uint8)
            return label(blank, name), label(blank.copy(), "")
        raw = label(fit_height(middle_frame(clip), h2), f"phone: {name}")
        if setup is None:
            return raw, None
        from .goal_from_photo import fit_clip
        _, _, _, rect = fit_clip(clip, setup, 60)
        return raw, label(fit_height(rect[:, :, ::-1], h2), f"tracked: {name}")

    def column(top, bottom, w):
        tiles = [t for t in (top, bottom) if t is not None]
        padded = [np.pad(t, ((0, 0), ((w - t.shape[1]) // 2, w - t.shape[1] - (w - t.shape[1]) // 2), (0, 0)),
                         constant_values=255) for t in tiles]
        col = np.vstack(padded)
        return np.pad(col, ((0, H - col.shape[0]), (0, 0), (0, 0)), constant_values=255)

    def banner(img, text, color):
        """a coloured strip along the bottom of the panel, text scaled to fit"""
        img = img.copy()
        bh = max(28, H // 10)
        img[-bh:] = (0.2 * img[-bh:] + 0.8 * np.array(color)).astype(np.uint8)
        sc, th = bh / 42, max(2, H // 200)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        sc *= min(1.0, (img.shape[1] - 24) / tw)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        cv2.putText(img, text, ((img.shape[1] - tw) // 2, img.shape[0] - int(bh * 0.3)), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, (255, 255, 255), th, cv2.LINE_AA)
        return img

    for k in range(len(res["success"])):
        if args.tasks_only and k not in args.tasks_only:
            continue
        start_clip, goal_clip = clips[k] if k < len(clips) else (None, None)
        goal = res["state_g"][k]
        states = trim_still_tail(res["e_states"][k])
        te = np.linalg.norm(states[:, 2:4] - goal[2:4], axis=1) / UNITS_PER_CM
        ta = np.degrees(np.abs((states[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi))
        hit = np.nonzero((te < 20 / UNITS_PER_CM) & (ta < 20))[0]
        t_hit = int(hit[0]) if len(hit) else None
        if t_hit is not None:  # stop the clip once the T is in place (the planner stops too)
            states, te, ta = states[: t_hit + 1], te[: t_hit + 1], ta[: t_hit + 1]

        rs, os_ = still(start_clip, "start")
        rg, og = still(goal_clip, "goal")
        cw = max(t.shape[1] for t in (rs, os_, rg, og) if t is not None)
        stills = np.hstack([column(rs, os_, cw), column(rg, og, cw)])
        push = []
        if args.push and start_clip:
            pclip = Path(start_clip).with_name(Path(start_clip).name.replace("start", "push", 1))
            if pclip.exists():
                cap = cv2.VideoCapture(str(pclip))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                cap.release()
                push = [fit_height(f, H) for f in all_frames(pclip, step=max(1, round(fps / args.fps)))]
            else:
                print(f"task {k}: no {pclip.name}, leaving the push panel out")

        goal_pose = goal[2:5]
        frames, core = [], []
        n = max(len(states) + args.fps, len(push))  # the plan, then a second on the result
        for t in range(n):
            i = min(t, len(states) - 1)
            s = states[i]
            sim = label(rend.render(s[:2], s[2:5], goal_pose), "Panda (simulation)",
                        f"step {i}   green ghost = goal")
            if t_hit is not None and i == len(states) - 1:
                sim = banner(sim, f"GOAL REACHED  (step {t_hit})", (40, 160, 70))
            elif t_hit is None and t >= len(states) - 1:
                sim = banner(sim, f"goal not reached: T {te[-1]:.1f} cm / {ta[-1]:.0f} deg off", (200, 120, 40))
            row = [stills, sim]
            core.append(np.hstack(row)) if t < len(states) + args.fps else None
            if push:
                row.append(label(push[min(t, len(push) - 1)].copy(), "my push (real)"))
            frames.append(np.hstack(row))

        mp4 = out / f"photo_{k}.mp4"
        w16, h16 = -(-frames[0].shape[1] // 16) * 16, -(-frames[0].shape[0] // 16) * 16  # pad the width for the H.264 encoder
        write_mp4([np.pad(f, ((0, h16 - f.shape[0]), (0, w16 - f.shape[1]), (0, 0)), constant_values=255) for f in frames],
                  mp4, fps=args.fps)
        # the GIF (for the README): stills + Panda, every other frame, small, so it stays a few MB
        import imageio.v2 as imageio
        g = [fit_height(f, args.gif_size) for f in core[::2]]
        imageio.mimsave(out / f"photo_{k}.gif", g, duration=2000 / args.fps, loop=0)
        result = f"goal reached at step {t_hit}" if t_hit is not None else f"not reached (T {te[-1]:.1f} cm, {ta[-1]:.0f} deg off)"
        print(f"task {k}: {Path(start_clip).name if start_clip else '-'} -> {Path(goal_clip).name if goal_clip else '-'}: "
              f"{result} -> {mp4}, {mp4.with_suffix('.gif')}")
    rend.close()


if __name__ == "__main__":
    main()
