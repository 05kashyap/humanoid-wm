"""Planning problems from short clips of the real T (the photo stretch demo).

Film 2-3 s of the T lying still in some pose, phone in the recording position, hand out of
frame (a video clip, not a photo: the tracker's homography is calibrated on the video frames).
Each clip is rectified with setup.json and the T pose is fitted as in track_video, using the
median over the clip's frames.

  --as goal   each clip is a GOAL; it is paired with --starts start states taken from the
              validation demos (my first tracked state of each)
  --as start  each clip is a START; the goal is the standard Push-T target (256, 256, 45 deg)
  --as pair   clips are taken two at a time: (start, goal)

The pusher is not in the clips. It starts just behind the T, on the side away from the goal
(where a person would put a finger), and the goal image shows it carried along with the T, as
it would be after the push. The planner's objective matches the whole goal image, pusher
included, so a pusher parked in a far corner pulls the plan toward that corner instead of
the T. Score these with T-only success (helpers/plan_report.py, column "T only").

Writes the goal file plan.py reads (same format as make_fulltask_targets) and a preview image:
one row per problem: real start, real goal (rectified, with the fitted T outline), Panda start,
Panda goal. A side with no clip (validation starts, the standard target) is a labelled blank.

Usage (--dir finds the clips by name: start<k>.mp4 / goal<k>.mp4, paired by <k>):
  python -m phone2panda.goal_from_photo --setup setup.json --dir raw/photo --as pair
  python -m phone2panda.goal_from_photo --setup setup.json --dir raw/photo --as goal \\
      --episodes episodes/ --split split.json
  (or list the clips yourself with --clips a.mp4 b.mp4 ...; for --as pair in start, goal order)
  bash run_scripts/plan_eval.sh photo both
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from .common import (ARENA, GOAL_POSE, PUSHER_RADIUS, T_CENTROID_LOCAL, WALL_MARGIN, conf_damping,
                     dist_to_t, load_episode, load_json, make_env, rot, t_polygons, wrap_angle)
from .track_video import track_frames

CORNERS = np.array([[60.0, 60.0], [452.0, 60.0], [60.0, 452.0], [452.0, 452.0]])


def free_corner(tee_pose):
    return CORNERS[np.argmax(np.linalg.norm(CORNERS - tee_pose[:2], axis=1))]


def centroid(pose):
    return pose[:2] + rot(pose[2]) @ T_CENTROID_LOCAL


def behind(start_pose, goal_pose, gap=8.0):
    """A pusher start just behind the T, on the side away from the goal: where a person
    would put their finger to push it there. Falls back to the stem's end for pure turns."""
    c = centroid(start_pose)
    d = centroid(goal_pose) - c
    if np.linalg.norm(d) < 10:
        d = rot(start_pose[2]) @ np.array([0.0, 1.0])  # along the stem, push from its end
        d = -d
    d = d / np.linalg.norm(d)
    # straight behind first; if the arena wall is in the way, swing round to either side
    for turn in np.radians([0, 30, -30, 60, -60, 90, -90, 120, -120, 180]):
        for r in np.arange(20.0, 250.0, 2.0):
            p = np.clip(c - r * (rot(turn) @ d), WALL_MARGIN, ARENA - WALL_MARGIN)
            if dist_to_t(p, start_pose) >= PUSHER_RADIUS + gap:
                return p
    return free_corner(start_pose)


def carry(p, start_pose, goal_pose):
    """The pusher point moved rigidly with the T from its start pose to its goal pose, so the
    goal image shows the pusher where it would be after pushing the T there."""
    local = rot(start_pose[2]).T @ (np.asarray(p, float) - start_pose[:2])
    return np.clip(goal_pose[:2] + rot(goal_pose[2]) @ local, WALL_MARGIN, ARENA - WALL_MARGIN)


def clips_from_dir(d, role):
    """start<k>/goal<k> clips in d, sorted by k; for pairs, only k with both (in start, goal order)."""
    import re

    def find(prefix):
        out = {}
        for f in d.iterdir():
            m = re.fullmatch(prefix + r"(\w*?)\.(mp4|mov|MOV|MP4)", f.name)
            if m:
                out[m.group(1)] = f
        return out

    key = lambda k: (int(k) if k.isdigit() else float("inf"), k)
    starts, goals = find("start"), find("goal")
    if role == "pair":
        ks = sorted(set(starts) & set(goals), key=key)
        missing = sorted(set(starts) ^ set(goals), key=key)
        if missing:
            print(f"no partner clip for k = {', '.join(missing)}: skipped")
        if not ks:
            raise SystemExit(f"no start<k>/goal<k> pairs in {d}")
        print("pairs: " + ", ".join(f"start{k} -> goal{k}" for k in ks))
        return [str(c) for k in ks for c in (starts[k], goals[k])]
    pick = goals if role == "goal" else starts
    if not pick:
        raise SystemExit(f"no {role}<k>.mp4 clips in {d}")
    return [str(pick[k]) for k in sorted(pick, key=key)]


def fit_clip(clip, setup, max_frames):
    """Median T pose over the clip, the fraction of frames fitted, and one rectified frame."""
    _, _, tee, tee_iou, drift = track_frames(clip, setup, max_frames=max_frames)
    ok = ~np.isnan(tee[:, 0])
    if not ok.any():
        raise SystemExit(f"{clip}: the T was not found in any frame (check t_hsv in setup.json)")
    th = np.arctan2(np.median(np.sin(tee[ok, 2])), np.median(np.cos(tee[ok, 2])))
    pose = np.array([np.median(tee[ok, 0]), np.median(tee[ok, 1]), wrap_angle(th)])
    # one rectified frame for the preview (the middle one, without the bump correction)
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(n, max_frames or n) // 2)
    _, frame = cap.read()
    cap.release()
    rect = cv2.warpPerspective(frame, np.array(setup["H_img_to_arena"]), (ARENA, ARENA))
    for P in t_polygons(pose):
        cv2.polylines(rect, [np.round(P).astype(np.int32)], True, (0, 255, 0), 2)
    return pose, float(ok.mean()), float(np.nanmean(np.where(ok, tee_iou, np.nan))), rect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--clips", nargs="+")
    ap.add_argument("--dir", help="folder with start<k>.mp4 / goal<k>.mp4 (other files, e.g. push<k>, are ignored)")
    ap.add_argument("--as", dest="role", choices=["goal", "start", "pair"], default="goal")
    ap.add_argument("--episodes", help="validation starts for --as goal")
    ap.add_argument("--split", help="validation starts for --as goal")
    ap.add_argument("--starts", type=int, default=5, help="validation starts per goal clip (--as goal)")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--pusher", choices=["behind", "corner"], default="behind",
                    help="behind: start just behind the T (away from the goal), goal pusher carried "
                         "rigidly with the T; corner: farthest arena corner (the old default)")
    ap.add_argument("--damping", type=float, default=None, help="default: conf/env/pusht_human.yaml")
    ap.add_argument("--goal-H", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plan_targets/photo.pkl")
    ap.add_argument("--preview", default="qa/photo_tasks.png")
    args = ap.parse_args()
    if args.dir:
        if args.clips:
            ap.error("give --dir or --clips, not both")
        args.clips = clips_from_dir(Path(args.dir), args.role)
    elif not args.clips:
        ap.error("give --dir or --clips")
    if args.role == "goal" and not (args.episodes and args.split):
        ap.error("--as goal needs --episodes and --split for the start states")
    if args.role == "pair" and len(args.clips) % 2:
        ap.error("--as pair needs an even number of clips: start1 goal1 start2 goal2 ...")
    damping = conf_damping() if args.damping is None else args.damping

    setup = load_json(args.setup)
    fits = []
    for c in args.clips:
        pose, frac, q, rect = fit_clip(c, setup, args.max_frames)
        flag = "" if q > 0.8 else "   <-- low IoU: check the fit in the preview"
        print(f"{Path(c).name}: T at ({pose[0]:.0f}, {pose[1]:.0f}) {np.degrees(pose[2]):.0f} deg, "
              f"fitted in {frac:.0%} of frames, IoU {q:.2f}{flag}")
        fits.append((pose, rect))

    def state(tee_pose, pusher):
        return np.r_[pusher, tee_pose[:2], wrap_angle(tee_pose[2]), 0.0, 0.0]

    def pair(p0, pg):
        """start and goal states; the pusher starts behind the T and ends carried along with it
        (--pusher corner: both in the arena corner farthest from the T)"""
        p0, pg = np.asarray(p0, float), np.asarray(pg, float)
        if args.pusher == "corner":
            return state(p0, free_corner(p0)), state(pg, free_corner(pg))
        ps = behind(p0, pg)
        return state(p0, ps), state(pg, carry(ps, p0, pg))

    problems = []  # (start state, goal state, real start frame or None, real goal frame or None)
    if args.role == "goal":
        rng = np.random.default_rng(args.seed)
        names = load_json(args.split)["val"]
        for pose, rect in fits:
            for n in rng.choice(names, size=min(args.starts, len(names)), replace=False):
                e = load_episode(Path(args.episodes) / f"{n}.npz")
                problems.append((*pair(e["tee"][0], pose), None, rect))
    elif args.role == "start":
        for pose, rect in fits:
            problems.append((*pair(pose, GOAL_POSE), rect, None))
    else:
        for (p0, r0), (pg, rg) in zip(fits[0::2], fits[1::2]):
            problems.append((*pair(p0, pg), r0, rg))

    # the clips behind each problem (start clip, goal clip; None where there is none), kept in
    # the task file for phone2panda.make_demo_videos
    clips = [str(Path(c).resolve()) for c in args.clips]
    if args.role == "goal":
        task_clips = [(None, c) for c in clips for _ in range(min(args.starts, len(load_json(args.split)["val"])))]
    elif args.role == "start":
        task_clips = [(c, None) for c in clips]
    else:
        task_clips = list(zip(clips[0::2], clips[1::2]))

    env = make_env(damping=damping, renderer="panda")
    obs0, obsg, s0, sg, rows = {"visual": [], "proprio": []}, {"visual": [], "proprio": []}, [], [], []
    def tile(img, text):
        img = np.ascontiguousarray(img if img is not None else np.full((ARENA, ARENA, 3), 235, np.uint8))
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        return img

    no_start = "start: validation demo" if args.role == "goal" else "no clip"
    no_goal = "goal: standard target" if args.role == "start" else "no clip"
    for i, (a, b, r_start, r_goal) in enumerate(problems):
        pics = []
        for st, o, s in ((a, obs0, s0), (b, obsg, sg)):
            ob, _ = env.prepare(0, st)
            o["visual"].append(ob["visual"][None]); o["proprio"].append(ob["proprio"][None]); s.append(st)
            pics.append(cv2.resize(np.asarray(ob["visual"], np.uint8), (ARENA, ARENA))[:, :, ::-1])
        rows.append(np.hstack([
            tile(None if r_start is None else r_start.copy(), f"{i}: real start" if r_start is not None else f"{i}: {no_start}"),
            tile(None if r_goal is None else r_goal.copy(), "real goal" if r_goal is not None else no_goal),
            tile(pics[0], "Panda start"), tile(pics[1], "Panda goal")]))

    data = {"obs_0": {k: np.stack(v) for k, v in obs0.items()}, "obs_g": {k: np.stack(v) for k, v in obsg.items()},
            "state_0": np.stack(s0), "state_g": np.stack(sg), "gt_actions": None, "goal_H": args.goal_H,
            "clips": task_clips, "role": args.role}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.preview, np.vstack(rows))
    print(f"wrote {args.out}: {len(problems)} problems ({args.role} from {len(args.clips)} clips), damping {damping}")
    print(f"wrote {args.preview}: per row real start, real goal, Panda start, Panda goal "
          f"(row number = task number in plan_outputs/photo/<model>/output_final_<task>_*.mp4)")


if __name__ == "__main__":
    main()
