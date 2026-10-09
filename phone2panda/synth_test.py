"""Smoke test for the whole video pipeline, with no camera needed.

It simulates two pushes in the repo's Push-T physics, draws them into a fake phone video (a
tilted camera, the iPad marker grid, a skin-coloured finger with green tape, a blue T, and a
phone bump halfway through), then runs setup_frame + track_video on that video and reports
how far the tracked fingertip and T are from the truth.

Usage:  python -m phone2panda.synth_test --out synth/
Expect: fingertip and T errors of a few mm and about a degree, and 2 episodes found.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .common import ARENA, GOAL_POSE, SQUARE_CM, detect_markers, make_env, save_json, t_polygons, write_mp4
from .setup_frame import hsv_range
from .track_video import segment_episodes, track_frames
from .make_dataset import replay

MM_PER_UNIT = SQUARE_CM * 10 / ARENA
PX_PER_MM = 2.0
CANVAS = (1400, 1000)               # top-down canvas, pixels
ARENA_ORIGIN_MM = np.array([300.0, 80.0])
FPS = 30


def unit_to_canvas(p):
    return (ARENA_ORIGIN_MM + np.asarray(p) * MM_PER_UNIT) * PX_PER_MM


def scripted_paths():
    """Two pushes in arena units at 10 Hz, each with a still start and a still end."""
    def seg(a, b, n):
        return np.linspace(a, b, n, endpoint=False)
    hold = lambda p, n: np.repeat(np.array([p], float), n, 0)
    p1 = np.concatenate([hold([300, 430], 8), seg([300, 430], [300, 300], 25), seg([300, 300], [280, 250], 15),
                         seg([280, 250], [220, 330], 20), hold([220, 330], 12)])
    p2 = np.concatenate([hold([90, 260], 8), seg([90, 260], [210, 270], 25), seg([210, 270], [250, 300], 15),
                         hold([250, 300], 12)])
    return [(p1, np.array([300.0, 180.0, 0.4])), (p2, np.array([260.0, 200.0, 1.9]))]


def draw_canvas(finger, tee, marker_img, rng):
    h, w = CANVAS[1], CANVAS[0]
    c = np.full((h, w, 3), (170, 190, 205), np.uint8)                     # table (BGR)
    c = cv2.add(c, rng.integers(0, 12, c.shape, dtype=np.uint8))
    # tape corners of the square and the target outline
    corners = [unit_to_canvas(p) for p in [(0, 0), (ARENA, 0), (ARENA, ARENA), (0, ARENA)]]
    for q in corners:
        cv2.circle(c, tuple(np.int32(q)), 7, (40, 40, 40), -1)
    for P in t_polygons(GOAL_POSE):
        cv2.polylines(c, [np.int32(unit_to_canvas(P))], True, (60, 60, 60), 2)
    # iPad: bezel + screen with the marker grid, left of the square
    sx0, sy0 = int(15 * PX_PER_MM), int(150 * PX_PER_MM)
    sw, sh = int(227 * PX_PER_MM), int(158 * PX_PER_MM)
    cv2.rectangle(c, (sx0 - 10, sy0 - 10), (sx0 + sw + 10, sy0 + sh + 10), (20, 20, 20), -1)
    c[sy0:sy0 + sh, sx0:sx0 + sw] = cv2.cvtColor(cv2.resize(marker_img, (sw, sh), interpolation=cv2.INTER_AREA),
                                                 cv2.COLOR_GRAY2BGR)
    # T: blue cardboard
    if tee is not None:
        for P in t_polygons(tee):
            cv2.fillPoly(c, [np.int32(unit_to_canvas(P) * 16)], (170, 90, 30), shift=4)
    # finger: skin-coloured capsule from the fingertip toward you (+v), green tape at the tip
    if finger is not None:
        tip = unit_to_canvas(finger)
        base = tip + np.array([25.0, 220.0])
        cv2.line(c, tuple(np.int32(tip)), tuple(np.int32(base)), (130, 160, 220), 34)
        cv2.circle(c, tuple(np.int32(tip)), 17, (130, 160, 220), -1)
        cv2.circle(c, tuple(np.int32(tip)), 11, (60, 200, 60), -1)
    return c


def camera_homographies():
    src = np.float32([[0, 0], [CANVAS[0], 0], [CANVAS[0], CANVAS[1]], [0, CANVAS[1]]])
    dst = np.float32([[150, 40], [1130, 60], [1270, 705], [10, 690]])      # a tilted phone
    H0 = cv2.getPerspectiveTransform(src, dst)
    a = np.deg2rad(0.6)
    bump = np.array([[np.cos(a), -np.sin(a), 7.0], [np.sin(a), np.cos(a), -5.0], [0, 0, 1]])
    return H0, bump @ H0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. ground truth from the repo's physics
    env = make_env(renderer="none")
    gt = []
    for path, tee0 in scripted_paths():
        _, states, _, _ = replay(env, path, tee0, render=False)
        gt.append((states[:, :2], states[:, 2:5]))   # the pusher follows the path; T from physics

    # 2. draw a 30 fps video: still-empty gap, episode 1, gap, episode 2 (phone bumped), gap
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = np.full((1640, 2360), 255, np.uint8)
    m, sep = 420, 105
    for r in range(3):
        for col in range(4):
            x, y = 80 + 58 + col * (m + sep), 80 + 42 + r * (m + sep)
            marker_img[y:y + m, x:x + m] = cv2.aruco.generateImageMarker(dictionary, r * 4 + col, m)
    H0, H1 = camera_homographies()
    frames, truth = [], []
    tee_rest = gt[0][1][0]
    timeline = [("gap", None, 30)]
    timeline += [("ep", 0, None), ("gap", None, 45), ("ep", 1, None), ("gap", None, 30)]
    t_global = 0
    for kind, idx, n in timeline:
        if kind == "gap":
            for _ in range(n):
                H = H0 if len(frames) < 200 else H1
                frames.append(cv2.warpPerspective(draw_canvas(None, tee_rest, marker_img, rng), H, (1280, 720)))
                truth.append((np.nan, np.nan, *tee_rest))
            continue
        fp, tp = gt[idx]
        tt = np.arange(len(fp)) / 10.0
        t30 = np.arange(0, tt[-1], 1 / FPS)
        f30 = np.stack([np.interp(t30, tt, fp[:, d]) for d in range(2)], 1)
        th = np.unwrap(tp[:, 2])
        p30 = np.stack([np.interp(t30, tt, tp[:, 0]), np.interp(t30, tt, tp[:, 1]), np.interp(t30, tt, th)], 1)
        for f, p in zip(f30, p30):
            H = H0 if len(frames) < 200 else H1
            img = cv2.warpPerspective(draw_canvas(f, p, marker_img, rng), H, (1280, 720))
            img = cv2.GaussianBlur(img, (3, 3), 0)
            frames.append(img)
            truth.append((*f, *p))
        tee_rest = p30[-1]
    video = out / "synthetic.mp4"
    write_mp4([f[:, :, ::-1] for f in frames], video, fps=FPS)
    truth = np.array(truth)
    print(f"wrote {video}: {len(frames)} frames, phone bumped at frame 200")

    # 3. setup from frame 10: corners, markers, colours (computed from known positions)
    ref = frames[10]
    proj = lambda pts_units: cv2.perspectiveTransform(unit_to_canvas(np.atleast_2d(pts_units)).reshape(-1, 1, 2)
                                                      .astype(np.float32), H0).reshape(-1, 2)
    corners = proj([(0, 0), (ARENA, 0), (ARENA, ARENA), (0, ARENA)])
    tee_pts = proj(np.array(t_polygons(truth[10, 2:5])).reshape(-1, 2).mean(0, keepdims=True)
                   + np.array([[0, 0], [8, 8], [-8, 8], [0, -6]]))
    first_ep = np.flatnonzero(~np.isnan(truth[:, 0]))[0] + 5
    ref_f = frames[first_ep]
    finger_pts = cv2.perspectiveTransform(unit_to_canvas(truth[first_ep, :2][None]).reshape(-1, 1, 2)
                                          .astype(np.float32), H0).reshape(-1, 2)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                    np.float32([[0, 0], [ARENA, 0], [ARENA, ARENA], [0, ARENA]]))
    setup = {"video": str(video), "frame": 10, "corners_px": corners.tolist(), "H_img_to_arena": H.tolist(),
             "markers_ref": {str(k): v.tolist() for k, v in detect_markers(frames[10]).items()},
             "t_hsv": hsv_range(ref, tee_pts), "finger_hsv": hsv_range(ref_f, finger_pts, patch=2)}
    save_json(setup, out / "setup.json")
    print(f"setup: {len(setup['markers_ref'])} markers in the reference frame")

    # 4. track and compare with the truth
    fps, finger, tee, tee_iou, drift = track_frames(video, setup, preview=out / "track_preview.mp4")
    eps = segment_episodes(finger, fps)
    print(f"found {len(eps)} episodes: {eps}")
    ok = ~np.isnan(finger[:, 0]) & ~np.isnan(truth[:len(finger), 0])
    f_err = np.linalg.norm(finger[ok] - truth[:len(finger)][ok, :2], axis=1) * MM_PER_UNIT
    okt = ~np.isnan(tee[:, 0])
    t_err = np.linalg.norm(tee[okt, :2] - truth[:len(tee)][okt, 2:4], axis=1) * MM_PER_UNIT
    a_err = np.rad2deg(np.abs((tee[okt, 2] - truth[:len(tee)][okt, 4] + np.pi) % (2 * np.pi) - np.pi))
    after_bump = np.arange(len(tee))[okt] >= 200
    print(f"fingertip error: median {np.median(f_err):.1f} mm, 95th pct {np.percentile(f_err, 95):.1f} mm")
    print(f"T position error: median {np.median(t_err):.1f} mm, 95th pct {np.percentile(t_err, 95):.1f} mm "
          f"(after the bump: median {np.median(t_err[after_bump]):.1f} mm)")
    print(f"T angle error: median {np.median(a_err):.2f} deg, 95th pct {np.percentile(a_err, 95):.2f} deg")
    print(f"T fitted in {okt.mean():.0%} of frames, mean IoU {np.mean(tee_iou[okt]):.2f}")
    save_json({"finger_mm_median": float(np.median(f_err)), "tee_mm_median": float(np.median(t_err)),
               "tee_deg_median": float(np.median(a_err)), "episodes": eps}, out / "synth_report.json")


if __name__ == "__main__":
    main()
