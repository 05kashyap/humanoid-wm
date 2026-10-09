"""Track the fingertip tape and the T in a phone video and cut it into demo episodes.

Per frame:
  1. iPad markers -> homography from this frame to the reference frame (undoes phone bumps)
  2. warp to a rectified 512x512 top-down view: one pixel = one Push-T arena unit
  3. fingertip = centroid of the largest fingertip-tape blob
  4. T pose = the repo's T polygon fitted to the T-colour mask (distance-transform loss)
Then episodes are cut where the fingertip is on the table and still at both ends, the
tracks are smoothed, resampled to the env's 10 Hz control rate and saved, one .npz each.
An episode with a tracking gap over 0.5 s (hand covering the T) or with the fingertip at the
arena edge is split there instead of dropped; pieces of 3 s or more are saved as _epNNa, _epNNb.
Rerunning on a clip replaces that clip's episodes in --out.

Usage:
  python -m phone2panda.track_video --setup setup.json --video raw/clip01.mp4 --out episodes/ \
      --preview qa/clip01_track.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from .common import (ARENA, CONTROL_HZ, PUSHER_RADIUS, T_BAR, T_CENTROID_LOCAL, T_STEM, UNITS_PER_CM,
                     WALL_MARGIN, detect_markers, dist_to_t, hsv_mask, load_json, rot, t_mask, t_polygons,
                     write_mp4)

T_AREA = 120 * 30 + 30 * 90  # arena units^2


class TFitter:
    """Fits (x, y, theta) so the repo's T polygon lies inside the observed T mask."""

    def __init__(self, step=5.0):
        pts = []
        for P in (T_BAR, T_STEM):
            xs = np.arange(P[:, 0].min() + step / 2, P[:, 0].max(), step)
            ys = np.arange(P[:, 1].min() + step / 2, P[:, 1].max(), step)
            pts.append(np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2))
        self.local = np.concatenate(pts)

    @staticmethod
    def _sample(dt, pts):
        x, y = pts[:, 0], pts[:, 1]
        inside = (x >= 0) & (y >= 0) & (x < dt.shape[1] - 1) & (y < dt.shape[0] - 1)
        out = np.full(len(pts), 10.0)
        xi, yi = x[inside], y[inside]
        x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
        ax, ay = xi - x0, yi - y0
        v = (dt[y0, x0] * (1 - ax) * (1 - ay) + dt[y0, x0 + 1] * ax * (1 - ay)
             + dt[y0 + 1, x0] * (1 - ax) * ay + dt[y0 + 1, x0 + 1] * ax * ay)
        out[inside] = v
        return out

    def loss(self, pose, dt):
        pts = self.local @ rot(pose[2]).T + pose[:2]
        return np.minimum(self._sample(dt, pts), 10.0).mean()

    def _refine(self, x0, dt):
        simplex = np.array([x0, x0 + [3, 0, 0], x0 + [0, 3, 0], x0 + [0, 0, 0.06]])
        r = minimize(self.loss, x0, args=(dt,), method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, xatol=0.05, fatol=1e-4, maxiter=400))
        return r.x, r.fun

    def fit(self, mask, init=None):
        dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
        if init is not None:
            pose, l = self._refine(np.asarray(init, float), dt)
            if l < 1.0:
                return pose, l
        ys, xs = np.nonzero(mask)
        c = np.array([xs.mean(), ys.mean()])
        cands = []
        for th in np.deg2rad(np.arange(0, 360, 10)):
            p = c - rot(th) @ T_CENTROID_LOCAL
            x0 = np.array([p[0], p[1], th])
            cands.append((self.loss(x0, dt), x0))
        cands.sort(key=lambda z: z[0])
        best = min((self._refine(x0, dt) for _, x0 in cands[:3]), key=lambda z: z[1])
        return best


def largest_blob(mask, min_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, None
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None, None
    return cent[i], (lab == i).astype(np.uint8)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def track_frames(video, setup, preview=None, max_frames=None):
    H_ref = np.array(setup["H_img_to_arena"])
    ref_markers = {int(k): np.array(v) for k, v in setup["markers_ref"].items()}
    corners_ref = np.array(setup["corners_px"], np.float32).reshape(-1, 1, 2)
    kernel = np.ones((3, 3), np.uint8)
    fitter = TFitter()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finger, tee, tee_iou, drift = [], [], [], []
    G = np.eye(3)
    prev_pose = None
    preview_frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        i += 1
        # 1. phone-bump correction from the iPad markers
        cur = detect_markers(frame)
        common_ids = [k for k in cur if k in ref_markers]
        if len(common_ids) >= 4:  # 4+ markers: a single partly hidden marker can't skew the fit
            src = np.concatenate([cur[k] for k in common_ids])
            dst = np.concatenate([ref_markers[k] for k in common_ids])
            G_new, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if G_new is not None:
                jump = np.abs(cv2.perspectiveTransform(corners_ref, G_new) - cv2.perspectiveTransform(corners_ref, G)).max()
                if jump < 15:  # pixels; a real bump is gradual or a one-off, a bad detection is a spike
                    G = G_new
        # drift = how far (pixels) the phone has moved the tape corners since the reference frame
        drift.append(float(np.abs(cv2.perspectiveTransform(corners_ref, G) - corners_ref).max()))
        # 2. rectify: one pixel = one arena unit
        rect = cv2.warpPerspective(frame, H_ref @ G, (ARENA, ARENA))
        # 3. fingertip tape
        fmask = cv2.morphologyEx(hsv_mask(rect, setup["finger_hsv"]), cv2.MORPH_OPEN, kernel)
        fc, fblob = largest_blob(fmask, min_area=15)
        finger.append(fc if fc is not None else [np.nan, np.nan])
        # 4. T, with the fingertip blob removed
        tmask = hsv_mask(rect, setup["t_hsv"])
        if fblob is not None:
            tmask[cv2.dilate(fblob, np.ones((9, 9), np.uint8)) > 0] = 0
        tmask = cv2.morphologyEx(cv2.morphologyEx(tmask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        _, tblob = largest_blob(tmask, min_area=0.25 * T_AREA)
        if tblob is None:
            tee.append([np.nan] * 3); tee_iou.append(0.0); prev_pose = None
        else:
            pose, _ = fitter.fit(tblob, init=prev_pose)
            q = iou(t_mask(pose), tblob)
            tee.append(pose if q > 0.6 else [np.nan] * 3)
            tee_iou.append(float(q))
            prev_pose = pose if q > 0.6 else None
        if preview is not None:
            vis = rect.copy()
            if not np.isnan(tee[-1][0]):
                for P in t_polygons(tee[-1]):
                    cv2.polylines(vis, [np.round(P).astype(np.int32)], True, (0, 255, 0), 2)
            if fc is not None:
                cv2.circle(vis, (int(fc[0]), int(fc[1])), 15, (0, 0, 255), 2)
            preview_frames.append(vis[:, :, ::-1])
    cap.release()
    if preview is not None and preview_frames:
        write_mp4(preview_frames, preview, fps=int(round(fps)))
    return fps, np.array(finger, float), np.array(tee, float), np.array(tee_iou), np.array(drift)


def fill_gaps(x, max_gap):
    """Linearly interpolate NaN runs no longer than max_gap samples (per column)."""
    x = x.copy()
    for d in range(x.shape[1]):
        col = x[:, d]
        bad = np.isnan(col)
        if bad.all() or not bad.any():
            continue
        idx = np.arange(len(col))
        run_start = None
        for j in range(len(col) + 1):
            b = j < len(col) and bad[j]
            if b and run_start is None:
                run_start = j
            elif not b and run_start is not None:
                if j - run_start <= max_gap and run_start > 0 and j < len(col):
                    col[run_start:j] = np.interp(idx[run_start:j], [run_start - 1, j], [col[run_start - 1], col[j]])
                run_start = None
        x[:, d] = col
    return x


def segment_episodes(finger, fps, min_len_s=3.0, still_speed_cm=2.0, still_s=0.4, gap_s=0.5):
    """Episodes = runs where the fingertip is visible, trimmed to the first and last still moments.

    Trimming to still moments drops the hand entering and leaving the view in the air, where
    projecting onto the table plane would put the fingertip in the wrong place.
    """
    f = fill_gaps(finger, int(gap_s * fps))
    visible = ~np.isnan(f[:, 0])
    speed = np.full(len(f), np.inf)
    v = np.linalg.norm(np.gradient(f, axis=0), axis=1) * fps / UNITS_PER_CM  # cm/s
    speed[visible] = v[visible]
    still = speed < still_speed_cm
    w = max(1, int(still_s * fps))
    still_win = np.convolve(still.astype(float), np.ones(w), "same") >= w - 0.5
    episodes = []
    j = 0
    while j < len(f):
        if not visible[j]:
            j += 1
            continue
        k = j
        while k < len(f) and visible[k]:
            k += 1
        run = np.nonzero(still_win[j:k])[0]
        if len(run):
            s, e = j + run[0], j + run[-1]
            if (e - s) / fps >= min_len_s:
                episodes.append((int(s), int(e)))
        j = k
    return episodes


def _runs(mask):
    """(start, end_exclusive) of each run of True values."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def split_at_gaps(finger, tee, fps, max_gap_s=0.5, min_len_s=3.0, edge_s=0.1):
    """Cut one episode where it cannot be used and keep the usable pieces (skip recovery).

    A cut goes wherever the T or the fingertip is missing for longer than max_gap_s (usually the
    hand covering the T), or the fingertip comes within WALL_MARGIN of the tape square (the sim
    pusher would hit the wall). Next to a cut, each piece loses edge_s (half-covered fits), only
    starts once the fingertip is clear of the T (so the sim does not begin with the pusher inside
    it), and is kept if it still lasts min_len_s.
    Returns ([(a, b)], was_cut) in episode-relative frames, b inclusive.
    """
    n = len(finger)
    max_gap, edge = int(max_gap_s * fps), max(1, int(round(edge_s * fps)))
    bad = np.zeros(n, bool)
    for x in (finger, tee):
        for j, k in _runs(np.isnan(x[:, 0])):
            if k - j > max_gap:
                bad[j:k] = True
    with np.errstate(invalid="ignore"):
        bad |= ((finger < WALL_MARGIN) | (finger > ARENA - WALL_MARGIN)).any(1)
    if not bad.any():
        return [(0, n - 1)], False

    def ok(i):
        return not (np.isnan(finger[i, 0]) or np.isnan(tee[i, 0]))

    pieces = []
    for j, k in _runs(~bad):
        a, b = j + (edge if j > 0 else 0), k - 1 - (edge if k < n else 0)
        while a <= b and not ok(a):
            a += 1
        while b >= a and not ok(b):
            b -= 1
        if j > 0:
            while a <= b and (not ok(a) or dist_to_t(finger[a], tee[a]) <= PUSHER_RADIUS + 2):
                a += 1
        if b - a >= min_len_s * fps:
            pieces.append((a, b))
    return pieces, True


def resample(t_src, x, t_dst):
    return np.stack([np.interp(t_dst, t_src, x[:, d]) for d in range(x.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="folder for episode .npz files")
    ap.add_argument("--preview", help="optional mp4 of the rectified view with the fits drawn on")
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    setup = load_json(args.setup)
    fps, finger, tee, tee_iou, drift = track_frames(args.video, setup, args.preview, args.max_frames)
    print(f"{args.video}: {len(finger)} frames at {fps:.1f} fps, fingertip seen in "
          f"{np.mean(~np.isnan(finger[:, 0])):.0%}, T fitted in {np.mean(~np.isnan(tee[:, 0])):.0%}, "
          f"max phone drift {drift.max():.1f} px")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    stale = sorted(out.glob(f"{stem}_ep*.npz"))
    for p in stale:  # a rerun replaces this clip's episodes (names change when demos get split)
        p.unlink()
    if stale:
        print(f"  removed {len(stale)} old episodes of {stem} from {out}")
    n_saved, n_recovered = 0, 0
    for e_idx, (s0, e0) in enumerate(segment_episodes(finger, fps)):
        pieces, was_cut = split_at_gaps(finger[s0:e0 + 1], tee[s0:e0 + 1], fps)
        if was_cut:
            print(f"  episode {e_idx} (frames {s0}-{e0}): tracking gap over 0.5 s or fingertip at the arena "
                  f"edge -> split, kept {len(pieces)} piece(s) of 3 s or more")
        for p_idx, (a, b) in enumerate(pieces):
            s, e = s0 + a, s0 + b
            name = f"{e_idx:02d}" + (chr(ord("a") + p_idx) if was_cut else "")
            sl = slice(s, e + 1)
            f_hf = fill_gaps(finger[sl], int(0.5 * fps))
            # interpolate the angle through (cos, sin) so gaps and wrap-around are handled together
            t4 = fill_gaps(np.c_[tee[sl, :2], np.cos(tee[sl, 2]), np.sin(tee[sl, 2])], int(0.5 * fps))
            frac_t = float(np.mean(~np.isnan(tee[sl, 0])))
            if np.isnan(f_hf).any() or np.isnan(t4).any() or frac_t < 0.7:
                print(f"  skip episode {name} (frames {s}-{e}): T fitted in only {frac_t:.0%} of frames")
                continue
            win = max(5, int(0.2 * fps) | 1)
            f_hf = savgol_filter(f_hf, win, 2, axis=0)
            t4 = savgol_filter(t4, win, 2, axis=0)
            t_hf = np.c_[t4[:, :2], np.unwrap(np.arctan2(t4[:, 3], t4[:, 2]))]
            t_src = np.arange(len(f_hf)) / fps
            t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / CONTROL_HZ)
            f10 = resample(t_src, f_hf, t_dst)
            t10 = resample(t_src, t_hf, t_dst)
            if (f10 < WALL_MARGIN).any() or (f10 > ARENA - WALL_MARGIN).any():
                print(f"  skip episode {name}: fingertip leaves the arena")
                continue
            np.savez(out / f"{stem}_ep{name}.npz", finger=f10, tee=t10, t=t_dst,
                     finger_hf=f_hf, tee_hf=t_hf, t_hf=t_src, fps=fps,
                     frames=np.array([s, e]), video=str(args.video),
                     tee_iou=float(np.nanmean(tee_iou[sl])), tee_fitted=frac_t)
            n_saved += 1
            n_recovered += was_cut
            print(f"  episode {name}: frames {s}-{e} ({(e - s) / fps:.1f} s), {len(f10)} steps, "
                  f"T fit IoU {np.nanmean(tee_iou[sl]):.2f}")
    print(f"saved {n_saved} episodes to {out} ({n_recovered} of them pieces of split demos)")


if __name__ == "__main__":
    main()
