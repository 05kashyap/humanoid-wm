"""One-off setup from a single reference frame: arena corners, iPad markers and colour ranges.

You click, in this order:
  1. the four tape corners of the 34 cm square: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT
     (far = away from you; these become arena (0,0), (512,0), (512,512), (0,512))
  2. 4-6 points on the T                     (press any key when done)
  3. 2-4 points on the fingertip tape        (press any key when done)

It writes setup.json with the image->arena homography, the iPad marker corners seen in this
frame (used to undo phone bumps later) and HSV ranges for the T and the fingertip tape.

Usage:
  python -m phone2panda.setup_frame --video raw/session1_clip01.mp4 --frame 30 --out setup.json
Headless (no window), pass the pixel coordinates yourself:
  python -m phone2panda.setup_frame --video ... --corners "x,y x,y x,y x,y" --t-points "x,y ..." --finger-points "x,y ..."
"""
import argparse

import cv2
import numpy as np

from .common import ARENA, detect_markers, save_json


def read_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {video}")
    return frame


def parse_points(s):
    return [tuple(float(v) for v in p.split(",")) for p in s.split()]


def click_points(frame, title, n=None):
    """Collect clicks in an OpenCV window; finish with any key (or automatically after n clicks)."""
    scale = min(1.0, 1400 / frame.shape[1], 800 / frame.shape[0])
    shown = cv2.resize(frame, None, fx=scale, fy=scale)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x / scale, y / scale))
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(shown, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(title, shown)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)                  # resizable; image scales to the window
    cv2.resizeWindow(title, shown.shape[1], shown.shape[0])
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, shown)
    while True:
        key = cv2.waitKey(30)
        if (n is not None and len(pts) >= n) or (n is None and key != -1 and pts):
            break
    cv2.destroyWindow(title)
    return pts


def hsv_range(frame, points, patch=4):
    """HSV range covering small patches around the clicked points, with a margin."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
    samples = []
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        samples.append(hsv[max(0, y - patch):y + patch + 1, max(0, x - patch):x + patch + 1].reshape(-1, 3))
    s = np.concatenate(samples)
    # circular hue statistics (OpenCV hue is 0..179)
    ang = s[:, 0] / 180 * 2 * np.pi
    mean_h = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * 180
    dh = np.abs(((s[:, 0] - mean_h + 90) % 180) - 90)
    half = max(8.0, np.percentile(dh, 98) + 4)
    lo_h, hi_h = (mean_h - half) % 180, (mean_h + half) % 180
    lo_s = max(40, np.percentile(s[:, 1], 2) - 40)
    lo_v = max(40, np.percentile(s[:, 2], 2) - 50)
    return {"lo": [int(lo_h), int(lo_s), int(lo_v)], "hi": [int(np.ceil(hi_h)), 255, 255],
            "mean_hsv": [round(mean_h, 1), round(float(s[:, 1].mean()), 1), round(float(s[:, 2].mean()), 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--out", default="setup.json")
    ap.add_argument("--corners", help="'x,y x,y x,y x,y' far-left, far-right, near-right, near-left")
    ap.add_argument("--t-points")
    ap.add_argument("--finger-points")
    args = ap.parse_args()

    frame = read_frame(args.video, args.frame)
    corners = parse_points(args.corners) if args.corners else click_points(
        frame, "Click tape corners: far-left, far-right, near-right, near-left", n=4)
    t_pts = parse_points(args.t_points) if args.t_points else click_points(
        frame, "Click 4-6 points on the T, then press a key")
    f_pts = parse_points(args.finger_points) if args.finger_points else click_points(
        frame, "Click 2-4 points on the fingertip tape, then press a key")

    src = np.array(corners, np.float32)
    dst = np.array([[0, 0], [ARENA, 0], [ARENA, ARENA], [0, ARENA]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    markers = detect_markers(frame)
    setup = {
        "video": str(args.video), "frame": args.frame, "image_size": [frame.shape[1], frame.shape[0]],
        "corners_px": src.tolist(), "H_img_to_arena": H.tolist(),
        "markers_ref": {str(k): v.tolist() for k, v in markers.items()},
        "t_hsv": hsv_range(frame, t_pts), "finger_hsv": hsv_range(frame, f_pts),
    }
    save_json(setup, args.out)

    rect = cv2.warpPerspective(frame, H, (ARENA, ARENA))
    cv2.imwrite(args.out.rsplit(".", 1)[0] + "_rectified.png", rect)
    print(f"wrote {args.out}: {len(markers)} iPad markers seen, "
          f"T hsv {setup['t_hsv']['lo']}..{setup['t_hsv']['hi']}, "
          f"finger hsv {setup['finger_hsv']['lo']}..{setup['finger_hsv']['hi']}")
    if len(markers) < 4:
        print("WARNING: fewer than 4 iPad markers detected; phone bumps cannot be corrected. Check glare.")


if __name__ == "__main__":
    main()
