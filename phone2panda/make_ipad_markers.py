"""Make the full-screen ArUco grid the iPad displays beside the workspace.

The markers are used to detect and undo small phone movements between the reference frame
(where you clicked the tape corners) and every later frame. Their physical size is only a
sanity check, so measure one displayed marker with a ruler and compare with the printout.

Usage:
  python -m phone2panda.make_ipad_markers --width 2360 --height 1640 --ppi 264 --out ipad_grid.png
  (Settings > General > About shows your iPad model; look up its native resolution.)
"""
import argparse

import cv2
import numpy as np

from .common import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2360, help="native screen width in pixels (landscape)")
    ap.add_argument("--height", type=int, default=1640, help="native screen height in pixels")
    ap.add_argument("--ppi", type=float, default=264.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--margin", type=int, default=80, help="white border in pixels")
    ap.add_argument("--out", default="ipad_grid.png")
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sep_ratio = 0.25
    avail_w = args.width - 2 * args.margin
    avail_h = args.height - 2 * args.margin
    m = int(min(avail_w / (args.cols + (args.cols - 1) * sep_ratio),
                avail_h / (args.rows + (args.rows - 1) * sep_ratio)))
    sep = int(m * sep_ratio)
    grid_w = args.cols * m + (args.cols - 1) * sep
    grid_h = args.rows * m + (args.rows - 1) * sep
    x0 = (args.width - grid_w) // 2
    y0 = (args.height - grid_h) // 2

    img = np.full((args.height, args.width), 255, np.uint8)
    layout = {}
    for r in range(args.rows):
        for c in range(args.cols):
            mid = r * args.cols + c
            marker = cv2.aruco.generateImageMarker(dictionary, mid, m)
            x, y = x0 + c * (m + sep), y0 + r * (m + sep)
            img[y:y + m, x:x + m] = marker
            layout[mid] = [[x, y], [x + m, y], [x + m, y + m], [x, y + m]]

    cv2.imwrite(args.out, img)
    mm_per_px = 25.4 / args.ppi
    meta = {"dictionary": "DICT_4X4_50", "marker_px": m, "marker_mm_expected": round(m * mm_per_px, 1),
            "screen_px": [args.width, args.height], "corners_px": layout}
    save_json(meta, args.out.rsplit(".", 1)[0] + ".json")
    print(f"wrote {args.out}: {args.cols}x{args.rows} markers of {m}px "
          f"(expected {m * mm_per_px:.1f} mm at {args.ppi:.0f} ppi; check with a ruler)")


if __name__ == "__main__":
    main()
