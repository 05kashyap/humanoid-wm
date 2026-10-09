"""Shared constants and helpers for the phone-to-Panda pipeline.

Coordinate convention (the repo's Push-T arena, checked in env/pusht/pusht_env.py):
  * arena units 0..512, u to the right, v DOWN (pygame image convention)
  * T pose (x, y, theta): (x, y) is the midpoint of the bar's OUTER edge, the stem points
    along local +y, and world = R(theta) @ local + (x, y)
  * the pusher is a kinematic circle of radius 15 units
"""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

ARENA = 512                      # Push-T arena side, sim units
SQUARE_CM = 34.0                 # side of the taped square on the table
UNITS_PER_CM = ARENA / SQUARE_CM # 15.06 units per cm
CONTROL_HZ = 10                  # repo env control rate (10 Hz, 100 Hz physics)
PUSHER_RADIUS = 15.0             # repo pusher radius, units (2 cm diameter at this scale)
WALL_MARGIN = 20.0               # keep pusher paths this far inside the arena
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])  # repo's fixed goal zone

# Repo T geometry (add_tee, scale=30), local frame.
T_BAR = np.array([[-60, 30], [60, 30], [60, 0], [-60, 0]], float)
T_STEM = np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], float)
_A_BAR, _A_STEM = 120 * 30, 30 * 90
T_CENTROID_LOCAL = np.array([0.0, (_A_BAR * 15 + _A_STEM * 75) / (_A_BAR + _A_STEM)])  # area centroid


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def t_polygons(pose):
    """The T's two rectangles in arena units for pose (x, y, theta)."""
    R = rot(pose[2])
    return [P @ R.T + np.asarray(pose[:2]) for P in (T_BAR, T_STEM)]


def t_mask(pose, size=ARENA, scale=1.0):
    """Binary mask (size x size) of the T at `pose`, drawn at `scale` pixels per unit."""
    m = np.zeros((size, size), np.uint8)
    for P in t_polygons(pose):
        cv2.fillPoly(m, [np.round(P * scale * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m


def dist_to_t(point, pose):
    """Signed distance (units) from a point to the T outline; negative means inside."""
    d = []
    for P in t_polygons(pose):
        d.append(-cv2.pointPolygonTest(P.astype(np.float32), (float(point[0]), float(point[1])), True))
    return min(d)


_ARUCO = None


def detect_markers(image):
    """Detect the iPad's ArUco markers. Returns {id: (4, 2) corner array} in image pixels."""
    global _ARUCO
    if _ARUCO is None:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _ARUCO = cv2.aruco.ArucoDetector(d, p)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _ARUCO.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def hsv_mask(bgr, rng):
    """Mask of pixels inside an HSV range {"lo": [h, s, v], "hi": [h, s, v]} (OpenCV H is 0..179)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(rng["lo"]), np.array(rng["hi"])
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    # hue wraps around 0 (reds): split into two ranges
    a = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]]), np.array([179, hi[1], hi[2]]))
    b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), np.array([hi[0], hi[1], hi[2]]))
    return a | b


def wrap_angle(a):
    return np.mod(a, 2 * np.pi)


def import_pusht_wrapper():
    """Import the repo's Push-T wrapper without running env/__init__.py.

    env/__init__.py registers every gym env and imports mujoco_py for PointMaze. The data
    scripts only need Push-T, so a stub `env` package skips that import. train.py and
    plan.py import `env` normally and are unaffected.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "env" not in sys.modules:
        pkg = types.ModuleType("env")
        pkg.__path__ = [str(REPO_ROOT / "env")]
        sys.modules["env"] = pkg
    from env.pusht.pusht_human_wrapper import PushTHumanWrapper
    return PushTHumanWrapper


def make_env(damping=None, renderer="native"):
    PushTHumanWrapper = import_pusht_wrapper()
    return PushTHumanWrapper(with_velocity=True, with_target=True, damping=damping, renderer=renderer)


def conf_damping():
    """The damping in conf/env/pusht_human.yaml (the fitted value the datasets and plan.py use)."""
    import re
    m = re.search(r"^\s+damping:\s*([0-9.eE+-]+)", (REPO_ROOT / "conf/env/pusht_human.yaml").read_text(), re.M)
    return float(m.group(1)) if m else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_episode(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def write_mp4(frames, path, fps=10):
    """Write uint8 (T, H, W, 3) frames to an H.264 mp4 that decord can read."""
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=16,
                           pixelformat="yuv420p", ffmpeg_params=["-crf", "17"])
    for f in frames:
        w.append_data(np.ascontiguousarray(f))
    w.close()
