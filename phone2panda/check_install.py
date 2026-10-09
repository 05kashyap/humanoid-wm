"""Check every dependency of the pipeline in a minute, before recording anything.

Usage:  MUJOCO_GL=egl MENAGERIE_DIR=/path/to/mujoco_menagerie python -m phone2panda.check_install
"""
import os
import tempfile
import traceback


def check(name, fn):
    try:
        msg = fn()
        print(f"OK    {name}{': ' + msg if msg else ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
        return False


def aruco():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    return f"OpenCV {cv2.__version__}"


def pusht():
    import numpy as np
    from phone2panda.common import make_env
    env = make_env(renderer="native")
    obs, state = env.prepare(0, np.array([256, 400, 256, 256, 0.5, 0, 0], float))
    env.step(np.array([0.0, -0.5]))
    return f"frame {obs['visual'].shape}"


def panda():
    import numpy as np
    from phone2panda.common import make_env
    env = make_env(renderer="panda")
    obs, _ = env.prepare(0, np.array([256, 400, 256, 256, 0.5, 0, 0], float))
    return f"frame {obs['visual'].shape}, IK error {env._panda.last_ik_error * 1000:.2f} mm (MUJOCO_GL={os.environ.get('MUJOCO_GL')})"


def video():
    import numpy as np
    from phone2panda.common import write_mp4
    p = os.path.join(tempfile.mkdtemp(), "t.mp4")
    write_mp4(np.zeros((5, 224, 224, 3), np.uint8), p)
    try:
        import decord
        n = len(decord.VideoReader(p))
        return f"mp4 written and read back by decord ({n} frames)"
    except ImportError:
        return "mp4 written (decord not installed here; the repo's loader needs it)"


def torch_ok():
    import torch
    return f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}"


if __name__ == "__main__":
    results = [check("OpenCV ArUco", aruco), check("repo Push-T env", pusht), check("Panda renderer", panda),
               check("video I/O", video), check("torch", torch_ok)]
    print("all good" if all(results) else "fix the FAIL lines before recording")
