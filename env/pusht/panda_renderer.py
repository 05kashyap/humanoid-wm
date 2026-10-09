"""Render a Push-T state with a MuJoCo Franka Panda holding the pusher.

Physics stays in the repo's pymunk simulator. This module only draws: the T and the goal
decal are mocap bodies posed from the state, and the arm's 7 joints are solved by damped
least-squares IK so the gripper holds a 2 cm pusher cylinder at the pusher position.

Needs:  pip install "mujoco>=3.1"  and a clone of MuJoCo Menagerie:
        git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie
        export MENAGERIE_DIR=/path/to/mujoco_menagerie
Headless: export MUJOCO_GL=egl (GPU nodes) or MUJOCO_GL=osmesa (CPU).
"""
import os
from pathlib import Path

import mujoco
import numpy as np

ARENA = 512.0
ARENA_M = 0.34                 # the arena is 34 cm on the table
S = ARENA_M / ARENA            # metres per arena unit
CENTER = np.array([0.5, 0.0])  # arena centre, metres in front of the robot base
T_H = 0.010                    # T thickness
PUSHER_R = 15 * S              # repo pusher radius (15 units)
PUSHER_H = 0.12                # a rod keeps the hand above the T, so it hides less of it
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # hand frame -> between the fingertips
GRIP_Z = PUSHER_H - 0.008      # fingertips hold the top of the cylinder
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


def _scene_xml():
    def t_geoms(prefix, h, z, rgba):
        return (f'<geom name="{prefix}_bar" type="box" pos="0 {15 * S} {z + h / 2}" size="{60 * S} {15 * S} {h / 2}" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                f'<geom name="{prefix}_stem" type="box" pos="0 {75 * S} {z + h / 2}" size="{15 * S} {45 * S} {h / 2}" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>')
    return f"""<mujoco model="panda pusht">
  <include file="panda.xml"/>
  <visual>
    <headlight diffuse="0.25 0.25 0.25" ambient="0.2 0.2 0.2" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light pos="0.6 0 1.6" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35" specular="0 0 0" castshadow="false"/>
    <geom name="table" type="box" pos="0.45 0 -0.02" size="0.7 0.6 0.02" rgba="0.62 0.60 0.57 1" contype="0" conaffinity="0"/>
    <geom name="arena" type="box" pos="{CENTER[0]} {CENTER[1]} 0.0002" size="{ARENA_M / 2} {ARENA_M / 2} 0.0002"
          rgba="1 1 1 1" contype="0" conaffinity="0"/>
    <body name="goal" mocap="true">{t_geoms("goal", 0.0004, 0.0004, "0.565 0.933 0.565 1")}</body>
    <body name="tee" mocap="true">{t_geoms("tee", T_H, 0.0, "0.30 0.36 0.45 1")}</body>
    <body name="pusher" mocap="true">
      <geom type="cylinder" pos="0 0 {PUSHER_H / 2}" size="{PUSHER_R} {PUSHER_H / 2}" rgba="0.255 0.412 0.882 1"
            contype="0" conaffinity="0"/>
    </body>
    <camera name="cam" pos="0.47 0.52 0.56" xyaxes="-1 0 0 0 -0.716 0.698" fovy="40"/>
  </worldbody>
</mujoco>"""


def _arena_to_world(uv):
    """Arena (u right, v down) -> table (X forward, Y left). One axis swap, so theta flips."""
    return np.array([CENTER[0] + (uv[1] - 256.0) * S, CENTER[1] + (uv[0] - 256.0) * S])


def _pose_quat(theta):
    # R_body = P R(theta) D with P = axis swap and D = mirror of local x; det = +1. The T is
    # symmetric about its stem, so mirroring local x leaves its shape unchanged.
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[0, 1], [1, 0]]) @ np.array([[c, -s], [s, c]]) @ np.diag([-1, 1])
    R = np.eye(3)
    R[:2, :2] = R2
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


class PandaPushTRenderer:
    def close(self):
        r = getattr(self, "renderer", None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
            self.renderer = None

    def __init__(self, size=224, menagerie_dir=None, arm_alpha=None):
        """arm_alpha < 1 draws the arm semi-transparent, an option if it hides the T too often.
        It defaults to the PANDA_ARM_ALPHA environment variable (1.0 if unset), so it reaches
        every subprocess env that plan.py and make_dataset.py start."""
        if arm_alpha is None:
            arm_alpha = float(os.environ.get("PANDA_ARM_ALPHA", 1.0))
        root = Path(menagerie_dir or os.environ.get("MENAGERIE_DIR", "mujoco_menagerie"))
        panda_dir = root / "franka_emika_panda"
        if not (panda_dir / "panda.xml").exists():
            raise FileNotFoundError(f"panda.xml not found under {panda_dir}; set MENAGERIE_DIR")
        scene = panda_dir / "pusht_scene.xml"   # must sit next to panda.xml for its mesh paths
        xml = _scene_xml()
        if not scene.exists() or scene.read_text() != xml:
            scene.write_text(xml)
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.m.light_castshadow[:] = 0          # menagerie's spotlight casts large dark shadows
        if arm_alpha < 1.0:
            own = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("world", "tee", "goal", "pusher")}
            mats = {int(self.m.geom_matid[g]) for g in range(self.m.ngeom)
                    if self.m.geom_bodyid[g] not in own and self.m.geom_matid[g] >= 0}
            self.m.mat_rgba[list(mats), 3] *= arm_alpha
        self.renderer = mujoco.Renderer(self.m, size, size)
        # free the GL context before MuJoCo's own exit hook tears EGL down (atexit runs in
        # reverse order); otherwise every script ends with a harmless EGL_NOT_INITIALIZED trace
        import atexit
        import weakref
        ref = weakref.ref(self)
        atexit.register(lambda: ref() is not None and ref().close())
        self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        names = [f"joint{i}" for i in range(1, 8)]
        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jids])
        self.lo, self.hi = self.m.jnt_range[jids, 0], self.m.jnt_range[jids, 1]
        for f in ("finger_joint1", "finger_joint2"):
            self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f)]] = PUSHER_R
        self.mocap = {n: self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)]
                      for n in ("goal", "tee", "pusher")}
        self.q = HOME.copy()
        self.d.qpos[self.qadr] = self.q
        mujoco.mj_kinematics(self.m, self.d)
        self.R_target = self.d.xmat[self.hand].reshape(3, 3).copy()  # keep the home gripper orientation
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.last_ik_error = 0.0

    def _ik(self, target, iters=60):
        q = self.q.copy()
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.xmat[self.hand].reshape(3, 3)
            tcp = self.d.xpos[self.hand] + R @ TCP_OFFSET
            ep = target - tcp
            er = 0.5 * sum(np.cross(R[:, i], self.R_target[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jac(self.m, self.d, self.jp, self.jr, tcp, self.hand)
            J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
            A = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, np.r_[ep, er])
            dq += (np.eye(7) - J.T @ np.linalg.solve(A, J)) @ (0.05 * (HOME - q))
            q = np.clip(q + dq, self.lo, self.hi)
        self.q = q
        self.last_ik_error = float(np.linalg.norm(ep))
        return q

    def render(self, agent_uv, block_pose, goal_pose):
        for name, pose in (("tee", block_pose), ("goal", goal_pose)):
            xy = _arena_to_world(pose[:2])
            self.d.mocap_pos[self.mocap[name]] = [xy[0], xy[1], 0.0]
            self.d.mocap_quat[self.mocap[name]] = _pose_quat(pose[2])
        pxy = _arena_to_world(agent_uv)
        self.d.mocap_pos[self.mocap["pusher"]] = [pxy[0], pxy[1], 0.0]
        self._ik(np.array([pxy[0], pxy[1], GRIP_Z]))
        self.d.qpos[self.qadr] = self.q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)   # camera and light poses are not part of mj_kinematics
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render().copy()
