"""Push-T for the phone-to-Panda project.

Same physics and interface as PushTWrapper (prepare / step_multiple / rollout / eval_state),
plus two options:
  damping   pymunk space damping from the system-ID fit (None keeps the repo's value, 0)
  renderer  'native' = the repo's 2D drawing, 'panda' = MuJoCo Panda posed by IK from the
            state, 'none' = blank frames for fast state-only replays
"""
import numpy as np

from env.pusht.pusht_env import PushTEnv
from env.pusht.pusht_wrapper import PushTWrapper


class PushTHumanWrapper(PushTWrapper):
    def __init__(self, with_velocity=True, with_target=True, damping=None, renderer="native",
                 render_size=224):
        PushTEnv.__init__(self, with_velocity=with_velocity, with_target=with_target,
                          damping=damping, render_size=render_size)
        self.action_dim = self.action_space.shape[0]
        self.renderer_name = renderer
        self._panda = None  # built lazily so each subprocess env owns its own GL context

    def _render_frame(self, mode):
        if self.renderer_name == "native" or mode == "human":
            return super()._render_frame(mode)
        if self.renderer_name == "none":
            return np.zeros((self.render_size, self.render_size, 3), np.uint8)
        if self._panda is None:
            from env.pusht.panda_renderer import PandaPushTRenderer
            self._panda = PandaPushTRenderer(size=self.render_size)
        block = np.array([self.block.position[0], self.block.position[1], self.block.angle])
        return self._panda.render(np.array(self.agent.position), block, np.asarray(self.goal_pose))

    def render_state(self, state):
        """Render a state without stepping physics (used by the pixels-only pass)."""
        self.agent.position = (float(state[0]), float(state[1]))
        self.block.angle = float(state[4])
        self.block.position = (float(state[2]), float(state[3]))
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)
        return self._render_frame("rgb_array")
