"""Per-episode end states and T errors from a planning run's final_eval_results.npz.

The state an episode is scored at is where it succeeded (closed loop stops there) or else the
end of its rollout. Files written by plan.py since this helper exist store it as final_state.
For older files it is rebuilt: episodes that met the repo's test are T-only successes by
definition, and their error is taken at the first step where the T was within tolerance
(older runs padded finished episodes with a non-zero action, so their last state drifted);
failed episodes ran to the end and were never padded, so their last state is exact.
"""
import numpy as np

UNITS_PER_CM = 512 / 34.0


def t_errors(goal, cur):
    te = np.linalg.norm(goal[..., 2:4] - cur[..., 2:4], axis=-1)
    ta = np.degrees(np.abs((goal[..., 4] - cur[..., 4] + np.pi) % (2 * np.pi) - np.pi))
    return te, ta


def load_results(path):
    d = np.load(path)
    success, goal, traj = d["success"].astype(bool), d["state_g"], d["e_states"]
    if "final_state" in d.files:
        fin = d["final_state"]
    else:
        fin = traj[:, -1].copy()
        for i in np.nonzero(success)[0]:
            te, ta = t_errors(goal[i][None], traj[i])
            hit = np.nonzero((te < 20) & (ta < 20))[0]
            if len(hit):
                fin[i] = traj[i, hit[0]]
    te, ta = t_errors(goal, fin)
    t_only = success | ((te < 20) & (ta < 20))
    return {"success": success, "t_only": t_only, "final": fin, "t_err": te, "t_err_deg": ta,
            "state_g": goal, "state_0": d["state_0"], "e_states": traj}
