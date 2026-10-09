"""Turn tracked episodes into a dataset in the exact format datasets/pusht_dset.py loads.

Each episode is physics-replayed in the repo's Push-T simulator: the pusher is commanded to
the fingertip position two control steps ahead (this cancels the PD controller's ~0.2 s lag),
and the T moves under the simulator's physics. The commands actually sent are stored as the
actions. States are simulated first; frames are rendered in a second pass.

Output (one folder per split, plus stats.pth computed on train):
  <out>/train/{states.pth, velocities.pth, rel_actions.pth, abs_actions.pth, seq_lengths.pkl, obses/episode_XXX.mp4}
  <out>/val/...   <out>/stats.pth   <out>/meta.json

Usage:
  # the split is created once and reused by every dataset, so all models share validation demos
  python -m phone2panda.make_dataset --episodes episodes/ --split split.json --out $DATASET_DIR/pusht_human_aug \
      --augment 16 --damping 0 --renderer panda --workers 16
  python -m phone2panda.make_dataset --episodes episodes/ --split split.json --out $DATASET_DIR/pusht_human_raw \
      --augment 0 --damping 0 --renderer panda --workers 16
  python -m phone2panda.make_dataset --episodes episodes/ --split split.json --out $DATASET_DIR/pusht_random \
      --random --match $DATASET_DIR/pusht_human_aug --damping 0 --renderer panda --workers 16
  # human + branch rollouts, trimmed to the same number of training steps as pusht_human_aug
  python -m phone2panda.make_dataset --episodes episodes/ --split split.json --out $DATASET_DIR/pusht_human_branch \
      --augment 8 --branches 24 --match $DATASET_DIR/pusht_human_aug --damping 0 --renderer panda --workers 16

Branch rollouts (--branches K): for each training demo, K times, replay the demo up to a random
step t0, then continue for 2-5 s with random pusher motion whose waypoints mostly sit near the T.
Each rollout is one continuous simulation; it is stored from 15 steps before t0, so it starts
mid-push. Validation is never augmented or branched.
"""
import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .common import (ARENA, PUSHER_RADIUS, T_CENTROID_LOCAL, WALL_MARGIN, dist_to_t, load_episode, load_json,
                     make_env, rot, save_json, t_polygons, wrap_angle, write_mp4)

LOOKAHEAD = 2      # control steps; matches the PD controller's steady-state lag (k_v / k_p = 0.2 s)
MAX_STEP = 100.0   # clip per-step commands (units) to absorb tracking glitches
BRANCH_CONTEXT = 15  # control steps of the human demo kept before a branch point (one history window)


# ---------------------------------------------------------------- replay
def replay(env, finger, tee0, lookahead=LOOKAHEAD, max_step=MAX_STEP, render=True):
    """Physics-replay one path. Returns frames (or None), states (T,5), velocities (T,2), actions (T,2).

    actions[t] is the relative pusher command applied at step t, in arena units (the repo's
    loader divides rel_actions by 100 and the env multiplies by 100). The last action is padding.
    """
    init = np.r_[finger[0], tee0[:2], wrap_angle(tee0[2]), 0.0, 0.0]
    obs, state = env.prepare(0, init)
    frames = [obs["visual"]] if render else None
    states, acts = [state], []
    T = len(finger)
    for t in range(T - 1):
        target = finger[min(t + lookahead, T - 1)]
        delta = np.clip(target - state[:2], -max_step, max_step)
        obs, _, _, info = env.step(delta / 100.0)
        state = info["state"]
        if render:
            frames.append(obs["visual"])
        states.append(state)
        acts.append(delta)
    acts.append(np.zeros(2))
    states = np.stack(states).astype(np.float32)
    return (np.stack(frames) if render else None), states[:, :5], states[:, 5:7], np.stack(acts).astype(np.float32)


# ---------------------------------------------------------------- augmentation
def valid_start(path, tee0):
    if (path < WALL_MARGIN).any() or (path > ARENA - WALL_MARGIN).any():
        return False
    pts = np.concatenate(t_polygons(tee0))
    if (pts < 8).any() or (pts > ARENA - 8).any():
        return False
    return dist_to_t(path[0], tee0) > PUSHER_RADIUS + 2


def aug_jitter(path, tee0, rng):
    return path, tee0 + np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), np.deg2rad(rng.uniform(-5, 5))])


def aug_se2(path, tee0, rng):
    phi = rng.uniform(-np.pi, np.pi)
    shift = rng.uniform(-60, 60, 2)
    c = np.array([ARENA / 2, ARENA / 2])
    R = rot(phi)
    return (path - c) @ R.T + c + shift, np.r_[R @ (tee0[:2] - c) + c + shift, tee0[2] + phi]


def aug_mirror(path, tee0, rng):
    # valid because the T is symmetric about its stem: pose (x, y, th) -> (512 - x, y, -th)
    p = path.copy()
    p[:, 0] = ARENA - p[:, 0]
    return p, np.array([ARENA - tee0[0], tee0[1], -tee0[2]])


def aug_noise(path, tee0, rng, sigma=5.0):
    n = gaussian_filter1d(rng.normal(0, 1, path.shape), 3, axis=0)
    n *= sigma / (n.std(0, keepdims=True) + 1e-8)
    return path + n, tee0


def aug_retime(path, tee0, rng):
    s = rng.uniform(0.7, 1.4)
    idx = np.arange(0, len(path) - 1, s)
    return np.stack([np.interp(idx, np.arange(len(path)), path[:, d]) for d in range(2)], 1), tee0


def make_variants(path, tee0, k, rng, max_tries=50):
    out = []
    tries = 0
    while len(out) < k and tries < k * max_tries:
        tries += 1
        p, t = path.copy(), tee0.copy()
        used = []
        if rng.random() < 0.5:
            p, t = aug_se2(p, t, rng); used.append("se2")
        if rng.random() < 0.5:
            p, t = aug_mirror(p, t, rng); used.append("mirror")
        if rng.random() < 0.8 or not used:
            p, t = aug_jitter(p, t, rng); used.append("jitter")
        if rng.random() < 0.5:
            p, t = aug_noise(p, t, rng); used.append("noise")
        if rng.random() < 0.5:
            p, t = aug_retime(p, t, rng); used.append("retime")
        if valid_start(p, t):
            out.append((p, t, "+".join(used)))
    return out


# ---------------------------------------------------------------- random-play baseline
def random_episode(rng, n_steps, speed, p_near=0.0):
    """Pusher heads to a new random waypoint every 1-3 s at `speed` units/step."""
    while True:
        tee0 = np.array([rng.uniform(100, ARENA - 100), rng.uniform(100, ARENA - 100), rng.uniform(0, 2 * np.pi)])
        p = rng.uniform(WALL_MARGIN + 20, ARENA - WALL_MARGIN - 20, 2)
        if valid_start(p[None], tee0):
            break
    path = [p]
    while len(path) < n_steps:
        if rng.random() < p_near:
            wp = tee0[:2] + rng.normal(0, 60, 2)
        else:
            wp = rng.uniform(WALL_MARGIN + 20, ARENA - WALL_MARGIN - 20, 2)
        wp = np.clip(wp, WALL_MARGIN + 5, ARENA - WALL_MARGIN - 5)
        n = int(rng.integers(10, 31))
        for _ in range(n):
            d = wp - path[-1]
            step = d if np.linalg.norm(d) < speed else d / np.linalg.norm(d) * speed
            path.append(path[-1] + step)
    return np.array(path[:n_steps]), tee0


# ---------------------------------------------------------------- branch rollouts
def branch_continuation(start, tee, rng, speed, p_near=0.7):
    """2-5 s of random pusher motion from `start`: waypoints mostly near the T's centre, some anywhere."""
    n = int(rng.integers(20, 51))
    centre = tee[:2] + rot(tee[2]) @ T_CENTROID_LOCAL
    pts = [np.asarray(start, float)]
    while len(pts) < n + 1:
        if rng.random() < p_near:
            wp = centre + rng.normal(0, 60, 2)
        else:
            wp = rng.uniform(WALL_MARGIN + 20, ARENA - WALL_MARGIN - 20, 2)
        wp = np.clip(wp, WALL_MARGIN + 5, ARENA - WALL_MARGIN - 5)
        for _ in range(int(rng.integers(10, 31))):
            d = wp - pts[-1]
            pts.append(pts[-1] + (d if np.linalg.norm(d) < speed else d / np.linalg.norm(d) * speed))
    return np.array(pts[1:n + 1])


# ---------------------------------------------------------------- workers
_ENV = None


def _init_worker(damping, renderer):
    global _ENV
    _ENV = make_env(damping=damping, renderer=renderer)


def _simulate(job):
    path, tee0 = job
    _, states, vels, acts = replay(_ENV, path, tee0, render=False)
    track_err = float(np.linalg.norm(states[:, :2] - path, axis=1).mean())
    return states, vels, acts, track_err


def _branch(job):
    """Replay the human path to step t0, continue with random pushing, keep BRANCH_CONTEXT steps before t0."""
    path, tee0, t0, seed, speed = job
    _, pre, _, _ = replay(_ENV, path[:t0 + 1], tee0, render=False)
    cont = branch_continuation(path[t0], pre[-1, 2:5], np.random.default_rng(seed), speed)
    full = np.concatenate([path[:t0 + 1], cont])
    _, states, vels, acts = replay(_ENV, full, tee0, render=False)
    track_err = float(np.linalg.norm(states[:, :2] - full, axis=1).mean())
    c = max(0, t0 - BRANCH_CONTEXT)
    return states[c:], vels[c:], acts[c:], track_err


def _render(job):
    states, video_path = job
    _ENV.prepare(0, np.r_[states[0], 0.0, 0.0])
    frames = np.stack([_ENV.render_state(s) for s in states])
    write_mp4(frames, video_path, fps=10)
    return video_path


def run_pool(fn, jobs, workers, damping, renderer):
    if workers <= 1:
        _init_worker(damping, renderer)
        return [fn(j) for j in jobs]
    # look the functions up by module name so they pickle even when this file runs as __main__
    import importlib
    mod = importlib.import_module("phone2panda.make_dataset")
    ctx = mp.get_context("spawn")  # each worker gets a fresh process with its own GL context
    with ctx.Pool(workers, initializer=mod._init_worker, initargs=(damping, renderer)) as pool:
        return pool.map(getattr(mod, fn.__name__), jobs, chunksize=4)


# ---------------------------------------------------------------- writing
def write_split(folder, sims, workers, damping, renderer):
    import torch
    folder = Path(folder)
    (folder / "obses").mkdir(parents=True, exist_ok=True)
    lengths = [len(s[0]) for s in sims]
    T_max = max(lengths)
    N = len(sims)
    states = np.zeros((N, T_max, 5), np.float32)
    vels = np.zeros((N, T_max, 2), np.float32)
    acts = np.zeros((N, T_max, 2), np.float32)
    for i, (s, v, a, _) in enumerate(sims):
        states[i, :len(s)], vels[i, :len(s)], acts[i, :len(s)] = s, v, a
    abs_acts = np.zeros_like(acts)
    abs_acts[:, :-1] = states[:, 1:, :2]  # pusher target positions, for completeness
    torch.save(torch.from_numpy(states), folder / "states.pth")
    torch.save(torch.from_numpy(vels), folder / "velocities.pth")
    torch.save(torch.from_numpy(acts), folder / "rel_actions.pth")
    torch.save(torch.from_numpy(abs_acts), folder / "abs_actions.pth")
    with open(folder / "seq_lengths.pkl", "wb") as f:
        pickle.dump(lengths, f)
    jobs = [(sims[i][0], str(folder / "obses" / f"episode_{i:03d}.mp4")) for i in range(N)]
    run_pool(_render, jobs, workers, damping, renderer)
    return states, vels, acts, lengths


def compute_stats(states, vels, acts, lengths):
    import torch
    m = np.zeros(states.shape[:2], bool)
    for i, L in enumerate(lengths):
        m[i, :L] = True
    a_m = m.copy()
    for i, L in enumerate(lengths):
        a_m[i, L - 1] = False                      # the padding action
    a = acts[a_m] / 100.0                          # the loader divides rel_actions by action_scale=100
    full = np.concatenate([states, vels], -1)[m]   # 7-D state the loader builds (with_velocity=True)
    prop = np.concatenate([states[..., :2], vels], -1)[m]
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return {"action_mean": t(a.mean(0)), "action_std": t(a.std(0) + 1e-6),
            "state_mean": t(full.mean(0)), "state_std": t(full.std(0) + 1e-6),
            "proprio_mean": t(prop.mean(0)), "proprio_std": t(prop.std(0) + 1e-6)}


def load_split(episodes_dir, split_path, val_frac, seed):
    names = sorted(p.stem for p in Path(episodes_dir).glob("*.npz"))
    if split_path and Path(split_path).exists():
        split = load_json(split_path)
        missing = [n for n in split["train"] + split["val"] if n not in names]
        if missing:
            raise SystemExit(f"{len(missing)} episodes in {split_path} are missing, e.g. {missing[:3]}")
        return split
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(names))
    n_val = min(max(1, int(round(val_frac * len(names)))), len(names) - 1)  # always keep >= 1 train demo
    split = {"train": sorted(perm[n_val:]), "val": sorted(perm[:n_val])}
    if split_path:
        save_json(split, split_path)
        print(f"created {split_path}: {len(split['train'])} train / {len(split['val'])} val demos")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="split.json", help="created on first use, then reused")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--augment", type=int, default=16, help="variants per training demo")
    ap.add_argument("--random", action="store_true", help="random-play baseline instead of human demos")
    ap.add_argument("--match", help="dataset folder whose number of training steps to match (random set, or "
                                     "the branch rollouts kept with --branches)")
    ap.add_argument("--p-near", type=float, default=0.0, help="random set: share of waypoints near the T")
    ap.add_argument("--branches", type=int, default=0,
                    help="branch rollouts per training demo: the demo up to a random step, then 2-5 s of random "
                         "pushing aimed mostly at the T (DART/DINO-WM-style coverage around the human states)")
    ap.add_argument("--max-demos", type=int, help="use only the first N training demos (data-efficiency runs)")
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--renderer", default="panda", choices=["panda", "native"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)

    split = load_split(args.episodes, args.split, args.val_frac, args.seed)
    eps = {n: load_episode(Path(args.episodes) / f"{n}.npz") for n in split["train"] + split["val"]}
    train_names = split["train"][:args.max_demos] if args.max_demos else split["train"]

    # training jobs
    jobs, meta = [], []
    if args.random:
        human_steps = [len(eps[n]["finger"]) for n in train_names]
        speeds = np.concatenate([np.linalg.norm(np.diff(eps[n]["finger"], axis=0), axis=1) for n in train_names])
        speed = float(np.median(speeds[speeds > 1.0]))
        target = sum(human_steps) if not args.match else sum(pickle.load(open(Path(args.match) / "train" / "seq_lengths.pkl", "rb")))
        n_steps = int(np.median(human_steps))
        while sum(len(j[0]) for j in jobs) < target:
            p, t = random_episode(rng, n_steps, speed, args.p_near)
            jobs.append((p, t)); meta.append({"source": "random"})
        print(f"random play: {len(jobs)} episodes of {n_steps} steps at {speed:.1f} units/step (target {target} steps)")
    else:
        for n in train_names:
            e = eps[n]
            jobs.append((e["finger"], e["tee"][0])); meta.append({"source": n, "aug": "none"})
            for p, t, used in make_variants(e["finger"], e["tee"][0], args.augment, rng):
                jobs.append((p, t)); meta.append({"source": n, "aug": used})
        print(f"human: {len(train_names)} demos -> {len(jobs)} training episodes")
    branch_jobs = []
    if args.branches and not args.random:
        for n in train_names:
            f = eps[n]["finger"]
            if len(f) < 15:
                continue
            seg = np.linalg.norm(np.diff(f, axis=0), axis=1)
            speed = float(np.median(seg[seg > 1.0])) if (seg > 1.0).any() else 5.0
            for _ in range(args.branches):
                t0 = int(rng.integers(5, len(f) - 5))
                branch_jobs.append((f, eps[n]["tee"][0], t0, int(rng.integers(2**31)), speed))
                meta.append({"source": n, "aug": f"branch@{t0}"})
    val_jobs = [(eps[n]["finger"], eps[n]["tee"][0]) for n in split["val"]]

    # pass 1: states (no rendering)
    sims = run_pool(_simulate, jobs, args.workers, args.damping, "none")
    if branch_jobs:
        branch_sims = run_pool(_branch, branch_jobs, args.workers, args.damping, "none")
        if args.match:  # keep a random subset of branches so the total matches the other dataset's steps
            target = sum(pickle.load(open(Path(args.match) / "train" / "seq_lengths.pkl", "rb")))
            room = target - sum(len(s[0]) for s in sims)
            keep, used = [], 0
            for i in rng.permutation(len(branch_sims)):
                if used + len(branch_sims[i][0]) <= room:
                    keep.append(i)
                    used += len(branch_sims[i][0])
            keep = sorted(keep)
            n_human = len(meta) - len(branch_sims)
            meta = meta[:n_human] + [meta[n_human + i] for i in keep]
            print(f"--match: kept {len(keep)} of {len(branch_sims)} branches to reach {target} steps")
            branch_sims = [branch_sims[i] for i in keep]
        print(f"branches: {len(branch_sims)} rollouts, {sum(len(s[0]) for s in branch_sims)} steps "
              f"(human episodes: {sum(len(s[0]) for s in sims)} steps)")
        sims = sims + branch_sims
    val_sims = run_pool(_simulate, val_jobs, args.workers, args.damping, "none")
    err = np.array([s[3] for s in sims])
    print(f"pusher tracking error vs commanded path: median {np.median(err) / 15.06 * 10:.1f} mm, "
          f"95th pct {np.percentile(err, 95) / 15.06 * 10:.1f} mm")

    # pass 2: pixels, then files
    states, vels, acts, lengths = write_split(out / "train", sims, args.workers, args.damping, args.renderer)
    if val_sims:
        write_split(out / "val", val_sims, args.workers, args.damping, args.renderer)
    else:
        print("no validation demos (fine for a pilot; train.py and plan.py need a val/ split)")
    import torch
    torch.save(compute_stats(states, vels, acts, lengths), out / "stats.pth")
    save_json({"episodes": meta, "val": split["val"], "damping": args.damping, "renderer": args.renderer,
               "lookahead": LOOKAHEAD, "augment": args.augment, "branches": args.branches, "random": args.random,
               "train_steps": int(sum(lengths))}, out / "meta.json")
    print(f"wrote {out}: {len(sims)} train / {len(val_sims)} val episodes, {sum(lengths)} train steps")


if __name__ == "__main__":
    main()
