"""Compare planning runs on the same tasks (the same plan_targets.pkl).

    python helpers/plan_report.py <plan_targets.pkl> aug=<run_dir> random=<run_dir> ...

--detail N lists the first N tasks of each run (task index = the number in the video names).

A run done on more tasks of the same task file (e.g. open loop on 200 when the file given
here is the closed-loop run's 50) is cut to the first tasks, so the comparison stays paired.

Each run is label=<plan output folder holding final_eval_results.npz> (plan.py writes it),
or label=<a T/F string, one letter per task> for runs made before plan.py saved that file.

Prints:
  - what the tasks ask for: how far the pusher and the T move between start and goal
  - the do-nothing baseline (start state scored against the goal, same success test)
  - each run's success rate, overall and split by how much the T has to move
  - paired comparisons (same tasks): who solves what, exact McNemar p-value
"""
import os
import sys
import pickle
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


DETAIL = 0


def success_test(goal, cur):
    """env/pusht/pusht_wrapper.py eval_state: pusher+T position within 20 px, angle within 20 deg."""
    pos = np.linalg.norm(goal[:, :4] - cur[:, :4], axis=1)
    ang = np.abs((goal[:, 4] - cur[:, 4] + np.pi) % (2 * np.pi) - np.pi)
    return (pos < 20) & (ang < np.pi / 9)


def load_run(spec, n):
    label, val = spec.split("=", 1)
    path = os.path.join(val, "final_eval_results.npz")
    final = None
    if os.path.isfile(path) or (os.path.isfile(val) and val.endswith(".npz")):
        from plan_metrics import load_results
        r = load_results(path if os.path.isfile(path) else val)
        s, final = r["success"], r["final"]  # final: where the episode is scored
    else:
        toks = [t for t in val.replace(",", " ").replace("True", "T").replace("False", "F").split() if t]
        if len(toks) == 1:
            toks = list(toks[0])
        s = np.array([t.upper().startswith("T") for t in toks])
    assert len(s) >= n, f"{label}: {len(s)} results for {n} tasks"
    if len(s) > n:  # a run on a longer task list from the same file: its first n tasks are these
        print(f"({label}: using its first {n} of {len(s)} tasks)")
    return label, s[:n], (None if final is None else final[:n])


def errors(goal, cur):
    """final T position error (px), T angle error (deg), pusher position error (px)"""
    te = np.linalg.norm(goal[:, 2:4] - cur[:, 2:4], axis=1)
    ta = np.degrees(np.abs((goal[:, 4] - cur[:, 4] + np.pi) % (2 * np.pi) - np.pi))
    pe = np.linalg.norm(goal[:, :2] - cur[:, :2], axis=1)
    return te, ta, pe


def mcnemar(a, b):
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return only_a, only_b, p


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    global DETAIL
    args = sys.argv[1:]
    if "--detail" in args:  # --detail N: per-task errors for the first N tasks of each run
        k = args.index("--detail")
        DETAIL = int(args[k + 1])
        del args[k:k + 2]
    sys.argv = [sys.argv[0]] + args
    with open(sys.argv[1], "rb") as f:
        t = pickle.load(f)
    s0, sg = np.asarray(t["state_0"]), np.asarray(t["state_g"])
    n = len(s0)
    runs = [load_run(a, n) for a in sys.argv[2:]]

    agent_d = np.linalg.norm(sg[:, :2] - s0[:, :2], axis=1)
    tee_d = np.linalg.norm(sg[:, 2:4] - s0[:, 2:4], axis=1)
    tee_a = np.degrees(np.abs((sg[:, 4] - s0[:, 4] + np.pi) % (2 * np.pi) - np.pi))
    noop = success_test(sg, s0)
    moved = (tee_d > 10) | (tee_a > 10)

    q = lambda x: "median %5.1f  [25%% %5.1f, 75%% %5.1f, max %5.1f]" % (
        np.median(x), *np.percentile(x, [25, 75]), x.max())
    print(f"{n} tasks (goal_H={t.get('goal_H')})")
    print(f"  pusher moves  (px):  {q(agent_d)}")
    print(f"  T moves       (px):  {q(tee_d)}")
    print(f"  T turns      (deg):  {q(tee_a)}")
    print(f"  T has to move (>10 px or >10 deg): {moved.sum()} tasks, nearly still: {(~moved).sum()}")
    print(f"  do-nothing baseline success: {noop.mean():.2f}")
    print()

    se = lambda s: np.sqrt(s.mean() * (1 - s.mean()) / len(s))
    print(f"{'run':>12} {'success':>9} {'+-se':>6} {'T moves':>9} {'T still':>9}"
          f" {'T only':>8} {'T err px':>9} {'T err deg':>10}")
    for label, s, fin in runs:
        mv = f"{s[moved].mean():.2f}" if moved.any() else "-"
        st = f"{s[~moved].mean():.2f}" if (~moved).any() else "-"
        if fin is not None:
            te, ta, _ = errors(sg, fin)
            extra = f" {(s | ((te < 20) & (ta < 20))).mean():8.2f} {np.median(te):9.1f} {np.median(ta):10.1f}"
        else:
            extra = f" {'-':>8} {'-':>9} {'-':>10}"
        print(f"{label:>12} {s.mean():9.2f} {se(s):6.2f} {mv:>9} {st:>9}{extra}")
    print("  success: pusher AND T within 20 px together (norm of the 4 position numbers) and T angle")
    print("  within 20 deg, in the real env.  T only: the same test ignoring the pusher.  T err: median")
    print("  final distance of the T from the goal T.")
    for label, s, fin in runs:
        if fin is None or DETAIL <= 0:
            continue
        te, ta, pe = errors(sg, fin)
        print(f"\n{label}: first {min(DETAIL, n)} tasks (the videos output_final_<task>_*.mp4)")
        print(f"  {'task':>4} {'result':>8} {'T px':>6} {'T deg':>6} {'pusher px':>10}")
        for i in range(min(DETAIL, n)):
            print(f"  {i:>4} {'success' if s[i] else 'failure':>8} {te[i]:6.1f} {ta[i]:6.1f} {pe[i]:10.1f}")
    if len(runs) >= 2:
        print("\npaired (same tasks):")
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                (la, a, _), (lb, b, _) = runs[i], runs[j]
                oa, ob, p = mcnemar(a, b)
                print(f"  {la} vs {lb}: both {int((a & b).sum())}, only {la} {oa}, "
                      f"only {lb} {ob}, neither {int((~a & ~b).sum())}  (McNemar p={p:.2f})")


if __name__ == "__main__":
    main()
