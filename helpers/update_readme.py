"""Fill the results tables in README.md from the planning outputs.

    python helpers/update_readme.py            # rewrites the block between the results markers

Every number comes from plan_outputs/<mode>/<model>/final_eval_results.npz (all models of a
mode share one task file); runs that have not been done yet show as "pending". The best value in
each column is bolded. (E3 and the photo goals are no longer reported in the README.)
"""
import re
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_metrics import load_results  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNITS_PER_CM = 512 / 34.0
PENDING = "*pending*"

E1 = [("aug", "my demos + augmentation"), ("random", "random play (same size)"),
      ("branch", "my demos + branch rollouts (same size)"), ("raw", "my demos only")]
E2 = [("noreg", "none (lower encoder learning rate)"), ("aug", "straightening"), ("pacing", "pacing"),
      ("both", "straightening + pacing")]


def load(mode, model):
    f = ROOT / "plan_outputs" / mode / model / "final_eval_results.npz"
    if not f.exists():
        return None
    r = load_results(f)
    return {"n": len(r["success"]), "success": r["success"].mean(), "t_only": r["t_only"].mean(),
            "t_err_cm": np.median(r["t_err"]) / UNITS_PER_CM, "t_err_deg": np.median(r["t_err_deg"])}


def cell(r, key="success"):
    return PENDING if r is None else f"{r[key]:.2f}"


def bold_best(rows):
    """Bold the highest number in each numeric column (ties all bold); pending cells are skipped."""
    for c in range(1, len(rows[0])):
        vals = [float(r[c]) for r in rows if r[c] != PENDING]
        if vals:
            best = max(vals)
            for r in rows:
                if r[c] != PENDING and float(r[c]) == best:
                    r[c] = f"**{r[c]}**"
    return rows


def table(rows, header, cols):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build():
    parts = []
    n = {m: next((load(m, x)["n"] for x, _ in E1 + E2 if load(m, x)), None) for m in ("ol", "cl")}
    ntxt = lambda m: f"{n[m]} tasks" if n[m] else "tasks pending"

    rows = [[desc, cell(load("ol", m)), cell(load("ol", m), "t_only"), cell(load("cl", m)),
             cell(load("cl", m), "t_only")] for m, desc in E1]
    bold_best(rows)
    parts.append("### E1: does my data beat random play?\n\nAll models use temporal straightening and the same "
                 "number of training steps.\n\n" + table(
                     rows, ["training data", f"open loop ({ntxt('ol')})", "T only", f"closed loop ({ntxt('cl')})", "T only"], 6))

    rows = [[desc, cell(load("ol", m)), cell(load("ol", m), "t_only"), cell(load("cl", m)),
             cell(load("cl", m), "t_only")] for m, desc in E2]
    bold_best(rows)
    parts.append("### E2: which regularizer helps planning?\n\nAll models train on human demos + augmentation.\n\n" + table(
        rows, ["regularizer", "open loop", "T only", "closed loop", "T only"], 5))

    parts.append("The best value in each column is in bold. One training run per model; the tasks, not training "
                 "seeds, are the sample. Regenerate with `python helpers/update_readme.py`.")
    return "\n\n".join(parts)


def main():
    readme = ROOT / "README.md"
    s = readme.read_text()
    new = "<!-- results:start -->\n" + build() + "\n<!-- results:end -->"
    s2, k = re.subn(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: new, s, flags=re.S)
    if k != 1:
        raise SystemExit("README.md needs one <!-- results:start --> ... <!-- results:end --> block")
    readme.write_text(s2)
    print("updated the results in README.md")


if __name__ == "__main__":
    main()
