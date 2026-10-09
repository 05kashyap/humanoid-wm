"""Results figure and table from plan_outputs/<mode>/<model>/final_eval_results.npz.

    python helpers/plot_results.py                       # every mode and model found
    python helpers/plot_results.py --modes ol cl --models noreg aug both random branch

Writes figures/results.png (two panels: the repo's success and T-only success, one bar per
model and mode, 95% Wilson intervals over tasks) and figures/results.md (the same numbers as
a table, plus task counts). Models keep the order given (default: the order below); modes
are colored in a fixed order: ol, cl, full, photo.
"""
import argparse
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_metrics import load_results  # noqa: E402

MODEL_ORDER = ["noreg", "aug", "pacing", "both", "random", "branch", "raw"]
MODE_ORDER = ["ol", "cl", "full", "photo"]
MODE_NAME = {"ol": "open loop", "cl": "closed loop (MPC)", "full": "full task (MPC)", "photo": "photo goals (MPC)"}
MODE_COLOR = {"ol": "#2a78d6", "cl": "#eb6834", "full": "#1baf7a", "photo": "#eda100"}  # fixed slots
LABEL = {"noreg": "no reg.", "aug": "straightening", "pacing": "pacing", "both": "straightening\n+ pacing",
         "random": "random data\n(straightening)", "branch": "branch data\n(straightening)",
         "raw": "raw demos\n(straightening)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def t_only(goal, cur):
    pos = np.linalg.norm(goal[:, 2:4] - cur[:, 2:4], axis=1)
    ang = np.abs((goal[:, 4] - cur[:, 4] + np.pi) % (2 * np.pi) - np.pi)
    return (pos < 20) & (ang < np.pi / 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plan_outputs")
    ap.add_argument("--modes", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--out", default="figures/results")
    args = ap.parse_args()
    root = Path(args.root)

    found = {}
    for f in sorted(root.glob("*/*/final_eval_results.npz")):
        mode, model = f.parent.parent.name, f.parent.name
        r = load_results(f)
        found[(mode, model)] = (r["success"], r["t_only"])
    if not found:
        raise SystemExit(f"no final_eval_results.npz under {root}/<mode>/<model>/")
    modes = args.modes or [m for m in MODE_ORDER if any(k[0] == m for k in found)]
    all_models = {k[1] for k in found}
    models = args.models or [m for m in MODEL_ORDER if m in all_models] + sorted(all_models - set(MODEL_ORDER))
    models = [m for m in models if any((md, m) in found for md in modes)]

    # table
    lines = ["| model | " + " | ".join(f"{MODE_NAME.get(md, md)}: success / T-only (n)" for md in modes) + " |",
             "|---|" + "---|" * len(modes)]
    for m in models:
        cells = []
        for md in modes:
            if (md, m) in found:
                s, t = found[(md, m)]
                cells.append(f"{s.mean():.2f} / {t.mean():.2f} ({len(s)})")
            else:
                cells.append("-")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".md").write_text(table + "\n\nsuccess: the repo's test (pusher and T within 20 px, T angle within "
                                      "20 deg); T-only: the same test without the pusher. n = tasks; all models in a "
                                      "column share the same tasks.\n")
    print(table)

    # figure: two panels on the same 0-1 scale, bars grouped by model, one color per mode
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dc"
    fig, axes = plt.subplots(1, 2, figsize=(max(7.5, 1.5 * len(models) * 2), 3.6), sharey=True)
    width = 0.8 / len(modes)
    x = np.arange(len(models))
    for ax, which, title in zip(axes, (0, 1), ("Success (repo test: pusher + T)", "T-only success")):
        for j, md in enumerate(modes):
            for i, m in enumerate(models):
                if (md, m) not in found:
                    continue
                v = found[(md, m)][which]
                lo, hi = wilson(v.sum(), len(v))
                xi = x[i] - 0.4 + width * (j + 0.5)
                ax.bar(xi, v.mean(), width - 0.04, color=MODE_COLOR.get(md, "#4a3aa7"),
                       label=MODE_NAME.get(md, md) if i == 0 or (md, models[0]) not in found else None)
                ax.plot([xi, xi], [lo, hi], color=ink, lw=1.2, solid_capstyle="round")
        ax.set_xticks(x, [LABEL.get(m, m) for m in models], fontsize=8, color=ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("fraction of tasks solved", fontsize=9, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, fontsize=8, ncol=len(uniq))
    fig.text(0.01, 0.01, "bars: fraction of tasks; lines: 95% interval over tasks (one training run per model)",
             fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(args.out + ".png", dpi=200)
    print(f"\nwrote {args.out}.png and {args.out}.md")


if __name__ == "__main__":
    main()
