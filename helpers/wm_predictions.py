"""How well do the world models predict? Open-loop rollouts on the held-out demos.

    python helpers/wm_predictions.py --models aug both branch random noreg raw
    python helpers/wm_predictions.py --models both random --horizon 10 --show both random

For every held-out demo and several start points in it, each model gets the first 3 real frames
and then only my recorded actions, and predicts the next --horizon latents by feeding its own
predictions back in (the same rollout the planner uses). Every prediction is compared with the
latent of the real frame at that time:

  - latent error: mean squared error of the visual latent, divided by the error of a
    "nothing moves" baseline that copies the last context latent. Below 1 means the model
    predicts motion better than assuming the scene stands still.
  - the decoder (trained for viewing only) turns predictions into images, for the figure.

Writes:
  <out>.png      real frames vs each --show model's decoded predictions, one held-out window
  <out>_error.png latent error vs prediction horizon, one line per model
  <out>.md       the error table (1, half and full horizon), also written into README.md
                 between the <!-- wm:start --> and <!-- wm:end --> markers

Needs CKPT and DATASET_DIR as for planning (CKPT/humanai_<model>/hydra.yaml). One model is on
the GPU at a time; --batch sets how many windows are rolled out together (lower it on small GPUs).
"""
import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import hydra  # noqa: E402
from plan import load_model  # noqa: E402  (also quiets gym/d4rl and registers the resolvers)

LABEL = {"noreg": "no regularizer", "aug": "straightening", "pacing": "pacing", "both": "straightening + pacing",
         "random": "random play", "branch": "branch rollouts",
         "raw": "my demos only"}
MODEL_ORDER = ["noreg", "aug", "pacing", "both", "random", "branch", "raw"]


def load(ckpt_root, name, device):
    path = Path(ckpt_root) / f"humanai_{name}"
    cfg = OmegaConf.load(path / "hydra.yaml")
    ckpt = path / "checkpoints" / "model_latest.pth"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    model = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
    model.eval()
    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    return model, traj["valid"], cfg


def windows(dset, fs, n_ctx, horizon, stride):
    """(episode, start) pairs whose window of n_ctx + horizon model steps fits in the demo."""
    need = (n_ctx + horizon) * fs + 1
    out = []
    for e in range(len(dset)):
        L = dset.get_seq_length(e)
        out += [(e, s) for s in range(0, L - need + 1, stride)]
    return out


def gather(dset, wins, fs, n_ctx, horizon, cache):
    """Real frames, proprio and model-step actions for a list of windows."""
    vis, pro, act, tpos = [], [], [], []
    T = n_ctx + horizon
    for e, s in wins:
        if e not in cache:
            cache.clear()
            cache[e] = dset[e]
        obs, a, state, _ = cache[e]
        idx = [s + k * fs for k in range(T + 1)]
        vis.append(obs["visual"][idx])
        pro.append(obs["proprio"][idx])
        act.append(rearrange(a[s:s + T * fs], "(t f) d -> t (f d)", f=fs))
        tpos.append(np.asarray(state)[idx][:, 2:5])
    return torch.stack(vis), torch.stack(pro), torch.stack(act), np.stack(tpos)


@torch.no_grad()
def evaluate(model, dset, cfg, wins, horizon, batch, device, keep=None):
    fs, n_ctx = cfg.frameskip, cfg.num_hist
    errs, base, kept, cache = [], [], None, {}
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        vis, pro, act, _ = gather(dset, chunk, fs, n_ctx, horizon, cache)
        vis, pro, act = vis.to(device), pro.to(device), act.to(device).float()
        obs0 = {"visual": vis[:, :n_ctx], "proprio": pro[:, :n_ctx].float()}
        z_obs, _ = model.rollout(obs0, act)                      # n_ctx + horizon + 1 latents
        pred = z_obs["visual"][:, n_ctx:]                         # predicted steps 1..horizon (+1 extra)
        real = model.encode_obs({"visual": vis, "proprio": pro.float()})["visual"][:, n_ctx:]
        h = horizon                                               # the rollout gives one extra step; drop it
        pred, real = pred[:, :h], real[:, :h]
        last = model.encode_obs({"visual": vis[:, n_ctx - 1:n_ctx], "proprio": pro[:, n_ctx - 1:n_ctx].float()})["visual"]
        errs.append(((pred - real) ** 2).mean(dim=(2, 3)).cpu())
        base.append(((last - real) ** 2).mean(dim=(2, 3)).cpu())
        if keep is not None and kept is None and keep in chunk and model.decoder is not None:
            j = chunk.index(keep)
            dec_pred, _ = model.decode_obs({"visual": pred[j:j + 1], "proprio": None})
            kept = {"real": vis[j, n_ctx:n_ctx + h].cpu(), "pred": dec_pred["visual"][0].cpu(),
                    "ctx": vis[j, n_ctx - 1].cpu()}
    errs, base = torch.cat(errs).numpy(), torch.cat(base).numpy()      # (windows, horizon)
    return errs, base, kept


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--show", nargs="+", help="models whose decoded predictions go in the figure (default: first two)")
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", str(Path.home() / "wm_ckpts/checkpoints/test")))
    ap.add_argument("--horizon", type=int, default=10, help="model steps to predict (0.5 s each)")
    ap.add_argument("--stride", type=int, default=25, help="env steps between window starts in a demo")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="docs/media/wm_predictions")
    args = ap.parse_args()
    show = args.show or args.models[:2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results, frames, keep, wins = {}, {}, None, None
    for name in args.models:
        print(f"[{name}] loading", flush=True)
        model, dset, cfg = load(args.ckpt, name, device)
        if wins is None:
            fs, n_ctx = cfg.frameskip, cfg.num_hist
            wins = windows(dset, fs, n_ctx, args.horizon, args.stride)
            # the figure's window: the one where the T moves most over the horizon
            if not wins:
                raise SystemExit(f"no held-out demo is {(n_ctx + args.horizon) * fs + 1} steps long; lower --horizon")
            tp = []
            cache = {}
            for w in wins:
                _, _, _, t = gather(dset, [w], fs, n_ctx, args.horizon, cache)
                tp.append(np.linalg.norm(t[0, -1, :2] - t[0, n_ctx - 1, :2]))
            keep = wins[int(np.argmax(tp))]
            print(f"{len(wins)} windows from {len(dset)} held-out demos; figure window: demo {keep[0]}, step {keep[1]}", flush=True)
        errs, base, kept = evaluate(model, dset, cfg, wins, args.horizon, args.batch, device,
                                    keep if name in show else None)
        results[name] = (errs, base)
        if kept is not None:
            frames[name] = kept
        print(f"[{name}] relative latent error at 1 / {args.horizon} steps: "
              f"{errs[:, 0].mean() / base[:, 0].mean():.2f} / {errs[:, -1].mean() / base[:, -1].mean():.2f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dc"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [m for m in MODEL_ORDER if m in results] + [m for m in results if m not in MODEL_ORDER]

    # table
    H = next(iter(results.values()))[0].shape[1]
    cols = sorted({1, max(1, H // 2), H})
    lines = ["| model | " + " | ".join(f"{c * 0.5:g} s ahead" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    rel = {m: results[m][0].mean(0) / results[m][1].mean(0) for m in names}
    for m in names:
        lines.append(f"| {LABEL.get(m, m)} | " + " | ".join(f"{rel[m][c - 1]:.2f}" for c in cols) + " |")
    note = (f"\nLatent prediction error divided by the error of assuming nothing moves (lower is better, below 1 "
            f"beats standing still), over {len(wins)} windows from the held-out demos.\n")
    Path(str(out) + ".md").write_text("\n".join(lines) + "\n" + note)
    readme = ROOT / "README.md"
    txt = readme.read_text()
    if "<!-- wm:start -->" in txt and "<!-- wm:end -->" in txt:
        a, b = txt.index("<!-- wm:start -->"), txt.index("<!-- wm:end -->")
        txt = txt[:a] + "<!-- wm:start -->\n" + "\n".join(lines) + "\n\n" + note.strip() + "\n" + txt[b:]
        readme.write_text(txt)
        print("updated the prediction table in README.md")
    print("\n" + "\n".join(lines) + note)

    # error vs horizon
    fig, ax = plt.subplots(figsize=(6, 3.4))
    t = np.arange(1, H + 1) * 0.5
    for m in names:
        ax.plot(t, rel[m], marker="o", ms=3, lw=1.6, label=LABEL.get(m, m))
    ax.axhline(1, color=muted, lw=1, ls="--")
    ax.text(t[len(t) // 2], 1.02, "nothing moves", color=muted, fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("seconds predicted ahead (my recorded actions)", fontsize=9, color=muted)
    ax.set_ylabel("latent error / standing-still error", fontsize=9, color=muted)
    ax.set_ylim(0, max(1.15, float(max(r.max() for r in rel.values())) * 1.05))
    ax.grid(axis="y", color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out) + "_error.png", dpi=200)
    plt.close(fig)

    # real vs predicted frames
    if frames:
        steps = [k for k in (1, 2, 4, 6, 8, 10) if k <= H]
        rows = ["real"] + [m for m in show if m in frames]
        fig, axes = plt.subplots(len(rows), len(steps) + 1, figsize=(1.6 * (len(steps) + 1), 1.75 * len(rows)))
        axes = np.atleast_2d(axes)
        first = frames[rows[1]]
        for r, row in enumerate(rows):
            axes[r, 0].imshow(to_img(first["ctx"]))
            for c, k in enumerate(steps):
                img = first["real"][k - 1] if row == "real" else frames[row]["pred"][k - 1]
                axes[r, c + 1].imshow(to_img(img))
                if r == 0:
                    axes[r, c + 1].set_title(f"+{k * 0.5:g} s", fontsize=9, color=ink)
            axes[r, 0].set_ylabel("real frames" if row == "real" else LABEL.get(row, row), fontsize=8, color=ink,
                                  rotation=0, ha="right", va="center")
            if r == 0:
                axes[r, 0].set_title("last real input", fontsize=9, color=ink)
        for a in axes.flat:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color(grid)
        fig.tight_layout()
        fig.savefig(str(out) + ".png", dpi=160)
        plt.close(fig)
    print(f"\nwrote {out}.png, {out}_error.png and {out}.md")


if __name__ == "__main__":
    main()
