"""Draw the README's method and architecture figures as plain SVG.

    python docs/make_diagrams.py        # writes docs/media/method.svg and docs/media/architecture.svg

Hand-placed shapes, no plotting library, so the figures stay crisp at any size and are easy
to tweak: edit the coordinates below and rerun.
"""
import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "media"

INK, MUTED, LINE, PANEL, PANEL_EDGE = "#1f2328", "#57606a", "#8c959f", "#f6f8fa", "#d0d7de"
REAL, REAL_BG = "#2a78d6", "#eaf2fc"      # the real world: my desk, the phone
SIM, SIM_BG = "#d9622b", "#fdf0e8"        # the simulator
MODEL, MODEL_BG = "#178f64", "#e7f5ee"    # the learned model and the planner
TEE_SIM, GOAL = "#7d8fa6", "#7fd27f"
FONT = "Helvetica, Arial, sans-serif"


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{label}" font-family="{FONT}">\n'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
            f'<marker id="arrM" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MODEL}"/></marker>'
            f'<marker id="arrS" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SIM}"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def sub(s):
    """x_t+1 -> x with a real subscript (t+1); stops at spaces, brackets and commas."""
    return re.sub(r"_([\w+−]+)", r'<tspan baseline-shift="sub" font-size="72%">\1</tspan>', s)


def text(x, y, s, size=13, fill=INK, anchor="middle", weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    s = sub(s)
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}>{s}</text>')


def rect(x, y, w, h, fill="none", stroke=PANEL_EDGE, r=6, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def arrow(x1, y1, x2, y2, color=MUTED, sw=1.6, dash=None, marker="arr"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
            f'marker-end="url(#{marker})"/>')


def path(d, stroke=MUTED, sw=1.6, fill="none", dash=None, marker=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{a}{m} stroke-linecap="round" stroke-linejoin="round"/>'


def tee(cx, cy, deg, s=1.0, fill=REAL, stroke="none", sw=1.2, opacity=1.0):
    """The Push-T T: a 4 x 1 bar on a 1 x 3 stem (scaled), centred near its centroid."""
    a = f' opacity="{opacity}"' if opacity < 1 else ""
    return (f'<g transform="translate({cx},{cy}) rotate({deg}) scale({s})"{a}>'
            f'<rect x="-20" y="-16" width="40" height="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw / s}"/>'
            f'<rect x="-5" y="-6" width="10" height="28" fill="{fill}" stroke="{stroke}" stroke-width="{sw / s}"/></g>')


def arena(x, y, size):
    return rect(x, y, size, size, fill="#ffffff", stroke=LINE, r=2)


def caption(x, y, letter, s):
    return text(x, y, f'<tspan font-weight="bold">({letter})</tspan> {s}', size=13, fill=INK)


def panda(x, y, s=1.0, rod_to=None):
    """A schematic arm: base, two links, wrist, gripper holding a rod."""
    sh, el, wr = (x, y), (x + 34 * s, y + 30 * s), (x + 22 * s, y + 62 * s)
    out = [f'<rect x="{x - 16 * s}" y="{y - 12 * s}" width="{32 * s}" height="{16 * s}" rx="{4 * s}" fill="#9aa4ae"/>',
           f'<line x1="{sh[0]}" y1="{sh[1]}" x2="{el[0]}" y2="{el[1]}" stroke="#c9ced3" stroke-width="{13 * s}" stroke-linecap="round"/>',
           f'<line x1="{el[0]}" y1="{el[1]}" x2="{wr[0]}" y2="{wr[1]}" stroke="#c9ced3" stroke-width="{11 * s}" stroke-linecap="round"/>',
           f'<circle cx="{el[0]}" cy="{el[1]}" r="{7 * s}" fill="#9aa4ae"/>',
           f'<rect x="{wr[0] - 9 * s}" y="{wr[1]}" width="{18 * s}" height="{7 * s}" rx="{2 * s}" fill="#5c656e"/>']
    rx, ry = rod_to if rod_to else (wr[0], wr[1] + 34 * s)
    out.append(f'<line x1="{wr[0]}" y1="{wr[1] + 7 * s}" x2="{rx}" y2="{ry}" stroke="#2b4c8c" stroke-width="{5 * s}" stroke-linecap="round"/>')
    return "\n".join(out)


def table(x, y, w, h, inset=18, fill="#e6d8c0"):
    """A table top seen at an angle (trapezoid)."""
    return (f'<polygon points="{x},{y + h} {x + w},{y + h} {x + w - inset},{y} {x + inset},{y}" '
            f'fill="{fill}" stroke="#bfae92" stroke-width="1"/>')


def markers(x, y):
    sq = "".join(f'<rect x="{x + 4 + 9 * (i % 4)}" y="{y + 4 + 9 * (i // 4)}" width="6" height="6" fill="#222"/>'
                 for i in range(8))
    return f'<rect x="{x}" y="{y}" width="40" height="24" fill="#fff" stroke="#888" stroke-width="0.8"/>{sq}'


# ----------------------------------------------------------------------------------------
def _surface(x0, y0, w, h, n=17):
    """A small 3-D loss surface (oblique view, painter's order) with a gradient-descent path.

    Returns (svg pieces, projected start, projected minimum). The surface is a tilted bowl with
    a bump on the way down, so the descent path curves like a real optimizer's would.
    """
    import numpy as np
    g = np.array([0.38, -0.30])                                   # minimum: the goal latent

    def f(u, v):
        bowl = 0.55 * (u - g[0]) ** 2 + 0.9 * (v - g[1]) ** 2 + 0.25 * (u - g[0]) * (v - g[1])
        bump = 0.32 * np.exp(-((u + 0.15) ** 2 + (v - 0.25) ** 2) / 0.09)
        return bowl + bump

    def grad(u, v, e=1e-4):
        return np.array([(f(u + e, v) - f(u - e, v)) / (2 * e), (f(u, v + e) - f(u, v - e)) / (2 * e)])

    us = np.linspace(-1, 1, n)
    U, V = np.meshgrid(us, us, indexing="ij")
    Z = f(U, V)
    zmax = Z.max()
    A, B, C = w / 4.0, h * 0.15, h * 0.38                           # oblique projection scales
    cx, cy = x0 + w / 2, y0 + 2 * B + C + 2

    def proj(u, v, z):
        return cx + (u - v) * A, cy + (u + v) * B - (z / zmax) * C

    lo, hi = np.array([0x9f, 0xd4, 0xbb]), np.array([0xf6, 0xfa, 0xf8])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            zc = np.mean([Z[a, b] for a, b in corners])
            t = min(1.0, (zc / zmax) ** 0.6)
            col = "#%02x%02x%02x" % tuple((lo + (hi - lo) * t).astype(int))
            pts = " ".join("%.1f,%.1f" % proj(U[a, b], V[a, b], Z[a, b]) for a, b in corners)
            faces.append((U[i, j] + V[i, j], f'<polygon points="{pts}" fill="{col}" stroke="#a7cbb9" stroke-width="0.5"/>'))
    out = [svg_ for _, svg_ in sorted(faces, key=lambda q: q[0])]

    # plain gradient descent from a corner of the action space
    p, traj = np.array([-0.82, 0.86]), []
    for _ in range(60):
        traj.append(p.copy())
        p = p - 0.16 * grad(*p)
    traj.append(g)
    keep = [traj[k] for k in (0, 2, 4, 6, 9, 12, 16, 22, 30, 42)] + [g]
    pts = [proj(u, v, f(u, v)) for u, v in keep]
    out.append(path("M" + " L".join("%.1f,%.1f" % q for q in pts), MODEL, 1.8))
    for q in pts[1:-1]:
        out.append(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="2.8" fill="#ffffff" stroke="{MODEL}" stroke-width="1.5"/>')
    return out, pts[0], pts[-1]


def method():
    """One row: training data -> world model -> planning from a phone clip."""
    W, H = 1200, 300
    b = []
    top, ph = 40, 214
    A, B, C = (20, 270), (312, 380), (714, 466)    # (x, width) of the three panels
    for (x, w), bg in zip((A, B, C), (SIM_BG, MODEL_BG, "#f3f4f6")):
        b.append(rect(x, top, w, ph, fill=bg, stroke="none", r=10))
    mid = top + ph / 2

    # (a) training data: my pushes -> replayed and rendered on the Panda
    x, yy = A[0], top + 20
    b.append(text(x + A[1] / 2, top + 26, "my pushes, replayed in simulation", 12, SIM, weight="bold"))
    b.append(table(x + 12, yy + 62, 104, 66, inset=14))
    b.append(tee(x + 66, yy + 92, -24, 0.62, REAL))
    b.append(f'<line x1="{x + 114}" y1="{yy + 122}" x2="{x + 84}" y2="{yy + 102}" stroke="#e8b48f" stroke-width="7" stroke-linecap="round"/>')
    b.append(f'<circle cx="{x + 84}" cy="{yy + 102}" r="3.5" fill="#38b24a"/>')
    b.append(text(x + 64, yy + 152, "phone video", 11, REAL))
    b.append(arrow(x + 122, yy + 96, x + 156, yy + 96))
    b.append(text(x + 139, yy + 88, "replay", 10, MUTED))
    b.append(f'<rect x="{x + 160}" y="{yy + 50}" width="100" height="88" fill="#e9e6e1"/>')
    b.append(f'<polygon points="{x + 168},{yy + 134} {x + 252},{yy + 134} {x + 244},{yy + 96} {x + 176},{yy + 96}" fill="#ffffff"/>')
    b.append(tee(x + 228, yy + 118, 18, 0.55, TEE_SIM))
    b.append(panda(x + 194, yy + 58, 0.62, rod_to=(x + 208, yy + 120)))
    b.append(text(x + 210, yy + 152, "Panda frames", 11, SIM))
    b.append(text(x + A[1] / 2, top + ph - 10, "my demos → simulated (action, frame) episodes", 10, MUTED))

    # (b) world model: online branch (o_t -> z_t -> predictor -> z^_t+1) and target branch (o_t+1 -> z_t+1)
    x = B[0]
    r1, r2 = top + 92, top + 168                       # row centres: prediction, target
    b.append(text(x + B[1] / 2, top + 26, "latent world model", 12, MODEL, weight="bold"))

    def frame(fx, fy, deg):
        return (f'<rect x="{fx}" y="{fy}" width="34" height="34" fill="#e9e6e1" stroke="#c4bfb7" rx="2"/>'
                + tee(fx + 17, fy + 16, deg, 0.36, TEE_SIM))

    for ry, lab, deg in ((r1, "o_t", 10), (r2, "o_t+1", 26)):
        b.append(frame(x + 14, ry - 17, deg))
        b.append(text(x + 31, ry + 31, lab, 11, INK, italic=True))
        b.append(arrow(x + 50, ry, x + 64, ry))
    b.append(rect(x + 66, r1 - 22, 92, 44, fill="#ffffff", stroke=MODEL, r=6))
    b.append(text(x + 112, r1 - 3, "encoder", 12))
    b.append(text(x + 112, r1 + 12, "DINOv2 + proj.", 10, MUTED))
    b.append(rect(x + 66, r2 - 22, 92, 44, fill="#ffffff", stroke=MODEL, r=6, dash="4 3"))
    b.append(text(x + 112, r2 - 3, "same encoder", 12))
    b.append(text(x + 112, r2 + 12, "stop-gradient", 10, MUTED))
    b.append(f'<line x1="{x + 112}" y1="{r1 + 23}" x2="{x + 112}" y2="{r2 - 23}" stroke="{MODEL}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    b.append(text(x + 118, (r1 + r2) / 2 + 4, "shared", 9, MUTED, "start"))
    # online branch
    b.append(arrow(x + 160, r1, x + 200, r1, MODEL, marker="arrM"))
    b.append(text(x + 180, r1 - 7, "z_t", 13, MODEL, italic=True))
    b.append(rect(x + 202, r1 - 22, 92, 44, fill="#ffffff", stroke=MODEL, r=6))
    b.append(text(x + 248, r1 - 3, "predictor", 12))
    b.append(text(x + 248, r1 + 12, "ViT, 3 frames", 10, MUTED))
    b.append(text(x + 248, top + 52, "actions a_t", 11, MODEL, italic=True))
    b.append(arrow(x + 248, top + 57, x + 248, r1 - 24, MODEL, marker="arrM"))
    b.append(arrow(x + 296, r1, x + 322, r1, MODEL, marker="arrM"))
    b.append(text(x + 348, r1 + 5, "ẑ_t+1", 14, MODEL, italic=True))
    # target branch
    b.append(arrow(x + 160, r2, x + 322, r2))
    b.append(text(x + 240, r2 - 7, "target", 10, MUTED))
    b.append(text(x + 348, r2 + 5, "z_t+1", 14, INK, italic=True))
    # prediction loss between the two
    b.append(f'<line x1="{x + 348}" y1="{r1 + 12}" x2="{x + 348}" y2="{r2 - 16}" stroke="{SIM}" stroke-width="1.6" '
             f'marker-start="url(#arrS)" marker-end="url(#arrS)"/>')
    b.append(text(x + 340, (r1 + r2) / 2 + 4, "ℒ_pred", 13, SIM, "end", italic=True))
    b.append(text(x + B[1] / 2, top + ph - 10, "+ regularizers on the path of z (Figure 3)", 10, MUTED))

    # (c) planning from a phone clip: stacked start/goal -> descent on the latent objective -> Panda
    x = C[0]
    b.append(text(x + C[1] / 2, top + 26, "planning from a phone clip of the real T", 12, INK, weight="bold"))
    for i, (lab, deg) in enumerate([("start", -24), ("goal", 30)]):
        ty = top + 46 + i * 78
        b.append(table(x + 14, ty, 72, 46, inset=9))
        b.append(tee(x + 50, ty + 23, deg, 0.48, REAL))
        b.append(text(x + 50, ty + 62, lab, 11, REAL))
    b.append(arrow(x + 89, mid, x + 103, mid))
    b.append(text(x + 96, mid - 8, "track", 9, MUTED))
    # the trained world model, small: encoder (images -> z) and predictor (z, actions -> z^)
    wx, wy, ww, wh = x + 104, mid - 46, 66, 92
    b.append(rect(wx, wy, ww, wh, fill=MODEL_BG, stroke=MODEL, r=6, sw=1.2))
    b.append(text(wx + ww / 2, wy + 13, "world model", 9, MODEL, weight="bold"))
    b.append(rect(wx + 6, wy + 20, ww - 12, 24, fill="#ffffff", stroke=MODEL, r=4))
    b.append(text(wx + ww / 2, wy + 36, "encoder", 10))
    b.append(arrow(wx + ww / 2, wy + 45, wx + ww / 2, wy + 57, MODEL, 1.2, marker="arrM"))
    b.append(rect(wx + 6, wy + 59, ww - 12, 24, fill="#ffffff", stroke=MODEL, r=4))
    b.append(text(wx + ww / 2, wy + 75, "predictor", 10))
    sx, sw_, sh = x + 196, 140, 108
    # encoder gives z_0 and z_goal; the predictor rolls the actions forward; gradients flow back
    b.append(arrow(wx + ww + 2, wy + 32, sx + 6, wy + 32, MODEL, 1.2, marker="arrM"))
    b.append(text(wx + ww + 13, wy + 26, "z", 11, MODEL, italic=True))
    b.append(f'<line x1="{wx + ww + 2}" y1="{wy + 71}" x2="{sx + 6}" y2="{wy + 71}" stroke="{MODEL}" stroke-width="1.2" '
             f'stroke-dasharray="3 2" marker-start="url(#arrM)" marker-end="url(#arrM)"/>')
    b.append(text(wx + ww + 14, wy + 64, "ẑ_H", 11, MODEL, italic=True))
    surf, p0, pg = _surface(sx, top + 46, sw_, sh)
    b += surf
    b.append(f'<circle cx="{p0[0]:.1f}" cy="{p0[1]:.1f}" r="5" fill="{MODEL}"/>')
    b.append(text(p0[0] - 2, p0[1] - 10, "z_0", 12, MODEL, italic=True))
    b.append(f'<path d="M{pg[0] - 6:.1f},{pg[1]:.1f} L{pg[0] + 6:.1f},{pg[1]:.1f} M{pg[0]:.1f},{pg[1] - 6:.1f} '
             f'L{pg[0]:.1f},{pg[1] + 6:.1f}" stroke="{INK}" stroke-width="2.2"/>')
    b.append(text(pg[0] + 10, pg[1] + 18, "z_goal", 12, INK, "start", italic=True))
    b.append(text(sx + sw_ / 2 - 14, top + 172, "gradient descent on actions a<tspan baseline-shift='sub' font-size='72%'>1:H</tspan>", 10, MODEL))
    b.append(text(sx + sw_ / 2 - 14, top + 185, "through the predictor, to minimise ‖ẑ_H − z_goal‖", 10, MODEL))
    b.append(arrow(sx + sw_ + 2, mid, sx + sw_ + 24, mid))
    b.append(text(sx + sw_ + 13, mid - 8, "act", 10, MUTED))
    px = sx + sw_ + 44
    b.append(f'<rect x="{px - 18}" y="{mid - 50}" width="96" height="94" rx="3" fill="#e9e6e1"/>')
    b.append(f'<polygon points="{px - 10},{mid + 40} {px + 70},{mid + 40} {px + 62},{mid + 4} {px - 2},{mid + 4}" fill="#ffffff"/>')
    b.append(tee(px + 44, mid + 24, 30, 0.5, TEE_SIM))
    b.append(tee(px + 44, mid + 24, 30, 0.5, "none", GOAL, 1.4))
    b.append(panda(px + 14, mid - 42, 0.6, rod_to=(px + 26, mid + 22)))
    yb = top + ph - 10
    b.append(path(f"M{px + 30},{mid + 46} L{px + 30},{yb} L{wx + ww / 2},{yb} L{wx + ww / 2},{wy + wh + 2}", MUTED, 1.2,
                  dash="4 3", marker="arr"))
    b.append(text(px + 36, mid + 60, "goal", 10, MUTED, "start"))
    b.append(text(px + 36, mid + 72, "reached?", 10, MUTED, "start"))
    b.append(f'<rect x="{x + 200}" y="{yb - 7}" width="150" height="14" fill="#f3f4f6"/>')
    b.append(text(x + 275, yb + 4, "no: observe again, replan", 10, MUTED))

    # flow between panels
    b.append(arrow(A[0] + A[1] + 2, mid, B[0] - 2, mid))
    b.append(text((A[0] + A[1] + B[0]) / 2, mid - 8, "train", 10, MUTED))
    b.append(arrow(B[0] + B[1] + 2, mid, C[0] - 2, mid, MODEL, marker="arrM"))
    for (xx, w), letter, cap in zip((A, B, C), "abc", ("Training data", "World model", "Planning (MPC)")):
        b.append(caption(xx + w / 2, top + ph + 26, letter, cap))
    return svg(W, H, "\n".join(b), "Method in one row: my pushes, replayed in simulation and rendered with a Panda, "
               "train a latent world model whose target latents come from the same encoder; at test time a phone clip "
               "of a start and a goal pose sets the task and gradient descent on the latent objective picks the "
               "Panda's actions, replanning until the goal is reached.")


# ----------------------------------------------------------------------------------------
def architecture():
    """Three context frames (drawn as a stack) -> latents z (visual patches | proprio embedding)
    -> causal predictor, conditioned on the actions -> the next latent of each frame, against the
    same encoder's latents of the following frames (stop-gradient).

    In the code the action embedding is concatenated to every patch before the predictor, and
    the predictor's output action slots are ignored by the loss and overwritten during
    rollouts, so the figure shows actions as a predictor input and z as visual + proprio."""
    W, H = 940, 280
    b = []
    y = 130                                  # centre line
    PROP, ACT = REAL, "#8a5cc2"
    D = 7                                    # offset between stacked copies

    def frames(x, degs, lab):
        out = []
        for i, deg in enumerate(degs):       # back to front
            k = 2 - i
            fx, fy = x + k * D, y - 26 - k * D
            out.append(rect(fx, fy, 52, 52, fill="#e9e6e1", stroke="#b9b3aa", r=2))
            out.append(tee(fx + 26, fy + 24, deg, 0.5, TEE_SIM))
        out.append(text(x + 33, y + 44, lab, 11, INK, italic=True))
        return "".join(out)

    def tokens(x, lab, color=MODEL, above=False):
        out = []
        for i in range(3):
            k = 2 - i
            tx, ty = x + k * D, y - 20 - k * D
            out.append(rect(tx, ty, 46, 40, fill="#ffffff", stroke=color, r=2, sw=1.3))
            for j in range(1, 4):
                out.append(f'<line x1="{tx + 11.5 * j}" y1="{ty}" x2="{tx + 11.5 * j}" y2="{ty + 40}" stroke="{color}" stroke-width="0.5" opacity="0.6"/>')
                out.append(f'<line x1="{tx}" y1="{ty + 10 * j}" x2="{tx + 46}" y2="{ty + 10 * j}" stroke="{color}" stroke-width="0.5" opacity="0.6"/>')
            out.append(rect(tx + 47, ty, 8, 40, fill=PROP, stroke="none", r=1))
        out.append(text(x + 34, y - 44 if above else y + 44, lab, 12, color, italic=True))
        return "".join(out)

    b.append(text(20, 26, "WORLD MODEL (training)", 12, MUTED, "start", "bold"))
    # context frames -> encoder
    b.append(frames(18, (8, 16, 24), "o_t−2 … o_t"))
    b.append(arrow(90, y, 112, y))
    b.append(rect(114, y - 52, 104, 104, fill="#eef1f4", stroke=LINE, r=8))
    b.append(text(166, y - 24, "DINOv2", 13, INK, weight="bold"))
    b.append(text(166, y - 9, "ViT-S/14, frozen", 9, MUTED))
    b.append(text(166, y + 12, "+ projector", 12, MODEL))
    b.append(text(166, y + 28, "14×14×8 patches", 9, MUTED))
    b.append(arrow(220, y, 242, y, MODEL, marker="arrM"))
    b.append(tokens(244, "z_t−2 … z_t", above=True))
    # proprio and action inputs
    b.append(rect(150, 216, 128, 40, fill="#ffffff", stroke=PROP, r=6))
    b.append(text(214, 233, "pusher xy + velocity", 11, PROP))
    b.append(text(214, 248, "proprio embedding", 9, MUTED))
    b.append(path(f"M214,216 L214,196 L295,196 L295,{y + 22}", PROP, 1.3))
    b.append(f'<polygon points="291,{y + 27} 295,{y + 21} 299,{y + 27}" fill="{PROP}"/>')
    b.append(rect(344, 216, 132, 40, fill="#ffffff", stroke=ACT, r=6))
    b.append(text(410, 233, "actions a_t−2 … a_t", 11, ACT, italic=True))
    b.append(text(410, 248, "5 env steps each, embedded", 9, MUTED))
    b.append(f'<line x1="410" y1="216" x2="410" y2="{y + 58}" stroke="{ACT}" stroke-width="1.4"/>')
    b.append(f'<polygon points="405,{y + 61} 410,{y + 53} 415,{y + 61}" fill="{ACT}"/>')
    # predictor
    b.append(arrow(330, y, 356, y, MODEL, marker="arrM"))
    b.append(rect(358, y - 52, 104, 104, fill=MODEL_BG, stroke=MODEL, r=8))
    b.append(text(410, y - 10, "predictor", 13, INK, weight="bold"))
    b.append(text(410, y + 8, "ViT, causal", 10, MUTED))
    b.append(text(410, y + 22, "over 3 frames", 10, MUTED))
    b.append(arrow(464, y, 488, y, MODEL, marker="arrM"))
    b.append(tokens(490, "ẑ_t−1 … ẑ_t+1"))
    # loss
    b.append(f'<line x1="584" y1="{y}" x2="640" y2="{y}" stroke="{SIM}" stroke-width="1.6" '
             f'marker-start="url(#arrS)" marker-end="url(#arrS)"/>')
    b.append(text(612, y - 42, "<tspan font-style='italic'>ℒ_pred</tspan>", 13, SIM))
    b.append(text(612, y - 22, "MSE", 9, SIM))
    # targets: same encoder, stop-gradient
    b.append(tokens(646, "z_t−1 … z_t+1", INK))
    b.append(arrow(758, y, 734, y))
    b.append(rect(760, y - 52, 84, 104, fill="#ffffff", stroke=LINE, r=8, dash="4 3"))
    b.append(text(802, y - 8, "same", 12))
    b.append(text(802, y + 8, "encoder", 12))
    b.append(text(802, y + 24, "+ proprio", 9, MUTED))
    b.append(text(802, y + 37, "stop-grad", 9, MUTED))
    b.append(arrow(866, y, 846, y))
    b.append(frames(868, (16, 24, 32), "o_t−1 … o_t+1"))
    # decoder and regularizer notes
    b.append(rect(496, 216, 120, 40, fill="#ffffff", stroke=LINE, r=8, dash="4 3"))
    b.append(text(556, 233, "decoder", 12))
    b.append(text(556, 248, "viz only, stop-grad", 9, MUTED))
    b.append(arrow(530, y + 52, 540, 214, LINE, 1.2, dash="4 3"))
    b.append(text(650, 232, "regularizers act on the latent", 10, MUTED, "start"))
    b.append(text(650, 246, "path z_t−2 → z_t+1 (Figure 3)", 10, MUTED, "start"))
    return svg(W, H, "\n".join(b), "World model: three frames pass through a frozen DINOv2 encoder and a trainable projector; "
               "each frame's patches are joined with an embedding of the pusher's position and velocity; a causal ViT, "
               "conditioned on the actions, predicts the next latent of each frame, matched against the same "
               "encoder's latents of the following frames with a stop-gradient.")


# ----------------------------------------------------------------------------------------
def regularizers():
    """The planning landscape and the latent path from start to goal, three ways.

    Filled contours: the planner's objective (latent distance to the goal) around the path.
    No regularizer: a warped landscape and a winding path with uneven steps.
    Straightening: a smoother landscape; the path still bends once, steps still uneven.
    + pacing: same path, but steps shrink where it turns and lengthen where it is straight.
    """
    import numpy as np
    from matplotlib.contour import QuadContourSet  # noqa: F401  (contours are computed, not drawn)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = 1140, 330
    PW, PH, TOP = 355, 230, 44
    S, G = np.array([46.0, 186.0]), np.array([300.0, 68.0])
    b = []

    def field(warp, seed):
        xs, ys = np.linspace(0, PW, 180), np.linspace(0, PH, 120)
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X - G[0], Y - G[1]
        r = np.sqrt((dx / 1.15) ** 2 + (dy / 0.8) ** 2)
        rng = np.random.default_rng(seed)
        bumps = sum(a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2))
                    for a, cx, cy, w in zip(rng.uniform(-60, 60, 6), rng.uniform(40, 320, 6),
                                            rng.uniform(30, 200, 6), rng.uniform(25, 55, 6)))
        ripple = 18 * np.sin(X / 37 + 0.6) * np.cos(Y / 29)
        return X, Y, r + warp * (bumps + ripple)

    def contour_svg(X, Y, Z, x0, clip):
        levels = np.linspace(Z.min(), np.percentile(Z, 96), 9)
        fig = plt.figure()
        cs = plt.contourf(X, Y, Z, levels=levels, extend="max")
        lines = plt.contour(X, Y, Z, levels=levels)
        out = []
        ramp = ["#cfe3da", "#d8e9e1", "#e1eee8", "#e8f2ed", "#eef5f1", "#f2f7f4", "#f5f9f7", "#f8fbf9", "#fafcfb", "#fbfcfc"]
        for i, pth in enumerate(cs.get_paths()):
            d = []
            for (vx, vy), code in zip(pth.vertices, pth.codes if pth.codes is not None else [2] * len(pth.vertices)):
                if code == 1:
                    d.append(f"M{x0 + vx:.1f},{TOP + vy:.1f}")
                elif code == 79:
                    d.append("Z")
                else:
                    d.append(f"L{x0 + vx:.1f},{TOP + vy:.1f}")
            if d:
                out.append(f'<path d="{" ".join(d)}" fill="{ramp[min(i, len(ramp) - 1)]}" fill-rule="evenodd" clip-path="url(#{clip})"/>')
        for segs in lines.allsegs:
            for seg in segs:
                if len(seg) > 2:
                    d = "M" + " L".join(f"{x0 + vx:.1f},{TOP + vy:.1f}" for vx, vy in seg)
                    out.append(f'<path d="{d}" fill="none" stroke="#a9c7ba" stroke-width="0.8" clip-path="url(#{clip})"/>')
        plt.close(fig)
        return out

    def bezier(P, n=400):
        t = np.linspace(0, 1, n)[:, None]
        return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])

    def winding(n=400):
        t = np.linspace(0, 1, n)
        base = S[None] + t[:, None] * (G - S)[None]
        nrm = np.array([-(G - S)[1], (G - S)[0]]) / np.linalg.norm(G - S)
        off = 34 * np.sin(2.4 * np.pi * t) * np.sin(np.pi * t) + 10 * np.sin(6.5 * np.pi * t) * np.sin(np.pi * t)
        return base + off[:, None] * nrm[None]

    def arclen(c):
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]

    def at(c, s_target):
        s = arclen(c)
        return np.stack([np.interp(s_target, s, c[:, 0]), np.interp(s_target, s, c[:, 1])], 1)

    def curvature(c):
        d1 = np.gradient(c, axis=0)
        d2 = np.gradient(d1, axis=0)
        return np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-9)

    uneven = np.array([0, .03, .17, .22, .41, .47, .63, .70, .88, .93, 1.0])
    bent = bezier([S, np.array([150.0, 214.0]), np.array([150.0, 52.0]), G])

    def paced(c, n=14, gain=900.0):
        s, k = arclen(c), curvature(c)
        dens = 1 + gain * np.convolve(k, np.ones(25) / 25, mode="same")
        cum = np.r_[0, np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(s))]
        targets = np.linspace(0, cum[-1], n)
        return at(c, np.interp(targets, cum, s))

    panels = [
        ("No regularizer", "a warped landscape and a winding, unevenly stepped path",
         field(1.0, 3), winding(), lambda c: at(c, uneven * arclen(c)[-1])),
        ("Straightening", "smoother, but a bend remains and the steps stay uneven",
         field(0.35, 3), bent, lambda c: at(c, uneven * arclen(c)[-1])),
        ("Straightening + pacing", "smaller steps through the bend, longer ones on the straights",
         field(0.35, 3), bent, paced),
    ]
    defs = []
    for k, (title, sub_, (X, Y, Z), curve, steps) in enumerate(panels):
        x0 = 20 + k * 375
        clip = f"panel{k}"
        defs.append(f'<clipPath id="{clip}"><rect x="{x0}" y="{TOP}" width="{PW}" height="{PH}" rx="10"/></clipPath>')
        b += contour_svg(X, Y, Z, x0, clip)
        b.append(rect(x0, TOP, PW, PH, fill="none", stroke=PANEL_EDGE, r=10))
        b.append(text(x0 + PW / 2, 30, title, 14, INK, weight="bold"))
        b.append(path("M" + " L".join(f"{x0 + x:.1f},{TOP + y:.1f}" for x, y in curve[::4]) + f" L{x0 + curve[-1][0]:.1f},{TOP + curve[-1][1]:.1f}",
                      MODEL, 2))
        pts = steps(curve)
        for x, y in pts[1:-1]:
            b.append(f'<circle cx="{x0 + x:.1f}" cy="{TOP + y:.1f}" r="4" fill="#ffffff" stroke="{MODEL}" stroke-width="2"/>')
        b.append(f'<circle cx="{x0 + S[0]:.1f}" cy="{TOP + S[1]:.1f}" r="6" fill="{MODEL}"/>')
        b.append(f'<path d="M{x0 + G[0] - 7},{TOP + G[1]} L{x0 + G[0] + 7},{TOP + G[1]} M{x0 + G[0]},{TOP + G[1] - 7} '
                 f'L{x0 + G[0]},{TOP + G[1] + 7}" stroke="{INK}" stroke-width="2.2"/>')
        b.append(text(x0 + S[0], TOP + S[1] + 24, "start", 11, MUTED))
        b.append(text(x0 + G[0] + 4, TOP + G[1] - 14, "goal", 11, MUTED))
        if k == 2:
            b.append(text(x0 + 168, TOP + 202, "more, smaller steps", 11, SIM, "start"))
            b.append(text(x0 + 168, TOP + 216, "where the path turns", 11, SIM, "start"))
            b.append(path(f"M{x0 + 164},{TOP + 199} Q{x0 + 140},{TOP + 200} {x0 + 112},{TOP + 186}", SIM, 1.3, marker="arrS"))
            b.append(path(f"M{x0 + 246},{TOP + 192} Q{x0 + 262},{TOP + 160} {x0 + 248},{TOP + 116}", SIM, 1.3, marker="arrS"))
        b.append(text(x0 + PW / 2, TOP + PH + 24, sub_, 12, INK))
    body = "<defs>" + "".join(defs) + "</defs>\n" + "\n".join(b)
    return svg(W, H, body, "The planning landscape and the latent path from start to goal: without regularization the "
               "landscape is warped and the path winds with uneven steps; straightening smooths it but a bend "
               "remains; pacing puts more, smaller steps through the bend.")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "method.svg").write_text(method())
    (OUT / "architecture.svg").write_text(architecture())
    (OUT / "regularizers.svg").write_text(regularizers())
    print(f"wrote method.svg, architecture.svg and regularizers.svg in {OUT}")
