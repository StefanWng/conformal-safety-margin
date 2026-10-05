# -*- coding: utf-8 -*-
"""Method overview figure: from the fixed policy to the certified safety margin.

Two rows inside one method box (stages 1-3 on top; the expected-spread test and
its split into group refinement or the single offset below), with the input and
the output outside the box. Writes figures/fig_method.pdf and .png (vector icons).
Not yet included in the paper. Run with the csc249 environment.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge, Polygon, Rectangle,
                                PathPatch, FancyArrowPatch)
from matplotlib.path import Path

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6})

NAVY = "#1B3358"                        # every piece of text
GREY = "#7A8591"
SLATE = "#44546A"
# (panel fill, accent) for each stage, in the palette of the other figures
P_IN = ("#EEF2F6", SLATE)
P1 = ("#E3EEF8", "#0072B2")
P2 = ("#FCEBDF", "#D55E00")
P3 = ("#E1F3EB", "#009E73")
PC = ("#FFF6DD", "#B07D00")
P4 = ("#F2E6EF", "#A0457A")
P_OUT = ("#E8EDF7", "#2F4B8A")
BOX = ("#F7F9FB", "#9AA8B8")            # the method box around stages 1-4

W_FIG, H_FIG = 5.8, 3.62
fig = plt.figure(figsize=(W_FIG, H_FIG))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_FIG); ax.set_ylim(0, H_FIG); ax.set_aspect("equal"); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=0.8, r=0.05, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r,
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(x, y, s, fs=5.5, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    kw.setdefault("linespacing", 1.22); kw.setdefault("zorder", 6)
    ax.text(x, y, s, fontsize=fs, color=NAVY, **kw)


def panel(x, y, w, h, title, pal, num=None, ty=None):
    fc, ec = pal
    rbox(x, y, w, h, fc, ec, lw=0.9, r=0.06, z=2)
    text(x + w / 2, y + h - 0.14 if ty is None else ty, title, fs=6.6, weight="bold",
         linespacing=1.05)
    if num is not None:
        ax.add_patch(Circle((x + 0.02, y + h - 0.02), 0.07, fc=ec, ec="white", lw=0.8, zorder=7))
        ax.text(x + 0.02, y + h - 0.022, str(num), ha="center", va="center", fontsize=6.0,
                color="white", weight="bold", zorder=8)


def card(cx, cy, w, h, body, pal, head=None, fs=6.0):
    rbox(cx - w / 2, cy - h / 2, w, h, "white", pal[1], lw=0.6, r=0.03, z=3)
    if head:
        text(cx, cy + h / 2 - 0.065, head, fs=5.2, weight="bold")
        cy -= 0.045
    text(cx, cy, body, fs=fs, linespacing=1.25)


def chevron(x, y, color=SLATE, right=True):
    d = 1 if right else -1
    ax.add_patch(Polygon([(x - 0.04 * d, y + 0.10), (x + 0.055 * d, y), (x - 0.04 * d, y - 0.10)],
                         closed=True, fc=color, ec="none", zorder=9))


def line(xs, ys, color=SLATE, lw=0.9, head=True, z=4):
    ax.plot(xs[:-1] + [xs[-1]] if not head else xs[:-1], ys[:-1] if head else ys,
            color=color, lw=lw, zorder=z, solid_joinstyle="round")
    if head:
        ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>",
                                     mutation_scale=7, lw=lw, color=color, shrinkA=0,
                                     shrinkB=0, zorder=z))


# ===================================================================== icons
def icon_policy(cx, cy, s=1.0):
    xs = [cx - 0.16 * s, cx, cx + 0.16 * s]
    layers = [[-0.08, 0, 0.08], [-0.12, -0.04, 0.04, 0.12], [-0.06, 0.06]]
    pts = [[(x, cy + dy * s) for dy in L] for x, L in zip(xs, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=0.35, zorder=3)
    for L in pts:
        for p in L:
            ax.add_patch(Circle(p, 0.026 * s, fc=SLATE, ec="white", lw=0.4, zorder=4))


def icon_gauge(cx, cy, r=0.17):
    """Tolerance dial in the slate tones of the input panel."""
    for a0, a1, c in ((120, 180, "#C3CCD7"), (60, 120, "#8A99AD"), (0, 60, SLATE)):
        ax.add_patch(Wedge((cx, cy), r, a0, a1, width=0.32 * r, fc=c, ec=P_IN[0], lw=0.8, zorder=3))
    for a in np.deg2rad([150, 90, 30]):
        ax.plot([cx + 0.58 * r * np.cos(a), cx + 0.64 * r * np.cos(a)],
                [cy + 0.58 * r * np.sin(a), cy + 0.64 * r * np.sin(a)], color=GREY, lw=0.4, zorder=3)
    ang = np.deg2rad(55)
    ax.plot([cx, cx + 0.78 * r * np.cos(ang)], [cy, cy + 0.78 * r * np.sin(ang)],
            color=NAVY, lw=1.0, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((cx, cy), 0.022, fc=NAVY, ec="none", zorder=5))


def icon_rollouts(x0, y0, w, h):
    """An anchor state, the clean rollout under pi, and perturbed rollouts whose
    first n steps are random and which end lower."""
    ec = P1[1]
    xb = x0 + 0.30 * w
    ax.add_patch(Rectangle((x0, y0 - 0.55 * h), xb - x0, 1.1 * h, fc="#C9DDF0", ec="none", zorder=3))
    text((x0 + xb) / 2, y0 - 0.60 * h, "$n$ random actions", fs=4.8, va="top")
    t = np.linspace(0, 1, 60)
    ax.plot(x0 + t * w, y0 + 0.30 * h + 0.05 * h * np.sin(5 * t), color=NAVY, lw=0.9, zorder=4)
    text(x0 + w + 0.02, y0 + 0.30 * h, r"$\pi$", fs=6.0, ha="left")
    rng = np.random.default_rng(3)
    for end, c in ((-0.05, "#5B9BD0"), (-0.30, "#2A78B8"), (-0.55, "#0B4F86")):
        k = 7
        xj = np.linspace(x0, xb, k)
        yj = y0 + np.r_[0, rng.uniform(-0.22, 0.22, k - 2) * h, 0.05 * h]
        ax.plot(xj, yj, color=c, lw=0.7, zorder=4)
        xs = np.linspace(xb, x0 + w, 30)
        ys = yj[-1] + (end * h - yj[-1] + y0) * ((xs - xb) / (x0 + w - xb)) ** 1.3
        ax.plot(xs, ys, color=c, lw=0.7, ls=(0, (2.2, 1.2)), zorder=4)
    ax.add_patch(Circle((x0, y0), 0.034, fc=ec, ec="white", lw=0.5, zorder=5))
    text(x0 - 0.05, y0, "$s$", fs=6.2, ha="right")
    ax.annotate("", xy=(x0 + w + 0.005, y0 - 0.55 * h), xytext=(x0 + w + 0.005, y0 + 0.28 * h),
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=GREY, shrinkA=0, shrinkB=0))
    text(x0 + w + 0.03, y0 - 0.14 * h, r"$\Delta$", fs=5.6, ha="left")


def icon_trees(cx, cy, s=1.0):
    c = P2[1]
    for dx in (-0.29, -0.08, 0.24):
        r = (cx + dx * s, cy + 0.10 * s)
        kids = [(r[0] - 0.05 * s, cy), (r[0] + 0.05 * s, cy)]
        leaves = [(kids[0][0] - 0.025 * s, cy - 0.10 * s), (kids[0][0] + 0.025 * s, cy - 0.10 * s),
                  (kids[1][0] - 0.025 * s, cy - 0.10 * s), (kids[1][0] + 0.025 * s, cy - 0.10 * s)]
        for k in kids:
            ax.plot([r[0], k[0]], [r[1], k[1]], color=c, lw=0.6, zorder=3)
        for i, l in enumerate(leaves):
            p = kids[i // 2]
            ax.plot([p[0], l[0]], [p[1], l[1]], color=c, lw=0.6, zorder=3)
        for p, rr in [(r, 0.024)] + [(k, 0.02) for k in kids] + [(l, 0.017) for l in leaves]:
            ax.add_patch(Circle(p, rr * s, fc=c, ec="white", lw=0.3, zorder=4))
    text(cx + 0.08 * s, cy, r"$\cdots$", fs=6.5)


def icon_hist(cx, cy, w, h):
    c = P3[1]
    hts = np.array([0.35, 0.75, 1.0, 0.85, 0.6, 0.42, 0.28, 0.18, 0.12, 0.07])
    bw = w / len(hts)
    for i, v in enumerate(hts):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy - h / 2), bw * 0.86, v * h,
                               fc=c if i < 7 else "#9FD9C2", ec="none", zorder=3))
    xq = cx - w / 2 + 7 * bw - 0.06 * bw
    ax.plot([xq, xq], [cy - h / 2, cy + h / 2 + 0.02], color=NAVY, lw=0.7, ls=(0, (2, 1)), zorder=4)
    text(xq + 0.02, cy + h / 2 - 0.02, r"$Q(n)$", fs=5.6, ha="left")
    ax.plot([cx - w / 2 - 0.02, cx + w / 2 + 0.02], [cy - h / 2, cy - h / 2], color=GREY, lw=0.5)


def icon_spread(cx, cy, w, h):
    """Group coverages under the single offset against the band that sampling
    alone would produce; the riskiest group falls outside it."""
    ax.add_patch(Rectangle((cx - w / 2, cy - 0.22 * h), w, 0.44 * h, fc="#F6E3A9", ec="none", zorder=3))
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy], color=NAVY, lw=0.6, ls=(0, (2, 1)), zorder=4)
    ys = [0.12, -0.05, 0.16, -0.12, -0.46]
    for i, dy in enumerate(ys):
        x = cx - w / 2 + (i + 0.5) * w / len(ys)
        out = abs(dy) > 0.22
        ax.plot([x, x], [cy, cy + dy * h], color=GREY, lw=0.5, zorder=4)
        ax.add_patch(Circle((x, cy + dy * h), 0.03, fc="#6E4C00" if out else PC[1], ec="white",
                            lw=0.4, zorder=5))
    text(cx + w / 2 + 0.01, cy + 0.30 * h, r"$1-\alpha$", fs=5.0, ha="right", va="bottom")


def icon_groups(cx, cy, w):
    shades = ["#E7C6DA", "#CF96B8", "#B46A96", "#7E2F5E"]
    labs = ["1", "2", r"$\cdots$", "$G$"]
    bw = w / 4
    for i, (c, lab) in enumerate(zip(shades, labs)):
        x = cx - w / 2 + i * bw
        rbox(x + 0.012, cy - 0.10, bw - 0.024, 0.20, "white", c, lw=0.7, r=0.03, z=3)
        for dx, dy in ((-0.03, 0.04), (0.03, 0.04), (-0.03, -0.035), (0.03, -0.035)):
            ax.add_patch(Circle((x + bw / 2 + dx * 0.9, cy + dy), 0.017, fc=c, ec="none", zorder=4))
        text(x + bw / 2, cy - 0.15, lab, fs=5.4)
    ax.annotate("", xy=(cx + w / 2, cy + 0.15), xytext=(cx - w / 2, cy + 0.15),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=GREY, mutation_scale=5))
    text(cx, cy + 0.19, "risk score", fs=4.9, va="bottom")


def icon_shield(cx, cy, s=0.24):
    c = P_OUT[1]
    verts = [(-0.42, 0.40), (0.0, 0.55), (0.42, 0.40), (0.42, 0.02), (0.40, -0.30),
             (0.0, -0.58), (-0.40, -0.30), (-0.42, 0.02), (-0.42, 0.40)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    verts = [(cx + x * s, cy + y * s) for x, y in verts]
    ax.add_patch(PathPatch(Path(verts, codes), fc="white", ec=c, lw=1.2, zorder=3))
    ax.plot([cx - 0.17 * s, cx - 0.03 * s, cx + 0.2 * s], [cy, cy - 0.15 * s, cy + 0.17 * s],
            color=c, lw=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=4)


# ==================================================================== layout
YB, YT = 0.12, 3.50                      # bottom and top of the columns
R1 = (1.86, YT - 0.06)                   # stage row 1 (inside the method box)
R2 = (YB + 0.06, 1.62)                   # spread-test row 2
XIN = (0.04, 0.74)                       # input column
XM = (0.84, 4.96)                        # method box
XOUT = (5.06, 5.76)                      # output column
PW, PG = 1.24, 0.14                      # row-1 panel width and gap
PX = [XM[0] + 0.08 + k * (PW + PG) for k in range(3)]
M1 = (R1[0] + R1[1]) / 2

# method box around stages 1-4
rbox(XM[0], YB, XM[1] - XM[0], YT - YB, BOX[0], BOX[1], lw=0.9, r=0.08, z=1)

# ------------------------------------------------------------------- input
panel(XIN[0], YB, XIN[1] - XIN[0], YT - YB, "Input", P_IN, ty=YT - 0.15)
cxi = sum(XIN) / 2
icon_policy(cxi, M1 + 0.18, 1.0)
text(cxi, M1 - 0.12, "Fixed trained\npolicy $\\pi$")
ax.plot([XIN[0] + 0.1, XIN[1] - 0.1], [R2[1] + 0.12, R2[1] + 0.12], color="#C9D1DA", lw=0.6)
icon_gauge(cxi, 1.12, 0.17)
text(cxi, 0.72, "User-defined\ntolerance $\\zeta$\nand level $\\alpha$")

# ----------------------------------------------------------- row 1: stages 1-3
h1 = R1[1] - R1[0]
ICON1, NOTE1, CARD1 = R1[1] - 0.52, R1[1] - 0.94, R1[0] + 0.22
panel(PX[0], R1[0], PW, h1, "Perturbation and\nCollection", P1, 1)
icon_rollouts(PX[0] + 0.21, ICON1 + 0.02, 0.80, 0.36)
text(PX[0] + PW / 2, NOTE1, "Anchor $s$ saved in the simulator;\none rollout under $\\pi$ and $N$ with\n"
                            "$n$ random actions, $n\\in\\{1,2,4,\\ldots,32\\}$", fs=5.3)
card(PX[0] + PW / 2, CARD1, PW - 0.12, 0.36, r"$\Delta(s,n)=R^{\pi}_{\gamma}-R^{\pi'(t,n)}_{\gamma}$",
     P1, head="Draws of the reduction", fs=6.0)

panel(PX[1], R1[0], PW, h1, "Quantile\nPrediction", P2, 2)
icon_trees(PX[1] + PW / 2, ICON1, 1.0)
text(PX[1] + PW / 2, NOTE1, "Gradient-boosted trees\non the observation\nor policy embedding", fs=5.3)
card(PX[1] + PW / 2, CARD1, PW - 0.12, 0.36, r"$\hat{q}_{1-\alpha}(s,n)$", P2,
     head="Upper-tail estimate", fs=6.6)

panel(PX[2], R1[0], PW, h1, "Conformal\nCalibration", P3, 3)
icon_hist(PX[2] + PW / 2 - 0.08, ICON1 - 0.02, 0.62, 0.30)
text(PX[2] + PW / 2, NOTE1, "Split anchors into train,\ncalibration and test; one\ndraw per calibration anchor",
     fs=5.3)
card(PX[2] + PW / 2, CARD1, PW - 0.12, 0.36, r"$M_\alpha(s,n)=\hat{q}_{1-\alpha}(s,n)+Q(n)$", P3,
     head="Single-offset bound", fs=5.6)

for k in range(2):
    chevron(PX[k] + PW + PG / 2, M1)
chevron(XIN[1] + (XM[0] - XIN[1]) / 2, M1)

# --------------------------------------------- row 2: spread test and the split
h2 = R2[1] - R2[0]
SX, SW = PX[0], PW                        # spread test sits under stage 1
panel(SX, R2[0], SW, h2, "Expected-Spread\nTest", PC)
icon_spread(SX + SW / 2 + 0.05, R2[1] - 0.48, 0.66, 0.34)
text(SX + SW / 2, R2[1] - 0.81, "Spread in group coverage\nunder the single offset,\nagainst sampling alone",
     fs=5.3)
card(SX + SW / 2, R2[0] + 0.24, SW - 0.12, 0.36, r"$d_G\sqrt{\alpha(1-\alpha)/n_g}$", PC,
     head="Expected spread", fs=6.0)

# return path from stage 3 down to the test
yr = (R1[0] + R2[1]) / 2
line([PX[2] + PW / 2, PX[2] + PW / 2, SX + SW / 2, SX + SW / 2],
     [R1[0], yr, yr, R2[1]], color=SLATE)

# the split
BX0, BX1 = SX + SW + 0.70, XM[1] - 0.14   # branch boxes
fork = SX + SW + 0.10
yG = R2[1] - 0.47                         # group-refinement branch
yS = R2[0] + 0.26                         # single-offset branch
ym = (yG + yS) / 2
ax.plot([SX + SW, fork], [ym, ym], color=SLATE, lw=0.9, zorder=4)
ax.plot([fork, fork], [yS, yG], color=SLATE, lw=0.9, zorder=4)
line([fork, BX0], [yG, yG], color=P4[1])
line([fork, BX0], [yS, yS], color=P3[1])
text((fork + BX0) / 2, yG + 0.03, "single-offset\nspread\n> expected", fs=5.0, va="bottom", weight="bold")
text((fork + BX0) / 2, yS + 0.03, "single-offset\nspread\n$\\leq$ expected", fs=5.0, va="bottom", weight="bold")

# branch: group-conditional refinement
gh = 0.98
panel(BX0, R2[1] - gh, BX1 - BX0, gh, "Group-Conditional Refinement", P4, 4, ty=R2[1] - 0.13)
icon_groups(BX0 + 0.47, R2[1] - 0.58, 0.74)
gx = BX0 + 0.98 + (BX1 - BX0 - 0.98) / 2
text(gx, R2[1] - 0.37, "Risk-score groups fixed\nacross $n$, one offset\n$Q_g(n)$ for each", fs=5.3)
card(gx, R2[1] - gh + 0.23, BX1 - BX0 - 1.04, 0.36,
     "$M_\\alpha(s,n)=$\n$\\hat{q}_{1-\\alpha}(s,n)+Q_{g(s)}(n)$", P4, fs=5.7)

# branch: keep the single offset
sh = R2[1] - gh - 0.10 - R2[0]
panel(BX0, R2[0], BX1 - BX0, sh, "No Refinement", P3, ty=R2[0] + sh - 0.11)
text((BX0 + BX1) / 2, R2[0] + 0.15, "keep the single offset, $M_\\alpha(s,n)=\\hat{q}_{1-\\alpha}(s,n)+Q(n)$",
     fs=5.4)

# merge the two branches into the output
xm = BX1 + 0.07
yo = (R2[0] + R2[1]) / 2 + 0.05
ax.plot([BX1, xm], [yG, yG], color=SLATE, lw=0.9, zorder=4)
ax.plot([BX1, xm], [R2[0] + sh / 2, R2[0] + sh / 2], color=SLATE, lw=0.9, zorder=4)
ax.plot([xm, xm], [R2[0] + sh / 2, yG], color=SLATE, lw=0.9, zorder=4)
ax.plot([xm, XM[1]], [yo, yo], color=SLATE, lw=0.9, zorder=4)
chevron(XM[1] + (XOUT[0] - XM[1]) / 2, yo)

# ------------------------------------------------------------------ output
panel(XOUT[0], YB, XOUT[1] - XOUT[0], YT - YB, "Output", P_OUT, ty=YT - 0.15)
cxo = sum(XOUT) / 2
icon_shield(cxo, 2.72, 0.34)
text(cxo, 2.22, "Safety margin\n$s^{*}(s,\\zeta)$", fs=6.0)
text(cxo, 1.62, "largest $n$ with\n$M_\\alpha(s,n')\\leq\\zeta$\nfor all $n'\\leq n$", fs=5.3)
card(cxo, 0.95, XOUT[1] - XOUT[0] - 0.08, 0.42, "$\\mathbb{P}[\\Delta\\leq M_\\alpha]$\n$\\geq 1-\\alpha$",
     P_OUT, fs=5.6)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_method.pdf"))
fig.savefig(os.path.join(OUT, "fig_method.png"), dpi=300)
print("wrote", os.path.join(OUT, "fig_method.pdf"))
