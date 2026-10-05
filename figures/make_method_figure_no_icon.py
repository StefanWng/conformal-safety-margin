# -*- coding: utf-8 -*-
"""Text-only method overview: the icon-free, compact variant of make_method_figure.py.

One row inside a method box (stages 1-3, the expected-spread test, and its split
into group refinement or the single offset), with the input and the output
outside the box. Writes figures/fig_method_no_icon.pdf and .png at the full
AISTATS text width (6.75 in). Run with the csc249 environment.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6})

NAVY = "#1B3358"                        # every piece of text
SLATE = "#44546A"
P_IN = ("#EEF2F6", SLATE)
P1 = ("#E3EEF8", "#0072B2")
P2 = ("#FCEBDF", "#D55E00")
P3 = ("#E1F3EB", "#009E73")
PC = ("#FFF6DD", "#B07D00")
P4 = ("#F2E6EF", "#A0457A")
P_OUT = ("#E8EDF7", "#2F4B8A")
BOX = ("#F7F9FB", "#9AA8B8")

W_FIG, H_FIG = 6.75, 1.88
FS = 1.36                               # font scale
fig = plt.figure(figsize=(W_FIG, H_FIG))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_FIG); ax.set_ylim(0, H_FIG); ax.set_aspect("equal"); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=0.8, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r,
                                fc=fc, ec=ec, lw=lw, zorder=z))


def text(x, y, s, fs=5.4, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    kw.setdefault("linespacing", 1.2); kw.setdefault("zorder", 6)
    ax.text(x, y, s, fontsize=fs * FS, color=NAVY, **kw)


def panel(x, y, w, h, title, pal, num=None, tdy=0.16, tfs=6.3):
    rbox(x, y, w, h, pal[0], pal[1], lw=0.9, r=0.05, z=2)
    text(x + w / 2, y + h - tdy, title, fs=tfs, weight="bold", linespacing=1.05)
    if num is not None:
        ax.add_patch(Circle((x + 0.015, y + h - 0.015), 0.075, fc=pal[1], ec="white", lw=0.7, zorder=7))
        ax.text(x + 0.015, y + h - 0.017, str(num), ha="center", va="center", fontsize=5.4 * FS,
                color="white", weight="bold", zorder=8)


def card(cx, cy, w, h, body, pal, fs=5.7):
    rbox(cx - w / 2, cy - h / 2, w, h, "white", pal[1], lw=0.6, r=0.03, z=3)
    text(cx, cy, body, fs=fs, linespacing=1.25)


def chevron(x, y, color=SLATE):
    ax.add_patch(Polygon([(x - 0.032, y + 0.08), (x + 0.045, y), (x - 0.032, y - 0.08)],
                         closed=True, fc=color, ec="none", zorder=9))


def arrow(x0, y0, x1, y1, color=SLATE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0, zorder=4))


# ==================================================================== layout
YB, YT = 0.03, H_FIG - 0.03              # outer columns and method box
PB, PT = YB + 0.06, YT - 0.06            # stage panels inside the box
PH = PT - PB
YM = (PB + PT) / 2
XIN = (0.02, 0.61)
XOUT = (6.10, 6.73)
XM = (0.68, 6.03)

G, GF = 0.09, 0.15                       # gap between stages, gap at the fork
WS = [1.06, 0.86, 1.00, 0.85, 1.06]      # stages 1-3, spread test, branches
xs = [XM[0] + 0.05]
for k, w in enumerate(WS[:-1]):
    xs.append(xs[-1] + w + (GF if k == 3 else G))
cxs = [x + w / 2 for x, w in zip(xs, WS)]
TITLE_Y = PT - 0.15
BODY_Y = PT - 0.69
CARD_Y = PB + 0.245

rbox(XM[0], YB, XM[1] - XM[0], YT - YB, BOX[0], BOX[1], lw=0.9, r=0.07, z=1)

# ------------------------------------------------------------------- input
panel(XIN[0], YB, XIN[1] - XIN[0], YT - YB, "Input", P_IN)
cxi = sum(XIN) / 2
text(cxi, YM + 0.36, "Fixed trained\npolicy $\\pi$", fs=5.1)
ax.plot([XIN[0] + 0.09, XIN[1] - 0.09], [YM + 0.05, YM + 0.05], color="#C9D1DA", lw=0.6)
text(cxi, YM - 0.38, "User-defined\ntolerance $\\zeta$\nand level $\\alpha$", fs=5.1)
chevron((XIN[1] + XM[0]) / 2, YM)

# ----------------------------------------------------------- stages 1-3
panel(xs[0], PB, WS[0], PH, "Perturbation and\nCollection", P1, 1)
text(cxs[0], BODY_Y, "Save anchor $s$; one\nrollout under $\\pi$ and $N$\n"
                     "with $n$ random actions,\n$n\\in\\{1,2,4,\\ldots,32\\}$")
card(cxs[0], CARD_Y, WS[0] - 0.07, 0.35, r"$\Delta(s,n)=R^{\pi}_{\gamma}-R^{\pi'(t,n)}_{\gamma}$", P1)

panel(xs[1], PB, WS[1], PH, "Quantile\nPrediction", P2, 2)
text(cxs[1], BODY_Y, "Gradient-boosted\ntrees on the\nobservation or\npolicy embedding")
card(cxs[1], CARD_Y, WS[1] - 0.10, 0.35, r"$\hat{q}_{1-\alpha}(s,n)$", P2, fs=6.2)

panel(xs[2], PB, WS[2], PH, "Conformal\nCalibration", P3, 3)
text(cxs[2], BODY_Y, "Split anchors into\ntrain, calibration and\ntest; one draw per\ncalibration anchor")
card(cxs[2], CARD_Y, WS[2] - 0.07, 0.45, "$M_\\alpha(s,n)=$\n$\\hat{q}_{1-\\alpha}(s,n)+Q(n)$", P3)

# ---------------------------------------------------- expected-spread test
panel(xs[3], PB, WS[3], PH, "Expected-\nSpread Test", PC)
text(cxs[3], BODY_Y - 0.04, "Spread in group\ncoverage under\nthe single offset,\nagainst sampling\nalone")
card(cxs[3], CARD_Y, WS[3] - 0.06, 0.35, r"$d_G\sqrt{\alpha(1-\alpha)/n_g}$", PC)

for k in range(3):
    chevron(xs[k] + WS[k] + G / 2, YM)

# ------------------------------------------------------ the split
bx, bw = xs[4], WS[4]
gh = 1.00                                # group-refinement box
nh = PH - gh - 0.06                      # no-refinement box
gy0, ny0 = PT - gh, PB
yG, yN = gy0 + gh / 2, ny0 + nh / 2
fork = xs[3] + WS[3] + 0.06
ax.plot([xs[3] + WS[3], fork], [YM, YM], color=SLATE, lw=0.8, zorder=4)
ax.plot([fork, fork], [yN, yG], color=SLATE, lw=0.8, zorder=4)
arrow(fork, yG, bx, yG, color=P4[1])
arrow(fork, yN, bx, yN, color=P3[1])

panel(bx, gy0, bw, gh, "Group-Conditional\nRefinement", P4, 4, tdy=0.18, tfs=5.8)
text(bx + bw / 2, gy0 + 0.51, "if spread $>$ expected:\none $Q_g(n)$ per risk group", fs=5.0)
card(bx + bw / 2, gy0 + 0.175, bw - 0.05, 0.25, r"$\hat{q}_{1-\alpha}(s,n)+Q_{g(s)}(n)$", P4, fs=5.4)

panel(bx, ny0, bw, nh, "No Refinement", P3, tdy=0.16)
text(bx + bw / 2, ny0 + 0.19, "otherwise: keep $Q(n)$", fs=5.2)

# merge into the output
xm = bx + bw + 0.045
ax.plot([bx + bw, xm], [yG, yG], color=SLATE, lw=0.8, zorder=4)
ax.plot([bx + bw, xm], [yN, yN], color=SLATE, lw=0.8, zorder=4)
ax.plot([xm, xm], [yN, yG], color=SLATE, lw=0.8, zorder=4)
ax.plot([xm, XM[1]], [YM, YM], color=SLATE, lw=0.8, zorder=4)
chevron((XM[1] + XOUT[0]) / 2, YM)

# ------------------------------------------------------------------ output
panel(XOUT[0], YB, XOUT[1] - XOUT[0], YT - YB, "Output", P_OUT)
cxo = sum(XOUT) / 2
text(cxo, YM + 0.40, "Safety margin\n$s^{*}(s,\\zeta)$", fs=5.2)
text(cxo, YM - 0.04, "largest $n$ with\n$M_\\alpha(s,n')\\leq\\zeta$\nfor all $n'\\leq n$", fs=5.0)
card(cxo, PB + 0.21, XOUT[1] - XOUT[0] - 0.05, 0.37, "$\\mathbb{P}[\\Delta\\leq M_\\alpha]$\n$\\geq 1-\\alpha$",
     P_OUT, fs=5.2)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_method_no_icon.pdf"))
fig.savefig(os.path.join(OUT, "fig_method_no_icon.png"), dpi=300)
print("wrote", os.path.join(OUT, "fig_method_no_icon.pdf"))
