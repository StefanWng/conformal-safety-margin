# -*- coding: utf-8 -*-
"""Generate every figure used in the Experiments section.

Run from the repository root after the experiments (see README):
  python figures/make_figures.py
"""
import json, io, os, math, shutil, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(io.open(os.path.join(ROOT, p), encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
})
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
     "purple": "#CC79A7", "grey": "#666666", "yellow": "#E69F00"}


PAST = os.path.join(OUT, "past")


def save(fig, name):
    """Write the figure, first moving any earlier version that differs into
    figures/past, labelled by the date and time it was made."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    png = os.path.join(OUT, name + ".png")
    if os.path.exists(png) and io.open(png, "rb").read() != buf.getvalue():
        os.makedirs(PAST, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M", time.localtime(os.path.getmtime(png)))
        for ext in (".pdf", ".png"):
            old = os.path.join(OUT, name + ext)
            if os.path.exists(old):
                shutil.move(old, os.path.join(PAST, "%s__%s%s" % (name, stamp, ext)))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    io.open(png, "wb").write(buf.getvalue())
    plt.close(fig)


MIN_CERT = 0.05          # tolerances certifying less than this are not reported


def usable(tol):
    """Drop tolerances at which almost nothing is certified. Below this the
    margin is of no use and the validity check rests on a handful of states."""
    return [t for t in tol if (1.0 - t["frac_margin_zero"]) >= MIN_CERT]


GR = [("SafetyPointGoal1 ($\\Delta$)", "runs/spg1/sweep/results.json", 1200),
      ("CartPole ($\\Delta^{c}$)", "runs/cartpole/sweep/results.json", 1000),
      ("Pendulum ($\\Delta$)", "runs/pendulum/sweep/results.json", 3000),
      ("BeamRider ($\\Delta$)", "runs/beamrider/sweep/results.json", 1000)]

# ------------------------------------------------------------------ Fig. 1
# Criticality against perturbation length: the monotonicity premise.
fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.6))
for k, (ax, (name, path, _)) in enumerate(zip(axes.flat, GR)):
    d = J(path)
    n = np.array([r["n"] for r in d["per_n"]], float)
    mu = np.array([r["emp_mean_drop"] for r in d["per_n"]])
    q9 = np.array([r["emp_upper_quantile"] for r in d["per_n"]])
    ax.plot(n, mu, "o-", color=C["blue"], label=r"mean $\Delta$")
    ax.plot(n, q9, "s--", color=C["orange"], label=r"$q_{0.9}(\Delta)$")
    ax.axhline(0, color=C["grey"], lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n); ax.set_xticklabels([int(v) for v in n])
    ax.set_title(name, fontsize=7.0, pad=3)
    if k >= 2:
        ax.set_xlabel("$n$", labelpad=1)
    if k % 2 == 0:
        ax.set_ylabel(r"reduction $\Delta$")
# BeamRider's curves stay low until n = 4, leaving its upper left corner free
axes[1, 1].legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.8,
                  borderaxespad=0.1)
fig.tight_layout(h_pad=0.6, w_pad=0.8)
save(fig, "fig_criticality_vs_n")

# ------------------------------------------------------------------ Fig. 2
# Marginal coverage of the certified margin at each perturbation length.
fig, ax = plt.subplots(figsize=(3.35, 2.2))
marks = ["o", "s", "^", "D", "v"]
for (name, path, nte), mk in zip(GR, marks):
    d = J(path)
    n = np.array([r["n"] for r in d["per_n"]], float)
    cov = np.array([r["coverage"] for r in d["per_n"]])
    ci = np.array([r.get("coverage_ci95_halfwidth") or 1.96 * math.sqrt(.9 * .1 / nte)
                   for r in d["per_n"]])
    ax.errorbar(n, cov, yerr=ci, fmt=mk + "-", capsize=1.6, elinewidth=0.7,
                label=name, alpha=0.9)
ax.axhline(0.9, color="k", ls="--", lw=0.9, label=r"target $1-\alpha=0.90$")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlabel("perturbation length $n$"); ax.set_ylabel("empirical coverage")
ax.set_ylim(0.785, 0.96)
ax.set_yticks([0.85, 0.90, 0.95])
# the key sits in the empty band below the data, on two rows of three
ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=6.2,
          handlelength=1.6, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
fig.tight_layout()
save(fig, "fig_coverage")

# ------------------------------------------------------------------ Fig. 3
# Distribution of the tolerable-perturbation margin as the tolerance is swept.
LEVELS = [0, 1, 2, 4, 8, 16, 32]
cmap = plt.get_cmap("viridis")
cols = [cmap(i / (len(LEVELS) - 1.0)) for i in range(len(LEVELS))]
fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.85), sharey=True)
for k, (ax, (name, path, _)) in enumerate(zip(axes.flat, GR)):
    d = J(path)
    tol = usable(d["tolerances"])
    if len(tol) > 8:                       # thin dense sweeps for legibility
        idx = np.linspace(0, len(tol) - 1, 8).astype(int)
        tol = [tol[i] for i in idx]
    x = np.arange(len(tol))
    dist = np.array([[t["margin_distribution"].get(str(L), 0) for L in LEVELS] for t in tol],
                    dtype=float)
    dist = dist / dist.sum(axis=1, keepdims=True)
    bottom = np.zeros(len(tol))
    for j, L in enumerate(LEVELS):
        ax.bar(x, dist[:, j], bottom=bottom, width=0.82, color=cols[j],
               label=((r"$s^{*}{=}%d$" if j == 0 else "$%d$") % L) if k == 0 else None)
        bottom += dist[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels(["%.4g" % t["tolerance"] for t in tol], rotation=90, fontsize=5.6)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_title(name, fontsize=7.0, pad=3)
    if k >= 2:
        ax.set_xlabel(r"tolerance $\zeta$", labelpad=1)
    if k % 2 == 0:
        ax.set_ylabel("fraction of states")
    ax.set_ylim(0, 1)
# one row of seven swatches above the panels, labelled by the margin s*
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=len(LEVELS), frameon=False, fontsize=6.4,
           handlelength=1.0,
           handletextpad=0.3, columnspacing=0.7, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.4, w_pad=0.6)
save(fig, "fig_margin_distribution")

# ------------------------------------------------------------------ Fig. 4
# Reliability audit and the recovery ceiling.
AD = [("SafetyPointGoal1", "runs/spg1/n16/results.json"),
      ("CartPole", "runs/cartpole/n8/results.json"),
      ("Pendulum", "runs/pendulum/n8/results.json"),
      ("BeamRider", "runs/beamrider/n8/results.json")]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.95))
CD, CE = C["orange"], C["blue"]                      # dispersion, exceedance
MARK = {"SafetyPointGoal1": "s", "CartPole": "^", "Pendulum": "D", "BeamRider": "v"}
w, xs = 0.36, np.arange(len(AD))
disp, exc = [], []
for _, path in AD:
    d = J(path)
    dv = [d["reliability"][k]["sb_reliability"] for k in ("std", "iqr", "mad")
          if np.isfinite(d["reliability"][k]["sb_reliability"])]
    ev = [v["sb_reliability"] for k, v in d["reliability"].items() if k.startswith("exceedance")]
    disp.append(max(dv) if dv else 0.0)
    exc.append(max(ev))
ax = axes[0]
ax.bar(xs - w / 2, disp, w, color=CD, label="dispersion")
ax.bar(xs + w / 2, exc, w, color=CE, label="exceedance")
ax.axhline(0.30, color="k", ls="--", lw=0.9, label=r"floor $\rho=0.3$")
ax.set_xticks(xs); ax.set_xticklabels([a for a, _ in AD], fontsize=6.8)
ax.set_ylabel(r"reliability $\rho$")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
          handlelength=1.4, columnspacing=1.2)
ax.text(-0.13, 1.08, "(a)", transform=ax.transAxes, fontsize=8, weight="bold")

ax = axes[1]
for nm, path in AD:
    d = J(path)
    for k, v in d["reliability"].items():
        rec = d["recovery"][k]["recovery_r"]
        if not (np.isfinite(v["ceiling"]) and np.isfinite(rec)):
            continue
        ax.scatter(v["ceiling"], rec, s=16, marker=MARK[nm], alpha=0.9, linewidths=0,
                   color=CE if k.startswith("exceedance") else CD)
ax.plot([0, 1], [0, 1], color="k", ls="--", lw=0.8)
ax.text(0.40, 0.47, "largest attainable", fontsize=5.8, rotation=33, ha="center")
hs = [plt.Line2D([], [], marker=MARK[nm], ls="", color=C["grey"], ms=3.8, label=nm)
      for nm, _ in AD]
ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
          frameon=False, handletextpad=0.2, columnspacing=0.9, fontsize=6.4)
ax.set_xlabel(r"largest attainable correlation $\sqrt{\rho}$", fontsize=7)
ax.set_ylabel("recovery", fontsize=7)
ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.09, 1.05)
ax.text(-0.13, 1.08, "(b)", transform=ax.transAxes, fontsize=8, weight="bold")
fig.tight_layout(w_pad=2.0)
save(fig, "fig_reliability")

# ------------------------------------------------------------------ Fig. 6
# The density-estimate bound of Grushin et al. against ours, on identical
# BeamRider anchors, draws and split.
KD = J("runs/beamrider/kde/results.json")
bt = np.load(os.path.join(ROOT, "runs", "beamrider", "kde", "bounds_test.npz"))
NLIST = [int(n) for n in KD["n_list"]]
CK, CO = C["orange"], C["blue"]                      # KDE baseline, our margin

fig = plt.figure(figsize=(3.35, 3.0))
gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# (a) how closely each quantity tracks the realized risk of a state, per n
ax = axes[0]
q90 = np.percentile(bt["drops_te"], 90, axis=2)          # [states, n]
px = bt["proxy_te"]
c_proxy = [np.corrcoef(px, q90[:, i])[0, 1] for i in range(len(NLIST))]
c_ours = [np.corrcoef(bt["M_te"][i], q90[:, i])[0, 1] for i in range(len(NLIST))]
ax.plot(NLIST, c_proxy, "s-", color=CK, label="proxy")
ax.plot(NLIST, c_ours, "o-", color=CO, label="certified margin")
ax.set_xscale("log", base=2)
ax.set_xticks(NLIST); ax.set_xticklabels(NLIST)
ax.set_ylim(0, 0.75)
ax.set_xlabel("perturbation length $n$", labelpad=1)
ax.set_ylabel(r"corr. with $q_{0.9}(\Delta)$")
ax.set_title("(a) tracking the risk of a state", fontsize=7.0, pad=3)
ax.legend(loc="upper left", frameon=False, fontsize=6.4)

# (b) the margin in units of n as the tolerance is swept
ax = axes[1]
z = np.array([t["tolerance"] for t in KD["tolerances"]])
mk = np.array([t["kde"]["mean_margin"] for t in KD["tolerances"]])
mc = np.array([t["conformal"]["mean_margin"] for t in KD["tolerances"]])
ax.plot(z, mk, "s-", color=CK, label="density estimate")
ax.plot(z, mc, "o-", color=CO, label="certified margin")
ax.set_xscale("log")
ax.set_xticks(z); ax.set_xticklabels(["%g" % v for v in z])
ax.minorticks_off()
ax.annotate("no state certified", xy=(z[0], mk[0]), textcoords="offset points",
            xytext=(3, -5), fontsize=5.6, color=CK, va="center")
ax.set_ylim(-4.5, 21)
ax.set_xlabel(r"tolerance $\zeta$", labelpad=1)
ax.set_ylabel("mean margin $s^{*}$")
ax.set_title("(b) certified length", fontsize=7.0, pad=3)
ax.legend(loc="upper left", frameon=False, fontsize=6.0, handlelength=1.5,
          borderaxespad=0.1)

# (c) the exceedance each margin actually incurs, against the level allowed
ax = axes[2]
ek = np.array([t["kde"]["worst_exceedance_at_margin"] for t in KD["tolerances"]])
ec = np.array([t["conformal"]["worst_exceedance_at_margin"] for t in KD["tolerances"]])
ax.axhline(KD["alpha"], color="k", ls="--", lw=0.9)
ax.annotate(r"allowed, $\alpha=0.1$", xy=(z[-1], KD["alpha"]), textcoords="offset points",
            xytext=(0, 2), fontsize=5.8, ha="right")
ax.plot(z[1:], ek[1:], "s-", color=CK, label="density estimate")
ax.plot(z, ec, "o-", color=CO, label="certified margin")
ax.scatter([z[0]], [ek[0]], s=14, facecolors="none", edgecolors=CK, zorder=3)
ax.annotate("no state certified", xy=(z[0], ek[0]), textcoords="offset points",
            xytext=(4, 0), fontsize=5.6, color=CK, va="center")
ax.set_xscale("log")
ax.set_xticks(z); ax.set_xticklabels(["%g" % v for v in z])
ax.minorticks_off()
ax.set_ylim(-0.008, 0.128)
ax.set_xlabel(r"tolerance $\zeta$", labelpad=1)
ax.set_ylabel("worst exceedance")
ax.set_title("(c) risk actually incurred", fontsize=7.0, pad=3)

fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "fig_kde_baseline")

# ------------------------------------------------------------------ Fig. 6
# The validity check at every tolerance setting tested.
VAL = [("SafetyPointGoal1 ($\\Delta$)", "runs/spg1/sweep/results.json",
        C["orange"], "s"),
       ("CartPole ($\\Delta^{c}$)", "runs/cartpole/sweep/results.json",
        C["green"], "^"),
       ("Pendulum ($\\Delta$)", "runs/pendulum/sweep/results.json",
        C["purple"], "D"),
       ("BeamRider ($\\Delta$)", "runs/beamrider/sweep/results.json",
        C["yellow"], "v")]
ALPHA = 0.1

fig, ax = plt.subplots(figsize=(3.35, 2.05))
n_fail = n_tot = 0
for nm, path, col, mk in VAL:
    tol = usable(J(path)["tolerances"])
    z = np.array([t["tolerance"] for t in tol], float)
    e = np.array([t["worst_exceedance_at_margin"] for t in tol], float)
    ok = e <= ALPHA
    n_fail += int((~ok).sum()); n_tot += len(tol)
    ax.plot(z, e, "-", color=col, lw=0.7, alpha=0.45, zorder=2)
    ax.scatter(z[ok], e[ok], s=17, marker=mk, color=col, linewidths=0, zorder=3)
    ax.scatter(z[~ok], e[~ok], s=30, marker=mk, facecolors="white",
               edgecolors=col, linewidths=1.0, zorder=4)
ax.axhline(ALPHA, color="k", ls="--", lw=0.9, zorder=1)
ax.set_xscale("log")
ax.set_xlabel(r"tolerance $\zeta$")
ax.set_ylabel("worst exceedance")
ax.set_ylim(-0.006, 0.205)
ax.set_yticks([0.0, 0.05, 0.10])
# labelling the level on its own line keeps it out of the key
ax.annotate(r"$\alpha=0.1$", xy=(1.0, ALPHA), xycoords=("axes fraction", "data"),
            xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=6.6)
hv = [plt.Line2D([], [], marker=mk, color=col, ls="", ms=4.0, label=nm)
      for nm, _, col, mk in VAL]
hv += [plt.Line2D([], [], marker="o", mfc="white", mec=C["grey"], mew=1.0,
                  color="none", ls="", ms=4.8, label="fails the check")]
# the key sits in the empty band above the data, on two rows of three
ax.legend(handles=hv, loc="upper center", ncol=3, frameon=False, fontsize=6.2,
          handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
fig.tight_layout()
save(fig, "fig_validity")
print("validity check: %d of %d settings fail at exceedance > %.2f"
      % (n_fail, n_tot, ALPHA))

# ------------------------------------------------------------------ Fig. 7
# Single offset against group offsets, by risk group, in the three settings
# whose single-offset spread exceeds the expected spread. Groups come from one
# score per state, shared across n (group_sweep/fixed_groups.py). Coverage is
# averaged over n, over-certification over the reported tolerances, and both
# over 200 calibration/test partitions (group_sweep/fixed_agree.py).
GM = [("pendulum", "Pendulum ($\\Delta$)"),
      ("beamrider", "BeamRider ($\\Delta$)")]
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.4))
for c, (key, title) in enumerate(GM):
    d = J("runs/grouping/fixed_all_%s.json" % key)
    a = J("runs/grouping/agree_%s.json" % key)
    x = np.arange(1, d["G"] + 1)
    ax = axes[0, c]
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=0.8)
    ax.plot(x, np.nanmean(np.array(d["gcov1"]), (0, 1)), "o-", color=C["blue"],
            label="single offset")
    ax.plot(x, np.nanmean(np.array(d["gcovG"]), (0, 1)), "s-", color=C["orange"],
            label="group offsets")
    ax.set_ylim(0.83, 0.99); ax.set_xticks(x); ax.set_title(title, fontsize=7.6)
    if c == 0:
        ax.set_ylabel("coverage within group")
    ax = axes[1, c]
    w = 0.38
    ax.bar(x - w / 2, np.nanmean(np.array(a["overg1"]), (0, 1)), w, color=C["blue"])
    ax.bar(x + w / 2, np.nanmean(np.array(a["overgG"]), (0, 1)), w, color=C["orange"])
    ax.set_xticks(x); ax.set_xlabel("risk group (low to high)")
    if c == 0:
        ax.set_ylabel("over-certified states")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig_group_margin")

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("   %-34s %7d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))
