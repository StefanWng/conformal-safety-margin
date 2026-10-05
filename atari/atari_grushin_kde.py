"""
Grushin et al. (2409.18289) KDE baseline for BeamRider, head-to-head vs our conformal method.

Grushin's original "safety margin" pipeline:
  * TRUE criticality C_n(s) = expected return drop from n random actions then policy (MC),
  * a cheap PROXY criticality from the agent's own Q-values (here max_a Q - mean_a Q, the advantage of
    the best action over the average = how much a random action costs),
  * a 2-D **KDE** over (proxy, true criticality); the confidence bound B_n(proxy) is the conditional
    (1-alpha) percentile of true criticality given proxy read off that KDE  -- NO finite-sample guarantee,
  * safety margin s*(s, zeta) = max{ n : B_n'(proxy(s)) <= zeta for all n' <= n }.

This file reproduces that faithfully AND runs OUR conformal method (direct HGB upper quantile + split
conformal, on the same 512-d embedding features) on the *identical* anchors and train/test split, so the
two bounds can be compared on equal data. Everything is recomputed from the banked final-run npz
(`beamrider_grushin_final/raw.npz`) with ZERO new rollouts: the Grushin proxy is a function of the QRDQN
Q-values, and QRDQN computes Q from the CNN embedding we already stored, so proxy = Q-head(embedding).

Same pretrained model (sb3/qrdqn-BeamRiderNoFrameskip-v4), same environment collection, all 5,000 anchors.

Run (env `atari`, from the `safety_margin` directory; instant, no collection):
    python atari/atari_grushin_kde.py --raw-npz atari/runs/beamrider_grushin_final/raw.npz \
        --alpha 0.1 --test-fraction 0.2 --seed 0 --kde-subsample 40000 \
        --tolerance-list 25 75 150 300 \
        --results-json atari/runs/beamrider_grushin_kde/results.json \
        --plot-dir atari/runs/beamrider_grushin_kde/plots \
        --save-path atari/runs/beamrider_grushin_kde/kde_net.pkl
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from adaptivity_core import conformal_offset, one_per_anchor, _pearson  # noqa: E402
from atari_grushin import _fit_quantile, _grushin_margin  # noqa: E402
from atari_core import ENV_ID, REPO, build_env, load_model  # noqa: E402


# ---------------------------------------------------------------------------
# Grushin proxy criticality from the frozen QRDQN Q-head (no rollouts)
# ---------------------------------------------------------------------------
def compute_proxy(features: np.ndarray, device: str = "cpu") -> np.ndarray:
    """proxy(s) = max_a Q(s,a) - mean_a Q(s,a), computed from the banked 512-d anchor embeddings.

    QRDQN's QuantileNetwork.forward is head(extract_features(obs)) reshaped to [B, n_quantiles, n_actions];
    the stored `features` ARE extract_features(obs), so head(features) reproduces the quantiles exactly and
    Q(s,a) = mean over quantiles. No environment stepping needed.
    """
    import torch
    torch.backends.mkldnn.enabled = False
    env = build_env(seed=0, episodic_life=False)
    model, _ = load_model(env, device=device)
    qnet = model.policy.quantile_net
    head = qnet.quantile_net
    nq = int(qnet.n_quantiles)
    na = int(model.action_space.n)
    with torch.no_grad():
        feats_t = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=model.device)
        q = head(feats_t).view(-1, nq, na).mean(dim=1)          # [A, n_actions] action values
        proxy = (q.max(dim=1).values - q.mean(dim=1)).cpu().numpy().astype(np.float64)
    print(f"Proxy (max-mean Q) over {proxy.shape[0]} anchors: "
          f"mean {proxy.mean():.3f} std {proxy.std():.3f} [{proxy.min():.3f}, {proxy.max():.3f}] "
          f"| n_quantiles={nq} n_actions={na}")
    try:
        env.close()
    except Exception:
        pass
    return proxy


# ---------------------------------------------------------------------------
# Grushin 2-D KDE conditional (1-alpha) upper bound
# ---------------------------------------------------------------------------
def kde_conditional_upper(kde: gaussian_kde, proxy_grid: np.ndarray, y_grid: np.ndarray,
                          level: float) -> np.ndarray:
    """B(proxy) = conditional `level` quantile of y given proxy, from a fitted 2-D KDE over (proxy, y).

    Evaluate the joint density on proxy_grid x y_grid, normalize each proxy column to a conditional pdf,
    integrate to a CDF, and invert at `level`. Rows with negligible mass fall back to the top of y_grid.
    """
    Pm, Ym = np.meshgrid(proxy_grid, y_grid, indexing="ij")       # [G1, G2]
    Z = kde(np.vstack([Pm.ravel(), Ym.ravel()])).reshape(Pm.shape)  # joint density
    Z = np.clip(Z, 0.0, None)
    cdf = np.cumsum(Z, axis=1)
    totals = cdf[:, -1:].copy()
    totals[totals <= 0] = 1.0
    cdf = cdf / totals
    bound = np.empty(len(proxy_grid))
    for i in range(len(proxy_grid)):
        idx = int(np.searchsorted(cdf[i], level))
        bound[i] = y_grid[min(idx, len(y_grid) - 1)]
    return bound


def fit_kde_bound(proxy_fit: np.ndarray, drops_fit_n: np.ndarray, proxy_grid: np.ndarray,
                  level: float, subsample: int, bandwidth, y_points: int, seed: int):
    """Fit a 2-D KDE over (proxy repeated over samples, drop) and return (B on proxy_grid, y_grid, kde)."""
    N = drops_fit_n.shape[1]
    x = np.repeat(proxy_fit, N)
    y = drops_fit_n.reshape(-1)
    if subsample and x.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(x.shape[0], size=subsample, replace=False)
        x, y = x[sel], y[sel]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    y_lo = float(np.min(y))
    y_hi = float(np.quantile(y, 0.999))
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    y_grid = np.linspace(y_lo, y_hi, y_points)
    bound = kde_conditional_upper(kde, proxy_grid, y_grid, level)
    return bound, y_grid, kde


# ---------------------------------------------------------------------------
# CLI + data
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Grushin KDE safety-margin baseline vs our conformal method "
                                            "on BeamRider, from the banked final-run npz.")
    p.add_argument("--raw-npz", type=str, required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--calibration-fraction", type=float, default=0.3,
                   help="For OUR conformal method's calibration split (carved from the non-test fit set).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    # KDE controls
    p.add_argument("--kde-subsample", type=int, default=40000,
                   help="Max (proxy, drop) points fed to gaussian_kde per n (for tractability).")
    p.add_argument("--kde-bandwidth", type=str, default="scott",
                   help="gaussian_kde bw_method: 'scott', 'silverman', or a float.")
    p.add_argument("--proxy-grid-points", type=int, default=80)
    p.add_argument("--y-grid-points", type=int, default=400)
    # tolerances + HGB (for our conformal, reused via _fit_quantile)
    p.add_argument("--tolerance-list", type=float, nargs="+", default=[25.0, 75.0, 150.0, 300.0])
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument("--hgb-lr", type=float, default=0.05)
    p.add_argument("--hgb-l2", type=float, default=1.0)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--plot-dir", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def _bandwidth(arg: str):
    try:
        return float(arg)
    except (TypeError, ValueError):
        return arg


def main():
    args = parse_args()
    alpha = args.alpha
    level = 1.0 - alpha
    drop_label = "ΔG (return drop)"

    d = np.load(args.raw_npz, allow_pickle=True)
    X = d["features"].astype(np.float64)
    drops = d["drops_r"].astype(np.float64)                    # [A, n_vals, N]
    n_list = [int(n) for n in d["n_list"]]
    A, n_vals, N = drops.shape
    print(f"Loaded {A} anchors | n_list={n_list} | drops {drops.shape} | "
          f"source_policy {str(d['model_path'])}")

    # ---- Grushin proxy from the Q-head (recomputed, no rollouts) ---------------
    proxy = compute_proxy(X, device=args.device)

    # ---- shared split: identical test set for both methods ---------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(A)
    n_te = max(1, int(A * args.test_fraction))
    n_cal = max(1, int(A * args.calibration_fraction))
    te = perm[:n_te]
    fit = perm[n_te:]                                          # KDE fits on the whole non-test set
    cal = fit[:n_cal]                                          # our conformal calibration
    tr = fit[n_cal:]                                           # our conformal training
    print(f"Split: fit {len(fit)} (train {len(tr)} + calib {len(cal)}) | test {len(te)}")

    proxy_grid = np.linspace(float(proxy.min()), float(proxy.max()), args.proxy_grid_points)
    bw = _bandwidth(args.kde_bandwidth)

    # ---- per-n: Grushin KDE bound  vs  our conformal bound ---------------------
    B_te = np.empty((n_vals, len(te)))                         # Grushin KDE bound on test
    M_te = np.empty((n_vals, len(te)))                         # our conformal margin on test
    kde_curves = {}                                            # for plotting
    per_n = []
    print("\n" + "=" * 84)
    print(f"PER-n BOUNDS  (target coverage {level:.2f})   Grushin 2-D KDE  vs  our HGB+conformal")
    print("=" * 84)
    print(f"  {'n':<5}{'cov_KDE':<10}{'cov_conf':<10}{'mean_B(KDE)':<13}{'mean_M(conf)':<14}"
          f"{'corr(proxy,ΔG)':<16}{'mean ΔG':<10}")
    for ni, n in enumerate(n_list):
        dtr_n, dcal_n, dte_n = drops[tr, ni, :], drops[cal, ni, :], drops[te, ni, :]
        dfit_n = drops[fit, ni, :]

        # Grushin: 2-D KDE over (proxy, ΔG) on the fit set -> conditional (1-a) bound
        bound_grid, y_grid, kde = fit_kde_bound(
            proxy[fit], dfit_n, proxy_grid, level, args.kde_subsample, bw, args.y_grid_points, args.seed)
        B_te[ni] = np.interp(proxy[te], proxy_grid, bound_grid)
        cov_kde = float((dte_n <= B_te[ni][:, None]).mean())
        kde_curves[n] = {"proxy_grid": proxy_grid, "bound_grid": bound_grid, "y_grid": y_grid}

        # Ours: direct HGB upper quantile on the 512-d embedding + split conformal
        model = _fit_quantile(args, X[tr], dtr_n, level)
        base_cal, base_te = model.predict(X[cal]), model.predict(X[te])
        Q = conformal_offset(one_per_anchor(dcal_n, seed=args.seed) - base_cal, alpha)
        M_te[ni] = base_te + Q
        cov_conf = float((dte_n <= M_te[ni][:, None]).mean())

        corr = _pearson(proxy[te], dte_n.mean(axis=1))
        emp_mean = float(dte_n.mean())
        per_n.append({"n": int(n), "coverage_kde": cov_kde, "coverage_conformal": cov_conf,
                      "mean_bound_kde": float(B_te[ni].mean()), "mean_margin_conformal": float(M_te[ni].mean()),
                      "proxy_true_corr": corr, "emp_mean_drop": emp_mean,
                      "conformal_q_one": float(Q)})
        print(f"  {n:<5}{cov_kde:<10.4f}{cov_conf:<10.4f}{float(B_te[ni].mean()):<13.2f}"
              f"{float(M_te[ni].mean()):<14.2f}{corr:<+16.3f}{emp_mean:<10.2f}")

    # ---- margins + validity, both methods, across zeta -------------------------
    dte_all = drops[te]                                        # [A_te, n_vals, N]
    tol_rows = []
    print("\n" + "=" * 84)
    print("SAFETY MARGIN s*(s,ζ) = max n with bound_n'(s) ≤ ζ ∀n'≤n   (KDE vs conformal)")
    print("=" * 84)
    print(f"  {'ζ':<9}{'mean_m_KDE':<12}{'mean_m_conf':<13}{'val_KDE(exc≤α)':<16}{'val_conf(exc≤α)':<16}")
    for zeta in args.tolerance_list:
        m_kde = _grushin_margin(B_te, n_list, zeta)
        m_conf = _grushin_margin(M_te, n_list, zeta)
        worst_kde, worst_conf = 0.0, 0.0
        for ni, n in enumerate(n_list):
            ck, cc = m_kde >= n, m_conf >= n
            if ck.sum():
                worst_kde = max(worst_kde, float((dte_all[ck, ni, :] > zeta).mean()))
            if cc.sum():
                worst_conf = max(worst_conf, float((dte_all[cc, ni, :] > zeta).mean()))
        dist_kde = {int(m): int((m_kde == m).sum()) for m in ([0] + n_list)}
        dist_conf = {int(m): int((m_conf == m).sum()) for m in ([0] + n_list)}
        tol_rows.append({"tolerance": float(zeta),
                         "kde": {"mean_margin": float(m_kde.mean()),
                                 "frac_margin_zero": float((m_kde == 0).mean()),
                                 "worst_exceedance_at_margin": worst_kde,
                                 "validity_ok": bool(worst_kde <= alpha + 0.02),
                                 "margin_distribution": dist_kde},
                         "conformal": {"mean_margin": float(m_conf.mean()),
                                       "frac_margin_zero": float((m_conf == 0).mean()),
                                       "worst_exceedance_at_margin": worst_conf,
                                       "validity_ok": bool(worst_conf <= alpha + 0.02),
                                       "margin_distribution": dist_conf}})
        okk = "OK" if worst_kde <= alpha + 0.02 else "VIOLATED"
        okc = "OK" if worst_conf <= alpha + 0.02 else "VIOLATED"
        print(f"  {zeta:<9.2f}{float(m_kde.mean()):<12.3f}{float(m_conf.mean()):<13.3f}"
              f"{worst_kde:.3f} [{okk}]      {worst_conf:.3f} [{okc}]")

    # ---- summary headline ------------------------------------------------------
    kde_cov = np.array([r["coverage_kde"] for r in per_n])
    conf_cov = np.array([r["coverage_conformal"] for r in per_n])
    print("\n" + "-" * 84)
    print(f"COVERAGE vs target {level:.2f}:  KDE mean {kde_cov.mean():.3f} (abs miss "
          f"{np.abs(kde_cov-level).mean():.3f})  |  conformal mean {conf_cov.mean():.3f} (abs miss "
          f"{np.abs(conf_cov-level).mean():.3f})")
    print("-" * 84)

    # ---- plots -----------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        _plots(args, n_list, proxy, drops, fit, te, per_n, tol_rows, kde_curves, B_te, M_te, level, alpha)
        print(f"Saved plots to {args.plot_dir}")

    # ---- results json ----------------------------------------------------------
    results = {
        "method": "grushin_kde_vs_conformal", "env": ENV_ID, "policy": REPO,
        "proxy": "max_minus_mean_Q", "bound": "2d_kde_conditional_percentile",
        "args": {k: getattr(args, k) for k in vars(args)},
        "n_list": n_list, "alpha": alpha, "target_coverage": level,
        "split": {"fit": int(len(fit)), "train": int(len(tr)), "calibration": int(len(cal)),
                  "test": int(len(te)), "samples_per_state": int(N)},
        "proxy_stats": {"mean": float(proxy.mean()), "std": float(proxy.std()),
                        "min": float(proxy.min()), "max": float(proxy.max())},
        "per_n": per_n, "tolerances": tol_rows,
        "coverage_summary": {"kde_mean": float(kde_cov.mean()), "kde_abs_miss": float(np.abs(kde_cov-level).mean()),
                             "conformal_mean": float(conf_cov.mean()),
                             "conformal_abs_miss": float(np.abs(conf_cov-level).mean())},
        "source_policy": str(d["model_path"]),
    }
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved results JSON to {args.results_json}")
        # per-test-state proxy, draws and both bounds, read by figures/make_figures.py
        bounds_path = os.path.join(os.path.dirname(os.path.abspath(args.results_json)), "bounds_test.npz")
        np.savez_compressed(bounds_path, proxy_te=proxy[te], drops_te=drops[te],
                            M_te=M_te, B_te=B_te, n_list=np.array(n_list))
        print(f"Saved test-state bounds to {bounds_path}")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        joblib.dump({"method": "grushin_kde", "n_list": n_list, "alpha": alpha,
                     "proxy_grid": proxy_grid, "kde_bounds": {n: kde_curves[n]["bound_grid"] for n in n_list},
                     "tolerances": list(args.tolerance_list), "results": results}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


# ---------------------------------------------------------------------------
# Plots ("his KDE plots and everything else")
# ---------------------------------------------------------------------------
def _plots(args, n_list, proxy, drops, fit, te, per_n, tol_rows, kde_curves, B_te, M_te, level, alpha):
    ns = np.asarray(n_list)

    # 1) Signature Grushin figure: 2-D KDE of (proxy, ΔG) with the (1-a) bound curve, per n (2x3 panel).
    ncol = 3
    nrow = int(np.ceil(len(n_list) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for k, n in enumerate(n_list):
        ax = axes[k // ncol][k % ncol]
        ni = n_list.index(n)
        xf = proxy[fit]
        yf = drops[fit, ni, :].mean(axis=1)                    # per-anchor mean drop for a readable scatter
        y_hi = float(np.quantile(drops[fit, ni, :], 0.99))
        y_lo = float(np.quantile(drops[fit, ni, :], 0.01))
        hb = ax.hexbin(np.repeat(xf, 1), yf, gridsize=30, cmap="Blues", mincnt=1)
        ax.plot(kde_curves[n]["proxy_grid"], kde_curves[n]["bound_grid"], "r-", lw=2,
                label=f"KDE {level:.2f} bound")
        ax.set_ylim(min(y_lo, 0) - 1, max(y_hi, kde_curves[n]["bound_grid"].max()) * 1.05 + 1)
        ax.set_title(f"n={n}  (corr={per_n[ni]['proxy_true_corr']:+.2f})")
        ax.set_xlabel("proxy  max−mean Q"); ax.set_ylabel("ΔG (mean over samples)")
        ax.legend(fontsize=7)
    for k in range(len(n_list), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Grushin proxy vs true criticality with 2-D KDE percentile bound")
    fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "kde_proxy_vs_criticality.png"), dpi=170)
    plt.close(fig)

    # 2) Per-n bound coverage: KDE vs conformal vs target.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, [r["coverage_kde"] for r in per_n], "s-", color="tab:red", label="Grushin KDE bound")
    ax.plot(ns, [r["coverage_conformal"] for r in per_n], "o-", color="tab:blue", label="our conformal")
    ax.axhline(level, c="k", ls="--", label=f"target {level:.2f}")
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("perturbation count n"); ax.set_ylabel("empirical coverage of the bound")
    ax.set_title("Bound coverage: KDE (no guarantee) vs conformal"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "coverage_by_n_kde_vs_conformal.png"), dpi=180)
    plt.close(fig)

    # 3) Mean safety margin vs tolerance, both methods.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    zs = [t["tolerance"] for t in tol_rows]
    ax.plot(zs, [t["kde"]["mean_margin"] for t in tol_rows], "s-", color="tab:red", label="Grushin KDE")
    ax.plot(zs, [t["conformal"]["mean_margin"] for t in tol_rows], "o-", color="tab:blue", label="our conformal")
    ax.set_xlabel("tolerance ζ"); ax.set_ylabel("mean safety margin (tolerable n)")
    ax.set_title("Safety margin vs tolerance"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "margin_vs_tolerance_kde_vs_conformal.png"), dpi=180)
    plt.close(fig)

    # 4) proxy<->true correlation by n.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(n_list)), [r["proxy_true_corr"] for r in per_n], color="tab:purple")
    ax.set_xticks(range(len(n_list))); ax.set_xticklabels(n_list)
    ax.set_xlabel("perturbation count n"); ax.set_ylabel("Pearson(proxy, mean ΔG)")
    ax.set_title("Does the Grushin proxy track true criticality?")
    fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "proxy_true_corr_by_n.png"), dpi=180)
    plt.close(fig)

    # 5) Margin-distribution comparison per zeta.
    levels = [0] + n_list
    for t in tol_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        w = 0.4
        xk = np.arange(len(levels))
        ax.bar(xk - w / 2, [t["kde"]["margin_distribution"][m] for m in levels], width=w,
               color="tab:red", label="Grushin KDE")
        ax.bar(xk + w / 2, [t["conformal"]["margin_distribution"][m] for m in levels], width=w,
               color="tab:blue", label="our conformal")
        ax.set_xticks(xk); ax.set_xticklabels([str(m) for m in levels])
        ax.set_xlabel("safety margin (tolerable n)"); ax.set_ylabel("test states")
        ax.set_title(f"Margin distribution (ζ={t['tolerance']:g})"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, f"margin_hist_zeta{t['tolerance']:g}_compare.png"), dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
