"""
Reliability-backed adaptive-width safety margin — DIRECT-PREDICTOR edition.

Twin of ``gmm_safety_margin_adaptivity.py`` with the MDN replaced by direct sklearn
``HistGradientBoosting`` predictors. The thesis-scale Mondrian run showed the adaptive margin was
valid but not useful (``width_exceedance_corr = -0.16``), because the NLL-trained MDN under-predicts
the quantities the margin uses. A predictability probe on the saved raw data found a direct learner
roughly doubles recovery: per-state exceedance 0.15 -> 0.33, per-state mean 0.28 -> 0.46. Conformal
prediction is model-agnostic, so we keep the entire conformal / Mondrian / reliability layer verbatim
and only swap the underlying predictor.

Predictors (HGB; trees are scale-invariant, so no input standardization):
  * median               : quantile regressor at 0.5           (diagnostics)
  * upper base q̂_{1-α}(s) : quantile regressor at 1-α           (one-sided base AND Mondrian center —
                            centering on the median forces Q_g to carry an unreliable dispersion
                            quantity and inverts the width; centering on the upper quantile lets each
                            group's offset REPAIR that group's residual miscoverage with a guarantee)
  * lo / hi              : quantile regressors at α/2, 1-α/2   (CQR two-sided)
  * mean                 : squared-loss regressor              (constant-δ base)
  * exceedance p̂(ΔG>τ|s) : regressor on per-state empirical exceedance  (Mondrian grouping + headline)
  * std/iqr/mad          : regressors on per-state emp stat    (recovery table; ~0 as those are noise)

The Mondrian grouping is keyed to the reliable, best-recovered target (an exceedance threshold).
The reduction is the reward reduction ΔG = G_clean - G_perturbed.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Labels contain "Δ"; keep stdout robust to a non-UTF-8 Windows console codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import HistGradientBoostingRegressor

from spg1_core import (
    collect_parallel,
    _scatter_identity,
    conformal_offset,
    one_per_anchor,
    split_half_reliability,
    _stat_std,
    _stat_iqr,
    _stat_mad,
    _stat_exceedance,
    _pearson,
    calibrate_global_center,
    mondrian_calibrate,
    mondrian_apply,
    _resolve_mondrian_tau,
)


# ---------------------------------------------------------------------------
# HGB helpers
# ---------------------------------------------------------------------------
def _hgb_regressor(args, loss: str = "squared_error", quantile: float = None):
    kw = dict(max_iter=args.hgb_max_iter, learning_rate=args.hgb_lr,
              l2_regularization=args.hgb_l2, validation_fraction=0.15,
              n_iter_no_change=30, random_state=args.seed)
    if loss == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


def _fit_flat(args, X_train: np.ndarray, drops_train: np.ndarray, loss: str, quantile: float = None):
    """Fit an HGB on flattened (obs repeated N times, drop) pairs -> conditional stat of ΔG."""
    N = drops_train.shape[1]
    X_rep = np.repeat(X_train, N, axis=0)
    y_flat = drops_train.reshape(-1)
    return _hgb_regressor(args, loss=loss, quantile=quantile).fit(X_rep, y_flat)


def _fit_peranchor(args, X_train: np.ndarray, target_train: np.ndarray):
    """Fit an HGB on per-anchor (obs, empirical stat) rows -> predicts that per-state stat."""
    return _hgb_regressor(args, loss="squared_error").fit(X_train, target_train)


# ---------------------------------------------------------------------------
# Direct-predictor conformal comparison (analogs of evaluate_conformal)
# ---------------------------------------------------------------------------
def _one_sided_cov(drops: np.ndarray, margin: np.ndarray) -> float:
    return float((drops <= margin[:, None]).mean())


def _interval_cov_width(drops: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    cov = float(((drops >= lo[:, None]) & (drops <= hi[:, None])).mean())
    width = float((hi - lo).mean())
    return cov, width


def direct_conformal_table(calib_y, test_drops, alpha,
                           mean_cal, mean_te, up_cal, up_te, lo_cal, lo_te, hi_cal, hi_te):
    """Constant-δ / CQR / one-sided conformal using direct-quantile bases (one calib score per anchor)."""
    # constant-δ around the predicted mean
    delta = conformal_offset(np.abs(calib_y - mean_cal), alpha)
    cd_cov, cd_w = _interval_cov_width(test_drops, mean_te - delta, mean_te + delta)
    # CQR two-sided
    q_cqr = conformal_offset(np.maximum(lo_cal - calib_y, calib_y - hi_cal), alpha)
    cqr_cov, cqr_w = _interval_cov_width(test_drops, lo_te - q_cqr, hi_te + q_cqr)
    # one-sided upper
    q_one = conformal_offset(calib_y - up_cal, alpha)
    os_margin = up_te + q_one
    os_cov = _one_sided_cov(test_drops, os_margin)
    return {
        "constant_delta": {"coverage": cd_cov, "mean_width": cd_w, "delta": float(delta)},
        "cqr_two_sided": {"coverage": cqr_cov, "mean_width": cqr_w, "q_cqr": float(q_cqr)},
        "one_sided_margin": {"coverage": os_cov, "mean_margin": float(os_margin.mean()),
                             "q_one": float(q_one), "side": "upper"},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Reliability-backed adaptive-width safety margin with DIRECT (HGB) predictors "
                     "in place of the MDN; Mondrian conformal grouped by predicted exceedance.")
    )
    p.add_argument("--eval-dir", type=str, required=True)
    p.add_argument("--raw-npz", type=str, default=None,
                   help="Load raw (features, drops, g_clean) from this .npz if it exists; "
                        "otherwise collect and save it here for reuse.")
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=2000)
    p.add_argument("--samples-per-state", type=int, default=64)
    p.add_argument("--random-prob", type=float, default=0.05)
    p.add_argument("--random-steps", "--random-step", dest="random_steps", type=int, default=16)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--history-len", type=int, default=2)
    # dispersion / risk targets
    p.add_argument("--tau-list", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    p.add_argument("--reliability-floor", type=float, default=0.30)
    p.add_argument("--mondrian-tau", type=str, default="auto")
    p.add_argument("--mondrian-groups", type=int, default=5)
    p.add_argument("--num-bins", type=int, default=4)
    # HGB hyperparams
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument("--hgb-lr", type=float, default=0.05)
    p.add_argument("--hgb-l2", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    # conformal
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--calibration-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    # output
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--plot-dir", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def _load_or_collect(args, device):
    if args.raw_npz and os.path.exists(args.raw_npz):
        print(f"Loading raw data from {args.raw_npz}")
        d = np.load(args.raw_npz, allow_pickle=True)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        print(f"Loaded {d['features'].shape[0]} anchors | input_dim={extra['input_dim']} | "
              f"source_policy {os.path.basename(extra['model_path'])}")
        return d["features"].astype(np.float64), d["drops_r"].astype(np.float64), extra

    feats, drops_r, g_clean_r, extra = collect_parallel(
        eval_dir=args.eval_dir, num_anchors=args.num_anchors,
        samples_per_state=args.samples_per_state, random_prob=args.random_prob,
        random_steps=args.random_steps, gamma=args.gamma,
        max_steps=args.max_steps, history_len=args.history_len,
        num_workers=args.num_workers, base_seed=args.seed,
    )
    feats = feats.astype(np.float64)
    drops_r = drops_r.astype(np.float64)
    if args.raw_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_npz)), exist_ok=True)
        np.savez_compressed(
            args.raw_npz, features=feats, drops_r=drops_r, g_clean_r=g_clean_r,
            model_path=extra["model_path"], obs_dim=extra["obs_dim"], input_dim=extra["input_dim"])
        print(f"Saved raw data to {args.raw_npz}")
    return feats, drops_r, extra


def main() -> None:
    args = parse_args()
    drop_label = "ΔG (reward drop)"

    features, drops, extra = _load_or_collect(args, torch.device("cpu"))
    X = features
    A, N = drops.shape
    if A < 10:
        raise RuntimeError("Need >=10 anchors.")

    # ---- reliability of candidate targets (model-free, on ALL anchors) ---------
    stat_fns = {"std": _stat_std, "iqr": _stat_iqr, "mad": _stat_mad}
    for tau in args.tau_list:
        stat_fns[f"exceedance@{tau:g}"] = _stat_exceedance(tau)

    reliability: Dict[str, Dict[str, float]] = {}
    print("\n" + "=" * 68)
    print(f"RELIABILITY of per-state targets  ({drop_label}, N={N} samples/anchor)")
    print("=" * 68)
    for name, fn in stat_fns.items():
        rel = split_half_reliability(drops, fn, seed=args.seed)
        rel["cleared"] = bool(np.isfinite(rel["sb_reliability"])
                              and rel["sb_reliability"] >= args.reliability_floor)
        reliability[name] = rel
        flag = "CLEARS" if rel["cleared"] else "noise "
        print(f"  {name:<16} split-half r={rel['r_half']:+.3f} | reliability={rel['sb_reliability']:+.3f}"
              f" | ceiling~{rel['ceiling']:.3f}  [{flag} floor {args.reliability_floor:.2f}]")

    # ---- 3-way anchor split ----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(A)
    n_calib = max(1, int(A * args.calibration_fraction))
    n_test = max(1, int(A * args.test_fraction))
    if A - n_calib - n_test < 1:
        raise RuntimeError("calibration+test fractions too large.")
    te_idx, cal_idx, tr_idx = perm[:n_test], perm[n_test:n_test + n_calib], perm[n_test + n_calib:]
    Xtr, Xcal, Xte = X[tr_idx], X[cal_idx], X[te_idx]
    dtr, dcal, dte = drops[tr_idx], drops[cal_idx], drops[te_idx]
    print(f"\nSplit: train {len(tr_idx)} | calib {len(cal_idx)} | test {len(te_idx)} | "
          f"input_dim {X.shape[1]}  (predictor: direct HGB)")

    # ---- direct predictors -----------------------------------------------------
    alpha = args.alpha
    print("Fitting HGB predictors (median / upper / lo / hi / mean) ...")
    m_median = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=0.5)
    m_upper = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=1.0 - alpha)
    m_lo = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=alpha / 2.0)
    m_hi = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=1.0 - alpha / 2.0)
    m_mean = _fit_flat(args, Xtr, dtr, loss="squared_error")

    # per-anchor target regressors (recovery table + exceedance grouping)
    emp_tr = {name: fn(dtr) for name, fn in stat_fns.items()}
    emp_te = {name: fn(dte) for name, fn in stat_fns.items()}
    target_models = {name: _fit_peranchor(args, Xtr, emp_tr[name]) for name in stat_fns}

    # ---- recovery of each target on TEST ---------------------------------------
    recovery: Dict[str, Dict[str, float]] = {}
    print("\n" + "=" * 68)
    print("RECOVERY of each target by a direct HGB predictor (test split)")
    print("=" * 68)
    for name in stat_fns:
        pred = target_models[name].predict(Xte)
        r = _pearson(emp_te[name], pred)
        rel = reliability[name]["sb_reliability"]
        disatt = float(np.clip(r / np.sqrt(rel), -1.0, 1.0)) if (np.isfinite(rel) and rel > 1e-6) else float("nan")
        recovery[name] = {"recovery_r": r, "disattenuated_r": disatt, "cleared": reliability[name]["cleared"]}
        print(f"  {name:<16} recovery_r={r:+.3f} | disattenuated={disatt:+.3f} "
              f"| {'reliable' if reliability[name]['cleared'] else 'noise-target'}")

    # sanity anchor: per-state mean recovery
    mean_recovery_r = _pearson(dte.mean(axis=1), _fit_peranchor(args, Xtr, dtr.mean(axis=1)).predict(Xte))
    std_recovery_r = recovery["std"]["recovery_r"]
    print(f"Legacy: mean_recovery_r={mean_recovery_r:+.3f} | std_recovery_r={std_recovery_r:+.3f}")

    cleared = [k for k in stat_fns if reliability[k]["cleared"]]
    headline_target = max(cleared, key=lambda k: recovery[k]["recovery_r"]) if cleared else None
    print(f"\nHeadline adaptive-width metric: {headline_target or 'NONE cleared'}"
          + (f"  recovery_r={recovery[headline_target]['recovery_r']:+.3f}" if headline_target else ""))

    # ---- Mondrian (exceedance-grouped) conformal margin ------------------------
    m_tau = _resolve_mondrian_tau(args.mondrian_tau, reliability, args.tau_list, headline_target)
    exc_model = target_models[f"exceedance@{m_tau:g}"]

    calib_y = one_per_anchor(torch.as_tensor(dcal, dtype=torch.float32), seed=args.seed)
    # Center the Mondrian on the PREDICTED UPPER QUANTILE q̂_{1-α}(s), not the median: the median
    # strips the (reliable) location signal and forces Q_g to encode an unreliable dispersion, which
    # inverts the width. With the upper-quantile center, each group's conformal offset repairs that
    # group's residual miscoverage, guaranteeing >= 1-α coverage per predicted-risk group.
    cen_cal = torch.as_tensor(m_upper.predict(Xcal), dtype=torch.float32)
    cen_te = torch.as_tensor(m_upper.predict(Xte), dtype=torch.float32)
    pexc_cal = exc_model.predict(Xcal)
    pexc_te = exc_model.predict(Xte)

    edges, Qg, q_global, counts = mondrian_calibrate(calib_y, cen_cal, pexc_cal, alpha, args.mondrian_groups)
    m_mond, g_te = mondrian_apply(cen_te, pexc_te, edges, Qg, q_global)
    m_glob = cen_te + q_global
    hw_mond = (m_mond - cen_te).detach().cpu().numpy()
    cov_mond = float((dte <= m_mond[:, None].numpy()).mean())
    cov_glob = float((dte <= m_glob[:, None].numpy()).mean())
    print("\n" + "=" * 68)
    print(f"MONDRIAN ADAPTIVE MARGIN  (center = q̂_{{{1-alpha:.2f}}}(s), exceedance τ={m_tau:g}, "
          f"{len(edges)+1} groups, target coverage {1-alpha:.2f})")
    print("=" * 68)
    print(f"  Mondrian : coverage {cov_mond:.4f} | mean margin {float(m_mond.mean()):.3f}")
    print(f"  global   : coverage {cov_glob:.4f} | mean margin {float(m_glob.mean()):.3f} "
          f"| Q_global {q_global:.4f}")

    emp_exc_all = _stat_exceedance(m_tau)(dte)
    m_mond_np = m_mond.detach().cpu().numpy()
    m_glob_np = m_glob.numpy()
    group_rows = []
    print(f"  {'grp':<4}{'n_cal':<7}{'n_test':<7}{'Q_g':<9}{'emp_exc(test)':<15}"
          f"{'cov_mond':<10}{'cov_glob':<10}{'ci95':<8}")
    for g in range(len(edges) + 1):
        mte = (g_te == g)
        if mte.sum() == 0:
            continue
        n_g = int(mte.sum())
        exc_lo, exc_hi = float(emp_exc_all[mte].min()), float(emp_exc_all[mte].max())
        cov_g = float((dte[mte] <= m_mond_np[mte][:, None]).mean())
        cov_g_glob = float((dte[mte] <= m_glob_np[mte][:, None]).mean())
        # 95% CI half-width, using the ANCHOR as the exchangeable unit (n_g anchors), not the
        # correlated N samples — the honest, conservative granularity for a coverage rate.
        ci95 = float(1.96 * np.sqrt(max(cov_g * (1.0 - cov_g), 1e-9) / max(n_g, 1)))
        group_rows.append({"group": g, "n_calib": counts.get(g, 0), "n_test": n_g,
                           "Q_g": float(Qg.get(g, q_global)), "emp_exc_lo": exc_lo,
                           "emp_exc_hi": exc_hi, "coverage": cov_g,
                           "coverage_global": cov_g_glob, "ci95_halfwidth": ci95})
        print(f"  {g:<4}{counts.get(g,0):<7}{n_g:<7}{Qg.get(g,q_global):<9.3f}"
              f"[{exc_lo:.2f},{exc_hi:.2f}]     {cov_g:<10.4f}{cov_g_glob:<10.4f}±{ci95:.3f}")

    # Per-PREDICTED-group coverage is the quantity Mondrian guarantees (>= 1-α per group), unlike
    # the global offset (marginal only). The safety-relevant number is the WORST group's coverage:
    # global can systematically under-protect a predicted-risk group; Mondrian bounds it per group.
    grp_min_mond = min(r["coverage"] for r in group_rows) if group_rows else float("nan")
    grp_min_glob = min(r["coverage_global"] for r in group_rows) if group_rows else float("nan")
    grp_spread_mond = (max(r["coverage"] for r in group_rows)
                       - min(r["coverage"] for r in group_rows)) if group_rows else float("nan")
    grp_spread_glob = (max(r["coverage_global"] for r in group_rows)
                       - min(r["coverage_global"] for r in group_rows)) if group_rows else float("nan")
    print(f"\n  worst predicted-group coverage: Mondrian {grp_min_mond:.4f}  vs  "
          f"global {grp_min_glob:.4f}  (higher=safer; target {1-alpha:.2f})")
    print(f"  predicted-group coverage spread: Mondrian {grp_spread_mond:.4f}  vs  "
          f"global {grp_spread_glob:.4f}  (smaller=more uniform)")

    # ---- conditional coverage by EMPIRICAL exceedance bin ----------------------
    order = np.argsort(emp_exc_all)
    ebins = [b for b in np.array_split(order, args.num_bins) if len(b) > 0]
    bin_rows = []
    print("\n--- Conditional coverage by empirical-exceedance bin (Mondrian vs global) ---")
    print(f"  {'bin':<5}{'emp_exc[lo,hi]':<20}{'cov_mond':<11}{'cov_glob':<11}{'hw_mond':<10}")
    for bi, idx in enumerate(ebins):
        e_lo, e_hi = float(emp_exc_all[idx].min()), float(emp_exc_all[idx].max())
        cm = float((dte[idx] <= m_mond_np[idx][:, None]).mean())
        cg = float((dte[idx] <= (m_glob.numpy())[idx][:, None]).mean())
        hm = float(hw_mond[idx].mean())
        bin_rows.append({"bin": bi, "emp_exc_lo": e_lo, "emp_exc_hi": e_hi, "cov_mond": cm,
                         "cov_glob": cg, "hw_mond": hm, "hw_glob": float(q_global), "n": int(len(idx))})
        print(f"  {bi:<5}[{e_lo:5.2f},{e_hi:5.2f}]      {cm:<11.4f}{cg:<11.4f}{hm:<10.3f}")

    cov_spread_mond = (max(r["cov_mond"] for r in bin_rows) - min(r["cov_mond"] for r in bin_rows)) if bin_rows else float("nan")
    cov_spread_glob = (max(r["cov_glob"] for r in bin_rows) - min(r["cov_glob"] for r in bin_rows)) if bin_rows else float("nan")
    width_exc_corr = _pearson(hw_mond, emp_exc_all)
    margin_exc_corr = _pearson(m_mond_np, emp_exc_all)
    print(f"\n  conditional-coverage spread (max-min over bins): "
          f"Mondrian {cov_spread_mond:.4f}  vs  global {cov_spread_glob:.4f}  (smaller=better)")
    print(f"  corr(Mondrian group offset, empirical exceedance) = {width_exc_corr:+.3f}")
    print(f"  corr(FULL Mondrian margin,  empirical exceedance) = {margin_exc_corr:+.3f} "
          f"(adaptivity now lives in the q̂ center + group offset)")

    # ---- direct-quantile conformal comparison table ----------------------------
    calib_y_np = calib_y.numpy()
    evalres = direct_conformal_table(
        calib_y_np, dte, alpha,
        m_mean.predict(Xcal), m_mean.predict(Xte),
        m_upper.predict(Xcal), m_upper.predict(Xte),
        m_lo.predict(Xcal), m_lo.predict(Xte),
        m_hi.predict(Xcal), m_hi.predict(Xte),
    )
    print(f"\nConformal comparison (direct quantiles) — target coverage {1-alpha:.2f}:")
    for name, row in evalres.items():
        k = "mean_margin" if "mean_margin" in row else "mean_width"
        print(f"  {name:<18} coverage {row['coverage']:.4f} | {k} {row[k]:.4f}")

    # ---- plots -----------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(reliability.keys())
        vals = [np.nan_to_num(reliability[n]["sb_reliability"]) for n in names]
        colors = ["tab:green" if reliability[n]["cleared"] else "tab:red" for n in names]
        ax.bar(names, vals, color=colors)
        ax.axhline(args.reliability_floor, ls="--", c="k", label=f"floor {args.reliability_floor}")
        ax.set_ylabel("Spearman-Brown reliability"); ax.set_title("Per-state target reliability")
        ax.legend(); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "reliability_bar.png"), dpi=180); plt.close(fig)

        htgt = headline_target or f"exceedance@{m_tau:g}"
        _scatter_identity(emp_te[htgt], target_models[htgt].predict(Xte),
                          f"empirical {htgt}", f"predicted {htgt}",
                          f"Recovery of {htgt} (r={recovery[htgt]['recovery_r']:+.3f})",
                          os.path.join(args.plot_dir, f"recovery_{htgt.replace('@','_')}.png"))

        # HEADLINE: conditional coverage by PREDICTED-exceedance group — this is the quantity the
        # Mondrian guarantee actually applies to (>= 1-α within each predicted-risk group), unlike
        # the empirical-exceedance bins (conditioning on the true outcome is not guaranteeable).
        if group_rows:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            gx = [r["group"] for r in group_rows]
            gcov = [r["coverage"] for r in group_rows]
            gcov_glob = [r["coverage_global"] for r in group_rows]
            gerr = [r["ci95_halfwidth"] for r in group_rows]
            w = 0.38
            ax.bar([g - w / 2 for g in gx], gcov, width=w, yerr=gerr, capsize=4,
                   label="Mondrian (per-group guarantee ≥ 1−α)", color="tab:blue",
                   error_kw={"ecolor": "black", "lw": 1})
            ax.bar([g + w / 2 for g in gx], gcov_glob, width=w, label="global offset (marginal only)",
                   color="tab:gray", alpha=0.85)
            ax.axhline(1 - alpha, c="k", ls="--", label=f"target {1-alpha:.2f}")
            ax.set_xticks(gx)
            ax.set_xticklabels([f"g{g}\n(pred risk↑)" for g in gx])
            ax.set_ylim(min(0.75, min(gcov + gcov_glob) - 0.05), 1.0)
            ax.set_xlabel("predicted-exceedance group (low → high risk)")
            ax.set_ylabel("conditional coverage")
            ax.set_title("Conditional coverage by predicted-risk group (95% CI on Mondrian)")
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, "coverage_by_predicted_group.png"), dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(emp_exc_all, hw_mond, s=10, alpha=0.5, label="Mondrian group offset")
        ax.axhline(q_global, c="tab:red", ls="--", label="global offset")
        ax.set_xlabel(f"empirical P(ΔG>{m_tau:g}) per state"); ax.set_ylabel("one-sided margin offset")
        ax.set_title(f"Adaptive width vs exceedance (r={width_exc_corr:+.3f})")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "width_vs_exceedance.png"), dpi=180); plt.close(fig)

        if bin_rows:
            bx = [r["bin"] for r in bin_rows]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
            a1.plot(bx, [r["cov_mond"] for r in bin_rows], "o-", label="Mondrian")
            a1.plot(bx, [r["cov_glob"] for r in bin_rows], "s-", label="global")
            a1.axhline(1 - alpha, c="k", ls="--", label=f"target {1-alpha:.2f}")
            a1.set_xlabel("empirical-exceedance bin (low→high)"); a1.set_ylabel("conditional coverage")
            a1.set_title("Conditional coverage"); a1.legend()
            a2.plot(bx, [r["hw_mond"] for r in bin_rows], "o-", label="Mondrian")
            a2.plot(bx, [r["hw_glob"] for r in bin_rows], "s-", label="global")
            a2.set_xlabel("empirical-exceedance bin (low→high)"); a2.set_ylabel("mean margin offset")
            a2.set_title("Adaptive width"); a2.legend()
            fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "sharpness_by_bin.png"), dpi=180); plt.close(fig)
        print(f"Saved plots to {args.plot_dir}")

    # ---- results json ----------------------------------------------------------
    results = {
        "predictor": "direct_hgb",
        "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "history_len": args.history_len,
                         "input_dim": extra["input_dim"]},
        "split": {"train": int(len(tr_idx)), "calibration": int(len(cal_idx)),
                  "test": int(len(te_idx)), "samples_per_state": int(N)},
        "reliability": reliability,
        "recovery": recovery,
        "headline_target": headline_target,
        "headline_recovery_r": (recovery[headline_target]["recovery_r"] if headline_target else None),
        "mondrian_margin": {
            "center": "upper_quantile",
            "tau": m_tau, "num_groups": int(len(edges) + 1), "edges": edges.tolist(),
            "q_global": q_global, "coverage_mondrian": cov_mond, "coverage_global": cov_glob,
            "mean_margin_mondrian": float(m_mond.mean()), "mean_margin_global": float(m_glob.mean()),
            "cond_coverage_spread_mondrian": cov_spread_mond,
            "cond_coverage_spread_global": cov_spread_glob,
            "width_exceedance_corr": width_exc_corr,
            "margin_exceedance_corr": margin_exc_corr,
            "worst_group_coverage_mondrian": grp_min_mond,
            "worst_group_coverage_global": grp_min_glob,
            "group_coverage_spread_mondrian": grp_spread_mond,
            "group_coverage_spread_global": grp_spread_glob,
            "groups": group_rows, "empirical_bins": bin_rows,
        },
        "conformal_comparison": evalres,
        "legacy": {"mean_recovery_r": mean_recovery_r, "std_recovery_r": std_recovery_r},
        "source_policy": extra["model_path"],
    }
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved results JSON to {args.results_json}")

    # ---- checkpoint (joblib; sklearn models) -----------------------------------
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        joblib.dump({
            "predictor": "direct_hgb", "obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
            "history_len": args.history_len, "alpha": alpha,
            "center_kind": "upper_quantile", "m_center": m_upper,
            "m_median": m_median, "exc_model": exc_model,
            "mondrian_tau": m_tau, "mondrian_edges": edges.tolist(),
            "mondrian_Q": [float(Qg.get(g, q_global)) for g in range(len(edges) + 1)],
            "mondrian_q_global": q_global, "headline_target": headline_target,
            "source_policy": extra["model_path"], "results": results,
        }, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


# ---------------------------------------------------------------------------
# Reload helper
# ---------------------------------------------------------------------------
def adaptive_certified_margin_direct(ckpt: dict, obs: np.ndarray) -> np.ndarray:
    """Mondrian certified upper margin M(s)=q̂_{1−α}(s)+Q_{g(s)} for raw stacked-obs features [B, input_dim].

    ``ckpt`` is the dict saved by ``--save-path`` (joblib). Guarantee: within each predicted-exceedance
    group, P(drop ≤ M(s)) ≥ 1−α. Center is the predicted upper quantile (falls back to the median for
    checkpoints saved before the upper-quantile centering change).
    """
    obs = np.asarray(obs, dtype=np.float64)
    center_model = ckpt.get("m_center", ckpt.get("m_median"))
    center = center_model.predict(obs)
    pexc = ckpt["exc_model"].predict(obs)
    edges = np.asarray(ckpt["mondrian_edges"], dtype=float)
    Qlist = ckpt["mondrian_Q"]
    g = np.digitize(pexc, edges)
    q = np.array([Qlist[int(gi)] if 0 <= int(gi) < len(Qlist) else ckpt["mondrian_q_global"]
                  for gi in g], dtype=np.float64)
    return center + q


if __name__ == "__main__":
    main()
