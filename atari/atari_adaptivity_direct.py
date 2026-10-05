"""
Reliability-backed adaptive-width safety margin on **BeamRiderNoFrameskip-v4** — DIRECT-PREDICTOR edition.

Port of ``safety_margin_adaptivity_direct.py`` / ``pendulum_adaptivity_direct.py`` to Atari with the
pretrained ``sb3/qrdqn-BeamRiderNoFrameskip-v4`` policy. The method is identical; only the data-collection
front-end changes (image obs -> QRDQN NatureCNN 512-d embedding; ALE + wrapper state save/restore). See
``atari_core.py`` for those Atari specifics and ``test_state_restore.py`` for the determinism gate that
must pass first.

Criticality target (reward-only; Atari has no cost signal):
    ΔG(s) = G_clean(s) - G_perturbed(s)      (large drop = bad -> one-sided UPPER margin on ΔG)
The per-state ΔG distribution is induced by a leading uniform random-action burst of length
``--random-steps`` over Discrete(9); thereafter the deterministic policy runs to the horizon.

Downstream is verbatim from the SafePO / Pendulum direct driver: split-half RELIABILITY of candidate
per-state targets (std / iqr / mad / exceedance at several tau) -> direct HistGradientBoosting predictors
and a RECOVERY table -> a **Mondrian (exceedance-grouped) conformal margin** centered on the predicted
upper quantile, M(s)=q_hat_{1-alpha}(s)+Q_{g(s)}, with per-predicted-group coverage >= 1-alpha.

Run (conda env `atari`, from the `safety_margin` directory) — SMOKE:
    python atari/atari_adaptivity_direct.py \
        --num-anchors 150 --samples-per-state 24 --random-steps 8 --num-workers 12 --max-steps 150 \
        --raw-npz atari/runs/beamrider_adaptivity_smoke/raw.npz \
        --plot-dir atari/runs/beamrider_adaptivity_smoke/plots \
        --results-json atari/runs/beamrider_adaptivity_smoke/results.json
"""
import argparse
import json
import os
import sys
from typing import Dict

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from adaptivity_core import (  # noqa: E402
    _scatter_identity, one_per_anchor,
    _stat_std, _stat_iqr, _stat_mad, _stat_exceedance, _pearson, split_half_reliability,
    mondrian_calibrate, mondrian_apply, _resolve_mondrian_tau,
    _fit_flat, _fit_peranchor, direct_conformal_table,
)
from atari_core import ENV_ID, REPO, collect_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reliability-backed adaptive-width safety margin on BeamRider with DIRECT (HGB) "
                    "predictors; Mondrian conformal grouped by predicted exceedance of the return drop.")
    p.add_argument("--raw-npz", type=str, default=None)
    p.add_argument("--sweep-npz", type=str, default=None,
                   help="raw.npz of atari_grushin.py; analyse its --sweep-n slice instead of collecting.")
    p.add_argument("--sweep-n", type=int, default=8)
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=1200)
    p.add_argument("--samples-per-state", type=int, default=48)
    p.add_argument("--random-prob", type=float, default=0.02)
    p.add_argument("--random-steps", "--random-step", dest="random_steps", type=int, default=8,
                   help="Number of uniform random-action disturbance steps at the start of each "
                        "perturbed rollout.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-steps", type=int, default=200, help="Rollout horizon after the anchor.")
    p.add_argument("--anchor-max-steps", type=int, default=800,
                   help="Cap on the Phase-1 nominal rollout while searching for an anchor trigger.")
    p.add_argument("--history-len", type=int, default=1,
                   help="Fixed 1 for Atari: the 4-frame stack already encodes history and the feature "
                        "is the CNN embedding of the current stacked obs.")
    p.add_argument("--episodic-life", action="store_true",
                   help="Terminal-on-life-loss (ΔG = drop up to the first lost life). Default off: ΔG is "
                        "the full max_steps-capped episode drop.")
    p.add_argument("--proxy-features", action="store_true",
                   help="Append 2 value-spread features (max-min Q, quantile std) to the 512-d embedding.")
    p.add_argument("--device", type=str, default="cpu")
    # dispersion / risk targets
    p.add_argument("--tau-list", type=float, nargs="+", default=[200.0, 500.0, 1000.0],
                   help="Fixed thresholds tau for exceedance targets P(ΔG>tau) (BeamRider point scale).")
    p.add_argument("--tau-quantiles", type=float, nargs="+", default=[0.7, 0.85, 0.95],
                   help="If given, OVERRIDES --tau-list: derive tau from these quantiles of the pooled "
                        "ΔG. Robust to BeamRider's large/unknown reward scale.")
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


def _load_or_collect(args):
    if args.sweep_npz:
        # reuse one perturbation length of the collection over n (atari_grushin.py) instead of collecting
        d = np.load(args.sweep_npz, allow_pickle=True)
        ni = [int(n) for n in d["n_list"]].index(args.sweep_n)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        print(f"Loaded n={args.sweep_n} slice of {args.sweep_npz}: {d['features'].shape[0]} anchors")
        return (d["features"].astype(np.float64), d["drops_r"][:, ni, :].astype(np.float64),
                d["g_clean"].astype(np.float64), extra)
    if args.raw_npz and os.path.exists(args.raw_npz):
        print(f"Loading raw data from {args.raw_npz}")
        d = np.load(args.raw_npz, allow_pickle=True)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        print(f"Loaded {d['features'].shape[0]} anchors | input_dim={extra['input_dim']} | "
              f"source_policy {extra['model_path']}")
        return (d["features"].astype(np.float64), d["drops_r"].astype(np.float64),
                d["g_clean"].astype(np.float64), extra)

    feats, drops, g_clean, extra = collect_parallel(
        args.num_anchors, args.samples_per_state, args.random_prob, args.random_steps, args.gamma,
        args.max_steps, args.episodic_life, args.device, args.proxy_features, args.anchor_max_steps,
        args.num_workers, args.seed)
    feats = feats.astype(np.float64); drops = drops.astype(np.float64); g_clean = g_clean.astype(np.float64)
    if args.raw_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_npz)), exist_ok=True)
        np.savez_compressed(args.raw_npz, features=feats, drops_r=drops, g_clean=g_clean,
                            model_path=extra["model_path"], obs_dim=extra["obs_dim"],
                            input_dim=extra["input_dim"])
        print(f"Saved raw data to {args.raw_npz}")
    return feats, drops, g_clean, extra


def main() -> None:
    args = parse_args()
    drop_label = "ΔG (return drop)"

    X, drops, g_clean, extra = _load_or_collect(args)
    A, N = drops.shape
    if A < 10:
        raise RuntimeError("Need >=10 anchors.")

    # ---- resolve exceedance thresholds from the pooled ΔG (unknown BeamRider scale) -----------
    if args.tau_quantiles:
        tau_list = [float(np.quantile(drops, q)) for q in args.tau_quantiles]
        tau_list = sorted(set(round(t, 6) for t in tau_list))
        print(f"tau from ΔG quantiles {args.tau_quantiles} -> {[f'{t:.2f}' for t in tau_list]}")
    else:
        tau_list = list(args.tau_list)
    args.tau_list = tau_list

    # ---- reliability of candidate targets (model-free, on ALL anchors) ------------------------
    stat_fns = {"std": _stat_std, "iqr": _stat_iqr, "mad": _stat_mad}
    for tau in tau_list:
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

    # ---- 3-way anchor split -------------------------------------------------------------------
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

    # ---- direct predictors --------------------------------------------------------------------
    alpha = args.alpha
    print("Fitting HGB predictors (median / upper / lo / hi / mean) ...")
    m_median = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=0.5)
    m_upper = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=1.0 - alpha)
    m_lo = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=alpha / 2.0)
    m_hi = _fit_flat(args, Xtr, dtr, loss="quantile", quantile=1.0 - alpha / 2.0)
    m_mean = _fit_flat(args, Xtr, dtr, loss="squared_error")

    emp_tr = {name: fn(dtr) for name, fn in stat_fns.items()}
    emp_te = {name: fn(dte) for name, fn in stat_fns.items()}
    target_models = {name: _fit_peranchor(args, Xtr, emp_tr[name]) for name in stat_fns}

    # ---- recovery of each target on TEST ------------------------------------------------------
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

    mean_recovery_r = _pearson(dte.mean(axis=1), _fit_peranchor(args, Xtr, dtr.mean(axis=1)).predict(Xte))
    std_recovery_r = recovery["std"]["recovery_r"]
    print(f"Legacy: mean_recovery_r={mean_recovery_r:+.3f} | std_recovery_r={std_recovery_r:+.3f}")

    cleared = [k for k in stat_fns if reliability[k]["cleared"]]
    headline_target = max(cleared, key=lambda k: recovery[k]["recovery_r"]) if cleared else None
    print(f"\nHeadline adaptive-width metric: {headline_target or 'NONE cleared'}"
          + (f"  recovery_r={recovery[headline_target]['recovery_r']:+.3f}" if headline_target else ""))

    # ---- Mondrian (exceedance-grouped) conformal margin ---------------------------------------
    m_tau = _resolve_mondrian_tau(args.mondrian_tau, reliability, tau_list, headline_target)
    exc_model = target_models[f"exceedance@{m_tau:g}"]

    calib_y = one_per_anchor(dcal, seed=args.seed)
    cen_cal = m_upper.predict(Xcal)
    cen_te = m_upper.predict(Xte)
    pexc_cal = exc_model.predict(Xcal)
    pexc_te = exc_model.predict(Xte)

    edges, Qg, q_global, counts = mondrian_calibrate(calib_y, cen_cal, pexc_cal, alpha, args.mondrian_groups)
    m_mond, g_te = mondrian_apply(cen_te, pexc_te, edges, Qg, q_global)
    m_glob = cen_te + q_global
    hw_mond = m_mond - cen_te
    cov_mond = float((dte <= m_mond[:, None]).mean())
    cov_glob = float((dte <= m_glob[:, None]).mean())
    print("\n" + "=" * 68)
    print(f"MONDRIAN ADAPTIVE MARGIN  (center = q_hat_{{{1-alpha:.2f}}}(s), exceedance tau={m_tau:g}, "
          f"{len(edges)+1} groups, target coverage {1-alpha:.2f})")
    print("=" * 68)
    print(f"  Mondrian : coverage {cov_mond:.4f} | mean margin {float(m_mond.mean()):.3f}")
    print(f"  global   : coverage {cov_glob:.4f} | mean margin {float(m_glob.mean()):.3f} "
          f"| Q_global {q_global:.4f}")

    emp_exc_all = _stat_exceedance(m_tau)(dte)
    group_rows = []
    print(f"  {'grp':<4}{'n_cal':<7}{'n_test':<7}{'Q_g':<9}{'emp_exc(test)':<15}"
          f"{'cov_mond':<10}{'cov_glob':<10}{'ci95':<8}")
    for g in range(len(edges) + 1):
        mte = (g_te == g)
        if mte.sum() == 0:
            continue
        n_g = int(mte.sum())
        exc_lo, exc_hi = float(emp_exc_all[mte].min()), float(emp_exc_all[mte].max())
        cov_g = float((dte[mte] <= m_mond[mte][:, None]).mean())
        cov_g_glob = float((dte[mte] <= m_glob[mte][:, None]).mean())
        ci95 = float(1.96 * np.sqrt(max(cov_g * (1.0 - cov_g), 1e-9) / max(n_g, 1)))
        group_rows.append({"group": g, "n_calib": counts.get(g, 0), "n_test": n_g,
                           "Q_g": float(Qg.get(g, q_global)), "emp_exc_lo": exc_lo,
                           "emp_exc_hi": exc_hi, "coverage": cov_g,
                           "coverage_global": cov_g_glob, "ci95_halfwidth": ci95})
        print(f"  {g:<4}{counts.get(g,0):<7}{n_g:<7}{Qg.get(g,q_global):<9.3f}"
              f"[{exc_lo:.2f},{exc_hi:.2f}]     {cov_g:<10.4f}{cov_g_glob:<10.4f}±{ci95:.3f}")

    grp_min_mond = min((r["coverage"] for r in group_rows), default=float("nan"))
    grp_min_glob = min((r["coverage_global"] for r in group_rows), default=float("nan"))
    grp_spread_mond = ((max(r["coverage"] for r in group_rows) - grp_min_mond) if group_rows else float("nan"))
    grp_spread_glob = ((max(r["coverage_global"] for r in group_rows) - grp_min_glob) if group_rows else float("nan"))
    print(f"\n  worst predicted-group coverage: Mondrian {grp_min_mond:.4f}  vs  "
          f"global {grp_min_glob:.4f}  (higher=safer; target {1-alpha:.2f})")
    print(f"  predicted-group coverage spread: Mondrian {grp_spread_mond:.4f}  vs  "
          f"global {grp_spread_glob:.4f}  (smaller=more uniform)")

    # ---- conditional coverage by EMPIRICAL exceedance bin -------------------------------------
    order = np.argsort(emp_exc_all)
    ebins = [b for b in np.array_split(order, args.num_bins) if len(b) > 0]
    bin_rows = []
    print("\n--- Conditional coverage by empirical-exceedance bin (Mondrian vs global) ---")
    print(f"  {'bin':<5}{'emp_exc[lo,hi]':<20}{'cov_mond':<11}{'cov_glob':<11}{'hw_mond':<10}")
    for bi, idx in enumerate(ebins):
        e_lo, e_hi = float(emp_exc_all[idx].min()), float(emp_exc_all[idx].max())
        cm = float((dte[idx] <= m_mond[idx][:, None]).mean())
        cg = float((dte[idx] <= m_glob[idx][:, None]).mean())
        hm = float(hw_mond[idx].mean())
        bin_rows.append({"bin": bi, "emp_exc_lo": e_lo, "emp_exc_hi": e_hi, "cov_mond": cm,
                         "cov_glob": cg, "hw_mond": hm, "hw_glob": float(q_global), "n": int(len(idx))})
        print(f"  {bi:<5}[{e_lo:5.2f},{e_hi:5.2f}]      {cm:<11.4f}{cg:<11.4f}{hm:<10.3f}")

    cov_spread_mond = ((max(r["cov_mond"] for r in bin_rows) - min(r["cov_mond"] for r in bin_rows))
                       if bin_rows else float("nan"))
    cov_spread_glob = ((max(r["cov_glob"] for r in bin_rows) - min(r["cov_glob"] for r in bin_rows))
                       if bin_rows else float("nan"))
    width_exc_corr = _pearson(hw_mond, emp_exc_all)
    margin_exc_corr = _pearson(m_mond, emp_exc_all)
    print(f"\n  conditional-coverage spread (max-min over bins): "
          f"Mondrian {cov_spread_mond:.4f}  vs  global {cov_spread_glob:.4f}  (smaller=better)")
    print(f"  corr(Mondrian group offset, empirical exceedance) = {width_exc_corr:+.3f}")
    print(f"  corr(FULL Mondrian margin,  empirical exceedance) = {margin_exc_corr:+.3f}")

    # ---- direct-quantile conformal comparison table -------------------------------------------
    evalres = direct_conformal_table(
        calib_y, dte, alpha,
        m_mean.predict(Xcal), m_mean.predict(Xte),
        m_upper.predict(Xcal), m_upper.predict(Xte),
        m_lo.predict(Xcal), m_lo.predict(Xte),
        m_hi.predict(Xcal), m_hi.predict(Xte),
    )
    print(f"\nConformal comparison (direct quantiles) — target coverage {1-alpha:.2f}:")
    for name, row in evalres.items():
        k = "mean_margin" if "mean_margin" in row else "mean_width"
        print(f"  {name:<18} coverage {row['coverage']:.4f} | {k} {row[k]:.4f}")

    # ---- plots --------------------------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(reliability.keys())
        vals = [np.nan_to_num(reliability[n]["sb_reliability"]) for n in names]
        colors = ["tab:green" if reliability[n]["cleared"] else "tab:red" for n in names]
        ax.bar(names, vals, color=colors)
        ax.axhline(args.reliability_floor, ls="--", c="k", label=f"floor {args.reliability_floor}")
        ax.set_ylabel("Spearman-Brown reliability"); ax.set_title("Per-state target reliability (BeamRider)")
        ax.legend(); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "reliability_bar.png"), dpi=180); plt.close(fig)

        htgt = headline_target or f"exceedance@{m_tau:g}"
        _scatter_identity(emp_te[htgt], target_models[htgt].predict(Xte),
                          f"empirical {htgt}", f"predicted {htgt}",
                          f"Recovery of {htgt} (r={recovery[htgt]['recovery_r']:+.3f})",
                          os.path.join(args.plot_dir, f"recovery_{htgt.replace('@','_')}.png"))

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.hist(drops.reshape(-1), bins=80, color="steelblue", edgecolor="none")
        a1.set_xlabel("ΔG (return drop)"); a1.set_ylabel("count")
        a1.set_title(f"Pooled ΔG (mean {drops.mean():.1f}, std {drops.std():.1f})")
        for t in tau_list:
            a1.axvline(t, c="tab:red", ls="--", lw=1)
        a2.hist(g_clean, bins=60, color="seagreen", edgecolor="none")
        a2.set_xlabel("g_clean (clean return from anchor)"); a2.set_ylabel("count")
        a2.set_title(f"g_clean spread (std {g_clean.std():.1f})")
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "drop_distribution.png"), dpi=180)
        plt.close(fig)

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
            ax.set_xticks(gx); ax.set_xticklabels([f"g{g}\n(pred risk↑)" for g in gx])
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

    # ---- results json -------------------------------------------------------------------------
    results = {
        "env": ENV_ID, "policy": REPO, "predictor": "direct_hgb",
        "perturbation": "action_burst_uniform",
        "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "history_len": args.history_len,
                         "input_dim": extra["input_dim"], "feature": "qrdqn_naturecnn_embedding"},
        "drop_stats": {"g_clean_mean": float(g_clean.mean()), "g_clean_std": float(g_clean.std()),
                       "g_clean_min": float(g_clean.min()), "g_clean_max": float(g_clean.max()),
                       "drop_mean": float(drops.mean()), "drop_std": float(drops.std()),
                       "drop_min": float(drops.min()), "drop_max": float(drops.max())},
        "split": {"train": int(len(tr_idx)), "calibration": int(len(cal_idx)),
                  "test": int(len(te_idx)), "samples_per_state": int(N)},
        "reliability": reliability, "recovery": recovery,
        "headline_target": headline_target,
        "headline_recovery_r": (recovery[headline_target]["recovery_r"] if headline_target else None),
        "mondrian_margin": {
            "center": "upper_quantile", "tau": m_tau, "num_groups": int(len(edges) + 1),
            "edges": edges.tolist(), "q_global": q_global, "coverage_mondrian": cov_mond,
            "coverage_global": cov_glob, "mean_margin_mondrian": float(m_mond.mean()),
            "mean_margin_global": float(m_glob.mean()),
            "cond_coverage_spread_mondrian": cov_spread_mond,
            "cond_coverage_spread_global": cov_spread_glob,
            "width_exceedance_corr": width_exc_corr, "margin_exceedance_corr": margin_exc_corr,
            "worst_group_coverage_mondrian": grp_min_mond, "worst_group_coverage_global": grp_min_glob,
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

    # ---- checkpoint ---------------------------------------------------------------------------
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        joblib.dump({
            "env": ENV_ID, "policy": REPO, "predictor": "direct_hgb",
            "perturbation": "action_burst_uniform",
            "obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
            "history_len": args.history_len, "alpha": alpha,
            "center_kind": "upper_quantile", "m_center": m_upper, "m_median": m_median,
            "exc_model": exc_model, "mondrian_tau": m_tau, "mondrian_edges": edges.tolist(),
            "mondrian_Q": [float(Qg.get(g, q_global)) for g in range(len(edges) + 1)],
            "mondrian_q_global": q_global, "headline_target": headline_target,
            "source_policy": extra["model_path"], "results": results,
        }, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


def adaptive_certified_margin_direct(ckpt: dict, obs: np.ndarray) -> np.ndarray:
    """Mondrian certified upper margin M(s)=q_hat_{1-alpha}(s)+Q_{g(s)} for embedded features [B, input_dim].

    Note: ``obs`` here is the QRDQN NatureCNN EMBEDDING (use ``atari_core.embed``), not the raw image.
    Guarantee: within each predicted-exceedance group, P(ΔG ≤ M(s)) ≥ 1−α.
    """
    obs = np.asarray(obs, dtype=np.float64)
    center = ckpt.get("m_center", ckpt.get("m_median")).predict(obs)
    pexc = ckpt["exc_model"].predict(obs)
    edges = np.asarray(ckpt["mondrian_edges"], dtype=float)
    Qlist = ckpt["mondrian_Q"]
    g = np.digitize(pexc, edges)
    q = np.array([Qlist[int(gi)] if 0 <= int(gi) < len(Qlist) else ckpt["mondrian_q_global"]
                  for gi in g], dtype=np.float64)
    return center + q


if __name__ == "__main__":
    main()
