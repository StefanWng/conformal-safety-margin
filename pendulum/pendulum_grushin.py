"""
Grushin-style safety margin (tolerable perturbation count) on **Pendulum-v1**.

Port of ``safety_margin_grushin.py`` (SafetyGymnasium / SafetyPointGoal1-v0) to Pendulum-v1 with the
pretrained SB3 PPO policy ``sb3/ppo-Pendulum-v1``.

Grushin et al. (2409.18289) define a safety margin as the maximum number of consecutive random
perturbations ``n`` an agent can tolerate before its criticality (expected drop) exceeds a tolerance ζ,
with high confidence. They estimate that confidence bound with a **2-D KDE percentile**, which carries no
formal guarantee. Following the SafetyGymnasium implementation here, the density estimator is **removed
and replaced by a direct predictor**: a HistGradientBoosting conditional-quantile regressor wrapped in a
distribution-free split-conformal bound, swept over n ∈ {1,2,4,8,16,32}:

  * collect ΔG(s,n) = G_clean − G_perturbed(n) for every n (exact state-restore Monte-Carlo),
  * per n, certify M_α(s,n) = q̂_{1−α}(ΔG(s,n)) + Q(n)  with  P(ΔG(s,n) ≤ M_α(s,n)) ≥ 1−α,
  * safety margin s*(s,ζ) = max{ n : M_α(s,n') ≤ ζ for all n' ≤ n }, else 0.

Equivalence to the reliability framing in ``pendulum_adaptivity_direct.py``:
``M_α(s,n) ≤ ζ  ⟺  certified P(ΔG(s,n) > ζ) ≤ α`` (τ = ζ). Where the adaptivity driver asks *how large*
the drop can be at a fixed perturbation strength, this one asks *how much perturbation* the state can
absorb at a fixed tolerance — the margin is denominated in units of n, not return.

Pendulum is reward-only (no cost signal); the criticality target is the return drop ΔG.
Environment / policy / rollout helpers are imported from ``pendulum_adaptivity_direct.py`` and the
conformal + HGB machinery from ``adaptivity_core.py`` (numpy/sklearn only — torch-free).

Run (in the ``csc249`` env, from the ``safety_margin`` directory):
    python pendulum/pendulum_grushin.py --num-anchors 1500 --samples-per-state 48 \
        --n-list 1 2 4 8 16 32 --tolerance-list 5 25 100 \
        --raw-npz ./pendulum/runs/pendulum_grushin/raw.npz \
        --save-path ./pendulum/runs/pendulum_grushin/grushin_net.pkl \
        --plot-dir ./pendulum/runs/pendulum_grushin/plots \
        --results-json ./pendulum/runs/pendulum_grushin/results.json
"""

import argparse
import json
import os
import sys
from collections import deque
from typing import Dict, List

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Labels contain "Δ"; keep stdout robust to a non-UTF-8 Windows console codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from adaptivity_core import (  # noqa: E402
    run_parallel_collection, conformal_offset, one_per_anchor, _hgb_regressor,
)
from pendulum_adaptivity_direct import (  # noqa: E402
    ENV_ID, HF_REPO, download_policy, _load_policy, _make_core_env,
    _save_state, _restore_state, _predict, _rollout_return,
)


# ---------------------------------------------------------------------------
# Multi-n collection
# ---------------------------------------------------------------------------
def _collect_grushin_worker(model_path, num_anchors, samples_per_state, random_prob, n_list,
                            burst_mode, noise_std, gamma, max_steps, history_len, seed, counter=None):
    """At each anchor: 1 clean rollout + N perturbed rollouts FOR EACH n in n_list.

    Returns features [A, obs_dim*history_len], drops [A, |n_list|, N], g_clean [A].
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_list = [int(n) for n in n_list]
    n_vals = len(n_list)

    model = _load_policy(model_path)
    core = _make_core_env()
    core.action_space.seed(seed)
    obs_dim = core.observation_space.shape[0]
    input_dim = obs_dim * history_len

    feat_buf: List[np.ndarray] = []
    drop_buf: List[np.ndarray] = []
    gclean_buf: List[float] = []
    collected, attempts = 0, 0
    max_attempts = num_anchors * 30 + 50
    while collected < num_anchors and attempts < max_attempts:
        attempts += 1
        obs, _ = core.reset(seed=seed * 7919 + attempts)

        # Phase 1: nominal rollout with a sliding obs-history window; bounded by max_steps because
        # Pendulum has no terminal state.
        hist: deque = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(np.zeros(obs_dim, dtype=np.float32))
        hist.append(np.asarray(obs, dtype=np.float32).copy())

        anchor_state, anchor_obs, anchor_feature = None, None, None
        steps, done = 0, False
        while not done and steps < max_steps:
            if np.random.random() < random_prob:
                anchor_state = _save_state(core)
                anchor_obs = np.asarray(obs, dtype=np.float32).copy()
                anchor_feature = np.concatenate(list(hist), axis=0)
                break
            obs, _, terminated, truncated, _ = core.step(_predict(model, obs))
            hist.append(np.asarray(obs, dtype=np.float32).copy())
            steps += 1
            done = bool(terminated or truncated)
        if anchor_state is None:
            continue

        # Shared clean baseline (no perturbation).
        _restore_state(core, anchor_state)
        g_clean = _rollout_return(core, model, anchor_obs, 0, burst_mode, noise_std, rng,
                                  gamma, max_steps)

        # Perturbed rollouts for every n.
        drops = np.empty((n_vals, samples_per_state), dtype=np.float32)
        for ni, n in enumerate(n_list):
            for i in range(samples_per_state):
                _restore_state(core, anchor_state)
                core.action_space.seed(seed * 100003 + collected * 911 + ni * 7919 + i)
                g_i = _rollout_return(core, model, anchor_obs, n, burst_mode, noise_std, rng,
                                      gamma, max_steps)
                drops[ni, i] = g_clean - g_i

        feat_buf.append(anchor_feature)
        drop_buf.append(drops)
        gclean_buf.append(g_clean)
        collected += 1
        if counter is not None:
            counter.value += 1

    if not feat_buf:
        return (np.empty((0, input_dim), np.float32),
                np.empty((0, n_vals, samples_per_state), np.float32),
                np.empty(0, np.float32))
    return (np.stack(feat_buf).astype(np.float32), np.stack(drop_buf).astype(np.float32),
            np.array(gclean_buf, dtype=np.float32))


def collect_grushin_parallel(model_path, num_anchors, samples_per_state, random_prob, n_list,
                             burst_mode, noise_std, gamma, max_steps, history_len,
                             num_workers, base_seed):
    num_workers = max(1, min(num_workers or os.cpu_count() or 1, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    counts = [c for c in counts if c > 0]
    print(f"Collecting {num_anchors} anchors x {samples_per_state} samples x {len(n_list)} n-values "
          f"| n_list={n_list} | burst_mode={burst_mode} | {len(counts)} workers")
    args = [(model_path, c, samples_per_state, random_prob, n_list, burst_mode, noise_std,
             gamma, max_steps, history_len, base_seed + i) for i, c in enumerate(counts)]
    results = run_parallel_collection(_collect_grushin_worker, args, num_anchors, "Anchors", len(counts))
    feat_parts = [r[0] for r in results if r[0].shape[0] > 0]
    drop_parts = [r[1] for r in results if r[1].shape[0] > 0]
    gcl_parts = [r[2] for r in results if r[2].shape[0] > 0]
    if not feat_parts:
        raise RuntimeError("No anchors collected; raise --random-prob or --num-anchors.")
    feats = np.concatenate(feat_parts, 0)
    drops = np.concatenate(drop_parts, 0)
    g_clean = np.concatenate(gcl_parts, 0)
    obs_dim = feats.shape[1] // history_len
    print(f"Collected {feats.shape[0]} anchors | input_dim={feats.shape[1]} (obs_dim={obs_dim}) | "
          f"drops shape {drops.shape}")
    print(f"  g_clean: mean {g_clean.mean():.3f} std {g_clean.std():.3f} "
          f"[{g_clean.min():.3f}, {g_clean.max():.3f}]")
    return feats, drops, g_clean, {"model_path": model_path, "obs_dim": obs_dim,
                                   "input_dim": feats.shape[1]}


# ---------------------------------------------------------------------------
# HGB quantile predictor + margin rule
# ---------------------------------------------------------------------------
def _fit_quantile(args, X_train, drops_n_train, q):
    """HGB conditional-q quantile of ΔG(s,n) from flattened (obs repeated N, drop) pairs.

    This is the component that REPLACES the density estimator (GMM/KDE) of the original method.
    """
    N = drops_n_train.shape[1]
    X_rep = np.repeat(X_train, N, axis=0)
    y = drops_n_train.reshape(-1)
    return _hgb_regressor(args, loss="quantile", quantile=q).fit(X_rep, y)


def _calibrate(mode: str, dcal_n: np.ndarray, base_cal: np.ndarray, alpha: float,
               seed: int, reps: int) -> float:
    """One-sided conformal offset Q from calibration scores (y − q̂), under one of four schemes.

    A multi-split study on cached data (12 independent splits) showed all SAMPLE-level schemes below
    give the same mean coverage — 0.895, 95% CI [0.889, 0.902], i.e. on target — and differ only in the
    VARIANCE of Q. They are not interchangeable in what they guarantee, though:

      single  : one draw per calibration anchor. The formally exact split-conformal choice (scores are
                genuinely exchangeable) but discards N−1 of every N samples, so Q is noisy.
      multi   : average Q over ``reps`` independent one-per-anchor draws. Same target, lower variance;
                approximately valid (an average of exact-but-noisy offsets). DEFAULT.
      pooled  : use all N scores per anchor. Lowest-variance Q, but within-anchor scores are correlated,
                so the finite-sample exchangeability argument no longer applies exactly.
      cluster : per-anchor empirical (1−α) quantile of scores, then conformal over anchors. Targets a
                STRONGER, group-conditional statement — ~(1−α) of anchors have ≥(1−α) of their mass
                covered — and therefore deliberately over-covers marginally (measured ~0.943).
    """
    scores_all = dcal_n - base_cal[:, None]                 # [A_cal, N]
    if mode == "pooled":
        return conformal_offset(scores_all.reshape(-1), alpha)
    if mode == "cluster":
        return conformal_offset(np.quantile(scores_all, 1.0 - alpha, axis=1), alpha)
    if mode == "multi":
        qs = [conformal_offset(one_per_anchor(dcal_n, seed=seed + 7919 * r) - base_cal, alpha)
              for r in range(max(1, reps))]
        return float(np.mean(qs))
    return conformal_offset(one_per_anchor(dcal_n, seed=seed) - base_cal, alpha)


def _coverage_with_ci(dte_n: np.ndarray, M: np.ndarray):
    """Coverage plus a 95% CI that uses the ANCHOR as the exchangeable unit.

    Averaging over all A×N samples is unbiased for the marginal coverage but its effective sample size
    is the number of anchors, not A×N — the per-anchor coverages are what actually vary independently.
    """
    per_anchor = (dte_n <= M[:, None]).mean(axis=1)          # [A_te]
    cov = float(per_anchor.mean())
    se = float(per_anchor.std(ddof=1) / np.sqrt(len(per_anchor))) if len(per_anchor) > 1 else float("nan")
    return cov, se, 1.96 * se


def _grushin_margin(M_by_n: np.ndarray, n_list: List[int], zeta: float) -> np.ndarray:
    """Per-state safety margin: largest n such that M_α(s,n') ≤ ζ for ALL n' ≤ n (else 0).

    M_by_n is [n_vals, A]; returns int margins [A] in {0} ∪ n_list.
    """
    safe = M_by_n <= zeta                                   # [n_vals, A]
    cum = np.logical_and.accumulate(safe, axis=0)           # leading-safe mask (∀ n'≤n clause)
    num_leading = cum.sum(axis=0)                           # count of leading Trues per state
    n_arr = np.asarray(n_list)
    return np.where(num_leading >= 1, n_arr[np.clip(num_leading - 1, 0, len(n_list) - 1)], 0)


# ---------------------------------------------------------------------------
# CLI + data
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Grushin-style tolerable-perturbation-count safety margin on Pendulum-v1, with the "
                    "density estimator replaced by a conformal-certified direct HGB quantile predictor.")
    p.add_argument("--raw-npz", type=str, default=None)
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=1500)
    p.add_argument("--samples-per-state", type=int, default=48)
    p.add_argument("--random-prob", type=float, default=0.05)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                   help="Perturbation counts to sweep (the margin is denominated in these units).")
    p.add_argument("--burst-mode", type=str, choices=["uniform", "gaussian"], default="uniform")
    p.add_argument("--noise-std", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--history-len", type=int, default=1)
    p.add_argument("--tolerance-list", type=float, nargs="+", default=[5.0, 25.0, 100.0],
                   help="Tolerances ζ on ΔG (Pendulum's drop scale, not SafetyGym's cost scale).")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--calib-mode", type=str, choices=["single", "multi", "pooled", "cluster"],
                   default="multi",
                   help="Conformal calibration scheme; see _calibrate(). 'multi' = variance-reduced "
                        "average over one-per-anchor draws (default); 'cluster' = stronger "
                        "group-conditional guarantee (over-covers marginally).")
    p.add_argument("--calib-reps", type=int, default=50,
                   help="Number of one-per-anchor draws averaged when --calib-mode multi.")
    p.add_argument("--calibration-fraction", type=float, default=0.3)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument("--hgb-lr", type=float, default=0.05)
    p.add_argument("--hgb-l2", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--plot-dir", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def _load_or_collect(args):
    if args.raw_npz and os.path.exists(args.raw_npz):
        print(f"Loading raw data from {args.raw_npz}")
        d = np.load(args.raw_npz, allow_pickle=True)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        n_list = [int(n) for n in d["n_list"]]
        print(f"Loaded {d['features'].shape[0]} anchors | n_list={n_list} | drops {d['drops_r'].shape}"
              f" | source_policy {os.path.basename(extra['model_path'])}")
        return (d["features"].astype(np.float64), d["drops_r"].astype(np.float64),
                d["g_clean"].astype(np.float64), n_list, extra)

    model_path = download_policy()
    print(f"Loaded baseline policy from {HF_REPO} -> {model_path}")
    feats, drops, g_clean, extra = collect_grushin_parallel(
        model_path, args.num_anchors, args.samples_per_state, args.random_prob, args.n_list,
        args.burst_mode, args.noise_std, args.gamma, args.max_steps, args.history_len,
        args.num_workers, args.seed)
    if args.raw_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_npz)), exist_ok=True)
        np.savez_compressed(args.raw_npz, features=feats, drops_r=drops, g_clean=g_clean,
                            n_list=np.array(args.n_list), model_path=extra["model_path"],
                            obs_dim=extra["obs_dim"], input_dim=extra["input_dim"])
        print(f"Saved raw data to {args.raw_npz}")
    return (feats.astype(np.float64), drops.astype(np.float64), g_clean.astype(np.float64),
            [int(n) for n in args.n_list], extra)


def main():
    args = parse_args()
    alpha = args.alpha
    drop_label = "ΔG (return drop)"

    X, drops, g_clean, n_list, extra = _load_or_collect(args)   # drops [A, n_vals, N]
    A, n_vals, N = drops.shape
    print(f"\nTarget=reward ({drop_label}) | A={A} anchors | n_list={n_list} | N={N}")

    # ---- 3-way anchor split (shared across all n) ------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(A)
    n_te = max(1, int(A * args.test_fraction))
    n_cal = max(1, int(A * args.calibration_fraction))
    te, cal, tr = perm[:n_te], perm[n_te:n_te + n_cal], perm[n_te + n_cal:]
    Xtr, Xcal, Xte = X[tr], X[cal], X[te]
    print(f"Split: train {len(tr)} | calib {len(cal)} | test {len(te)}")

    # ---- per-n certified bound M_α(s,n) ----------------------------------------
    models: Dict[int, object] = {}
    q_one: Dict[int, float] = {}
    M_te = np.empty((n_vals, len(te)), dtype=np.float64)
    per_n = []
    print("\n" + "=" * 66)
    print(f"PER-n CERTIFIED BOUND  ({drop_label}, target coverage {1-alpha:.2f})")
    print("=" * 66)
    print(f"  calibration scheme: {args.calib_mode}"
          + (f" (reps={args.calib_reps})" if args.calib_mode == "multi" else ""))
    print(f"  {'n':<5}{'coverage':<11}{'±95%CI':<10}{'mean_margin':<13}{'Q(n)':<9}"
          f"{'mean ΔG':<10}{'q_{1-a} ΔG':<11}")
    for ni, n in enumerate(n_list):
        dtr_n, dcal_n, dte_n = drops[tr, ni, :], drops[cal, ni, :], drops[te, ni, :]
        model = _fit_quantile(args, Xtr, dtr_n, 1.0 - alpha)
        base_cal, base_te = model.predict(Xcal), model.predict(Xte)
        Q = _calibrate(args.calib_mode, dcal_n, base_cal, alpha, args.seed, args.calib_reps)
        M = base_te + Q
        M_te[ni] = M
        cov, cov_se, cov_ci = _coverage_with_ci(dte_n, M)
        models[n], q_one[n] = model, float(Q)
        emp_mean = float(dte_n.mean())
        emp_q = float(np.quantile(dte_n, 1.0 - alpha))
        per_n.append({"n": int(n), "coverage": cov, "coverage_se": cov_se,
                      "coverage_ci95_halfwidth": cov_ci,
                      "coverage_within_ci_of_target": bool(abs(cov - (1 - alpha)) <= cov_ci),
                      "mean_margin": float(M.mean()), "q_one": float(Q),
                      "emp_mean_drop": emp_mean, "emp_upper_quantile": emp_q})
        print(f"  {n:<5}{cov:<11.4f}±{cov_ci:<9.4f}{float(M.mean()):<13.3f}{Q:<9.3f}"
              f"{emp_mean:<10.3f}{emp_q:<11.3f}")
    n_off = sum(1 for r in per_n if not r["coverage_within_ci_of_target"])
    print(f"  -> {len(per_n)-n_off}/{len(per_n)} n-values within their 95% CI of the {1-alpha:.2f} target"
          f"  (all n share ONE split, so these are correlated, not independent, estimates)")

    # ---- Grushin safety margin per tolerance ζ ---------------------------------
    dte_all = drops[te]                                     # [A_te, n_vals, N]
    tol_results = []
    print("\n" + "=" * 66)
    print("SAFETY MARGIN  s*(s,ζ) = max n with M_α(s,n')≤ζ ∀n'≤n")
    print("=" * 66)
    print(f"  {'ζ':<9}{'mean margin':<13}{'frac margin=0':<15}{'validity (exc@margin≤α?)':<26}")
    for zeta in args.tolerance_list:
        margins = _grushin_margin(M_te, n_list, zeta)       # [A_te]
        # validity: among states certified to tolerate n, empirical P(ΔG(s,n) > ζ) must be ≤ α
        worst_exc = 0.0
        for ni, n in enumerate(n_list):
            cert = margins >= n
            if cert.sum() == 0:
                continue
            exc = float((dte_all[cert, ni, :] > zeta).mean())
            worst_exc = max(worst_exc, exc)
        dist = {int(m): int((margins == m).sum()) for m in ([0] + n_list)}
        tol_results.append({"tolerance": float(zeta), "mean_margin": float(margins.mean()),
                            "frac_margin_zero": float((margins == 0).mean()),
                            "worst_exceedance_at_margin": worst_exc,
                            "validity_ok": bool(worst_exc <= alpha + 0.02),
                            "margin_distribution": dist})
        ok = "OK" if worst_exc <= alpha + 0.02 else "VIOLATED"
        print(f"  {zeta:<9.2f}{float(margins.mean()):<13.3f}{float((margins==0).mean()):<15.3f}"
              f"worst {worst_exc:.3f} vs α={alpha:.2f}  [{ok}]")

    # ---- plots -----------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        ns = np.asarray(n_list)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ns, [r["emp_mean_drop"] for r in per_n], "o-", label="mean ΔG")
        ax.plot(ns, [r["emp_upper_quantile"] for r in per_n], "s-", label=f"empirical q_{{{1-alpha:.2f}}}")
        ax.plot(ns, [r["mean_margin"] for r in per_n], "^--", label="certified M_α (mean)")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel(drop_label)
        ax.set_title("Criticality vs perturbation count"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "criticality_vs_n.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [r["coverage"] for r in per_n], "o-")
        ax.axhline(1 - alpha, c="k", ls="--", label=f"target {1-alpha:.2f}")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel("conformal coverage")
        ax.set_title("Per-n certified coverage"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "coverage_by_n.png"), dpi=180)
        plt.close(fig)

        levels = [0] + n_list
        for tr_ in tol_results:
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = [tr_["margin_distribution"][m] for m in levels]
            ax.bar([str(m) for m in levels], counts, color="tab:blue")
            ax.set_xlabel("safety margin (tolerable n)"); ax.set_ylabel("test states")
            ax.set_title(f"Safety-margin distribution (ζ={tr_['tolerance']:g})")
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"margin_hist_zeta{tr_['tolerance']:g}.png"), dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([t["tolerance"] for t in tol_results], [t["mean_margin"] for t in tol_results], "o-")
        ax.set_xlabel("tolerance ζ"); ax.set_ylabel("mean safety margin")
        ax.set_title("Safety margin vs tolerance"); fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "margin_vs_tolerance.png"), dpi=180); plt.close(fig)
        print(f"Saved plots to {args.plot_dir}")

    # ---- results json ----------------------------------------------------------
    results = {
        "method": "grushin_conformal", "env": ENV_ID, "policy": HF_REPO,
        "predictor": "direct_hgb_quantile",
        "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
                         "history_len": args.history_len},
        "drop_stats": {"g_clean_mean": float(g_clean.mean()), "g_clean_std": float(g_clean.std()),
                       "g_clean_min": float(g_clean.min()), "g_clean_max": float(g_clean.max())},
        "split": {"train": int(len(tr)), "calibration": int(len(cal)), "test": int(len(te)),
                  "samples_per_state": int(N)},
        "n_list": n_list, "per_n": per_n, "tolerances": tol_results,
        "source_policy": extra["model_path"],
    }
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved results JSON to {args.results_json}")

    # ---- checkpoint ------------------------------------------------------------
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        joblib.dump({"method": "grushin_conformal", "env": ENV_ID, "policy": HF_REPO,
                     "n_list": n_list, "alpha": alpha, "target": "reward",
                     "input_dim": extra["input_dim"], "history_len": args.history_len,
                     "models": models, "q_one": q_one, "tolerances": list(args.tolerance_list),
                     "source_policy": extra["model_path"], "results": results}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


# ---------------------------------------------------------------------------
# Reload helper
# ---------------------------------------------------------------------------
def safety_margin_for_obs(ckpt: dict, obs: np.ndarray, zeta: float) -> np.ndarray:
    """Per-state Grushin safety margin (tolerable perturbation count) for raw obs [B, input_dim].

    Uses the stored per-n certified bounds M_α(s,n) = model_n(obs) + Q(n); returns int margins in
    {0} ∪ n_list, the largest n with M_α(s,n') ≤ ζ for all n' ≤ n.
    """
    obs = np.asarray(obs, dtype=np.float64)
    n_list = list(ckpt["n_list"])
    M = np.stack([ckpt["models"][n].predict(obs) + ckpt["q_one"][n] for n in n_list], axis=0)
    return _grushin_margin(M, n_list, zeta)


if __name__ == "__main__":
    main()
