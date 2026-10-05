"""
Grushin-style safety margin (tolerable perturbation count) on **CartPole-v1**.

Port of ``safety_margin_grushin.py`` (SafetyGymnasium) and ``pendulum/pendulum_grushin.py`` to CartPole,
with the pretrained SB3 PPO policy ``sb3/ppo-CartPole-v1``.

Grushin et al. (2409.18289) define a safety margin as the maximum number of consecutive random
perturbations ``n`` an agent can tolerate before its criticality exceeds a tolerance ζ, with high
confidence. They estimate that bound with a 2-D KDE percentile (no formal guarantee); following the
SafetyGymnasium implementation, the density estimator is **removed and replaced by a direct predictor** —
a HistGradientBoosting conditional-quantile regressor wrapped in a distribution-free split-conformal
bound, swept over n ∈ {1,2,4,8,16,32}:

  * per n, certify M_α(s,n) = q̂_{1−α}(Δ(s,n)) + Q(n)  with  P(Δ(s,n) ≤ M_α(s,n)) ≥ 1−α,
  * safety margin s*(s,ζ) = max{ n : M_α(s,n') ≤ ζ for all n' ≤ n }, else 0.

CRITICALITY TARGET. CartPole's reward drop ΔG is DEGENERATE (bimodal: recover≈0 or fall≈cap), so its
Grushin bound would jump between two values. We instead default to the **shaped state-based cost** ΔC
that broke the degeneracy in ``cartpole_adaptivity_direct.py`` (continuous, reliable, graded per state):
``c_t = min(1, max(|x|/x_thr, |θ|/θ_thr))`` (``--cost-shape proximity``), ΔC = discounted cost increase vs
the clean baseline. Both channels are collected & cached, and ``--target {reward,cost}`` selects which the
analysis uses (reward available for the contrast).

TERMINAL FILL (default ON for Grushin, opposite of the adaptivity driver). Without filling the
post-failure horizon, heavy perturbation makes the pole fall SOONER, which truncates the cost integral, so
ΔC becomes NON-monotone in n (rises then falls) and breaks Grushin's premise. Filling the remaining horizon
with cost 1.0 (absorbing-unsafe) keeps criticality monotone increasing in n. Pass ``--no-terminal-fill``
for the continuous-but-non-monotone variant.

WHY THIS IS INTERESTING. The single-level adaptive *margin* on ΔC was nearly flat (its upper tail is
homoscedastic). Grushin's tolerable-n aggregates across perturbation levels, so it may recover per-state
adaptivity that the single-level method lacked — different states cross ζ at different n. The premise
Grushin needs (criticality MONOTONE increasing in n) should hold on CartPole cost (more random steps =
more disturbance = more cost), unlike SafePO cost where random actions REDUCED cost and the method collapsed.

Env / policy / rollout / cost helpers are imported from ``cartpole_adaptivity_direct.py``; the conformal +
HGB machinery from ``adaptivity_core.py`` / ``gmm_core.py``.

Run (in the ``csc249`` env, from the ``safety_margin`` directory):
    python cartpole/cartpole_grushin.py --num-anchors 5000 --samples-per-state 64 \
        --n-list 1 2 4 8 16 32 --target cost --tolerance-quantiles 0.6 0.8 0.9 0.95 \
        --raw-npz ./cartpole/runs/cartpole_grushin/raw.npz \
        --save-path ./cartpole/runs/cartpole_grushin/grushin_net.pkl \
        --plot-dir ./cartpole/runs/cartpole_grushin/plots \
        --results-json ./cartpole/runs/cartpole_grushin/results.json
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

import torch  # noqa: E402  (gmm_core.one_per_anchor is torch-based)

from gmm_core import run_parallel_collection, conformal_offset, one_per_anchor  # noqa: E402
from adaptivity_core import _hgb_regressor  # noqa: E402
from cartpole_adaptivity_direct import (  # noqa: E402
    ENV_ID, HF_REPO, download_policy, _load_policy, _make_core_env,
    _save_state, _restore_state, _predict, _rollout,
)


def _opa_np(samples: np.ndarray, seed: int) -> np.ndarray:
    """One-per-anchor draw, numpy in/out (wraps gmm_core's torch implementation)."""
    return one_per_anchor(torch.as_tensor(samples, dtype=torch.float32), seed=seed).numpy()


# ---------------------------------------------------------------------------
# Multi-n dual-channel collection
# ---------------------------------------------------------------------------
def _collect_grushin_worker(model_path, num_anchors, samples_per_state, random_prob, n_list,
                            gamma, max_steps, history_len, cost_shape, cost_kappa, terminal_fill,
                            seed, counter=None):
    """At each anchor: 1 clean rollout + N perturbed rollouts FOR EACH n in n_list, BOTH channels.

    Returns features [A, obs_dim*history_len], drops_r/drops_c [A, |n_list|, N], g_clean_r/g_clean_c [A].
    """
    np.random.seed(seed)
    n_list = [int(n) for n in n_list]
    n_vals = len(n_list)

    model = _load_policy(model_path)
    core = _make_core_env()
    core.action_space.seed(seed)
    obs_dim = core.observation_space.shape[0]
    input_dim = obs_dim * history_len
    x_thr, th_thr = float(core.x_threshold), float(core.theta_threshold_radians)

    def roll(start_obs, rsteps):
        return _rollout(core, model, start_obs, rsteps, gamma, max_steps,
                        x_thr, th_thr, cost_shape, cost_kappa, terminal_fill)

    feat_buf, dr_buf, dc_buf, gcr_buf, gcc_buf = [], [], [], [], []
    collected, attempts = 0, 0
    max_attempts = num_anchors * 30 + 50
    while collected < num_anchors and attempts < max_attempts:
        attempts += 1
        obs, _ = core.reset(seed=seed * 7919 + attempts)

        # Phase 1: nominal rollout, tracking the last history_len obs. CartPole terminates on failure,
        # so `while not done` is safe; the random_prob trigger fires within ~1/random_prob steps.
        hist: deque = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(np.zeros(obs_dim, dtype=np.float32))
        hist.append(np.asarray(obs, dtype=np.float32).copy())

        anchor_state, anchor_obs, anchor_feature = None, None, None
        done = False
        while not done:
            if np.random.random() < random_prob:
                anchor_state = _save_state(core)
                anchor_obs = np.asarray(obs, dtype=np.float32).copy()
                anchor_feature = np.concatenate(list(hist), axis=0)
                break
            obs, _, terminated, truncated, _ = core.step(_predict(model, obs))
            hist.append(np.asarray(obs, dtype=np.float32).copy())
            done = bool(terminated or truncated)
        if anchor_state is None:
            continue

        # Shared clean baseline (no perturbation) -> reward & cost.
        _restore_state(core, anchor_state)
        g_r_clean, g_c_clean = roll(anchor_obs, 0)

        # Perturbed rollouts for every n; drops_r = reward drop, drops_c = cost increase (SafePO sign).
        drops_r = np.empty((n_vals, samples_per_state), dtype=np.float32)
        drops_c = np.empty((n_vals, samples_per_state), dtype=np.float32)
        for ni, n in enumerate(n_list):
            for i in range(samples_per_state):
                _restore_state(core, anchor_state)
                core.action_space.seed(seed * 100003 + collected * 911 + ni * 7919 + i)
                g_r_i, g_c_i = roll(anchor_obs, n)
                drops_r[ni, i] = g_r_clean - g_r_i
                drops_c[ni, i] = g_c_i - g_c_clean

        feat_buf.append(anchor_feature)
        dr_buf.append(drops_r); dc_buf.append(drops_c)
        gcr_buf.append(g_r_clean); gcc_buf.append(g_c_clean)
        collected += 1
        if counter is not None:
            counter.value += 1

    if not feat_buf:
        empty = np.empty((0, n_vals, samples_per_state), np.float32)
        return (np.empty((0, input_dim), np.float32), empty, empty,
                np.empty(0, np.float32), np.empty(0, np.float32))
    return (np.stack(feat_buf).astype(np.float32),
            np.stack(dr_buf).astype(np.float32), np.stack(dc_buf).astype(np.float32),
            np.array(gcr_buf, np.float32), np.array(gcc_buf, np.float32))


def collect_grushin_parallel(model_path, num_anchors, samples_per_state, random_prob, n_list,
                             gamma, max_steps, history_len, cost_shape, cost_kappa, terminal_fill,
                             num_workers, base_seed):
    num_workers = max(1, min(num_workers or os.cpu_count() or 1, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    counts = [c for c in counts if c > 0]
    print(f"Collecting {num_anchors} anchors x {samples_per_state} samples x {len(n_list)} n-values "
          f"| n_list={n_list} | cost_shape={cost_shape} terminal_fill={terminal_fill} | {len(counts)} workers")
    args = [(model_path, c, samples_per_state, random_prob, n_list, gamma, max_steps, history_len,
             cost_shape, cost_kappa, terminal_fill, base_seed + i) for i, c in enumerate(counts)]
    results = run_parallel_collection(_collect_grushin_worker, args, num_anchors, "Anchors", len(counts))
    feat_parts = [r[0] for r in results if r[0].shape[0] > 0]
    if not feat_parts:
        raise RuntimeError("No anchors collected; raise --random-prob or --num-anchors.")
    feats = np.concatenate(feat_parts, 0)
    drops_r = np.concatenate([r[1] for r in results if r[1].shape[0] > 0], 0)
    drops_c = np.concatenate([r[2] for r in results if r[2].shape[0] > 0], 0)
    g_clean_r = np.concatenate([r[3] for r in results if r[3].shape[0] > 0], 0)
    g_clean_c = np.concatenate([r[4] for r in results if r[4].shape[0] > 0], 0)
    obs_dim = feats.shape[1] // history_len
    print(f"Collected {feats.shape[0]} anchors | input_dim={feats.shape[1]} (obs_dim={obs_dim}) | "
          f"drops shape {drops_c.shape}")
    print(f"  reward: g_clean std {g_clean_r.std():.3f} | ΔG mean {drops_r.mean():.3f}")
    print(f"  cost  : g_clean std {g_clean_c.std():.3f} | ΔC mean {drops_c.mean():.3f}")
    return feats, drops_r, drops_c, g_clean_r, g_clean_c, {
        "model_path": model_path, "obs_dim": obs_dim, "input_dim": feats.shape[1]}


# ---------------------------------------------------------------------------
# HGB quantile predictor + conformal calibration + margin rule
# ---------------------------------------------------------------------------
def _fit_quantile(args, X_train, drops_n_train, q):
    """HGB conditional-q quantile of Δ(s,n) — the piece REPLACING the density estimator (GMM/KDE)."""
    N = drops_n_train.shape[1]
    X_rep = np.repeat(X_train, N, axis=0)
    return _hgb_regressor(args, loss="quantile", quantile=q).fit(X_rep, drops_n_train.reshape(-1))


def _calibrate(mode: str, dcal_n: np.ndarray, base_cal: np.ndarray, alpha: float,
               seed: int, reps: int) -> float:
    """One-sided conformal offset Q from calibration scores (y − q̂). See pendulum_grushin for the
    single/multi/pooled/cluster trade-off; 'multi' (variance-reduced one-per-anchor average) is default."""
    scores_all = dcal_n - base_cal[:, None]
    if mode == "pooled":
        return conformal_offset(scores_all.reshape(-1), alpha)
    if mode == "cluster":
        return conformal_offset(np.quantile(scores_all, 1.0 - alpha, axis=1), alpha)
    if mode == "multi":
        qs = [conformal_offset(_opa_np(dcal_n, seed + 7919 * r) - base_cal, alpha)
              for r in range(max(1, reps))]
        return float(np.mean(qs))
    return conformal_offset(_opa_np(dcal_n, seed) - base_cal, alpha)


def _coverage_with_ci(dte_n: np.ndarray, M: np.ndarray):
    """Coverage + 95% CI using the ANCHOR as the exchangeable unit (effective n = #anchors)."""
    per_anchor = (dte_n <= M[:, None]).mean(axis=1)
    cov = float(per_anchor.mean())
    se = float(per_anchor.std(ddof=1) / np.sqrt(len(per_anchor))) if len(per_anchor) > 1 else float("nan")
    return cov, se, 1.96 * se


def _grushin_margin(M_by_n: np.ndarray, n_list: List[int], zeta: float) -> np.ndarray:
    """Per-state safety margin: largest n such that M_α(s,n') ≤ ζ for ALL n' ≤ n (else 0)."""
    safe = M_by_n <= zeta
    cum = np.logical_and.accumulate(safe, axis=0)
    num_leading = cum.sum(axis=0)
    n_arr = np.asarray(n_list)
    return np.where(num_leading >= 1, n_arr[np.clip(num_leading - 1, 0, len(n_list) - 1)], 0)


# ---------------------------------------------------------------------------
# CLI + data
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Grushin-style tolerable-perturbation-count safety margin on CartPole-v1, density "
                    "estimator replaced by a conformal-certified direct HGB quantile predictor.")
    p.add_argument("--raw-npz", type=str, default=None)
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=2000)
    p.add_argument("--samples-per-state", type=int, default=64)
    p.add_argument("--random-prob", type=float, default=0.05)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                   help="Perturbation counts to sweep (the margin is denominated in these units).")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--history-len", type=int, default=1)
    # criticality target (shaped cost by default; reward drop is bimodal-degenerate)
    p.add_argument("--target", type=str, choices=["reward", "cost"], default="cost")
    p.add_argument("--cost-shape", type=str, choices=["proximity", "zone"], default="proximity")
    p.add_argument("--cost-zone-kappa", type=float, default=0.5)
    p.add_argument("--no-terminal-fill", dest="terminal_fill", action="store_false",
                   help="Do NOT fill the post-failure horizon with cost 1.0. DEFAULT IS FILL for Grushin: "
                        "without it, heavy perturbation makes the pole fall sooner and TRUNCATES the cost "
                        "integral, so ΔC becomes NON-monotone in n (rises then falls ~n=8) and breaks "
                        "Grushin's premise. Filling the post-failure horizon (absorbing-unsafe) keeps "
                        "criticality monotone increasing in n. (The single-level adaptivity driver prefers "
                        "no-fill for continuity; the n-sweep prefers fill for monotonicity.)")
    p.set_defaults(terminal_fill=True)
    # tolerances (ΔC scale unknown a priori -> derive from quantiles)
    p.add_argument("--tolerance-list", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0],
                   help="Fixed tolerances ζ on the selected drop.")
    p.add_argument("--tolerance-quantiles", type=float, nargs="+", default=None,
                   help="If given, OVERRIDES --tolerance-list: derive ζ from these quantiles of the "
                        "pooled drop (e.g. 0.6 0.8 0.9 0.95).")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--calib-mode", type=str, choices=["single", "multi", "pooled", "cluster"],
                   default="multi")
    p.add_argument("--calib-reps", type=int, default=50)
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


def _select(args, drops_r, drops_c, g_clean_r, g_clean_c):
    if args.target == "cost":
        if drops_c is None:
            raise RuntimeError("This raw npz predates cost collection (no 'drops_c'); recollect.")
        return drops_c, g_clean_c
    return drops_r, g_clean_r


def _load_or_collect(args):
    if args.raw_npz and os.path.exists(args.raw_npz):
        print(f"Loading raw data from {args.raw_npz}")
        d = np.load(args.raw_npz, allow_pickle=True)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        n_list = [int(n) for n in d["n_list"]]
        keys = set(d.files)
        drops_c = d["drops_c"].astype(np.float64) if "drops_c" in keys else None
        g_clean_c = d["g_clean_c"].astype(np.float64) if "g_clean_c" in keys else None
        g_clean_r = (d["g_clean_r"] if "g_clean_r" in keys else d["g_clean"]).astype(np.float64)
        print(f"Loaded {d['features'].shape[0]} anchors | n_list={n_list} | drops {d['drops_r'].shape} | "
              f"channels={'reward+cost' if drops_c is not None else 'reward-only'} | "
              f"source_policy {os.path.basename(extra['model_path'])}")
        drops, g_clean = _select(args, d["drops_r"].astype(np.float64), drops_c, g_clean_r, g_clean_c)
        return d["features"].astype(np.float64), drops, g_clean, n_list, extra

    model_path = download_policy()
    print(f"Loaded baseline policy from {HF_REPO} -> {model_path}")
    feats, dr, dc, gcr, gcc, extra = collect_grushin_parallel(
        model_path, args.num_anchors, args.samples_per_state, args.random_prob, args.n_list,
        args.gamma, args.max_steps, args.history_len, args.cost_shape, args.cost_zone_kappa,
        args.terminal_fill, args.num_workers, args.seed)
    if args.raw_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_npz)), exist_ok=True)
        np.savez_compressed(args.raw_npz, features=feats, drops_r=dr, drops_c=dc,
                            g_clean_r=gcr, g_clean_c=gcc, n_list=np.array(args.n_list),
                            cost_shape=args.cost_shape, terminal_fill=args.terminal_fill,
                            model_path=extra["model_path"], obs_dim=extra["obs_dim"],
                            input_dim=extra["input_dim"])
        print(f"Saved raw data to {args.raw_npz}")
    drops, g_clean = _select(args, dr.astype(np.float64), dc.astype(np.float64),
                             gcr.astype(np.float64), gcc.astype(np.float64))
    return feats.astype(np.float64), drops, g_clean, [int(n) for n in args.n_list], extra


def main():
    args = parse_args()
    alpha = args.alpha
    drop_label = "ΔC (shaped cost increase)" if args.target == "cost" else "ΔG (return drop)"

    X, drops, g_clean, n_list, extra = _load_or_collect(args)   # drops [A, n_vals, N]
    A, n_vals, N = drops.shape

    # resolve tolerances (ζ) — derive from pooled-drop quantiles when the scale is unknown
    if args.tolerance_quantiles:
        tol_list = sorted(set(round(float(np.quantile(drops, q)), 4) for q in args.tolerance_quantiles))
        print(f"ζ from {drop_label} quantiles {args.tolerance_quantiles} -> {tol_list}")
    else:
        tol_list = list(args.tolerance_list)
    args.tolerance_list = tol_list

    print(f"\nTarget={args.target} ({drop_label}) | A={A} anchors | n_list={n_list} | N={N}")

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
          f"{'mean Δ':<10}{'q_{1-a} Δ':<11}")
    for ni, n in enumerate(n_list):
        dtr_n, dcal_n, dte_n = drops[tr, ni, :], drops[cal, ni, :], drops[te, ni, :]
        model = _fit_quantile(args, Xtr, dtr_n, 1.0 - alpha)
        base_cal, base_te = model.predict(Xcal), model.predict(Xte)
        Q = _calibrate(args.calib_mode, dcal_n, base_cal, alpha, args.seed, args.calib_reps)
        M = base_te + Q
        M_te[ni] = M
        cov, cov_se, cov_ci = _coverage_with_ci(dte_n, M)
        models[n], q_one[n] = model, float(Q)
        per_n.append({"n": int(n), "coverage": cov, "coverage_se": cov_se,
                      "coverage_ci95_halfwidth": cov_ci,
                      "coverage_within_ci_of_target": bool(abs(cov - (1 - alpha)) <= cov_ci),
                      "mean_margin": float(M.mean()), "q_one": float(Q),
                      "emp_mean_drop": float(dte_n.mean()),
                      "emp_upper_quantile": float(np.quantile(dte_n, 1.0 - alpha))})
        print(f"  {n:<5}{cov:<11.4f}±{cov_ci:<9.4f}{float(M.mean()):<13.3f}{Q:<9.3f}"
              f"{float(dte_n.mean()):<10.3f}{float(np.quantile(dte_n,1-alpha)):<11.3f}")
    n_off = sum(1 for r in per_n if not r["coverage_within_ci_of_target"])
    print(f"  -> {len(per_n)-n_off}/{len(per_n)} n-values within their 95% CI of the {1-alpha:.2f} target"
          f"  (all n share ONE split, so these are correlated estimates)")

    # ---- Grushin safety margin per tolerance ζ ---------------------------------
    dte_all = drops[te]                                     # [A_te, n_vals, N]
    tol_results = []
    print("\n" + "=" * 66)
    print("SAFETY MARGIN  s*(s,ζ) = max n with M_α(s,n')≤ζ ∀n'≤n")
    print("=" * 66)
    print(f"  {'ζ':<9}{'mean margin':<13}{'frac margin=0':<15}{'validity (exc@margin≤α?)':<26}")
    for zeta in tol_list:
        margins = _grushin_margin(M_te, n_list, zeta)
        worst_exc = 0.0
        for ni, n in enumerate(n_list):
            cert = margins >= n
            if cert.sum() == 0:
                continue
            worst_exc = max(worst_exc, float((dte_all[cert, ni, :] > zeta).mean()))
        dist = {int(m): int((margins == m).sum()) for m in ([0] + n_list)}
        tol_results.append({"tolerance": float(zeta), "mean_margin": float(margins.mean()),
                            "frac_margin_zero": float((margins == 0).mean()),
                            "worst_exceedance_at_margin": worst_exc,
                            "validity_ok": bool(worst_exc <= alpha + 0.02),
                            "margin_distribution": dist})
        ok = "OK" if worst_exc <= alpha + 0.02 else "VIOLATED"
        print(f"  {zeta:<9.3f}{float(margins.mean()):<13.3f}{float((margins==0).mean()):<15.3f}"
              f"worst {worst_exc:.3f} vs α={alpha:.2f}  [{ok}]")

    # ---- plots -----------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        ns = np.asarray(n_list)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ns, [r["emp_mean_drop"] for r in per_n], "o-", label=f"mean {drop_label}")
        ax.plot(ns, [r["emp_upper_quantile"] for r in per_n], "s-", label=f"empirical q_{{{1-alpha:.2f}}}")
        ax.plot(ns, [r["mean_margin"] for r in per_n], "^--", label="certified M_α (mean)")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel(drop_label)
        ax.set_title("Criticality vs perturbation count"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "criticality_vs_n.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ns, [r["coverage"] for r in per_n],
                    yerr=[r["coverage_ci95_halfwidth"] for r in per_n], fmt="o-", capsize=4)
        ax.axhline(1 - alpha, c="k", ls="--", label=f"target {1-alpha:.2f}")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel("conformal coverage")
        ax.set_title("Per-n certified coverage (95% CI)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "coverage_by_n.png"), dpi=180)
        plt.close(fig)

        levels = [0] + n_list
        for tr_ in tol_results:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([str(m) for m in levels], [tr_["margin_distribution"][m] for m in levels], color="tab:blue")
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
        "predictor": "direct_hgb_quantile", "target": args.target, "drop_label": drop_label,
        "cost_shape": (args.cost_shape if args.target == "cost" else None),
        "terminal_fill": (args.terminal_fill if args.target == "cost" else None),
        "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
                         "history_len": args.history_len},
        "drop_stats": {"g_clean_std": float(g_clean.std()),
                       "drop_pooled_mean": float(drops.mean()), "drop_pooled_std": float(drops.std()),
                       "drop_min": float(drops.min()), "drop_max": float(drops.max())},
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
                     "n_list": n_list, "alpha": alpha, "target": args.target,
                     "cost_shape": (args.cost_shape if args.target == "cost" else None),
                     "input_dim": extra["input_dim"], "history_len": args.history_len,
                     "models": models, "q_one": q_one, "tolerances": tol_list,
                     "source_policy": extra["model_path"], "results": results}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


# ---------------------------------------------------------------------------
# Reload helper
# ---------------------------------------------------------------------------
def safety_margin_for_obs(ckpt: dict, obs: np.ndarray, zeta: float) -> np.ndarray:
    """Per-state Grushin safety margin (tolerable perturbation count) for raw obs [B, input_dim]."""
    obs = np.asarray(obs, dtype=np.float64)
    n_list = list(ckpt["n_list"])
    M = np.stack([ckpt["models"][n].predict(obs) + ckpt["q_one"][n] for n in n_list], axis=0)
    return _grushin_margin(M, n_list, zeta)


if __name__ == "__main__":
    main()
