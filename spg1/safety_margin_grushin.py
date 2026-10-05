"""
SafetyPointGoal1: certified reduction margin and safety margin swept over the perturbation length.

  * collect ΔG(s,n) = G_clean - G_perturbed(n) for every n in --n-list (state-restore Monte-Carlo),
  * per n, certify M_α(s,n) = q̂_{1-α}(s,n) + Q(n) with P(ΔG(s,n) ≤ M_α(s,n)) ≥ 1-α,
  * safety margin s*(s,ζ) = max{ n : M_α(s,n') ≤ ζ for all n' ≤ n }, else 0.

--eval-dir is a SafePO CPO run directory (config.json, torch_save/, state*.pkl).
"""

import argparse
import json
import multiprocessing
import os
import sys
import threading
from collections import deque
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import HistGradientBoostingRegressor

from spg1_core import (
    _load_actor, _get_sim, _save_sim_state, _restore_sim_state, _rollout_return_from_here,
    _rollout_return_clean, _progress_monitor, conformal_offset, one_per_anchor,
)


# ---------------------------------------------------------------------------
# Multi-n collection
# ---------------------------------------------------------------------------
def _collect_grushin_worker(eval_dir, num_anchors, samples_per_state, random_prob,
                            n_list, gamma_override, max_steps, history_len, seed, counter=None):
    """At each anchor: 1 clean rollout + N perturbed rollouts FOR EACH n in n_list.

    Returns features [A, obs_dim*history_len], drops_r [A, |n_list|, N], g_clean_r [A],
    model_path, anchor_steps.
    """
    device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_list = list(n_list)
    n_vals = len(n_list)

    policy, extra = _load_actor(eval_dir, device)
    env = extra["env"]
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    config = extra["config"]
    gamma = gamma_override if gamma_override is not None else config.get("gamma", 0.99)
    sim_model, sim_data = _get_sim(env)
    obs_dim = extra["obs_space"].shape[0]
    input_dim = obs_dim * history_len

    feat_buffer, dr_buffer, gcr_buffer, anchor_steps = [], [], [], []

    collected, attempts = 0, 0
    max_attempts = num_anchors * 20 + 50

    while collected < num_anchors and attempts < max_attempts:
        attempts += 1
        hist: deque = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(np.zeros(obs_dim, dtype=np.float32))

        obs_np, _ = env.reset()
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        obs_flat = obs_torch.view(-1).detach().cpu().numpy()
        hist.append(obs_flat.copy())

        anchor_state = anchor_obs_np = anchor_feature = None
        t0, step, done = -1, 0, False
        while not done:
            if np.random.random() < random_prob:
                anchor_state = _save_sim_state(sim_data)
                anchor_obs_np = obs_flat.copy()
                anchor_feature = np.concatenate(list(hist), axis=0)
                t0 = step
                break
            with torch.no_grad():
                act_t, _, _, _ = policy.step(obs_torch, deterministic=True)
            action = act_t.squeeze(0).detach().cpu().numpy()
            next_obs, _, _, terminated, truncated, _ = env.step(action)
            obs_torch = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            obs_flat = obs_torch.view(-1).detach().cpu().numpy()
            hist.append(obs_flat.copy())
            step += 1
            done = bool(terminated[0] or truncated[0])

        if anchor_state is None:
            continue

        # shared clean baseline
        _restore_sim_state(sim_model, sim_data, anchor_state)
        g_r_clean = _rollout_return_clean(env, policy, device, gamma, max_steps, anchor_obs_np)

        # perturbed rollouts for every n
        drops_r = np.empty((n_vals, samples_per_state), dtype=np.float32)
        for ni, n in enumerate(n_list):
            for i in range(samples_per_state):
                _restore_sim_state(sim_model, sim_data, anchor_state)
                try:
                    env.action_space.seed(seed * 100003 + collected * 911 + ni * 7919 + i)
                except Exception:
                    pass
                g_r_i, _ = _rollout_return_from_here(env, policy, device, n, gamma, max_steps)
                drops_r[ni, i] = g_r_clean - g_r_i

        feat_buffer.append(anchor_feature)
        dr_buffer.append(drops_r)
        gcr_buffer.append(g_r_clean)
        anchor_steps.append(t0)
        collected += 1
        if counter is not None:
            counter.value += 1

    if not feat_buffer:
        empty = np.empty((0, n_vals, samples_per_state), dtype=np.float32)
        return (np.empty((0, input_dim), np.float32), empty, np.empty(0, np.float32),
                extra["model_path"], [])
    return (np.stack(feat_buffer).astype(np.float32), np.stack(dr_buffer).astype(np.float32),
            np.array(gcr_buffer, np.float32), extra["model_path"], anchor_steps)


def collect_grushin_parallel(eval_dir, num_anchors, samples_per_state, random_prob, n_list,
                             gamma, max_steps, history_len, num_workers, base_seed):
    num_workers = max(1, min(num_workers or os.cpu_count() or 1, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    counts = [c for c in counts if c > 0]
    print(f"Collecting {num_anchors} anchors x {samples_per_state} samples x {len(n_list)} n-values "
          f"| n_list={n_list} | {len(counts)} workers")

    stop_event = threading.Event()
    with multiprocessing.Manager() as manager:
        counter = manager.Value("i", 0)
        with tqdm(total=num_anchors, desc="Anchors", unit="anchor") as pbar:
            monitor = threading.Thread(target=_progress_monitor,
                                       args=(counter, num_anchors, pbar, stop_event), daemon=True)
            monitor.start()
            results = joblib.Parallel(n_jobs=len(counts), backend="loky", verbose=0)(
                joblib.delayed(_collect_grushin_worker)(
                    eval_dir, c, samples_per_state, random_prob, n_list, gamma, max_steps,
                    history_len, base_seed + i, counter)
                for i, c in enumerate(counts))
            stop_event.set()
            monitor.join(timeout=2.0)

    feats = np.concatenate([r[0] for r in results if r[0].shape[0] > 0], axis=0)
    dr = np.concatenate([r[1] for r in results if r[1].shape[0] > 0], axis=0)
    gcr = np.concatenate([r[2] for r in results if r[2].shape[0] > 0])
    model_path = results[0][3]
    obs_dim = feats.shape[1] // history_len
    print(f"Collected {feats.shape[0]} anchors | input_dim={feats.shape[1]} (obs_dim={obs_dim}) | "
          f"drops shape {dr.shape}")
    return feats, dr, gcr, {"model_path": model_path, "obs_dim": obs_dim,
                            "input_dim": feats.shape[1]}


# ---------------------------------------------------------------------------
# HGB quantile predictor
# ---------------------------------------------------------------------------
def _fit_quantile(args, X_train, drops_n_train, q):
    """HGB conditional-q quantile of ΔG(s,n) from flattened (obs repeated N, drop) pairs."""
    N = drops_n_train.shape[1]
    X_rep = np.repeat(X_train, N, axis=0)
    y = drops_n_train.reshape(-1)
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=args.hgb_max_iter, learning_rate=args.hgb_lr,
        l2_regularization=args.hgb_l2, validation_fraction=0.15, n_iter_no_change=30,
        random_state=args.seed).fit(X_rep, y)


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
    p = argparse.ArgumentParser(description="Grushin-style tolerable-perturbation-count safety "
                                            "margin with conformal certified bounds.")
    p.add_argument("--eval-dir", type=str, required=True)
    p.add_argument("--raw-npz", type=str, default=None)
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=2000)
    p.add_argument("--samples-per-state", type=int, default=64)
    p.add_argument("--random-prob", type=float, default=0.05)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--history-len", type=int, default=2)
    p.add_argument("--tolerance-list", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    p.add_argument("--alpha", type=float, default=0.1)
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
        n_list = list(d["n_list"])
        print(f"Loaded {d['features'].shape[0]} anchors | n_list={n_list} | drops {d['drops_r'].shape}"
              f" | source_policy {os.path.basename(extra['model_path'])}")
        return (d["features"].astype(np.float64), d["drops_r"].astype(np.float64),
                d["g_clean_r"].astype(np.float64), n_list, extra)

    feats, dr, gcr, extra = collect_grushin_parallel(
        args.eval_dir, args.num_anchors, args.samples_per_state, args.random_prob, args.n_list,
        args.gamma, args.max_steps, args.history_len, args.num_workers, args.seed)
    if args.raw_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_npz)), exist_ok=True)
        np.savez_compressed(args.raw_npz, features=feats, drops_r=dr,
                            g_clean_r=gcr, n_list=np.array(args.n_list),
                            model_path=extra["model_path"], obs_dim=extra["obs_dim"],
                            input_dim=extra["input_dim"])
        print(f"Saved raw data to {args.raw_npz}")
    return (feats.astype(np.float64), dr.astype(np.float64), gcr.astype(np.float64),
            list(args.n_list), extra)


def main():
    args = parse_args()
    alpha = args.alpha
    drop_label = "ΔG (reward drop)"

    X, drops, g_clean, n_list, extra = _load_or_collect(args)   # drops [A, n_vals, N]
    A, n_vals, N = drops.shape
    n_list = list(int(n) for n in n_list)
    print(f"\n{drop_label} | A={A} anchors | n_list={n_list} | N={N}")

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
    M_te = np.empty((n_vals, len(te)), dtype=np.float64)   # certified upper margin per (n, test state)
    per_n = []
    print("\n" + "=" * 66)
    print(f"PER-n CERTIFIED BOUND  ({drop_label}, target coverage {1-alpha:.2f})")
    print("=" * 66)
    print(f"  {'n':<5}{'coverage':<11}{'mean_margin':<13}{'Q(n)':<9}{'mean ΔG':<10}{'q_{1-a} ΔG':<11}")
    for ni, n in enumerate(n_list):
        dtr_n, dcal_n, dte_n = drops[tr, ni, :], drops[cal, ni, :], drops[te, ni, :]
        model = _fit_quantile(args, Xtr, dtr_n, 1.0 - alpha)
        calib_y = one_per_anchor(torch.as_tensor(dcal_n, dtype=torch.float32), seed=args.seed).numpy()
        base_cal, base_te = model.predict(Xcal), model.predict(Xte)
        Q = conformal_offset(calib_y - base_cal, alpha)
        M = base_te + Q
        M_te[ni] = M
        cov = float((dte_n <= M[:, None]).mean())
        models[n], q_one[n] = model, float(Q)
        emp_mean = float(dte_n.mean())
        emp_q = float(np.quantile(dte_n, 1.0 - alpha))
        per_n.append({"n": int(n), "coverage": cov, "mean_margin": float(M.mean()), "q_one": float(Q),
                      "emp_mean_drop": emp_mean, "emp_upper_quantile": emp_q})
        print(f"  {n:<5}{cov:<11.4f}{float(M.mean()):<13.3f}{Q:<9.3f}{emp_mean:<10.3f}{emp_q:<11.3f}")

    # ---- Grushin safety margin per tolerance ζ ---------------------------------
    dte_all = drops[te]                                     # [A_te, n_vals, N]
    tol_results = []
    print("\n" + "=" * 66)
    print("SAFETY MARGIN  s*(s,ζ) = max n with M_α(s,n')≤ζ ∀n'≤n")
    print("=" * 66)
    print(f"  {'ζ':<7}{'mean margin':<13}{'frac margin=0':<15}{'validity (exc@margin≤α?)':<26}")
    for zeta in args.tolerance_list:
        margins = _grushin_margin(M_te, n_list, zeta)      # [A_te]
        # validity: among states certified to tolerate n, empirical P(ΔG(s,n) > ζ) must be ≤ α
        worst_exc = 0.0
        for ni, n in enumerate(n_list):
            cert = margins >= n                            # certified to tolerate at least n
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
        print(f"  {zeta:<7.2f}{float(margins.mean()):<13.3f}{float((margins==0).mean()):<15.3f}"
              f"worst {worst_exc:.3f} vs α={alpha:.2f}  [{ok}]")

    # ---- plots -----------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        ns = np.asarray(n_list)
        # criticality vs n
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ns, [r["emp_mean_drop"] for r in per_n], "o-", label="mean ΔG")
        ax.plot(ns, [r["emp_upper_quantile"] for r in per_n], "s-", label=f"empirical q_{{{1-alpha:.2f}}}")
        ax.plot(ns, [r["mean_margin"] for r in per_n], "^--", label="certified M_α (mean)")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel(drop_label)
        ax.set_title("Criticality vs perturbation count"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "criticality_vs_n.png"), dpi=180)
        plt.close(fig)
        # coverage by n
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [r["coverage"] for r in per_n], "o-")
        ax.axhline(1 - alpha, c="k", ls="--", label=f"target {1-alpha:.2f}")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel("conformal coverage")
        ax.set_title("Per-n certified coverage"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "coverage_by_n.png"), dpi=180)
        plt.close(fig)
        # margin histograms + margin vs tolerance
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
        "method": "grushin_conformal", "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
                         "history_len": args.history_len},
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
        joblib.dump({"method": "grushin_conformal", "n_list": n_list, "alpha": alpha,
                     "input_dim": extra["input_dim"],
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
