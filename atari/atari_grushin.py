"""
Grushin-style safety margin (tolerable perturbation count) on **BeamRiderNoFrameskip-v4** (QRDQN).

Port of ``safety_margin_grushin.py`` / ``pendulum_grushin.py`` to Atari — Grushin et al.'s original
domain. A safety margin is the maximum number of consecutive random perturbations n an agent tolerates
before its criticality (return drop) exceeds a tolerance ζ, with high confidence. Grushin estimate that
confidence bound with a 2-D KDE percentile (no formal guarantee); here the density estimator is replaced
by a direct HGB conditional-quantile predictor wrapped in a distribution-free split-conformal bound,
swept over n ∈ {1,2,4,8,16,32}:

  * collect ΔG(s,n) = G_clean - G_perturbed(n) for every n (exact ALE+wrapper state-restore MC),
  * per n, certify M_α(s,n) = q_hat_{1-α}(ΔG(s,n)) + Q(n)  with  P(ΔG(s,n) ≤ M_α(s,n)) ≥ 1-α,
  * safety margin s*(s,ζ) = max{ n : M_α(s,n') ≤ ζ for all n' ≤ n }, else 0.

Reward-only (Atari has no cost). Features are QRDQN NatureCNN embeddings; state save/restore + determinism
gate live in ``atari_core.py`` / ``test_state_restore.py``.

Run (conda env `atari`, from the `safety_margin` directory) — SMOKE:
    python atari/atari_grushin.py \
        --num-anchors 150 --samples-per-state 24 --n-list 1 2 4 8 16 32 --num-workers 12 \
        --max-steps 150 --tolerance-quantiles 0.5 0.8 0.95 \
        --raw-npz atari/runs/beamrider_grushin_smoke/raw.npz \
        --plot-dir atari/runs/beamrider_grushin_smoke/plots \
        --results-json atari/runs/beamrider_grushin_smoke/results.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List

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

from adaptivity_core import conformal_offset, one_per_anchor, _fit_flat  # noqa: E402
from atari_core import ENV_ID, REPO, collect_grushin_parallel  # noqa: E402


# ---------------------------------------------------------------------------
# HGB quantile predictor + margin rule
# ---------------------------------------------------------------------------
def _fit_quantile(args, X_train, drops_n_train, q):
    """HGB conditional-q quantile of ΔG(s,n) from flattened (obs repeated N, drop) pairs. This is the
    component that REPLACES the density estimator (GMM/KDE) of the original Grushin method."""
    return _fit_flat(args, X_train, drops_n_train, loss="quantile", quantile=q)


def _grushin_margin(M_by_n: np.ndarray, n_list: List[int], zeta: float) -> np.ndarray:
    """Per-state safety margin: largest n such that M_α(s,n') ≤ ζ for ALL n' ≤ n (else 0).

    M_by_n is [n_vals, A]; returns int margins [A] in {0} ∪ n_list.
    """
    safe = M_by_n <= zeta                                   # [n_vals, A]
    cum = np.logical_and.accumulate(safe, axis=0)           # leading-safe mask (∀ n'≤n clause)
    num_leading = cum.sum(axis=0)
    n_arr = np.asarray(n_list)
    return np.where(num_leading >= 1, n_arr[np.clip(num_leading - 1, 0, len(n_list) - 1)], 0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Grushin-style tolerable-perturbation-count safety margin on BeamRider, with the "
                    "density estimator replaced by a conformal-certified direct HGB quantile predictor.")
    p.add_argument("--raw-npz", type=str, default=None)
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=1200)
    p.add_argument("--samples-per-state", type=int, default=32)
    p.add_argument("--random-prob", type=float, default=0.02)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                   help="Perturbation counts to sweep (the margin is denominated in these units).")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-steps", type=int, default=200, help="Rollout horizon after the anchor.")
    p.add_argument("--anchor-max-steps", type=int, default=800,
                   help="Cap on the Phase-1 nominal rollout while searching for an anchor trigger.")
    p.add_argument("--history-len", type=int, default=1)
    p.add_argument("--episodic-life", action="store_true",
                   help="Terminal-on-life-loss. Default off: ΔG is the full max_steps-capped episode drop.")
    p.add_argument("--proxy-features", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tolerance-list", type=float, nargs="+", default=None,
                   help="Tolerances ζ on ΔG (BeamRider point scale). If omitted, use --tolerance-quantiles.")
    p.add_argument("--tolerance-quantiles", type=float, nargs="+", default=[0.5, 0.7, 0.85, 0.95],
                   help="Derive ζ from these quantiles of the pooled ΔG (robust to BeamRider's scale). "
                        "Ignored if --tolerance-list is given.")
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
        n_list = [int(n) for n in d["n_list"]]
        print(f"Loaded {d['features'].shape[0]} anchors | n_list={n_list} | drops {d['drops_r'].shape}"
              f" | source_policy {extra['model_path']}")
        return (d["features"].astype(np.float64), d["drops_r"].astype(np.float64),
                d["g_clean"].astype(np.float64), n_list, extra)

    feats, drops, g_clean, extra = collect_grushin_parallel(
        args.num_anchors, args.samples_per_state, args.random_prob, args.n_list, args.gamma,
        args.max_steps, args.episodic_life, args.device, args.proxy_features, args.anchor_max_steps,
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

    # ---- resolve tolerances from the pooled ΔG (unknown BeamRider scale) -----------------------
    if args.tolerance_list:
        tol_list = [float(z) for z in args.tolerance_list]
    else:
        tol_list = sorted(set(round(float(np.quantile(drops, q)), 4) for q in args.tolerance_quantiles))
        print(f"ζ from ΔG quantiles {args.tolerance_quantiles} -> {[f'{z:.2f}' for z in tol_list]}")
    args.tolerance_list = tol_list

    # ---- 3-way anchor split (shared across all n) ---------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(A)
    n_te = max(1, int(A * args.test_fraction))
    n_cal = max(1, int(A * args.calibration_fraction))
    te, cal, tr = perm[:n_te], perm[n_te:n_te + n_cal], perm[n_te + n_cal:]
    Xtr, Xcal, Xte = X[tr], X[cal], X[te]
    print(f"Split: train {len(tr)} | calib {len(cal)} | test {len(te)}")

    # ---- per-n certified bound M_α(s,n) -------------------------------------------------------
    models: Dict[int, object] = {}
    q_one: Dict[int, float] = {}
    M_te = np.empty((n_vals, len(te)), dtype=np.float64)
    per_n = []
    print("\n" + "=" * 66)
    print(f"PER-n CERTIFIED BOUND  ({drop_label}, target coverage {1-alpha:.2f})")
    print("=" * 66)
    print(f"  {'n':<5}{'coverage':<11}{'mean_margin':<13}{'Q(n)':<10}{'mean ΔG':<11}{'q_{1-a} ΔG':<12}")
    for ni, n in enumerate(n_list):
        dtr_n, dcal_n, dte_n = drops[tr, ni, :], drops[cal, ni, :], drops[te, ni, :]
        model = _fit_quantile(args, Xtr, dtr_n, 1.0 - alpha)
        calib_y = one_per_anchor(dcal_n, seed=args.seed)
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
        print(f"  {n:<5}{cov:<11.4f}{float(M.mean()):<13.2f}{Q:<10.2f}{emp_mean:<11.2f}{emp_q:<12.2f}")

    # ---- Grushin safety margin per tolerance ζ ------------------------------------------------
    dte_all = drops[te]                                     # [A_te, n_vals, N]
    tol_results = []
    print("\n" + "=" * 66)
    print("SAFETY MARGIN  s*(s,ζ) = max n with M_α(s,n')≤ζ ∀n'≤n")
    print("=" * 66)
    print(f"  {'ζ':<10}{'mean margin':<13}{'frac margin=0':<15}{'validity (exc@margin≤α?)':<26}")
    for zeta in tol_list:
        margins = _grushin_margin(M_te, n_list, zeta)
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
        print(f"  {zeta:<10.2f}{float(margins.mean()):<13.3f}{float((margins==0).mean()):<15.3f}"
              f"worst {worst_exc:.3f} vs α={alpha:.2f}  [{ok}]")

    # ---- plots --------------------------------------------------------------------------------
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        ns = np.asarray(n_list)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ns, [r["emp_mean_drop"] for r in per_n], "o-", label="mean ΔG")
        ax.plot(ns, [r["emp_upper_quantile"] for r in per_n], "s-", label=f"empirical q_{{{1-alpha:.2f}}}")
        ax.plot(ns, [r["mean_margin"] for r in per_n], "^--", label="certified M_α (mean)")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("perturbation count n"); ax.set_ylabel(drop_label)
        ax.set_title("Criticality vs perturbation count (BeamRider)"); ax.legend()
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

    # ---- results json -------------------------------------------------------------------------
    results = {
        "method": "grushin_conformal", "env": ENV_ID, "policy": REPO,
        "predictor": "direct_hgb_quantile", "feature": "qrdqn_naturecnn_embedding",
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

    # ---- checkpoint ---------------------------------------------------------------------------
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        joblib.dump({"method": "grushin_conformal", "env": ENV_ID, "policy": REPO,
                     "n_list": n_list, "alpha": alpha, "target": "reward",
                     "input_dim": extra["input_dim"], "history_len": args.history_len,
                     "models": models, "q_one": q_one, "tolerances": list(tol_list),
                     "source_policy": extra["model_path"], "results": results}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


def safety_margin_for_obs(ckpt: dict, obs: np.ndarray, zeta: float) -> np.ndarray:
    """Per-state Grushin safety margin (tolerable perturbation count) for embedded features [B, input_dim].

    Note: ``obs`` is the QRDQN NatureCNN EMBEDDING (use ``atari_core.embed``), not the raw image.
    Returns int margins in {0} ∪ n_list.
    """
    obs = np.asarray(obs, dtype=np.float64)
    n_list = list(ckpt["n_list"])
    M = np.stack([ckpt["models"][n].predict(obs) + ckpt["q_one"][n] for n in n_list], axis=0)
    return _grushin_margin(M, n_list, zeta)


if __name__ == "__main__":
    main()
