"""
Reliability-backed adaptive-width safety margin on **Pendulum-v1** — DIRECT-PREDICTOR edition.

Port of ``safety_margin_adaptivity_direct.py`` (SafePO / safety-gym) to Pendulum-v1 with the pretrained
SB3 PPO policy ``sb3/ppo-Pendulum-v1`` (HF Hub). The method is identical; only the environment /
data-collection front-end changes.

WHY PENDULUM. The CartPole port turned out to be a degenerate testbed: the policy balanced to the step cap
from every anchor (``g_clean`` constant) and the return drop was strictly bimodal (recover≈0 or fall≈86,
~nothing between), which collapsed every exceedance threshold onto a single "fall probability" axis and
left the adaptive margin with almost no dynamic range. Pendulum fixes this at the root:

  * **no termination** — episodes are fixed-length, so a disturbance produces a graded accumulation of
    per-step cost rather than an all-or-nothing outcome;
  * **dense quadratic reward** ``r = −(θ² + 0.1·θ̇² + 0.001·a²)`` — partial degradation is expressible;
  * **basin structure** — cheap recovery near upright vs an expensive full swing-up if knocked past the
    boundary, giving ΔG a genuine heavy right tail so ``P(ΔG>τ)`` means different things at different τ;
  * ``g_clean`` genuinely **varies** across anchors (the policy's per-episode return spans ≈ −4 … −435).

Criticality target (reward-only; Pendulum has no cost signal — its reward is already cost-like)::

    ΔG(s) = G_clean(s) − G_perturbed(s)      (large drop = bad → one-sided UPPER margin on ΔG)

The per-state ΔG distribution is induced by an **action burst** over Pendulum's continuous torque
``Box(-2, 2)``, in one of two modes (``--burst-mode``):
  * ``uniform``  — the first ``random_steps`` actions are drawn uniformly from the action space
                   (direct analog of the SafePO / CartPole discrete random burst);
  * ``gaussian`` — the first ``random_steps`` actions are ``clip(π(s) + N(0, σ²))`` for a continuously
                   tunable disturbance magnitude (``--noise-std``).
Thereafter the deterministic policy is followed to the horizon.

Downstream is verbatim from the SafePO direct driver: split-half RELIABILITY of candidate per-state
targets (std / iqr / mad / exceedance at several τ) → direct sklearn ``HistGradientBoosting`` predictors
and a RECOVERY table → a **Mondrian (exceedance-grouped) conformal margin** centered on the predicted
upper quantile, M(s)=q̂_{1−α}(s)+Q_{g(s)}, with per-predicted-group coverage ≥ 1−α.

All helpers live in ``adaptivity_core.py`` (numpy/sklearn only — this package is torch-free).

Run (in the ``csc249`` env, from the ``safety_margin`` directory):
    python pendulum/pendulum_adaptivity_direct.py --num-anchors 5000 --samples-per-state 64 \
        --random-steps 8 --random-prob 0.05 --tau-quantiles 0.7 0.85 0.95 --mondrian-groups 5 \
        --raw-npz ./pendulum/runs/pendulum_adaptivity_direct/raw.npz \
        --save-path ./pendulum/runs/pendulum_adaptivity_direct/ckpt.joblib \
        --plot-dir ./pendulum/runs/pendulum_adaptivity_direct/plots \
        --results-json ./pendulum/runs/pendulum_adaptivity_direct/results.json
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
    discount_cumsum, run_parallel_collection, _scatter_identity, one_per_anchor,
    _stat_std, _stat_iqr, _stat_mad, _stat_exceedance, _pearson, split_half_reliability,
    mondrian_calibrate, mondrian_apply, _resolve_mondrian_tau, _interior_spread,
    _fit_flat, _fit_peranchor, direct_conformal_table,
)

ENV_ID = "Pendulum-v1"
HF_REPO = "sb3/ppo-Pendulum-v1"
HF_FILENAME = "ppo-Pendulum-v1.zip"


# ---------------------------------------------------------------------------
# Policy + environment (HF SB3 PPO; exact state save/restore)
# ---------------------------------------------------------------------------
def download_policy() -> str:
    """Fetch the SB3 PPO zip from the HF Hub (cached); robust to the exact zip filename."""
    from huggingface_sb3 import load_from_hub
    try:
        return load_from_hub(HF_REPO, HF_FILENAME)
    except Exception:
        from huggingface_hub import list_repo_files
        zips = [f for f in list_repo_files(HF_REPO) if f.endswith(".zip")]
        if not zips:
            raise
        return load_from_hub(HF_REPO, zips[0])


def _load_policy(model_path: str):
    import gymnasium as gym
    from stable_baselines3 import PPO
    # custom_objects avoids deserialization issues with schedules saved by older SB3, and supplies the
    # spaces directly so the hub model's pickled (legacy gym) spaces never need to be unpickled.
    env = gym.make(ENV_ID)
    custom = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0,
              "observation_space": env.observation_space, "action_space": env.action_space}
    env.close()
    return PPO.load(model_path, device="cpu", custom_objects=custom)


def _make_core_env():
    """Raw (unwrapped) env: exposes ``state`` for exact restore and drops the TimeLimit wrapper, so the
    horizon is controlled entirely by ``max_steps``."""
    import gymnasium as gym
    return gym.make(ENV_ID).unwrapped


def _save_state(core) -> np.ndarray:
    """Pendulum's full simulator state is the 2-vector [theta, theta_dot]."""
    return np.array(core.state, dtype=np.float64).copy()


def _restore_state(core, state: np.ndarray) -> None:
    core.state = np.array(state, dtype=np.float64).copy()
    core.last_u = None          # render-only bookkeeping; reset for cleanliness


def _obs_from_state(state) -> np.ndarray:
    """Pendulum observation is [cos(theta), sin(theta), theta_dot]."""
    th, thdot = float(state[0]), float(state[1])
    return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)


def _predict(model, obs) -> np.ndarray:
    """Deterministic policy action — CONTINUOUS (shape (1,)) for Pendulum, unlike CartPole's int."""
    action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _burst_action(core, model, obs, rng, burst_mode: str, noise_std: float) -> np.ndarray:
    """Disturbance action during the burst window."""
    if burst_mode == "uniform":
        return np.asarray(core.action_space.sample(), dtype=np.float32).reshape(-1)
    a = _predict(model, obs) + rng.normal(0.0, noise_std, size=core.action_space.shape)
    return np.clip(a, core.action_space.low, core.action_space.high).astype(np.float32)


def _rollout_return(core, model, start_obs, burst_steps, burst_mode, noise_std, rng,
                    gamma, max_steps) -> float:
    """Discounted return from the *current* (already-restored) core state.

    The first ``burst_steps`` actions are disturbance actions; thereafter the deterministic policy.
    Pendulum never terminates, so the loop is bounded purely by ``max_steps`` (the terminated/truncated
    check is kept defensively).
    """
    o, rewards, steps, done = start_obs, [], 0, False
    while not done and steps < max_steps:
        a = (_burst_action(core, model, o, rng, burst_mode, noise_std) if steps < burst_steps
             else _predict(model, o))
        o, r, terminated, truncated, _ = core.step(a)
        rewards.append(float(r)); steps += 1
        done = bool(terminated or truncated)
    return discount_cumsum(rewards, gamma)


# ---------------------------------------------------------------------------
# Collection worker (action burst -> per-state return drops)
# ---------------------------------------------------------------------------
def _collect_worker(model_path, num_anchors, samples_per_state, random_prob, random_steps,
                    burst_mode, noise_std, gamma, max_steps, history_len, seed, counter=None):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
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

        # Phase 1: roll the nominal policy, tracking a sliding window of the last history_len obs.
        # Bounded by max_steps -- Pendulum has no terminal state, so this would otherwise never exit.
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
                anchor_feature = np.concatenate(list(hist), axis=0)   # [history_len * obs_dim]
                break
            obs, _, terminated, truncated, _ = core.step(_predict(model, obs))
            hist.append(np.asarray(obs, dtype=np.float32).copy())
            steps += 1
            done = bool(terminated or truncated)
        if anchor_state is None:
            continue  # no trigger fired this episode; retry

        # Phase 2a: one clean (deterministic policy, no burst) rollout from the anchor -> g_clean.
        _restore_state(core, anchor_state)
        g_clean = _rollout_return(core, model, anchor_obs, 0, burst_mode, noise_std, rng,
                                  gamma, max_steps)

        # Phase 2b: N burst rollouts from the same anchor; drop_i = g_clean - g_i.
        drops = np.empty(samples_per_state, dtype=np.float32)
        for i in range(samples_per_state):
            _restore_state(core, anchor_state)
            core.action_space.seed(seed * 100003 + collected * 911 + i)
            g_i = _rollout_return(core, model, anchor_obs, random_steps, burst_mode, noise_std, rng,
                                  gamma, max_steps)
            drops[i] = g_clean - g_i
        feat_buf.append(anchor_feature)
        drop_buf.append(drops)
        gclean_buf.append(g_clean)
        collected += 1
        if counter is not None:
            counter.value += 1

    if not feat_buf:
        return (np.empty((0, input_dim), np.float32), np.empty((0, samples_per_state), np.float32),
                np.empty(0, np.float32))
    return (np.stack(feat_buf).astype(np.float32), np.stack(drop_buf).astype(np.float32),
            np.array(gclean_buf, dtype=np.float32))


def collect_parallel(model_path, num_anchors, samples_per_state, random_prob, random_steps,
                     burst_mode, noise_std, gamma, max_steps, history_len, num_workers, base_seed):
    num_workers = max(1, min(num_workers, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    counts = [c for c in counts if c > 0]
    print(f"Collecting {num_anchors} anchors x {samples_per_state} burst-samples across {len(counts)} "
          f"workers | burst_mode={burst_mode}"
          + (f" (sigma={noise_std})" if burst_mode == "gaussian" else "")
          + f" | random_steps={random_steps} | history_len={history_len}")
    args = [(model_path, c, samples_per_state, random_prob, random_steps, burst_mode, noise_std,
             gamma, max_steps, history_len, base_seed + i) for i, c in enumerate(counts)]
    results = run_parallel_collection(_collect_worker, args, num_anchors, "Anchors", len(counts))
    feat_parts = [r[0] for r in results if r[0].shape[0] > 0]
    drop_parts = [r[1] for r in results if r[1].shape[0] > 0]
    gcl_parts = [r[2] for r in results if r[2].shape[0] > 0]
    if not feat_parts:
        raise RuntimeError("No anchors collected; raise --random-prob or --num-anchors.")
    feats = np.concatenate(feat_parts, 0)
    drops = np.concatenate(drop_parts, 0)
    g_clean = np.concatenate(gcl_parts, 0)
    input_dim = feats.shape[1]
    obs_dim = input_dim // history_len
    print(f"Collected {feats.shape[0]} anchors | input_dim={input_dim} (obs_dim={obs_dim} x "
          f"history={history_len})")
    print(f"  g_clean: mean {g_clean.mean():.3f} std {g_clean.std():.3f} "
          f"[{g_clean.min():.3f}, {g_clean.max():.3f}]")
    print(f"  ΔG     : mean {drops.mean():.3f} std {drops.std():.3f} "
          f"[{drops.min():.3f}, {drops.max():.3f}]")
    extra = {"model_path": model_path, "obs_dim": obs_dim, "input_dim": input_dim}
    return feats, drops, g_clean, extra


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Reliability-backed adaptive-width safety margin on Pendulum-v1 with DIRECT (HGB) "
                     "predictors; Mondrian conformal grouped by predicted exceedance of the return drop.")
    )
    p.add_argument("--raw-npz", type=str, default=None,
                   help="Load raw (features, drops, g_clean) from this .npz if it exists; "
                        "otherwise collect and save it here for reuse.")
    p.add_argument("--num-anchors", "--episodes", dest="num_anchors", type=int, default=2000)
    p.add_argument("--samples-per-state", type=int, default=64)
    p.add_argument("--random-prob", type=float, default=0.05)
    p.add_argument("--random-steps", "--random-step", dest="random_steps", type=int, default=8,
                   help="Number of disturbance actions at the start of each perturbed rollout.")
    p.add_argument("--burst-mode", type=str, choices=["uniform", "gaussian"], default="uniform",
                   help="uniform = sample torque from the action space; "
                        "gaussian = clip(policy action + N(0, noise_std^2)).")
    p.add_argument("--noise-std", type=float, default=1.0,
                   help="Disturbance sigma for --burst-mode gaussian (torque units; action range is +-2).")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount for the MC returns. 0.99 over a 200-step horizon captures the full "
                        "swing-up recovery (the zoo trained the policy at 0.9, but gamma is only an "
                        "analysis choice here -- no value function is used).")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Rollout horizon (Pendulum's native episode length; it never terminates).")
    p.add_argument("--history-len", type=int, default=1,
                   help="Stacked-obs history feeding the predictors (Pendulum obs is Markov -> 1).")
    # dispersion / risk targets
    p.add_argument("--tau-list", type=float, nargs="+", default=[5.0, 25.0, 100.0],
                   help="Fixed thresholds τ for exceedance targets P(ΔG>τ).")
    p.add_argument("--tau-quantiles", type=float, nargs="+", default=None,
                   help="If given, OVERRIDES --tau-list: derive τ from these quantiles of the pooled ΔG "
                        "(e.g. 0.7 0.85 0.95). Robust to an unknown reward scale on a new environment.")
    p.add_argument("--reliability-floor", type=float, default=0.30,
                   help="A target counts as a valid adaptivity axis iff its SB reliability >= this.")
    p.add_argument("--mondrian-tau", type=str, default="auto",
                   help="τ for the exceedance grouping axis ('auto' = the cleared exceedance with the most "
                        "interior spread, NOT the highest-recovery one which may be saturated).")
    p.add_argument("--mondrian-group-by", type=str, choices=["exceedance", "mean", "center"],
                   default="center",
                   help="Quantity the Mondrian bins on. DEFAULT 'center' = the predicted q̂_{1-α} margin "
                        "base: continuous (always well-spread, saturation-proof) and it exposes exactly "
                        "where a single global offset under-covers (the high-risk group), so it gives the "
                        "cleanest adaptive>global gap. 'mean' = predicted mean drop. 'exceedance' = the "
                        "interior-spread-selected τ (an exceedance axis can still leave global already "
                        "group-uniform, so Mondrian may not beat it there).")
    p.add_argument("--spread-floor", type=float, default=0.10,
                   help="Min interior spread (mass in (0.05,0.95)) for an exceedance target to be eligible "
                        "as the headline adaptive-width metric — avoids rewarding a saturated 0/1 axis.")
    p.add_argument("--mondrian-groups", type=int, default=5)
    p.add_argument("--num-bins", type=int, default=4,
                   help="Empirical-exceedance bins for the conditional-coverage plot.")
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
    if args.raw_npz and os.path.exists(args.raw_npz):
        print(f"Loading raw data from {args.raw_npz}")
        d = np.load(args.raw_npz, allow_pickle=True)
        extra = {"model_path": str(d["model_path"]), "obs_dim": int(d["obs_dim"]),
                 "input_dim": int(d["input_dim"])}
        print(f"Loaded {d['features'].shape[0]} anchors | input_dim={extra['input_dim']} | "
              f"source_policy {os.path.basename(extra['model_path'])}")
        return (d["features"].astype(np.float64), d["drops_r"].astype(np.float64),
                d["g_clean"].astype(np.float64), extra)

    model_path = download_policy()
    print(f"Loaded baseline policy from {HF_REPO} -> {model_path}")
    feats, drops, g_clean, extra = collect_parallel(
        model_path, args.num_anchors, args.samples_per_state, args.random_prob, args.random_steps,
        args.burst_mode, args.noise_std, args.gamma, args.max_steps, args.history_len,
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

    # ---- resolve exceedance thresholds ----------------------------------------
    # On a new environment the ΔG scale is unknown, so τ can be derived from the pooled drop
    # distribution rather than guessed; this keeps the exceedance targets in an informative range.
    if args.tau_quantiles:
        tau_list = [float(np.quantile(drops, q)) for q in args.tau_quantiles]
        tau_list = sorted(set(round(t, 6) for t in tau_list))
        print(f"τ from ΔG quantiles {args.tau_quantiles} -> "
              f"{[f'{t:.3f}' for t in tau_list]}")
    else:
        tau_list = list(args.tau_list)
    args.tau_list = tau_list   # so it lands in the results JSON

    # ---- reliability of candidate targets (model-free, on ALL anchors) ---------
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

    # Interior spread of each exceedance axis (mass strictly inside (0,1)). A saturated 0/1 axis is
    # trivially recoverable but useless for grouping, so headline/grouping selection must not be keyed to
    # recovery alone. Non-exceedance targets are treated as always-eligible (default spread 1.0).
    interior_spread = {name: _interior_spread(fn(drops)) for name, fn in stat_fns.items()
                       if name.startswith("exceedance@")}
    for name in stat_fns:
        recovery[name]["interior_spread"] = interior_spread.get(name)
    if interior_spread:
        print("Interior spread (mass in (0.05,0.95)) of exceedance axes: "
              + " | ".join(f"{k.split('@')[1]}:{v:.3f}" for k, v in interior_spread.items()))

    cleared = [k for k in stat_fns if reliability[k]["cleared"]]
    # spread-gated headline: among cleared targets prefer those that also clear the spread floor (a
    # saturated exceedance axis wins recovery but is a degenerate grouping axis); fall back to all cleared.
    eligible = [k for k in cleared if interior_spread.get(k, 1.0) >= args.spread_floor] or cleared
    headline_target = max(eligible, key=lambda k: recovery[k]["recovery_r"]) if eligible else None
    print(f"\nHeadline adaptive-width metric: {headline_target or 'NONE cleared'}"
          + (f"  recovery_r={recovery[headline_target]['recovery_r']:+.3f}"
             f"  interior_spread={interior_spread.get(headline_target, float('nan')):.3f}"
             if headline_target else ""))

    # ---- Mondrian (risk-grouped) conformal margin ------------------------------
    calib_y = one_per_anchor(dcal, seed=args.seed)
    # Center the Mondrian on the PREDICTED UPPER QUANTILE q̂_{1−α}(s), not the median: the median strips
    # the (reliable) location signal and forces Q_g to encode an unreliable dispersion, which inverts the
    # width. With the upper-quantile center, each group's conformal offset repairs that group's residual
    # miscoverage, guaranteeing >= 1−α coverage per predicted-risk group.
    cen_cal = m_upper.predict(Xcal)
    cen_te = m_upper.predict(Xte)

    # Grouping axis. τ is chosen by INTERIOR SPREAD (not recovery); mean/center bin on a continuous
    # predicted-risk quantity (always well-spread, saturation-proof). m_tau is kept as the exceedance
    # threshold used for the empirical-exceedance conditional-coverage DIAGNOSTIC regardless of axis.
    m_tau = _resolve_mondrian_tau(args.mondrian_tau, reliability, tau_list, headline_target,
                                  spread=interior_spread)
    if args.mondrian_group_by == "mean":
        group_model = m_mean
        group_desc = "predicted mean drop"
    elif args.mondrian_group_by == "center":
        group_model = m_upper
        group_desc = f"q̂_{{{1-alpha:.2f}}} center"
    else:
        group_model = target_models[f"exceedance@{m_tau:g}"]
        group_desc = (f"exceedance τ={m_tau:g} "
                      f"(interior spread {interior_spread.get(f'exceedance@{m_tau:g}', float('nan')):.3f})")
    pexc_cal, pexc_te = group_model.predict(Xcal), group_model.predict(Xte)
    print(f"Mondrian grouping axis: {group_desc}")

    edges, Qg, q_global, counts = mondrian_calibrate(calib_y, cen_cal, pexc_cal, alpha, args.mondrian_groups)
    m_mond, g_te = mondrian_apply(cen_te, pexc_te, edges, Qg, q_global)
    m_glob = cen_te + q_global
    hw_mond = m_mond - cen_te
    cov_mond = float((dte <= m_mond[:, None]).mean())
    cov_glob = float((dte <= m_glob[:, None]).mean())
    print("\n" + "=" * 68)
    print(f"MONDRIAN ADAPTIVE MARGIN  (center = q̂_{{{1-alpha:.2f}}}(s), group-by {group_desc}, "
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
        # 95% CI half-width, using the ANCHOR as the exchangeable unit (n_g anchors), not the correlated
        # N samples — the honest, conservative granularity for a coverage rate.
        ci95 = float(1.96 * np.sqrt(max(cov_g * (1.0 - cov_g), 1e-9) / max(n_g, 1)))
        group_rows.append({"group": g, "n_calib": counts.get(g, 0), "n_test": n_g,
                           "Q_g": float(Qg.get(g, q_global)), "emp_exc_lo": exc_lo,
                           "emp_exc_hi": exc_hi, "coverage": cov_g,
                           "coverage_global": cov_g_glob, "ci95_halfwidth": ci95})
        print(f"  {g:<4}{counts.get(g,0):<7}{n_g:<7}{Qg.get(g,q_global):<9.3f}"
              f"[{exc_lo:.2f},{exc_hi:.2f}]     {cov_g:<10.4f}{cov_g_glob:<10.4f}±{ci95:.3f}")

    # Per-PREDICTED-group coverage is the quantity Mondrian guarantees (>= 1−α per group), unlike the
    # global offset (marginal only). The safety-relevant number is the WORST group's coverage.
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
        cm = float((dte[idx] <= m_mond[idx][:, None]).mean())
        cg = float((dte[idx] <= m_glob[idx][:, None]).mean())
        hm = float(hw_mond[idx].mean())
        bin_rows.append({"bin": bi, "emp_exc_lo": e_lo, "emp_exc_hi": e_hi, "cov_mond": cm,
                         "cov_glob": cg, "hw_mond": hm, "hw_glob": float(q_global), "n": int(len(idx))})
        print(f"  {bi:<5}[{e_lo:5.2f},{e_hi:5.2f}]      {cm:<11.4f}{cg:<11.4f}{hm:<10.3f}")

    cov_spread_mond = (max(r["cov_mond"] for r in bin_rows) - min(r["cov_mond"] for r in bin_rows)) if bin_rows else float("nan")
    cov_spread_glob = (max(r["cov_glob"] for r in bin_rows) - min(r["cov_glob"] for r in bin_rows)) if bin_rows else float("nan")
    width_exc_corr = _pearson(hw_mond, emp_exc_all)
    margin_exc_corr = _pearson(m_mond, emp_exc_all)
    print(f"\n  conditional-coverage spread (max-min over bins): "
          f"Mondrian {cov_spread_mond:.4f}  vs  global {cov_spread_glob:.4f}  (smaller=better)")
    print(f"  corr(Mondrian group offset, empirical exceedance) = {width_exc_corr:+.3f}")
    print(f"  corr(FULL Mondrian margin,  empirical exceedance) = {margin_exc_corr:+.3f} "
          f"(adaptivity lives in the q̂ center + group offset)")

    # ---- direct-quantile conformal comparison table ----------------------------
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

        # ΔG distribution — the diagnostic that exposed CartPole's degeneracy (bimodal vs continuous).
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.hist(drops.reshape(-1), bins=80, color="steelblue", edgecolor="none")
        a1.set_xlabel("ΔG (return drop)"); a1.set_ylabel("count")
        a1.set_title(f"Pooled ΔG distribution (mean {drops.mean():.2f}, std {drops.std():.2f})")
        for t in tau_list:
            a1.axvline(t, c="tab:red", ls="--", lw=1)
        a2.hist(g_clean, bins=60, color="seagreen", edgecolor="none")
        a2.set_xlabel("g_clean (clean return from anchor)"); a2.set_ylabel("count")
        a2.set_title(f"g_clean spread (std {g_clean.std():.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(args.plot_dir, "drop_distribution.png"), dpi=180)
        plt.close(fig)

        # HEADLINE: conditional coverage by PREDICTED-exceedance group — the quantity the Mondrian
        # guarantee applies to (>= 1−α within each predicted-risk group), vs the global offset (marginal).
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
            ax.set_xlabel(f"predicted-risk group by {group_desc} (low → high risk)")
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
        "env": ENV_ID, "policy": HF_REPO, "predictor": "direct_hgb",
        "perturbation": f"action_burst_{args.burst_mode}",
        "args": {k: getattr(args, k) for k in vars(args)},
        "feature_info": {"obs_dim": extra["obs_dim"], "history_len": args.history_len,
                         "input_dim": extra["input_dim"]},
        "drop_stats": {"g_clean_mean": float(g_clean.mean()), "g_clean_std": float(g_clean.std()),
                       "g_clean_min": float(g_clean.min()), "g_clean_max": float(g_clean.max()),
                       "drop_mean": float(drops.mean()), "drop_std": float(drops.std()),
                       "drop_min": float(drops.min()), "drop_max": float(drops.max())},
        "split": {"train": int(len(tr_idx)), "calibration": int(len(cal_idx)),
                  "test": int(len(te_idx)), "samples_per_state": int(N)},
        "reliability": reliability,
        "recovery": recovery,
        "headline_target": headline_target,
        "headline_recovery_r": (recovery[headline_target]["recovery_r"] if headline_target else None),
        "headline_interior_spread": (interior_spread.get(headline_target) if headline_target else None),
        "interior_spread": interior_spread,
        "mondrian_margin": {
            "center": "upper_quantile",
            "group_by": args.mondrian_group_by, "group_desc": group_desc,
            "grouping_interior_spread": (interior_spread.get(f"exceedance@{m_tau:g}")
                                         if args.mondrian_group_by == "exceedance" else None),
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
            "env": ENV_ID, "policy": HF_REPO, "predictor": "direct_hgb",
            "perturbation": f"action_burst_{args.burst_mode}",
            "obs_dim": extra["obs_dim"], "input_dim": extra["input_dim"],
            "history_len": args.history_len, "alpha": alpha,
            "center_kind": "upper_quantile", "m_center": m_upper,
            "m_median": m_median, "group_by": args.mondrian_group_by, "group_model": group_model,
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

    ``ckpt`` is the dict saved by ``--save-path`` (joblib). Guarantee: within each predicted-risk
    group (grouped on ``ckpt['group_by']``), P(ΔG ≤ M(s)) ≥ 1−α.
    """
    obs = np.asarray(obs, dtype=np.float64)
    center_model = ckpt.get("m_center", ckpt.get("m_median"))
    center = center_model.predict(obs)
    group_model = ckpt.get("group_model", ckpt.get("exc_model"))   # exc_model: pre-group-by checkpoints
    pexc = group_model.predict(obs)
    edges = np.asarray(ckpt["mondrian_edges"], dtype=float)
    Qlist = ckpt["mondrian_Q"]
    g = np.digitize(pexc, edges)
    q = np.array([Qlist[int(gi)] if 0 <= int(gi) < len(Qlist) else ckpt["mondrian_q_global"]
                  for gi in g], dtype=np.float64)
    return center + q


if __name__ == "__main__":
    main()
