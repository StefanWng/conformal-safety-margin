"""
Self-contained core for the reliability-backed adaptive-width safety margin (direct-predictor edition).

Everything the Pendulum driver needs, with **no cross-folder imports and no torch**: numpy / sklearn /
joblib / tqdm / matplotlib only. (The direct-HGB pipeline uses only a small slice of the original
``gmm_core.py`` — none of the MDN / Hurdle / NLL-training / density-diagnostic machinery — so that slice is
inlined here rather than depended upon.)

Contents:
  * rollout + conformal utilities : ``discount_cumsum``, ``conformal_offset``, ``one_per_anchor``,
                                    ``run_parallel_collection``, ``_scatter_identity``
  * per-state risk statistics     : ``_stat_std/_stat_iqr/_stat_mad/_stat_exceedance``, ``_pearson``
  * split-half RELIABILITY        : ``split_half_reliability`` (Spearman-Brown)
  * Mondrian conformal            : ``mondrian_calibrate``, ``mondrian_apply``, ``_resolve_mondrian_tau``
  * direct HGB predictors         : ``_hgb_regressor``, ``_fit_flat``, ``_fit_peranchor``
  * conformal comparison table    : ``direct_conformal_table``

Array conventions: samples ``[A, N]`` (A anchors x N draws per anchor), per-anchor statistics ``[A]``,
features ``[A, input_dim]``.
"""

import math
import multiprocessing
import threading
import time
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Rollout / conformal utilities
# ---------------------------------------------------------------------------
def discount_cumsum(rewards, gamma: float) -> float:
    """Discounted return of a reward list (computed backwards)."""
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
    return g


def conformal_offset(scores, alpha: float) -> float:
    """Split-conformal offset: the ceil((n+1)(1-alpha))-th smallest score."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("No calibration scores.")
    q_index = min(max(math.ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(np.partition(scores, q_index)[q_index])


def one_per_anchor(samples: np.ndarray, seed: int = 0) -> np.ndarray:
    """Draw ONE sample per anchor — the exchangeable unit for conformal calibration. [A,N] -> [A]."""
    samples = np.asarray(samples, dtype=np.float64)
    A, N = samples.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=A)
    return samples[np.arange(A), idx]


def _progress_monitor(counter, total, pbar, stop):
    last = 0
    while not stop.is_set():
        cur = counter.value
        if cur != last:
            pbar.update(cur - last); last = cur
        if last >= total:
            break
        time.sleep(0.05)
    cur = counter.value
    if cur > last:
        pbar.update(cur - last)


def run_parallel_collection(worker_fn, worker_args_list, total, desc, num_workers):
    """Run ``worker_fn(*args, counter)`` across processes with a shared tqdm progress counter."""
    stop = threading.Event()
    with multiprocessing.Manager() as manager:
        counter = manager.Value("i", 0)
        with tqdm(total=total, desc=desc, unit="anchor") as pbar:
            monitor = threading.Thread(target=_progress_monitor,
                                       args=(counter, total, pbar, stop), daemon=True)
            monitor.start()
            results = joblib.Parallel(n_jobs=len(worker_args_list), backend="loky", verbose=0)(
                joblib.delayed(worker_fn)(*args, counter) for args in worker_args_list)
            stop.set(); monitor.join(timeout=2.0)
    return results


def _scatter_identity(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.scatter(x, y, s=10, alpha=0.5); ax.plot([lo, hi], [lo, hi], "r--", label="ideal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------
# Per-state dispersion / risk statistics  (operate on [A, N] -> [A])
# ---------------------------------------------------------------------------
def _stat_std(x: np.ndarray) -> np.ndarray:
    return x.std(axis=1)


def _stat_iqr(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)


def _stat_mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    return np.median(np.abs(x - med), axis=1)


def _stat_exceedance(tau: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return (x > tau).mean(axis=1)
    return f


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    am, bm = a - a.mean(), b - b.mean()
    return float((am * bm).sum() / (np.sqrt((am * am).sum()) * np.sqrt((bm * bm).sum()) + 1e-12))


def split_half_reliability(samples: np.ndarray, stat_fn: Callable[[np.ndarray], np.ndarray],
                           seed: int = 0) -> Dict[str, float]:
    """Split the N columns into two halves, compute stat on each, correlate across anchors.

    Returns r_half (raw split-half correlation), sb_reliability (Spearman-Brown extrapolation to the
    full N samples), and ceiling (~sqrt(reliability): the max correlation any predictor of the
    full-N statistic can reach, given the target's own sampling noise).
    """
    A, N = samples.shape
    rng = np.random.default_rng(seed)
    cols = rng.permutation(N)
    half = N // 2
    a = stat_fn(samples[:, cols[:half]])
    b = stat_fn(samples[:, cols[half:2 * half]])
    r = _pearson(a, b)
    sb = (2.0 * r / (1.0 + r)) if (np.isfinite(r) and r > -0.999) else float("nan")
    ceiling = float(np.sqrt(sb)) if (np.isfinite(sb) and sb > 0) else 0.0
    return {"r_half": r, "sb_reliability": sb, "ceiling": ceiling}


# ---------------------------------------------------------------------------
# Mondrian (exceedance-grouped) one-sided conformal
# ---------------------------------------------------------------------------
def mondrian_calibrate(calib_y: np.ndarray, center: np.ndarray, pexc_cal: np.ndarray,
                       alpha: float, groups: int) -> Tuple[np.ndarray, Dict[int, float], float, Dict[int, int]]:
    """Group-conditional (Mondrian) one-sided conformal keyed to predicted exceedance.

    Bin calibration states into `groups` quantile bins of predicted exceedance p̂(ΔG>τ|s); within each
    bin compute the one-sided offset of scores (y - center). Returns bin `edges`, per-group `Q` (dict
    group->offset), a global fallback `q_global`, and per-group calibration `counts`. The test margin is
    M(s)=center(s)+Q[g(s)] where g(s) uses the same edges -> width widens for high-risk groups, with
    coverage >= 1-alpha within each group.
    """
    scores = np.asarray(calib_y, float) - np.asarray(center, float)
    pexc = np.asarray(pexc_cal, float)
    qs = np.linspace(0.0, 1.0, groups + 1)[1:-1]
    edges = np.unique(np.quantile(pexc, qs)) if qs.size else np.array([])
    g_cal = np.digitize(pexc, edges)
    q_global = conformal_offset(scores, alpha)
    Q: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for g in range(len(edges) + 1):
        m = g_cal == g
        counts[g] = int(m.sum())
        Q[g] = conformal_offset(scores[m], alpha) if m.sum() >= 1 else q_global
    return edges, Q, q_global, counts


def mondrian_apply(center: np.ndarray, pexc: np.ndarray, edges: np.ndarray,
                   Q: Dict[int, float], q_global: float) -> Tuple[np.ndarray, np.ndarray]:
    """Assign states to exceedance groups via `edges`; return (Mondrian margin array, group ids)."""
    g = np.digitize(np.asarray(pexc, float), edges)
    q_arr = np.array([Q.get(int(gi), q_global) for gi in g], dtype=np.float64)
    return np.asarray(center, float) + q_arr, g


def _interior_spread(exc: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> float:
    """Fraction of per-anchor exceedance values strictly inside (lo, hi).

    Measures how well an exceedance axis SEPARATES anchors into distinct risk groups. A near-0 value
    means the axis is saturated at 0/1: it is trivially recoverable (predicting ~1 everywhere scores
    high) yet useless as a Mondrian grouping axis, because quantile bins of a saturated axis all land at
    the same value. Reliability/recovery-based axis selection is blind to this and can pick a degenerate
    grouping axis; selecting by interior spread avoids it.
    """
    exc = np.asarray(exc, float)
    return float(np.mean((exc > lo) & (exc < hi)))


def _resolve_mondrian_tau(arg: str, reliability: Dict[str, dict], tau_list: List[float],
                          headline: str, spread: Dict[str, float] = None) -> float:
    if arg != "auto":
        return float(arg)
    cleared = [t for t in tau_list if reliability[f"exceedance@{t:g}"]["cleared"]]
    # Prefer the cleared exceedance axis with the most INTERIOR SPREAD — a saturated 0/1 axis has high
    # recovery but cannot separate risk groups, so a recovery/reliability-keyed choice picks a useless
    # grouping axis (the saturation-rewarding flaw). Fall back to reliability, then the middle τ.
    if spread is not None and cleared:
        return float(max(cleared, key=lambda t: spread.get(f"exceedance@{t:g}", 0.0)))
    if headline and headline.startswith("exceedance@"):
        return float(headline.split("@")[1])
    exc = [(t, reliability[f"exceedance@{t:g}"]["sb_reliability"]) for t in cleared]
    if exc:
        return float(max(exc, key=lambda x: x[1])[0])
    return float(tau_list[len(tau_list) // 2])


# ---------------------------------------------------------------------------
# Direct HGB predictor factories
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
# Direct-predictor conformal comparison
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
