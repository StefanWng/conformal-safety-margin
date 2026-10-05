"""
Environment-agnostic reliability + Mondrian + direct-HGB helpers for the adaptive-width safety margin.

Numpy/sklearn-only distillation of the pieces that ``safety_margin_adaptivity_direct.py`` imports from
the SafePO modules (``gmm_safety_margin_adaptivity`` etc). Those modules pull in ``safepo``/mujoco and
cannot load in the classic-control ``csc249`` env, so the reusable, dependency-free logic is ported here:

  * per-state dispersion / risk statistics of a drop distribution ΔG  (std / iqr / mad / exceedance),
  * split-half (Spearman-Brown) RELIABILITY of those statistics,
  * Mondrian (group-conditional) one-sided conformal keyed to a predicted exceedance axis,
  * direct HistGradientBoosting predictor factories,
  * the constant-δ / CQR / one-sided conformal comparison table.

The only cross-module dependency is ``conformal_offset`` / ``one_per_anchor`` from ``gmm_core`` (also
numpy/torch-only). Everything here operates on numpy arrays: samples ``[A, N]`` (A anchors, N draws per
anchor), per-anchor statistics ``[A]``, and features ``[A, input_dim]``.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from gmm_core import conformal_offset, one_per_anchor  # noqa: F401  (re-exported for the driver)


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
# Mondrian (exceedance-grouped) one-sided conformal  (numpy-native)
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


def _resolve_mondrian_tau(arg: str, reliability: Dict[str, dict], tau_list: List[float],
                          headline: str) -> float:
    if arg != "auto":
        return float(arg)
    if headline and headline.startswith("exceedance@"):
        return float(headline.split("@")[1])
    exc = [(t, reliability[f"exceedance@{t:g}"]["sb_reliability"]) for t in tau_list
           if reliability[f"exceedance@{t:g}"]["cleared"]]
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
