"""
Environment-agnostic core for the GMM (MDN) + split-conformal safety-margin method.

Portable distillation of ``gmm_safety_margin_state_restore.py`` + ``gmm_safety_margin_mc_conformal.py``
with **no SafePO / MuJoCo / Stable-Baselines3 dependency** -- only ``torch``, ``numpy``,
``matplotlib``. The per-environment driver file (``gmm_conformal_cartpole.py``) supplies policy
loading, the environment, discrete-action perturbation, and MuJoCo-free state save/restore.

Density models (all expose the same transform-aware interface ``cdf / quantile / mean / std /
sample / pit`` on the RAW return scale, so the conformal layer + metrics + plots are model-agnostic):

  * ``MixtureDensityNetwork`` -- K-component 1-D GMM. ``transform='none'`` models the raw return;
    ``transform='logit'`` models ``logit(G/cap)`` so the density respects a bounded support
    ``[0, cap]`` (no mass leaking past the discounted-return ceiling).
  * ``HurdleMDN`` -- spike-plus-continuous: a Bernoulli "survives-to-cap" head (point mass at ``cap``)
    plus a continuous ``MixtureDensityNetwork`` (logit-transformed) for the sub-cap returns. This is
    the right shape for bounded, bimodal returns like CartPole (survive-to-cap vs fall-early).

The "safety margin" for these single-objective control tasks is a **one-sided lower bound on the
return**: ``P(G >= M(s)) >= 1 - alpha`` (CartPole: "balanced at least M discounted steps under
perturbation"). ``side='lower'`` is the default.
"""

import math
import multiprocessing
import os
import threading
import time
from typing import Callable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def build_mlp_network(sizes: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine.weight, a=math.sqrt(5))
        layers += [affine, act()]
    return nn.Sequential(*layers)


def discount_cumsum(rewards, gamma: float) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
    return g


def parse_hidden_sizes(spec) -> List[int]:
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    return [int(x) for x in str(spec).replace(",", " ").split()]


def return_cap(gamma: float, max_steps: int) -> float:
    """Discounted-return ceiling: sum_{k=0}^{T-1} gamma^k for a +1/step bounded task."""
    if gamma >= 1.0:
        return float(max_steps)
    return float((1.0 - gamma ** max_steps) / (1.0 - gamma))


_LOG_2PI = math.log(2.0 * math.pi)
_LOGIT_EPS = 1e-4


# ---------------------------------------------------------------------------
# Mixture Density Network with optional bounded (logit) transform
# ---------------------------------------------------------------------------
class MixtureDensityNetwork(nn.Module):
    """MLP -> K-component 1-D GMM over a *modeled* variable z; ``z = transform(G)``.

    transform='none'  : z = G            (original behaviour)
    transform='logit' : z = logit(G/cap) : the GMM lives in unbounded z-space but maps back into the
                        bounded support [0, cap], so predicted mass cannot leak past the ceiling.
    All public methods (cdf/quantile/mean/std/sample/pit) operate on the RAW return G.
    """

    def __init__(self, obs_dim, num_components=3, hidden_sizes=(64, 64), sigma_floor=0.1,
                 return_mean=0.0, return_std=1.0, transform="none", cap=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_components = num_components
        self.sigma_floor = sigma_floor
        self.transform = transform
        self.register_buffer("return_mean", torch.tensor(float(return_mean)))
        self.register_buffer("return_std", torch.tensor(float(return_std)))
        self.register_buffer("cap", torch.tensor(float(cap)))
        self.net = build_mlp_network(list((obs_dim, *hidden_sizes, 3 * num_components)))

    # --- transform between raw return G and modeled variable z -------------
    def _to_z(self, g):
        if self.transform == "logit":
            u = torch.clamp(g / self.cap, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
            return torch.log(u / (1.0 - u))
        return g

    def _to_g(self, z):
        if self.transform == "logit":
            return self.cap * torch.sigmoid(z)
        return z

    # --- raw GMM head (normalized) and its z-space denormalization ---------
    def forward(self, obs):
        out = self.net(obs)
        logits, mu, raw_sigma = torch.chunk(out, 3, dim=-1)
        return F.log_softmax(logits, dim=-1), mu, F.softplus(raw_sigma) + self.sigma_floor

    def _params_z(self, obs):
        """GMM params in z-space (de-normalized)."""
        log_pi, mu, sigma = self.forward(obs)
        return log_pi, mu * self.return_std + self.return_mean, sigma * self.return_std

    def transform_normalize(self, g):
        """Value the inner net directly models: normalized z."""
        return (self._to_z(g) - self.return_mean) / self.return_std

    def log_prob(self, obs, y_norm):
        log_pi, mu, sigma = self.forward(obs)
        y = y_norm.unsqueeze(-1)
        log_comp = log_pi - torch.log(sigma) - 0.5 * _LOG_2PI - 0.5 * ((y - mu) / sigma) ** 2
        return torch.logsumexp(log_comp, dim=-1)

    # --- CDF / quantile / sampling on raw scale ---------------------------
    def _cdf_z(self, obs, z):
        log_pi, mu, sigma = self._params_z(obs)
        zz = z.unsqueeze(-1)
        comp_cdf = 0.5 * (1.0 + torch.erf((zz - mu) / (sigma * math.sqrt(2.0))))
        return torch.sum(torch.exp(log_pi) * comp_cdf, dim=-1)

    def cdf(self, obs, g):
        return self._cdf_z(obs, self._to_z(g))

    def pit(self, obs, g):
        return self.cdf(obs, g).clamp(0.0, 1.0)

    def quantile(self, obs, q, iters=40):
        """Inverse CDF on the raw scale. ``q`` may be a float or a per-row tensor."""
        with torch.no_grad():
            log_pi, mu, sigma = self._params_z(obs)
            mean = torch.sum(torch.exp(log_pi) * mu, dim=-1)
            spread = mu.max(dim=-1).values - mu.min(dim=-1).values + 4.0 * sigma.max(dim=-1).values
            lo, hi = mean - spread, mean + spread
            q_t = q if torch.is_tensor(q) else torch.full_like(mean, float(q))
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                too_high = self._cdf_z(obs, mid) > q_t
                hi = torch.where(too_high, mid, hi)
                lo = torch.where(too_high, lo, mid)
            return self._to_g(0.5 * (lo + hi))

    def sample(self, obs, num_samples=1):
        log_pi, mu, sigma = self._params_z(obs)
        pi = torch.exp(log_pi)
        comp = torch.multinomial(pi, num_samples, replacement=True)
        z = torch.gather(mu, 1, comp) + torch.gather(sigma, 1, comp) * torch.randn_like(torch.gather(mu, 1, comp))
        return self._to_g(z)

    def mean(self, obs):
        if self.transform == "none":
            log_pi, mu, _ = self._params_z(obs)
            return torch.sum(torch.exp(log_pi) * mu, dim=-1)
        return self.sample(obs, 1024).mean(dim=-1)

    def std(self, obs):
        if self.transform == "none":
            log_pi, mu, sigma = self._params_z(obs)
            w = torch.exp(log_pi)
            m = torch.sum(w * mu, dim=-1, keepdim=True)
            var = torch.sum(w * (sigma ** 2 + (mu - m) ** 2), dim=-1)
            return torch.sqrt(torch.clamp(var, min=1e-12))
        return self.sample(obs, 1024).std(dim=-1)


# ---------------------------------------------------------------------------
# Hurdle model: point mass at cap (survive) + continuous sub-cap density
# ---------------------------------------------------------------------------
class HurdleMDN(nn.Module):
    """P(G=cap)=sigmoid(head(s)) point mass + continuous logit-MDN for G<cap.

    CDF:   F(g) = (1-p)*F_cont(g)         for g < cap-tol;   = 1 for g >= cap-tol
    Quant: Q(q) = cap                     if q >= 1-p;       = F_cont^{-1}(q/(1-p)) otherwise
    """

    def __init__(self, obs_dim, num_components, hidden_sizes, sigma_floor, cap, tol,
                 return_mean=0.0, return_std=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.register_buffer("cap", torch.tensor(float(cap)))
        self.register_buffer("tol", torch.tensor(float(tol)))
        self.survive_head = build_mlp_network(list((obs_dim, *hidden_sizes, 1)))
        self.cont = MixtureDensityNetwork(obs_dim, num_components, hidden_sizes, sigma_floor,
                                          return_mean=return_mean, return_std=return_std,
                                          transform="logit", cap=cap)

    def survive_logit(self, obs):
        return self.survive_head(obs).squeeze(-1)

    def p_survive(self, obs):
        return torch.sigmoid(self.survive_logit(obs))

    def cdf(self, obs, g):
        p = self.p_survive(obs)
        below = (1.0 - p) * self.cont.cdf(obs, g)
        return torch.where(g >= self.cap - self.tol, torch.ones_like(below), below)

    def pit(self, obs, g):
        p = self.p_survive(obs)
        below = (1.0 - p) * self.cont.cdf(obs, g)
        surv = g >= self.cap - self.tol
        u = torch.rand_like(p)
        return torch.where(surv, (1.0 - p) + u * p, below).clamp(0.0, 1.0)

    def quantile(self, obs, q, iters=40):
        with torch.no_grad():
            p = self.p_survive(obs)
            q_t = q if torch.is_tensor(q) else torch.full_like(p, float(q))
            denom = (1.0 - p).clamp_min(1e-6)
            q_cont = (q_t / denom).clamp(0.0, 1.0 - 1e-6)
            cont_q = self.cont.quantile(obs, q_cont, iters=iters)
            return torch.where(q_t >= (1.0 - p), torch.full_like(cont_q, float(self.cap)), cont_q)

    def sample(self, obs, num_samples=1):
        p = self.p_survive(obs).unsqueeze(-1)
        surv = torch.rand(obs.shape[0], num_samples, device=obs.device) < p
        cont_s = self.cont.sample(obs, num_samples)
        return torch.where(surv, torch.full_like(cont_s, float(self.cap)), cont_s)

    def mean(self, obs):
        return self.sample(obs, 1024).mean(dim=-1)

    def std(self, obs):
        return self.sample(obs, 1024).std(dim=-1)


def mdn_nll(log_pi, mu, sigma, y_norm):
    y = y_norm.unsqueeze(-1)
    log_comp = log_pi - torch.log(sigma) - 0.5 * _LOG_2PI - 0.5 * ((y - mu) / sigma) ** 2
    return -torch.logsumexp(log_comp, dim=-1).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _kmeans_1d(values, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    values = values.reshape(-1)
    if values.shape[0] <= k:
        lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
        return np.linspace(lo, hi, k).astype(np.float32)
    centroids = rng.choice(values, size=k, replace=False).astype(np.float64)
    for _ in range(iters):
        assign = np.abs(values[:, None] - centroids[None, :]).argmin(axis=1)
        new = centroids.copy()
        for j in range(k):
            members = values[assign == j]
            if members.size:
                new[j] = members.mean()
        if np.allclose(new, centroids):
            break
        centroids = new
    return np.sort(centroids).astype(np.float32)


def _init_mu_bias(model, z_norm_np, num_components, seed, device):
    centroids = _kmeans_1d(z_norm_np, num_components, seed=seed)
    final_linear = [m for m in model.net if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        final_linear.bias[num_components:2 * num_components] = torch.tensor(
            centroids, dtype=final_linear.bias.dtype, device=device)


def train_mdn(obs, targets, num_components, hidden_sizes, sigma_floor, batch_size, epochs, lr,
              device, seed=0, transform="none", cap=1.0, verbose=True):
    """Fit a (optionally logit-transformed) MDN by NLL on the modeled variable z = transform(G)."""
    tmp = MixtureDensityNetwork(obs.shape[1], num_components, list(hidden_sizes), sigma_floor,
                                transform=transform, cap=cap)  # only for _to_z
    z = tmp._to_z(targets)
    z_mean, z_std = float(z.mean().item()), float(z.std().item())
    if z_std < 1e-8:
        z_std = 1.0
    z_norm = (z - z_mean) / z_std

    model = MixtureDensityNetwork(obs.shape[1], num_components, list(hidden_sizes), sigma_floor,
                                  return_mean=z_mean, return_std=z_std, transform=transform,
                                  cap=cap).to(device)
    _init_mu_bias(model, z_norm.detach().cpu().numpy(), num_components, seed, device)

    loader = DataLoader(TensorDataset(obs, z_norm), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        tot, n = 0.0, 0
        for obs_b, y_b in loader:
            optimizer.zero_grad()
            loss = mdn_nll(*model(obs_b), y_b)
            loss.backward(); optimizer.step()
            tot += loss.item() * obs_b.shape[0]; n += obs_b.shape[0]
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - NLL (z, normalized): {tot / max(n, 1):.6f}")
    return model


def train_hurdle_mdn(obs, targets, num_components, hidden_sizes, sigma_floor, batch_size, epochs, lr,
                     device, cap, tol, seed=0, verbose=True):
    """Fit a hurdle model: BCE survive-to-cap head + continuous logit-MDN NLL on sub-cap returns."""
    surv = (targets >= cap - tol)
    sub = targets[~surv]
    if sub.numel() < num_components + 1:
        raise RuntimeError("Almost all returns are at the cap; hurdle has no sub-cap data to fit. "
                           "Lower --reward perturbation or raise --random-steps.")
    cont_tmp = MixtureDensityNetwork(obs.shape[1], num_components, list(hidden_sizes), sigma_floor,
                                     transform="logit", cap=cap)
    z_sub = cont_tmp._to_z(sub)
    z_mean, z_std = float(z_sub.mean().item()), float(z_sub.std().item())
    if z_std < 1e-8:
        z_std = 1.0

    model = HurdleMDN(obs.shape[1], num_components, list(hidden_sizes), sigma_floor, cap, tol,
                      return_mean=z_mean, return_std=z_std).to(device)
    _init_mu_bias(model.cont, ((z_sub - z_mean) / z_std).detach().cpu().numpy(),
                  num_components, seed, device)

    loader = DataLoader(TensorDataset(obs, targets), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        tot_b, tot_c, n = 0.0, 0.0, 0
        for obs_b, g_b in loader:
            optimizer.zero_grad()
            s_b = (g_b >= cap - tol).float()
            loss_b = bce(model.survive_logit(obs_b), s_b)
            m = g_b < cap - tol
            loss_c = torch.zeros((), device=device)
            if m.any():
                y_norm = model.cont.transform_normalize(g_b[m])
                loss_c = mdn_nll(*model.cont(obs_b[m]), y_norm)
            (loss_b + loss_c).backward(); optimizer.step()
            tot_b += loss_b.item() * obs_b.shape[0]; tot_c += float(loss_c) * obs_b.shape[0]
            n += obs_b.shape[0]
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - survive BCE {tot_b / max(n,1):.4f} | "
                  f"cont NLL {tot_c / max(n,1):.4f} | p(survive)~{model.p_survive(obs[:512]).mean().item():.3f}")
    return model


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------
def _normal_abs_expectation(m, var):
    s = torch.sqrt(torch.clamp(var, min=1e-12))
    ratio = m / s
    phi = torch.exp(-0.5 * ratio ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(ratio / math.sqrt(2.0)))
    return m * (2.0 * Phi - 1.0) + 2.0 * s * phi


def gmm_crps_closed(model, obs, y):
    log_pi, mu, sigma = model._params_z(obs)
    w = torch.exp(log_pi); var = sigma ** 2
    t1 = (w * _normal_abs_expectation(mu - y.unsqueeze(-1), var)).sum(dim=-1)
    mu_diff = mu.unsqueeze(-1) - mu.unsqueeze(-2)
    var_sum = var.unsqueeze(-1) + var.unsqueeze(-2)
    w_outer = w.unsqueeze(-1) * w.unsqueeze(-2)
    t2 = 0.5 * (w_outer * _normal_abs_expectation(mu_diff, var_sum)).sum(dim=(-1, -2))
    return t1 - t2


def crps_mc(model, obs, y, n=200):
    """Sample-based CRPS (energy form): E|X-y| - 0.5 E|X-X'|. Works for any transform/hurdle."""
    with torch.no_grad():
        x = model.sample(obs, n); xp = model.sample(obs, n)
        return (x - y[:, None]).abs().mean(-1) - 0.5 * (x - xp).abs().mean(-1)


def _gmm_quantile(model, obs, q, iters=40):
    return model.quantile(obs, q, iters=iters)


def evaluate_flat(model, obs_flat, y_flat, alpha):
    model.eval()
    is_plain_gmm = isinstance(model, MixtureDensityNetwork) and model.transform == "none"
    with torch.no_grad():
        if isinstance(model, MixtureDensityNetwork):
            nll = -model.log_prob(obs_flat, model.transform_normalize(y_flat)).mean().item()
        else:
            nll = float("nan")  # hurdle mixes a point mass + continuous; NLL not comparable
        crps = (gmm_crps_closed(model, obs_flat, y_flat) if is_plain_gmm
                else crps_mc(model, obs_flat, y_flat)).mean().item()
        pit = model.pit(obs_flat, y_flat)
        lower = model.quantile(obs_flat, alpha / 2.0)
        upper = model.quantile(obs_flat, 1.0 - alpha / 2.0)
        covered = ((y_flat >= lower) & (y_flat <= upper)).float().mean().item()
        mean_width = (upper - lower).mean().item()
    metrics = {"samples": int(y_flat.shape[0]), "nll": float(nll), "crps": float(crps),
               "alpha": float(alpha), "interval_coverage": float(covered),
               "mean_interval_width": float(mean_width),
               "pit_mean": float(pit.mean().item()), "pit_std": float(pit.std().item())}
    print("MDN eval | samples: {samples} | NLL: {nll:.4f} | CRPS: {crps:.4f} | "
          "{:.0f}% coverage: {interval_coverage:.4f} | width: {mean_interval_width:.4f} | "
          "PIT mean/std: {pit_mean:.3f}/{pit_std:.3f}".format((1 - alpha) * 100, **metrics))
    return metrics, pit.detach().cpu().numpy()


def per_state_metrics(model, obs, returns, alpha=None, cap=None, tol=None):
    """Per-anchor predicted-vs-empirical statistics.

    Beyond mean/std/KS/Wasserstein, this optionally computes two **monotonic, decision-relevant**
    adaptivity metrics that (unlike Pearson-on-std, which is non-monotonic for hurdle models --
    std is a cap*sqrt(p(1-p))-shaped function of survival prob) actually reflect whether the model
    resolves per-state risk:
      * lower-quantile recovery: predicted q_alpha(s) vs empirical q_alpha(s)   (needs ``alpha``)
      * survival-rate recovery : predicted P(G>=cap-tol|s) vs empirical fraction (needs ``cap,tol``)
    """
    model.eval()
    A, N = returns.shape
    ret_np = returns.detach().cpu().numpy()
    emp_mean, emp_std = ret_np.mean(axis=1), ret_np.std(axis=1)
    out = {}
    with torch.no_grad():
        pred_mean = model.mean(obs).detach().cpu().numpy()
        pred_std = model.std(obs).detach().cpu().numpy()
        sorted_ret = np.sort(ret_np, axis=1)
        # Two-sample KS (model draws vs empirical) -- atom-safe, unlike analytic-CDF-vs-empirical,
        # which breaks for spiked/atomic distributions (e.g. the hurdle's point mass at the cap).
        model_draws = model.sample(obs, N).detach().cpu().numpy()
        sorted_draws = np.sort(model_draws, axis=1)
        emp_cdf = np.arange(1, N + 1) / N
        ks = np.empty(A)
        for a in range(A):
            pred_at = np.searchsorted(sorted_draws[a], sorted_ret[a], side="right") / N
            ks[a] = float(np.max(np.abs(pred_at - emp_cdf)))
        wass = np.mean(np.abs(sorted_draws - sorted_ret), axis=1)
        if alpha is not None:
            out["pred_q_lower"] = model.quantile(obs, alpha).detach().cpu().numpy()
            out["emp_q_lower"] = np.quantile(ret_np, alpha, axis=1)
        if cap is not None and tol is not None:
            if hasattr(model, "p_survive"):          # hurdle: exact survival head
                pred_surv = model.p_survive(obs)
            else:                                     # plain/transformed MDN: P(G >= cap - tol)
                pred_surv = 1.0 - model.cdf(obs, torch.full((A,), float(cap - tol), device=obs.device))
            out["pred_survival"] = pred_surv.detach().cpu().numpy()
            out["emp_survival"] = (ret_np >= (cap - tol)).mean(axis=1)
    out.update({"pred_mean": pred_mean, "emp_mean": emp_mean, "pred_std": pred_std,
                "emp_std": emp_std, "ks": ks, "wasserstein": wass})
    return out


# ---------------------------------------------------------------------------
# Split-conformal layer
# ---------------------------------------------------------------------------
def conformal_offset(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("No calibration scores.")
    q_index = min(max(math.ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(np.partition(scores, q_index)[q_index])


def one_per_anchor(returns, seed=0):
    A, N = returns.shape
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N, (A,), generator=g)
    return returns[torch.arange(A), idx]


def calibrate_constant_delta(model, obs, y, alpha):
    with torch.no_grad():
        mu = model.mean(obs)
    return conformal_offset(torch.abs(y - mu).cpu().numpy(), alpha)


def calibrate_cqr(model, obs, y, alpha):
    with torch.no_grad():
        lo = model.quantile(obs, alpha / 2.0); hi = model.quantile(obs, 1.0 - alpha / 2.0)
    return conformal_offset(torch.maximum(lo - y, y - hi).cpu().numpy(), alpha)


def calibrate_one_sided(model, obs, y, alpha, side):
    with torch.no_grad():
        if side == "upper":
            scores = (y - model.quantile(obs, 1.0 - alpha)).cpu().numpy()
        else:
            scores = (model.quantile(obs, alpha) - y).cpu().numpy()
    return conformal_offset(scores, alpha)


def certified_margin(model, obs, q_one, alpha, side):
    with torch.no_grad():
        if side == "upper":
            return model.quantile(obs, 1.0 - alpha) + q_one
        return model.quantile(obs, alpha) - q_one


def conformal_interval(model, obs, q_cqr, alpha):
    with torch.no_grad():
        return model.quantile(obs, alpha / 2.0) - q_cqr, model.quantile(obs, 1.0 - alpha / 2.0) + q_cqr


def _interval_cov_width(returns, lo, hi):
    cov = ((returns >= lo[:, None]) & (returns <= hi[:, None])).float().mean().item()
    return cov, (hi - lo).mean().item()


def _one_sided_cov(returns, margin, side):
    if side == "upper":
        return (returns <= margin[:, None]).float().mean().item()
    return (returns >= margin[:, None]).float().mean().item()


def evaluate_conformal(model, test_obs, test_returns, alpha, side, delta_const, q_cqr, q_one):
    model.eval()
    with torch.no_grad():
        mu = model.mean(test_obs)
        lo_raw = model.quantile(test_obs, alpha / 2.0)
        hi_raw = model.quantile(test_obs, 1.0 - alpha / 2.0)
        base_one = model.quantile(test_obs, 1.0 - alpha) if side == "upper" \
            else model.quantile(test_obs, alpha)
        pred_std = model.std(test_obs)
    raw_cov, raw_w = _interval_cov_width(test_returns, lo_raw, hi_raw)
    const_cov, const_w = _interval_cov_width(test_returns, mu - delta_const, mu + delta_const)
    cqr_lo, cqr_hi = lo_raw - q_cqr, hi_raw + q_cqr
    cqr_cov, cqr_w = _interval_cov_width(test_returns, cqr_lo, cqr_hi)
    margin = base_one + q_one if side == "upper" else base_one - q_one
    one_cov = _one_sided_cov(test_returns, margin, side)
    table = {
        "raw_model": {"coverage": raw_cov, "mean_width": raw_w},
        "constant_delta": {"coverage": const_cov, "mean_width": const_w},
        "cqr_two_sided": {"coverage": cqr_cov, "mean_width": cqr_w},
        "one_sided_margin": {"coverage": one_cov, "mean_margin": float(margin.mean().item()), "side": side},
    }
    arrays = {"const_width": np.full(test_obs.shape[0], 2.0 * delta_const),
              "cqr_width": (cqr_hi - cqr_lo).detach().cpu().numpy(),
              "margin": margin.detach().cpu().numpy(),
              "pred_std": pred_std.detach().cpu().numpy()}
    return {"table": table, "arrays": arrays}


def coverage_one_per_anchor(model, test_obs, test_returns, alpha, side, q_cqr, q_one, reps, seed=0):
    """Coverage measured with ONE random sample per test anchor (the exchangeable unit),
    averaged over ``reps`` random draws. Returns mean/std for CQR interval and one-sided margin."""
    model.eval()
    A, N = test_returns.shape
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lo = model.quantile(test_obs, alpha / 2.0) - q_cqr
        hi = model.quantile(test_obs, 1.0 - alpha / 2.0) + q_cqr
        base = model.quantile(test_obs, 1.0 - alpha) if side == "upper" else model.quantile(test_obs, alpha)
        margin = base + q_one if side == "upper" else base - q_one
    cqr_cov, one_cov = [], []
    for _ in range(reps):
        idx = torch.randint(0, N, (A,), generator=g)
        y = test_returns[torch.arange(A), idx]
        cqr_cov.append(((y >= lo) & (y <= hi)).float().mean().item())
        one_cov.append((y <= margin).float().mean().item() if side == "upper"
                       else (y >= margin).float().mean().item())
    return {"reps": reps, "cqr_coverage_mean": float(np.mean(cqr_cov)), "cqr_coverage_std": float(np.std(cqr_cov)),
            "one_sided_coverage_mean": float(np.mean(one_cov)), "one_sided_coverage_std": float(np.std(one_cov))}


def threshold_decision(model, test_obs, test_returns, alpha, q_one, side, threshold):
    model.eval()
    margin = certified_margin(model, test_obs, q_one, alpha, side).detach().cpu()
    if side == "lower":
        certified_safe = margin >= threshold; viol = lambda g: (g < threshold)
    else:
        certified_safe = margin <= threshold; viol = lambda g: (g > threshold)
    n_safe = int(certified_safe.sum().item()); A = test_obs.shape[0]
    safe_returns = test_returns[certified_safe]
    violation_rate = float(viol(safe_returns).float().mean().item()) if n_safe > 0 else float("nan")
    return {"threshold": float(threshold), "side": side, "alpha": float(alpha),
            "fraction_certified_safe": n_safe / A, "fraction_flagged_unsafe": (A - n_safe) / A,
            "violation_rate_among_certified_safe": violation_rate,
            "n_test_anchors": A, "n_certified_safe": n_safe}


# ---------------------------------------------------------------------------
# Parallel collection harness
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Diagnostics (separate image files)
# ---------------------------------------------------------------------------
def _corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _scatter_identity(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.scatter(x, y, s=10, alpha=0.5); ax.plot([lo, hi], [lo, hi], "r--", label="ideal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_diagnostics(model, val_obs, val_returns, val_obs_flat, val_y_flat, pit, psm, alpha,
                     plot_dir, num_density_examples=6):
    os.makedirs(plot_dir, exist_ok=True)
    P = lambda name: os.path.join(plot_dir, name)

    _scatter_identity(psm["emp_mean"], psm["pred_mean"], "Empirical conditional mean E[G|s]",
                      "Predicted mixture mean", f"Mean recovery (r={_corr(psm['emp_mean'], psm['pred_mean']):.3f})",
                      P("mean_vs_empirical_mean.png"))
    _scatter_identity(psm["emp_std"], psm["pred_std"], "Empirical conditional std",
                      "Predicted mixture std", f"Spread calibration (r={_corr(psm['emp_std'], psm['pred_std']):.3f})",
                      P("std_vs_empirical_std.png"))

    n_extra = 0
    if "emp_q_lower" in psm:
        _scatter_identity(psm["emp_q_lower"], psm["pred_q_lower"], "Empirical q_alpha (lower tail)",
                          "Predicted q_alpha", f"Lower-quantile recovery (r={_corr(psm['emp_q_lower'], psm['pred_q_lower']):.3f})",
                          P("q_lower_recovery.png")); n_extra += 1
    if "emp_survival" in psm:
        _scatter_identity(psm["emp_survival"], psm["pred_survival"], "Empirical survival rate",
                          "Predicted survival prob", f"Survival recovery (r={_corr(psm['emp_survival'], psm['pred_survival']):.3f})",
                          P("survival_recovery.png")); n_extra += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pit, bins=20, range=(0, 1), density=True, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axhline(1.0, color="r", ls="--")
    ax.set_xlabel("PIT  F(G | s)"); ax.set_ylabel("density"); ax.set_title("PIT histogram")
    fig.tight_layout(); fig.savefig(P("pit_histogram.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    sp = np.sort(pit); uq = (np.arange(1, len(sp) + 1) - 0.5) / len(sp)
    ax.plot(uq, sp, ".", ms=3); ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("Uniform quantile"); ax.set_ylabel("PIT quantile"); ax.set_title("PIT QQ-plot")
    fig.tight_layout(); fig.savefig(P("pit_qq.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    levels = np.linspace(0.05, 0.95, 19); cov = []
    with torch.no_grad():
        for lv in levels:
            lo = model.quantile(val_obs_flat, (1 - lv) / 2.0)
            hi = model.quantile(val_obs_flat, 1 - (1 - lv) / 2.0)
            cov.append(float(((val_y_flat >= lo) & (val_y_flat <= hi)).float().mean().item()))
    ax.plot(levels, cov, "o-", label="empirical"); ax.plot([0, 1], [0, 1], "r--", label="ideal")
    ax.set_xlabel("nominal coverage 1-a"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability (interval coverage)"); ax.legend()
    fig.tight_layout(); fig.savefig(P("reliability.png"), dpi=180); plt.close(fig)

    for key, color, fname in [("ks", "darkorange", "ks_hist.png"),
                              ("wasserstein", "seagreen", "wasserstein_hist.png")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(psm[key], bins=20, color=color, edgecolor="k", alpha=0.8)
        ax.set_xlabel(key); ax.set_ylabel("count")
        ax.set_title(f"Per-state {key} (median {np.median(psm[key]):.3f})")
        fig.tight_layout(); fig.savefig(P(fname), dpi=180); plt.close(fig)

    # Per-state density: predicted SAMPLES (transform/hurdle-agnostic) vs the N empirical returns.
    A, N = val_returns.shape
    n_ex = min(num_density_examples, A)
    examples = np.argsort(psm["emp_std"])[np.linspace(0, A - 1, n_ex).astype(int)]
    ret_np = val_returns.detach().cpu().numpy()
    with torch.no_grad():
        for j, a in enumerate(examples):
            ys = ret_np[a]
            draws = model.sample(val_obs[a:a + 1], 3000)[0].cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 4))
            lo = float(min(ys.min(), draws.min())); hi = float(max(ys.max(), draws.max()))
            bins = np.linspace(lo, hi, 40)
            ax.hist(ys, bins=bins, density=True, alpha=0.45, color="gray", label=f"empirical (N={N})")
            ax.hist(draws, bins=bins, density=True, histtype="step", color="b", lw=2, label="predicted")
            ax.set_xlabel("Return G"); ax.set_ylabel("density")
            ax.set_title(f"Anchor #{int(a)} | emp std {psm['emp_std'][a]:.2f} pred std {psm['pred_std'][a]:.2f}")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(P(f"per_state_density_{j:02d}.png"), dpi=180); plt.close(fig)
    print(f"Saved {7 + n_ex + n_extra} diagnostic images to {plot_dir}")


def save_conformal_plots(model, calib_obs, calib_y, test_obs, test_returns, alpha, side,
                         offsets, evalres, plot_dir, threshold=None):
    os.makedirs(plot_dir, exist_ok=True)
    P = lambda name: os.path.join(plot_dir, name)
    delta_const, q_cqr, q_one = offsets

    levels = np.linspace(0.05, 0.95, 19); raw_cov, conf_cov = [], []
    with torch.no_grad():
        for lv in levels:
            a = 1.0 - lv
            lo_t = model.quantile(test_obs, a / 2.0); hi_t = model.quantile(test_obs, 1.0 - a / 2.0)
            raw_cov.append(_interval_cov_width(test_returns, lo_t, hi_t)[0])
            lo_c = model.quantile(calib_obs, a / 2.0); hi_c = model.quantile(calib_obs, 1.0 - a / 2.0)
            q = conformal_offset(torch.maximum(lo_c - calib_y, calib_y - hi_c).cpu().numpy(), a)
            conf_cov.append(_interval_cov_width(test_returns, lo_t - q, hi_t + q)[0])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(levels, raw_cov, "o-", color="gray", label="raw model")
    ax.plot(levels, conf_cov, "s-", color="b", label="conformal (CQR)")
    ax.plot([0, 1], [0, 1], "r--", label="ideal")
    ax.set_xlabel("nominal coverage 1-alpha"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability: raw vs conformal"); ax.legend()
    fig.tight_layout(); fig.savefig(P("reliability_raw_vs_conformal.png"), dpi=180); plt.close(fig)

    t = evalres["table"]; names = ["raw_model", "constant_delta", "cqr_two_sided", "one_sided_margin"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, [t[n]["coverage"] for n in names],
           color=["gray", "indianred", "steelblue", "seagreen"], alpha=0.85)
    ax.axhline(1.0 - alpha, color="r", ls="--", label=f"target {1-alpha:.2f}")
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1.02); ax.set_title("Coverage by method"); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(P("coverage_comparison.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(evalres["arrays"]["cqr_width"], bins=25, color="steelblue", alpha=0.8, label="CQR (adaptive)")
    ax.axvline(2.0 * delta_const, color="indianred", lw=2, ls="--", label="constant-delta")
    ax.set_xlabel("interval width"); ax.set_ylabel("count")
    ax.set_title("Interval width: constant vs adaptive"); ax.legend()
    fig.tight_layout(); fig.savefig(P("width_comparison.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(evalres["arrays"]["pred_std"], evalres["arrays"]["cqr_width"], s=10, alpha=0.5)
    ax.set_xlabel("predicted std"); ax.set_ylabel("CQR interval width")
    ax.set_title("Adaptivity: width vs uncertainty")
    fig.tight_layout(); fig.savefig(P("margin_vs_uncertainty.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(evalres["arrays"]["margin"], bins=25, color="seagreen", alpha=0.85)
    ax.set_xlabel(f"certified {side} margin M(s)"); ax.set_ylabel("count")
    ax.set_title("Certified margin distribution")
    fig.tight_layout(); fig.savefig(P("margin_hist.png"), dpi=180); plt.close(fig)

    if threshold is not None:
        margin = evalres["arrays"]["margin"]; std = evalres["arrays"]["pred_std"]
        safe = (margin >= threshold) if side == "lower" else (margin <= threshold)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(std[safe], margin[safe], s=12, c="seagreen", alpha=0.6, label="certified safe")
        ax.scatter(std[~safe], margin[~safe], s=12, c="indianred", alpha=0.6, label="flagged unsafe")
        ax.axhline(threshold, color="r", ls="--", label=f"threshold {threshold:g}")
        ax.set_xlabel("predicted std"); ax.set_ylabel(f"certified {side} margin M(s)")
        ax.set_title("Per-state safety decision"); ax.legend()
        fig.tight_layout(); fig.savefig(P("safety_decision.png"), dpi=180); plt.close(fig)
    print(f"Saved conformal plots to {plot_dir}")
