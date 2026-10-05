"""
Shared SafetyPointGoal1 machinery: policy loading, MuJoCo state save/restore, Monte-Carlo rollouts,
parallel anchor collection, split-conformal helpers and the per-state statistics used by the
grouping analysis. The reduction is always the reward reduction, ΔG = G_clean - G_perturbed.

The policy is a CPO agent trained with SafePO (https://github.com/PKU-Alignment/Safe-Policy-Optimization);
``eval_dir`` is a SafePO run directory holding ``config.json``, ``torch_save/model*.pt`` and the
observation normalizer ``state*.pkl``.
"""
import json
import math
import multiprocessing
import os
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco
from safepo.common.buffer import discount_cumsum
from safepo.common.env import make_sa_mujoco_env
from safepo.common.model import ActorVCritic


# ---------------------------------------------------------------------------
# Policy and simulator state
# ---------------------------------------------------------------------------
def _ckpt_key(f: str) -> int:
    """Numeric epoch key for checkpoint / normalizer filenames (``model599.pt`` -> 599).

    A lexicographic sort would rank ``model99.pt`` after ``model599.pt``. Files without digits sort first.
    """
    digits = "".join(ch for ch in f if ch.isdigit())
    return int(digits) if digits else -1


def _load_actor(eval_dir: str, device: torch.device) -> Tuple[ActorVCritic, dict]:
    """Load the policy, its environment and observation normalizer from a SafePO run dir.

    Checkpoints may be actor-only (keys ``mean.*`` / ``log_std``) or full ActorVCritic dumps
    (``actor.`` / ``reward_critic.`` / ``cost_critic.`` prefixes); both are handled.
    """
    config = json.load(open(os.path.join(eval_dir, "config.json"), "r"))
    env_id = config["task"] if "task" in config else config["env_name"]
    env, obs_space, act_space = make_sa_mujoco_env(num_envs=1, env_id=env_id, seed=None)

    model_dir = os.path.join(eval_dir, "torch_save")
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No policy checkpoints found under {model_dir}")
    model_path = os.path.join(model_dir, sorted(model_files, key=_ckpt_key)[-1])

    policy = ActorVCritic(obs_dim=obs_space.shape[0], act_dim=act_space.shape[0],
                          hidden_sizes=config["hidden_sizes"]).to(device)
    state_dict = torch.load(model_path, map_location=device)
    is_actor_only = not any(k.startswith(("actor.", "reward_critic.", "cost_critic."))
                            for k in state_dict.keys())
    target = policy.actor if is_actor_only else policy
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    actor_missing = [m for m in missing if m.startswith(("mean.", "actor.")) or m == "log_std"]
    if actor_missing:
        raise RuntimeError("Checkpoint is missing actor parameters: " + ", ".join(actor_missing))
    if unexpected:
        print(f"Warning: unused checkpoint keys: {unexpected}")
    policy.eval()

    norm_files = [f for f in os.listdir(eval_dir) if f.endswith(".pkl")]
    if norm_files:
        norm_path = os.path.join(eval_dir, sorted(norm_files, key=_ckpt_key)[-1])
        env.obs_rms = joblib.load(open(norm_path, "rb"))["Normalizer"]

    return policy, {"env": env, "obs_space": obs_space, "act_space": act_space, "config": config,
                    "model_path": model_path}


def _get_sim(env):
    """Return the underlying (MjModel, MjData) for a wrapped single-agent env."""
    task = env.unwrapped.task
    return task.model, task.data


def _save_sim_state(data) -> dict:
    """Snapshot the full physics state (mj_getState is unavailable on mujoco 2.3.3)."""
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy() if data.act.size else None,
        "time": float(data.time),
        "mocap_pos": data.mocap_pos.copy(),
        "mocap_quat": data.mocap_quat.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
    }


def _restore_sim_state(model, data, state: dict) -> None:
    """Write a saved physics state back and recompute derived quantities."""
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    if state["act"] is not None and data.act.size:
        data.act[:] = state["act"]
    data.time = state["time"]
    data.mocap_pos[:] = state["mocap_pos"]
    data.mocap_quat[:] = state["mocap_quat"]
    data.qacc_warmstart[:] = state["qacc_warmstart"]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Rollouts (discounted reward return)
# ---------------------------------------------------------------------------
def _rollout_return_from_here(env, policy, device, random_steps, gamma, max_steps):
    """Random burst of ``random_steps`` actions, then the nominal policy to episode end.

    Returns the discounted reward return from the restored state and the number of steps taken.
    The burst's actions need no observation, so the (wrapper-normalized) observation is read from
    ``env.step``; requires ``random_steps >= 1``.
    """
    rewards: List[float] = []
    obs_torch = None
    done = False
    steps = 0
    while not done and steps < max_steps:
        if steps < random_steps or obs_torch is None:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                act_t, _, _, _ = policy.step(obs_torch, deterministic=True)
            action = act_t.squeeze(0).detach().cpu().numpy()
        next_obs, reward, _, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward[0]))
        obs_torch = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        steps += 1
        done = bool(terminated[0] or truncated[0])
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    g_r = float(discount_cumsum(rewards_tensor, gamma)[0]) if rewards else 0.0
    return g_r, steps


def _rollout_return_clean(env, policy, device, gamma: float, max_steps: int,
                          anchor_obs_np: np.ndarray) -> float:
    """Deterministic policy from the already-restored state; ``anchor_obs_np`` is the first input."""
    rewards: List[float] = []
    obs_torch = torch.as_tensor(anchor_obs_np, dtype=torch.float32, device=device)
    done = False
    steps = 0
    while not done and steps < max_steps:
        with torch.no_grad():
            act_t, _, _, _ = policy.step(obs_torch, deterministic=True)
        action = act_t.squeeze(0).detach().cpu().numpy()
        next_obs, reward, _, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward[0]))
        obs_torch = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        steps += 1
        done = bool(terminated[0] or truncated[0])
    return float(discount_cumsum(torch.tensor(rewards), gamma)[0]) if rewards else 0.0


def _progress_monitor(counter, total: int, pbar: tqdm, stop: threading.Event) -> None:
    """Thread target: polls the shared counter and updates the main-process progress bar."""
    last = 0
    while not stop.is_set():
        current = counter.value
        if current != last:
            pbar.update(current - last)
            last = current
        if last >= total:
            break
        time.sleep(0.05)
    current = counter.value
    if current > last:
        pbar.update(current - last)


# ---------------------------------------------------------------------------
# Single-length collection (used by safety_margin_adaptivity_direct.py)
# ---------------------------------------------------------------------------
def _collect_worker(eval_dir: str, num_anchors: int, samples_per_state: int, random_prob: float,
                    random_steps: int, gamma_override: float, max_steps: int, history_len: int,
                    seed: int, counter=None):
    """One worker: anchors by a per-step Bernoulli trigger, one clean and N perturbed rollouts each.

    Returns features [A, obs_dim*history_len], drops_r [A, N], g_clean_r [A], model_path, anchor_steps.
    """
    device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    feat_buffer, drops_buffer, g_clean_buffer, anchor_steps = [], [], [], []
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
                anchor_feature = np.concatenate(list(hist), axis=0)     # [history_len * obs_dim]
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

        _restore_sim_state(sim_model, sim_data, anchor_state)
        g_clean = _rollout_return_clean(env, policy, device, gamma, max_steps, anchor_obs_np)
        drops = np.empty(samples_per_state, dtype=np.float32)
        for i in range(samples_per_state):
            _restore_sim_state(sim_model, sim_data, anchor_state)
            try:
                env.action_space.seed(seed * 100003 + collected * 911 + i)
            except Exception:
                pass
            g_i, _ = _rollout_return_from_here(env, policy, device, random_steps, gamma, max_steps)
            drops[i] = g_clean - g_i

        feat_buffer.append(anchor_feature)
        drops_buffer.append(drops)
        g_clean_buffer.append(g_clean)
        anchor_steps.append(t0)
        collected += 1
        if counter is not None:
            counter.value += 1

    if not feat_buffer:
        return (np.empty((0, input_dim), np.float32), np.empty((0, samples_per_state), np.float32),
                np.empty(0, np.float32), extra["model_path"], anchor_steps)
    return (np.stack(feat_buffer).astype(np.float32), np.stack(drops_buffer).astype(np.float32),
            np.array(g_clean_buffer, np.float32), extra["model_path"], anchor_steps)


def collect_parallel(eval_dir: str, num_anchors: int, samples_per_state: int, random_prob: float,
                     random_steps: int = 1, gamma: float = None, max_steps: int = 1000,
                     history_len: int = 2, num_workers: int = None, base_seed: int = 0):
    """Parallel single-length collector. Returns features [A, D], drops_r [A, N], g_clean_r [A], extra."""
    num_workers = max(1, min(num_workers or os.cpu_count() or 1, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    counts = [c for c in counts if c > 0]
    print(f"Collecting {num_anchors} anchors x {samples_per_state} perturbed + 1 clean rollout "
          f"| n={random_steps} | history_len={history_len} | {len(counts)} workers")

    stop_event = threading.Event()
    with multiprocessing.Manager() as manager:
        counter = manager.Value("i", 0)
        with tqdm(total=num_anchors, desc="Anchors", unit="anchor") as pbar:
            monitor = threading.Thread(target=_progress_monitor,
                                       args=(counter, num_anchors, pbar, stop_event), daemon=True)
            monitor.start()
            results = joblib.Parallel(n_jobs=len(counts), backend="loky", verbose=0)(
                joblib.delayed(_collect_worker)(eval_dir, c, samples_per_state, random_prob,
                                                random_steps, gamma, max_steps, history_len,
                                                base_seed + i, counter)
                for i, c in enumerate(counts))
            stop_event.set()
            monitor.join(timeout=2.0)

    if not any(r[0].shape[0] > 0 for r in results):
        raise RuntimeError("No anchors collected; increase --random-prob or --num-anchors.")
    feats = np.concatenate([r[0] for r in results if r[0].shape[0] > 0], axis=0)
    drops = np.concatenate([r[1] for r in results if r[1].shape[0] > 0], axis=0)
    g_clean = np.concatenate([r[2] for r in results if r[2].shape[0] > 0], axis=0)
    input_dim = feats.shape[1]
    obs_dim = input_dim // history_len
    print(f"Collected {feats.shape[0]} anchors | input_dim={input_dim} | "
          f"reward drop mean {drops.mean():.3f} std {drops.std():.3f}")
    return feats, drops, g_clean, {"model_path": results[0][3], "obs_dim": obs_dim,
                                   "input_dim": input_dim}


# ---------------------------------------------------------------------------
# Split-conformal helpers
# ---------------------------------------------------------------------------
def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest of the (possibly signed) conformity scores."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("No calibration scores available for conformal calibration.")
    q_index = min(max(math.ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(np.partition(scores, q_index)[q_index])


def one_per_anchor(returns: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Pick one draw per anchor (random column) so calibration points stay exchangeable. [A, N] -> [A]."""
    A, N = returns.shape
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N, (A,), generator=g)
    return returns[torch.arange(A), idx]


def calibrate_global_center(calib_y: torch.Tensor, center: torch.Tensor, alpha: float) -> float:
    """Single one-sided offset Q for M(s) = center(s) + Q."""
    return conformal_offset((calib_y - center).detach().cpu().numpy(), alpha)


def mondrian_calibrate(calib_y: torch.Tensor, center: torch.Tensor, pexc_cal: np.ndarray,
                       alpha: float, groups: int):
    """Group-conditional one-sided offsets keyed to quantile bins of a predicted risk score.

    Returns bin ``edges``, per-group offsets ``Q``, the global fallback ``q_global`` and per-group counts.
    """
    scores = (calib_y - center).detach().cpu().numpy()
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


def mondrian_apply(center: torch.Tensor, pexc: np.ndarray, edges: np.ndarray,
                   Q: Dict[int, float], q_global: float):
    """Assign states to risk groups via ``edges``; return (group bound, group ids)."""
    g = np.digitize(np.asarray(pexc, float), edges)
    q_arr = np.array([Q.get(int(gi), q_global) for gi in g], dtype=np.float64)
    return center + torch.as_tensor(q_arr, dtype=torch.float32, device=center.device), g


# ---------------------------------------------------------------------------
# Per-state statistics and their split-half reliability
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
    """Split-half correlation of a per-state statistic, its Spearman-Brown reliability at full N,
    and the ceiling sqrt(reliability) on the correlation any predictor can reach."""
    A, N = samples.shape
    cols = np.random.default_rng(seed).permutation(N)
    half = N // 2
    r = _pearson(stat_fn(samples[:, cols[:half]]), stat_fn(samples[:, cols[half:2 * half]]))
    sb = (2.0 * r / (1.0 + r)) if (np.isfinite(r) and r > -0.999) else float("nan")
    ceiling = float(np.sqrt(sb)) if (np.isfinite(sb) and sb > 0) else 0.0
    return {"r_half": r, "sb_reliability": sb, "ceiling": ceiling}


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


def _scatter_identity(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
    ax.scatter(x, y, s=10, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], "r--", label="ideal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
