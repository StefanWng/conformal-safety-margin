"""
Atari front-end for the certified safety-margin pipeline (BeamRider / QRDQN).

This is the ONLY genuinely new code in the Atari port. It supplies, for the two drivers
(``atari_adaptivity_direct.py`` and ``atari_grushin.py``), everything the SafetyGymnasium / Pendulum
front-ends supplied there, adapted to an image-observation, emulator-state setting:

  * ``build_env`` / ``load_model`` — the exact validated SB3-zoo preprocessing + the 4-part
    ``custom_objects`` loading cascade (see ``eval_beamrider.py`` / [[atari-extension-setup]]);
  * ``embed`` — the anchor FEATURE: the QRDQN NatureCNN 512-d embedding (HGB/MDN cannot consume the
    84x84x4 image, so we use the policy's own learned state representation — the direct analog of
    SafetyGym's 60-d obs). Optional 2 value-spread proxy features from the return quantiles;
  * ``save_sim_state`` / ``restore_sim_state`` — THE CRUX. Grushin's method needs N perturbed rollouts
    from an identical anchor. ``clone_state``/``restore_state`` only cover the ALE emulator; the SB3
    wrapper stack also holds the 4-frame stack (``VecFrameStack``) and the life counter
    (``EpisodicLifeEnv``). Both are snapshotted here. Correctness is gated by ``determinism_self_test``;
  * ``rollout_return`` — discounted raw-score return from the current (restored) state, with a leading
    uniform random-action burst as the perturbation;
  * collection workers (single-n for adaptivity, multi-n for Grushin) + parallel drivers.

Reward-only: Atari has no cost signal, so the criticality target is the return drop
ΔG = G_clean - G_perturbed (large drop = bad -> one-sided UPPER margin), the monotone/graded regime
where the Grushin n-sweep works cleanly.

Environment: requirements.txt.
"""

import os
import sys
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# ale-py registers the legacy "BeamRiderNoFrameskip-v4" id in gymnasium on import.
import ale_py  # noqa: F401
import gymnasium  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptivity_core import discount_cumsum, run_parallel_collection  # noqa: E402

REPO = "sb3/qrdqn-BeamRiderNoFrameskip-v4"
FILE = "qrdqn-BeamRiderNoFrameskip-v4.zip"
ENV_ID = "BeamRiderNoFrameskip-v4"
N_STACK = 4
EMBED_DIM = 512  # NatureCNN output width


# ---------------------------------------------------------------------------
# Environment + policy
# ---------------------------------------------------------------------------
def build_env(seed: int = 0, episodic_life: bool = False):
    """SB3-zoo BeamRider eval env: AtariWrapper (frameskip 4, grayscale, 84x84) + 4-frame stack +
    channels-first transpose, matching the policy's training preprocessing exactly.

    ``clip_reward=False`` so returns are on the RAW BeamRider score scale (the drop ΔG is then
    interpretable in points). ``terminal_on_life_loss`` defaults to False so ΔG measures the full
    (max_steps-capped) episode drop rather than a single life; set True to make a lost life terminal.
    """
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed,
                         wrapper_kwargs={"clip_reward": False,
                                         "terminal_on_life_loss": episodic_life})
    env = VecFrameStack(env, n_stack=N_STACK)   # (1, 84, 84, 4) channels-last buffer
    env = VecTransposeImage(env)                # (1, 4, 84, 84) for the CNN
    return env


def load_model(env, device: str = "cpu"):
    """Load the HF QRDQN checkpoint with the validated old-gym->gymnasium ``custom_objects`` cascade.

    ``torch.set_num_threads(1)`` keeps each loky worker single-threaded so the CNN forwards don't
    oversubscribe cores (loky x torch intra-op contention otherwise dominates).

    ``torch.backends.mkldnn.enabled = False`` routes Conv2d around oneDNN/MKL-DNN to the native kernel.
    oneDNN's CPU convolution derives its thread/blocking factors from the CPU topology the process
    sees; inside a restricted cgroup cpuset (e.g. an HTCondor slot) one of those factors can resolve to
    0 and the kernel divides by it -> SIGFPE (Floating point exception) in _conv_forward. MLP policies
    (SafetyGym) never hit this because Linear uses the GEMM/BLAS path, not oneDNN conv. This flag is a
    process-global torch setting, so setting it once per worker (here) covers every later forward.
    """
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    from sb3_contrib import QRDQN
    from huggingface_sb3 import load_from_hub
    ckpt = load_from_hub(REPO, FILE)
    model = QRDQN.load(
        ckpt, env=env, device=device,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                        "exploration_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0,
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                        "optimize_memory_usage": False},
    )
    model.policy.set_training_mode(False)
    return model, ckpt


# ---------------------------------------------------------------------------
# Anchor features: QRDQN NatureCNN embedding (+ optional value-spread proxy)
# ---------------------------------------------------------------------------
def embed(model, obs: np.ndarray, proxy_features: bool = False) -> np.ndarray:
    """Feature vector(s) for a batch of stacked+transposed obs [B, 4, 84, 84] (uint8) -> [B, D].

    Primary feature is the 512-d NatureCNN embedding (the QRDQN's own state representation); this is
    exactly what ``QuantileNetwork.forward`` computes before its quantile head. With
    ``proxy_features``, append [max_a Q(s,a) - min_a Q(s,a), mean_a std_quantiles(s,a)] — a richer
    analog of Grushin's max-min Q, straight from the distributional critic.
    """
    import torch
    qnet = model.policy.quantile_net
    obs_t = torch.as_tensor(np.asarray(obs), device=model.device)
    with torch.no_grad():
        feats = qnet.extract_features(obs_t, qnet.features_extractor)   # [B, 512]
        out = feats.cpu().numpy().astype(np.float32)
        if proxy_features:
            quantiles = qnet(obs_t)                                     # [B, n_quantiles, n_actions]
            q_sa = quantiles.mean(dim=1)                               # [B, n_actions] action values
            spread = (q_sa.max(dim=1).values - q_sa.min(dim=1).values)  # [B]
            qstd = quantiles.std(dim=1).mean(dim=1)                    # [B] mean per-action quantile std
            extra = torch.stack([spread, qstd], dim=1).cpu().numpy().astype(np.float32)
            out = np.concatenate([out, extra], axis=1)
    return out


# ---------------------------------------------------------------------------
# State save / restore  (THE CRUX)
# ---------------------------------------------------------------------------
def _handles(env):
    """Resolve the wrapper handles needed for snapshotting from the top VecTransposeImage env.

    Chain: VecTransposeImage(env) -> .venv VecFrameStack -> .venv DummyVecEnv -> .envs[0] gym
    AtariWrapper chain -> .unwrapped gymnasium AtariEnv (exposes clone_state/restore_state).
    EpisodicLifeEnv is present in the gym chain ONLY when terminal_on_life_loss=True.
    """
    from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
    vfs = env.venv                                   # VecFrameStack
    dummy = vfs.venv                                 # DummyVecEnv
    gym_env = dummy.envs[0]                           # AtariWrapper-wrapped gym env
    atari_env = gym_env.unwrapped                     # gymnasium AtariEnv
    epi = None
    w = gym_env
    while w is not None and w is not atari_env:
        if isinstance(w, EpisodicLifeEnv):
            epi = w
            break
        w = getattr(w, "env", None)
    return env, vfs, atari_env, epi


def save_sim_state(env) -> tuple:
    """Full snapshot: (ALE state incl. RNG, frame-stack buffer, EpisodicLife counters).

    ``include_rng=True`` makes replay bit-exact; combined with ``-v4``'s repeat_action_probability=0
    this satisfies Grushin's determinism assumption (Assumption I).
    """
    _, vfs, atari_env, epi = _handles(env)
    ale_state = atari_env.clone_state(include_rng=True)
    stack = vfs.stacked_obs.stacked_obs.copy()
    epi_state = (int(epi.lives), bool(epi.was_real_done)) if epi is not None else None
    return (ale_state, stack, epi_state)


def restore_sim_state(env, snap: tuple) -> np.ndarray:
    """Restore emulator + wrapper buffers; return the stacked+transposed obs the policy last saw."""
    top, vfs, atari_env, epi = _handles(env)
    ale_state, stack, epi_state = snap
    atari_env.restore_state(ale_state)
    vfs.stacked_obs.stacked_obs[...] = stack
    if epi is not None and epi_state is not None:
        epi.lives, epi.was_real_done = epi_state[0], epi_state[1]
    # VecTransposeImage is stateless -> just re-apply the transpose to the restored stack buffer.
    return top.transpose_observations(vfs.stacked_obs.stacked_obs)


# ---------------------------------------------------------------------------
# Rollout (random-action burst perturbation -> discounted raw-score return)
# ---------------------------------------------------------------------------
def rollout_return(env, model, start_obs: np.ndarray, burst_steps: int,
                   gamma: float, max_steps: int) -> float:
    """Discounted raw-score return from the CURRENT (already-restored) state.

    The first ``burst_steps`` actions are uniform random (Discrete(9)) = the perturbation; thereafter
    the deterministic policy. Break on done: DummyVecEnv auto-resets on done, so stepping past it would
    leak a fresh episode's reward into the return.
    """
    obs, rewards, steps, done = start_obs, [], 0, False
    while not done and steps < max_steps:
        if steps < burst_steps:
            action = np.array([env.action_space.sample()])
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        steps += 1
        done = bool(dones[0])
    return discount_cumsum(rewards, gamma)


# ---------------------------------------------------------------------------
# Collection workers
# ---------------------------------------------------------------------------
def _roll_to_anchor(env, model, random_prob, max_steps, seed_offset):
    """Phase-1: roll the nominal policy from a fresh reset; at the first random_prob trigger, snapshot
    the anchor and return (snapshot, current_obs). Returns None if the episode ended with no trigger."""
    obs = env.reset()
    steps, done = 0, False
    while not done and steps < max_steps:
        if np.random.random() < random_prob:
            return save_sim_state(env), obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        steps += 1
        done = bool(dones[0])
    return None


def _collect_worker(num_anchors, samples_per_state, random_prob, random_steps, gamma, max_steps,
                    episodic_life, device, proxy_features, anchor_max_steps, seed, counter=None):
    """Single perturbation-strength collection (adaptivity driver).

    Returns features [A, D], drops [A, N], g_clean [A].
    """
    np.random.seed(seed)
    env = build_env(seed=seed, episodic_life=episodic_life)
    env.action_space.seed(seed)
    model, ckpt = load_model(env, device=device)

    feat_buf: List[np.ndarray] = []
    drop_buf: List[np.ndarray] = []
    gclean_buf: List[float] = []
    collected, attempts = 0, 0
    max_attempts = num_anchors * 30 + 50
    while collected < num_anchors and attempts < max_attempts:
        attempts += 1
        got = _roll_to_anchor(env, model, random_prob, anchor_max_steps, attempts)
        if got is None:
            continue
        snap, anchor_obs = got
        feat = embed(model, anchor_obs, proxy_features)[0]

        restore_sim_state(env, snap)
        g_clean = rollout_return(env, model, anchor_obs, 0, gamma, max_steps)

        drops = np.empty(samples_per_state, dtype=np.float32)
        for i in range(samples_per_state):
            obs0 = restore_sim_state(env, snap)
            env.action_space.seed(seed * 100003 + collected * 911 + i)
            g_i = rollout_return(env, model, obs0, random_steps, gamma, max_steps)
            drops[i] = g_clean - g_i

        feat_buf.append(feat.astype(np.float32))
        drop_buf.append(drops)
        gclean_buf.append(g_clean)
        collected += 1
        if counter is not None:
            counter.value += 1

    D = (EMBED_DIM + 2) if proxy_features else EMBED_DIM
    if not feat_buf:
        return (np.empty((0, D), np.float32), np.empty((0, samples_per_state), np.float32),
                np.empty(0, np.float32))
    return (np.stack(feat_buf).astype(np.float32), np.stack(drop_buf).astype(np.float32),
            np.array(gclean_buf, dtype=np.float32))


def _collect_grushin_worker(num_anchors, samples_per_state, random_prob, n_list, gamma, max_steps,
                            episodic_life, device, proxy_features, anchor_max_steps, seed, counter=None):
    """Multi-n collection (Grushin driver): 1 clean rollout + N perturbed rollouts FOR EACH n.

    Returns features [A, D], drops [A, |n_list|, N], g_clean [A].
    """
    np.random.seed(seed)
    n_list = [int(n) for n in n_list]
    n_vals = len(n_list)
    env = build_env(seed=seed, episodic_life=episodic_life)
    env.action_space.seed(seed)
    model, ckpt = load_model(env, device=device)

    feat_buf: List[np.ndarray] = []
    drop_buf: List[np.ndarray] = []
    gclean_buf: List[float] = []
    collected, attempts = 0, 0
    max_attempts = num_anchors * 30 + 50
    while collected < num_anchors and attempts < max_attempts:
        attempts += 1
        got = _roll_to_anchor(env, model, random_prob, anchor_max_steps, attempts)
        if got is None:
            continue
        snap, anchor_obs = got
        feat = embed(model, anchor_obs, proxy_features)[0]

        restore_sim_state(env, snap)
        g_clean = rollout_return(env, model, anchor_obs, 0, gamma, max_steps)

        drops = np.empty((n_vals, samples_per_state), dtype=np.float32)
        for ni, n in enumerate(n_list):
            for i in range(samples_per_state):
                obs0 = restore_sim_state(env, snap)
                env.action_space.seed(seed * 100003 + collected * 911 + ni * 7919 + i)
                g_i = rollout_return(env, model, obs0, n, gamma, max_steps)
                drops[ni, i] = g_clean - g_i

        feat_buf.append(feat.astype(np.float32))
        drop_buf.append(drops)
        gclean_buf.append(g_clean)
        collected += 1
        if counter is not None:
            counter.value += 1

    D = (EMBED_DIM + 2) if proxy_features else EMBED_DIM
    if not feat_buf:
        return (np.empty((0, D), np.float32),
                np.empty((0, n_vals, samples_per_state), np.float32), np.empty(0, np.float32))
    return (np.stack(feat_buf).astype(np.float32), np.stack(drop_buf).astype(np.float32),
            np.array(gclean_buf, dtype=np.float32))


def _split_counts(num_anchors, num_workers):
    num_workers = max(1, min(num_workers or os.cpu_count() or 1, num_anchors))
    counts = [num_anchors // num_workers + (1 if i < num_anchors % num_workers else 0)
              for i in range(num_workers)]
    return [c for c in counts if c > 0]


def collect_parallel(num_anchors, samples_per_state, random_prob, random_steps, gamma, max_steps,
                     episodic_life, device, proxy_features, anchor_max_steps, num_workers, base_seed):
    counts = _split_counts(num_anchors, num_workers)
    print(f"Collecting {num_anchors} anchors x {samples_per_state} burst-samples | random_steps="
          f"{random_steps} | {len(counts)} workers | device={device} | proxy_features={proxy_features}")
    args = [(c, samples_per_state, random_prob, random_steps, gamma, max_steps, episodic_life,
             device, proxy_features, anchor_max_steps, base_seed + i) for i, c in enumerate(counts)]
    results = run_parallel_collection(_collect_worker, args, num_anchors, "Anchors", len(counts))
    return _assemble(results, num_workers=len(counts))


def collect_grushin_parallel(num_anchors, samples_per_state, random_prob, n_list, gamma, max_steps,
                             episodic_life, device, proxy_features, anchor_max_steps, num_workers, base_seed):
    counts = _split_counts(num_anchors, num_workers)
    print(f"Collecting {num_anchors} anchors x {samples_per_state} samples x {len(n_list)} n-values | "
          f"n_list={n_list} | {len(counts)} workers | device={device}")
    args = [(c, samples_per_state, random_prob, n_list, gamma, max_steps, episodic_life,
             device, proxy_features, anchor_max_steps, base_seed + i) for i, c in enumerate(counts)]
    results = run_parallel_collection(_collect_grushin_worker, args, num_anchors, "Anchors", len(counts))
    return _assemble(results, num_workers=len(counts))


def _assemble(results, num_workers):
    feat_parts = [r[0] for r in results if r[0].shape[0] > 0]
    drop_parts = [r[1] for r in results if r[1].shape[0] > 0]
    gcl_parts = [r[2] for r in results if r[2].shape[0] > 0]
    if not feat_parts:
        raise RuntimeError("No anchors collected; raise --random-prob or --num-anchors.")
    feats = np.concatenate(feat_parts, 0)
    drops = np.concatenate(drop_parts, 0)
    g_clean = np.concatenate(gcl_parts, 0)
    print(f"Collected {feats.shape[0]} anchors | feature_dim={feats.shape[1]} | drops shape {drops.shape}")
    print(f"  g_clean: mean {g_clean.mean():.2f} std {g_clean.std():.2f} "
          f"[{g_clean.min():.2f}, {g_clean.max():.2f}]")
    print(f"  drop   : mean {drops.mean():.2f} std {drops.std():.2f} "
          f"[{drops.min():.2f}, {drops.max():.2f}]")
    extra = {"model_path": REPO, "obs_dim": int(feats.shape[1]), "input_dim": int(feats.shape[1])}
    return feats, drops, g_clean, extra


# ---------------------------------------------------------------------------
# Determinism self-test (the hard gate before any collection)
# ---------------------------------------------------------------------------
def determinism_self_test(num_anchors: int = 20, random_prob: float = 0.02, gamma: float = 0.99,
                          max_steps: int = 150, anchor_max_steps: int = 800, burst_steps: int = 8,
                          device: str = "cpu", seed: int = 0) -> Dict[str, object]:
    """Validate that save/restore reproduces the deterministic dynamics bit-exactly.

    For each anchor: snapshot, then run TWO clean (no-burst) rollouts from the restored state and check
    their per-step reward sequences are identical. If the frame-stack buffer or life counter were not
    restored, the policy would see a different observation and the two returns would diverge.
    Also runs one burst rollout to confirm the perturbation actually changes the return.
    """
    np.random.seed(seed)
    env = build_env(seed=seed, episodic_life=False)
    env.action_space.seed(seed)
    model, _ = load_model(env, device=device)

    passes, diffs, clean_returns, burst_changed = 0, [], [], 0
    checked = 0
    attempts = 0
    while checked < num_anchors and attempts < num_anchors * 40 + 50:
        attempts += 1
        got = _roll_to_anchor(env, model, random_prob, anchor_max_steps, attempts)
        if got is None:
            continue
        snap, anchor_obs = got

        o1 = restore_sim_state(env, snap)
        r1 = _reward_seq(env, model, o1, 0, max_steps)
        o2 = restore_sim_state(env, snap)
        r2 = _reward_seq(env, model, o2, 0, max_steps)
        max_diff = float(np.abs(np.array(r1) - np.array(r2)).max()) if (len(r1) == len(r2) and r1) else float("inf")
        ok = (len(r1) == len(r2)) and (max_diff == 0.0)
        diffs.append(max_diff)
        clean_returns.append(discount_cumsum(r1, gamma))
        passes += int(ok)

        ob = restore_sim_state(env, snap)
        env.action_space.seed(seed * 13 + checked)
        gb = rollout_return(env, model, ob, burst_steps, gamma, max_steps)
        burst_changed += int(gb != clean_returns[-1])

        status = "PASS" if ok else "FAIL"
        print(f"  anchor {checked:3d} | steps r1={len(r1)} r2={len(r2)} | max reward diff {max_diff:.3g} "
              f"| clean G={clean_returns[-1]:.1f} burst G={gb:.1f}  [{status}]")
        checked += 1

    result = {"anchors": checked, "passes": passes, "all_pass": bool(passes == checked and checked > 0),
              "max_reward_diff": float(max(diffs)) if diffs else float("inf"),
              "burst_changed_frac": (burst_changed / checked) if checked else 0.0}
    print(f"\n=== determinism self-test: {passes}/{checked} anchors PASS | "
          f"max reward diff {result['max_reward_diff']:.3g} | "
          f"burst changed return in {result['burst_changed_frac']*100:.0f}% of anchors ===")
    return result


def _reward_seq(env, model, start_obs, burst_steps, max_steps):
    """Per-step raw reward list from the current state (used only by the self-test)."""
    obs, rewards, steps, done = start_obs, [], 0, False
    while not done and steps < max_steps:
        if steps < burst_steps:
            action = np.array([env.action_space.sample()])
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        steps += 1
        done = bool(dones[0])
    return rewards
