# -*- coding: utf-8 -*-
"""Shared helpers for the group-conditional analysis, and the export stage.

Stage 'export': reproduce each run's train/calibration/test split and predict
q_hat_{1-a}(s,n) on the calibration and test anchors with that run's SAVED per-n models.
Run it with the python environment that produced the run, so its models unpickle:
    python grouping/grouped_sweep.py export spg1                         (safety environment)
    python grouping/grouped_sweep.py export cartpole pendulum beamrider   (main environment)
fixed_groups.py and fixed_agree.py then read these exports.
"""
import math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "grouping")
ALPHA, SEED, MIN_CERT = 0.1, 0, 0.05

RUNS = {
    "spg1":      dict(raw="runs/spg1/sweep/raw.npz", key="drops_r",
                      pkl="runs/spg1/sweep/grushin_net.pkl",
                      js="runs/spg1/sweep/results.json", G=3, label="SafetyPointGoal1"),
    "cartpole":  dict(raw="runs/cartpole/sweep/raw.npz", key="drops_c",
                      pkl="runs/cartpole/sweep/grushin_net.pkl",
                      js="runs/cartpole/sweep/results.json", G=5, label="CartPole"),
    "pendulum":  dict(raw="runs/pendulum/sweep/raw.npz", key="drops_r",
                      pkl="runs/pendulum/sweep/grushin_net.pkl",
                      js="runs/pendulum/sweep/results.json", G=5, label="Pendulum"),
    "beamrider": dict(raw="runs/beamrider/sweep/raw.npz", key="drops_r",
                      pkl="runs/beamrider/sweep/grushin_net.pkl",
                      js="runs/beamrider/sweep/results.json", G=6, label="BeamRider"),
}
P = lambda p: os.path.join(ROOT, p)


def split(A):
    perm = np.random.default_rng(SEED).permutation(A)
    n_te, n_cal = int(A * 0.2), int(A * 0.3)
    return perm[:n_te], perm[n_te:n_te + n_cal], perm[n_te + n_cal:]


def load(name):
    r = RUNS[name]
    d = np.load(P(r["raw"]), allow_pickle=True)
    X = d["features"].astype(np.float64)
    D = d[r["key"]].astype(np.float64)
    return X, D, [int(n) for n in d["n_list"]]


def export(name):
    import joblib
    X, D, n_list = load(name)
    te, cal, tr = split(len(X))
    ck = joblib.load(P(RUNS[name]["pkl"]))
    base_cal = np.stack([ck["models"][n].predict(X[cal]) for n in n_list])
    base_te = np.stack([ck["models"][n].predict(X[te]) for n in n_list])
    q_one = np.array([ck["q_one"][n] for n in n_list])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "export_%s.npz" % name), base_cal=base_cal, base_te=base_te,
             q_one=q_one)
    print("exported", name)


def conformal_offset(scores, alpha):
    scores = np.asarray(scores, float)
    k = min(max(math.ceil((len(scores) + 1) * (1 - alpha)) - 1, 0), len(scores) - 1)
    return float(np.partition(scores, k)[k])


def one_per_anchor(S, seed):
    idx = np.random.default_rng(seed).integers(0, S.shape[1], size=S.shape[0])
    return S[np.arange(S.shape[0]), idx]


def s_star(M, n_list, zeta):
    cum = np.logical_and.accumulate(M <= zeta, axis=0)
    k = cum.sum(0)
    n_arr = np.asarray(n_list)
    return np.where(k >= 1, n_arr[np.clip(k - 1, 0, len(n_list) - 1)], 0)


if __name__ == "__main__":
    stage, names = sys.argv[1], sys.argv[2:]
    if stage != "export":
        raise SystemExit("usage: python grouping/grouped_sweep.py export <run> [<run> ...]")
    for nm in names:
        export(nm)
