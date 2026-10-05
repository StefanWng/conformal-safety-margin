# -*- coding: utf-8 -*-
"""Group-conditional safety margins with ONE risk score per state, shared across n.

Score target  pbar(s) = mean_n P(Delta(s,n) > tau_n),  tau_n = pooled training upper
decile at length n. One HGB fitted on the training split predicts it. Groups are
G quantiles of the predicted score on the calibration split and are the same at
every n; only the offsets Q_g(n) are calibrated per n.

The quantile models (and the score model) are fitted on the training split only,
so the calibration and test anchors can be re-partitioned freely. Split 0 is the
paper's own partition; splits 1..R-1 re-randomize it, with a fresh one-per-anchor
calibration draw each time. Single and group offsets always share that draw.
"""
import io, json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import grouped_sweep as g

ALPHA, R, MIN_CERT = 0.1, 200, 0.05
DG = {3: 1.69, 5: 2.33, 6: 2.53}
OUT = g.OUT


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                         validation_fraction=0.15, n_iter_no_change=30,
                                         random_state=0)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def run(name, score="all"):
    r = g.RUNS[name]; G = r["G"]
    X, D, n_list = g.load(name); nv, N = len(n_list), D.shape[2]
    te, cal, tr = g.split(len(X))
    ex = np.load(os.path.join(OUT, "export_%s.npz" % name))
    pool = np.concatenate([cal, te])
    base = np.concatenate([ex["base_cal"], ex["base_te"]], axis=1)        # [nv, P]
    P, n_cal, n_te = len(pool), len(cal), len(te)
    taus = [float(np.quantile(D[tr, ni], 1 - ALPHA)) for ni in range(nv)]
    use = list(range(nv)) if score == "all" else [n_list.index(8)]
    target = lambda idx, cols=slice(None): np.mean(
        [(D[idx, ni][:, cols] > taus[ni]).mean(1) for ni in use], axis=0)

    # ---- the score, and whether it passes the paper's own measurability check
    h = hgb().fit(X[tr], target(tr))
    sc = h.predict(X[pool])
    perm = np.random.default_rng(1).permutation(N)
    allA = np.arange(len(X))
    rh = corr(target(allA, perm[:N // 2]), target(allA, perm[N // 2:]))
    rho = 2 * rh / (1 + rh)
    recov = corr(sc[n_cal:], target(pool[n_cal:]))

    js = json.load(io.open(g.P(r["js"]), encoding="utf-8"))
    tols = [t["tolerance"] for t in js["tolerances"] if (1 - t["frac_margin_zero"]) >= MIN_CERT]
    nz = len(tols)
    # per-anchor exceedance of each tolerance, for the validity check  [nz, P, nv]
    Eexc = np.stack([(D[pool] > z).mean(2) for z in tols])

    A = lambda *s: np.full(s, np.nan)
    res = dict(cov1=A(R, nv), covG=A(R, nv), gcov1=A(R, nv, G), gcovG=A(R, nv, G),
               w1=A(R, nv), wG=A(R, nv), cert1=A(R, nz), certG=A(R, nz), m1=A(R, nz), mG=A(R, nz),
               mc1=A(R, nz), mcG=A(R, nz), x1=A(R, nz), xG=A(R, nz), gs1=A(R, nz, G),
               gsG=A(R, nz, G), gain=A(R, nz), lose=A(R, nz), qg=A(R, nv, G), q1=A(R, nv))
    for rr in range(R):
        if rr == 0:
            ic, it = np.arange(n_cal), np.arange(n_cal, P)
        else:
            pp = np.random.default_rng(1000 + rr).permutation(P)
            ic, it = pp[:n_cal], pp[n_cal:]
        edges = np.unique(np.quantile(sc[ic], np.linspace(0, 1, G + 1)[1:-1]))
        gc, gt = np.digitize(sc[ic], edges), np.digitize(sc[it], edges)
        M1, MG = np.empty((nv, len(it))), np.empty((nv, len(it)))
        for ni in range(nv):
            y = g.one_per_anchor(D[pool[ic], ni], rr)
            s = y - base[ni, ic]
            q1 = g.conformal_offset(s, ALPHA)
            qg = np.array([g.conformal_offset(s[gc == k], ALPHA) if (gc == k).any() else q1
                           for k in range(G)])
            M1[ni], MG[ni] = base[ni, it] + q1, base[ni, it] + qg[gt]
            res["q1"][rr, ni], res["qg"][rr, ni] = q1, qg
            Dt = D[pool[it], ni]
            c1a, cGa = (Dt <= M1[ni][:, None]).mean(1), (Dt <= MG[ni][:, None]).mean(1)
            res["cov1"][rr, ni], res["covG"][rr, ni] = c1a.mean(), cGa.mean()
            for k in range(G):
                if (gt == k).any():
                    res["gcov1"][rr, ni, k] = c1a[gt == k].mean()
                    res["gcovG"][rr, ni, k] = cGa[gt == k].mean()
            res["w1"][rr, ni], res["wG"][rr, ni] = M1[ni].mean(), MG[ni].mean()
        for zi, z in enumerate(tols):
            for tag, M in (("1", M1), ("G", MG)):
                ss = g.s_star(M, n_list, z)
                worst = 0.0
                for ni, n in enumerate(n_list):
                    c = ss >= n
                    if c.any():
                        worst = max(worst, float(Eexc[zi, it][c, ni].mean()))
                res["cert" + tag][rr, zi] = (ss > 0).mean()
                res["m" + tag][rr, zi] = ss.mean()
                res["mc" + tag][rr, zi] = ss[ss > 0].mean() if (ss > 0).any() else np.nan
                res["x" + tag][rr, zi] = worst
                for k in range(G):
                    if (gt == k).any():
                        res["gs" + tag][rr, zi, k] = ss[gt == k].mean()
                if tag == "1":
                    s1 = ss
                else:
                    res["gain"][rr, zi] = (ss > s1).mean()
                    res["lose"][rr, zi] = (ss < s1).mean()
    out = dict(name=name, label=r["label"], G=G, score=score, n_list=n_list, tols=tols,
               rho=rho, recovery=recov, n_te=n_te, n_cal=n_cal,
               expected_spread=DG[G] * np.sqrt(ALPHA * (1 - ALPHA) / (n_te / G)))
    out.update({k: v.tolist() for k, v in res.items()})
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    score = sys.argv[1]
    for nm in sys.argv[2:]:
        o = run(nm, score)
        with io.open(os.path.join(OUT, "fixed_%s_%s.json" % (score, nm)), "w", encoding="utf-8") as f:
            json.dump(o, f)
        print("done", nm, score, "rho %.3f recovery %.3f" % (o["rho"], o["recovery"]))
