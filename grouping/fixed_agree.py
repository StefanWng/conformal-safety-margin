# -*- coding: utf-8 -*-
"""Does the group margin predict each state's safety margin better than the single one?

Reference per test state: s_emp(s, zeta), the s* rule applied to the state's OWN
empirical upper decile q_emp(s, n) over its N draws. A method over-certifies a
state when s* > s_emp, i.e. it promises more steps than the state's draws support
(at some n <= s*, more than a tenth of that state's own draws exceed zeta). It
under-certifies when s* < s_emp. Same fixed groups, splits and draws as
fixed_groups.py.
"""
import io, json, os, sys
import numpy as np
from scipy.stats import spearmanr
import grouped_sweep as g
from fixed_groups import hgb, ALPHA, R, MIN_CERT, OUT

NAMES = tuple(sys.argv[1:]) or ("spg1", "cartpole", "pendulum", "beamrider")


def run(name):
    r = g.RUNS[name]; G = r["G"]
    X, D, n_list = g.load(name); nv = len(n_list)
    te, cal, tr = g.split(len(X))
    ex = np.load(os.path.join(OUT, "export_%s.npz" % name))
    pool = np.concatenate([cal, te])
    base = np.concatenate([ex["base_cal"], ex["base_te"]], axis=1)
    P, n_cal = len(pool), len(cal)
    taus = [float(np.quantile(D[tr, ni], 1 - ALPHA)) for ni in range(nv)]
    ytr = np.mean([(D[tr, ni] > taus[ni]).mean(1) for ni in range(nv)], axis=0)
    sc = hgb().fit(X[tr], ytr).predict(X[pool])
    js = json.load(io.open(g.P(r["js"]), encoding="utf-8"))
    tols = [t["tolerance"] for t in js["tolerances"] if (1 - t["frac_margin_zero"]) >= MIN_CERT]
    q_emp = np.quantile(D[pool], 1 - ALPHA, axis=2).T              # [nv, P]
    s_emp = np.stack([g.s_star(q_emp, n_list, z) for z in tols])    # [nz, P]

    nz = len(tols)
    out = {k: np.full((R, nz), np.nan) for k in
           ("over1", "overG", "under1", "underG", "agree1", "agreeG", "rho1", "rhoG")}
    out["overg1"] = np.full((R, nz, G), np.nan); out["overgG"] = np.full((R, nz, G), np.nan)
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
            s = g.one_per_anchor(D[pool[ic], ni], rr) - base[ni, ic]
            q1 = g.conformal_offset(s, ALPHA)
            qg = np.array([g.conformal_offset(s[gc == k], ALPHA) if (gc == k).any() else q1
                           for k in range(G)])
            M1[ni], MG[ni] = base[ni, it] + q1, base[ni, it] + qg[gt]
        for zi, z in enumerate(tols):
            ref = s_emp[zi, it]
            for tag, M in (("1", M1), ("G", MG)):
                ss = g.s_star(M, n_list, z)
                out["over" + tag][rr, zi] = (ss > ref).mean()
                out["under" + tag][rr, zi] = (ss < ref).mean()
                out["agree" + tag][rr, zi] = (ss == ref).mean()
                if ss.std() > 0 and ref.std() > 0:
                    out["rho" + tag][rr, zi] = spearmanr(ss, ref)[0]
                for k in range(G):
                    m = gt == k
                    if m.any():
                        out["overg" + tag][rr, zi, k] = (ss[m] > ref[m]).mean()
    return dict(name=name, label=r["label"], G=G, tols=tols, **{k: v.tolist() for k, v in out.items()})


if __name__ == "__main__":
    for nm in NAMES:
        o = run(nm)
        with io.open(os.path.join(OUT, "agree_%s.json" % nm), "w", encoding="utf-8") as f:
            json.dump(o, f)
        a = lambda k: np.array(o[k], float)
        d_over = a("overG") - a("over1")                                   # [R, nz]
        print("=" * 92)
        print("%s  (means over %d splits)" % (o["label"], R))
        print("  %-8s | %-13s | %-13s | %-13s | %-13s | %s" % ("zeta", "over-cert", "under-cert",
              "exact agree", "Spearman", "splits where group over-certifies less"))
        for zi, z in enumerate(o["tols"]):
            print("  %-8.3g | %.3f  %.3f  | %.3f  %.3f  | %.3f  %.3f  | %.3f  %.3f  | %.2f" %
                  (z, a("over1")[:, zi].mean(), a("overG")[:, zi].mean(), a("under1")[:, zi].mean(),
                   a("underG")[:, zi].mean(), a("agree1")[:, zi].mean(), a("agreeG")[:, zi].mean(),
                   np.nanmean(a("rho1")[:, zi]), np.nanmean(a("rhoG")[:, zi]),
                   (d_over[:, zi] < 0).mean()))
        print("  ALL      | %.3f  %.3f  | %.3f  %.3f  | %.3f  %.3f  | %.3f  %.3f  | %.2f (paired mean diff %+.4f, 95%% of splits in [%+.4f, %+.4f])" %
              (a("over1").mean(), a("overG").mean(), a("under1").mean(), a("underG").mean(),
               a("agree1").mean(), a("agreeG").mean(), np.nanmean(a("rho1")), np.nanmean(a("rhoG")),
               (d_over.mean(1) < 0).mean(), d_over.mean(), *np.percentile(d_over.mean(1), [2.5, 97.5])))
        print("  over-certification by risk group (low -> high), mean over tolerances and splits")
        print("     single: " + " ".join("%.3f" % v for v in np.nanmean(a("overg1"), (0, 1))))
        print("     group : " + " ".join("%.3f" % v for v in np.nanmean(a("overgG"), (0, 1))))
