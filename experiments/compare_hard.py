"""A harder data-generating process than Sims 1-2: interactions, thresholds, heavy-tailed noise and
weaker overlap, which favour trees over additive splines. Compares the stacked default with
boosting alone, and with npcausal (R). Usage: python experiments/compare_hard.py REPS [BUDGET] [--no-r]"""
import json, subprocess, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drcurves import GPS, BalancingWeights, HierarchicalDoseResponse, cross_fit_nuisances, default_model, pseudo_outcomes  # noqa
from drcurves.nuisance import Nuisances  # noqa
from drcurves.simulate import CURVES  # noqa
warnings.filterwarnings("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 1e9
USE_R = "--no-r" not in sys.argv
OUT = ROOT / "results" / "cache"; T0 = time.time()
f = CURVES["inverted_u_centred"]


def sim_hard(n=3000, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    A = (0.7 * (X[:, 0] > 0) + 0.5 * X[:, 1] * X[:, 2] + 0.4 * np.abs(X[:, 3]) - 0.5
         + 0.5 * rng.standard_t(4, size=n))
    Y = (f(A) + 1.2 * (X[:, 0] > 0.5) + 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.maximum(X[:, 4], 0)
         + 0.5 * X[:, 3] * (X[:, 0] > 0) + 0.8 * rng.standard_t(3, size=n))
    return X, A, Y


def boosted_gps(rep):
    return GPS(mean_model=HGB(max_iter=200, learning_rate=0.05, random_state=rep),
               scale_model=HGB(max_iter=100, learning_rate=0.05, min_samples_leaf=100, random_state=rep),
               random_state=rep)


def score(est, se, t):
    return dict(rmse=float(np.sqrt(np.mean((est - t) ** 2))), cover=float(np.mean(np.abs(est - t) <= 1.96 * se)),
                width=float(np.mean(2 * 1.96 * se)))


if __name__ == "__main__":
    for rep in range(REPS):
        fp = OUT / f"hard_{rep:03d}.json"
        res = json.loads(fp.read_text()) if fp.exists() else {}
        need_r = USE_R and "npcausal (R)" not in res
        if "stacked, density" in res and not need_r:
            continue
        if time.time() - T0 > BUDGET:
            print("budget reached", flush=True); sys.exit(0)
        t0 = time.time()
        X, A, Y = sim_hard(seed=3000 + rep)
        g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); t = f(g) - f(g).mean()
        if "stacked, density" not in res:
            nbal = cross_fit_nuisances(X, A, Y, None, BalancingWeights(), random_state=rep)
            for name, om, gps in [("stacked", default_model(), GPS(random_state=rep)),
                                  ("boosted", HGB(max_iter=300, learning_rate=0.05, random_state=0), boosted_gps(rep))]:
                nu = cross_fit_nuisances(X, A, Y, om, gps, random_state=rep)
                for wname, nn in [("density", nu), ("balancing", Nuisances(mu=nu.mu, m=nu.m, pi=nbal.pi, varpi=nbal.varpi))]:
                    est, se = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nn, Y, "dr")).curve(g, centre=True)
                    res[f"{name}, {wname}"] = score(est, se, t)
                est, se = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nu, Y, "plugin")).curve(g, centre=True)
                res[f"{name}, S-learner"] = score(est, se, t)
        if need_r:
            with tempfile.TemporaryDirectory() as td:
                df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]); df["A"] = A; df["Y"] = Y
                df.to_csv(f"{td}/d.csv", index=False)
                r = subprocess.run(["Rscript", str(ROOT / "experiments/r/run_ctseff.R"), f"{td}/d.csv", f"{td}/o.csv",
                                    str(g[0]), str(g[-1]), "80", "0"], capture_output=True, text=True)
                if r.returncode == 0:
                    o = pd.read_csv(f"{td}/o.csv"); av = o["a.vals"].values
                    est, lo, hi = (np.interp(g, av, o[c]) for c in ("est", "ci.ll", "ci.ul"))
                    c = est.mean()
                    res["npcausal (R)"] = dict(rmse=float(np.sqrt(np.mean((est - c - t) ** 2))),
                                               cover=float(np.mean((t >= lo - c) & (t <= hi - c))), width=float(np.mean(hi - lo)))
                else:
                    print("npcausal failed", r.stderr[-200:], flush=True)
        fp.write_text(json.dumps(res))
        print(rep, f"{time.time()-t0:.0f}s", {k: round(v["rmse"], 3) for k, v in res.items()}, flush=True)
