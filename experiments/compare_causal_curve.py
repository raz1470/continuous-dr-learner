"""Compare causal-curve's TMLE_Regressor with this repo's DR curve on Sim 1 (5 confounders)
and Sim 2 (50 confounders). Scored on the centred curve over the middle 90% of doses:
RMSE and pointwise 95% coverage. Usage: python experiments/compare_causal_curve.py REPS"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcurves import fit_dose_response, BalancingWeights, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa
from drcurves.nuisance import Nuisances  # noqa
from drcurves.simulate import sim_nonlinear, sim_high_dim  # noqa
from causal_curve import TMLE_Regressor  # noqa
warnings.filterwarnings("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
_T0 = time.time()
OUT = Path(__file__).resolve().parents[1] / "results" / "cache"
hgb = lambda: HGB(max_iter=300, learning_rate=0.05, random_state=0)  # noqa

def score(g, est, lo, hi, truth):
    c = lambda v: np.asarray(v) - np.mean(est)  # noqa
    t = truth - truth.mean()
    e, l, h = est - est.mean(), np.asarray(lo) - est.mean(), np.asarray(hi) - est.mean()
    return dict(rmse=float(np.sqrt(np.mean((e - t) ** 2))), cover=float(np.mean((t >= l) & (t <= h))),
                width=float(np.mean(h - l)))

rows = []
for rep in range(REPS):
    for setting in ["p5", "p50"]:
        if time.time() - _T0 > BUDGET and not (OUT / f"{'cc' if 'causal_curve' in __file__ else 'np'}_{setting}_{rep:03d}.json").exists():
            print('budget reached', flush=True); sys.exit(0)
        fp = OUT / f"cc_{setting}_{rep:03d}.json"
        if fp.exists():
            rows += json.loads(fp.read_text()); continue
        t0 = time.time()
        if setting == "p5":
            X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u", seed=900 + rep)
        else:
            X, A, Y, f = sim_high_dim(n=3000, p=50, seed=900 + rep)
        X = np.asarray(X); g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); truth = f(g)
        out = []
        # causal-curve
        cols = [f"x{i}" for i in range(X.shape[1])]
        tm = TMLE_Regressor(random_seed=0)
        tm.fit(T=pd.Series(A, name="T"), X=pd.DataFrame(X, columns=cols), y=pd.Series(Y, name="Y"))
        pe = tm.point_estimate(g); iv = np.asarray(tm.point_estimate_interval(g, 0.95))
        out.append(dict(setting=setting, rep=rep, method="causal-curve TMLE", **score(g, np.ravel(pe), iv[:, 0], iv[:, 1], truth)))
        # this repo, density weights (default)
        m = fit_dose_response(X, A, Y, outcome_model=hgb())
        est, se = m.curve(g)
        out.append(dict(setting=setting, rep=rep, method="DR, density weights", **score(g, est, est - 1.96 * se, est + 1.96 * se, truth)))
        # this repo, balancing weights
        nu = cross_fit_nuisances(X, A, Y, hgb(), None, random_state=rep)
        nw = cross_fit_nuisances(X, A, Y, None, BalancingWeights(), random_state=rep)
        hm = HierarchicalDoseResponse().fit(A, pseudo_outcomes(Nuisances(mu=nu.mu, m=nu.m, pi=nw.pi, varpi=nw.varpi), Y, "dr"))
        est, se = hm.curve(g)
        out.append(dict(setting=setting, rep=rep, method="DR, balancing weights", **score(g, est, est - 1.96 * se, est + 1.96 * se, truth)))
        fp.write_text(json.dumps(out)); rows += out
        print(setting, rep, f"{time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows).groupby(["setting", "method"])[["rmse", "cover", "width"]].mean().round(3)
res.to_csv(OUT.parent / "compare_causal_curve.csv")
print(res.to_string())
