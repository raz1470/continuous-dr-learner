"""This package with stacked nuisance models (spline-ridge + boosting), on the same datasets as
compare_causal_curve.py / compare_npcausal.py. Tests whether an npcausal-style ensemble closes the
accuracy gap. Usage: python experiments/compare_stacked.py REPS [BUDGET_SECONDS]   (resumable)"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drcurves import GPS, BalancingWeights, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa
from drcurves.nuisance import Nuisances  # noqa
from drcurves.simulate import sim_nonlinear, sim_high_dim  # noqa
warnings.filterwarnings("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
OUT = ROOT / "results" / "cache"
T0 = time.time()


def stack():
    """Additive spline ridge (a GAM-like learner) + boosting, combined with non-negative weights."""
    gam = make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return StackingRegressor([("gam", gam), ("hgb", HGB(max_iter=200, learning_rate=0.05, random_state=0))],
                             final_estimator=LinearRegression(positive=True), cv=3)


def score(est, se, t):
    return dict(rmse=float(np.sqrt(np.mean((est - t) ** 2))), cover=float(np.mean(np.abs(est - t) <= 1.96 * se)),
                width=float(np.mean(2 * 1.96 * se)))


for rep in range(REPS):
    for setting in ["p5", "p50"]:
        fp = OUT / f"st_{setting}_{rep:03d}.json"
        if fp.exists():
            continue
        if time.time() - T0 > BUDGET:
            print("budget reached", flush=True); sys.exit(0)
        t0 = time.time()
        if setting == "p5":
            X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u", seed=900 + rep)
        else:
            X, A, Y, f = sim_high_dim(n=3000, p=50, seed=900 + rep)
        X = np.asarray(X); g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); t = f(g) - f(g).mean()
        out = []
        nu = cross_fit_nuisances(X, A, Y, stack(), GPS(mean_model=stack(), scale_model=stack(), random_state=rep),
                                 random_state=rep)
        est, se = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nu, Y, "dr")).curve(g, centre=True)
        out.append(dict(setting=setting, rep=rep, method="DR, stacked, density weights", **score(est, se, t)))
        nw = cross_fit_nuisances(X, A, Y, None, BalancingWeights(), random_state=rep)
        nn = Nuisances(mu=nu.mu, m=nu.m, pi=nw.pi, varpi=nw.varpi)
        est, se = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nn, Y, "dr")).curve(g, centre=True)
        out.append(dict(setting=setting, rep=rep, method="DR, stacked, balancing weights", **score(est, se, t)))
        fp.write_text(json.dumps(out))
        print(setting, rep, f"{time.time()-t0:.0f}s", {o["method"][12:]: round(o["rmse"], 3) for o in out}, flush=True)
