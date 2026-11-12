"""Sim 2 - Robustness, on the 50-confounder data (4 strong nonlinear + 46 weak).

Part A (the curve): a correct and a wrong version of the outcome model and of
the dose model, combined four ways; three estimators from the same pieces:
  S-learner      : outcome model averaged over X (needs the outcome model right)
  weighting only : outcome reweighted by varpi/pi (needs the dose model right)
  DR             : pseudo-outcome (needs one of the two right)
"Wrong" outcome model: correct in the dose (A, A^2) but linear in the raw
covariates, so it mishandles the confounding, not the shape.
"Wrong" dose model: linear mean in the raw covariates, constant normal error.

Part B (in practice): boosted outcome model, with boosted density weights or
balancing weights; S-learner from the same boosted model.

Part C (DoubleML, linear true effect so theta is defined): partialling-out vs
IV-type score with right/wrong nuisance models.

Usage: python experiments/sim2_double_robustness.py REPS   (resumable cache)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcurves import GPS, BalancingWeights, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa: E402
from drcurves.nuisance import Nuisances  # noqa: E402
from drcurves.simulate import sim_high_dim  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
P, N = 50, 3000


# ---------------------------------------------------------------- model specs
def f_conf(X):         # the true confounding features (strong nonlinear + weak linear)
    return np.column_stack([np.sin(1.5 * X[:, 0]), X[:, 1] ** 2, X[:, 2] > 0, X[:, 3], X[:, 4:]])


def f_out_right(Z):    # Z = [X, A]
    a = Z[:, -1]
    return np.column_stack([f_conf(Z[:, :-1]), a, a ** 2])


def f_out_wrong(Z):    # right in the dose, linear in raw covariates
    a = Z[:, -1]
    return np.column_stack([Z[:, :-1], a, a ** 2])


def f_scale(X):
    return 1 / (1 + np.exp(-2 * X[:, [0]]))


def lin(fn):
    return make_pipeline(FunctionTransformer(fn), LinearRegression())


hgb = lambda: HGB(max_iter=300, learning_rate=0.05, random_state=0)  # noqa: E731
OUTCOME = {"right": lin(f_out_right), "wrong": lin(f_out_wrong)}
DOSE = {"right": GPS(mean_model=lin(f_conf), scale_model=lin(f_scale), residual="gaussian"),
        "wrong": GPS(mean_model=LinearRegression(), heteroscedastic=False, residual="gaussian")}
SCEN = [("right", "right"), ("wrong", "right"), ("right", "wrong"), ("wrong", "wrong")]

_, A_big, _, f = sim_high_dim(n=200_000, p=P, seed=0)
GRID = np.linspace(*np.quantile(A_big, [0.05, 0.95]), 50)
TRUTH = f(GRID)


def centred(v):
    return np.asarray(v) - np.mean(v)


def curve(A, nuis, Y, kind):
    est, _ = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nuis, Y, kind)).curve(GRID)
    return centred(est).tolist()


for rep in range(REPS):
    path = CACHE / f"sim2hd_{rep:03d}.json"
    if path.exists():
        continue
    X, A, Y, _ = sim_high_dim(n=N, p=P, seed=100 + rep)
    out = {}
    # Part A
    nu = {k: cross_fit_nuisances(X, A, Y, OUTCOME[k], None, random_state=rep) for k in OUTCOME}
    nd = {k: cross_fit_nuisances(X, A, Y, None, DOSE[k], random_state=rep) for k in DOSE}
    for om, dm in SCEN:
        nn = Nuisances(mu=nu[om].mu, m=nu[om].m, pi=nd[dm].pi, varpi=nd[dm].varpi)
        for kind in ["plugin", "ipw", "dr"]:
            out[f"A|{om}|{dm}|{kind}"] = curve(A, nn, Y, kind)
    # Part B
    nb = cross_fit_nuisances(X, A, Y, hgb(), GPS(random_state=rep), random_state=rep)
    nbal = cross_fit_nuisances(X, A, Y, None, BalancingWeights(), random_state=rep)
    out["B|slearner"] = curve(A, nb, Y, "plugin")
    out["B|dr_density"] = curve(A, nb, Y, "dr")
    out["B|dr_balancing"] = curve(A, Nuisances(mu=nb.mu, m=nb.m, pi=nbal.pi, varpi=nbal.varpi), Y, "dr")
    # Part C
    Xl, Al, Yl, _ = sim_high_dim(n=N, p=P, seed=500 + rep, effect="linear")
    data = DoubleMLData.from_arrays(Xl, Yl, Al)
    pick = {"right": lin(f_conf), "wrong": LinearRegression()}
    for om, dm in SCEN:
        for score in ["partialling out", "IV-type"]:
            kw = dict(ml_g=pick[om]) if score == "IV-type" else {}
            np.random.seed(rep)
            plr = DoubleMLPLR(data, pick[om], pick[dm], score=score, n_folds=5, **kw).fit()
            out[f"C|{score}|{om}|{dm}"] = [float(plr.coef[0]), float(plr.se[0])]
    path.write_text(json.dumps(out))
    print(f"rep {rep} done", flush=True)

# ----------------------------------------------------------------- summarise
files = sorted(CACHE.glob("sim2hd_*.json"))
D = [json.loads(p.read_text()) for p in files]
t = centred(TRUTH)
rows, curves = [], {}
for key in [k for k in D[0] if not k.startswith("C|")]:
    cs = np.array([d[key] for d in D])
    curves[key] = cs
    rows.append(dict(key=key, reps=len(cs), abs_bias=np.mean(np.abs(cs.mean(0) - t)),
                     sd=np.mean(cs.std(0)), rmse=np.sqrt(np.mean((cs - t) ** 2))))
summ = pd.DataFrame(rows).round(3)
summ.to_csv(ROOT / "results" / "sim2_curve_summary.csv", index=False)
np.savez(ROOT / "results" / "sim2_curves.npz", grid=GRID, truth=TRUTH,
         **{k: v + TRUTH.mean() for k, v in curves.items()})
prow = []
for key in [k for k in D[0] if k.startswith("C|")]:
    _, score, om, dm = key.split("|")
    th = np.array([d[key][0] for d in D])
    se = np.array([d[key][1] for d in D])
    prow.append(dict(score=score, outcome=om, treatment_model=dm, bias=th.mean() - 1.0, sd=th.std(),
                     coverage=np.mean(np.abs(th - 1.0) <= 1.96 * se), reps=len(th)))
psum = pd.DataFrame(prow).round(3)
psum.to_csv(ROOT / "results" / "sim2_plr_summary.csv", index=False)
print(summ.to_string(index=False))
print(psum.to_string(index=False))
