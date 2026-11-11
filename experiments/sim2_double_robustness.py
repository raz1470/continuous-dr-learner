"""Sim 2 - Double robustness for the *curve*.

Part A (curve): fit a right (boosted) and a wrong (linear) version of each
nuisance model, then combine them four ways. Plug-in needs the outcome model,
IPW needs the GPS, the DR pseudo-outcome needs only one of the two.

Part B (DoubleML PLR, linear true effect so theta is well defined): the default
'partialling out' score needs the treatment model E[D|X] right; it is not
protected by a correct outcome model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _style import C, INK, plt  # noqa: E402
from drcurves import GPS, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa: E402
from drcurves.nuisance import Nuisances  # noqa: E402
from drcurves.simulate import sim_double_robust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
hgb = lambda: HGB(max_iter=300, learning_rate=0.05, random_state=0)  # noqa: E731


# "right" = correctly specified parametric model (the textbook test of double
# robustness); "wrong" = linear in the raw covariates; "boosted" = what you'd
# actually use in practice (flexible, approximately right).
def f_conf(X):     # confounder features that enter both E[A|X] and E[Y|X]
    return np.column_stack([np.sin(1.5 * X[:, 0]), X[:, 1] ** 2, X[:, 2] > 0, X[:, 3]])


def f_outcome(Z):  # Z = [X, A]
    a = Z[:, -1]
    return np.column_stack([f_conf(Z[:, :-1]), a, a ** 2])


def f_scale(X):
    return (1 / (1 + np.exp(-2 * X[:, [0]])))


def lin(fn):
    return make_pipeline(FunctionTransformer(fn), LinearRegression())


OUTCOME = {"right": lin(f_outcome), "wrong": LinearRegression(), "boosted": hgb()}
GPSM = {"right": GPS(mean_model=lin(f_conf), scale_model=lin(f_scale), residual="gaussian"),
        "wrong": GPS(mean_model=LinearRegression(), heteroscedastic=False, residual="gaussian"),
        "boosted": GPS()}
SCEN = [("right", "right"), ("wrong", "right"), ("right", "wrong"), ("wrong", "wrong"),
        ("boosted", "boosted")]
LABEL = {("right", "right"): "Both right", ("wrong", "right"): "Outcome model wrong",
         ("right", "wrong"): "GPS wrong", ("wrong", "wrong"): "Both wrong",
         ("boosted", "boosted"): "Both boosted"}


def centred(v):
    return v - v.mean()


# ------------------------------------------------------------------ Part A
_, A_big, _, f = sim_double_robust(n=200_000, seed=0)
grid = np.linspace(*np.quantile(A_big, [0.05, 0.95]), 60)   # fixed grid across reps
truth = centred(f(grid))
curves = {}
for rep in range(REPS):
    X, A, Y, _ = sim_double_robust(n=3000, seed=100 + rep)
    nu = {k: cross_fit_nuisances(X, A, Y, OUTCOME[k], None, random_state=rep) for k in OUTCOME}
    ng = {k: cross_fit_nuisances(X, A, Y, None, GPSM[k], random_state=rep) for k in GPSM}
    for om, gm in SCEN:
        nuis = Nuisances(mu=nu[om].mu, m=nu[om].m, pi=ng[gm].pi, varpi=ng[gm].varpi)
        for kind in ["plugin", "ipw", "dr"]:
            est, _ = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nuis, Y, kind)).curve(grid)
            curves.setdefault((om, gm, kind), []).append(centred(est))
    print(f"rep {rep} done", flush=True)

rows = []
for (om, gm, kind), cs in curves.items():
    cs = np.array(cs)
    rows.append(dict(outcome=om, gps=gm, estimator=kind,
                     abs_bias=np.mean(np.abs(cs.mean(0) - truth)),      # integrated |bias|
                     sd=np.mean(cs.std(0)),                             # integrated sd
                     rmse=np.sqrt(np.mean((cs - truth) ** 2))))
summ = pd.DataFrame(rows).round(3)
summ.to_csv(ROOT / "results" / "sim2_curve_summary.csv", index=False)
np.savez(ROOT / "results" / "sim2_curves.npz", grid=grid, truth=truth + f(grid).mean(),
         **{"|".join(k): np.array(v) + f(grid).mean() for k, v in curves.items()})
print(summ.to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
names = {"plugin": "Plug-in (outcome only)", "ipw": "IPW (GPS only)", "dr": "Doubly robust"}
for ax, sc in zip(axes, SCEN[:4]):
    for i, kind in enumerate(["plugin", "ipw", "dr"]):
        cs = np.array(curves[(*sc, kind)]) + f(grid).mean()
        col = C[[1, 3, 0][i]]
        if kind == "dr":
            lo, hi = np.percentile(cs, [5, 95], axis=0)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(grid, cs.mean(0), color=col, label=names[kind], lw=2.4 if kind == "dr" else 1.8)
    ax.plot(grid, f(grid), color=INK, ls="--", label="Truth")
    ax.set_title(LABEL[sc])
    ax.set_xlabel("Treatment a")
axes[0].set_ylabel("E[Y(a)]  (mean over reps)")
axes[0].legend(loc="lower center", fontsize=9)
fig.suptitle("Sim 2 · Doubly robust: right if either nuisance model is right", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "sim2_double_robustness.png")

# ------------------------------------------------------------------ Part B
prow = []
for rep in range(REPS):
    X, A, Y, _ = sim_double_robust(n=3000, effect="linear", seed=500 + rep)
    data = DoubleMLData.from_arrays(X, Y, A)
    for om, gm in SCEN:
        pick = {"right": lin(f_conf), "wrong": LinearRegression(), "boosted": hgb()}
        l, m = pick[om], pick[gm]
        for score in ["partialling out", "IV-type"]:
            kw = dict(ml_g=pick[om]) if score == "IV-type" else {}
            plr = DoubleMLPLR(data, l, m, score=score, n_folds=5, **kw).fit()
            prow.append(dict(rep=rep, outcome=om, treatment_model=gm, score=score,
                             theta=plr.coef[0], covered=abs(plr.coef[0] - 1) < 1.96 * plr.se[0]))
pres = pd.DataFrame(prow)
pres.to_csv(ROOT / "results" / "sim2_plr_reps.csv", index=False)
psum = pres.groupby(["score", "outcome", "treatment_model"]).agg(
    bias=("theta", lambda v: v.mean() - 1.0), sd=("theta", "std"), coverage=("covered", "mean")).round(3).reset_index()
psum.to_csv(ROOT / "results" / "sim2_plr_summary.csv", index=False)
print(psum.to_string(index=False))
