"""Sim 5 - Many confounders: plug-in vs DR with density weights vs DR with
automatically debiased (Riesz) weights, plus DR with the true weights as a ceiling.

4 strong nonlinear confounders plus many weak ones (p = 5, 20, 50 in total);
boosted outcome model throughout.
Usage: python experiments/sim5_dimension.py REPS [time_budget_seconds]  (resumable)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcurves import GPS, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa: E402
from drcurves.riesz import RieszWeights  # noqa: E402
from drcurves.nuisance import Nuisances  # noqa: E402
from drcurves.simulate import sim_high_dim  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PS = [5, 20, 50]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9

grids = {}
for p in PS:
    _, Ab, _, f = sim_high_dim(n=200_000, p=p, seed=0)
    grids[p] = np.linspace(*np.quantile(Ab, [0.05, 0.95]), 50)

t0 = time.time()
for rep in range(REPS):
    for p in PS:
        path = CACHE / f"sim5_p{p}_{rep:03d}.json"
        if path.exists():
            continue
        if time.time() - t0 > BUDGET:
            print("time budget reached; re-run to resume", flush=True)
            sys.exit(0)
        t = time.time()
        X, A, Y, f, w_true = sim_high_dim(n=3000, p=p, seed=500 + rep, return_oracle=True)
        hgb = HGB(max_iter=300, learning_rate=0.05, random_state=0)
        nu = cross_fit_nuisances(X, A, Y, hgb, GPS(), random_state=rep)
        nr = cross_fit_nuisances(X, A, Y, None(random_state=rep), random_state=rep)
        no = Nuisances(mu=nu.mu, m=nu.m, pi=np.ones_like(w_true), varpi=w_true)
        nrz = Nuisances(mu=nu.mu, m=nu.m, pi=nr.pi, varpi=nr.varpi)
        g = grids[p]
        out = {}
        for kind, nn, pk in [("plugin", nu, "plugin"), ("dr_density", nu, "dr"),
                             ("dr_riesz", nrz, "dr"), ("dr_true_weights", no, "dr")]:
            est, _ = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nn, Y, pk)).curve(g)
            out[kind] = (est - est.mean()).tolist()
        path.write_text(json.dumps(out))
        print(f"p={p} rep {rep} done in {time.time() - t:.0f}s", flush=True)

rows, curves = [], {}
for p in PS:
    _, _, _, f = sim_high_dim(n=10, p=p, seed=0)
    t = f(grids[p]) - f(grids[p]).mean()
    files = sorted(CACHE.glob(f"sim5_p{p}_*.json"))
    for kind in ["plugin", "dr_density", "dr_riesz", "dr_true_weights"]:
        cs = np.array([json.loads(fp.read_text())[kind] for fp in files])
        curves[(p, kind)] = cs
        rows.append(dict(p=p, estimator=kind, reps=len(cs),
                         abs_bias=np.mean(np.abs(cs.mean(0) - t)), sd=np.mean(cs.std(0)),
                         rmse=np.sqrt(np.mean((cs - t) ** 2))))
res = pd.DataFrame(rows).round(3)
res.to_csv(ROOT / "results" / "sim5_dimension_summary.csv", index=False)
np.savez(ROOT / "results" / "sim5_curves.npz",
         **{f"{p}|{k}": v for (p, k), v in curves.items()},
         **{f"grid{p}": grids[p] for p in PS})
print(res.to_string(index=False))
