"""Sim 6 - Honest intervals with many confounders?

Population dose-response curve on the Sim 5 data (4 strong + many weak
confounders), boosted outcome model, three weight models:
  density   : GPS() (Kennedy et al. density weights)
  learned   : RieszWeights() (automatic debiasing)
  balancing : BalancingWeights() (entropy balancing for a continuous dose)
Scored on the shape (centred curve): bias, RMSE, coverage of 95% pointwise and
simultaneous bands (Bayesian covariance), and whether the reported s.e. matches
the actual spread across replications (se_ratio = mean s.e. / empirical s.d.).

Usage: python experiments/sim6_intervals.py REPS [time_budget_seconds]  (resumable)
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
from drcurves import (GPS, BalancingWeights, HierarchicalDoseResponse, RieszWeights,  # noqa: E402
                      cross_fit_nuisances, pseudo_outcomes)
from drcurves.nuisance import Nuisances  # noqa: E402
from drcurves.simulate import sim_high_dim  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PS = [5, 50]
METHODS = {"density": lambda r: GPS(random_state=r),
           "learned": lambda r: RieszWeights(random_state=r),
           "balancing": lambda r: BalancingWeights()}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9

grids = {}
for p in PS:
    _, Ab, _, f = sim_high_dim(n=200_000, p=p, seed=0)
    grids[p] = np.linspace(*np.quantile(Ab, [0.05, 0.95]), 40)

t0 = time.time()
for rep in range(REPS):
    for p in PS:
        path = CACHE / f"sim6_p{p}_{rep:03d}.json"
        if path.exists():
            continue
        if time.time() - t0 > BUDGET:
            print("time budget reached; re-run to resume", flush=True)
            sys.exit(0)
        t = time.time()
        X, A, Y, f = sim_high_dim(n=3000, p=p, seed=700 + rep)
        g = grids[p]
        truth = f(g) - f(g).mean()
        nu = cross_fit_nuisances(X, A, Y, HGB(max_iter=300, learning_rate=0.05, random_state=0),
                                 None, random_state=rep)
        out = {}
        for name, mk in METHODS.items():
            nw = cross_fit_nuisances(X, A, Y, None, mk(rep), random_state=rep)
            nn = Nuisances(mu=nu.mu, m=nu.m, pi=nw.pi, varpi=nw.varpi)
            m = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nn, Y, "dr"))
            est, se = m.curve(g, centre=True)
            _, lo_s, hi_s, _ = m.band(g, simultaneous=True, centre=True, n_draws=4000)
            out[name] = {"est": est.tolist(), "se": se.tolist(),
                         "pointwise": float(np.mean(np.abs(est - truth) <= 1.96 * se)),
                         "simultaneous": float(np.all((truth >= lo_s) & (truth <= hi_s)))}
        path.write_text(json.dumps(out))
        print(f"p={p} rep {rep} done in {time.time() - t:.0f}s", flush=True)

rows = []
for p in PS:
    g = grids[p]
    _, _, _, f = sim_high_dim(n=10, p=p, seed=0)
    truth = f(g) - f(g).mean()
    files = sorted(CACHE.glob(f"sim6_p{p}_*.json"))
    for name in METHODS:
        d = [json.loads(fp.read_text())[name] for fp in files]
        E = np.array([x["est"] for x in d])
        S = np.array([x["se"] for x in d])
        rows.append(dict(p=p, weights=name, reps=len(d),
                         abs_bias=np.mean(np.abs(E.mean(0) - truth)),
                         rmse=np.sqrt(np.mean((E - truth) ** 2)),
                         se_ratio=np.mean(S.mean(0) / E.std(0)),
                         pointwise=np.mean([x["pointwise"] for x in d]),
                         simultaneous=np.mean([x["simultaneous"] for x in d])))
res = pd.DataFrame(rows).round(3)
res.to_csv(ROOT / "results" / "sim6_intervals_summary.csv", index=False)
print(res.to_string(index=False))
