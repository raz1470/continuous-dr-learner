"""Example for the intervals section: one 50-confounder dataset, DR curve with
balancing weights, its default 95% pointwise band, and the band widened by a
critical value calibrated with a plasmode check (coverage_check) on the same data.

Usage: python experiments/sim5_calibrated_band.py [n_reps] [seed]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcurves import BalancingWeights, HierarchicalDoseResponse, coverage_check, cross_fit_nuisances, pseudo_outcomes  # noqa
from drcurves.nuisance import Nuisances  # noqa
from drcurves.simulate import sim_high_dim  # noqa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 700

_, Ab, _, f = sim_high_dim(n=200_000, p=50, seed=0)
g = np.linspace(*np.quantile(Ab, [0.05, 0.95]), 40)
X, A, Y, f = sim_high_dim(n=3000, p=50, seed=SEED)
truth = f(g) - f(g).mean()
om = HGB(max_iter=300, learning_rate=0.05, random_state=0)
nu = cross_fit_nuisances(X, A, Y, om, None, random_state=0)
nw = cross_fit_nuisances(X, A, Y, None, BalancingWeights(), random_state=0)
nn = Nuisances(mu=nu.mu, m=nu.m, pi=nw.pi, varpi=nw.varpi)
m = HierarchicalDoseResponse().fit(A, pseudo_outcomes(nn, Y, "dr"))
est, se = m.curve(g, centre=True)

summ, _ = coverage_check(X, A, Y, outcome_model=om, weights={"balancing": lambda r: BalancingWeights()},
                         n_reps=N_REPS, random_state=1)
print(summ.to_string())
crit = float(summ["critical_pointwise"].iloc[0])
out = dict(grid=g.tolist(), truth=truth.tolist(), est=est.tolist(), se=se.tolist(), critical=crit,
           cover_default=float(np.mean(np.abs(est - truth) <= 1.96 * se)),
           cover_calibrated=float(np.mean(np.abs(est - truth) <= crit * se)),
           check=summ.round(3).to_dict(orient="records"), n_reps=N_REPS, seed=SEED)
(ROOT / "results" / "sim5_calibrated_band.json").write_text(json.dumps(out))
print({k: out[k] for k in ["critical", "cover_default", "cover_calibrated"]})
