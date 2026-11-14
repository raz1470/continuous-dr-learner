"""Sim 4 - Are the uncertainty bands honest?

Monte Carlo coverage of 95% intervals for
  * the curve shape (pointwise and simultaneous bands, centred curves)
  * a dose contrast theta(a_75) - theta(a_25)
  * the optimal dose (interval from the argmax of posterior draws)
under three covariance choices: sandwich, bayes, robust_bayes.

Part A: one population curve (Sim 1 data, inverted U).
Part B: 200 store curves in 8 regions (Sim 3 data), by store size.

Usage: python experiments/sim4_uncertainty.py {single|stores} REPS [time_budget_seconds]
Each replicate is cached in results/cache/, so the script can be re-run to resume.
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
from drcurves import GPS, HierarchicalDoseResponse, cross_fit_nuisances, default_model, pseudo_outcomes  # noqa: E402
from drcurves.simulate import CURVES, sim_hierarchical, sim_nonlinear  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
COVS = ["sandwich", "bayes", "robust_bayes"]


def score(model, grid, truth, path, a_lo, a_hi, f_true):
    """Coverage indicators for one curve under each covariance type."""
    out = []
    t_c = truth - truth.mean()
    true_contrast = f_true(a_hi) - f_true(a_lo)
    true_peak = grid[np.argmax(truth)]
    for ct in COVS:
        est, lo_p, hi_p, _ = model.band(grid, path, simultaneous=False, centre=True, cov_type=ct)
        _, lo_s, hi_s, crit = model.band(grid, path, simultaneous=True, centre=True, cov_type=ct,
                                         n_draws=4000)
        c, se = model.contrast(a_lo, a_hi, path, cov_type=ct)
        opt = model.optimal_dose(grid, path, cov_type=ct, n_draws=2000)
        out.append(dict(cov_type=ct,
                        pointwise=float(np.mean((t_c >= lo_p) & (t_c <= hi_p))),
                        simultaneous=float(np.all((t_c >= lo_s) & (t_c <= hi_s))),
                        contrast=float(abs(c - true_contrast) <= 1.96 * se),
                        optimum=float(opt["lo"] - 1e-9 <= true_peak <= opt["hi"] + 1e-9),
                        band_width=float(np.mean(hi_p - lo_p)), sup_crit=crit,
                        opt_width=opt["hi"] - opt["lo"]))
    return out


def rep_single(rep):
    X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u_centred", seed=2000 + rep)
    # package default: spline-ridge + boosting stack for every nuisance model
    nuis = cross_fit_nuisances(X, A, Y, default_model(), GPS(random_state=rep), random_state=rep)
    xi = pseudo_outcomes(nuis, Y)
    m = HierarchicalDoseResponse().fit(A, xi)
    grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 50)
    a_lo, a_hi = np.quantile(A, [0.25, 0.75])
    rows = score(m, grid, f(grid), None, a_lo, a_hi, f)
    for r in rows:
        r.update(rep=rep)
    return rows


def rep_stores(rep):
    df, truth, _ = sim_hierarchical(seed=3000 + rep)
    X = np.column_stack([pd.Categorical(df.region).codes, pd.Categorical(df.store).codes,
                         df[["x0", "x1", "x2"]].to_numpy()])
    A, Y = df.A.to_numpy(), df.Y.to_numpy()
    gps = GPS(categorical_features=[0, 1], random_state=rep)      # package default models
    nuis = cross_fit_nuisances(X, A, Y, default_model([0, 1]), gps, groups=df.store.to_numpy(), random_state=rep)
    xi = pseudo_outcomes(nuis, Y)
    m = HierarchicalDoseResponse(pooling="partial").fit(A, xi, df[["region", "store"]])
    grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 40)
    a_lo, a_hi = np.quantile(A, [0.25, 0.75])
    sizes = df.groupby("store").size()
    rows = []
    for (r, s), fs in truth.items():
        for row in score(m, grid, fs(grid), (r, s), a_lo, a_hi, fs):
            row.update(rep=rep, store=s, n=int(sizes[s]))
            rows.append(row)
    return rows


if __name__ == "__main__":
    part, reps = sys.argv[1], int(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
    t0 = time.time()
    fn = rep_single if part == "single" else rep_stores
    for rep in range(reps):
        path = CACHE / f"sim4_{part}_{rep:03d}.json"
        if path.exists():
            continue
        if time.time() - t0 > budget:
            print("time budget reached; re-run to resume", flush=True)
            sys.exit(0)
        t = time.time()
        path.write_text(json.dumps(fn(rep)))
        print(f"{part} rep {rep} done in {time.time() - t:.0f}s", flush=True)

    rows = [r for p in sorted(CACHE.glob(f"sim4_{part}_*.json")) for r in json.loads(p.read_text())]
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / f"sim4_{part}_reps.csv", index=False)
    cols = ["pointwise", "simultaneous", "contrast", "optimum", "band_width", "opt_width"]
    if part == "single":
        summ = res.groupby("cov_type")[cols].mean().round(3)
    else:
        res["size_band"] = pd.cut(res.n, [0, 25, 60, 150, 10_000], labels=["<25", "25–60", "60–150", "150+"])
        summ = res.groupby(["cov_type", "size_band"], observed=True)[cols].mean().round(3)
        summ = pd.concat([summ, res.groupby("cov_type")[cols].mean().round(3)
                          .assign(size_band="all").set_index("size_band", append=True)])
    summ.to_csv(ROOT / "results" / f"sim4_{part}_summary.csv")
    print(f"{res.rep.nunique()} replications")
    print(summ.to_string())
