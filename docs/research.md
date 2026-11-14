# Research notes and older results

These tables come from simulations run **before** the stacked default model (`default_model()`); the
nuisance models were boosted trees alone. They are kept for the record and will be rerun (see
`BACKLOG.md`). The article's results and the package comparison in the README use the stacked default.

## Uncertainty tools and interval coverage (boosted models)

```python
est, se = model.curve(a_grid, path)                         # pointwise s.e.
est, lo, hi, crit = model.band(a_grid, path)                # simultaneous 95% band (whole curve)
est, lo, hi, _ = model.band(a_grid, path, simultaneous=False)
lift, se = model.contrast(0.10, 0.20, path)                 # theta(0.20) - theta(0.10)
gap, se = model.contrast(0.15, 0.15, path_a, path_b)        # two groups at the same dose
best = model.optimal_dose(a_grid, path)                     # {"dose", "lo", "hi", "at_edge", "draws"}
draws = model.draws(a_grid, path, n_draws=4000)             # curves for any other summary
```

- `centre=True` (on `curve`, `band`, `draws`) gives the curve's shape (minus its mean) with matching uncertainty. Use it when group baselines are not what you care about.
- `cov_type`: `"bayes"` (default) is the posterior covariance σ²(X′X+P)⁻¹ of the equivalent Bayesian model, as used by `mgcv`; its intervals widen where the penalty pulls the curve and have close to nominal coverage *averaged across the curve* (Marra & Wood 2012). `"robust_bayes"` swaps in a heteroscedasticity-robust data term (our own hybrid; it did slightly worse in the coverage study). `"sandwich"` ignores penalty bias and is overconfident under shrinkage.
- `cluster=` (in `fit_dose_response`) makes the robust part cluster-robust, e.g. by customer when customers repeat. Don't cluster at the level whose curves you report.
- `optimal_dose(...)["at_edge"]` is the share of draws whose optimum sits at the end of the grid. If it's large, the optimum may be outside your data.
- `bootstrap_curves(...)` refits the whole pipeline on resampled rows (within each leaf group; duplicates of a row stay in the same cross-fitting fold). It is the only option that includes nuisance-model uncertainty. The percentile band is pointwise, and it is centred on a smoothed estimate, so it does not capture smoothing bias. Slow; use it to check a decision-driving number. `X` must be numeric (`drcurves.prepare_X`).

### Intervals with many covariates (read this before using on real data)

`experiments/sim6_intervals.py` (run before the stacked default; to be rerun, see BACKLOG.md): population curve, boosted outcome model, Sim 5 data, 15 replications. Coverage of 95% shape bands; SE ratio = reported s.e. / actual spread (1 = right).

| Confounders | Weights | Bias | RMSE | SE ratio | Pointwise | Simultaneous |
|---|---|---|---|---|---|---|
| 5 | density | 0.09 | 0.22 | 1.46 | 94% | 87% |
| 5 | learned | 0.09 | 0.13 | 0.85 | 58% | 7% |
| 5 | balancing | 0.06 | 0.10 | 0.89 | 76% | 20% |
| 50 | density | 2.21 | 4.13 | 1.27 | 82% | 20% |
| 50 | learned | 0.26 | 0.32 | 0.95 | 30% | 0% |
| 50 | balancing | 0.08 | 0.14 | 1.09 | 72% | 0% |

- **Few covariates:** density weights (the default) give honest intervals.
- **Many covariates:** density weights from the default boosted dose model break, and learned (Riesz) weights carry too much bias. **`BalancingWeights()`** (entropy balancing for a continuous dose; in-sample, population curve only) is clearly best: lowest error, and standard errors that match the real spread. The bands still under-cover, because a small bias remains; richer balancing bases, prognostic scores, removing the weight cap and undersmoothing didn't remove it.
- **So calibrate on your own data.** `coverage_check` keeps your real X and doses and simulates outcomes from a known curve, then reports the critical value that gave 95% coverage:

```python
from drcurves import coverage_check, BalancingWeights, fit_dose_response
summary, _ = coverage_check(X, A, Y, n_reps=30)              # density / learned / balancing
print(summary)                                                # coverage, se_ratio, critical values
model = fit_dose_response(X, A, Y, outcome_model=HGB(), gps=BalancingWeights())
crit = summary.set_index("weights").loc["balancing", "critical_pointwise"]
est, lo, hi, _ = model.band(a_grid, simultaneous=False, critical=crit)
```

Validation: on one 50-covariate dataset, `coverage_check` estimated a pointwise critical value of 3.35 (true value from the full simulation: 3.55) and coverage of 66% (true: 72%). Its simultaneous critical value was too low (4.0 vs 5.5). For whole-curve bands use `n_reps >= 50`, or add a margin. The check assumes unconfoundedness (it tests estimation, not identification) and builds outcomes from a fitted model, so treat it as optimistic.

### How honest are the 95% intervals?

Monte Carlo coverage (`experiments/sim4_uncertainty.py`; share of 95% intervals containing the truth):

Shape (centred curve) coverage; one curve: 42 replications of Sim 1 (centred inverted U), stores: 15 replications of Sim 3 (200 stores each).

| Setting | Pointwise band | Simultaneous band | Dose contrast | Optimal dose |
|---|---|---|---|---|
| One curve, default (`bayes`) | 91% | 81% | 95% | 100% |
| One curve, `robust_bayes` | 93% | 83% | 90% | 100% |
| One curve, `sandwich` | 90% | 76% | 90% | 100% |
| 200 stores, default (`bayes`) | 88% | 78% | 92% | 95% |
| 200 stores, `robust_bayes` | 87% | 74% | 91% | 96% |
| 200 stores, `sandwich` | 64% | 34% | 72% | 78% |
| Stores with < 25 obs, default | 90% | 82% | 93% | 94% |
| Stores with 150+ obs, default | 84% | 69% | 90% | 97% |

Read these as follows:
- For one curve the default pointwise band is a little below nominal and the simultaneous band falls further short; contrasts are close to nominal.
- With many shrunk curves, the default bands are much better than a sandwich but still below 95%. Coverage falls as stores get bigger; a plausible reason (not isolated by the simulation) is a small bias from the nuisance models that stays while the band narrows.
- For decisions, prefer `level=0.99`, or check the key number with `bootstrap_curves`.


## Layout

```
src/drcurves/
  nuisance.py   GPS (location-scale density, KDE residuals), cross-fitting, pseudo-outcomes
  pooled.py     HierarchicalDoseResponse: P-splines with nested ridge deviations, GCV-tuned;
                bands, contrasts, optimal dose, draws
  bootstrap.py  full-pipeline bootstrap
  models.py     default_model: spline-ridge + boosting stack used for every nuisance model
  riesz.py      experimental: RieszWeights (weights learned directly, no density)
  balance.py    BalancingWeights: entropy balancing for a continuous dose
  check.py      coverage_check: plasmode coverage check and calibration on your data
  simulate.py   the three data-generating processes (true curves known)
experiments/
  sim1_nonlinear.py               shape: DR curve vs DoubleML θ, and θ under policy shifts
  sim2_double_robustness.py       right/wrong nuisance models: plug-in vs IPW vs DR; PLR scores
  sim3_hierarchical_shrinkage.py  8 regions x 25 stores: no / complete / partial pooling
  sim4_uncertainty.py             coverage of bands, contrasts and optimal-dose intervals
  sim5_dimension.py               5 / 20 / 50 confounders: plug-in vs DR (density vs Riesz weights)
  sim6_intervals.py               interval coverage with many confounders: density / learned / balancing
tools/build_article.py            builds article/index.html from results/
figures/  results/  article/
```

## Reproduce

```bash
pip install -e .
python experiments/sim1_nonlinear.py
python experiments/sim2_double_robustness.py 30
python experiments/sim3_hierarchical_shrinkage.py 5
python experiments/sim4_uncertainty.py single 42   # resumable; add a time budget in seconds as 3rd arg
python experiments/sim4_uncertainty.py stores 15
python experiments/sim5_dimension.py 10            # resumable
python experiments/sim6_intervals.py 15            # resumable
python tools/build_article.py      # -> article/index.html
pytest
```

Runtime is about 45 minutes on two cores, mostly simulations 2 and 3.

