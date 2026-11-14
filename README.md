# Continuous treatments: curves, double robustness and shrinkage

Companion code for the article **"Continuous Treatments Need Curves, Not Coefficients"**.

DoubleML's partially linear model (`DoubleMLPLR`) is the usual tool for a continuous treatment. It has three limitations:

1. **Linear effect.** It returns one slope θ. When the true dose-response bends, θ is a slope averaged over wherever doses happen to fall, so it moves with the treatment policy.
2. **Default score isn't doubly robust.** The partialling-out score is biased when the treatment model is wrong, even with a correct outcome model.
3. **One answer per population.** A curve per store / product / country means looping, and small groups get noise.

`drcurves` implements the doubly robust dose-response estimator of Kennedy et al. (2017) with a hierarchical, partially pooled penalised-spline second stage:

```
1. cross-fit   mu(x, a) = E[Y | X, A]    and    pi(a | x)  (generalised propensity score)
2. pseudo-outcome   xi = (Y - mu(X, A)) * varpi(A) / pi(A | X) + m(A)
3. regress xi on A:   theta_store(a) = B(a)' (beta_global + b_region + b_store)
                      ridge on b = random effects -> shrinkage toward the parent
```

## Quick start

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from drcurves import fit_dose_response
from drcurves.simulate import sim_hierarchical

df, _, _ = sim_hierarchical(seed=0)                 # region, store, x0..x2, A, Y
model = fit_dose_response(X=df[["x0", "x1", "x2"]], A=df["A"], Y=df["Y"],
                          outcome_model=HGB(), groups=df[["region", "store"]])
print(model.diagnostics())                          # read this first

a_grid = np.linspace(*np.quantile(df["A"], [0.05, 0.95]), 50)
curve, se = model.curve(a_grid, ("r0", "r0_s03"))   # store curve
region, _ = model.curve(a_grid, ("r0",))            # region curve
new, _    = model.curve(a_grid, ("r0", "unseen"))   # unseen store -> its region
```

Inputs:
- `X` can be an array or a DataFrame (non-numeric columns are integer-coded). Group codes are appended automatically (`add_group_codes=True`) so the nuisance models see the groups.
- Models: by default every nuisance model (outcome, dose mean, dose scale) is `default_model()`, a stack of a spline regression and gradient boosting blended by cross-validation. Boosting alone made density weights explode with many confounders; the stack fixed that in our simulations (see the comparison below). Pass your own sklearn regressors to override.
- Weights: for a continuous `A` the default is `GPS()`, Kennedy et al.'s density-based weights. `BalancingWeights()` is an alternative for a population curve.
- `gps=RieszWeights()` (automatic debiasing) learns the weight varpi(a)/pi(a|x) directly by minimising the Riesz loss with boosted trees; no density is estimated. With 50 confounders, density weights made DR worse than plug-in, while learned weights beat it (Sim 5). **But** the intervals don't account for the bias learned weights leave behind: 95% bands covered about half the time on the Sim 1 data. Use it for the point estimate when there are many confounders, and check intervals with `bootstrap_curves`.
- `BalancingWeights()`: entropy-balancing weights that decorrelate the dose from the covariates (and their squares). Best with many covariates; population curve only (`groups=None`).
- `StackedWeights()` (experimental) mixes the two, choosing the mix by held-out Riesz loss. That loss is noisy, dominated by a few extreme weights, so it isn't the default.
- `A` with ≤ 30 distinct values (tiers, counts) is treated as discrete: `DiscreteGPS` (a classifier over the dose levels) is used automatically, because a continuous density is not a valid weight for a discrete dose.
- A point mass in a continuous dose (e.g. many zeros) triggers a warning; model "any vs none" separately.
- `fold_groups=` keeps ids (e.g. customers) together in cross-fitting; `cluster=` makes robust covariances cluster-robust.
- NaN in `A`/`Y`/groups raises; NaN in `X` warns (use models that accept missing values).
- Scale: the second stage uses a sparse solver above 300 coefficients; 2,000 stores × 2 levels fits in about 1.5 minutes on 2 cores. The integrals in the nuisance step use at most 60,000 reference rows in total.

`model.diagnostics()` reports: weight quantiles, the cap, share capped and effective-sample-size share; the fitted dose range (outside it curves are flat extrapolations, and `curve()` warns); tuned penalties and the grid edges (a warning is raised if a penalty hits an edge); groups with a constant dose (they inherit their parent's curve).

Grouping options:

| Your data | `groups=` | What you get |
|---|---|---|
| One population / one store | `None` | One curve |
| Groups with no hierarchy | `df[["store"]]` (or `df["store"]`) | One curve per store, shrunk toward the global curve |
| Nested hierarchy | `df[["region", "store"]]` (top → bottom) | Store curves shrunk toward their region, regions toward global |

Each level must be nested in the one above; crossed factors (store × channel) are not supported. `pooling` is `"partial"` (default), `"none"` or `"complete"`. Shrinkage acts on each group's *shape*; its overall level is left nearly free, so very different baselines don't weaken the pooling of shapes.

`path=None` under partial pooling is the global block, a shrinkage-weighted "typical group" curve, not the population average E[Y(a)]. For the population curve, fit with `groups=None`.

## Uncertainty

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
  riesz.py      RieszWeights: automatic debiasing (weights learned directly, no density)
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

## Caveats

- Everything assumes unconfoundedness: X must explain how doses were assigned.
- Report curves over the observed dose range. Thin overlap means large weights; weights are capped at the 99th percentile by default (`max_weight_quantile`), which trades a little bias for a lot of variance.
- Analytic intervals treat the nuisance models as fixed; see the coverage table above.

## Compared with existing packages

`experiments/compare_causal_curve.py`, `compare_npcausal.py` (needs R with npcausal and SuperLearner), `compare_stacked.py` and `compare_hard.py`: one population curve, 10 datasets per setting. RMSE of the centred curve / pointwise coverage of the 95% band.

| | 5 confounders | 50 confounders | Harder data |
|---|---|---|---|
| this package, stacked default (density weights) | **0.06** / 77% | **0.05** / 94% | **0.23** / 89% |
| this package, stacked default + `BalancingWeights()` | 0.07 / 89% | 0.05 / 91% | 0.36 / 19% |
| [npcausal](https://github.com/ehkennedy/npcausal) (R, Kennedy) | 0.08 / 90% | 0.18 / 77% | 0.30 / 13% |
| [causal-curve](https://github.com/ronikobrosly/causal-curve) TMLE (Python) | 0.54 / 100% | 0.30 / 100% | 0.44 / 100% |
| boosting alone (old default), density weights | 0.19 / 94% | 3.43 / 87% | 2.81 / 97% |

The harder data has thresholds, interactions and heavy-tailed noise. There, boosting alone gave the best outcome model (0.18 with balancing weights, 43% coverage): no learner wins everywhere, which is why the default stacks a spline regression with boosting. causal-curve's bands are 25+ times wider than the others'.

## References

- Kennedy, Ma, McHugh & Small (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *JRSS-B* 79(4).
- Chernozhukov et al. (2018). Double/debiased machine learning. *Econometrics Journal* 21(1).
- Colangelo & Lee (2020). Double debiased ML nonparametric inference with continuous treatments. arXiv:2004.03036.
- Chernozhukov, Newey & Singh (2022). Automatic debiased machine learning of causal and structural effects. *Econometrica* 90(3).
- Chernozhukov, Newey, Quintas-Martínez & Syrgkanis (2022). RieszNet and ForestRiesz. *ICML*.
- Valentin, Detommaso, Li, Opper & Brueckner (2025). On double robustness in double machine learning. NeurIPS 2025 Workshop CauScien.
- Tübbicke (2022). Entropy balancing for continuous treatments. *Journal of Econometric Methods* 11(1), 71–89.
- Vegetabile et al. (2021). Nonparametric estimation of population average dose-response curves using entropy balancing weights for continuous exposures. *Health Services and Outcomes Research Methodology* 21(1), 69–110.
- Marra & Wood (2012). Coverage properties of confidence intervals for GAM components. *Scand. J. Statistics* 39(1).
- Eilers & Marx (1996). Flexible smoothing with B-splines and penalties. *Statistical Science* 11(2).

MIT licence.
