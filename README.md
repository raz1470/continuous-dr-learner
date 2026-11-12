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
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from drcurves import fit_dose_response, GPS

model = fit_dose_response(X, A, Y, outcome_model=HGB(), gps=GPS(),
                          groups=df[["region", "store"]], pooling="partial")
curve, se = model.curve(a_grid, ("north", "store_17"))     # store curve
region, _ = model.curve(a_grid, ("north",))                 # region curve
new, _    = model.curve(a_grid, ("north", "unseen_store"))  # falls back to region
```

Grouping options:

| Your data | `groups=` | What you get |
|---|---|---|
| One population / one store | `None` | One curve |
| Groups with no hierarchy | `df[["store"]]` | One curve per store, shrunk toward the global curve |
| Nested hierarchy | `df[["region", "store"]]` (top → bottom) | Store curves shrunk toward their region, regions toward global |

Each level must be nested in the one above; crossed factors (store × channel) are not supported. `pooling` is `"partial"` (default), `"none"` or `"complete"`.

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
- `cov_type`: `"robust_bayes"` (default) = heteroscedasticity-robust data term + the prior (penalty) term of the equivalent Bayesian model, so bands widen where smoothing/shrinkage biases the curve (as in `mgcv`; Marra & Wood 2012). `"bayes"` is the plain posterior covariance; `"sandwich"` ignores penalty bias and is overconfident under shrinkage.
- `cluster=` (in `fit_dose_response`) makes the robust part cluster-robust, e.g. by customer when customers repeat. Don't cluster at the level whose curves you report.
- `optimal_dose(...)["at_edge"]` is the share of draws whose optimum sits at the end of the grid. If it's large, the optimum may be outside your data.
- `bootstrap_curves(...)` refits the whole pipeline on resampled rows (within each leaf group). It is the only option that includes nuisance-model uncertainty. Slow; use it to check a decision-driving number.

### How honest are the 95% intervals?

Monte Carlo coverage (`experiments/sim4_uncertainty.py`; share of 95% intervals containing the truth):

| Setting | Pointwise band | Simultaneous band | Dose contrast | Optimal dose |
|---|---|---|---|---|
| One curve, default (`robust_bayes`) | 95% | 88% | 100% | 100% |
| One curve, `sandwich` | 94% | 86% | 100% | 100% |
| 200 stores, default | 84% | 70% | 90% | 94% |
| 200 stores, `sandwich` | 64% | 34% | 73% | 78% |
| Stores with < 25 obs, default | 89% | 79% | 93% | 94% |
| Stores with 150+ obs, default | 80% | 60% | 87% | 95% |

Read these as follows:
- For one curve the defaults are close to nominal. Only the simultaneous band is a bit short.
- With many shrunk curves, the default bands are much better than a sandwich but still below 95%. Coverage *falls* as stores get bigger: bands shrink with data, but a small bias from the nuisance models doesn't.
- For decisions, prefer `level=0.99`, or check the key number with `bootstrap_curves`.

## Layout

```
src/drcurves/
  nuisance.py   GPS (location-scale density, KDE residuals), cross-fitting, pseudo-outcomes
  pooled.py     HierarchicalDoseResponse: P-splines with nested ridge deviations, GCV-tuned;
                bands, contrasts, optimal dose, draws
  bootstrap.py  full-pipeline bootstrap
  simulate.py   the three data-generating processes (true curves known)
experiments/
  sim1_nonlinear.py               shape: DR curve vs DoubleML θ, and θ under policy shifts
  sim2_double_robustness.py       right/wrong nuisance models: plug-in vs IPW vs DR; PLR scores
  sim3_hierarchical_shrinkage.py  8 regions x 25 stores: no / complete / partial pooling
  sim4_uncertainty.py             coverage of bands, contrasts and optimal-dose intervals
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
python tools/build_article.py      # -> article/index.html
pytest
```

Runtime is about 45 minutes on two cores, mostly simulations 2 and 3.

## Caveats

- Everything assumes unconfoundedness: X must explain how doses were assigned.
- Report curves over the observed dose range. Thin overlap means large weights; weights are capped at the 99th percentile by default (`max_weight_quantile`), which trades a little bias for a lot of variance.
- Analytic intervals treat the nuisance models as fixed; see the coverage table above.

## References

- Kennedy, Ma, McHugh & Small (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *JRSS-B* 79(4).
- Chernozhukov et al. (2018). Double/debiased machine learning. *Econometrics Journal* 21(1).
- Colangelo & Lee (2020). Double debiased ML nonparametric inference with continuous treatments. arXiv:2004.03036.
- Marra & Wood (2012). Coverage properties of confidence intervals for GAM components. *Scand. J. Statistics* 39(1).
- Eilers & Marx (1996). Flexible smoothing with B-splines and penalties. *Statistical Science* 11(2).

MIT licence.
