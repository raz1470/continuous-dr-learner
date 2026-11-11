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

`groups=None` gives one population curve. `pooling` is `"partial"` (default), `"none"` or `"complete"`.

## Layout

```
src/drcurves/
  nuisance.py   GPS (location-scale density, KDE residuals), cross-fitting, pseudo-outcomes
  pooled.py     HierarchicalDoseResponse: P-splines with nested ridge deviations, GCV-tuned
  simulate.py   the three data-generating processes (true curves known)
experiments/
  sim1_nonlinear.py               shape: DR curve vs DoubleML θ, and θ under policy shifts
  sim2_double_robustness.py       right/wrong nuisance models: plug-in vs IPW vs DR; PLR scores
  sim3_hierarchical_shrinkage.py  8 regions x 25 stores: no / complete / partial pooling
tools/build_article.py            builds article/index.html from results/
figures/  results/  article/
```

## Reproduce

```bash
pip install -e .
python experiments/sim1_nonlinear.py
python experiments/sim2_double_robustness.py 30
python experiments/sim3_hierarchical_shrinkage.py 5
python tools/build_article.py      # -> article/index.html
pytest
```

Runtime is about 45 minutes on two cores, mostly simulations 2 and 3.

## Caveats

- Everything assumes unconfoundedness: X must explain how doses were assigned.
- Report curves over the observed dose range. Thin overlap means large weights; weights are capped at the 99th percentile by default (`max_weight_quantile`), which trades a little bias for a lot of variance.
- Pointwise intervals are sandwich intervals from the second stage and ignore smoothing bias.

## References

- Kennedy, Ma, McHugh & Small (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *JRSS-B* 79(4).
- Chernozhukov et al. (2018). Double/debiased machine learning. *Econometrics Journal* 21(1).
- Colangelo & Lee (2020). Double debiased ML nonparametric inference with continuous treatments. arXiv:2004.03036.
- Eilers & Marx (1996). Flexible smoothing with B-splines and penalties. *Statistical Science* 11(2).

MIT licence.
