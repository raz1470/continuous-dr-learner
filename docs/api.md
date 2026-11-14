# API reference

Generated from the docstrings by `tools/build_api_docs.py`. Everything listed here is
the supported public API; other modules (`drcurves.riesz`, `drcurves.simulate`) may change before 1.0.

## One call

### `fit_dose_response`

```python
fit_dose_response(X, A, Y, outcome_model=None, gps=None, groups=None, pooling='partial', n_folds=5, n_basis=10, cluster=None, fold_groups=None, cov_type='bayes', max_weight_quantile=0.99, add_group_codes=True, max_levels=30, y_pre=None, design=None, n_jobs=None, random_state=0)
```

```text
One call: cross-fitted nuisances -> DR pseudo-outcomes -> pooled splines.

X        : covariates (array or DataFrame; categorical columns are coded).
A, Y     : dose and outcome.
groups   : None (one population curve); a DataFrame with one column (flat
           groups, each shrunk toward the global curve); or several columns
           top -> bottom (e.g. ["region", "store"]). Levels must be nested.
outcome_model : sklearn regressor for E[Y | X, A]. None uses default_model(), a
           spline-ridge + boosting stack.
gps      : the weight model. None picks DiscreteGPS when A has <= max_levels
           distinct values, otherwise GPS() (density-based weights, Kennedy
           et al. 2017). BalancingWeights() is an alternative for a
           population curve (groups=None).
cluster  : cluster ids for the robust covariance types.
fold_groups : ids kept together in cross-fitting (e.g. customer id).
cov_type : "bayes" (default), "robust_bayes" or "sandwich".
n_jobs   : parallel jobs for the default models (None = 1, -1 = all cores).
y_pre    : the outcome before the dose (same units). With it, choose `design`:
           "did" (default): fit on the change Y - y_pre. Removes stable unobserved traits,
               but comparing doses needs strong parallel trends: units on different doses
               would have followed the same path had their doses been equal.
           "lagged": keep Y and adjust for y_pre as a confounder. Right when the dose
               responds to past outcomes rather than to a stable trait.
           The two bracket the truth when you don't know which applies; check "did" with
           placebo_check on two pre-periods.
Returns a fitted HierarchicalDoseResponse with .nuisances_, .pseudo_outcomes_
and .diagnostics().
```

### `HierarchicalDoseResponse`

```python
HierarchicalDoseResponse(n_basis=10, pooling='partial', lam_grid=None, n_sweeps=2, gcv_gamma=1.4, cov_type='bayes', lam_int=0.0001, dense_max=300, n_probes=40)
```

```text
Partially pooled dose-response curves.

Parameters
----------
n_basis : number of B-spline basis functions (reduced automatically when the
    dose takes only a few distinct values).
pooling : "partial" (GCV-tuned shrinkage), "none" (each group on its own,
    only smoothness penalised) or "complete" (one curve for everybody).
cov_type : default covariance for intervals: "bayes" (posterior covariance
    of the equivalent Bayesian model, as used by mgcv), "robust_bayes"
    (heteroscedasticity-robust data term + prior term) or "sandwich"
    (ignores penalty bias; overconfident under shrinkage).
```

#### `.band`

```python
band(self, a_grid, path=None, level=0.95, simultaneous=True, centre=False, cov_type=None, n_draws=10000, seed=0, critical=None)
```

```text
Confidence band; returns (est, lo, hi, critical_value).

simultaneous=True uses a sup-t critical value (the 95th percentile of
the largest standardised deviation across the grid), so the band holds
for the whole curve at once - what you need for statements like "the
curve peaks between x and y".
critical: use this critical value instead (e.g. the calibrated value
from coverage_check), multiplying the standard errors.
```

#### `.contrast`

```python
contrast(self, a_from, a_to, path=None, path_to=None, cov_type=None)
```

```text
theta_{path_to}(a_to) - theta_{path}(a_from) and its s.e.

Same path: the effect of moving the dose from a_from to a_to.
Different paths, same dose: how two groups differ at that dose.
```

#### `.curve`

```python
curve(self, a_grid, path=None, centre=False, cov_type=None)
```

```text
Curve and pointwise s.e. for a group path, e.g. ("north", "store_17").

path=None returns the global block. Under partial pooling that is a
shrinkage-weighted "typical group" curve, not the population average
E[Y(a)]; for the population curve fit with groups=None.
Unknown levels fall back to their parent (a new store gets its region's
curve). centre=True returns the shape (curve minus its mean over a_grid).
```

#### `.draws`

```python
draws(self, a_grid, path=None, n_draws=4000, centre=False, cov_type=None, seed=0)
```

```text
Curves drawn from N(coef, cov): array (n_draws, len(a_grid)).
```

#### `.fit`

```python
fit(self, a, y, groups=None, cluster=None, penalties=None)
```

```text
Fit the pooled curves.

groups    : DataFrame (or Series) with one column per level, top -> bottom;
            one column = flat groups; None = one curve.
cluster   : optional cluster ids for the robust ("sandwich" /
            "robust_bayes") covariance, e.g. customer id when customers
            repeat. Don't cluster at the level whose curves you report.
penalties : optional (lam_smooth, [lam_level, ...]) to skip GCV tuning.
```

#### `.optimal_dose`

```python
optimal_dose(self, a_grid, path=None, level=0.95, n_draws=4000, maximize=True, cov_type=None, seed=0)
```

```text
Dose maximising (or minimising) the curve on a_grid, with an interval
from the argmax across draws. `at_edge` = share of draws whose optimum is
at an end of a_grid: if large, the optimum may lie outside the data.
```

#### `.plot`

```python
plot(self, a_grid=None, paths=(None,), labels=None, band='pointwise', level=0.95, centre=False, truth=None, rug=True, ax=None, cov_type=None)
```

```text
Plot one or more curves with their bands (needs matplotlib).

paths  : group paths to draw, e.g. [None] or [("north",), ("north", "s17")].
band   : "pointwise", "simultaneous" or None.
truth  : optional callable a -> true curve, drawn dashed (for simulations).
rug    : tick marks along the x axis at the observed doses.
Returns the matplotlib Axes.
```

## Models and weights

### `default_model`

```python
default_model(categorical_features=None, n_knots=5, max_iter=500, learning_rate=0.05, cv=2, n_jobs=None, random_state=0)
```

```text
Spline-ridge + boosting stack, usable for the outcome model and the dose mean/scale models.

categorical_features : column indices to one-hot encode in the spline member and treat as
    categorical in boosting (e.g. group codes). Other columns get cubic B-splines.
cv : folds used to choose the blend. 2 was as accurate as 3 in our simulations and a third faster.
n_jobs : fit the two members in parallel (passed to StackingRegressor).
Missing values: the spline member imputes the column mean; boosting handles NaN natively.
Boosting stops early once a 10% validation split stops improving, so max_iter is a ceiling.
```

### `GPS`

```python
GPS(mean_model=None, scale_model=None, heteroscedastic=True, residual='kde', categorical_features=None, n_jobs=None, random_state=0)
```

```text
Location-scale model for the conditional density of a continuous treatment.

    A = m(X) + s(X) * e,   e ~ f  (f estimated by KDE or assumed Gaussian)

Parameters
----------
mean_model : sklearn regressor for E[A | X]. Default: drcurves.models.default_model,
    a spline-ridge + boosting stack (boosting alone gave unstable weights in our simulations).
scale_model : sklearn regressor for E[|A - m(X)| | X]. Ignored if
    ``heteroscedastic=False``. (Absolute residuals are far less noisy than
    log squared residuals, which produce near-zero scales and exploding weights.)
heteroscedastic : model s(X) (True) or use a constant scale (False).
categorical_features : column indices treated as categorical by the default models.
n_jobs : parallel jobs for the default models.
residual : "kde" (non-parametric shape for e) or "gaussian".
```

#### `.density`

```python
density(self, X, a)
```

```text
pi(a | x) evaluated row-wise.
```

#### `.fit`

```python
fit(self, X, A)
```

```text
Fit the dose mean and scale models and the residual density on (X, A).
```

### `DiscreteGPS`

```python
DiscreteGPS(classifier=None, defensive=0.05, min_count=20, floor=0.0001, n_jobs=None, random_state=0)
```

```text
Generalised propensity score for a dose with a few distinct values
(tiers, counts such as notifications per week): a classifier's predicted
probability of each level. With a discrete dose, varpi/pi is a ratio of
probabilities, which is the right weight (a continuous density is not).

classifier : default stacks a spline logistic regression with boosting, which gives
    far better-calibrated probabilities for rare levels than boosting alone.
defensive  : share of the marginal level probabilities mixed into every prediction,
    p = (1 - defensive) * p(a | x) + defensive * p(a). This bounds each weight by
    1 / defensive (20 by default), as the continuous GPS's defensive mixture does.
    Without it, rare top or bottom levels got near-zero probabilities and their
    weights blew the ends of the curve up in smaller samples.
```

#### `.density`

```python
density(self, X, a)
```

```text
Probability of dose level a given X (a is snapped to the nearest observed level).
```

#### `.fit`

```python
fit(self, X, A)
```

```text
Fit the level classifier on (X, A), merging levels with fewer than min_count rows.
```

### `BalancingWeights`

```python
BalancingWeights(basis='quadratic', dose_powers=(1, 2), ridge=0.01, max_weight=50.0, scores=False, score_model=None, n_knots=4)
```

```text
Entropy-balancing weights for a continuous dose.

basis   : "quadratic" (h(X) = standardised X and X^2) or "linear".
dose_powers : powers of the standardised dose used in g(A) (default 1, 2).
ridge   : penalty on the dual variables; larger = smoother weights, looser balance.
```

#### `.fit_weights`

```python
fit_weights(self, X, A, groups=None, Y=None)
```

```text
Weights for every row, chosen so the weighted dose is uncorrelated with the balance terms.
```

## Checks

### `coverage_check`

```python
coverage_check(X, A, Y, groups=None, outcome_model=None, weights=None, n_reps=20, shape='inverted_u', effect_size=None, level_points=40, n_folds=5, add_group_codes=True, random_state=0, verbose=True)
```

```text
Plasmode coverage check of the curve's 95% intervals on your data.

X, A, Y   : your covariates, dose and outcome (Y is used only to build h and e).
groups    : optional hierarchy (DataFrame, top -> bottom). If given, curves are
            fitted with partial pooling and every group's curve is scored
            against the common true shape (balancing weights are skipped,
            as they support one population only).
outcome_model : default: default_model() (spline-ridge + boosting stack).
weights   : dict name -> callable(seed) returning a weight model; default
            density (GPS) and balancing.
shape     : "inverted_u", "saturating", "linear" or a callable of the
            standardised dose z = (a - median) / sd.
effect_size : peak-to-trough size of the true curve; default 0.5 * sd(Y).

Returns (summary DataFrame, per-replication DataFrame). Read `pointwise`
and `simultaneous` (share of 95% intervals containing the truth), `se_ratio`
(reported s.e. / actual spread; < 1 means overconfident) and `abs_bias`.
`critical_pointwise` / `critical_simultaneous` are the critical values that
gave 95% coverage on your data: pass one to model.band(..., critical=...)
for calibrated intervals (compare with 1.96 and the usual sup-t value).
With few replications these are noisy; use n_reps >= 30 for reporting.
```

### `bootstrap_curves`

```python
bootstrap_curves(X, A, Y, outcome_model, gps, a_grid, paths=(None,), groups=None, pooling='partial', n_boot=100, refit_nuisances=True, retune=True, n_folds=5, n_basis=10, level=0.95, centre=False, random_state=0, verbose=False)
```

```text
Percentile bootstrap bands for one or more curves.

paths            : group paths to return, e.g. [None, ("north",), ("north", "s17")].
refit_nuisances  : refit outcome model and GPS in every replicate (slow, complete)
                   or keep the original pseudo-outcomes (fast, misses nuisance noise).
retune           : re-select the penalties by GCV in every replicate.
outcome_model, gps : None uses the package defaults (default_model() and GPS()).
Rows are resampled within each leaf group, so every group appears in every
replicate and its curve stays defined.

X must already be numeric (see drcurves.prepare_X). The percentile band is
pointwise; use the stored draws for any other summary.

Returns {path: {"est", "lo", "hi", "draws"}}.
```

### `placebo_check`

```python
placebo_check(X, A, y_pre, y_prepre, main=None, a_grid=None, groups=None, path=None, **fit_kwargs)
```

```text
Fit the curve on the change between two pre-periods, where the dose can't have acted yet.

Under (strong) parallel trends the placebo curve is flat. Judge it by its *size* relative to
the effect you care about, not by whether its band excludes zero: with a well-fitted model the
band is narrow enough to flag deviations too small to matter.

X, A        : covariates and the (later) dose.
y_pre       : outcome in the last period before the dose.
y_prepre    : outcome in an earlier period before the dose.
main        : optional fitted model of the real effect (e.g. fit_dose_response(..., y_pre=...)),
              used to express the placebo's size relative to the effect.
a_grid      : doses at which to compare; default the 5th to 95th percentile of A.
groups, path: optional hierarchy and the group path whose curves to compare.
fit_kwargs  : passed to fit_dose_response (models, weights, n_jobs, ...).

Returns a dict: grid, placebo curve and s.e. (centred), max_abs (largest deviation from flat),
effect_range (range of the main curve, if given), relative (max_abs / effect_range) and
share_band_excludes_zero (for reference only).
```

## Building blocks

### `cross_fit_nuisances`

```python
cross_fit_nuisances(X, A, Y, outcome_model, gps, groups=None, n_folds=5, n_grid=60, n_ref=1000, max_ref_total=60000, max_levels=30, fold_groups=None, n_jobs=None, random_state=0) -> 'Nuisances'
```

```text
Cross-fit outcome model and GPS; return everything the pseudo-outcomes need.

groups      : finest grouping level (length n). Integrals over P(x) are taken
              *within* each group, so regressing the pseudo-outcome on
              (A, group) targets theta_g(a) = E[Y(a) | G = g]. The group
              should also be (encoded) in X so the nuisance models can use it.
fold_groups : ids kept together in cross-fitting (e.g. customer id).
n_jobs      : folds fitted in parallel threads (None = 1, -1 = all cores).
max_levels  : if A has at most this many distinct values it is treated as
              discrete: integrals are evaluated at those values exactly.
n_ref / max_ref_total : rows per group (and in total) used for the integrals.
Either model may be None, in which case its outputs are NaN.
```

### `pseudo_outcomes`

```python
pseudo_outcomes(nuis: 'Nuisances', Y, kind='dr', max_weight_quantile=0.99)
```

```text
Pseudo-outcomes whose regression on A estimates the dose-response curve.

kind = "dr"      : (Y - mu) * varpi/pi + m      (Kennedy et al. 2017)
       "ipw"     : Y * varpi/pi                 (needs GPS right)
       "plugin"  : m                            (needs outcome model right)
```

### `Nuisances`

```python
Nuisances(mu: 'np.ndarray', pi: 'np.ndarray', varpi: 'np.ndarray', m: 'np.ndarray') -> None
```

```text
Cross-fitted nuisance values per row: mu = E[Y | X, A], pi = density of A given X,
varpi = marginal density of A, m = mu averaged over X at the row's dose.
```
