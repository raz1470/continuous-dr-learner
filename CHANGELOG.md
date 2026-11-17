# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may change the API.

## [0.1.0] - unreleased

First public release, alongside the article *Estimating Continuous Treatment Effects: Doubly Robust
Dose-Response Curves*.

### Estimation

- `fit_dose_response`: cross-fitted outcome and dose models, the doubly robust pseudo-outcome of
  Kennedy et al. (2017), and penalised splines, in one call.
- `HierarchicalDoseResponse`: curves for one population or many groups, partially pooled across a
  nested hierarchy (region → store), with penalties tuned by GCV. `pooling="none"` and
  `"complete"` for comparison.
- `default_model()`: a stack of spline-ridge regression and early-stopped gradient boosting, the
  default for every nuisance model.
- Weights: `GPS` (density-based, location-scale with a KDE residual), `DiscreteGPS` (used
  automatically for a dose with up to 30 distinct values; rare levels merged and a defensive
  mixture keeps the ends stable) and `BalancingWeights` (entropy balancing, population curve only).
- Designs with a pre-period outcome: `y_pre=` for difference-in-differences and `design="lagged"`
  to adjust for the lagged outcome instead.

### Reading the curves

- `curve`, `band` (pointwise or simultaneous), `draws`, `contrast`, `optimal_dose` and `plot`.
- Bands include the variance from estimating m(a) (`nuisance_var=True`, Kennedy et al. 2017).
- A group with no data gets its parent's curve with a predictive band that adds the spread between
  groups.
- `diagnostics()`: weight distribution, effective sample size, fitted dose range and penalties.

### Checks

- `coverage_check`: plasmode simulation on your own data; returns coverage and calibrated critical
  values for `band(critical=...)`.
- `bootstrap_curves`: percentile bootstrap that refits the whole pipeline.
- `placebo_check`: the curve on two pre-periods, judged by its size relative to the effect.
- Input checks: mismatched lengths, a constant or missing dose, groups with fewer than 10 rows, a
  point mass in the dose, and poor overlap after fitting.

### Known limitations

- The 95% bands are slightly optimistic: 75–93% coverage in our simulations, mostly from the bias
  left by capping the weights. Run `coverage_check` before relying on them.
- Groups must be nested; crossed groupings (store × channel) aren't supported.
- `drcurves.riesz` (learned Riesz weights) is experimental and not part of the public API.

[0.1.0]: https://github.com/raz1470/continuous-dr-learner/releases/tag/v0.1.0
