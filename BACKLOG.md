# Backlog: before publishing `drcurves` on PyPI

## Done (phase 1)
- [x] Public API frozen in `drcurves.__all__`; Riesz/stacked weights moved to experimental `drcurves.riesz`.
- [x] Default fit ~2x faster (2-fold stacking, early-stopped boosting, 2-fold GPS residuals); `n_jobs`.
- [x] Input checks (lengths, finite/constant dose, missing group labels, tiny groups, poor overlap); NaN in X handled.
- [x] Fast test suite (~40 s; slow experimental test behind `-m slow`) with an end-to-end recovery test of the defaults.

## Features
- [x] DiD option: `fit_dose_response(..., y_pre=...)` fits on the change in outcome.
- [x] Lagged-outcome option: `design="lagged"`.
- [x] `placebo_check`, judged by size relative to the main effect.
- [x] Add the integral term to the pointwise variance (Kennedy et al. 2017, as in npcausal's
      `ctseff`) so the intervals account for estimating m(a).
- [x] Use the stacked default inside `coverage_check` and `bootstrap_curves`.
- [x] Discrete doses (stacked classifier, rare levels merged, defensive mixture): with n=2,500 and rare top/bottom levels, `DiscreteGPS` weights blew the curve up at
      the edges (error 5.6 vs 0.2 at n=4,000). Stabilise (stacked classifier, pooling rare levels, or
      warning) before release.
- [x] `.plot()` on the fitted model.
- [x] Predictive band for an unseen group: add the between-group variance, so a new store's band is wider than its region's.

## Evidence
- [x] Harder simulation: interactions, thresholds, heavy-tailed noise, weaker overlap. Compare the
      stacked default, boosting alone and npcausal, so the README claims hold beyond smooth data.
- [x] DiD simulation with all three: levels, DiD, lagged outcome (stable trait vs selection on
      past outcomes).
- [x] Rerun `sim4_uncertainty` and `sim6_intervals` with the stacked default (docs/research.md).
- [ ] Bands under-cover a little (77-92%) from capping bias; an adaptive cap or a bias correction could close it.

## Packaging
- [ ] CI (GitHub Actions): tests on Python 3.10–3.13, latest sklearn/pandas.
- [ ] Pin minimum versions in `pyproject.toml`; check optional deps (doubleml, causal-curve) are
      only needed for experiments.
- [ ] Version 0.1.0, CHANGELOG, tagged release; trusted publishing to PyPI from GitHub Actions.
- [ ] Faster tests (smaller n) so CI runs in a couple of minutes.

## Docs
- [x] README: lead with what it's for (pick your assumption, get dose-response curves, many groups,
      many confounders) and when to use something else.
- [x] Two worked examples (notebooks): one population curve; group curves with a hierarchy.
- [x] Docstrings for every public function; a short API page (`docs/api.md`, generated).
- [ ] Link the TDS article once it's live.
