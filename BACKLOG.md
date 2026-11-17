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
- [ ] Bands under-cover a little (75-93%) from capping bias; an adaptive cap or a bias correction could close it.

## Packaging
- [x] CI (GitHub Actions, `ci.yml`): lint, tests on Python 3.10 with the oldest supported deps and
      3.13 with the latest, and a clean-install run of the README quick start.
- [x] Minimum versions in `pyproject.toml` (tested in CI); doubleml and causal-curve are only in the
      `experiments` extra.
- [x] Version 0.1.0 and CHANGELOG; trusted publishing to PyPI on a `v*` tag (`publish.yml`).
- [ ] Tag `v0.1.0` and publish (set up the PyPI pending publisher first; see `publish.yml`).
- [ ] Faster tests (smaller n) so CI runs in a couple of minutes.

## Docs
- [x] README: lead with what it's for (pick your assumption, get dose-response curves, many groups,
      many confounders) and when to use something else.
- [x] Two worked examples (notebooks): one population curve; group curves with a hierarchy.
- [x] Numpydoc docstrings for every public function and class.
- [x] Docs site (Sphinx, as in bayesian_vecm): user guide, rendered examples, research, API, and the
      article at /walkthrough.html; deployed to GitHub Pages by `docs.yml`.
- [ ] Link the TDS article once it's live.
