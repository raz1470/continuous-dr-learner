# Backlog: before publishing `drcurves` on PyPI

## Before anything else
- [ ] Confirm with Monzo in writing that publishing this is fine (IP clause).

## Done (phase 1)
- [x] Public API frozen in `drcurves.__all__`; Riesz/stacked weights moved to experimental `drcurves.riesz`.
- [x] Default fit ~2x faster (2-fold stacking, early-stopped boosting, 2-fold GPS residuals); `n_jobs`.
- [x] Input checks (lengths, finite/constant dose, missing group labels, tiny groups, poor overlap); NaN in X handled.
- [x] Fast test suite (~40 s; slow experimental test behind `-m slow`) with an end-to-end recovery test of the defaults.

## Features
- [ ] DiD option: `fit_dose_response(..., y_pre=...)` fits on the change in outcome; document that
      comparing across doses needs strong parallel trends.
- [ ] Lagged-outcome option (adjust for `y_pre` as a confounder), so users can bracket the two
      assumptions as in the DiD vs lagged-outcome article.
- [ ] `placebo_check(X, A, y_pre, y_prepre)`: curve on a pre-period change; judge it by the size of
      the curve (max deviation vs the main effect), not by whether its band excludes zero, which
      is far too sensitive.
- [ ] Add the integral term to the pointwise variance (Kennedy et al. 2017, as in npcausal's
      `ctseff`) so the intervals account for estimating m(a).
- [x] Use the stacked default inside `coverage_check` and `bootstrap_curves`.
- [ ] Discrete doses: with n=2,500 and rare top/bottom levels, `DiscreteGPS` weights blew the curve up at
      the edges (error 5.6 vs 0.2 at n=4,000). Stabilise (stacked classifier, pooling rare levels, or
      warning) before release.
- [ ] `.plot()` on the fitted model.

## Evidence
- [ ] Harder simulation: interactions, thresholds, heavy-tailed noise, weaker overlap. Compare the
      stacked default, boosting alone and npcausal, so the README claims hold beyond smooth data.
- [ ] DiD simulation with all three: levels, DiD, lagged outcome (stable trait vs selection on
      past outcomes).
- [ ] Rerun `sim4_uncertainty` and `sim6_intervals` with the stacked default; update README tables.

## Packaging
- [ ] CI (GitHub Actions): tests on Python 3.10–3.13, latest sklearn/pandas.
- [ ] Pin minimum versions in `pyproject.toml`; check optional deps (doubleml, causal-curve) are
      only needed for experiments.
- [ ] Version 0.1.0, CHANGELOG, tagged release; trusted publishing to PyPI from GitHub Actions.
- [ ] Faster tests (smaller n) so CI runs in a couple of minutes.

## Docs
- [ ] README: lead with what it's for (pick your assumption, get dose-response curves, many groups,
      many confounders) and when to use something else.
- [ ] Two worked examples (notebooks): one population curve; group curves with a hierarchy.
- [ ] Docstrings for every public function; a short API page.
- [ ] Link the TDS article once it's live.
