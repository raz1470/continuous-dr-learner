# Backlog: before publishing `drcurves` on PyPI

## Before anything else
- [ ] Confirm with Monzo in writing that publishing this is fine (IP clause).

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
- [ ] Use the stacked default inside `coverage_check`, `bootstrap_curves` and the Riesz features
      (some still build their own boosted models).

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
