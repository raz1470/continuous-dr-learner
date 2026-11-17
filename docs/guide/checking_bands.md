# Checking the bands on your data

In our simulations the default 95% bands contained the true curve 75–93% of the time: a little
short, mostly because capping the weights leaves a small bias. On your data they could be better or
worse. You can't measure that from the data alone, because the problem is bias and bias needs a known
truth.

{func}`~drcurves.coverage_check` gets one. It keeps your real confounders and doses, replaces the
outcome with a simulated one whose curve is known, reruns the whole pipeline and counts how often the
band contains the truth. This is a plasmode simulation (Franklin et al. 2014). It's the same idea as
injecting a known effect into real data to check that a pipeline gets it back.

```python
from drcurves import coverage_check, fit_dose_response

summary, reps = coverage_check(X, A, Y, n_reps=30)
summary[["weights", "pointwise", "simultaneous", "se_ratio", "abs_bias",
         "critical_pointwise", "critical_simultaneous"]]
```

How to read it:

- **`pointwise`, `simultaneous`**: the share of 95% bands that contained the true curve. Near 0.95
  is good.
- **`se_ratio`**: reported standard error divided by the actual spread across runs. Below 1 means the
  bands are overconfident.
- **`abs_bias`**: how far the curves sit from the truth on average. If coverage is short and
  `se_ratio` is close to 1, bias is the reason.
- **`critical_pointwise`, `critical_simultaneous`**: the multipliers that gave 95% coverage on your
  data. Pass one to `band` for calibrated bands:

```python
model = fit_dose_response(X, A, Y)
crit = summary.loc[summary.weights == "density", "critical_simultaneous"].item()
est, lo, hi, _ = model.band(grid, critical=crit)
```

Use at least 30 replications for anything you report; with fewer, the critical values are noisy.
With groups, pass `groups=` and every group's curve is scored against the common true shape.

The check is optimistic. The simulated outcome comes from your own fitted model, so it's probably
easier to fit than reality, and it assumes no unmeasured confounding. Treat the calibrated band as a
minimum width.

The full coverage studies behind these numbers are on the [research page](../research.md).
