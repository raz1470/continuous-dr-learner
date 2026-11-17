# drcurves

**Doubly robust dose-response curves for continuous treatments, for one population or many groups.**

How does the outcome change as the dose changes: discount depth and margin, handle time and
satisfaction, notifications per week and retention? `drcurves` estimates the whole curve, not one
slope, from observational data.

- **Doubly robust.** The curve stays close to the truth if either the outcome model or the dose
  model is right (Kennedy et al. 2017).
- **Many curves at once.** A curve per store, product or customer issue, partially pooled across a
  hierarchy (region → store), so small groups borrow strength from their parents.
- **Choose your assumption.** Adjust for measured confounders, use difference-in-differences when a
  stable trait you can't measure drives the dose, or adjust for the lagged outcome when doses respond
  to past outcomes.
- **Sensible defaults.** Every nuisance model is a stack of a spline regression and gradient boosting.

```bash
pip install drcurves
pip install "drcurves[plot]"    # adds matplotlib for model.plot()
```

```python
import numpy as np
from drcurves import fit_dose_response
from drcurves.simulate import sim_nonlinear

X, A, Y, truth = sim_nonlinear(n=3000, curve="inverted_u_centred", seed=1)

model = fit_dose_response(X, A, Y)
print(model.diagnostics())                  # overlap and fit checks: read these first

grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 50)
est, lo, hi, _ = model.band(grid)           # curve and 95% band
best = model.optimal_dose(grid)             # dose that maximises the outcome, with an interval
model.plot(truth=truth)
```

## Where to start

- **[The walkthrough](https://raz1470.github.io/continuous-dr-learner/walkthrough.html)**: the
  article. Why DoubleML's partially linear model falls short for a continuous dose, and the method
  step by step, checked on simulated data.
- **[The user guide](user_guide.md)**: groups, choosing the assumption, checking the bands, and when
  to use something else.
- **[Examples](examples/index.md)**: one curve, and curves for many groups, as notebooks.
- **[Research](research.md)**: how often the 95% bands contain the true curve.
- **[API reference](api/index.md)**: every public function and class.

```{toctree}
:hidden:

user_guide
examples/index
research
api/index
```
