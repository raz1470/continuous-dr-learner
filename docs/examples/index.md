# Examples

Both notebooks run on simulated data with a known curve, so you can see how close the estimates get.
To use your own data, swap the simulation for your confounders `X`, dose `A` and outcome `Y`.

| Notebook | What it covers |
|---|---|
| [One curve](01_one_curve.ipynb) | Fit with the defaults, read the diagnostics, the band, a contrast and the optimal dose |
| [Curves for many groups](02_group_curves.ipynb) | A curve per store nested in regions: pooling, small stores, and a new store with no data |

The source notebooks are in
[`examples/`](https://github.com/raz1470/continuous-dr-learner/tree/main/examples).

```{toctree}
:hidden:

01_one_curve
02_group_curves
```
