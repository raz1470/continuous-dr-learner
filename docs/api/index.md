# API reference

Everything below is importable from `drcurves`. Anything not listed here (`drcurves.riesz`,
`drcurves.simulate`, `prepare_X`) is experimental or internal and may change before 1.0.

| Step | Object |
|---|---|
| Fit the curves in one call | {func}`~drcurves.fit_dose_response` |
| Read, plot and compare the curves | {class}`~drcurves.HierarchicalDoseResponse` |
| Models and weights | {func}`~drcurves.default_model`, {class}`~drcurves.GPS`, {class}`~drcurves.DiscreteGPS`, {class}`~drcurves.BalancingWeights` |
| Check the bands and the assumptions | {func}`~drcurves.coverage_check`, {func}`~drcurves.bootstrap_curves`, {func}`~drcurves.placebo_check` |
| Build your own pipeline | {func}`~drcurves.cross_fit_nuisances`, {func}`~drcurves.pseudo_outcomes`, {class}`~drcurves.Nuisances` |

```{toctree}
:maxdepth: 1

fitting
models
checks
building_blocks
```
