# User guide

Start with the [walkthrough](https://raz1470.github.io/continuous-dr-learner/walkthrough.html) for
the idea and the evidence, then the [examples](examples/index.md). These pages cover each decision.

| Page | What it covers |
|---|---|
| [Many groups](guide/many_groups.md) | A curve per store or product, pooled across a hierarchy; what `groups=` takes |
| [Choosing the assumption](guide/assumptions.md) | Measured confounders, difference-in-differences or the lagged outcome, and the placebo check |
| [Checking the bands](guide/checking_bands.md) | `coverage_check`: how often the bands contain the truth on data shaped like yours |
| [Options worth knowing](guide/options.md) | Models, weights, uncertainty, repeated units, inputs and speed |
| [When to use something else](guide/when_not.md) | Other tools, and the caveats |
| [How it works](guide/how_it_works.md) | The estimator in three steps, and how it compares with other packages |

```{toctree}
:hidden:

guide/many_groups
guide/assumptions
guide/checking_bands
guide/options
guide/when_not
guide/how_it_works
```
