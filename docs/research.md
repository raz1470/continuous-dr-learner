# Interval coverage with the stacked default

Monte Carlo coverage of 95% intervals (share containing the truth), all with `default_model()` for
every nuisance model. Older results with boosting alone are in `results/cache_boosted/`.

## One curve (`experiments/sim4_uncertainty.py single 42`; Sim 1 data, 5 confounders, n = 4,000)

| cov_type | pointwise | simultaneous | contrast | optimum | band_width | opt_width |
|---|---|---|---|---|---|---|
| bayes | 77% | 48% | 74% | 100% | 0.161 | 0.185 |
| robust_bayes | 77% | 45% | 74% | 100% | 0.155 | 0.172 |
| sandwich | 74% | 40% | 71% | 95% | 0.146 | 0.153 |

## 200 stores (`experiments/sim4_uncertainty.py stores 15`; Sim 3 data)

| cov_type | size_band | pointwise | simultaneous | contrast | optimum | band_width | opt_width |
|---|---|---|---|---|---|---|---|
| bayes | <25 | 81% | 65% | 85% | 88% | 0.823 | 0.826 |
| bayes | 25–60 | 81% | 66% | 85% | 90% | 0.736 | 0.767 |
| bayes | 60–150 | 82% | 67% | 87% | 90% | 0.594 | 0.663 |
| bayes | 150+ | 79% | 61% | 84% | 88% | 0.475 | 0.538 |
| robust_bayes | <25 | 81% | 64% | 84% | 88% | 0.789 | 0.804 |
| robust_bayes | 25–60 | 80% | 64% | 84% | 89% | 0.723 | 0.746 |
| robust_bayes | 60–150 | 80% | 61% | 85% | 89% | 0.585 | 0.649 |
| robust_bayes | 150+ | 78% | 58% | 84% | 88% | 0.466 | 0.526 |
| sandwich | <25 | 42% | 14% | 46% | 56% | 0.348 | 0.360 |
| sandwich | 25–60 | 52% | 21% | 56% | 66% | 0.416 | 0.407 |
| sandwich | 60–150 | 59% | 26% | 65% | 70% | 0.388 | 0.394 |
| sandwich | 150+ | 65% | 32% | 73% | 76% | 0.353 | 0.362 |
| bayes | all | 81% | 65% | 85% | 89% | 0.664 | 0.708 |
| robust_bayes | all | 80% | 62% | 84% | 89% | 0.650 | 0.691 |
| sandwich | all | 55% | 23% | 60% | 67% | 0.386 | 0.388 |

## Many confounders (`experiments/sim6_intervals.py 15`; Sim 5 data, n = 3,000)

| p | weights | reps | abs_bias | rmse | se_ratio | pointwise | simultaneous |
|---|---|---|---|---|---|---|---|
| 5 | density | 15 | 0.009 | 0.042 | 1.025 | 91% | 80% |
| 5 | balancing | 15 | 0.011 | 0.048 | 0.939 | 89% | 73% |
| 50 | density | 15 | 0.015 | 0.057 | 0.974 | 87% | 73% |
| 50 | balancing | 15 | 0.013 | 0.053 | 0.976 | 92% | 87% |

## Reading these

- With 50 confounders (Sim 5) the pointwise bands cover 87-92% and the standard errors match the real
  spread (se_ratio close to 1).
- With the strongly confounded Sim 1 data, pointwise coverage is 77%. The bands are 4.5x narrower than
  with boosting alone (0.16 vs 0.73), and the shortfall is mostly bias, not variance: a small tilt in
  the curve from capping the weights at their 99th percentile. Capping at the 99.9th percentile gave
  95% coverage there but blew up on the harder data, so the default stays at 0.99.
- Store curves cover about 81% pointwise.
- The variance term for estimating m(a) (`nuisance_var=True`, Kennedy et al. 2017) is included by
  default. With additive confounding it only affects the curve's level, so centred bands don't change.
- So: treat the default bands as slightly optimistic, and run `coverage_check` on your own data before
  relying on them for a decision.
