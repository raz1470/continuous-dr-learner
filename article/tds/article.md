# Estimating Continuous Treatment Effects: Doubly Robust Dose-Response Curves

*A simulation study of where DoubleML's partially linear model falls short for continuous treatments, and a doubly robust alternative that estimates the whole curve.*

Not all treatments are binary: marketing spend, price, discount depth, app notifications, customer service response times. Rather than asking "did this treatment work?", we often need to know "what's the right amount?". When the treatment is continuous and chosen rather than randomised, a common choice is double machine learning ([Chernozhukov et al. 2018](https://arxiv.org/abs/1608.00060)), usually the partially linear model in [DoubleML](https://docs.doubleml.org/) ([Bach et al. 2022](https://jmlr.org/papers/v23/21-0862.html)). For a binary treatment, DoubleML's interactive model is doubly robust; for a continuous treatment, the partially linear model is the usual choice, and it has three limitations that matter in practice:

- **It returns one number.** The effect is assumed linear, so a rise-then-fall or a saturating curve comes back as a single slope.
- **Its default score isn't doubly robust.** With a wrong treatment model the estimate is biased, however good the outcome model is.
- **It gives one answer per population.** If you need a curve for each of 200 groups, looping over them gives noisy slopes for the small ones.

This article runs one simulation per limitation and shows an estimator that addresses all three: the doubly robust dose-response estimator of [Kennedy et al. (2017)](https://arxiv.org/abs/1507.00747), with a second stage that partially pools curves across a hierarchy. It then looks at three practical questions: which machine learning models to use, how existing packages compare, and what to do when a confounder is unmeasured.

## 1. One slope versus a whole curve

Let's start with the partially linear model. DoubleML's `DoubleMLPLR` fits

Y = θD + g(X) + ε

where D is the treatment, X the confounders and g(X) the effect of the confounders on the outcome. DoubleML estimates θ with two machine learning models: an **outcome model**, which predicts Y from X, and a **treatment model**, which predicts D from X. It then regresses what's left of Y on what's left of D (this is called partialling out).

Models like these are called nuisance models: we need them, but don't care about them for their own sake. Cross-fitting (each row's predictions come from models trained on other rows) and an orthogonal score (an estimating equation that is insensitive to small errors in the nuisance models) stop the regularisation in these models from biasing θ, which is what makes its confidence interval valid. The catch is the θD term: it assumes the effect is linear, so every extra unit of treatment has the same effect wherever you start.

If the true effect is a curve, you still get a single slope back: roughly the curve's average slope across the doses in your data.

The alternative is to estimate the whole curve: for every dose a, the average outcome if everyone received dose a. It takes three steps:

**Step 1: Cross-fit two nuisance models.** 

An outcome model μ(x, a) = E[Y | X, A]; unlike DoubleML's outcome model, this one includes the dose as an input. And a generalised propensity score (GPS; [Hirano and Imbens 2004](https://doi.org/10.1002/0470090456.ch7)) π(a | x): the density of the dose given X.

**Step 2: Build a pseudo-outcome.** 

Combine them into one number per row, ξ, whose average at each dose is the causal curve.

**Step 3: Smooth, with pooling.** 

Regress ξ on the dose with penalised splines ([Eilers and Marx 1996](https://doi.org/10.1214/ss/1038425655)), sharing strength across a hierarchy of groups.

Step 2 is where the robustness comes from. It starts with a regression estimate, the outcome model's prediction at dose a averaged over everyone, which on its own is an S-learner. It then adds a correction: the residuals of units that actually got a dose near a, weighted by how unusual that dose was for them so that they represent everyone. If the outcome model is right, the correction is just noise; if it's wrong, the correction removes the error, provided the model of who gets which dose is right. One of the two has to be right, not both: that's double robustness.1

Step 3 turns these noisy per-row numbers into a smooth curve and, when there are groups, lets small groups borrow strength from larger ones.

1 The pseudo-outcome is ξ = (Y − μ(X, A)) · ϖ(A)/π(A | X) + m(A), where μ is the outcome model, π the density of the dose given X, ϖ the overall density of the dose and m(a) the outcome model averaged over X at dose a (Kennedy et al. 2017). For a binary treatment this is the AIPW estimator behind DoubleML's `DoubleMLIRM`.

## 2. Nonlinear effects

Two nonlinear shapes are common:

- **Saturating.** Diminishing returns. The first £10k of ad spend in a region buys more conversions than the next £10k, and the 20th app notification this month adds almost nothing.
- **Inverted U.** Too much of a good thing. Deeper discounts win orders but give away margin, and a higher price raises revenue per unit until volume falls away.

In practice you rarely know the shape in advance, and the decision you need to make (how much, and where to stop) depends on it. So the method should find the shape rather than assume it.

The simulation:

- 4,000 units and five covariates.
- Three covariates drive both the dose (through nonlinear functions) and the outcome, so they confound the effect. A fourth changes how much the dose varies between units; the fifth does nothing.
- The true effect is an inverted U in one dataset and a saturating curve in the other.
- Both methods use the same learner for every nuisance model: a stack of gradient boosting and a spline regression (section 4 explains why).

[[IMAGE 1: fig1\_shape.png | caption: Each panel is one simulated dataset. Dashed black: the true dose-response curve. Blue: the DR curve, with a 95% band. Orange: DoubleML's estimate, a straight line with slope θ. Curves cover the middle 90% of observed doses and are centred, since only the shape is being compared. Image by author.]]

The DR curve follows both shapes. DoubleML's line can only tilt. On the inverted U its slope is close to zero: read on its own, it says the dose doesn't matter, when moving away from the middle in either direction costs a lot. On the saturating curve it gives every unit the same return, understating it at low doses and overstating it at high ones.

## 3. Robustness to a wrong nuisance model

Both nuisance models are approximations, and with a continuous dose the model of who gets which dose is especially easy to get wrong: it has to describe the whole spread of doses each unit might get, not just the average one. Doses are often set by unwritten business rules (pricing tiers, budgets split by last year's revenue, a cap on notifications), and some segments see a narrow range of doses while others see a wide one. The outcome model can be wrong too. So the test is what happens when one of them is wrong.

DoubleML's partially linear model has two scores (estimating equations). The default, partialling out, only protects against a wrong outcome model: if the treatment model is wrong, θ is biased however good the outcome model is ([Valentin et al. 2025](https://neurips.cc/virtual/2025/124095)). The alternative IV-type score is doubly robust, but only for the single slope θ. What we want is a doubly robust *curve*, which is what the pseudo-outcome from section 1 provides.

The simulation:

- 3,000 units and 50 confounders: four with strong nonlinear effects on both the dose and the outcome, and 46 weak ones.
- "Right" nuisance models are correctly specified. A "wrong" model treats every confounder as linear.
- The true effect is the inverted U.

Each dataset gets a correct and a wrong version of the outcome model and of the dose model. The S-learner uses the outcome model alone, so it shows what the wrong outcome model costs you.

[[IMAGE 2: fig2\_robust.png | caption: Each panel shows each estimator's error: its average estimated curve over 30 simulated datasets, minus the true curve. The dashed line at zero is the truth, so the closer to it the better. The blue band shows where 90% of the individual DR curves fall. Curves are centred. Image by author.]]

With the outcome model wrong, the S-learner misses the curve, while DR stays much closer, because the weighted residuals correct most of the error. With the dose model wrong, DR stays close. With both wrong, DR misses too: it needs at least one model right. DR's remaining error sits at the ends of the dose range, where the wrong outcome model is most wrong and the fewest units are available to correct it.

## 4. Choosing the nuisance models

Real data rarely has a handful of confounders. A customer table has dozens of features: tenure, spend history, products held, past contacts. Most of them nudge both the dose a customer gets and their outcome a little. You don't know which matter or how, so both nuisance models end up as flexible, regularised machine learning models. The choice of learner changed the answer a lot.

The simulation:

- The same data as section 3: 3,000 units and 50 confounders.
- Each estimator is fitted twice, changing only the machine learning model used for the nuisance models:
  - **Boosting alone:** gradient-boosted trees.
  - **A stack:** two models fitted side by side, boosted trees and a spline regression (a smooth curve in each confounder, like a GAM). Their predictions are combined as a weighted average, with the weights chosen by cross-validation.
- Three estimators: the S-learner, DR with density weights, and DR with balancing weights. Balancing weights skip the density: they are chosen so that, after weighting, the dose is uncorrelated with each confounder ([Tübbicke 2022](https://doi.org/10.1515/jem-2021-0002)).

[[IMAGE 3: fig3\_confounders.png | caption: Average estimated curve over 30 simulated datasets. Left: every nuisance model is boosted trees. Right: every nuisance model is the stack. Dashed black: the true curve. Curves are centred; the density-weight curve on the left runs off the chart. Image by author.]]

With boosting alone, the S-learner's curve is too flat, because boosted trees shrink the dose's effect. Density weights are far off: the boosted dose model gives some doses a near-zero probability, so their weights explode. Balancing weights, which need no density, do best. With the stack, all three recover the curve, because the spline regression captures the smooth relationships that trees can only approximate with steps.

In practice: with the stack, density weights work well; if you use boosting alone, use balancing weights instead. That doesn't make the stack better everywhere: section 6 tests it on harder data, where trees have the edge.

## 5. Many curves: borrow strength across the hierarchy

The decision is rarely one curve for the whole business. Pricing teams set discounts by category, marketing teams set budgets by country and channel, CRM teams set cadence by segment. The groups are nested (stores in regions, products in categories, campaigns in channels), and their sizes follow a long tail: a few big ones with plenty of data and many small ones with very little.

There are three ways to handle the groups:

- **Complete pooling:** one curve for everyone. Stable, but it ignores real differences between groups.
- **No pooling:** a separate curve per group. A store with 12 customers is treated as if it knew as much as one with 500. Looping DoubleML over stores does the same, and it also refits the nuisance models on each store's handful of rows.
- **Partial pooling:** in between ([Gelman and Hill 2007](https://doi.org/10.1017/CBO9780511790942)). Each store's curve is the global curve, plus a regional deviation, plus a store deviation:

θ store (a) = B(a)′ ( β global + b region + b store )

B(a) is a set of spline basis functions. A small store's deviation is shrunk toward its region, and a region's toward the global curve. A big store has enough data to overrule the shrinkage. How hard to shrink at each level is chosen from the data. Because the DR pseudo-outcome is just a number per row, the pooling sits on top of it unchanged: the DR step handles confounding, and the pooling handles the hierarchy.

The simulation:

- 200 stores in 8 regions. Each row is a customer who got a personalised discount (the dose); the outcome is their spend. Stores range from 8 to several hundred customers, with a long tail of small ones.
- Each store's true curve is the inverted U plus a regional and a store deviation.
- Each store has its own baseline, which raises both its doses and its outcomes, so there is store-level confounding.
- Four approaches: no pooling, complete pooling, partial pooling, and DoubleML fitted per store.

[[IMAGE 4: fig4\_many.png | caption: One simulated dataset. Root mean squared error of each store's estimated change in outcome between the 25th and 75th percentiles of the dose, grouped by the number of customers in the store. Image by author.]]

Without pooling, the error grows as stores get smaller. DoubleML per store is worst in all but the biggest stores. Even there, with 150 or more customers, its error stays around 0.7, because a straight line can't follow the curve; the extra error in small stores comes from refitting the nuisance models on a few dozen rows. Complete pooling's error stays high even in big stores, because every store gets the same curve. Partial pooling has the lowest error in small stores, because it pulls them toward their region; in big stores it matches no pooling, because the data overrules the shrinkage.

## 6. How does this compare with existing packages?

Two packages already estimate a single doubly robust curve with the pseudo-outcome from section 1:

- **[npcausal](https://github.com/ehkennedy/npcausal) (R):** Kennedy's own implementation. It fits the nuisance models with SuperLearner, an ensemble of smooth regressions and random forests, and chooses how much to smooth the curve from the data.
- **[causal-curve](https://github.com/ronikobrosly/causal-curve) (Python):** follows the same steps (its estimator is called TMLE), with gradient boosting and a fixed amount of smoothing.

Neither cross-fits the nuisance models, and neither handles groups. Below, we compare them with [drcurves](https://github.com/raz1470/continuous-dr-learner), the open-source Python package written for this article, which does both.

We ran all three on three settings, ten simulated datasets each:

- The data from section 2 (5 confounders) and section 3 (50 confounders). These are the simulations used to choose drcurves' default learner in section 4, so they favour it.
- Harder data that favours trees, added as a check: thresholds, interactions between confounders and heavy-tailed noise.

Each package used its defaults, with two exceptions for npcausal: we gave it a range of bandwidths to choose from, and with 50 confounders we dropped the pairwise-interaction learner from its library, which didn't fit in memory.

[[IMAGE 5: fig5\_packages.png | caption: Root mean squared error between the estimated and the true curve, averaged over the simulated datasets in each setting. Curves are centred. On the harder data, npcausal failed on 1 of the 10 datasets, so all three are compared on the other 9. Image by author.]]

- **Smooth data (5 and 50 confounders):** drcurves is the most accurate, and its lead grows with more confounders. npcausal is close with 5 confounders but falls behind with 50, likely because of its library of learners rather than the cross-fitting (section 4 showed how much the learner matters).
- **Harder data:** drcurves is still ahead of npcausal. Within drcurves, boosting alone with balancing weights was more accurate still (0.17 vs 0.24), but boosting alone with density weights failed badly (2.2). No learner wins everywhere; the stack is rarely far from the best.
- **causal-curve** trails throughout, and its intervals are 10 to 60 times wider than the others', so they always cover the truth but tell you little.

## 7. When confounders are unmeasured: difference-in-differences

Everything so far assumes you've measured the confounders. Often you haven't. A customer's underlying complexity, a team's culture, a store manager's skill: stable traits that push both the dose and the outcome, and never appear in the data. No estimator fixes that, but with pre-period data a different assumption can.

If you observe the outcome before the dose changed, and the hidden trait is stable over time, difference-in-differences removes it: fit the same method on the *change* in outcome rather than its level. To compare doses, rather than just treated against untreated, this needs a stronger form of parallel trends: units on different doses would have followed the same path had their doses been the same ([Callaway, Goodman-Bacon and Sant'Anna 2024](https://arxiv.org/abs/2107.02637)).

The simulation:

- 3,000 units, five observed confounders and one hidden, stable trait that raises both the dose and the outcome.
- Two periods before the dose, one after. The true effect is the inverted U.
- Three ways to use the same DR estimator: adjust for X; adjust for X and the pre-period outcome; or fit on the change in outcome (DiD).

[[IMAGE 6: fig6\_did.png | caption: Root mean squared error between the estimated and the true curve, averaged over 10 simulated datasets. Curves are centred. Image by author.]]

Adjusting for X alone leaves the trait confounding the curve. Adding the pre-period outcome helps, because it partly stands in for the trait, but the curve is still biased. Fitting on the change recovers it, because the trait cancels.

The ranking flips if the dose responds to past outcomes rather than to a stable trait: then adjusting for the pre-period outcome is right and DiD is biased. You can't tell which from the data alone. A placebo helps: fit the curve on the change between two pre-periods, where the dose can't have had an effect yet. A clearly non-flat curve warns that parallel trends doesn't hold. Judge it by its size relative to the effect you care about, not by whether its band includes zero: even when parallel trends held in our simulation, about 30% of the placebo curve fell outside its 95% band.

## Frequently asked questions

### Can I trust the intervals?

Treat them as optimistic. With the stacked default, the 95% bands contained the true curve 75-93% of the time across our simulations. Capping the most extreme weights, which trades a small bias for a lot of stability, is one likely reason. When the nuisance models are off the bands are too narrow, because a band doesn't account for bias. Check on your own data: keep your real confounders and doses, simulate an outcome with a known curve, rerun the whole pipeline 20 to 30 times, and count how often the band contains it. This is a plasmode simulation ([Franklin et al. 2014](https://doi.org/10.1016/j.csda.2013.10.018)), and if the bands fall short it tells you how much to widen them.

### When shouldn't I use this?

- **Unmeasured confounding.** Double robustness protects against a wrong model, not a missing variable.
- **Poor overlap.** If some units only ever get low doses, the data can't say what high doses would do for them. Report the curve only over doses you actually see.
- **Dose set at the group level.** If every customer in a store gets the store's discount, the store is the unit of analysis. Aggregate first.
- **A spike at zero.** "Any versus none" and "how much" are different questions; analyse them separately.
- **Crossed groups.** The pooling assumes nesting (stores within regions). Store × channel needs a different model.

### Why not bin the treatment and use a discrete estimator?

DoubleML's `DoubleMLAPOS` estimates average potential outcomes for discrete treatment levels, so binning the dose is an option. It throws away the ordering of the doses, each bin needs its own propensity model and enough data, and the answer depends on where you put the cut points. Within a bin, units with different doses are treated as if they got the same one.

### What about a polynomial in the treatment, as in EconML?

EconML's DML estimators accept a treatment featurizer, so you can fit θ₁D + θ₂D². That handles a simple curve, but you choose the shape in advance, it is still a partially linear model with the same score, and it does not pool across groups.

### Is there a kernel version of double machine learning for this?

Yes. [Colangelo and Lee (2020)](https://arxiv.org/abs/2004.03036) estimate the curve one dose at a time with a kernel-localised doubly robust score, and [Klosin (2021)](https://arxiv.org/abs/2104.10334) does it without estimating the density. Both are good options for a single population curve with careful inference. The pseudo-outcome approach here was chosen because the second stage can be any regression, which is what makes the hierarchical pooling simple.

### Could I do the pooling in a Bayesian model instead?

Yes. The ridge penalties are Gaussian priors on the deviations, tuned by cross-validation rather than given hyperpriors. Fitting the same spline structure in PyMC or brms on the pseudo-outcomes gives full posteriors for each store's curve, at more computational cost.

## Conclusion

With a continuous treatment, the partially linear model answers a narrower question than you usually need. The question is about a curve:

- **Shape.** A single slope can't describe a curve. On an inverted U it can come out near zero, as if the dose didn't matter. Estimate the whole curve instead.
- **Robustness.** With one nuisance model wrong, the DR curve removed most of the error the S-learner made, though some remained at the ends of the dose range.
- **Learners.** Boosting alone flattens the curve and breaks density weights. Stack a smooth learner with boosting; no single learner wins everywhere.
- **Many curves.** Partial pooling gives every group a curve, shrinks the small ones toward their parent, and leaves big ones close to their own data.
- **Assumptions.** If a stable trait you can't measure drives both dose and outcome, fit the same method on the change in outcome (DiD), and run a placebo on the pre-periods.

The simulations use one data-generating process per question, known confounders and moderate sample sizes. They do not show how often each problem occurs in your data. The questions to ask are the same, though: is the curve likely to bend, which nuisance model are you least sure of, how many groups need their own answer, and what are you assuming about the confounders you can't see?

The package, [drcurves](https://github.com/raz1470/continuous-dr-learner), has the estimator, the simulations and the script that builds every chart and number on this page.

### References

- Bach, P., Chernozhukov, V., Kurz, M. S. and Spindler, M. (2022). [DoubleML: an object-oriented implementation of double machine learning in Python](https://jmlr.org/papers/v23/21-0862.html). *Journal of Machine Learning Research*, 23(53), 1–6.
- Chernozhukov, V. et al. (2018). [Double/debiased machine learning for treatment and structural parameters](https://arxiv.org/abs/1608.00060). *The Econometrics Journal*, 21(1), C1–C68.
- Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024). [Difference-in-differences with a continuous treatment](https://arxiv.org/abs/2107.02637). arXiv:2107.02637.
- Colangelo, K. and Lee, Y.-Y. (2020). [Double debiased machine learning nonparametric inference with continuous treatments](https://arxiv.org/abs/2004.03036). arXiv:2004.03036.
- Eilers, P. and Marx, B. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.
- Franklin, J. M., Schneeweiss, S., Polinski, J. M. and Rassen, J. A. (2014). [Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases](https://doi.org/10.1016/j.csda.2013.10.018). *Computational Statistics & Data Analysis*, 72, 219–226.
- Gelman, A. and Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Hirano, K. and Imbens, G. (2004). The propensity score with continuous treatments. In *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*, Wiley.
- Kennedy, E., Ma, Z., McHugh, M. and Small, D. (2017). [Non-parametric methods for doubly robust estimation of continuous treatment effects](https://arxiv.org/abs/1507.00747). *Journal of the Royal Statistical Society: Series B*, 79(4), 1229–1245.
- Klosin, S. (2021). [Automatic double machine learning for continuous treatment effects](https://arxiv.org/abs/2104.10334). arXiv:2104.10334.
- Tübbicke, S. (2022). [Entropy balancing for continuous treatments](https://doi.org/10.1515/jem-2021-0002). *Journal of Econometric Methods*, 11(1), 71–89.
- Valentin, S., Detommaso, G., Li, Y., Opper, M. and Brueckner, M. (2025). [On double robustness in double machine learning](https://neurips.cc/virtual/2025/124095). NeurIPS 2025 Workshop CauScien.