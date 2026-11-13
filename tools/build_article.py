"""Build article/index.html from the results/ folder. Every number and chart in
the article comes from here, so re-running the experiments and this script
regenerates the whole page."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg import funnel_panel, hist_panel, legend, line_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CSS = (ROOT / "tools" / "article.css").read_text()
REPO = "https://github.com/raz1470/continuous-dr-learner"

BLUE, ORANGE, AQUA, YELLOW, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#111827", "#6b7280"


def f2(v, sign=False):
    if abs(v) < 0.005:
        return "0.00"
    return f"{v:+.2f}".replace("-", "−") if sign else f"{v:.2f}".replace("-", "−")


def pct(v):
    return f"{100 * v:.0f}%"


# ======================================================================= data
s1 = json.loads((RES / "sim1_curves.json").read_text())
s1tab = pd.read_csv(RES / "sim1_nonlinear.csv").set_index("curve")
s2plr = pd.read_csv(RES / "sim2_plr_summary.csv")
s2c = np.load(RES / "sim2_curves.npz")
s3 = json.loads((RES / "sim3_curves.json").read_text())
s3size = pd.read_csv(RES / "sim3_summary_by_size.csv")
s3rmse = pd.read_csv(RES / "sim3_store_rmse.csv")
s3plr = pd.read_csv(RES / "sim3_plr_per_store.csv")
N_REPS2 = int(s2plr.reps.min())
N_REPS3 = int(s3rmse.rep.nunique())
s4a = pd.read_csv(RES / "sim4_single_summary.csv").set_index("cov_type")
s4b = pd.read_csv(RES / "sim4_stores_summary.csv").set_index(["cov_type", "size_band"])
N_REPS4A = int(pd.read_csv(RES / "sim4_single_reps.csv").rep.nunique())
s6 = pd.read_csv(RES / "sim6_intervals_summary.csv").set_index(["p", "weights"])

N_REPS4B = int(pd.read_csv(RES / "sim4_stores_reps.csv").rep.nunique())


def covcls(v):
    return "good" if v >= 0.92 else ("bad" if v < 0.8 else "")


cov_rows = []
for label, ct, src in [("One curve: default (Bayesian)", "bayes", "a"),
                       ("One curve: plain sandwich", "sandwich", "a"),
                       ("200 stores: default (Bayesian)", "bayes", "b"),
                       ("200 stores: plain sandwich", "sandwich", "b")]:
    r = s4a.loc[ct] if src == "a" else s4b.loc[(ct, "all")]
    cells = "".join(f"<td class='num {covcls(r[c])}'>{pct(r[c])}</td>"
                    for c in ["pointwise", "simultaneous", "contrast", "optimum"])
    cov_rows.append(f"<tr><td>{label}</td>{cells}</tr>")
table4 = ("<table class='dtable'><caption>Share of 95% intervals that contain the truth. One curve: "
          f"{N_REPS4A} replications of the Sim 1 data. Stores: {N_REPS4B} replications of the Sim 3 data, "
          "averaged over all 200 stores.</caption><thead><tr><th>Setting</th><th class='num'>Shape: pointwise band</th>"
          "<th class='num'>Shape: simultaneous band</th><th class='num'>Dose contrast</th>"
          "<th class='num'>Optimal dose</th></tr></thead><tbody>" + "".join(cov_rows) + "</tbody></table>")
big4 = s4b.loc[("bayes", "150+")]
small4 = s4b.loc[("bayes", "<25")]


def p2(score, om, gm, col="bias"):
    return float(s2plr.query("score==@score and outcome==@om and treatment_model==@gm")[col].iloc[0])


# ===================================================================== sim 1
pol = s1["policy_shift"]
sim1_panels = []
for key, title in [("inverted_u", "Inverted-U effect"), ("saturating", "Saturating effect")]:
    d = s1[key]
    g, dr, se = np.array(d["grid"]), np.array(d["dr"]), np.array(d["dr_se"])
    sub = f"DoubleML θ = {d['theta']:.2f} (s.e. {d['theta_se']:.2f})"
    sim1_panels.append(line_panel(g, [
        {"y": dr, "lo": dr - 1.96 * se, "hi": dr + 1.96 * se, "color": BLUE},
        {"y": d["plr"], "color": ORANGE},
        {"y": d["truth"], "color": INK, "dash": "5 4", "width": 2},
    ], title, sub))

# ===================================================================== sim 2
g2, t2 = s2c["grid"], s2c["truth"]
SC = [("right", "right", "Both models right"), ("wrong", "right", "Outcome model wrong"),
      ("right", "wrong", "Dose model wrong"), ("wrong", "wrong", "Both wrong")]
ylim2 = (t2.min() - 1.0, t2.max() + 0.8)
errs = {k: s2c[k] - t2 for k in s2c.files if k.startswith("A|")}
lim_e = float(np.ceil(max(np.abs(v.mean(0)).max() for v in errs.values()) * 1.25 * 10) / 10)
sim2_panels = []
for om, dm, title in SC:
    ser = [{"y": np.zeros_like(t2), "color": INK, "dash": "5 4", "width": 2}]
    for est, col in [("plugin", YELLOW), ("ipw", AQUA), ("dr", BLUE)]:
        e = errs[f"A|{om}|{dm}|{est}"]
        item = {"y": e.mean(0), "color": col, "width": 2.6 if est == "dr" else 1.8}
        if est == "dr":
            item["lo"], item["hi"] = np.percentile(e, [5, 95], axis=0)
        ser.append(item)
    sim2_panels.append(line_panel(g2, ser, title, "", ylim=(-lim_e, lim_e)))

bal = s2c["B|dr_balancing"]
blo, bhi = np.percentile(bal, [5, 95], axis=0)
practice_panel = line_panel(g2, [
    {"y": s2c["B|slearner"].mean(0), "color": YELLOW, "width": 2},
    {"y": s2c["B|dr_density"].mean(0), "color": ORANGE, "width": 2},
    {"y": bal.mean(0), "lo": blo, "hi": bhi, "color": BLUE, "width": 2.6},
    {"y": t2, "color": INK, "dash": "5 4", "width": 2},
], "", "", ylim=ylim2, w=680, h=280)

cov_po = p2("partialling out", "right", "wrong", "coverage")
plr_cov_text = "never contained" if cov_po == 0 else f"contained only {pct(cov_po)} of the time"

# ===================================================================== sim 3
grid3 = np.array(s3["grid"])
stores = sorted(s3["stores"], key=lambda d: d["n"])
pick_idx = [0, len(stores) - 1]
sim3_panels = []
allv = np.concatenate([np.r_[d["truth"], d["partial"]] for d in stores])
ylim3 = (np.percentile(allv, 0.5), np.percentile(allv, 99.5))
for i in pick_idx:
    d = stores[i]
    p, se = np.array(d["partial"]), np.array(d["partial_se"])
    sim3_panels.append(line_panel(grid3, [
        {"y": d["none"], "color": ORANGE, "width": 1.6},
        {"y": d["complete"], "color": GREY, "width": 1.6},
        {"y": p, "lo": p - 1.96 * se, "hi": p + 1.96 * se, "color": BLUE, "width": 2.6},
        {"y": d["truth"], "color": INK, "dash": "5 4", "width": 2},
    ], f"{d['store']} · {d['region']}", f"n = {d['n']}", ylim=ylim3))

overall3 = s3rmse.groupby("pooling").rmse.mean()
small = s3rmse[s3rmse.n < 25].groupby("pooling").rmse.mean()
big = s3rmse[s3rmse.n >= 150].groupby("pooling").rmse.mean()

# Funnel: error in each store's dose effect, theta_s(a_hi) - theta_s(a_lo), by store size.
L3 = len(grid3)
i_lo, i_hi = round(0.25 * (L3 - 1)), round(0.75 * (L3 - 1))
a_lo, a_hi = grid3[i_lo], grid3[i_hi]
st_n = np.array([d["n"] for d in stores])
true_eff = np.array([d["truth"][i_hi] - d["truth"][i_lo] for d in stores])
eff_err = {p: np.array([d[p][i_hi] - d[p][i_lo] for d in stores]) - true_eff
           for p in ["partial", "none", "complete"]}
plr_theta = s3plr.set_index("store").theta
eff_err["doubleml"] = np.array([plr_theta.get(d["store"], np.nan) for d in stores]) * (a_hi - a_lo) - true_eff
lim3 = float(np.ceil(np.nanpercentile(np.abs(eff_err["none"]), 99) * 2) / 2)


def rmse_by(err, mask):
    e = err[mask & np.isfinite(err)]
    return float(np.sqrt(np.mean(e ** 2)))


small_m, big_m = st_n < 25, st_n >= 150
funnel = []
for key, title, col in [("partial", "Partial pooling", BLUE), ("none", "No pooling", ORANGE),
                        ("complete", "Complete pooling", GREY), ("doubleml", "DoubleML per store", YELLOW)]:
    sub = f"RMSE {rmse_by(eff_err[key], small_m):.2f} in stores under 25 obs, {rmse_by(eff_err[key], big_m):.2f} at 150+"
    funnel.append(funnel_panel(st_n, eff_err[key], title, sub, col, ylim=(-lim3, lim3)))
n_fail = int(np.isnan(eff_err["doubleml"]).sum())

# ===================================================================== sim 4 example
eb = json.loads((RES / "sim4_example_band.json").read_text())
gE = np.array(eb["grid"])
est_e = np.array(eb["est"])
zero = np.zeros_like(est_e)
band_panel = line_panel(gE, [
    {"y": zero, "lo": np.array(eb["lo_s"]) - est_e, "hi": np.array(eb["hi_s"]) - est_e, "color": BLUE, "width": 0, "opacity": 0},
    {"y": zero, "lo": np.array(eb["lo_p"]) - est_e, "hi": np.array(eb["hi_p"]) - est_e, "color": BLUE, "width": 2},
    {"y": np.array(eb["truth"]) - est_e, "color": INK, "dash": "5 4", "width": 2},
], "Is the true curve inside the bands?",
   f"Everything relative to the estimate (the blue line at 0). Inner band: each dose, ±{eb['z_p']:.2f} s.e. Outer: whole curve, ±{eb['z_s']:.2f} s.e.")
op = eb["opt"]
peak_panel = hist_panel(op["draws"], "Where the peak is",
                        f"Estimate {f2(op['dose'])}, 95% interval {f2(op['lo'])} to {f2(op['hi'])}",
                        bins=28, color=BLUE, shade=(op["lo"], op["hi"]),
                        marks=[(eb["true_peak"], "true peak", "5 4")], xlabel="Dose at which the curve peaks")
ct = eb["contrast"]
LEG4 = legend([("True curve", INK, "5 4"), ("Estimate and its 95% bands", BLUE, "")])

explorer_data = json.dumps({"grid": s3["grid"], "stores": [
    {k: d[k] for k in ["store", "region", "n", "truth", "none", "complete", "partial", "partial_se"]} for d in stores]})

# =================================================================== article
LEG1 = legend([("True curve", INK, "5 4"), ("DR curve (95% band)", BLUE, ""), ("DoubleML PLR", ORANGE, "")])
LEG2 = legend([("Truth (zero error)", INK, "5 4"), ("S-learner", YELLOW, ""), ("Weighting only", AQUA, ""),
               ("Doubly robust", BLUE, "")])
LEG5 = legend([("True curve", INK, "5 4"), ("S-learner", YELLOW, ""), ("DR, density weights", ORANGE, ""),
               ("DR, balancing weights", BLUE, "")])
LEG3 = legend([("True store curve", INK, "5 4"), ("No pooling", ORANGE, ""), ("Complete pooling", GREY, ""),
               ("Partial pooling (95% band)", BLUE, "")])

inv = s1tab.loc["inverted_u"]
sat = s1tab.loc["saturating"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Continuous Treatments Need Curves, Not Coefficients by Ryan O'Sullivan</title>
<style>
{CSS}
select {{ font: inherit; font-size: .85rem; padding: .3rem .5rem; border: 1px solid var(--border); border-radius: 6px; background: #fff; }}
.flow {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 1.5rem 0; }}
@media (max-width: 599px) {{ .flow {{ grid-template-columns: 1fr; }} }}
.step {{ border: 1px solid var(--border); border-radius: 8px; padding: .9rem 1rem; background: var(--bg); }}
.step .n {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
.step .t {{ font-weight: 700; margin: .15rem 0 .35rem; }}
.step p {{ font-size: .86rem; margin: 0; line-height: 1.5; }}
</style>
</head>
<body>

<header class="hdr">
  <h1>Continuous Treatments Need Curves, Not Coefficients</h1>
  <p class="tagline">Where double machine learning's partially linear model falls short for continuous treatments, a doubly robust and partially pooled alternative, and how far to trust its intervals.</p>
  <p class="byline">Ryan O'Sullivan · Code on <a href="{REPO}" style="color:#cbd5e1">GitHub</a></p>
</header>

<nav class="snav">
  <a href="#plr" class="active">1 · One slope vs a curve</a>
  <a href="#shape">2 · Shape</a>
  <a href="#robust">3 · Robustness</a>
  <a href="#many">4 · Many curves</a>
  <a href="#method">5 · In practice</a>
  <a href="#limits">Limits</a>
  <a href="#faq">Questions</a>
  <a href="#together">Conclusion</a>
</nav>

<main>

<p>Not all treatments are binary: marketing spend, price, discount depth, app notifications, customer service response times. Rather than asking "did this treatment work?", we often need to know "what's the right amount?". When the treatment is continuous and chosen rather than randomised, most analysts reach for double machine learning, usually the partially linear model in <a href="https://docs.doubleml.org/" target="_blank" rel="noopener">DoubleML</a>. It is a good default for a binary treatment, but for a continuous one it has three limitations that matter in practice:</p>

<ul class="pts">
  <li><strong>It returns one number.</strong> The effect is assumed linear, so a rise-then-fall or a saturating curve comes back as a single slope.</li>
  <li><strong>Its default score isn't doubly robust.</strong> With a wrong treatment model the estimate is biased, however good the outcome model is.</li>
  <li><strong>It gives one answer per population.</strong> If you need a curve for each of 200 stores, looping over them gives noisy curves for the small ones.</li>
</ul>

<p>This article runs one simulation per limitation and shows an estimator that addresses all three: the doubly robust dose-response estimator of <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Kennedy et al. (2017)</a>, with a second stage that partially pools curves across a hierarchy.</p>

<!-- 1 -->
<section id="plr">
  <h2>1. One slope versus a whole curve</h2>

  <p>Let's start with the partially linear model. DoubleML's <code>DoubleMLPLR</code> fits</p>

  <div class="eqn">Y = θD + g(X) + ε</div>

  <p>where D is the treatment, X the confounders and g(X) is learned with any machine learning model. Models like g are called nuisance models: we need them, but don't care about them for their own sake. Cross-fitting (each row's predictions come from models trained on other rows) and an orthogonal score (an estimating equation that is insensitive to small errors in the nuisance models) stop the regularisation in g from biasing θ, which is what makes its confidence interval valid. The catch is the θD term: it assumes the effect is linear, so every extra unit of treatment has the same effect wherever you start.</p>

  <p>If the true effect is a curve, you still get a single slope back: roughly the curve's average slope across the doses in your data.</p>

  <p>The alternative is to estimate the whole curve: for every dose a, the average outcome if everyone received dose a. It takes three steps:</p>

  <div class="flow">
    <div class="step"><div class="n">Step 1</div><div class="t">Cross-fit two nuisance models</div><p>An outcome model μ(x, a) = E[Y | X, A], and a generalised propensity score (GPS) π(a | x): the density of the dose given X.</p></div>
    <div class="step"><div class="n">Step 2</div><div class="t">Build a pseudo-outcome</div><p>Combine them into one number per row, ξ, whose average at each dose is the causal curve.</p></div>
    <div class="step"><div class="n">Step 3</div><div class="t">Smooth, with pooling</div><p>Regress ξ on the dose with penalised splines, sharing strength across a hierarchy of groups.</p></div>
  </div>

  <p>Step 2 is where the robustness comes from. It starts with a regression estimate, the outcome model's prediction at dose a averaged over everyone, which on its own is an S-learner. It then adds a correction: the residuals of units that actually got a dose near a, weighted by how unusual that dose was for them so that they represent everyone. If the outcome model is right, the correction is just noise; if it's wrong, the correction removes the error, provided the model of who gets which dose is right. One of the two has to be right, not both: that's double robustness.<sup>1</sup></p>

  <p>Step 3 turns these noisy per-row numbers into a smooth curve and, when there are groups, lets small groups borrow strength from larger ones (section 4).</p>

  <p class="fig-cap" style="padding:0"><sup>1</sup> The pseudo-outcome is ξ = (Y − μ(X, A)) · ϖ(A)/π(A | X) + m(A), where μ is the outcome model, π the density of the dose given X, ϖ the overall density of the dose and m(a) the outcome model averaged over X at dose a (Kennedy et al. 2017). For a binary treatment this is the AIPW estimator behind DoubleML's <code>DoubleMLIRM</code>.</p>
</section>

<!-- 2 -->
<section id="shape">
  <p class="s-label">Simulation 1</p>
  <h2>2. Shape: one slope cannot describe a curve</h2>

  <p>Nonlinear dose-response curves are common. Two shapes come up again and again:</p>

  <ul class="pts">
    <li><strong>Saturating.</strong> Diminishing returns. The first £10k of ad spend in a region buys more conversions than the next £10k, and the 20th app notification this month adds almost nothing.</li>
    <li><strong>Inverted U.</strong> Too much of a good thing. Deeper discounts win orders but give away margin, and a higher price raises revenue per unit until volume falls away.</li>
  </ul>

  <p>In practice you rarely know the shape in advance, and the decision you need to make (how much, and where to stop) depends on it. So the method should find the shape rather than assume it.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>4,000 units and five covariates.</li>
    <li>Three covariates drive both the dose (through nonlinear functions) and the outcome, so they confound the effect. A fourth changes how much the dose varies between units; the fifth does nothing.</li>
    <li>The true effect is an inverted U in one dataset and a saturating curve in the other.</li>
    <li>Both methods use the same gradient-boosted models for every nuisance model.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimated dose-response curves: DR vs DoubleML</div>
    </div>
    <div class="fig-body">{LEG1}<div class="panels">{''.join(sim1_panels)}</div></div>
    <p class="fig-cap">Each panel is one simulated dataset. Dashed black: the true dose-response curve. Blue: the DR curve, with a 95% band. Orange: DoubleML's estimate, a straight line with slope θ. Curves cover the middle 90% of observed doses and are centred, since only the shape is being compared.</p>
  </div>

  <p>The DR curve follows both shapes. DoubleML's line can only tilt. On the inverted U its slope is close to zero: read on its own, it says the dose doesn't matter, when moving away from the middle in either direction costs a lot. On the saturating curve it gives every unit the same return, understating it at low doses and overstating it at high ones.</p>
</section>

<!-- 3 -->
<section id="robust">
  <p class="s-label">Simulation 2</p>
  <h2>3. Robustness to a wrong nuisance model</h2>

  <p>Every observational estimate leans on models that are approximations. With a continuous dose, the model of who gets which dose is especially easy to get wrong. Doses are often set by unwritten business rules (pricing tiers, budgets split by last year's revenue, a cap on notifications), and some segments see a narrow range of doses while others see a wide one. The outcome model can be wrong too. So the useful question isn't whether your models are right, but what happens when one of them is wrong.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>3,000 units and 50 confounders: four with strong nonlinear effects on both the dose and the outcome, and 46 weak ones with small effects on both.</li>
    <li>"Right" nuisance models are correctly specified. A "wrong" model gets the confounding wrong: it treats every confounder as linear.</li>
    <li>For DoubleML the true effect is linear, so θ is well defined. For the curve it is the inverted U.</li>
  </ul>

  <p>DoubleML's partially linear model offers two scores. The default, partialling out (Robinson's estimator), only protects against errors in the outcome model: if the treatment model is wrong, θ is biased however good the outcome model is (<a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">Valentin et al. 2025</a>). In our simulation with a wrong treatment model, the default score's 95% intervals {plr_cov_text} the true θ. The IV-type score is doubly robust, but only for θ, the single slope from section 2. What we want is a doubly robust <em>curve</em>, which is what the pseudo-outcome from section 1 provides.</p>

  <h3>The curve</h3>

  <p>Each dataset gets a correct and a wrong version of the outcome model and of the dose model, combined four ways, and three estimators are built from the same pieces:</p>

  <ul class="pts">
    <li><strong>S-learner:</strong> the outcome model's predictions, averaged over everyone. Needs the outcome model right.</li>
    <li><strong>Weighting only:</strong> the outcome reweighted by how unusual each unit's dose was, with no outcome model. Needs the dose model right.</li>
    <li><strong>Doubly robust:</strong> the pseudo-outcome. Needs one of the two right.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimation error with right and wrong nuisance models</div>
    </div>
    <div class="fig-body">{LEG2}<div class="panels">{''.join(sim2_panels)}</div></div>
    <p class="fig-cap">Each panel combines a right or wrong outcome model with a right or wrong dose model and shows each estimator's error: its average estimated curve over {N_REPS2} simulated datasets, minus the true curve. The dashed line at zero is the truth, so the closer to it the better. The blue band shows where 90% of the individual DR curves fall. Curves are centred.</p>
  </div>

  <ul class="pts">
    <li><strong>Outcome model wrong:</strong> the S-learner misses the curve; DR gets much closer, because the weighted residuals correct most of the error.</li>
    <li><strong>Dose model wrong:</strong> weighting only misses the curve; DR stays close, because the outcome model was already right.</li>
    <li><strong>Both wrong:</strong> DR misses too. It needs at least one of the two models to be right.</li>
  </ul>

  <p>Weighting only is the weakest throughout. Even with the right dose model, a few units get extreme weights, and capping them, a standard safeguard, leaves some bias. DR inherits little of this, because it only weights the residuals.</p>

  <h3>In practice</h3>

  <p>Nobody knows which model is right, so in practice both are flexible machine learning models. With 50 confounders the S-learner's weakness shows: a boosted outcome model shrinks the dose's effect and lets correlated features absorb part of it, so the curve flattens. DR can correct that, but only with good weights, and weights built from a boosted density aren't: small errors in the density turn into very large errors in the weights. Entropy balancing weights avoid the density altogether. They make the dose uncorrelated with each confounder in the weighted sample (<a href="https://doi.org/10.1515/jem-2021-0002" target="_blank" rel="noopener">Tübbicke 2022</a>).</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimated curves with boosted nuisance models, 50 confounders</div>
    </div>
    <div class="fig-body">{LEG5}<div class="panels" style="grid-template-columns:1fr">{practice_panel}</div></div>
    <p class="fig-cap">Average estimated curve over {N_REPS2} simulated datasets, all with the same boosted outcome model. The blue band shows where 90% of the individual curves with balancing weights fall. Dashed black: the true curve. Lines that leave the chart are cut off at its edge.</p>
  </div>

  <p>The S-learner flattens the curve, DR with density weights is far off, and DR with balancing weights recovers the shape. Its intervals need widening, though, which section 5 comes back to.</p>
</section>

<!-- 4 -->
<section id="many">
  <p class="s-label">Simulation 3</p>
  <h2>4. Many curves: borrow strength across the hierarchy</h2>

  <p>The decision is rarely one curve for the whole business. Pricing teams set discounts by category, marketing teams set budgets by country and channel, CRM teams set cadence by segment. The groups are nested (stores in regions, products in categories, campaigns in channels), and their sizes follow a long tail: a few big ones with plenty of data and many small ones with very little.</p>

  <p>Two simple options both fail on that long tail. One curve for everyone (complete pooling) ignores real differences between groups. A separate curve per group (no pooling) treats a store with 12 observations as if it knew as much as one with 500. Looping DoubleML over categories has the same problem and is worse, because it also refits the nuisance models on each store's handful of rows.</p>

  <p>Partial pooling sits between them. Each store's curve is the global curve, plus a regional deviation, plus a store deviation:</p>

  <div class="eqn">θ<sub>store</sub>(a) = B(a)′ ( β<sub>global</sub> + b<sub>region</sub> + b<sub>store</sub> )</div>

  <p>B(a) is a set of spline basis functions. The deviations get a ridge penalty, which is the same as treating them as random effects. A small store's deviation is shrunk toward its region, and a region's toward the global curve. The penalty applies to each group's shape; its overall level is left almost free, so stores with very different baselines can still share a shape. A big store has enough data to overrule the shrinkage. How strongly to shrink at each level is chosen from the data by generalised cross-validation (GCV, a fast approximation to leave-one-out error), so you do not set it by hand.</p>

  <p>Because the DR pseudo-outcome is just a number per row, this second stage can be any regression. The DR step handles confounding; the pooling step handles the hierarchy. For store-level curves, the averages inside m(a) and ϖ(a) are taken within each store.</p>

  <p>The simulation has 8 regions with 25 stores each. Store sizes are log-normal, from 8 to several hundred observations, and each store's true curve is the inverted U plus a random regional and store deviation. Each store also has its own baseline, which raises both its doses and its outcomes, so there is store-level confounding to remove.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">The smallest and the largest store</div>
      <div class="fig-sub">Same DR pseudo-outcomes; only the pooling changes. Curves centred.</div>
    </div>
    <div class="fig-body">{LEG3}<div class="panels">{''.join(sim3_panels)}</div></div>
    <p class="fig-cap">Store baselines are not the target, so curves are compared up to a constant.</p>
  </div>

  <p>Complete pooling gives every store the same curve, so it misses stores whose regions bend differently. In small stores, no pooling leans on a handful of points, while partial pooling leans on the region's curve and widens its band. In the largest store, no pooling and partial pooling agree, because the data overrules the shrinkage.</p>

  <p>Two stores are anecdotes. The next chart shows all 200. For each store it takes one number you might act on, the change in outcome from moving the dose from {f2(a_lo)} to {f2(a_hi)}, and plots the estimation error against the store's size. The fourth panel is the usual workaround: one DoubleMLPLR per store, with its slope multiplied by the same change in dose. In the smallest stores that means boosted nuisance models fitted on a few dozen rows with two folds, so treat it as what the workaround gives you, not DoubleML at its best.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error in each store's dose effect, all 200 stores</div>
      <div class="fig-sub">Estimated minus true effect of moving the dose from {f2(a_lo)} to {f2(a_hi)}. One dot per store.</div>
    </div>
    <div class="fig-body"><div class="panels">{''.join(funnel)}</div></div>
    <p class="fig-cap">Dots outside the plotted range are drawn at the edge, faded.{" DoubleML failed to fit in " + str(n_fail) + " stores." if n_fail else ""}</p>
  </div>

  <p>DoubleML fans out into a wide funnel and no pooling into a narrower one: the smaller the store, the larger the error. Complete pooling has the opposite problem. Its errors don't shrink as stores get bigger, because it gives every store the same curve. Partial pooling stays tight in small stores and matches no pooling in big ones. Across all {N_REPS3} replications of the full curve, its error is {overall3['partial']:.2f}, against {overall3['none']:.2f} with no pooling and {overall3['complete']:.2f} with complete pooling, and in stores with fewer than 25 observations it is {small['partial']:.2f} against {small['none']:.2f}.</p>

  <p>The explorer below shows any single store's curves.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Explore every store</div>
      <div class="fig-sub">Choose a store to compare the three second stages against its true curve</div>
    </div>
    <div class="fig-body">
      <div style="margin-bottom:.6rem"><select id="storeSel" aria-label="Store"></select></div>
      {LEG3}
      <div id="explorer"></div>
    </div>
  </div>
</section>

<!-- 5 -->
<section id="method">
  <h2>5. Using it</h2>

  <p>The companion repo wraps the three steps in one call. This runs as written, on the store data from section 4:</p>

<pre><code>import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from drcurves import fit_dose_response
from drcurves.simulate import sim_hierarchical

df, _, _ = sim_hierarchical(seed=0)            # columns: region, store, x0..x2, A, Y

model = fit_dose_response(
    X=df[["x0", "x1", "x2"]], A=df["A"], Y=df["Y"],
    outcome_model=HGB(),
    groups=df[["region", "store"]],   # top -> bottom; df[["store"]] for flat groups; None for one curve
)                                     # group codes are added to X for the nuisance models
print(model.diagnostics())            # weights, fitted dose range, penalties, groups with constant dose

a_grid = np.linspace(*np.quantile(df["A"], [0.05, 0.95]), 50)
curve, se = model.curve(a_grid, ("r0", "r0_s03"))      # one store
region, _ = model.curve(a_grid, ("r0",))               # its region
new, _ = model.curve(a_grid, ("r0", "a_new_store"))    # unseen store: falls back to its region</code></pre>

  <p>If the dose takes only a few values (tiers, or counts such as notifications per week), the package detects it and models the propensity with a classifier over those values instead of a density, because a continuous density is not a valid weight for a discrete dose. Three checks are worth running every time, and <code>diagnostics()</code> reports the numbers for them:</p>

  <ol class="pts">
    <li><strong>Overlap.</strong> Look at the weights ϖ(A)/π(A | X). A long right tail, or a small effective sample size share, means some doses are rare for some kinds of unit, and the curve there is an extrapolation. Report the curve over the doses you actually observe; the package warns if you ask for doses outside them. Weights are capped at the 99th percentile by default. The cap adds a little bias; without it, a handful of near-zero density estimates can swamp the curve. <code>max_weight_quantile</code> sets the trade-off.</li>
    <li><strong>The weights.</strong> For a continuous dose the default learns the weights directly (automatic debiasing). If you switch to density weights with <code>gps=GPS()</code>, check that the dose model's residuals look roughly the same shape across units.</li>
    <li><strong>Shrinkage.</strong> Read the chosen penalties. Very large ones mean the level adds little beyond its parent; very small ones mean the groups really are different. Groups with a constant dose have no curve of their own to estimate and simply inherit their parent's.</li>
  </ol>

  <h3>Uncertainty</h3>

  <p>A curve is only useful for a decision if you know how far to trust it. The package gives four kinds of interval, all from the same fitted model:</p>

<pre><code>est, lo, hi, _ = model.band(a_grid, ("north", "store_17"))           # whole-curve band
lift, se = model.contrast(0.10, 0.20, ("north", "store_17"))         # dose 0.10 -> 0.20
gap, se = model.contrast(0.15, 0.15, ("north", "store_17"), ("north", "store_04"))  # same dose, two stores
best = model.optimal_dose(a_grid, ("north", "store_17"))             # dose, lo, hi, at_edge</code></pre>

  <ul class="pts">
    <li><strong>Bands.</strong> A pointwise band covers each dose separately. A simultaneous band covers the whole curve at once, which is what you need before saying "the curve peaks between x and y". It is wider, by a critical value found from draws of the fitted curve.</li>
    <li><strong>Contrasts.</strong> The change from one dose to another, or the gap between two groups at the same dose. These are linear in the spline coefficients, so their standard errors are exact given the coefficient covariance.</li>
    <li><strong>Optimal dose.</strong> The argmax of each drawn curve gives an interval for where the peak is. <code>at_edge</code> reports how often the peak lands at the end of the observed doses, a warning that the true optimum may lie outside your data.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">What the intervals look like</div>
      <div class="fig-sub">One dataset from Sim 1 (inverted U, 4,000 units), default settings</div>
    </div>
    <div class="fig-body">{LEG4}<div class="panels">{band_panel}{peak_panel}</div></div>
    <p class="fig-cap">Right: the peak of each of 4,000 curves drawn from the fitted model; the shaded range is the 95% interval.</p>
  </div>

  <p>The whole-curve band is about {eb['z_s'] / eb['z_p']:.1f} times as wide as the dose-by-dose one, and it is the one to use before saying anything about the shape. This example shows why: at the lowest doses the true curve drifts outside the dose-by-dose band, but it stays inside the whole-curve band throughout. The peak is pinned down to a range rather than a point ({f2(op['lo'])} to {f2(op['hi'])}, with the true peak at {f2(eb['true_peak'])}), which is usually the more honest answer to "what dose should we use?". Moving the dose from {f2(ct['a_lo'])} to {f2(ct['a_hi'])} changes the outcome by {f2(ct['est'])} ± {1.96 * ct['se']:.2f} (truth {f2(ct['truth'])}).</p>

  <p>The covariance matters more than the method. Penalised splines and shrinkage are biased on purpose, and a standard sandwich covariance treats the fitted curve as if it were unbiased. The default is instead the posterior covariance of the equivalent Bayesian model, σ²(X′X + P)⁻¹ with P the penalty, which is what <code>mgcv</code> uses for its intervals. Its intervals widen where the penalty pulls the curve. <a href="https://doi.org/10.1111/j.1467-9469.2011.00760.x" target="_blank" rel="noopener">Marra and Wood (2012)</a> show this gives close to nominal coverage averaged across the curve, rather than at every point.</p>

  <p>To check the intervals, the Sim 1 and Sim 3 data were regenerated many times and each interval was scored against the truth. The bands are for each curve's shape (the curve minus its average), since group baselines are not the target:</p>

  <div class="dtable-wrap">{table4}</div>

  <ul class="pts">
    <li><strong>One curve:</strong> the default pointwise band is a little below nominal ({pct(s4a.loc['bayes','pointwise'])}), and the simultaneous band falls further short ({pct(s4a.loc['bayes','simultaneous'])}). Contrasts cover {pct(s4a.loc['bayes','contrast'])}. Optimal-dose intervals cover {pct(s4a.loc['bayes','optimum'])}.</li>
    <li><strong>Many curves:</strong> the plain sandwich is badly overconfident ({pct(s4b.loc[('sandwich','all'),'pointwise'])} pointwise), because it ignores shrinkage. The default does much better ({pct(s4b.loc[('bayes','all'),'pointwise'])} pointwise, {pct(s4b.loc[('bayes','all'),'optimum'])} for the optimal dose) but is still below 95% for bands. Coverage is {pct(small4['pointwise'])} pointwise in stores under 25 observations and {pct(big4['pointwise'])} above 150. A likely reason for any shortfall in big stores is a small bias from the nuisance models, which stays while the band narrows; the simulation does not isolate it.</li>
  </ul>

  <p>In practice, use the simultaneous band for any statement about the shape, ask for <code>level=0.99</code> when a decision rests on it, and treat a 95% band over many shrunk curves as closer to an 85–90% band.</p>

  <p>None of these intervals include the uncertainty from fitting the outcome model and the GPS. For a number that will drive a decision, run <code>bootstrap_curves</code> once. It refits the whole pipeline on resampled data (within each store, so every store's curve stays defined). It is slow, so use it as a check rather than routinely.</p>
</section>

<section id="limits">
  <h2>When not to use this</h2>
  <p>Every method here rests on the same assumptions, and some data breaks them:</p>
  <ul class="pts">
    <li><strong>Unmeasured confounding.</strong> X must explain how doses were assigned. Double robustness protects against a wrong model, not a missing variable.</li>
    <li><strong>Poor overlap.</strong> If some kinds of unit only ever get low doses, nothing in the data says what high doses would do for them. Check the weights, and restrict the dose range you report.</li>
    <li><strong>Many confounders.</strong> With dozens of features, density weights break and the intervals in section 5 are too narrow. Use balancing weights and widen the bands using the package's coverage check on your own data.</li>
    <li><strong>Doses set at the group level.</strong> If every customer in a store gets the store's discount, the store is the unit of analysis. Aggregate to stores first; row-level pooling across customers overstates what you know.</li>
    <li><strong>A spike at zero.</strong> Zero spend next to a continuous range of positive spend is a mixture the density model cannot represent. The package warns; analyse "any versus none" separately from "how much".</li>
    <li><strong>Crossed groupings.</strong> Store × channel, where channels are not nested in stores, is not supported. Each level must sit inside the one above.</li>
    <li><strong>A population curve from a hierarchical fit.</strong> The top-level curve of a pooled fit is a "typical group" curve, not the average over all units. Fit with <code>groups=None</code> for that.</li>
  </ul>
</section>

<!-- FAQ -->
<section id="faq">
  <h2>Frequently asked questions</h2>

  <details class="faq"><summary>Why not bin the treatment and use a discrete estimator?</summary>
    <p>DoubleML's <code>DoubleMLAPOS</code> estimates average potential outcomes for discrete treatment levels, so binning the dose is an option. It throws away the ordering of the doses, each bin needs its own propensity model and enough data, and the answer depends on where you put the cut points. Within a bin, units with different doses are treated as if they got the same one, which brings back a version of the policy dependence from section 2.</p>
  </details>

  <details class="faq"><summary>What about a polynomial in the treatment, as in EconML?</summary>
    <p>EconML's DML estimators accept a treatment featurizer, so you can fit θ₁D + θ₂D². That handles a simple curve, but you choose the shape in advance, it is still a partially linear model with the same score, and it does not pool across groups.</p>
  </details>

  <details class="faq"><summary>Is there a kernel version of double machine learning for this?</summary>
    <p>Yes. <a href="https://arxiv.org/abs/2004.03036" target="_blank" rel="noopener">Colangelo and Lee</a> estimate the curve one dose at a time with a kernel-localised doubly robust score, and <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Klosin (2021)</a> does it without estimating the density. Both are good options for a single population curve with careful inference. The pseudo-outcome approach here was chosen because the second stage can be any regression, which is what makes the hierarchical pooling simple.</p>
  </details>

  <details class="faq"><summary>Could I do the pooling in a Bayesian model instead?</summary>
    <p>Yes. The ridge penalties are Gaussian priors on the deviations, tuned by cross-validation rather than given hyperpriors. Fitting the same spline structure in PyMC or brms on the pseudo-outcomes gives full posteriors for each store's curve, at more computational cost.</p>
  </details>

</section>

<!-- Conclusion -->
<section id="together">
  <h2>Conclusion</h2>

  <p>The partially linear model is a good tool used for the wrong question. With a continuous treatment, the question is about a curve, and the three simulations show where a single coefficient falls short:</p>

  <ul class="pts">
    <li><strong>Shape.</strong> θ is an average slope weighted by where the doses fall, so it moves with the treatment policy. Estimate θ(a) instead.</li>
    <li><strong>Robustness.</strong> The default partialling-out score breaks when the treatment model is wrong; the IV-type score and ADML fix that, but only for θ. The DR pseudo-outcome survives either model being wrong for the whole curve. With few confounders a good outcome model alone can match it; with many, DR pays off only if the weights are estimated well, which rules out density weights.</li>
    <li><strong>Many curves.</strong> Looping over groups gives noise for the small ones. Partial pooling gives every group a curve, shrinks the small ones toward their parent, and lets the big ones speak for themselves.</li>
  </ul>

  <p>The simulations use one data-generating process per question, known confounders and moderate sample sizes. They do not show how often each problem occurs in your data. The questions to ask are the same, though: is the curve likely to bend, which nuisance model are you least sure of, and how many groups need their own answer?</p>

  <p>The estimator, simulations and the script that builds every chart and number on this page are in the <a href="{REPO}" target="_blank" rel="noopener">companion repo</a>.</p>

  <h3>References</h3>
  <ul class="refs">
    <li>Bach, P., Chernozhukov, V., Kurz, M. S. and Spindler, M. (2022). <a href="https://jmlr.org/papers/v23/21-0862.html" target="_blank" rel="noopener">DoubleML: an object-oriented implementation of double machine learning in Python</a>. <em>Journal of Machine Learning Research</em>, 23(53), 1–6.</li>
    <li>Chernozhukov, V. et al. (2018). <a href="https://arxiv.org/abs/1608.00060" target="_blank" rel="noopener">Double/debiased machine learning for treatment and structural parameters</a>. <em>The Econometrics Journal</em>, 21(1), C1–C68.</li>
    <li>Colangelo, K. and Lee, Y.-Y. (2020). <a href="https://arxiv.org/abs/2004.03036" target="_blank" rel="noopener">Double debiased machine learning nonparametric inference with continuous treatments</a>. arXiv:2004.03036.</li>
    <li>Eilers, P. and Marx, B. (1996). Flexible smoothing with B-splines and penalties. <em>Statistical Science</em>, 11(2), 89–121.</li>
    <li>Gelman, A. and Hill, J. (2007). <em>Data Analysis Using Regression and Multilevel/Hierarchical Models</em>. Cambridge University Press.</li>
    <li>Hirano, K. and Imbens, G. (2004). The propensity score with continuous treatments. In <em>Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives</em>, Wiley.</li>
    <li>Kennedy, E., Ma, Z., McHugh, M. and Small, D. (2017). <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Non-parametric methods for doubly robust estimation of continuous treatment effects</a>. <em>Journal of the Royal Statistical Society: Series B</em>, 79(4), 1229–1245.</li>
    <li>Kim, Y.-J. and Gu, C. (2004). Smoothing spline Gaussian regression: more scalable computation via efficient approximation. <em>Journal of the Royal Statistical Society: Series B</em>, 66(2), 337–356.</li>
    <li>Klosin, S. (2021). <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Automatic double machine learning for continuous treatment effects</a>. arXiv:2104.10334.</li>
    <li>Marra, G. and Wood, S. (2012). <a href="https://doi.org/10.1111/j.1467-9469.2011.00760.x" target="_blank" rel="noopener">Coverage properties of confidence intervals for generalized additive model components</a>. <em>Scandinavian Journal of Statistics</em>, 39(1), 53–74.</li>
    <li>Tübbicke, S. (2022). <a href="https://doi.org/10.1515/jem-2021-0002" target="_blank" rel="noopener">Entropy balancing for continuous treatments</a>. <em>Journal of Econometric Methods</em>, 11(1), 71–89.</li>
    <li>Valentin, S., Detommaso, G., Li, Y., Opper, M. and Brueckner, M. (2025). <a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">On double robustness in double machine learning</a>. NeurIPS 2025 Workshop CauScien.</li>
  </ul>
</section>

</main>

<footer>Simulated data throughout. Every chart and number on this page is generated by <code>tools/build_article.py</code> in the companion repo from the outputs of the three experiment scripts.</footer>

<script>
const D = {explorer_data};
(function () {{
  const sel = document.getElementById('storeSel'), box = document.getElementById('explorer');
  D.stores.forEach((s, i) => {{
    const o = document.createElement('option');
    o.value = i; o.textContent = s.store + ' · ' + s.region + ' · n = ' + s.n;
    sel.appendChild(o);
  }});
  const W = 680, H = 300, L = 44, R = W - 12, T = 12, B = H - 36, g = D.grid;
  const all = D.stores.flatMap(s => s.truth.concat(s.partial));
  all.sort((a, b) => a - b);
  const ylo = all[Math.floor(all.length * .005)] - .3, yhi = all[Math.floor(all.length * .995)] + .3;
  const sx = v => L + (v - g[0]) / (g[g.length - 1] - g[0]) * (R - L);
  const sy = v => B - (Math.min(Math.max(v, ylo), yhi) - ylo) / (yhi - ylo) * (B - T);
  const line = (ys, col, w, dash) => '<path d="M' + ys.map((y, i) => sx(g[i]).toFixed(1) + ',' + sy(y).toFixed(1)).join('L') +
      '" fill="none" stroke="' + col + '" stroke-width="' + w + '" stroke-dasharray="' + (dash || '') + '"/>';
  function draw(i) {{
    const s = D.stores[i];
    let h = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '">';
    for (let t = Math.ceil(ylo); t <= yhi; t++)
      h += '<line x1="' + L + '" x2="' + R + '" y1="' + sy(t) + '" y2="' + sy(t) + '" stroke="#e5e7eb"/><text x="' + (L - 6) + '" y="' + (sy(t) + 4) + '" text-anchor="end" font-size="11" fill="#6b7280">' + t + '</text>';
    [-1, -0.5, 0, 0.5, 1, 1.5].forEach(t => {{ if (t >= g[0] && t <= g[g.length - 1])
      h += '<text x="' + sx(t) + '" y="' + (B + 16) + '" text-anchor="middle" font-size="11" fill="#6b7280">' + t + '</text>'; }});
    h += '<text x="' + ((L + R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle" font-size="11" fill="#6b7280">Treatment a</text>';
    const up = s.partial.map((v, k) => v + 1.96 * s.partial_se[k]), lo = s.partial.map((v, k) => v - 1.96 * s.partial_se[k]);
    h += '<path d="M' + up.map((y, k) => sx(g[k]).toFixed(1) + ',' + sy(y).toFixed(1)).join('L') + 'L' +
         lo.map((y, k) => [sx(g[k]).toFixed(1), sy(y).toFixed(1)]).reverse().map(p => p.join(',')).join('L') + 'Z" fill="#2a78d6" opacity=".15"/>';
    h += line(s.none, '#eb6834', 1.6) + line(s.complete, '#6b7280', 1.6) + line(s.partial, '#2a78d6', 2.6) + line(s.truth, '#111827', 2, '5 4');
    box.innerHTML = h + '</svg>';
  }}
  sel.addEventListener('change', e => draw(+e.target.value));
  draw(0);
}})();
(function () {{
  const links = document.querySelectorAll('.snav a');
  const secs = [...links].map(a => document.querySelector(a.getAttribute('href')));
  addEventListener('scroll', () => {{
    let k = 0; secs.forEach((s, i) => {{ if (s && s.getBoundingClientRect().top < 120) k = i; }});
    links.forEach((a, i) => a.classList.toggle('active', i === k));
  }}, {{ passive: true }});
}})();
</script>
</body>
</html>
"""
(ROOT / "article").mkdir(exist_ok=True)
(ROOT / "article" / "index.html").write_text(html)
print("wrote article/index.html", len(html) // 1024, "KB")
