"""Build article/index.html from the results/ folder. Every number and chart in
the article comes from here, so re-running the experiments and this script
regenerates the whole page."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg import grouped_bars, legend, line_panel, scatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CSS = (ROOT / "tools" / "article.css").read_text()
REPO = "https://github.com/raz1470/continuous-treatments"

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
s2 = pd.read_csv(RES / "sim2_curve_summary.csv")
s2plr = pd.read_csv(RES / "sim2_plr_summary.csv")
s2c = np.load(RES / "sim2_curves.npz")
s3 = json.loads((RES / "sim3_curves.json").read_text())
s3size = pd.read_csv(RES / "sim3_summary_by_size.csv")
s3rmse = pd.read_csv(RES / "sim3_store_rmse.csv")
s3plr = pd.read_csv(RES / "sim3_plr_per_store.csv")
N_REPS2 = int(pd.read_csv(RES / "sim2_plr_reps.csv").rep.nunique())
N_REPS3 = int(s3rmse.rep.nunique())
s4a = pd.read_csv(RES / "sim4_single_summary.csv").set_index("cov_type")
s4b = pd.read_csv(RES / "sim4_stores_summary.csv").set_index(["cov_type", "size_band"])
N_REPS4A = int(pd.read_csv(RES / "sim4_single_reps.csv").rep.nunique())
s5 = pd.read_csv(RES / "sim5_dimension_summary.csv").set_index(["p", "estimator"])
N_REPS5 = int(s5.reps.min())


def b5(p, est):
    return float(s5.loc[(p, est), "abs_bias"])


rows5 = []
for est, lab in [("plugin", "Plug-in (outcome model only)"), ("dr_density", "DR, density weights"),
                 ("dr_riesz", "DR, learned (Riesz) weights"), ("dr_true_weights", "DR, true weights (benchmark)")]:
    vals = [b5(p, est) for p in (5, 20, 50)]
    best = min(b5(p, e) for p in (5, 20, 50) for e in ("plugin", "dr_density", "dr_riesz"))
    cells = "".join(f"<td class='num'>{v:.2f}</td>" for v in vals)
    rows5.append(f"<tr><td>{lab}</td>{cells}</tr>")
table5 = ("<table class='dtable'><caption>Average absolute bias of the mean estimated curve, boosted outcome model, "
          f"n = 3,000, {N_REPS5} replications per column</caption><thead><tr><th>Estimator</th>"
          "<th class='num'>5 confounders</th><th class='num'>20</th><th class='num'>50</th></tr></thead><tbody>"
          + "".join(rows5) + "</tbody></table>")
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


def c2(om, gm, est, col="abs_bias"):
    return float(s2.query("outcome==@om and gps==@gm and estimator==@est")[col].iloc[0])


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
SC = [("right", "right", "Both right", "All three estimators recover the curve"),
      ("wrong", "right", "Outcome model wrong", "Plug-in fails; IPW and DR hold"),
      ("right", "wrong", "GPS wrong", "IPW fails; plug-in and DR hold"),
      ("wrong", "wrong", "Both wrong", "Nothing can save this")]
sim2_panels = []
for om, gm, title, sub in SC:
    ser = []
    for est, col in [("plugin", YELLOW), ("ipw", AQUA), ("dr", BLUE)]:
        cs = s2c[f"{om}|{gm}|{est}"]
        item = {"y": cs.mean(0), "color": col, "width": 2.6 if est == "dr" else 1.8}
        if est == "dr":
            item["lo"], item["hi"] = np.percentile(cs, [5, 95], axis=0)
        ser.append(item)
    ser.append({"y": t2, "color": INK, "dash": "5 4", "width": 2})
    sim2_panels.append(line_panel(g2, ser, title, sub, ylim=(t2.min() - 1.6, t2.max() + 1.0)))


def cls(v, good=0.15, bad=0.3):
    return "good" if v < good else ("bad" if v > bad else "")


rows2 = []
for om, gm, title, _ in SC + [("boosted", "boosted", "Both boosted (practice)", "")]:
    cells = "".join(f'<td class="num {cls(c2(om, gm, e))}">{c2(om, gm, e):.2f}</td>' for e in ["plugin", "ipw", "dr"])
    rows2.append(f"<tr><td>{title}</td>{cells}<td class='num'>{c2(om, gm, 'dr', 'sd'):.2f}</td></tr>")
table2 = ("<table class='dtable'><caption>Average absolute bias of the mean estimated curve over the 5–95% "
          f"treatment range, {N_REPS2} replications (true curve spans about {t2.max()-t2.min():.1f} units)</caption>"
          "<thead><tr><th>Nuisance models</th><th class='num'>Plug-in</th><th class='num'>IPW</th>"
          "<th class='num'>Doubly robust</th><th class='num'>DR spread (sd)</th></tr></thead><tbody>"
          + "".join(rows2) + "</tbody></table>")

def cov(c):
    return "good" if c >= 0.85 else "bad"


rowsp = []
for om, gm, title, _ in SC + [("boosted", "boosted", "Both boosted (practice)", "")]:
    a, b = p2("partialling out", om, gm), p2("IV-type", om, gm)
    ca, cb = p2("partialling out", om, gm, "coverage"), p2("IV-type", om, gm, "coverage")
    rowsp.append(f"<tr><td>{title.replace('GPS', 'Treatment model')}</td>"
                 f"<td class='num'>{f2(a, True)}</td><td class='num {cov(ca)}'>{pct(ca)}</td>"
                 f"<td class='num'>{f2(b, True)}</td><td class='num {cov(cb)}'>{pct(cb)}</td></tr>")
tablep = ("<table class='dtable'><caption>DoubleMLPLR on a linear effect (true θ = 1), "
          f"{N_REPS2} replications</caption><thead><tr><th>Nuisance models</th>"
          "<th class='num'>Partialling out: bias</th><th class='num'>95% CI coverage</th>"
          "<th class='num'>IV-type: bias</th><th class='num'>95% CI coverage</th></tr></thead><tbody>"
          + "".join(rowsp) + "</tbody></table>")

# ===================================================================== sim 3
grid3 = np.array(s3["grid"])
stores = sorted(s3["stores"], key=lambda d: d["n"])
pick_idx = [0, len(stores) // 4, len(stores) // 2, len(stores) - 1]
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

bands = s3size.size_band.tolist()
bars3 = grouped_bars(
    [f"{b}\n{int(n)} stores" for b, n in zip(bands, s3size.n_stores)],
    [("No pooling", ORANGE, s3size["none"].tolist()),
     ("Complete pooling", GREY, s3size["complete"].tolist()),
     ("Partial pooling", BLUE, s3size["partial"].tolist())],
    ylabel="RMSE of store curve")
overall3 = s3rmse.groupby("pooling").rmse.mean()
small = s3rmse[s3rmse.n < 25].groupby("pooling").rmse.mean()
big = s3rmse[s3rmse.n >= 150].groupby("pooling").rmse.mean()

# true average slope of each store over the grid (what a per-store PLR should land near)
slope_true = {d["store"]: (d["truth"][-1] - d["truth"][0]) / (grid3[-1] - grid3[0]) for d in stores}
s3plr["true_slope"] = s3plr.store.map(slope_true)
ok = s3plr.theta.notna()
plr_small = s3plr[ok & (s3plr.n < 25)]
plr_big = s3plr[ok & (s3plr.n >= 150)]
sl = np.array(list(slope_true.values()))
scat = scatter(s3plr.n[ok], s3plr.theta[ok], xlabel="Observations in store (log scale)",
               ylabel="Per-store DoubleML θ", logx=True, ylim=(-3, 5),
               refs=(np.percentile(sl, 5), np.percentile(sl, 95)))

explorer_data = json.dumps({"grid": s3["grid"], "stores": [
    {k: d[k] for k in ["store", "region", "n", "truth", "none", "complete", "partial", "partial_se"]} for d in stores]})

# =================================================================== article
LEG1 = legend([("True curve", INK, "5 4"), ("DR curve, this repo (95% band)", BLUE, ""), ("DoubleML PLR", ORANGE, "")])
LEG2 = legend([("True curve", INK, "5 4"), ("Plug-in (outcome model only)", YELLOW, ""),
               ("IPW (GPS only)", AQUA, ""), ("Doubly robust (5–95% across reps)", BLUE, "")])
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
  <a href="#plr" class="active">1 · The usual approach</a>
  <a href="#shape">2 · Shape</a>
  <a href="#robust">3 · Robustness</a>
  <a href="#many">4 · Many curves</a>
  <a href="#method">5 · In practice</a>
  <a href="#limits">Limits</a>
  <a href="#faq">Questions</a>
  <a href="#together">Conclusion</a>
</nav>

<main>

<p>A lot of the treatments we care about are not on or off. A discount can be 5% or 40%. A marketing team can spend £10k or £200k in a region. A user can get one notification a week or fifteen. The question is rarely "did it work?" It is "how much is the right amount?", and that question is about the shape of a curve.</p>

<p>When the treatment is continuous and chosen rather than randomised, most analysts reach for double machine learning, usually the partially linear model in <a href="https://docs.doubleml.org/" target="_blank" rel="noopener">DoubleML</a>. It is a good default for a binary treatment. For a continuous one it has three limitations that matter in practice:</p>

<ul class="pts">
  <li><strong>It returns one number.</strong> The effect is assumed linear, so a rise-then-fall or a saturating curve comes back as a single slope.</li>
  <li><strong>Its default score is not doubly robust.</strong> With a wrong treatment model the estimate is biased, however good the outcome model is. The alternative score is doubly robust, but only for that single number.</li>
  <li><strong>It gives one answer per population.</strong> If you need a curve for each of 200 stores, you loop, and the small stores get noise.</li>
</ul>

<p>This article runs one simulation per limitation, plus a fourth on uncertainty, and shows an estimator that addresses all three: the doubly robust dose-response estimator of <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Kennedy et al. (2017)</a>, with a second stage that partially pools curves across a hierarchy.</p>

<!-- 1 -->
<section id="plr">
  <h2>1. The partially linear model, and what replaces it</h2>

  <p>DoubleML's <code>DoubleMLPLR</code> fits</p>

  <div class="eqn">Y = θD + g(X) + ε</div>

  <p>where D is the treatment, X the confounders and g(X) is learned with any machine learning model. Models like g, which we need but do not care about for their own sake, are called nuisance models. Cross-fitting (each row's predictions come from models trained on other rows) and an orthogonal score (an estimating equation that is insensitive to small errors in the nuisance models) remove the regularisation bias from g, so θ comes with a valid confidence interval. The cost is the θD term: the effect of every extra unit of treatment is the same, wherever you start.</p>

  <p>When the truth is a curve f(D), θ still converges to something. If the treatment is roughly normal given X, it is a weighted average of the curve's slope (a consequence of Stein's lemma):</p>

  <div class="eqn">θ = E[ Var(D | X) · f′(D) ] / E[ Var(D | X) ]</div>

  <p>The weights come from where the treatment happens to vary in your data. Change who gets how much treatment and θ changes, even if the causal curve does not. Section 2 shows this happening.</p>

  <p>The replacement estimates the whole curve, θ(a) = E[Y(a)], the average outcome if everyone received dose a. It runs in three steps:</p>

  <div class="flow">
    <div class="step"><div class="n">Step 1</div><div class="t">Cross-fit two nuisance models</div><p>An outcome model μ(x, a) = E[Y | X, A], and a generalised propensity score π(a | x): the density of the dose given X.</p></div>
    <div class="step"><div class="n">Step 2</div><div class="t">Build a pseudo-outcome</div><p>Combine them into one number per row, ξ, whose average at each dose is the causal curve.</p></div>
    <div class="step"><div class="n">Step 3</div><div class="t">Smooth, with pooling</div><p>Regress ξ on the dose with penalised splines, sharing strength across a hierarchy of groups.</p></div>
  </div>

  <p>The pseudo-outcome is</p>

  <div class="eqn">ξ = (Y − μ(X, A)) · ϖ(A) / π(A | X) + m(A)</div>

  <p>where m(a) is the outcome model averaged over everyone's X at dose a, and ϖ(a) is the marginal density of the dose. The first term is an inverse-propensity weighted residual and the second a regression prediction. If the outcome model is right, the residuals average to zero and m(a) is the answer. If the propensity model is right, the weighted residual corrects whatever m(a) got wrong. That is what doubly robust means: either model can be wrong, as long as one of them is right. The ratio ϖ(A)/π(A | X) acts as a weight; section 3 shows it is better learned directly than built from a density when there are many confounders.</p>

  <p>The binary-treatment AIPW estimator in DoubleML's <code>DoubleMLIRM</code> works the same way. The difference is that π is a density rather than a probability, and the final step is a curve rather than a mean.</p>
</section>

<!-- 2 -->
<section id="shape">
  <p class="s-label">Simulation 1</p>
  <h2>2. Shape: one slope cannot describe a curve</h2>

  <p>Nonlinear dose-response is the norm in practice, not the exception. Two shapes cover most of what you will meet:</p>

  <ul class="pts">
    <li><strong>Saturating.</strong> Diminishing returns. The first £10k of ad spend in a region buys more conversions than the next £10k, and the hundredth sales call to the same account adds almost nothing. Marketing mix models build this in with saturation curves for a reason.</li>
    <li><strong>Inverted U.</strong> Too much of a good thing. Deeper discounts win orders but give away margin, more notifications drive opens until users switch them off, and a higher price raises revenue per unit until volume falls away.</li>
  </ul>

  <p>Either way, the decision you need to make (how much, and where to stop) lives in the curvature. A linear model removes the curvature by assumption.</p>

  <p>The simulation draws 4,000 units with five covariates. Three of them drive the dose through nonlinear functions, a fourth changes how much the dose varies between units, and the first three also drive the outcome, so they confound the effect. The true effect is an inverted U in one dataset and a saturating curve in the other. Both methods use the same gradient boosting models for every nuisance function.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">One dataset, two estimates of the dose-response</div>
      <div class="fig-sub">Middle 90% of the observed doses. Curves centred on the truth, since only the shape is being compared.</div>
    </div>
    <div class="fig-body">{LEG1}<div class="panels">{''.join(sim1_panels)}</div></div>
    <p class="fig-cap">The DoubleML line is θ times the dose, drawn through the middle of the truth. Bands are pointwise 95% intervals.</p>
  </div>

  <p>The DR curve follows both shapes (root mean squared error of the centred curve {inv.dr_curve_rmse:.2f} and {sat.dr_curve_rmse:.2f}, against {inv.plr_line_rmse:.2f} and {sat.plr_line_rmse:.2f} for the DoubleML line). DoubleML's θ is not wrong: {inv.plr_theta:.2f} is in the neighbourhood of the plain average slope of the inverted U across these units ({inv.true_avg_derivative:.2f}). It is not equal to it, because θ weights each unit by how much its dose varies. It is the right answer to a question you probably did not mean to ask. It says nothing about where the curve turns over, and in the saturating case it gives every unit the same marginal return, when units at high doses get almost none.</p>

  <h3>θ depends on the treatment policy</h3>

  <p>The second problem is subtler. The formula in section 1 says θ averages the slope over wherever doses happen to fall. To check, the inverted-U data was regenerated three times with the same causal curve, moving the whole dose distribution down by 0.5, leaving it alone, and moving it up by 0.5.</p>

  <div class="knums">
    <div class="kbox"><div class="lbl">Doses shifted down</div><div class="num">{pol[0]['theta']:.2f}</div><div class="lab">DoubleML θ (s.e. {pol[0]['se']:.2f})</div></div>
    <div class="kbox"><div class="lbl">Original doses</div><div class="num">{pol[1]['theta']:.2f}</div><div class="lab">DoubleML θ (s.e. {pol[1]['se']:.2f})</div></div>
    <div class="kbox"><div class="lbl">Doses shifted up</div><div class="num">{pol[2]['theta']:.2f}</div><div class="lab">DoubleML θ (s.e. {pol[2]['se']:.2f})</div></div>
  </div>

  <p>Three tight, non-overlapping intervals for the same causal effect. A team that cut discounts last year and a team that raised them would report different "effects of discount", and both would be right about their own data. A curve does not have this problem: θ(a) is defined at each dose and does not depend on how doses were handed out, only on there being enough of them nearby.</p>

  <div class="callout"><p>With a continuous treatment, θ describes your treatment policy as much as the effect.</p></div>
</section>

<!-- 3 -->
<section id="robust">
  <p class="s-label">Simulation 2</p>
  <h2>3. Robustness: one of two models will be wrong</h2>

  <p>Every observational estimate leans on models we know are approximations, and for continuous treatments the treatment side is the hard one. The generalised propensity score is a full conditional density, not a probability. In practice the dose depends on confounders through business rules nobody wrote down: discount depth set by a pricing team's tiers, budget split by last year's revenue, notification cadence set by an engagement score with a cap. The variance changes too: some segments get a narrow band of doses, others a wide one. A model of the mean with a constant normal error, the usual default, is easy to get wrong. So is the outcome model, especially when the response depends on interactions you have not thought of.</p>

  <p>So the useful question is not "are my models right?" but "what happens when one of them is wrong?"</p>

  <h3>DoubleML's partially linear model</h3>

  <p>DoubleMLPLR offers two scores. The default, partialling out, regresses outcome residuals on treatment residuals. To isolate the robustness question from the shape question, this part uses a linear true effect (θ = 1), so the partially linear model is correctly specified. "Right" means a correctly specified parametric model; "wrong" means a linear regression on the raw confounders.</p>

  <div class="dtable-wrap">{tablep}</div>

  <p>With the default score, a wrong treatment model biases θ by {f2(p2('partialling out','right','wrong'), True)}. That looks small, but the intervals contain the truth in {pct(p2('partialling out','right','wrong','coverage'))} of replications, even though the outcome model is correct. The partialling-out score is only protected against errors in the outcome model. The IV-type score is doubly robust: bias {f2(p2('IV-type','right','wrong'), True)} and {pct(p2('IV-type','right','wrong','coverage'))} coverage in the same row. So DoubleML can give you double robustness if you change the score away from the default. The same point is made formally by <a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">Valentin et al. (2025)</a>, who show the partialling-out (Robinson) estimator depends on the treatment model, and propose an augmented estimator (ADML) that is doubly robust for θ. What none of these give you is a doubly robust curve: the robustness is for θ, the single slope from section 2. The last row, with boosted models for both, shows the same pattern in practice: {pct(p2('partialling out','boosted','boosted','coverage'))} coverage with the default score against {pct(p2('IV-type','boosted','boosted','coverage'))} with IV-type.</p>

  <h3>The curve</h3>

  <p>Now the true effect is the inverted U again. The dose still has plenty of random variation given X, so every kind of unit is seen at a range of doses (good overlap). Without that, inverse density weights become extreme and every weighting estimator, DR included, struggles. Each dataset gets a right and a wrong version of the outcome model and the GPS, combined four ways, and three estimators are fitted from the same pieces:</p>

  <ul class="pts">
    <li><strong>Plug-in:</strong> m(a) only, the outcome model averaged over X. Needs the outcome model right.</li>
    <li><strong>IPW:</strong> Y · ϖ(A)/π(A | X) only. Needs the GPS right.</li>
    <li><strong>Doubly robust:</strong> the pseudo-outcome ξ. Needs one of the two.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Mean estimated curve across {N_REPS2} replications</div>
      <div class="fig-sub">n = 3,000 per replication. Blue band: 5th to 95th percentile of the DR curve across replications.</div>
    </div>
    <div class="fig-body">{LEG2}<div class="panels">{''.join(sim2_panels)}</div></div>
    <p class="fig-cap">Every estimator uses the same penalised-spline second stage; only the pseudo-outcome changes.</p>
  </div>

  <div class="dtable-wrap">{table2}</div>

  <ul class="pts">
    <li><strong>Outcome model wrong:</strong> plug-in is off by {c2('wrong','right','plugin'):.2f} on average. DR is off by {c2('wrong','right','dr'):.2f}, because the weighted residuals correct it. What is left comes mostly from capping the largest weights (see section 5).</li>
    <li><strong>GPS wrong:</strong> IPW is off by {c2('right','wrong','ipw'):.2f}. DR is off by less than 0.01, because the outcome model was already right.</li>
    <li><strong>Both wrong:</strong> DR is off by {c2('wrong','wrong','dr'):.2f}. Double robustness gives you two chances, not a guarantee.</li>
  </ul>

  <p>Double robustness is about bias, not noise. When the outcome model is badly wrong the residuals are large, and multiplying them by inverse density weights makes the DR curve vary more between datasets (sd {c2('wrong','right','dr','sd'):.2f} against {c2('right','right','dr','sd'):.2f} when both are right). <a href="https://projecteuclid.org/journals/statistical-science/volume-22/issue-4/Demystifying-Double-Robustness--A-Comparison-of-Alternative-Strategies-for/10.1214/07-STS227.full" target="_blank" rel="noopener">Kang and Schafer (2007)</a> made the same point for binary treatments: when the weights are extreme, DR can do worse than a good regression.</p>

  <p>The last row is what you would do in practice: boosted models for both. Here plug-in wins: error {c2('boosted','boosted','plugin'):.2f} against {c2('boosted','boosted','dr'):.2f} for DR, with IPW far behind at {c2('boosted','boosted','ipw'):.2f}. When both nuisance models are good but not perfect, DR adds the product of their errors back in, and an imperfect boosted density makes that product noticeable. But this simulation has only four confounders, and with four smooth confounders and 3,000 rows a boosted outcome model has almost no regularisation bias left to correct. That is not the situation double machine learning was built for.</p>

  <h3>With many confounders</h3>

  <p>Real problems rarely have four confounders. Customer-level analyses typically have dozens of features, most with small effects. A regularised outcome model then shrinks the dose's effect and lets correlated features absorb part of it, so the plug-in curve flattens. This is the regularisation bias that the DR correction exists to remove, provided the weights can be estimated.</p>

  <p>Estimating the weights through a density is where it breaks. To test this, the Sim 2 data was extended with weak confounders (16 or 46 extra, correlated, each with a small effect on both dose and outcome; the total amount of confounding is held fixed). Four versions of the curve were compared:</p>

  <div class="dtable-wrap">{table5}</div>

  <ul class="pts">
    <li><strong>Plug-in gets worse as confounders are added</strong> ({b5(5,'plugin'):.2f} → {b5(50,'plugin'):.2f}): regularisation bias, as expected. (Even at five, its bias is larger than in Sim 2, because these covariates are correlated with each other.)</li>
    <li><strong>Perfect weights would fix most of it</strong> ({b5(50,'dr_true_weights'):.2f} at 50 confounders).</li>
    <li><strong>Density-based weights make DR worse, not better</strong> ({b5(50,'dr_density'):.2f}). A boosted model misses part of the dose's mean when it is spread thinly over many weak features, and the density turns small errors in the mean into very large errors in the weights.</li>
    <li><strong>Weights learned directly recover most of the benefit</strong> ({b5(50,'dr_riesz'):.2f}, against {b5(50,'plugin'):.2f} for plug-in).</li>
  </ul>

  <p>The fix is <em>automatic debiasing</em> (<a href="https://doi.org/10.3982/ECTA18515" target="_blank" rel="noopener">Chernozhukov, Newey and Singh 2022</a>; <a href="https://arxiv.org/abs/2110.03031" target="_blank" rel="noopener">Chernozhukov et al. 2022</a>). The weight r(x, a) = ϖ(a)/π(a | x) is the unique function that minimises a simple loss, so it can be learned with any machine learning model, without estimating a density:</p>

  <div class="eqn">r = argmin E[ r(X, A)² ] − 2 E[ r(X, A′) ]</div>

  <p>where A′ is a dose shuffled across rows, so it carries no information about X. The package minimises this loss with gradient-boosted trees. It also gives the trees the dose's predicted mean and the residual (from a ridge plus boosting regression of A on X) as extra features, because the weight depends mainly on how unusual a dose is for that unit. In the package it is <code>gps=RieszWeights()</code>.</p>

  <p>There is a catch, found when rerunning the coverage study of section 5 with learned weights. They give more accurate curves, but they leave a small bias that the intervals do not know about. Density weights are noisier, and their bands widen to match. On the Sim 1 data, 95% pointwise bands built on learned weights covered the truth only about half the time, against 95% with density weights. So the package keeps density weights as the default, and Simulations 1 to 4 use them. Use learned weights when you have many confounders and need the best point estimate, and check any interval from them with the bootstrap.</p>

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
      <div class="fig-title">Four stores, from the smallest to the largest</div>
      <div class="fig-sub">Same DR pseudo-outcomes; only the pooling changes. Curves centred.</div>
    </div>
    <div class="fig-body">{LEG3}<div class="panels">{''.join(sim3_panels)}</div></div>
    <p class="fig-cap">Store baselines are not the target, so curves are compared up to a constant.</p>
  </div>

  <p>Complete pooling gives every store the same curve, so it misses stores whose regions bend differently. In small stores, no pooling leans on a handful of points, while partial pooling leans on the region's curve and widens its band. In the largest store, no pooling and partial pooling agree, because the data overrules the shrinkage.</p>

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

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error by store size</div>
      <div class="fig-sub">Average RMSE of each store's curve against its true curve, {N_REPS3} replications of 200 stores</div>
    </div>
    <div class="fig-body">{bars3}</div>
  </div>

  <p>Partial pooling is the best or tied for best in every size band, and best overall ({overall3['partial']:.2f}, against {overall3['none']:.2f} with no pooling and {overall3['complete']:.2f} with complete pooling). The gain is largest where it matters most: in stores with fewer than 25 observations, error falls from {small['none']:.2f} to {small['partial']:.2f}. In stores with 150 or more it ties with no pooling ({big['partial']:.2f} against {big['none']:.2f}), and complete pooling is the worst option there at {big['complete']:.2f}.</p>

  <h3>Looping DoubleML over stores</h3>

  <p>For comparison, one DoubleMLPLR was fitted per store, the usual workaround. In the smallest stores that means boosted nuisance models fitted on a few dozen rows or fewer with two folds, so treat this as what the workaround gives you, not as DoubleML at its best. Each gives a slope, so the comparison point is each store's true average slope across the dose range (grey band: middle 90% of true store slopes).</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">One DoubleML θ per store</div>
      <div class="fig-sub">Each dot is a store. Grey: range of the true average slopes.</div>
    </div>
    <div class="fig-body">{scat}</div>
    <p class="fig-cap">Values outside the plotted range are drawn at the edge, faded.</p>
  </div>

  <p>Among stores with fewer than 25 observations, the per-store θ has a spread of {plr_small.theta.std():.1f} against {plr_small.true_slope.std():.1f} for the true slopes, so most of what you see is noise. Among stores with 150 or more it is {plr_big.theta.std():.1f}. And each estimate is still a single slope with the policy dependence from section 2.</p>
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

  <p>The covariance matters more than the method. Penalised splines and shrinkage are biased on purpose, and a standard sandwich covariance treats the fitted curve as if it were unbiased. The default is instead the posterior covariance of the equivalent Bayesian model, σ²(X′X + P)⁻¹ with P the penalty, which is what <code>mgcv</code> uses for its intervals. Its intervals widen where the penalty pulls the curve. <a href="https://doi.org/10.1111/j.1467-9469.2011.00760.x" target="_blank" rel="noopener">Marra and Wood (2012)</a> show this gives close to nominal coverage averaged across the curve, rather than at every point.</p>

  <p>To check the intervals, the Sim 1 and Sim 3 data were regenerated many times and each interval was scored against the truth. The bands are for each curve's shape (the curve minus its average), since group baselines are not the target:</p>

  <div class="dtable-wrap">{table4}</div>

  <ul class="pts">
    <li><strong>One curve:</strong> the default pointwise band is close to nominal ({pct(s4a.loc['bayes','pointwise'])}), and the simultaneous band falls a little short ({pct(s4a.loc['bayes','simultaneous'])}). Contrasts cover {pct(s4a.loc['bayes','contrast'])}, so their intervals are, if anything, too wide. Optimal-dose intervals cover {pct(s4a.loc['bayes','optimum'])}.</li>
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
    <li><strong>Robustness.</strong> The default partialling-out score breaks when the treatment model is wrong; the IV-type score and ADML fix that, but only for θ. The DR pseudo-outcome survives either model being wrong for the whole curve. With few confounders a good outcome model alone can match it; with many, DR pays off, but only if the weights are learned directly rather than through a density.</li>
    <li><strong>Many curves.</strong> Looping over groups gives noise for the small ones. Partial pooling gives every group a curve, shrinks the small ones toward their parent, and lets the big ones speak for themselves.</li>
  </ul>

  <p>The simulations use one data-generating process per question, known confounders and moderate sample sizes. They do not show how often each problem occurs in your data. The questions to ask are the same, though: is the curve likely to bend, which nuisance model are you least sure of, and how many groups need their own answer?</p>

  <p>The estimator, simulations and the script that builds every chart and number on this page are in the <a href="{REPO}" target="_blank" rel="noopener">companion repo</a>.</p>

  <h3>References</h3>
  <ul class="refs">
    <li>Bach, P., Chernozhukov, V., Kurz, M. S. and Spindler, M. (2022). <a href="https://jmlr.org/papers/v23/21-0862.html" target="_blank" rel="noopener">DoubleML: an object-oriented implementation of double machine learning in Python</a>. <em>Journal of Machine Learning Research</em>, 23(53), 1–6.</li>
    <li>Chernozhukov, V., Newey, W. and Singh, R. (2022). <a href="https://doi.org/10.3982/ECTA18515" target="_blank" rel="noopener">Automatic debiased machine learning of causal and structural effects</a>. <em>Econometrica</em>, 90(3), 967–1027.</li>
    <li>Chernozhukov, V., Newey, W., Quintas-Martínez, V. and Syrgkanis, V. (2022). <a href="https://arxiv.org/abs/2110.03031" target="_blank" rel="noopener">RieszNet and ForestRiesz: automatic debiased machine learning with neural nets and random forests</a>. <em>ICML</em>, PMLR 162.</li>
    <li>Chernozhukov, V. et al. (2018). <a href="https://arxiv.org/abs/1608.00060" target="_blank" rel="noopener">Double/debiased machine learning for treatment and structural parameters</a>. <em>The Econometrics Journal</em>, 21(1), C1–C68.</li>
    <li>Colangelo, K. and Lee, Y.-Y. (2020). <a href="https://arxiv.org/abs/2004.03036" target="_blank" rel="noopener">Double debiased machine learning nonparametric inference with continuous treatments</a>. arXiv:2004.03036.</li>
    <li>Eilers, P. and Marx, B. (1996). Flexible smoothing with B-splines and penalties. <em>Statistical Science</em>, 11(2), 89–121.</li>
    <li>Gelman, A. and Hill, J. (2007). <em>Data Analysis Using Regression and Multilevel/Hierarchical Models</em>. Cambridge University Press.</li>
    <li>Hirano, K. and Imbens, G. (2004). The propensity score with continuous treatments. In <em>Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives</em>, Wiley.</li>
    <li>Kang, J. and Schafer, J. (2007). <a href="https://projecteuclid.org/journals/statistical-science/volume-22/issue-4/Demystifying-Double-Robustness--A-Comparison-of-Alternative-Strategies-for/10.1214/07-STS227.full" target="_blank" rel="noopener">Demystifying double robustness</a>. <em>Statistical Science</em>, 22(4), 523–539.</li>
    <li>Kennedy, E., Ma, Z., McHugh, M. and Small, D. (2017). <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Non-parametric methods for doubly robust estimation of continuous treatment effects</a>. <em>Journal of the Royal Statistical Society: Series B</em>, 79(4), 1229–1245.</li>
    <li>Kim, Y.-J. and Gu, C. (2004). Smoothing spline Gaussian regression: more scalable computation via efficient approximation. <em>Journal of the Royal Statistical Society: Series B</em>, 66(2), 337–356.</li>
    <li>Klosin, S. (2021). <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Automatic double machine learning for continuous treatment effects</a>. arXiv:2104.10334.</li>
    <li>Marra, G. and Wood, S. (2012). <a href="https://doi.org/10.1111/j.1467-9469.2011.00760.x" target="_blank" rel="noopener">Coverage properties of confidence intervals for generalized additive model components</a>. <em>Scandinavian Journal of Statistics</em>, 39(1), 53–74.</li>
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
