"""Build article/index.html from the results/ folder. Every number and chart in
the article comes from here, so re-running the experiments and this script
regenerates the whole page."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg import funnel_panel, grouped_bars, hist_panel, legend, line_panel, shrink_panel  # noqa: E402

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
SC = [("wrong", "right", "Outcome model wrong"), ("right", "wrong", "Dose model wrong"),
      ("wrong", "wrong", "Both wrong")]
ylim2 = (t2.min() - 1.0, t2.max() + 0.8)
errs = {k: s2c[k] - t2 for k in s2c.files if k.startswith("A|") and not k.endswith("|ipw")}
lim_e = float(np.ceil(max(np.abs(v.mean(0)).max() for v in errs.values()) * 1.25 * 10) / 10)
sim2_panels = []
for om, dm, title in SC:
    ser = [{"y": np.zeros_like(t2), "color": INK, "dash": "5 4", "width": 2}]
    for est, col in ([("plugin", YELLOW)] if om == "wrong" else []) + [("dr", BLUE)]:
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

# ============================================================ calibrated band
s6 = pd.read_csv(RES / "sim6_intervals_summary.csv")
s6bal = s6[(s6.p == 50) & (s6.weights == "balancing")].iloc[0]
cb = json.loads((RES / "sim5_calibrated_band.json").read_text())
sys.path.insert(0, str(ROOT / "src"))
from drcurves.simulate import sim_high_dim  # noqa: E402
_, _Ab6, _, _f6 = sim_high_dim(n=200_000, p=50, seed=0)
_g6 = np.linspace(*np.quantile(_Ab6, [0.05, 0.95]), 40)
t6 = _f6(_g6) - _f6(_g6).mean()
CRIT = cb["critical"]
cov_cal = []
for fp in sorted((RES / "cache").glob("sim6_p50_*.json")):
    d6 = json.loads(fp.read_text())["balancing"]
    cov_cal.append([np.mean(np.abs(np.array(d6["est"]) - t6) <= c * np.array(d6["se"])) for c in (1.96, CRIT)])
cov_cal = np.mean(cov_cal, axis=0)
cal_panel = grouped_bars(["Default band\n1.96 standard errors", f"Calibrated band\n{CRIT:.1f} standard errors"],
                         [("Coverage", BLUE, list(cov_cal))], h=240, ylabel="Doses covered",
                         ref=(0.95, "95% target"), fmt=lambda v: f"{v:.0%}", vmax=1.08)

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

est_eff = {p: np.array([d[p][i_hi] - d[p][i_lo] for d in stores]) for p in ["partial", "none"]}
ylim6 = tuple(np.percentile(np.r_[est_eff["none"], est_eff["partial"]], [1, 99]) + np.array([-0.2, 0.2]))
shrink_chart = shrink_panel(st_n, est_eff["none"], est_eff["partial"], ORANGE, BLUE, ylim6,
                            ylabel=f"Effect of dose {f2(a_lo)} → {f2(a_hi)}")
BINS6 = [(0, 25, "<25"), (25, 60, "25–60"), (60, 150, "60–150"), (150, 10**6, "150+")]
size_cats = [f"{lab}\n{int(((st_n >= lo) & (st_n < hi)).sum())} stores" for lo, hi, lab in BINS6]
size_bars = grouped_bars(size_cats, [
    (lab, col, [rmse_by(eff_err[key], (st_n >= lo) & (st_n < hi)) for lo, hi, _ in BINS6])
    for key, lab, col in [("partial", "Partial pooling", BLUE), ("none", "No pooling", ORANGE),
                          ("complete", "Complete pooling", GREY), ("doubleml", "DoubleML per store", YELLOW)]],
    ylabel="RMSE of the store's dose effect")

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
LEG7 = legend([("No pooling", ORANGE, ""), ("Partial pooling", BLUE, "")])
LEG8 = legend([("Partial pooling", BLUE, ""), ("No pooling", ORANGE, ""), ("Complete pooling", GREY, ""), ("DoubleML per store", YELLOW, "")])
LEG6 = legend([("Truth (zero error)", INK, "5 4"), ("DR error, balancing weights", BLUE, "")])
LEG4 = legend([("True curve", INK, "5 4"), ("Estimate and its 95% bands", BLUE, "")])

explorer_data = json.dumps({"grid": s3["grid"], "stores": [
    {k: d[k] for k in ["store", "region", "n", "truth", "none", "complete", "partial", "partial_se"]} for d in stores]})

# =================================================================== article
LEG1 = legend([("True curve", INK, "5 4"), ("DR curve (95% band)", BLUE, ""), ("DoubleML PLR", ORANGE, "")])
LEG2 = legend([("Truth (zero error)", INK, "5 4"), ("S-learner", YELLOW, ""), ("Doubly robust (90% of runs)", BLUE, "")])
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
<title>Estimating Continuous Treatment Effects: Doubly Robust Dose-Response Curves by Ryan O'Sullivan</title>
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
  <h1>Estimating Continuous Treatment Effects: Doubly Robust Dose-Response Curves</h1>
  <p class="tagline">A simulation study of where DoubleML's partially linear model falls short for continuous treatments, and a doubly robust alternative that estimates the whole curve.</p>
  <p class="byline">Ryan O'Sullivan · Code on <a href="{REPO}" style="color:#cbd5e1">GitHub</a></p>
</header>

<nav class="snav">
  <a href="#plr" class="active">1 · One slope vs a curve</a>
  <a href="#shape">2 · Nonlinear effects</a>
  <a href="#robust">3 · Robustness</a>
  <a href="#confounders">4 · Many confounders</a>
  <a href="#method">5 · Intervals</a>
  <a href="#many">6 · Many curves</a>
  <a href="#faq">Questions</a>
  <a href="#together">Conclusion</a>
</nav>

<main>

<p>Not all treatments are binary: marketing spend, price, discount depth, app notifications, customer service response times. Rather than asking "did this treatment work?", we often need to know "what's the right amount?". When the treatment is continuous and chosen rather than randomised, most analysts reach for double machine learning, usually the partially linear model in <a href="https://docs.doubleml.org/" target="_blank" rel="noopener">DoubleML</a>. It is a good default for a binary treatment, but for a continuous one it has three limitations that matter in practice:</p>

<ul class="pts">
  <li><strong>It returns one number.</strong> The effect is assumed linear, so a rise-then-fall or a saturating curve comes back as a single slope.</li>
  <li><strong>Its default score isn't doubly robust.</strong> With a wrong treatment model the estimate is biased, however good the outcome model is.</li>
  <li><strong>It gives one answer per population.</strong> If you need a curve for each of 200 groups, looping over them gives noisy curves for the small ones.</li>
</ul>

<p>This article runs one simulation per limitation and shows an estimator that addresses all three: the doubly robust dose-response estimator of <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Kennedy et al. (2017)</a>, with a second stage that partially pools curves across a hierarchy.</p>

<!-- 1 -->
<section id="plr">
  <h2>1. One slope versus a whole curve</h2>

  <p>Let's start with the partially linear model. DoubleML's <code>DoubleMLPLR</code> fits</p>

  <div class="eqn">Y = θD + g(X) + ε</div>

  <p>where D is the treatment, X the confounders and g(X) the effect of the confounders on the outcome. DoubleML estimates θ with two machine learning models: an <strong>outcome model</strong>, which predicts Y from X, and a <strong>treatment model</strong>, which predicts D from X. It then regresses what's left of Y on what's left of D (this is called partialling out). Models like these are called nuisance models: we need them, but don't care about them for their own sake. Cross-fitting (each row's predictions come from models trained on other rows) and an orthogonal score (an estimating equation that is insensitive to small errors in the nuisance models) stop the regularisation in these models from biasing θ, which is what makes its confidence interval valid. The catch is the θD term: it assumes the effect is linear, so every extra unit of treatment has the same effect wherever you start.</p>

  <p>If the true effect is a curve, you still get a single slope back: roughly the curve's average slope across the doses in your data.</p>

  <p>The alternative is to estimate the whole curve: for every dose a, the average outcome if everyone received dose a. It takes three steps:</p>

  <div class="flow">
    <div class="step"><div class="n">Step 1</div><div class="t">Cross-fit two nuisance models</div><p>An outcome model μ(x, a) = E[Y | X, A], and a generalised propensity score (GPS) π(a | x): the density of the dose given X.</p></div>
    <div class="step"><div class="n">Step 2</div><div class="t">Build a pseudo-outcome</div><p>Combine them into one number per row, ξ, whose average at each dose is the causal curve.</p></div>
    <div class="step"><div class="n">Step 3</div><div class="t">Smooth, with pooling</div><p>Regress ξ on the dose with penalised splines, sharing strength across a hierarchy of groups.</p></div>
  </div>

  <p>Step 2 is where the robustness comes from. It starts with a regression estimate, the outcome model's prediction at dose a averaged over everyone, which on its own is an S-learner. It then adds a correction: the residuals of units that actually got a dose near a, weighted by how unusual that dose was for them so that they represent everyone. If the outcome model is right, the correction is just noise; if it's wrong, the correction removes the error, provided the model of who gets which dose is right. One of the two has to be right, not both: that's double robustness.<sup>1</sup></p>

  <p>Step 3 turns these noisy per-row numbers into a smooth curve and, when there are groups, lets small groups borrow strength from larger ones.</p>

  <p class="fig-cap" style="padding:0"><sup>1</sup> The pseudo-outcome is ξ = (Y − μ(X, A)) · ϖ(A)/π(A | X) + m(A), where μ is the outcome model, π the density of the dose given X, ϖ the overall density of the dose and m(a) the outcome model averaged over X at dose a (Kennedy et al. 2017). For a binary treatment this is the AIPW estimator behind DoubleML's <code>DoubleMLIRM</code>.</p>
</section>

<!-- 2 -->
<section id="shape">
  <p class="s-label">Simulation 1</p>
  <h2>2. Nonlinear effects</h2>

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

  <p>DoubleML's partially linear model has two scores. The default, partialling out, only protects against a wrong outcome model. If the treatment model is wrong, θ is biased however good the outcome model is (<a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">Valentin et al. 2025</a>). The IV-type score is doubly robust, but only for the single slope θ. What we want is a doubly robust <em>curve</em>, which is what the pseudo-outcome from section 1 provides.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>3,000 units and 50 confounders: four with strong nonlinear effects on both the dose and the outcome, and 46 weak ones.</li>
    <li>"Right" nuisance models are correctly specified. A "wrong" model treats every confounder as linear.</li>
    <li>The true effect is the inverted U.</li>
  </ul>

  <p>Each dataset gets a correct and a wrong version of the outcome model and of the dose model. The S-learner uses the outcome model alone, so it shows what the wrong outcome model costs you.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimation error with right and wrong nuisance models</div>
    </div>
    <div class="fig-body">{LEG2}<div class="panels three">{''.join(sim2_panels)}</div></div>
    <p class="fig-cap">Each panel shows each estimator's error: its average estimated curve over {N_REPS2} simulated datasets, minus the true curve. The dashed line at zero is the truth, so the closer to it the better. The blue band shows where 90% of the individual DR curves fall. Curves are centred.</p>
  </div>

  <ul class="pts">
    <li><strong>Outcome model wrong:</strong> the S-learner misses the curve, but DR stays close, because the weighted residuals correct most of the error.</li>
    <li><strong>Dose model wrong:</strong> DR stays close.</li>
    <li><strong>Both wrong:</strong> DR misses too. It needs at least one model right.</li>
  </ul>

  <p>DR's remaining error sits at the ends of the dose range, where the wrong outcome model is most wrong and the fewest units are available to correct it.</p>

</section>

<!-- 4 -->
<section id="confounders">
  <p class="s-label">Simulation 2, continued</p>
  <h2>4. Dealing with many confounders</h2>

  <p>Real data rarely has a handful of confounders. A customer table has dozens of features: tenure, spend history, products held, past contacts. Most of them nudge both the dose a customer gets and their outcome a little. You don't know which matter or how, so both nuisance models end up as flexible, regularised machine learning models. Regularisation is where the trouble starts.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>The same data as section 3: 3,000 units and 50 confounders.</li>
    <li>All nuisance models are boosted trees, since in practice you don't know which model is right.</li>
    <li>Three estimators: the S-learner, DR with density weights, and DR with balancing weights.</li>
  </ul>

  <p>A boosted outcome model shrinks the dose's effect. Correlated features absorb part of it, so the S-learner's curve flattens. DR corrects this with weights, so everything rests on how good those weights are.</p>

  <p>The goal of the weights is to make the data look as if doses had been handed out at random. A unit that got an unusual dose for someone like them is rare evidence, so it counts more. A unit that got the dose you'd expect counts less. There are two ways to build those weights.</p>

  <p><strong>Density weights</strong> measure "unusual" directly. A boosted model predicts each unit's expected dose and how much it typically varies. From that you get the probability of the dose the unit actually received, and the weight is roughly one over it. The problem is the tails. When the model says a dose was very unlikely, the probability is close to zero, so the weight is huge. Small errors in a near-zero probability become very large errors in the weight. A handful of units end up dominating the curve.</p>

  <p><strong>Balancing weights</strong> skip the probability altogether. They target the property we actually want: after weighting, the dose shouldn't be predictable from the confounders. They find the set of weights that are as close to equal as possible, subject to the weighted dose being uncorrelated with every confounder (and its square) (<a href="https://doi.org/10.1515/jem-2021-0002" target="_blank" rel="noopener">Tübbicke 2022</a>). It's the same idea as reweighting a survey so its age and region mix matches the census. Because there's no probability to divide by, there's nothing to blow up.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimated curves with boosted nuisance models, 50 confounders</div>
    </div>
    <div class="fig-body">{LEG5}<div class="panels" style="grid-template-columns:1fr">{practice_panel}</div></div>
    <p class="fig-cap">Average estimated curve over {N_REPS2} simulated datasets, all with the same boosted outcome model. The blue band shows where 90% of the individual curves with balancing weights fall. Dashed black: the true curve. Lines that leave the chart are cut off at its edge.</p>
  </div>

  <ul class="pts">
    <li><strong>S-learner:</strong> roughly the right shape, but too flat.</li>
    <li><strong>DR with density weights:</strong> far off. A few units got doses the boosted model thought were nearly impossible, so their weights are huge and they drag the curve with them.</li>
    <li><strong>DR with balancing weights:</strong> recovers the shape, but its 95% intervals are too narrow and miss the true curve more often than they should.</li>
  </ul>
</section>

<!-- 5 -->
<section id="method">
  <h2>5. Intervals</h2>

  <p>A curve only helps a decision if you know how far to trust it. The DR curve comes with a 95% band from step 3. With five confounders (section 2) that band was close to right. With 50 confounders and balancing weights, it contained the true curve at only {pct(s6bal['pointwise'])} of doses.</p>

  <p>The band's width is fine: it matches how much the curve moves between datasets. The problem is that the curve is slightly off, because balancing removes most of the confounding, not all of it. A band centred in the wrong place misses, and more data won't fix that: the band shrinks, but the bias doesn't.</p>

  <p>The fix is to measure the miss on your own data: inject a known effect and see if you get it back.</p>
  <ol class="pts">
    <li>Keep your real confounders and doses.</li>
    <li>Replace the outcome with a synthetic one whose curve you chose.</li>
    <li>Rerun the whole pipeline 20–30 times and count how often the band misses.</li>
    <li>Use the multiplier that would have given 95% in place of 1.96.</li>
  </ol>

  <p>This is a plasmode simulation (<a href="https://doi.org/10.1016/j.csda.2013.10.018" target="_blank" rel="noopener">Franklin et al. 2014</a>). It works because the synthetic data keep your confounding, so the pipeline is off by about as much as on the real data.</p>

  <p>Treat the multiplier as a rough guide. The synthetic outcome comes from your own fitted model, so it's easier than reality, which pushes the multiplier down. Keeping the real doses can make weighting methods look more biased than they are (<a href="https://arxiv.org/abs/2504.11740" target="_blank" rel="noopener">Shaw et al. 2025</a>), which pushes it up.</p>

  <p>We ran it on one simulated dataset: {cb['n_reps']} runs gave a multiplier of {CRIT:.1f}. The chart applies it to all {int(s6bal['reps'])}.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">How often the 95% band contains the true curve, 50 confounders</div>
    </div>
    <div class="fig-body">{cal_panel}</div>
    <p class="fig-cap">Share of doses at which the band contains the true curve, averaged over {int(s6bal['reps'])} simulated datasets with 50 confounders and DR with balancing weights. The dashed line is the 95% a band should reach.</p>
  </div>

  <p>The default band falls well short of 95%; the calibrated one gets back to it.</p>

  <p>This calibrates the band dose by dose. For claims about the whole shape ("the curve peaks between x and y"), the same check underestimated the multiplier needed, so be more cautious there.</p>
</section>

<!-- 6 -->
<section id="many">
  <p class="s-label">Simulation 3</p>
  <h2>6. Many curves: borrow strength across the hierarchy</h2>

  <p>The decision is rarely one curve for the whole business. Pricing teams set discounts by category, marketing teams set budgets by country and channel, CRM teams set cadence by segment. The groups are nested (stores in regions, products in categories, campaigns in channels), and their sizes follow a long tail: a few big ones with plenty of data and many small ones with very little.</p>

  <p>There are three ways to handle the groups:</p>
  <ul class="pts">
    <li><strong>Complete pooling:</strong> one curve for everyone. Stable, but it ignores real differences between groups.</li>
    <li><strong>No pooling:</strong> a separate curve per group. A store with 12 observations is treated as if it knew as much as one with 500. Looping DoubleML over stores does the same, and it also refits the nuisance models on each store's handful of rows.</li>
    <li><strong>Partial pooling:</strong> in between. Each store's curve is the global curve, plus a regional deviation, plus a store deviation:</li>
  </ul>

  <div class="eqn">θ<sub>store</sub>(a) = B(a)′ ( β<sub>global</sub> + b<sub>region</sub> + b<sub>store</sub> )</div>

  <p>B(a) is a set of spline basis functions. A small store's deviation is shrunk toward its region, and a region's toward the global curve. A big store has enough data to overrule the shrinkage. How hard to shrink at each level is chosen from the data. Because the DR pseudo-outcome is just a number per row, the pooling sits on top of it unchanged: the DR step handles confounding, and the pooling handles the hierarchy.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>200 stores in 8 regions. Store sizes are long-tailed, from 8 to several hundred observations.</li>
    <li>Each store's true curve is the inverted U plus a regional and a store deviation.</li>
    <li>Each store has its own baseline, which raises both its doses and its outcomes, so there is store-level confounding.</li>
    <li>Four approaches: no pooling, complete pooling, partial pooling, and DoubleML fitted per store.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error in each store's dose effect, by store size</div>
    </div>
    <div class="fig-body">{LEG8}{size_bars}</div>
    <p class="fig-cap">Root mean squared error of each store's estimated change in outcome from moving the dose from {f2(a_lo)} to {f2(a_hi)}, grouped by the number of observations in the store.{" DoubleML failed to fit in " + str(n_fail) + " stores, which are left out." if n_fail else ""}</p>
  </div>

  <ul class="pts">
    <li><strong>No pooling and DoubleML per store:</strong> the error grows as stores get smaller, worst for DoubleML.</li>
    <li><strong>Complete pooling:</strong> the error stays high even in big stores, because every store gets the same curve.</li>
    <li><strong>Partial pooling:</strong> the lowest error in small stores, because it pulls them toward their region. In big stores it matches no pooling, because the data overrules the shrinkage.</li>
  </ul>

  <p>The 95% bands for each store's curve are a little short, covering the truth about {pct(s4b.loc[('bayes','all'),'pointwise'])} of the time.</p>
</section>



<!-- FAQ -->
<section id="faq">
  <h2>Frequently asked questions</h2>

  <details class="faq"><summary>When shouldn't I use this?</summary>
    <ul class="pts">
      <li><strong>Unmeasured confounding.</strong> Double robustness protects against a wrong model, not a missing variable.</li>
      <li><strong>Poor overlap.</strong> If some units only ever get low doses, the data can't say what high doses would do for them. Report the curve only over doses you actually see.</li>
      <li><strong>Dose set at the group level.</strong> If every customer in a store gets the store's discount, the store is the unit of analysis. Aggregate first.</li>
      <li><strong>A spike at zero.</strong> "Any versus none" and "how much" are different questions; analyse them separately.</li>
      <li><strong>Crossed groups.</strong> The pooling assumes nesting (stores within regions). Store × channel needs a different model.</li>
    </ul>
  </details>

  <details class="faq"><summary>Why not bin the treatment and use a discrete estimator?</summary>
    <p>DoubleML's <code>DoubleMLAPOS</code> estimates average potential outcomes for discrete treatment levels, so binning the dose is an option. It throws away the ordering of the doses, each bin needs its own propensity model and enough data, and the answer depends on where you put the cut points. Within a bin, units with different doses are treated as if they got the same one.</p>
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
    <li><strong>Shape.</strong> A single slope can't describe a curve. On an inverted U it can come out near zero, as if the dose didn't matter. Estimate the whole curve instead.</li>
    <li><strong>Robustness.</strong> The default partialling-out score breaks when the treatment model is wrong; the IV-type score fixes that, but only for θ. The DR pseudo-outcome survives either model being wrong, for the whole curve.</li>
    <li><strong>Many confounders.</strong> With dozens of confounders, the S-learner flattens the curve and density weights blow up. Balancing weights hold up.</li>
    <li><strong>Intervals.</strong> With many confounders the default band misses, because the curve is slightly biased. Calibrate it with a plasmode simulation on your own data.</li>
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
    <li>Franklin, J. M., Schneeweiss, S., Polinski, J. M. and Rassen, J. A. (2014). <a href="https://doi.org/10.1016/j.csda.2013.10.018" target="_blank" rel="noopener">Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases</a>. <em>Computational Statistics &amp; Data Analysis</em>, 72, 219–226.</li>
    <li>Gelman, A. and Hill, J. (2007). <em>Data Analysis Using Regression and Multilevel/Hierarchical Models</em>. Cambridge University Press.</li>
    <li>Hirano, K. and Imbens, G. (2004). The propensity score with continuous treatments. In <em>Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives</em>, Wiley.</li>
    <li>Kennedy, E., Ma, Z., McHugh, M. and Small, D. (2017). <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Non-parametric methods for doubly robust estimation of continuous treatment effects</a>. <em>Journal of the Royal Statistical Society: Series B</em>, 79(4), 1229–1245.</li>
    <li>Kim, Y.-J. and Gu, C. (2004). Smoothing spline Gaussian regression: more scalable computation via efficient approximation. <em>Journal of the Royal Statistical Society: Series B</em>, 66(2), 337–356.</li>
    <li>Klosin, S. (2021). <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Automatic double machine learning for continuous treatment effects</a>. arXiv:2104.10334.</li>
    <li>Marra, G. and Wood, S. (2012). <a href="https://doi.org/10.1111/j.1467-9469.2011.00760.x" target="_blank" rel="noopener">Coverage properties of confidence intervals for generalized additive model components</a>. <em>Scandinavian Journal of Statistics</em>, 39(1), 53–74.</li>
    <li>Shaw, P. A., Gruber, S., Williamson, B. D., Desai, R., Shortreed, S. M., Krakauer, C., Nelson, J. C. and van der Laan, M. J. (2025). <a href="https://arxiv.org/abs/2504.11740" target="_blank" rel="noopener">A cautionary note for plasmode simulation studies in the setting of causal inference</a>. arXiv:2504.11740.</li>
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
