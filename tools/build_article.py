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
    sub = ""
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

def practice(prefix, title):
    return line_panel(g2, [
        {"y": s2c[f"{prefix}|slearner"].mean(0), "color": YELLOW, "width": 2},
        {"y": s2c[f"{prefix}|dr_density"].mean(0), "color": ORANGE, "width": 2},
        {"y": s2c[f"{prefix}|dr_balancing"].mean(0), "color": BLUE, "width": 2.4},
        {"y": t2, "color": INK, "dash": "5 4", "width": 2},
    ], title, "", ylim=ylim2)


practice_panels = practice("B", "Boosting alone") + practice("D", "Stacked: splines + boosting")

# ============================================================ package comparison
cmp = pd.read_csv(RES / "compare_packages_summary.csv").set_index(["setting", "method"])
_hard = []
for fp in sorted((RES / "cache").glob("hard_*.json")):
    for k, v in json.loads(fp.read_text()).items():
        _hard.append(dict(method=k, **v))
_hd = pd.DataFrame(_hard)
hard = _hd.groupby("method")[["rmse", "cover", "width"]].mean()          # all datasets (within-drcurves comparisons)
N_HARD = len(list((RES / "cache").glob("hard_*.json")))
# package comparison on matched datasets: those where every package ran
_files = sorted((RES / "cache").glob("hard_*.json"))
_ok = [json.loads(fp.read_text()) for fp in _files]
_ok = [d for d in _ok if all(k in d for k in ("npcausal (R)", "causal-curve TMLE", "stacked, density"))]
N_HARD_MATCH, N_HARD_FAIL = len(_ok), N_HARD - len(_ok)
hard_match = pd.Series({k: np.mean([d[k]["rmse"] for d in _ok]) for k in ("npcausal (R)", "causal-curve TMLE", "stacked, density")})
hard_match_w = pd.Series({k: np.mean([d[k]["width"] for d in _ok]) for k in ("npcausal (R)", "causal-curve TMLE", "stacked, density")})
for key, hk in [("npcausal (R)", "npcausal (R)"), ("causal-curve TMLE", "causal-curve TMLE"),
                ("DR, stacked, density weights", "stacked, density")]:
    for col in ("rmse", "cover", "width"):
        cmp.loc[("hard", key), col] = (hard_match[hk] if col == "rmse" else hard_match_w[hk] if col == "width"
                                       else hard.loc[hk, col])
CMP = [("npcausal (R)", "npcausal (R)", AQUA), ("causal-curve TMLE", "causal-curve (Python)", GREY),
       ("DR, stacked, density weights", "drcurves (Python)", BLUE)]
cmp_bars = grouped_bars(["5 confounders", "50 confounders", "Harder data"],
                        [(lab, col, [cmp.loc[(st, key), "rmse"] for st in ("p5", "p50", "hard")]) for key, lab, col in CMP],
                        ylabel="RMSE of the curve")
LEG9 = legend([(lab, col, "") for _, lab, col in CMP])
hardall = pd.Series({k: np.mean([d[k]["rmse"] for d in _ok]) for k in _ok[0]})   # matched datasets
_ratios = [cmp.loc[(st, "causal-curve TMLE"), "width"] / cmp.loc[(st, k), "width"]
           for st in ("p5", "p50", "hard") for k in ("npcausal (R)", "DR, stacked, density weights")]
CC_RATIO = f"{int(np.floor(min(_ratios) / 5) * 5) or 5} to {int(np.ceil(max(_ratios) / 10) * 10)}"

# ============================================================ DiD
_did = [json.loads(fp.read_text()) for fp in sorted((RES / "cache").glob("did_*.json"))]
N_DID = len(_did)
did = {k: (np.mean([d[k]["rmse"] for d in _did]), np.mean([d[k]["cover"] for d in _did]))
       for k in ("levels", "lagged", "did", "placebo")}
did_bars = grouped_bars(["Adjust for X", "Adjust for X and\nthe pre-period outcome", "DiD: fit on the change"],
                        [("Error", BLUE, [did[k][0] for k in ("levels", "lagged", "did")])],
                        h=240, ylabel="RMSE of the curve")


cov_po = p2("partialling out", "right", "wrong", "coverage")
plr_cov_text = "never contained" if cov_po == 0 else f"contained only {pct(cov_po)} of the time"
plr_bias_po = p2("partialling out", "right", "wrong", "bias")
cov_iv = p2("IV-type", "right", "wrong", "coverage")

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


def rmse_by(err, mask):  # noqa: E302
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
dml_big = rmse_by(eff_err["doubleml"], st_n >= 150)
size_cats = [f"{lab} customers\n{int(((st_n >= lo) & (st_n < hi)).sum())} stores" for lo, hi, lab in BINS6]
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
  <a href="#confounders">4 · Nuisance models</a>
  <a href="#many">5 · Many curves</a>
  <a href="#packages">6 · Existing packages</a>
  <a href="#did">7 · Assumptions</a>
  <a href="#faq">Questions</a>
  <a href="#together">Conclusion</a>
</nav>

<main>

<p>Not all treatments are binary: marketing spend, price, discount depth, app notifications, customer service response times. Rather than asking "did this treatment work?", we often need to know "what's the right amount?". When the treatment is continuous and chosen rather than randomised, a common choice is double machine learning (<a href="https://arxiv.org/abs/1608.00060" target="_blank" rel="noopener">Chernozhukov et al. 2018</a>), usually the partially linear model in <a href="https://docs.doubleml.org/" target="_blank" rel="noopener">DoubleML</a> (<a href="https://jmlr.org/papers/v23/21-0862.html" target="_blank" rel="noopener">Bach et al. 2022</a>). For a binary treatment, DoubleML's interactive model is doubly robust; for a continuous treatment, the partially linear model is the usual choice, and it has three limitations that matter in practice:</p>

<ul class="pts">
  <li><strong>It returns one number.</strong> The effect is assumed linear, so a rise-then-fall or a saturating curve comes back as a single slope.</li>
  <li><strong>Its default score isn't doubly robust.</strong> With a wrong treatment model the estimate is biased, however good the outcome model is.</li>
  <li><strong>It gives one answer per population.</strong> If you need a curve for each of 200 groups, looping over them gives noisy slopes for the small ones.</li>
</ul>

<p>This article runs one simulation per limitation and shows an estimator that addresses all three: the doubly robust dose-response estimator of <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Kennedy et al. (2017)</a>, with a second stage that partially pools curves across a hierarchy. It then looks at three practical questions: which machine learning models to use, how existing packages compare, and what to do when a confounder is unmeasured.</p>

<!-- 1 -->
<section id="plr">
  <h2>1. One slope versus a whole curve</h2>

  <p>Let's start with the partially linear model. DoubleML's <code>DoubleMLPLR</code> fits</p>

  <div class="eqn">Y = θD + g(X) + ε</div>

  <p>where D is the treatment, X the confounders and g(X) the effect of the confounders on the outcome. DoubleML estimates θ with two machine learning models: an <strong>outcome model</strong>, which predicts Y from X, and a <strong>treatment model</strong>, which predicts D from X. It then regresses what's left of Y on what's left of D (this is called partialling out).</p>

  <p>Models like these are called nuisance models: we need them, but don't care about them for their own sake. Cross-fitting (each row's predictions come from models trained on other rows) and an orthogonal score (an estimating equation that is insensitive to small errors in the nuisance models) stop the regularisation in these models from biasing θ, which is what makes its confidence interval valid. The catch is the θD term: it assumes the effect is linear, so every extra unit of treatment has the same effect wherever you start.</p>

  <p>If the true effect is a curve, you still get a single slope back: roughly the curve's average slope across the doses in your data.</p>

  <p>The alternative is to estimate the whole curve: for every dose a, the average outcome if everyone received dose a. It takes three steps:</p>

  <div class="flow">
    <div class="step"><div class="n">Step 1</div><div class="t">Cross-fit two nuisance models</div><p>An outcome model μ(x, a) = E[Y | X, A]; unlike DoubleML's outcome model, this one includes the dose as an input. And a generalised propensity score (GPS; <a href="https://doi.org/10.1002/0470090456.ch7" target="_blank" rel="noopener">Hirano and Imbens 2004</a>) π(a | x): the density of the dose given X.</p></div>
    <div class="step"><div class="n">Step 2</div><div class="t">Build a pseudo-outcome</div><p>Combine them into one number per row, ξ, whose average at each dose is the causal curve.</p></div>
    <div class="step"><div class="n">Step 3</div><div class="t">Smooth, with pooling</div><p>Regress ξ on the dose with penalised splines (<a href="https://doi.org/10.1214/ss/1038425655" target="_blank" rel="noopener">Eilers and Marx 1996</a>), sharing strength across a hierarchy of groups.</p></div>
  </div>

  <p>Step 2 is where the robustness comes from. It starts with a regression estimate, the outcome model's prediction at dose a averaged over everyone, which on its own is an S-learner. It then adds a correction: the residuals of units that actually got a dose near a, weighted by how unusual that dose was for them so that they represent everyone. If the outcome model is right, the correction is just noise; if it's wrong, the correction removes the error, provided the model of who gets which dose is right. One of the two has to be right, not both: that's double robustness.<sup>1</sup></p>

  <p>Step 3 turns these noisy per-row numbers into a smooth curve and, when there are groups, lets small groups borrow strength from larger ones.</p>

  <p class="fig-cap" style="padding:0"><sup>1</sup> The pseudo-outcome is ξ = (Y − μ(X, A)) · ϖ(A)/π(A | X) + m(A), where μ is the outcome model, π the density of the dose given X, ϖ the overall density of the dose and m(a) the outcome model averaged over X at dose a (Kennedy et al. 2017). For a binary treatment this is the AIPW estimator behind DoubleML's <code>DoubleMLIRM</code>.</p>
</section>

<!-- 2 -->
<section id="shape">
  <h2>2. Nonlinear effects</h2>

  <p>Two nonlinear shapes are common:</p>

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
    <li>Both methods use the same learner for every nuisance model: a stack of gradient boosting and a spline regression (section 4 explains why).</li>
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
  <h2>3. Robustness to a wrong nuisance model</h2>

  <p>Both nuisance models are approximations, and with a continuous dose the model of who gets which dose is especially easy to get wrong: it has to describe the whole spread of doses each unit might get, not just the average one. Doses are often set by unwritten business rules (pricing tiers, budgets split by last year's revenue, a cap on notifications), and some segments see a narrow range of doses while others see a wide one. The outcome model can be wrong too. So the test is what happens when one of them is wrong.</p>

  <p>DoubleML's partially linear model has two scores (estimating equations). The default, partialling out, only protects against a wrong outcome model: if the treatment model is wrong, θ is biased however good the outcome model is (<a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">Valentin et al. 2025</a>). The alternative IV-type score is doubly robust, but only for the single slope θ. What we want is a doubly robust <em>curve</em>, which is what the pseudo-outcome from section 1 provides.</p>

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

  <p>With the outcome model wrong, the S-learner misses the curve, while DR stays much closer, because the weighted residuals correct most of the error. With the dose model wrong, DR stays close. With both wrong, DR misses too: it needs at least one model right. DR's remaining error sits at the ends of the dose range, where the wrong outcome model is most wrong and the fewest units are available to correct it.</p>

</section>

<!-- 4 -->
<section id="confounders">
  <h2>4. Choosing the nuisance models</h2>

  <p>Real data rarely has a handful of confounders. A customer table has dozens of features: tenure, spend history, products held, past contacts. Most of them nudge both the dose a customer gets and their outcome a little. You don't know which matter or how, so both nuisance models end up as flexible, regularised machine learning models. The choice of learner changed the answer a lot.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>The same data as section 3: 3,000 units and 50 confounders.</li>
    <li>Each estimator is fitted twice, changing only the machine learning model used for the nuisance models:
      <ul>
        <li><strong>Boosting alone:</strong> gradient-boosted trees.</li>
        <li><strong>A stack:</strong> two models fitted side by side, boosted trees and a spline regression (a smooth curve in each confounder, like a GAM). Their predictions are combined as a weighted average, with the weights chosen by cross-validation.</li>
      </ul>
    </li>
    <li>Three estimators: the S-learner, DR with density weights, and DR with balancing weights. Balancing weights skip the density: they are chosen so that, after weighting, the dose is uncorrelated with each confounder (<a href="https://doi.org/10.1515/jem-2021-0002" target="_blank" rel="noopener">Tübbicke 2022</a>).</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Estimated curves by choice of learner, 50 confounders</div>
    </div>
    <div class="fig-body">{LEG5}<div class="panels">{practice_panels}</div></div>
    <p class="fig-cap">Average estimated curve over {N_REPS2} simulated datasets. Left: every nuisance model is boosted trees. Right: every nuisance model is the stack. Dashed black: the true curve. Curves are centred; the density-weight curve on the left runs off the chart.</p>
  </div>

  <ul class="pts">
    <li><strong>Boosting alone, S-learner:</strong> the curve is too flat, because boosted trees shrink the dose's effect.</li>
    <li><strong>Boosting alone, weights:</strong> density weights are far off, because the boosted dose model gives some doses a near-zero probability, so their weights explode. Balancing weights, which need no density, do best.</li>
    <li><strong>Stacked:</strong> all three recover the curve. The spline regression captures the smooth relationships that trees can only approximate with steps.</li>
  </ul>

  <p>In practice: with the stack, density weights work well; if you use boosting alone, use balancing weights instead. That doesn't make the stack better everywhere: section 6 tests it on harder data, where trees have the edge.</p>
</section>

<!-- 5 -->
<section id="many">
  <h2>5. Many curves: borrow strength across the hierarchy</h2>

  <p>The decision is rarely one curve for the whole business. Pricing teams set discounts by category, marketing teams set budgets by country and channel, CRM teams set cadence by segment. The groups are nested (stores in regions, products in categories, campaigns in channels), and their sizes follow a long tail: a few big ones with plenty of data and many small ones with very little.</p>

  <p>There are three ways to handle the groups:</p>
  <ul class="pts">
    <li><strong>Complete pooling:</strong> one curve for everyone. Stable, but it ignores real differences between groups.</li>
    <li><strong>No pooling:</strong> a separate curve per group. A store with 12 customers is treated as if it knew as much as one with 500. Looping DoubleML over stores does the same, and it also refits the nuisance models on each store's handful of rows.</li>
    <li><strong>Partial pooling:</strong> in between (<a href="https://doi.org/10.1017/CBO9780511790942" target="_blank" rel="noopener">Gelman and Hill 2007</a>). Each store's curve is the global curve, plus a regional deviation, plus a store deviation:</li>
  </ul>

  <div class="eqn">θ<sub>store</sub>(a) = B(a)′ ( β<sub>global</sub> + b<sub>region</sub> + b<sub>store</sub> )</div>

  <p>B(a) is a set of spline basis functions. A small store's deviation is shrunk toward its region, and a region's toward the global curve. A big store has enough data to overrule the shrinkage. How hard to shrink at each level is chosen from the data. Because the DR pseudo-outcome is just a number per row, the pooling sits on top of it unchanged: the DR step handles confounding, and the pooling handles the hierarchy.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>200 stores in 8 regions. Each row is a customer who got a personalised discount (the dose); the outcome is their spend. Stores range from 8 to several hundred customers, with a long tail of small ones.</li>
    <li>Each store's true curve is the inverted U plus a regional and a store deviation.</li>
    <li>Each store has its own baseline, which raises both its doses and its outcomes, so there is store-level confounding.</li>
    <li>Four approaches: no pooling, complete pooling, partial pooling, and DoubleML fitted per store.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error in each store's dose effect, by number of customers in the store</div>
    </div>
    <div class="fig-body">{LEG8}{size_bars}</div>
    <p class="fig-cap">One simulated dataset. Root mean squared error of each store's estimated change in outcome between the 25th and 75th percentiles of the dose, grouped by the number of customers in the store.{" DoubleML failed to fit in " + str(n_fail) + " stores, which are left out." if n_fail else ""}</p>
  </div>

  <p>Without pooling, the error grows as stores get smaller. DoubleML per store is worst in all but the biggest stores. Even there, with 150 or more customers, its error stays around {dml_big:.1f}, because a straight line can't follow the curve; the extra error in small stores comes from refitting the nuisance models on a few dozen rows. Complete pooling's error stays high even in big stores, because every store gets the same curve. Partial pooling has the lowest error in small stores, because it pulls them toward their region; in big stores it matches no pooling, because the data overrules the shrinkage.</p>

</section>



<!-- 6 -->
<section id="packages">
  <h2>6. How does this compare with existing packages?</h2>

  <p>Two packages already estimate a single doubly robust curve with the pseudo-outcome from section 1:</p>
  <ul class="pts">
    <li><strong><a href="https://github.com/ehkennedy/npcausal" target="_blank" rel="noopener">npcausal</a> (R):</strong> Kennedy's own implementation. It fits the nuisance models with SuperLearner, an ensemble of smooth regressions and random forests, and chooses how much to smooth the curve from the data.</li>
    <li><strong><a href="https://github.com/ronikobrosly/causal-curve" target="_blank" rel="noopener">causal-curve</a> (Python):</strong> follows the same steps (its estimator is called TMLE), with gradient boosting and a fixed amount of smoothing.</li>
  </ul>

  <p>Neither cross-fits the nuisance models, and neither handles groups. I wrote <a href="{REPO}" target="_blank" rel="noopener">drcurves</a>, the open-source Python package used for every result in this article, partly because I wanted both. Since I'm comparing my own package, read the comparison below with that in mind.</p>

  <p>We ran all three on three settings, ten simulated datasets each:</p>
  <ul class="pts">
    <li>The data from section 2 (5 confounders) and section 3 (50 confounders). These are the simulations used to choose drcurves' default learner in section 4, so they favour it.</li>
    <li>Harder data that favours trees, added as a check: thresholds, interactions between confounders and heavy-tailed noise.</li>
  </ul>

  <p>Each package used its defaults, with two exceptions for npcausal: we gave it a range of bandwidths to choose from, and with 50 confounders we dropped the pairwise-interaction learner from its library, which didn't fit in memory.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error in the estimated curve, by package</div>
    </div>
    <div class="fig-body">{LEG9}{cmp_bars}</div>
    <p class="fig-cap">Root mean squared error between the estimated and the true curve, averaged over the simulated datasets in each setting. Curves are centred.{f" On the harder data, npcausal failed on {N_HARD_FAIL} of the {N_HARD} datasets, so all three are compared on the other {N_HARD_MATCH}." if N_HARD_FAIL else ""}</p>
  </div>

  <ul class="pts">
    <li><strong>Smooth data (5 and 50 confounders):</strong> drcurves is the most accurate, and its lead grows with more confounders. npcausal is close with 5 confounders but falls behind with 50, likely because of its library of learners rather than the cross-fitting (section 4 showed how much the learner matters).</li>
    <li><strong>Harder data:</strong> drcurves is still ahead of npcausal. Within drcurves, boosting alone with balancing weights was more accurate still ({hardall['boosted, balancing']:.2f} vs {hardall['stacked, density']:.2f}), but boosting alone with density weights failed badly ({hardall['boosted, density']:.1f}). No learner wins everywhere; the stack is rarely far from the best.</li>
    <li><strong>causal-curve</strong> trails throughout, and its intervals are {CC_RATIO} times wider than the others', so they always cover the truth but tell you little.</li>
  </ul>

  <p>These are our simulations, run with each package's default settings; different data could reorder them.</p>
</section>

<!-- 7 -->
<section id="did">
  <h2>7. When confounders are unmeasured: difference-in-differences</h2>

  <p>Everything so far assumes you've measured the confounders. Often you haven't. A customer's underlying complexity, a team's culture, a store manager's skill: stable traits that push both the dose and the outcome, and never appear in the data. No estimator fixes that, but with pre-period data a different assumption can.</p>

  <p>If you observe the outcome before the dose changed, and the hidden trait is stable over time, difference-in-differences removes it: fit the same method on the <em>change</em> in outcome rather than its level. To compare doses, rather than just treated against untreated, this needs a stronger form of parallel trends: units on different doses would have followed the same path had their doses been the same (<a href="https://arxiv.org/abs/2107.02637" target="_blank" rel="noopener">Callaway, Goodman-Bacon and Sant'Anna 2024</a>).</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>3,000 units, five observed confounders and one hidden, stable trait that raises both the dose and the outcome.</li>
    <li>Two periods before the dose, one after. The true effect is the inverted U.</li>
    <li>Three ways to use the same DR estimator: adjust for X; adjust for X and the pre-period outcome; or fit on the change in outcome (DiD).</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Error in the estimated curve with a hidden, stable trait</div>
    </div>
    <div class="fig-body">{did_bars}</div>
    <p class="fig-cap">Root mean squared error between the estimated and the true curve, averaged over {N_DID} simulated datasets. Curves are centred.</p>
  </div>

  <p>Adjusting for X alone leaves the trait confounding the curve. Adding the pre-period outcome helps, because it partly stands in for the trait, but the curve is still biased. Fitting on the change recovers it, because the trait cancels.</p>

  <p>The ranking flips if the dose responds to past outcomes rather than to a stable trait: then adjusting for the pre-period outcome is right and DiD is biased. You can't tell which from the data alone. A placebo helps: fit the curve on the change between two pre-periods, where the dose can't have had an effect yet. A clearly non-flat curve warns that parallel trends doesn't hold. Judge it by its size relative to the effect you care about, not by whether its band includes zero: even when parallel trends held in our simulation, about {pct(1 - did["placebo"][1])} of the placebo curve fell outside its 95% band.</p>
</section>

<!-- FAQ -->
<section id="faq">
  <h2>Frequently asked questions</h2>

  <details class="faq"><summary>Can I trust the intervals?</summary>
    <p>Treat them as optimistic. With the stacked default, the 95% bands contained the true curve 75-93% of the time across our simulations. Capping the most extreme weights, which trades a small bias for a lot of stability, is one likely reason. When the nuisance models are off the bands are too narrow, because a band doesn't account for bias. Check on your own data: keep your real confounders and doses, simulate an outcome with a known curve, rerun the whole pipeline 20 to 30 times, and count how often the band contains it. This is a plasmode simulation (<a href="https://doi.org/10.1016/j.csda.2013.10.018" target="_blank" rel="noopener">Franklin et al. 2014</a>), and if the bands fall short it tells you how much to widen them.</p>
  </details>

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
    <p>Yes. <a href="https://arxiv.org/abs/2004.03036" target="_blank" rel="noopener">Colangelo and Lee (2020)</a> estimate the curve one dose at a time with a kernel-localised doubly robust score, and <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Klosin (2021)</a> does it without estimating the density. Both are good options for a single population curve with careful inference. The pseudo-outcome approach here was chosen because the second stage can be any regression, which is what makes the hierarchical pooling simple.</p>
  </details>

  <details class="faq"><summary>Could I do the pooling in a Bayesian model instead?</summary>
    <p>Yes. The ridge penalties are Gaussian priors on the deviations, tuned by cross-validation rather than given hyperpriors. Fitting the same spline structure in PyMC or brms on the pseudo-outcomes gives full posteriors for each store's curve, at more computational cost.</p>
  </details>

</section>

<!-- Conclusion -->
<section id="together">
  <h2>Conclusion</h2>

  <p>With a continuous treatment, the partially linear model answers a narrower question than you usually need. The question is about a curve:</p>

  <ul class="pts">
    <li><strong>Shape.</strong> A single slope can't describe a curve. On an inverted U it can come out near zero, as if the dose didn't matter. Estimate the whole curve instead.</li>
    <li><strong>Robustness.</strong> With one nuisance model wrong, the DR curve removed most of the error the S-learner made, though some remained at the ends of the dose range.</li>
    <li><strong>Learners.</strong> Boosting alone flattens the curve and breaks density weights. Stack a smooth learner with boosting; no single learner wins everywhere.</li>
    <li><strong>Many curves.</strong> Partial pooling gives every group a curve, shrinks the small ones toward their parent, and leaves big ones close to their own data.</li>
    <li><strong>Assumptions.</strong> If a stable trait you can't measure drives both dose and outcome, fit the same method on the change in outcome (DiD), and run a placebo on the pre-periods.</li>
  </ul>

  <p>The simulations use one data-generating process per question, known confounders and moderate sample sizes. They do not show how often each problem occurs in your data. The questions to ask are the same, though: is the curve likely to bend, which nuisance model are you least sure of, how many groups need their own answer, and what are you assuming about the confounders you can't see?</p>

  <p>The package, <a href="{REPO}" target="_blank" rel="noopener">drcurves</a>, has the estimator, the simulations and the script that builds every chart and number on this page. Install it with <code>pip install git+{REPO}</code>.</p>

  <h3>References</h3>
  <ul class="refs">
    <li>Bach, P., Chernozhukov, V., Kurz, M. S. and Spindler, M. (2022). <a href="https://jmlr.org/papers/v23/21-0862.html" target="_blank" rel="noopener">DoubleML: an object-oriented implementation of double machine learning in Python</a>. <em>Journal of Machine Learning Research</em>, 23(53), 1–6.</li>
    <li>Chernozhukov, V. et al. (2018). <a href="https://arxiv.org/abs/1608.00060" target="_blank" rel="noopener">Double/debiased machine learning for treatment and structural parameters</a>. <em>The Econometrics Journal</em>, 21(1), C1–C68.</li>
    <li>Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024). <a href="https://arxiv.org/abs/2107.02637" target="_blank" rel="noopener">Difference-in-differences with a continuous treatment</a>. arXiv:2107.02637.</li>
    <li>Colangelo, K. and Lee, Y.-Y. (2020). <a href="https://arxiv.org/abs/2004.03036" target="_blank" rel="noopener">Double debiased machine learning nonparametric inference with continuous treatments</a>. arXiv:2004.03036.</li>
    <li>Eilers, P. and Marx, B. (1996). Flexible smoothing with B-splines and penalties. <em>Statistical Science</em>, 11(2), 89–121.</li>
    <li>Franklin, J. M., Schneeweiss, S., Polinski, J. M. and Rassen, J. A. (2014). <a href="https://doi.org/10.1016/j.csda.2013.10.018" target="_blank" rel="noopener">Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases</a>. <em>Computational Statistics &amp; Data Analysis</em>, 72, 219–226.</li>
    <li>Gelman, A. and Hill, J. (2007). <em>Data Analysis Using Regression and Multilevel/Hierarchical Models</em>. Cambridge University Press.</li>
    <li>Hirano, K. and Imbens, G. (2004). The propensity score with continuous treatments. In <em>Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives</em>, Wiley.</li>
    <li>Kennedy, E., Ma, Z., McHugh, M. and Small, D. (2017). <a href="https://arxiv.org/abs/1507.00747" target="_blank" rel="noopener">Non-parametric methods for doubly robust estimation of continuous treatment effects</a>. <em>Journal of the Royal Statistical Society: Series B</em>, 79(4), 1229–1245.</li>
    <li>Klosin, S. (2021). <a href="https://arxiv.org/abs/2104.10334" target="_blank" rel="noopener">Automatic double machine learning for continuous treatment effects</a>. arXiv:2104.10334.</li>
    <li>Tübbicke, S. (2022). <a href="https://doi.org/10.1515/jem-2021-0002" target="_blank" rel="noopener">Entropy balancing for continuous treatments</a>. <em>Journal of Econometric Methods</em>, 11(1), 71–89.</li>
    <li>Valentin, S., Detommaso, G., Li, Y., Opper, M. and Brueckner, M. (2025). <a href="https://neurips.cc/virtual/2025/124095" target="_blank" rel="noopener">On double robustness in double machine learning</a>. NeurIPS 2025 Workshop CauScien.</li>
  </ul>
</section>

</main>

<footer>Simulated data throughout. Every chart and number on this page is generated by <code>tools/build_article.py</code> in the drcurves repo from the outputs of the experiment scripts.</footer>

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
