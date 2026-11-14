"""Sim 1 - DoubleML's PLR gives one number (theta) for a continuous treatment.
When the true dose-response is nonlinear that number hides the shape - and its
value depends on where the treatment happens to vary, not on the effect alone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _style import C, INK, plt  # noqa: E402
from drcurves import default_model, fit_dose_response  # noqa: E402
from drcurves.simulate import sim_nonlinear  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
hgb = lambda: HGB(max_iter=300, learning_rate=0.05, random_state=0)  # noqa: E731

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
rows, curves_out = [], {}
for ax, curve in zip(axes, ["inverted_u", "saturating"]):
    gen = "inverted_u_centred" if curve == "inverted_u" else curve
    X, A, Y, f = sim_nonlinear(n=4000, curve=gen, seed=1)

    np.random.seed(0)   # DoubleML draws its fold split from numpy's global RNG
    plr = DoubleMLPLR(DoubleMLData.from_arrays(X, Y, A), default_model(), default_model(), n_folds=5)
    plr.fit()
    theta, theta_se = plr.coef[0], plr.se[0]

    model = fit_dose_response(X, A, Y)  # package default: spline-ridge + boosting stack
    grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 100)
    est, se = model.curve(grid)

    # put everything on the same vertical anchor: E[Y(a)] up to a constant
    truth = f(grid)
    shift_true = truth.mean()
    est_c = est - est.mean() + shift_true
    plr_line = theta * (grid - grid.mean()) + shift_true

    avg_deriv = np.mean(np.gradient(f(np.sort(A)), np.sort(A)))
    rows.append(dict(curve=curve, plr_theta=theta, plr_se=theta_se,
                     true_avg_derivative=avg_deriv,
                     dr_curve_rmse=np.sqrt(np.mean((est_c - truth) ** 2)),
                     plr_line_rmse=np.sqrt(np.mean((plr_line - truth) ** 2))))

    curves_out[curve] = dict(grid=grid.round(4).tolist(), truth=truth.round(4).tolist(),
                             dr=est_c.round(4).tolist(), dr_se=se.round(4).tolist(),
                             plr=plr_line.round(4).tolist(), theta=float(theta), theta_se=float(theta_se),
                             a_quantiles=np.quantile(A, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist())
    ax.fill_between(grid, est_c - 1.96 * se, est_c + 1.96 * se, color=C[0], alpha=0.18, lw=0)
    ax.plot(grid, truth, color=INK, ls="--", lw=2, label="True E[Y(a)]")
    ax.plot(grid, est_c, color=C[0], label="DR curve (this repo)")
    ax.plot(grid, plr_line, color=C[1], label=f"DoubleML PLR, θ = {theta:.2f}")
    ax.set_title({"inverted_u": "Inverted-U effect", "saturating": "Saturating effect"}[curve])
    ax.set_xlabel("Treatment a")
    ax.set_ylabel("Outcome (centred)")
axes[0].legend(loc="lower center")
fig.suptitle("Sim 1 · One slope cannot describe a curve", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "sim1_nonlinear.png")

# same causal curve, different treatment policy: does theta stay put?
policy = []
for shift in [-0.5, 0.0, 0.5]:
    X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u_centred", seed=1, shift=shift)
    np.random.seed(0)
    plr = DoubleMLPLR(DoubleMLData.from_arrays(X, Y, A), default_model(), default_model(), n_folds=5).fit()
    policy.append(dict(shift=shift, mean_a=float(A.mean()), theta=float(plr.coef[0]),
                       se=float(plr.se[0])))
curves_out["policy_shift"] = policy
print(pd.DataFrame(policy).round(3).to_string(index=False))
(ROOT / "results" / "sim1_curves.json").write_text(json.dumps(curves_out))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "sim1_nonlinear.csv", index=False)
print(res.round(3).to_string(index=False))
