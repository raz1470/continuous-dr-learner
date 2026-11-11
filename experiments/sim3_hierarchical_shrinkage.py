"""Sim 3 - Hundreds of curves: 8 regions x 25 stores, heavy-tailed store sizes.

Same DR pseudo-outcomes for every method; only the second stage changes:
  complete pooling : one curve for everyone
  no pooling       : every store fitted on its own data
  partial pooling  : store = global + region deviation + store deviation, shrunk
Score: RMSE of each store's curve against its true curve, by store size.
Also: per-store DoubleML PLR slopes, the "loop over categories" workaround.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _style import C, INK, INK2, plt  # noqa: E402
from drcurves import GPS, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa: E402
from drcurves.simulate import sim_hierarchical  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def hgb_cat(n_cat_cols):
    return HGB(max_iter=300, learning_rate=0.05, random_state=0,
               categorical_features=list(range(n_cat_cols)))


def features(df):
    reg = pd.Categorical(df.region).codes
    sto = pd.Categorical(df.store).codes
    return np.column_stack([reg, sto, df[["x0", "x1", "x2"]].to_numpy()])


rows, examples, plr_rows = [], None, []
for rep in range(REPS):
    df, truth, f0 = sim_hierarchical(seed=rep)
    X, A, Y = features(df), df.A.to_numpy(), df.Y.to_numpy()
    gps = GPS(mean_model=hgb_cat(2), scale_model=HGB(max_iter=100, learning_rate=0.05,
              min_samples_leaf=100, categorical_features=[0, 1]))
    nuis = cross_fit_nuisances(X, A, Y, HGB(max_iter=300, learning_rate=0.05, random_state=0,
                                            categorical_features=[0, 1]),
                               gps, groups=df.store.to_numpy(), random_state=rep)
    xi = pseudo_outcomes(nuis, Y, "dr")
    grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 40)
    groups = df[["region", "store"]]
    fits = {p: HierarchicalDoseResponse(pooling=p).fit(A, xi, groups)
            for p in ["complete", "none", "partial"]}
    sizes = df.groupby("store").size()
    for (r, s), fs in truth.items():
        t = fs(grid)
        for p, m in fits.items():
            est, _ = m.curve(grid, (r, s))
            # curves are compared up to a constant (store baselines are not the target)
            err = (est - est.mean()) - (t - t.mean())
            rows.append(dict(rep=rep, region=r, store=s, n=sizes[s], pooling=p,
                             rmse=np.sqrt(np.mean(err ** 2))))
    if rep == 0:
        examples = (grid, fits, truth, sizes, df)
        print("tuned penalties:", {p: (m.lam_smooth_, m.lam_levels_) for p, m in fits.items()})
        # the per-category workaround: one DoubleML PLR per store
        for s, d in df.groupby("store"):
            Xs = d[["x0", "x1", "x2"]].to_numpy()
            try:
                plr = DoubleMLPLR(DoubleMLData.from_arrays(Xs, d.Y.to_numpy(), d.A.to_numpy()),
                                  HGB(max_iter=100, min_samples_leaf=5), HGB(max_iter=100, min_samples_leaf=5),
                                  n_folds=min(5, max(2, len(d) // 10))).fit()
                plr_rows.append(dict(store=s, n=len(d), theta=plr.coef[0], se=plr.se[0]))
            except Exception as e:  # noqa: BLE001
                plr_rows.append(dict(store=s, n=len(d), theta=np.nan, se=np.nan, error=str(e)[:60]))
    print(f"rep {rep} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "sim3_store_rmse.csv", index=False)
res["size_band"] = pd.cut(res.n, [0, 25, 60, 150, 10_000], labels=["<25", "25–60", "60–150", "150+"])
summ = res.pivot_table(index="size_band", columns="pooling", values="rmse", aggfunc="mean",
                       observed=True).round(3)
summ["n_stores"] = res[(res.pooling == "partial")].groupby("size_band", observed=True).size() / REPS
summ.to_csv(ROOT / "results" / "sim3_summary_by_size.csv")
overall = res.groupby("pooling").rmse.mean().round(3)
print(summ.to_string())
print(overall.to_string())
pd.DataFrame(plr_rows).to_csv(ROOT / "results" / "sim3_plr_per_store.csv", index=False)

# ------------------------------------------------------------------ figures
grid, fits, truth, sizes, df = examples
out = {"grid": grid.round(4).tolist(), "stores": []}
for (r, s), fs in truth.items():
    t = fs(grid)
    item = {"region": r, "store": s, "n": int(sizes[s]), "truth": (t - t.mean()).round(4).tolist()}
    for p, m in fits.items():
        e, se = m.curve(grid, (r, s))
        item[p] = (e - e.mean()).round(4).tolist()
        if p == "partial":
            item["partial_se"] = se.round(4).tolist()
    out["stores"].append(item)
out["penalties"] = {p: [float(m.lam_smooth_), [float(v) for v in m.lam_levels_]] for p, m in fits.items()}
(ROOT / "results" / "sim3_curves.json").write_text(json.dumps(out))
picks = [sizes.idxmin(), sizes.sort_values().index[len(sizes) // 4],
         sizes.sort_values().index[len(sizes) // 2], sizes.idxmax()]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, s in zip(axes, picks):
    r = df.loc[df.store == s, "region"].iloc[0]
    t = truth[(r, s)](grid)
    ax.plot(grid, t - t.mean(), color=INK, ls="--", label="True store curve")
    for p, col, lab in [("none", C[1], "No pooling"), ("complete", INK2, "Complete pooling"),
                        ("partial", C[0], "Partial pooling")]:
        e, se = fits[p].curve(grid, (r, s))
        if p == "partial":
            ax.fill_between(grid, e - e.mean() - 1.96 * se, e - e.mean() + 1.96 * se, color=col,
                            alpha=0.15, lw=0)
        ax.plot(grid, e - e.mean(), color=col, label=lab, lw=2.4 if p == "partial" else 1.6)
    ax.set_title(f"{s}  (n = {sizes[s]})")
    ax.set_xlabel("Treatment a")
    ax.set_ylim(-4.5, 3)
axes[0].set_ylabel("Store curve (centred)")
axes[0].legend(loc="lower center", fontsize=9)
fig.suptitle("Sim 3 · Small stores borrow strength from their region", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "sim3_example_stores.png")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(summ))
for i, (p, col, lab) in enumerate([("none", C[1], "No pooling"), ("complete", INK2, "Complete pooling"),
                                   ("partial", C[0], "Partial pooling")]):
    ax.bar(x + (i - 1) * 0.26, summ[p], 0.24, color=col, label=lab)
ax.set_xticks(x, [f"{b}\n({int(n)} stores)" for b, n in zip(summ.index, summ.n_stores)])
ax.set_xlabel("Observations per store")
ax.set_ylabel("RMSE of store curve")
ax.legend()
ax.set_title("Error by store size")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "sim3_rmse_by_size.png")
