"""DiD with a continuous dose. Units carry an unobserved, time-invariant trait that raises both
their dose and their outcome level; observed X drives the dose and the outcome trend.
Three periods: two before treatment (t = -1, 0) and one after (t = 1).

Compared (all DR, stacked nuisance models, density weights):
  levels   - outcome = Y_post            (biased: the hidden trait doesn't cancel)
  did      - outcome = Y_post - Y_pre    (DiD: the trait cancels)
  placebo  - outcome = Y_pre - Y_prepre  (should be flat)
  lagged   - outcome = Y_post, adjusting for Y_pre as a confounder (lagged-outcome regression)
Usage: python experiments/sim_did.py REPS [BUDGET]   (resumable)"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from drcurves import GPS, HierarchicalDoseResponse, cross_fit_nuisances, pseudo_outcomes  # noqa
from drcurves.simulate import CURVES  # noqa
from compare_stacked import stack  # noqa
warnings.filterwarnings("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
OUT = ROOT / "results" / "cache"; T0 = time.time()
f = CURVES["inverted_u_centred"]


def sim_did(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)); alpha = rng.normal(size=n)          # alpha: unobserved
    A = 0.5 * alpha + 0.4 * X[:, 0] + 0.3 * np.sin(2 * X[:, 1]) + 0.6 * rng.normal(size=n)
    g = 1.0 * X[:, 0] + 0.5 * X[:, 2] ** 2                            # level
    h = 0.6 * X[:, 1] + 0.3 * X[:, 0]                                 # trend, depends on X only
    Y = {t: 1.5 * alpha + g + t * h + (t == 1) * f(A) + 0.7 * rng.normal(size=n) for t in (-1, 0, 1)}
    return X, A, Y


def fit(X, A, y, rep, g):
    nu = cross_fit_nuisances(X, A, y, stack(), GPS(mean_model=stack(), scale_model=stack(), random_state=rep),
                             random_state=rep)
    return HierarchicalDoseResponse().fit(A, pseudo_outcomes(nu, y, "dr")).curve(g, centre=True)


if __name__ == "__main__":
    for rep in range(REPS):
        fp = OUT / f"did_{rep:03d}.json"
        if fp.exists():
            old = json.loads(fp.read_text())
            if "lagged" not in old:   # added later: adjust for the pre-period outcome instead of differencing
                X, A, Y = sim_did(seed=1000 + rep)
                g = np.array(old["grid"]); t = np.array(old["truth"])
                est, se = fit(np.column_stack([X, Y[0]]), A, Y[1], rep, g)
                old["lagged"] = dict(est=est.tolist(), se=se.tolist(), rmse=float(np.sqrt(np.mean((est - t) ** 2))),
                                     cover=float(np.mean(np.abs(est - t) <= 1.96 * se)))
                fp.write_text(json.dumps(old)); print(rep, "lagged", round(old["lagged"]["rmse"], 3), flush=True)
            continue
        if time.time() - T0 > BUDGET:
            print("budget reached", flush=True); sys.exit(0)
        t0 = time.time()
        X, A, Y = sim_did(seed=1000 + rep)
        g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); t = f(g) - f(g).mean()
        out = {"grid": g.tolist(), "truth": t.tolist()}
        for name, y, truth in [("levels", Y[1], t), ("did", Y[1] - Y[0], t), ("placebo", Y[0] - Y[-1], 0 * t)]:
            est, se = fit(X, A, y, rep, g)
            out[name] = dict(est=est.tolist(), se=se.tolist(), rmse=float(np.sqrt(np.mean((est - truth) ** 2))),
                             cover=float(np.mean(np.abs(est - truth) <= 1.96 * se)))
        fp.write_text(json.dumps(out))
        print(rep, f"{time.time()-t0:.0f}s", {k: (round(out[k]["rmse"], 3), round(out[k]["cover"], 2))
                                            for k in ("levels", "did", "placebo")}, flush=True)
