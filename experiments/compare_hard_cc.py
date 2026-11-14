"""causal-curve's TMLE_Regressor on the harder data from compare_hard.py (adds to the same cache)."""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from compare_hard import sim_hard, f, OUT  # noqa
from causal_curve import TMLE_Regressor  # noqa
warnings.filterwarnings("ignore")
for rep in range(10):
    fp = OUT / f"hard_{rep:03d}.json"
    res = json.loads(fp.read_text())
    if "causal-curve TMLE" in res:
        continue
    t0 = time.time()
    X, A, Y = sim_hard(seed=3000 + rep)
    g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); t = f(g) - f(g).mean()
    tm = TMLE_Regressor(random_seed=0)
    tm.fit(T=pd.Series(A, name="T"), X=pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y=pd.Series(Y, name="Y"))
    est = np.ravel(tm.point_estimate(g)); iv = np.asarray(tm.point_estimate_interval(g, 0.95)); c = est.mean()
    res["causal-curve TMLE"] = dict(rmse=float(np.sqrt(np.mean((est - c - t) ** 2))),
                                    cover=float(np.mean((t >= iv[:, 0] - c) & (t <= iv[:, 1] - c))), width=float(np.mean(iv[:, 1] - iv[:, 0])))
    fp.write_text(json.dumps(res)); print(rep, f"{time.time()-t0:.0f}s", round(res["causal-curve TMLE"]["rmse"], 3), flush=True)
