"""npcausal's ctseff (Kennedy's R implementation) on the same datasets as
compare_causal_curve.py. Needs R with npcausal and SuperLearner installed.
Usage: python experiments/compare_npcausal.py REPS   (resumable)"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drcurves.simulate import sim_nonlinear, sim_high_dim  # noqa
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
_T0 = time.time()
OUT = ROOT / "results" / "cache"
for rep in range(REPS):
    for setting in ["p5", "p50"]:
        if time.time() - _T0 > BUDGET and not (OUT / f"{'cc' if 'causal_curve' in __file__ else 'np'}_{setting}_{rep:03d}.json").exists():
            print('budget reached', flush=True); sys.exit(0)
        fp = OUT / f"np_{setting}_{rep:03d}.json"
        if fp.exists():
            continue
        t0 = time.time()
        if setting == "p5":
            X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u", seed=900 + rep)
        else:
            X, A, Y, f = sim_high_dim(n=3000, p=50, seed=900 + rep)
        X = np.asarray(X); g = np.linspace(*np.quantile(A, [0.05, 0.95]), 40); truth = f(g)
        with tempfile.TemporaryDirectory() as td:
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]); df["A"] = A; df["Y"] = Y
            df.to_csv(f"{td}/d.csv", index=False)
            r = subprocess.run(["Rscript", str(ROOT / "experiments/r/run_ctseff.R"), f"{td}/d.csv", f"{td}/o.csv",
                                str(g[0]), str(g[-1]), "80", "1" if setting == "p50" else "0"], capture_output=True, text=True)
            if r.returncode:
                print(setting, rep, "failed", r.stderr[-300:], flush=True); continue
            o = pd.read_csv(f"{td}/o.csv")
        av = o["a.vals"].values
        est, lo, hi = (np.interp(g, av, o[c]) for c in ("est", "ci.ll", "ci.ul"))
        c, t = est.mean(), truth - truth.mean()
        res = dict(setting=setting, rep=rep, method="npcausal (R)",
                   rmse=float(np.sqrt(np.mean((est - c - t) ** 2))),
                   cover=float(np.mean((t >= lo - c) & (t <= hi - c))), width=float(np.mean(hi - lo)))
        fp.write_text(json.dumps([res]))
        print(setting, rep, f"{time.time()-t0:.0f}s", flush=True)
