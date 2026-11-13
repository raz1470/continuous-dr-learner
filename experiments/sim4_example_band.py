"""What the default intervals look like on one dataset (Sim 1, inverted U):
pointwise and simultaneous 95% bands, the optimal dose with its interval, and a
dose contrast. Writes results/sim4_example_band.json for the article."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcurves import fit_dose_response  # noqa: E402
from drcurves.simulate import sim_nonlinear  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

X, A, Y, f = sim_nonlinear(n=4000, curve="inverted_u_centred", seed=1)
model = fit_dose_response(X, A, Y, outcome_model=HGB(max_iter=300, learning_rate=0.05, random_state=0))
grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 80)
est, lo_p, hi_p, z_p = model.band(grid, simultaneous=False, centre=True)
_, lo_s, hi_s, z_s = model.band(grid, simultaneous=True, centre=True)
truth = f(grid) - f(grid).mean()
opt = model.optimal_dose(grid, n_draws=4000)
a_lo, a_hi = np.quantile(A, [0.10, 0.50])   # low dose -> middle dose
c, c_se = model.contrast(a_lo, a_hi)
out = dict(grid=grid.tolist(), est=est.tolist(), lo_p=lo_p.tolist(), hi_p=hi_p.tolist(),
           lo_s=lo_s.tolist(), hi_s=hi_s.tolist(), truth=truth.tolist(), z_p=z_p, z_s=z_s,
           opt=dict(dose=opt["dose"], lo=opt["lo"], hi=opt["hi"], at_edge=opt["at_edge"],
                    draws=np.round(opt["draws"], 4).tolist()),
           true_peak=float(grid[np.argmax(truth)]),
           contrast=dict(a_lo=float(a_lo), a_hi=float(a_hi), est=c, se=c_se,
                         truth=float(f(a_hi) - f(a_lo))))
(ROOT / "results" / "sim4_example_band.json").write_text(json.dumps(out))
print(f"crit pointwise {z_p:.2f}, simultaneous {z_s:.2f}; peak {opt['dose']:.2f} "
      f"[{opt['lo']:.2f}, {opt['hi']:.2f}] (true {out['true_peak']:.2f}); "
      f"contrast {c:.2f} ± {1.96 * c_se:.2f} (true {out['contrast']['truth']:.2f})")
