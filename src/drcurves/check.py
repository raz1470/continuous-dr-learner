"""Check interval coverage on *your* data before reporting (a plasmode simulation).

Coverage depends on your covariates, dose distribution and sample size, so a
simulation with invented confounders cannot tell you whether intervals are
honest for your problem. This keeps your real X, A (and groups) and simulates
only the outcome:

    Y* = h(X) + f(A) + e

  h(X) : E[Y | X] fitted on your data, so the confounding structure (X drives
         both the dose and the outcome) is realistic;
  f(A) : a known dose-response curve, so the truth is known;
  e    : residuals Y - h(X), resampled.

Each replication resamples units with replacement, draws Y*, fits every weight
model, and scores its intervals for the shape of the curve against f.

What it does and doesn't tell you: unconfoundedness holds by construction, so
this checks *estimation* (weights, smoothing, intervals), not identification.
And h is fitted by a model, so it is smoother than the real E[Y|X] and the check
is, if anything, optimistic.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from .nuisance import GPS, Nuisances, cross_fit_nuisances, pseudo_outcomes
from .pooled import HierarchicalDoseResponse

SHAPES = {
    "inverted_u": lambda z: z - 0.5 * z ** 2,
    "saturating": lambda z: 1 - np.exp(-1.2 * (z + 1.5)),
    "linear": lambda z: z,
}


def _default_weights(cat=None):
    from .balance import BalancingWeights
    return {"density": lambda r: GPS(categorical_features=cat, random_state=r),
            "balancing": lambda r: BalancingWeights()}


def coverage_check(X, A, Y, groups=None, outcome_model=None, weights=None, n_reps=20,
                   shape="inverted_u", effect_size=None, level_points=40, n_folds=5,
                   add_group_codes=True, random_state=0, verbose=True):
    """Plasmode check of the curve's 95% intervals on your own data.

    Keeps your confounders and doses, replaces the outcome with a simulated
    one whose dose-response curve is known, reruns the whole pipeline
    ``n_reps`` times and counts how often the intervals contain the truth
    (Franklin et al. 2014).

    Parameters
    ----------
    X, A, Y : array-like
        Your confounders, dose and outcome. Y is used only to build the
        simulated outcome's baseline and noise.
    groups : DataFrame, optional
        Hierarchy, top to bottom. If given, curves are fitted with partial
        pooling and every group's curve is scored against the common true
        shape (balancing weights are skipped, as they support one population
        only).
    outcome_model : sklearn regressor, optional
        Default :func:`default_model`.
    weights : dict, optional
        ``name -> callable(seed)`` returning a weight model. Default density
        (:class:`GPS`) and balancing (:class:`BalancingWeights`).
    n_reps : int, default 20
        Replications. With few, the calibrated critical values are noisy; use
        30 or more for reporting.
    shape : {"inverted_u", "saturating", "linear"} or callable, default "inverted_u"
        The true curve, as a function of the standardised dose
        ``z = (a - median) / sd``.
    effect_size : float, optional
        Peak-to-trough size of the true curve; default ``0.5 * sd(Y)``.
    level_points : int, default 40
        Grid points at which coverage is scored.
    n_folds : int, default 5
        Cross-fitting folds.
    add_group_codes : bool, default True
        Add the group labels to X.
    random_state : int, default 0
        Seed.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    summary : DataFrame
        One row per weight model. Read ``pointwise`` and ``simultaneous``
        (share of 95% intervals containing the truth), ``se_ratio`` (reported
        standard error / actual spread; below 1 means overconfident) and
        ``abs_bias``. ``critical_pointwise`` and ``critical_simultaneous`` are
        the critical values that gave 95% coverage on your data: pass one to
        ``model.band(..., critical=...)`` for calibrated intervals (compare
        with 1.96 and the usual sup-t value).
    reps : DataFrame
        The same measures for every replication.
    """
    from . import _code_columns, prepare_X
    from .models import default_model
    rng = np.random.default_rng(random_state)
    A = np.asarray(A, float)
    Y = np.asarray(Y, float)
    if groups is not None:
        groups = groups.to_frame() if isinstance(groups, pd.Series) else pd.DataFrame(groups)
        groups = groups.reset_index(drop=True)
    Xn = prepare_X(X, groups, add_group_codes)
    cat = _code_columns(Xn, groups, add_group_codes)
    om = outcome_model if outcome_model is not None else default_model(cat, random_state=random_state)
    weights = _default_weights(cat) if weights is None else weights
    if groups is not None:
        weights = {k: v for k, v in weights.items() if not getattr(v(0), "in_sample", False)}

    # h(X) and residuals from the real data (out-of-fold, so residuals are honest)
    from sklearn.model_selection import cross_val_predict
    h = cross_val_predict(clone(om), Xn, Y, cv=5)
    resid = Y - h
    med, sd = np.median(A), A.std()
    fz = SHAPES[shape] if isinstance(shape, str) else shape
    grid = np.linspace(*np.quantile(A, [0.05, 0.95]), level_points)
    raw = fz((grid - med) / sd)
    scale = (effect_size if effect_size is not None else 0.5 * Y.std()) / max(np.ptp(raw), 1e-12)
    f = lambda a: scale * fz((np.asarray(a) - med) / sd)    # noqa: E731
    truth = f(grid) - f(grid).mean()

    leaf = None if groups is None else groups.astype(str).agg("\x1f".join, axis=1).to_numpy()
    rows = []
    for rep in range(n_reps):
        t0 = time.time()
        idx = rng.integers(0, len(A), len(A))
        Xb, Ab = Xn[idx], A[idx]
        Yb = h[idx] + f(Ab) + rng.choice(resid, len(A), replace=True)
        gb = None if groups is None else groups.iloc[idx].reset_index(drop=True)
        lb = None if leaf is None else leaf[idx]
        nu = cross_fit_nuisances(Xb, Ab, Yb, clone(om), None, groups=lb, n_folds=n_folds,
                                 random_state=rep)
        for name, mk in weights.items():
            nw = cross_fit_nuisances(Xb, Ab, Yb, None, mk(rep), groups=lb, n_folds=n_folds,
                                     random_state=rep)
            xi = pseudo_outcomes(Nuisances(mu=nu.mu, m=nu.m, pi=nw.pi, varpi=nw.varpi), Yb, "dr")
            model = HierarchicalDoseResponse().fit(Ab, xi, gb)
            paths = [None] if gb is None else [tuple(r) for r in gb.drop_duplicates().itertuples(index=False)]
            for path in paths:
                est, se = model.curve(grid, path, centre=True)
                _, lo, hi, _ = model.band(grid, path, simultaneous=True, centre=True, n_draws=2000)
                z = np.abs(est - truth) / np.maximum(se, 1e-12)
                rows.append(dict(rep=rep, weights=name, path=None if path is None else "/".join(map(str, path)),
                                 est=est, se=se, z=z,
                                 pointwise=float(np.mean(np.abs(est - truth) <= 1.96 * se)),
                                 simultaneous=float(np.all((truth >= lo) & (truth <= hi)))))
        if verbose:
            print(f"coverage_check: replication {rep + 1}/{n_reps} ({time.time() - t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows)
    out = []
    for name, d in per.groupby("weights"):
        E = np.vstack(d.est.to_numpy())
        S = np.vstack(d.se.to_numpy())
        if groups is None:
            sd_emp = E.std(0)
            se_ratio = float(np.mean(S.mean(0) / np.maximum(sd_emp, 1e-12)))
            bias = float(np.mean(np.abs(E.mean(0) - truth)))
        else:
            se_ratio, bias = [], []
            for _, dg in d.groupby("path"):
                Eg, Sg = np.vstack(dg.est.to_numpy()), np.vstack(dg.se.to_numpy())
                se_ratio.append(np.mean(Sg.mean(0) / np.maximum(Eg.std(0), 1e-12)))
                bias.append(np.mean(np.abs(Eg.mean(0) - truth)))
            se_ratio, bias = float(np.mean(se_ratio)), float(np.mean(bias))
        Z = np.vstack(d.z.to_numpy())
        out.append(dict(weights=name, pointwise=d.pointwise.mean(), simultaneous=d.simultaneous.mean(),
                        # critical values that would have given 95% coverage on this data
                        critical_pointwise=float(np.quantile(Z, 0.95)),
                        critical_simultaneous=float(np.quantile(Z.max(1), 0.95)),
                        se_ratio=se_ratio, abs_bias=bias,
                        rmse=float(np.sqrt(np.mean((E - truth) ** 2))), curve_range=float(np.ptp(truth))))
    summary = pd.DataFrame(out).sort_values("pointwise", ascending=False).round(3).reset_index(drop=True)
    return summary, per.drop(columns=["est", "se", "z"])
