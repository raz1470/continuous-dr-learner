"""Bootstrap for the full pipeline (nuisances -> pseudo-outcomes -> pooled splines).

The analytic bands from HierarchicalDoseResponse treat the nuisance models and
the tuned penalties as fixed. The bootstrap refits everything, so it also
captures their variability. It is slow (every replicate refits the boosted
models), so use it to check the analytic bands once, not routinely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nuisance import cross_fit_nuisances, pseudo_outcomes
from .pooled import HierarchicalDoseResponse


def _resample(n, groups, stratify, rng):
    if groups is None or not stratify:
        return rng.integers(0, n, n)
    leaf = groups.astype(str).agg("\x1f".join, axis=1).to_numpy()
    idx = []
    for g in pd.unique(leaf):
        rows = np.flatnonzero(leaf == g)
        idx.append(rng.choice(rows, len(rows), replace=True))
    return np.concatenate(idx)


def bootstrap_curves(X, A, Y, outcome_model, gps, a_grid, paths=(None,), groups=None,
                     pooling="partial", n_boot=100, refit_nuisances=True, retune=True,
                     n_folds=5, n_basis=10, level=0.95, centre=False, random_state=0,
                     verbose=False):
    """Percentile bootstrap bands for one or more curves.

    paths            : group paths to return, e.g. [None, ("north",), ("north", "s17")].
    refit_nuisances  : refit outcome model and GPS in every replicate (slow, complete)
                       or keep the original pseudo-outcomes (fast, misses nuisance noise).
    retune           : re-select the penalties by GCV in every replicate.
    Rows are resampled within each leaf group, so every group appears in every
    replicate and its curve stays defined.

    X must already be numeric (see drcurves.prepare_X). The percentile band is
    pointwise; use the stored draws for any other summary.

    Returns {path: {"est", "lo", "hi", "draws"}}.
    """
    X, A, Y = np.asarray(X, float), np.asarray(A, float), np.asarray(Y, float)
    a_grid = np.asarray(a_grid, float)
    rng = np.random.default_rng(random_state)
    if groups is not None:
        groups = pd.DataFrame(groups).reset_index(drop=True)
    leaf = None if groups is None else groups.astype(str).agg("\x1f".join, axis=1).to_numpy()

    nuis = cross_fit_nuisances(X, A, Y, outcome_model, gps, groups=leaf, n_folds=n_folds,
                               random_state=random_state)
    xi = pseudo_outcomes(nuis, Y)
    base = HierarchicalDoseResponse(n_basis=n_basis, pooling=pooling).fit(A, xi, groups)
    pen = (base.lam_smooth_, base.lam_levels_)
    est = {p: base.curve(a_grid, p, centre)[0] for p in paths}

    draws = {p: [] for p in paths}
    for b in range(n_boot):
        idx = _resample(len(Y), groups, True, rng)
        g_b = None if groups is None else groups.iloc[idx].reset_index(drop=True)
        if refit_nuisances:
            leaf_b = None if leaf is None else leaf[idx]
            # duplicates of one original row stay in the same fold (no leakage)
            nb = cross_fit_nuisances(X[idx], A[idx], Y[idx], outcome_model, gps, groups=leaf_b,
                                     n_folds=n_folds, fold_groups=idx,
                                     random_state=random_state + b + 1)
            xi_b = pseudo_outcomes(nb, Y[idx])
        else:
            xi_b = xi[idx]
        m = HierarchicalDoseResponse(n_basis=n_basis, pooling=pooling).fit(
            A[idx], xi_b, g_b, penalties=None if retune else pen)
        for p in paths:
            draws[p].append(m.curve(a_grid, p, centre)[0])
        if verbose:
            print(f"bootstrap {b + 1}/{n_boot}", flush=True)

    out = {}
    q = [(1 - level) / 2, (1 + level) / 2]
    for p in paths:
        d = np.array(draws[p])
        lo, hi = np.quantile(d, q, axis=0)
        out[p] = {"est": est[p], "lo": lo, "hi": hi, "draws": d}
    return out
