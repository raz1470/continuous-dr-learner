"""Doubly robust, nonlinear, hierarchically pooled dose-response curves."""
import warnings

import numpy as np
import pandas as pd

from .models import default_model
from .nuisance import GPS, DiscreteGPS, Nuisances, cross_fit_nuisances, pseudo_outcomes
from .pooled import HierarchicalDoseResponse
from .bootstrap import bootstrap_curves
from .balance import BalancingWeights
from .check import coverage_check
from .did import placebo_check

# Supported public API. Everything else (drcurves.riesz, drcurves.simulate, prepare_X) is
# experimental or internal and may change without notice before 1.0.
__all__ = [
    # one call
    "fit_dose_response", "HierarchicalDoseResponse",
    # models and weights
    "default_model", "GPS", "DiscreteGPS", "BalancingWeights",
    # checks
    "coverage_check", "bootstrap_curves", "placebo_check",
    # building blocks for custom pipelines
    "cross_fit_nuisances", "pseudo_outcomes", "Nuisances",
]


def prepare_X(X, groups=None, add_group_codes=True):
    """Numeric feature matrix for the nuisance models.

    DataFrame columns of dtype object/category/bool are integer-coded. With
    add_group_codes, one integer code per hierarchy level is appended so the
    nuisance models can learn group effects (tree models split on codes fine;
    pass categorical_features to HistGradientBoosting for native handling,
    which is limited to 255 levels).
    """
    if isinstance(X, pd.DataFrame):
        cols = []
        for c in X.columns:
            col = X[c]
            if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                cols.append(pd.factorize(col)[0].astype(float))
            else:
                cols.append(col.to_numpy(float))
        Xn = np.column_stack(cols) if cols else np.empty((len(X), 0))
    else:
        Xn = np.asarray(X, float)
        if Xn.ndim == 1:
            Xn = Xn[:, None]
    if groups is not None and add_group_codes:
        g = groups.to_frame() if isinstance(groups, pd.Series) else pd.DataFrame(groups)
        codes = [pd.factorize(g.iloc[:, : l + 1].astype(str).agg("|".join, axis=1))[0]
                 for l in range(g.shape[1])]
        Xn = np.column_stack([np.column_stack(codes).astype(float), Xn])
    if np.isinf(Xn).any():
        raise ValueError("X contains inf.")
    return Xn


def _check_dose(A, max_levels):
    A = np.asarray(A, float)
    if not np.all(np.isfinite(A)):
        raise ValueError("A (the dose) must be finite.")
    vals, counts = np.unique(A, return_counts=True)
    discrete = len(vals) <= max_levels
    top = counts.max() / len(A)
    if not discrete and top > 0.05:
        v = vals[np.argmax(counts)]
        warnings.warn(f"{top:.0%} of doses equal {v:g}: a point mass the continuous GPS cannot "
                      "represent (e.g. zero spend). Weights near that value are unreliable; "
                      "consider modelling 'any dose' separately or analysing A > 0 only.",
                      stacklevel=3)
    return discrete


def _code_columns(Xn, groups, add_group_codes):
    """Indices of the group-code columns prepare_X puts first (treated as categorical)."""
    n_codes = groups.shape[1] if (groups is not None and add_group_codes) else 0
    return list(range(n_codes)) or None


def fit_dose_response(X, A, Y, outcome_model=None, gps=None, groups=None, pooling="partial",
                      n_folds=5, n_basis=10, cluster=None, fold_groups=None,
                      cov_type="bayes", max_weight_quantile=0.99, add_group_codes=True,
                      max_levels=30, y_pre=None, design=None, n_jobs=None, random_state=0):
    """One call: cross-fitted nuisances -> DR pseudo-outcomes -> pooled splines.

    X        : covariates (array or DataFrame; categorical columns are coded).
    A, Y     : dose and outcome.
    groups   : None (one population curve); a DataFrame with one column (flat
               groups, each shrunk toward the global curve); or several columns
               top -> bottom (e.g. ["region", "store"]). Levels must be nested.
    outcome_model : sklearn regressor for E[Y | X, A]. None uses default_model(), a
               spline-ridge + boosting stack.
    gps      : the weight model. None picks DiscreteGPS when A has <= max_levels
               distinct values, otherwise GPS() (density-based weights, Kennedy
               et al. 2017). BalancingWeights() is an alternative for a
               population curve (groups=None).
    cluster  : cluster ids for the robust covariance types.
    fold_groups : ids kept together in cross-fitting (e.g. customer id).
    cov_type : "bayes" (default), "robust_bayes" or "sandwich".
    n_jobs   : parallel jobs for the default models (None = 1, -1 = all cores).
    y_pre    : the outcome before the dose (same units). With it, choose `design`:
               "did" (default): fit on the change Y - y_pre. Removes stable unobserved traits,
                   but comparing doses needs strong parallel trends: units on different doses
                   would have followed the same path had their doses been equal.
               "lagged": keep Y and adjust for y_pre as a confounder. Right when the dose
                   responds to past outcomes rather than to a stable trait.
               The two bracket the truth when you don't know which applies; check "did" with
               placebo_check on two pre-periods.
    Returns a fitted HierarchicalDoseResponse with .nuisances_, .pseudo_outcomes_
    and .diagnostics().
    """
    Y = np.asarray(Y, float)
    A = np.asarray(A, float)
    if design is not None and y_pre is None:
        raise ValueError("design needs y_pre (the outcome before the dose).")
    if y_pre is not None:
        y_pre = np.asarray(y_pre, float)
        design = design or "did"
        if design not in ("did", "lagged"):
            raise ValueError('design must be "did" or "lagged".')
        if y_pre.shape != Y.shape or not np.all(np.isfinite(y_pre)):
            raise ValueError("y_pre must be finite and the same length as Y.")
        if design == "did":
            Y = Y - y_pre
        else:
            X = pd.concat([X.reset_index(drop=True), pd.Series(y_pre, name="_y_pre")], axis=1) \
                if isinstance(X, pd.DataFrame) else np.column_stack([np.asarray(X, float), y_pre])
    _check_inputs(X, A, Y, groups)
    if groups is not None:
        groups = groups.to_frame() if isinstance(groups, pd.Series) else pd.DataFrame(groups)
        groups = groups.reset_index(drop=True)
    discrete = _check_dose(A, max_levels)
    Xn = prepare_X(X, groups, add_group_codes)
    cat = _code_columns(Xn, groups, add_group_codes)
    if outcome_model is None:
        # the outcome model sees [X, A]; A is appended last, so the code columns keep their indices
        outcome_model = default_model(cat, n_jobs=n_jobs, random_state=random_state)
    if gps is None:
        gps = (DiscreteGPS(random_state=random_state) if discrete
               else GPS(categorical_features=cat, n_jobs=n_jobs, random_state=random_state))
    elif discrete and type(gps) is GPS:
        warnings.warn(f"A has only {len(np.unique(A))} distinct values; a continuous GPS is not "
                      "a valid density for it. Use DiscreteGPS (the default when gps=None).",
                      stacklevel=2)
    leaf = None if groups is None else groups.astype(str).agg("\x1f".join, axis=1).to_numpy()
    nuis = cross_fit_nuisances(Xn, A, Y, outcome_model, gps, groups=leaf, n_folds=n_folds,
                               max_levels=max_levels, fold_groups=fold_groups,
                               n_jobs=n_jobs, random_state=random_state)
    xi = pseudo_outcomes(nuis, Y, kind="dr", max_weight_quantile=max_weight_quantile)
    model = HierarchicalDoseResponse(n_basis=n_basis, pooling=pooling, cov_type=cov_type).fit(
        A, xi, groups, cluster=cluster)
    model.nuisances_ = nuis
    model.pseudo_outcomes_ = xi
    model._diag_data = (A, leaf, max_weight_quantile)
    model.diagnostics = lambda: _diagnostics(model)
    model.design_ = design or "levels"
    ess = model.diagnostics()["weights"]["ess_share"]
    if np.isfinite(ess) and ess < 0.1:
        warnings.warn(f"Poor overlap: the weights' effective sample size is {ess:.0%} of n, so a few "
                      "units dominate the curve. Check model.diagnostics(), consider trimming the dose "
                      "range, or try gps=BalancingWeights().", stacklevel=2)
    return model


MIN_GROUP_ROWS = 10


def _check_inputs(X, A, Y, groups):
    """Fail early, with a clear message, on inputs the pipeline cannot handle."""
    n = len(Y)
    if Y.ndim != 1 or A.ndim != 1:
        raise ValueError("A and Y must be one-dimensional.")
    if len(A) != n or len(X) != n:
        raise ValueError(f"X, A and Y must have the same length (got {len(X)}, {len(A)}, {n}).")
    if groups is not None and len(groups) != n:
        raise ValueError(f"groups must have one row per observation (got {len(groups)}, expected {n}).")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must be finite (found NaN or inf).")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite (found NaN or inf).")
    if np.ptp(A) == 0:
        raise ValueError("A is constant: there is no variation in the dose to estimate a curve from.")
    if n < 200:
        warnings.warn(f"Only {n} rows: cross-fitted nuisance models and the curve will be very noisy.",
                      stacklevel=3)
    if groups is not None:
        groups = groups.to_frame() if isinstance(groups, pd.Series) else pd.DataFrame(groups)
        if groups.isna().any().any():
            raise ValueError("groups contains missing values.")
        leaf_n = groups.astype(str).agg("\x1f".join, axis=1).value_counts()
        small = int((leaf_n < MIN_GROUP_ROWS).sum())
        if small:
            warnings.warn(f"{small} of {len(leaf_n)} groups have fewer than {MIN_GROUP_ROWS} rows; their curves "
                          "will lean almost entirely on their parent group.", stacklevel=3)


def _diagnostics(model):
    """Overlap and fit checks worth reading before trusting the curves."""
    A, leaf, q = model._diag_data
    nuis = model.nuisances_
    w = nuis.varpi / nuis.pi
    cap = np.quantile(w, q) if q is not None else np.inf
    wc = np.minimum(w, cap)                       # the weights actually used
    out = {
        "weights": {"median": float(np.median(w)), "p99": float(np.quantile(w, 0.99)),
                    "max_raw": float(w.max()), "cap": float(cap),
                    "share_capped": float(np.mean(w > cap)),
                    # effective sample size of the capped weights, as a share of n
                    "ess_share": float(wc.sum() ** 2 / (wc ** 2).sum() / len(wc))},
        "dose_range_fitted": (float(model.lo_), float(model.hi_)),
        "penalties": {"smooth": float(model.lam_smooth_),
                      "levels": [float(v) for v in model.lam_levels_],
                      "grid_edges": (float(model.lam_grid.min()), float(model.lam_grid.max()))},
        "edf": float(model.edf_),
    }
    if leaf is not None:
        df = pd.DataFrame({"g": leaf, "A": A})
        stats = df.groupby("g").A.agg(n="size", iqr=lambda v: np.subtract(*np.quantile(v, [.75, .25])))
        out["groups"] = {"n_groups": int(len(stats)), "min_rows": int(stats.n.min()),
                         "groups_with_constant_dose": [k.replace("\x1f", "/")
                                                       for k in stats.index[stats.iqr == 0]][:20]}
    return out
