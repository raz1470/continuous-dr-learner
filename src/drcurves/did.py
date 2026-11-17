"""Placebo check for the difference-in-differences design with a continuous dose."""
import numpy as np


def placebo_check(X, A, y_pre, y_prepre, main=None, a_grid=None, groups=None, path=None, **fit_kwargs):
    """Fit the curve on the change between two pre-periods, before the dose could act.

    Under (strong) parallel trends the placebo curve is flat. Judge it by its
    size relative to the effect you care about, not by whether its band
    excludes zero: with a well-fitted model the band is narrow enough to flag
    deviations too small to matter.

    Parameters
    ----------
    X, A : array-like
        Confounders and the (later) dose.
    y_pre : array-like
        Outcome in the last period before the dose.
    y_prepre : array-like
        Outcome in an earlier period before the dose.
    main : HierarchicalDoseResponse, optional
        Fitted model of the real effect (e.g. ``fit_dose_response(...,
        y_pre=...)``), used to express the placebo's size relative to the
        effect.
    a_grid : array-like, optional
        Doses at which to compare; default the 5th to 95th percentile of A.
    groups : DataFrame, optional
        Hierarchy, top to bottom.
    path : tuple, optional
        Group path whose curves to compare.
    **fit_kwargs
        Passed to :func:`fit_dose_response` (models, weights, ``n_jobs``, ...).

    Returns
    -------
    dict
        ``grid``; the centred placebo curve and its standard errors;
        ``max_abs`` (largest deviation from flat); ``effect_range`` (range of
        the main curve, if given); ``relative`` (``max_abs / effect_range``;
        small is good) and ``share_band_excludes_zero`` (for reference only).
    """
    from . import fit_dose_response
    A = np.asarray(A, float)
    if a_grid is None:
        a_grid = np.linspace(*np.quantile(A, [0.05, 0.95]), 40)
    a_grid = np.asarray(a_grid, float)
    m = fit_dose_response(X, A, y_pre, groups=groups, y_pre=y_prepre, design="did", **fit_kwargs)
    est, se = m.curve(a_grid, path, centre=True)
    out = {"grid": a_grid, "placebo": est, "se": se, "max_abs": float(np.max(np.abs(est))),
           "share_band_excludes_zero": float(np.mean(np.abs(est) > 1.96 * se)), "model": m}
    if main is not None:
        eff, _ = main.curve(a_grid, path, centre=True)
        rng = float(np.ptp(eff))
        out["effect_range"] = rng
        out["relative"] = out["max_abs"] / rng if rng > 0 else np.inf
    return out
