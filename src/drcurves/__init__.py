"""Doubly robust, nonlinear, hierarchically pooled dose-response curves."""
from .nuisance import GPS, Nuisances, cross_fit_nuisances, pseudo_outcomes
from .pooled import HierarchicalDoseResponse

__all__ = ["GPS", "Nuisances", "cross_fit_nuisances", "pseudo_outcomes",
           "HierarchicalDoseResponse", "fit_dose_response"]


def fit_dose_response(X, A, Y, outcome_model, gps=None, groups=None, pooling="partial",
                      n_folds=5, n_basis=10, random_state=0):
    """One-call pipeline: cross-fitted nuisances -> DR pseudo-outcomes -> pooled splines.

    groups : pandas DataFrame with one column per hierarchy level, top -> bottom
             (e.g. ["region", "store"]), or None for a single population curve.
    """
    gps = GPS(random_state=random_state) if gps is None else gps
    leaf = None if groups is None else groups.astype(str).agg("/".join, axis=1).to_numpy()
    nuis = cross_fit_nuisances(X, A, Y, outcome_model, gps, groups=leaf,
                               n_folds=n_folds, random_state=random_state)
    xi = pseudo_outcomes(nuis, Y, kind="dr")
    model = HierarchicalDoseResponse(n_basis=n_basis, pooling=pooling).fit(A, xi, groups)
    model.nuisances_ = nuis
    model.pseudo_outcomes_ = xi
    return model
