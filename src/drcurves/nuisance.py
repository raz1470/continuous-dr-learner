"""Nuisance models: generalised propensity score (conditional treatment density)
and cross-fitted pieces needed for doubly robust dose-response pseudo-outcomes.

Notation follows Kennedy, Ma, McHugh & Small (2017, JRSS-B):
    mu(x, a)      = E[Y | X=x, A=a]                (outcome model)
    pi(a | x)     = conditional density of A given X  (generalised propensity score)
    varpi(a)      = int pi(a | x) dP(x)            (marginal treatment density)
    m(a)          = int mu(x, a) dP(x)             (plug-in dose-response)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict


class GPS(BaseEstimator):
    """Location-scale model for the conditional density of a continuous treatment.

        A = m(X) + s(X) * e,   e ~ f  (f estimated by KDE or assumed Gaussian)

    Parameters
    ----------
    mean_model : sklearn regressor for E[A | X]. Default: HistGradientBoosting.
    scale_model : sklearn regressor for E[|A - m(X)| | X]. Ignored if
        ``heteroscedastic=False``. (Absolute residuals are far less noisy than
        log squared residuals, which produce near-zero scales and exploding weights.)
    heteroscedastic : model s(X) (True) or use a constant scale (False).
    residual : "kde" (non-parametric shape for e) or "gaussian".
    """

    def __init__(self, mean_model=None, scale_model=None, heteroscedastic=True,
                 residual="kde", random_state=0):
        self.mean_model = mean_model
        self.scale_model = scale_model
        self.heteroscedastic = heteroscedastic
        self.residual = residual
        self.random_state = random_state

    def fit(self, X, A):
        A = np.asarray(A, float)
        mm = self.mean_model if self.mean_model is not None else HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, random_state=self.random_state)
        # out-of-fold residuals so the scale/shape are not shrunk by overfitting
        r = A - cross_val_predict(clone(mm), X, A, cv=KFold(3, shuffle=True, random_state=self.random_state))
        self.mean_model_ = clone(mm).fit(X, A)
        if self.heteroscedastic:
            sm = self.scale_model if self.scale_model is not None else HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.05, min_samples_leaf=100, random_state=self.random_state)
            self.scale_model_ = clone(sm).fit(X, np.abs(r))
            self.scale_floor_ = 0.25 * np.mean(np.abs(r))
            s = self._raw_scale(X)
            self.scale_const_ = np.sqrt(np.mean((r / s) ** 2))  # calibrate so Var(e)=1
        else:
            self.scale_const_ = np.std(r)
        e = r / self._scale(X)
        if self.residual == "kde":
            kde = gaussian_kde(e)
            self.z_grid_ = np.linspace(e.min() - 1.0, e.max() + 1.0, 2001)
            self.f_grid_ = kde(self.z_grid_)
        return self

    def _raw_scale(self, X):
        return np.maximum(self.scale_model_.predict(X), self.scale_floor_)

    def _scale(self, X):
        if self.heteroscedastic:
            return self.scale_const_ * self._raw_scale(X)
        return np.full(len(X), self.scale_const_)

    def density(self, X, a):
        """pi(a | x) evaluated row-wise."""
        s = self._scale(X)
        z = (np.asarray(a, float) - self.mean_model_.predict(X)) / s
        if self.residual == "kde":
            # defensive mixture: KDE shape in the body, Gaussian tails beyond the data
            f = 0.9 * np.interp(z, self.z_grid_, self.f_grid_, left=0.0, right=0.0) + 0.1 * norm.pdf(z)
        else:
            f = norm.pdf(z)
        return np.maximum(f / s, 1e-12)


@dataclass
class Nuisances:
    mu: np.ndarray          # mu_hat(X_i, A_i)
    pi: np.ndarray          # pi_hat(A_i | X_i)
    varpi: np.ndarray       # varpi_hat_g(A_i)   (marginal density, within group g_i)
    m: np.ndarray           # m_hat_g(A_i) = mean_x mu_hat(x, A_i) over x in group g_i


def _feat(X, a):
    return np.column_stack([X, a])


def cross_fit_nuisances(X, A, Y, outcome_model, gps: GPS, groups=None, n_folds=5,
                        n_grid=60, n_ref=1000, random_state=0) -> Nuisances:
    """Cross-fit outcome model and GPS; return everything the pseudo-outcomes need.

    ``groups`` (optional, length n) is the finest grouping level. When given, the
    integrals over P(x) are taken *within* each group, so that regressing the
    pseudo-outcome on (A, group) targets the group-specific curve
    theta_g(a) = E[Y(a) | G = g].  The group should also be (encoded) in X so the
    nuisance models can use it.
    """
    X = np.asarray(X, float)
    A = np.asarray(A, float)
    Y = np.asarray(Y, float)
    n = len(Y)
    g = np.zeros(n, int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    G = g.max() + 1
    rng = np.random.default_rng(random_state)

    lo, hi = np.quantile(A, [0.001, 0.999])
    a_grid = np.linspace(lo, hi, n_grid)

    # reference rows per group for the integrals (subsample big groups)
    ref_idx = [rng.permutation(np.flatnonzero(g == k))[:n_ref] for k in range(G)]
    ref_all = np.concatenate(ref_idx)
    ref_grp = np.concatenate([np.full(len(r), k) for k, r in enumerate(ref_idx)])
    counts = np.bincount(ref_grp, minlength=G)
    Xr = np.repeat(X[ref_all], n_grid, axis=0)
    ar = np.tile(a_grid, len(ref_all))
    gr = np.repeat(ref_grp, n_grid)
    jr = np.tile(np.arange(n_grid), len(ref_all))

    mu = np.full(n, np.nan)
    pi = np.full(n, np.nan)
    m_tab = np.zeros((G, n_grid))
    w_tab = np.zeros((G, n_grid))
    folds = KFold(n_folds, shuffle=True, random_state=random_state).split(X)
    for tr, te in folds:
        if outcome_model is not None:   # either model may be skipped (None)
            om = clone(outcome_model).fit(_feat(X[tr], A[tr]), Y[tr])
            mu[te] = om.predict(_feat(X[te], A[te]))
            np.add.at(m_tab, (gr, jr), om.predict(_feat(Xr, ar)))
        if gps is not None:
            gm = clone(gps).fit(X[tr], A[tr])
            pi[te] = gm.density(X[te], A[te])
            np.add.at(w_tab, (gr, jr), gm.density(Xr, ar))
    m_tab /= (counts[:, None] * n_folds)
    w_tab /= (counts[:, None] * n_folds)

    m = np.empty(n)
    varpi = np.empty(n)
    for k in range(G):
        idx = g == k
        m[idx] = np.interp(A[idx], a_grid, m_tab[k])
        varpi[idx] = np.interp(A[idx], a_grid, w_tab[k])
    return Nuisances(mu=mu, pi=pi, varpi=varpi, m=m)


def pseudo_outcomes(nuis: Nuisances, Y, kind="dr", max_weight_quantile=0.99):
    """Pseudo-outcomes whose regression on A estimates the dose-response curve.

    kind = "dr"      : (Y - mu) * varpi/pi + m      (Kennedy et al. 2017)
           "ipw"     : Y * varpi/pi                 (needs GPS right)
           "plugin"  : m                            (needs outcome model right)
    """
    w = nuis.varpi / nuis.pi
    if max_weight_quantile is not None:
        w = np.minimum(w, np.quantile(w, max_weight_quantile))
    Y = np.asarray(Y, float)
    if kind == "dr":
        return (Y - nuis.mu) * w + nuis.m
    if kind == "ipw":
        return Y * w
    if kind == "plugin":
        return nuis.m.copy()
    raise ValueError(kind)
