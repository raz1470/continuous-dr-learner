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
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd


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


class DiscreteGPS(BaseEstimator):
    """Generalised propensity score for a dose with a few distinct values
    (tiers, counts such as notifications per week): a classifier's predicted
    probability of each level. With a discrete dose, varpi/pi is a ratio of
    probabilities, which is the right weight (a continuous density is not).
    """

    def __init__(self, classifier=None, floor=1e-4, random_state=0):
        self.classifier = classifier
        self.floor = floor
        self.random_state = random_state

    def fit(self, X, A):
        A = np.asarray(A, float)
        self.levels_ = np.unique(A)
        clf = self.classifier if self.classifier is not None else HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=self.random_state)
        codes = np.searchsorted(self.levels_, A)
        self.clf_ = clone(clf).fit(X, codes)
        return self

    def density(self, X, a):
        a = np.asarray(a, float)
        code = np.clip(np.searchsorted(self.levels_, a), 0, len(self.levels_) - 1)
        left = np.clip(code - 1, 0, None)
        code = np.where(np.abs(self.levels_[left] - a) < np.abs(self.levels_[code] - a), left, code)
        proba = self.clf_.predict_proba(X)
        col = {c: j for j, c in enumerate(self.clf_.classes_)}
        out = np.full(len(a), self.floor)
        for c, j in col.items():
            hit = code == c
            out[hit] = proba[hit, j]
        return np.maximum(out, self.floor)


@dataclass
class Nuisances:
    mu: np.ndarray          # mu_hat(X_i, A_i)
    pi: np.ndarray          # pi_hat(A_i | X_i)
    varpi: np.ndarray       # varpi_hat_g(A_i)   (marginal density, within group g_i)
    m: np.ndarray           # m_hat_g(A_i) = mean_x mu_hat(x, A_i) over x in group g_i


def _feat(X, a):
    return np.column_stack([X, a])


def _folds(n, n_folds, fold_groups, random_state):
    """Fold labels. Rows sharing a fold_groups id (e.g. a customer, or a
    bootstrap duplicate of the same original row) always land in the same fold."""
    rng = np.random.default_rng(random_state)
    if fold_groups is None:
        return rng.permutation(np.arange(n) % n_folds)
    codes = pd.factorize(np.asarray(fold_groups))[0]
    u = rng.permutation(codes.max() + 1)
    return (u % n_folds)[codes]


def _predict_chunked(fn, Xr, ar, chunk=200_000):
    out = np.empty(len(ar))
    for s in range(0, len(ar), chunk):
        out[s:s + chunk] = fn(Xr[s:s + chunk], ar[s:s + chunk])
    return out


def cross_fit_nuisances(X, A, Y, outcome_model, gps, groups=None, n_folds=5,
                        n_grid=60, n_ref=1000, max_ref_total=60_000, max_levels=30,
                        fold_groups=None, random_state=0) -> Nuisances:
    """Cross-fit outcome model and GPS; return everything the pseudo-outcomes need.

    groups      : finest grouping level (length n). Integrals over P(x) are taken
                  *within* each group, so regressing the pseudo-outcome on
                  (A, group) targets theta_g(a) = E[Y(a) | G = g]. The group
                  should also be (encoded) in X so the nuisance models can use it.
    fold_groups : ids kept together in cross-fitting (e.g. customer id).
    max_levels  : if A has at most this many distinct values it is treated as
                  discrete: integrals are evaluated at those values exactly.
    n_ref / max_ref_total : rows per group (and in total) used for the integrals.
    Either model may be None, in which case its outputs are NaN.
    """
    X = np.asarray(X, float)
    A = np.asarray(A, float)
    Y = np.asarray(Y, float)
    n = len(Y)
    g = np.zeros(n, int) if groups is None else pd.factorize(np.asarray(groups))[0]
    G = g.max() + 1
    rng = np.random.default_rng(random_state)

    levels = np.unique(A)
    if len(levels) <= max_levels:
        a_grid = levels                       # discrete dose: exact evaluation points
    else:
        a_grid = np.linspace(*np.quantile(A, [0.001, 0.999]), n_grid)
    J = len(a_grid)

    # reference rows per group for the integrals (subsample big groups, cap total)
    per_group = int(max(30, min(n_ref, max_ref_total // G)))
    ref_idx = [rng.permutation(np.flatnonzero(g == k))[:per_group] for k in range(G)]
    ref_all = np.concatenate(ref_idx)
    ref_grp = np.concatenate([np.full(len(r), k) for k, r in enumerate(ref_idx)])
    counts = np.bincount(ref_grp, minlength=G)
    Xr = np.repeat(X[ref_all], J, axis=0)
    ar = np.tile(a_grid, len(ref_all))
    gr = np.repeat(ref_grp, J)
    jr = np.tile(np.arange(J), len(ref_all))

    mu = np.full(n, np.nan)
    pi = np.full(n, np.nan)
    m_tab = np.zeros((G, J))
    w_tab = np.zeros((G, J))
    fold = _folds(n, n_folds, fold_groups, random_state)
    for k in range(n_folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if outcome_model is not None:
            om = clone(outcome_model).fit(_feat(X[tr], A[tr]), Y[tr])
            mu[te] = om.predict(_feat(X[te], A[te]))
            np.add.at(m_tab, (gr, jr), _predict_chunked(lambda x, a: om.predict(_feat(x, a)), Xr, ar))
        if gps is not None and getattr(gps, "in_sample", False):
            pass                 # in-sample weights (balancing) are computed once below
        elif gps is not None and getattr(gps, "is_ratio", False):
            # weight learned directly (automatic debiasing): no density needed
            gm = clone(gps).fit(X[tr], A[tr], None if groups is None else g[tr])
            pi[te] = 1.0
            ratio_w = ratio_w if "ratio_w" in dir() else np.full(n, np.nan)
            ratio_w[te] = gm.weight(X[te], A[te], None if groups is None else g[te])
        elif gps is not None:
            gm = clone(gps).fit(X[tr], A[tr])
            pi[te] = gm.density(X[te], A[te])
            np.add.at(w_tab, (gr, jr), _predict_chunked(gm.density, Xr, ar))
    m_tab /= (counts[:, None] * n_folds)
    w_tab /= (counts[:, None] * n_folds)

    m = np.empty(n)
    varpi = np.empty(n)
    for k in range(G):
        idx = g == k
        m[idx] = np.interp(A[idx], a_grid, m_tab[k])
        varpi[idx] = np.interp(A[idx], a_grid, w_tab[k])
    if gps is not None and getattr(gps, "in_sample", False):
        pi = np.ones(n)
        varpi = gps.fit_weights(X, A, None if groups is None else g, Y=Y)
    elif gps is not None and getattr(gps, "is_ratio", False):
        varpi = ratio_w          # pseudo-outcomes use varpi / pi = learned weight
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
