"""EXPERIMENTAL: not part of the supported API. Learned (Riesz) weights gave accurate curves but
intervals that under-covered badly in our simulations; may change or be removed before 1.0.

Automatic debiasing for dose-response curves: learn the weight directly.

The DR pseudo-outcome needs the weight  r0(x, a) = varpi(a) / pi(a | x)
= p(x) p(a) / p(x, a). Instead of estimating the density pi (hard with many
covariates: small errors in its mean become huge errors in the weight), we learn
r0 as the Riesz representer of the functional

    L(g) = E_X E_A' [ g(A', X) ]     (A' drawn independently of X)

which satisfies E[r0(X, A) g(X, A)] = L(g) for every g. It minimises the Riesz loss

    R(r) = E_joint[ r(X, A)^2 ]  -  2 E_product[ r(X, A') ]

(Chernozhukov, Newey & Singh 2022, "Automatic debiased machine learning";
Chernozhukov, Newey, Quintas-Martinez & Syrgkanis 2022, RieszNet/ForestRiesz).
"Product" pairs are made by shuffling the dose across rows (within groups, to
target group-specific curves). We minimise R by gradient boosting with
regression trees and exact Newton leaf values, with early stopping on held-out
Riesz loss. No density is estimated anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor


def default_dose_model():
    """Ridge + boosting stack: ridge finds dense linear signals, boosting the rest."""
    return StackingRegressor([("ridge", RidgeCV(alphas=np.logspace(-2, 4, 13))),
                              ("hgb", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05))],
                             final_estimator=RidgeCV(), cv=3)


def _shuffle(A, groups, rng):
    if groups is None:
        return rng.permutation(A)
    out = A.copy()
    for idx in pd.Series(np.arange(len(A))).groupby(pd.factorize(groups)[0]).indices.values():
        out[idx] = A[rng.permutation(idx)]
    return out


class RieszWeights(BaseEstimator):
    """Gradient-boosted Riesz regression for the weight varpi(a)/pi(a|x).

    Use it in place of a GPS: fit_dose_response(..., gps=RieszWeights()).

    dose_model : regressor for E[A | X] (default: ridge + boosting stack), or
        None. Its prediction m(X) and the residual A - m(X) are added as
        features. The weight depends mostly on how far a dose is from what is
        typical for that unit, which trees on raw X cannot find when many weak
        covariates drive the dose. No density is estimated; the regression is
        only a feature.
    """
    is_ratio = True

    def __init__(self, dose_model="default", n_estimators=1500, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=20, n_shuffles=3, validation_fraction=0.2, patience=60,
                 max_weight=50.0, random_state=0):
        self.dose_model = dose_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_shuffles = n_shuffles
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_weight = max_weight
        self.random_state = random_state

    def _feats(self, X, A, mX):
        if mX is None:
            return np.column_stack([X, A])
        return np.column_stack([X, A, mX, A - mX])

    def _stack(self, X, A, groups, rng, mX=None):
        Zj = self._feats(X, A, mX)
        Zp = np.vstack([self._feats(X, _shuffle(A, groups, rng), mX) for _ in range(self.n_shuffles)])
        return Zj, Zp

    @staticmethod
    def _loss(rj, rp):
        return float(np.mean(rj ** 2) - 2 * np.mean(rp))

    def fit(self, X, A, groups=None):
        X = np.asarray(X, float)
        A = np.asarray(A, float)
        rng = np.random.default_rng(self.random_state)
        n = len(A)
        val = rng.random(n) < self.validation_fraction
        g = None if groups is None else np.asarray(groups)
        if self.dose_model is None:
            mX = None
            self.dose_model_ = None
        else:
            dm = default_dose_model() if isinstance(self.dose_model, str) else self.dose_model
            # out-of-fold predictions for training, so the residual is honest
            mX = cross_val_predict(clone(dm), X, A, cv=KFold(3, shuffle=True, random_state=self.random_state))
            self.dose_model_ = clone(dm).fit(X, A)
        tr_m = None if mX is None else mX[~val]
        va_m = None if mX is None else mX[val]
        Zj, Zp = self._stack(X[~val], A[~val], None if g is None else g[~val], rng, tr_m)
        Vj, Vp = self._stack(X[val], A[val], None if g is None else g[val], rng, va_m)
        Z = np.vstack([Zj, Zp])
        is_j = np.r_[np.ones(len(Zj), bool), np.zeros(len(Zp), bool)]
        sw = np.where(is_j, 1.0 / len(Zj), 1.0 / len(Zp))
        r = np.ones(len(Z))
        rv_j, rv_p = np.ones(len(Vj)), np.ones(len(Vp))
        self.trees_, self.leaf_values_ = [], []
        best, best_m, since = self._loss(rv_j, rv_p), 0, 0
        for m in range(self.n_estimators):
            grad = np.where(is_j, -r, 1.0)                 # negative gradient (up to a factor 2)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=self.random_state + m)
            tree.fit(Z, grad, sample_weight=sw)
            leaf = tree.apply(Z)
            vals = {}
            for lf in np.unique(leaf):
                inl = leaf == lf
                nj = np.sum(inl & is_j)
                sum_r = np.sum(r[inl & is_j])
                np_ = np.sum(inl & ~is_j)
                # exact minimiser of the Riesz loss in this leaf
                d = (np_ * len(Zj) / len(Zp) - sum_r) / nj if nj > 0 else 0.0
                vals[lf] = float(np.clip(d, -self.max_weight, self.max_weight))
            step = self.learning_rate * np.vectorize(vals.get)(leaf)
            r = np.clip(r + step, 0.0, self.max_weight)
            self.trees_.append(tree)
            self.leaf_values_.append(vals)
            rv_j = np.clip(rv_j + self.learning_rate * np.vectorize(vals.get)(tree.apply(Vj)), 0, self.max_weight)
            rv_p = np.clip(rv_p + self.learning_rate * np.vectorize(vals.get)(tree.apply(Vp)), 0, self.max_weight)
            lv = self._loss(rv_j, rv_p)
            if lv < best - 1e-6:
                best, best_m, since = lv, m + 1, 0
            else:
                since += 1
                if since >= self.patience:
                    break
        self.n_trees_ = best_m
        return self

    def weight(self, X, A, groups=None):
        X = np.asarray(X, float)
        A = np.asarray(A, float)
        Z = self._feats(X, A, None if self.dose_model_ is None else self.dose_model_.predict(X))
        r = np.ones(len(Z))
        for tree, vals in zip(self.trees_[: self.n_trees_], self.leaf_values_[: self.n_trees_]):
            r = np.clip(r + self.learning_rate * np.vectorize(vals.get)(tree.apply(Z)), 0, self.max_weight)
        return r


def _riesz_loss(rj, rp):
    return float(np.mean(rj ** 2) - 2 * np.mean(rp))


class StackedWeights(BaseEstimator):
    """Best mix of density weights (GPS) and learned weights (RieszWeights).

    Density weights are excellent when the dose model is close to right (few
    covariates); learned weights hold up when it is not (many covariates). The
    Riesz loss measures the quality of *any* weight function, since
    R(r) = E[(r - r0)^2] - E[r0^2]. So we fit both on a training split and
    choose alpha in [0, 1] minimising the held-out Riesz loss of
    alpha * w_density + (1 - alpha) * w_learned (a quadratic, solved exactly).
    """
    is_ratio = True

    def __init__(self, gps=None, riesz=None, validation_fraction=0.25, n_shuffles=5,
                 n_grid=80, n_ref=1500, max_weight=50.0, random_state=0):
        self.gps = gps
        self.riesz = riesz
        self.validation_fraction = validation_fraction
        self.n_shuffles = n_shuffles
        self.n_grid = n_grid
        self.n_ref = n_ref
        self.max_weight = max_weight
        self.random_state = random_state

    # density ratio varpi_g(a) / pi(a|x), with varpi_g on a grid per group
    def _density_weight(self, X, A, g):
        num = np.empty(len(A))
        for k, (grid, tab) in self.varpi_.items():
            idx = np.flatnonzero(g == k) if g is not None else np.arange(len(A))
            num[idx] = np.interp(A[idx], grid, tab)
        if g is not None:   # unseen groups: pooled marginal
            seen = np.isin(g, list(self.varpi_.keys()))
            if (~seen).any():
                grid, tab = self.varpi_all_
                num[~seen] = np.interp(A[~seen], grid, tab)
        return np.clip(num / self.gps_.density(X, A), 0, self.max_weight)

    def _fit_varpi(self, X, A, g, rng):
        lo, hi = np.quantile(A, [0.001, 0.999])
        grid = np.linspace(lo, hi, self.n_grid)

        def table(rows):
            rows = rng.permutation(rows)[: self.n_ref]
            Xr = np.repeat(X[rows], len(grid), axis=0)
            ar = np.tile(grid, len(rows))
            return grid, self.gps_.density(Xr, ar).reshape(len(rows), -1).mean(0)
        self.varpi_all_ = table(np.arange(len(A)))
        if g is None:
            self.varpi_ = {None: self.varpi_all_}
        else:
            cap = max(30, 60_000 // max(1, len(np.unique(g))))
            self.varpi_ = {}
            for k in np.unique(g):
                rows = np.flatnonzero(g == k)
                rows = rng.permutation(rows)[:cap]
                Xr = np.repeat(X[rows], len(grid), axis=0)
                ar = np.tile(grid, len(rows))
                self.varpi_[k] = (grid, self.gps_.density(Xr, ar).reshape(len(rows), -1).mean(0))

    def fit(self, X, A, groups=None):
        from .nuisance import GPS
        X = np.asarray(X, float)
        A = np.asarray(A, float)
        g = None if groups is None else np.asarray(groups)
        rng = np.random.default_rng(self.random_state)
        val = rng.random(len(A)) < self.validation_fraction
        gt = None if g is None else g[~val]
        self.gps_ = clone(self.gps if self.gps is not None else GPS(random_state=self.random_state))
        self.gps_.fit(X[~val], A[~val])
        self._fit_varpi(X[~val], A[~val], gt, rng)
        self.riesz_ = clone(self.riesz if self.riesz is not None else RieszWeights(random_state=self.random_state))
        self.riesz_.fit(X[~val], A[~val], gt)
        # held-out Riesz loss of the mixture
        Xv, Av = X[val], A[val]
        gv = None if g is None else g[val]
        Ap = np.concatenate([_shuffle(Av, gv, rng) for _ in range(self.n_shuffles)])
        Xp = np.vstack([Xv] * self.n_shuffles)
        gp = None if gv is None else np.concatenate([gv] * self.n_shuffles)
        d_j, d_p = self._density_weight(Xv, Av, gv), self._density_weight(Xp, Ap, gp)
        l_j, l_p = self.riesz_.weight(Xv, Av), self.riesz_.weight(Xp, Ap)
        dj, dp = d_j - l_j, d_p - l_p
        denom = np.mean(dj ** 2)
        alpha = (np.mean(dp) - np.mean(l_j * dj)) / denom if denom > 0 else 0.0
        self.alpha_ = float(np.clip(alpha, 0.0, 1.0))
        self.val_loss_ = {"density": _riesz_loss(d_j, d_p), "learned": _riesz_loss(l_j, l_p),
                          "mixture": _riesz_loss(self.alpha_ * d_j + (1 - self.alpha_) * l_j,
                                                 self.alpha_ * d_p + (1 - self.alpha_) * l_p)}
        self._g_seen = g is not None
        return self

    def weight(self, X, A, groups=None):
        X = np.asarray(X, float)
        A = np.asarray(A, float)
        g = None if groups is None else np.asarray(groups)
        if self._g_seen and g is None:
            raise ValueError("fitted with groups: pass groups to weight().")
        return (self.alpha_ * self._density_weight(X, A, g)
                + (1 - self.alpha_) * self.riesz_.weight(X, A))
