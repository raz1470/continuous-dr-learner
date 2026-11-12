"""Automatic debiasing for dose-response curves: learn the weight directly.

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
                             final_estimator=RidgeCV())


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
        k = self.n_shuffles
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

    def weight(self, X, A):
        X = np.asarray(X, float)
        A = np.asarray(A, float)
        Z = self._feats(X, A, None if self.dose_model_ is None else self.dose_model_.predict(X))
        r = np.ones(len(Z))
        for tree, vals in zip(self.trees_[: self.n_trees_], self.leaf_values_[: self.n_trees_]):
            r = np.clip(r + self.learning_rate * np.vectorize(vals.get)(tree.apply(Z)), 0, self.max_weight)
        return r
