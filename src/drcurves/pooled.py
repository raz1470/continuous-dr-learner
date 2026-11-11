"""Second stage: hierarchical penalised-spline regression of pseudo-outcomes on A.

Curve for a leaf group (e.g. store s in region r):

    theta_s(a) = B(a) @ (beta_global + b_r + b_s)

Penalty (all deviation blocks are shrunk towards their parent):

    lam_smooth * sum_blocks ||D2 beta_block||^2  +  sum_levels lam_level * ||b_level||^2

The ridge on deviations is exactly a Gaussian random-effects prior
b ~ N(0, sigma^2 / lam_level I): partial pooling / shrinkage. Groups with little
data borrow strength from their parent; groups with lots of data keep their shape.
Penalties are tuned by GCV with a coordinate search. GCV is inflated by
gamma = 1.4 (Kim & Gu 2004) because pseudo-outcomes are noisy and heavy-tailed,
and plain GCV undersmooths them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve


def _bspline_basis(x, lo, hi, n_basis, degree=3):
    n_int = n_basis - degree - 1
    inner = np.linspace(lo, hi, n_int + 2)
    t = np.r_[[lo] * degree, inner, [hi] * degree]
    x = np.clip(x, lo, hi - 1e-12 * (hi - lo))
    return BSpline.design_matrix(x, t, degree).toarray()


class HierarchicalDoseResponse:
    """Partially pooled dose-response curves.

    Parameters
    ----------
    n_basis : number of cubic B-spline basis functions.
    pooling : "partial" (GCV-tuned shrinkage), "none" (each group on its own,
        only smoothness penalised) or "complete" (one curve for everybody).
    """

    def __init__(self, n_basis=10, pooling="partial", lam_grid=None, n_sweeps=2, gcv_gamma=1.4):
        self.n_basis = n_basis
        self.pooling = pooling
        self.lam_grid = np.logspace(-2, 6, 9) if lam_grid is None else lam_grid
        self.n_sweeps = n_sweeps
        self.gcv_gamma = gcv_gamma

    # ---------------------------------------------------------------- design
    def _design(self, a, groups):
        B = _bspline_basis(a, self.lo_, self.hi_, self.n_basis)
        n, K = B.shape
        blocks = [sp.csr_matrix(B)]
        for lvl, keys in enumerate(self.level_keys_):
            col = self._path_codes(groups, lvl)
            idx = np.array([keys.get(c, -1) for c in col])
            ok = idx >= 0
            rows = np.repeat(np.flatnonzero(ok), K)
            cols = (idx[ok][:, None] * K + np.arange(K)).ravel()
            vals = B[ok].ravel()
            blocks.append(sp.csr_matrix((vals, (rows, cols)), shape=(n, len(keys) * K)))
        return sp.hstack(blocks).tocsr(), B

    @staticmethod
    def _path_codes(groups, lvl):
        if groups is None:
            return []
        return groups.iloc[:, : lvl + 1].astype(str).agg("/".join, axis=1).to_numpy()

    def _penalty(self, lam_s, lams):
        K = self.n_basis
        D = np.diff(np.eye(K), 2, axis=0)
        S = D.T @ D
        nb = 1 + sum(len(k) for k in self.level_keys_)
        P = np.kron(np.eye(nb), S) * lam_s
        ridge = [np.full(K, 1e-8)]
        for lam, keys in zip(lams, self.level_keys_):
            ridge.append(np.full(K * len(keys), lam))
        P[np.diag_indices_from(P)] += np.concatenate(ridge)
        return P

    def _solve(self, lam_s, lams):
        A = self.XtX_ + self._penalty(lam_s, lams)
        c = cho_factor(A)
        beta = cho_solve(c, self.Xty_)
        rss = self.yty_ - 2 * beta @ self.Xty_ + beta @ self.XtX_ @ beta
        edf = np.trace(cho_solve(c, self.XtX_))
        gcv = self.n_ * rss / (self.n_ - self.gcv_gamma * edf) ** 2
        return gcv, beta, c, edf

    # ------------------------------------------------------------------- fit
    def fit(self, a, y, groups: pd.DataFrame | None = None):
        a = np.asarray(a, float)
        y = np.asarray(y, float)
        self.lo_, self.hi_ = np.quantile(a, [0.005, 0.995])
        if groups is not None:
            groups = pd.DataFrame(groups).reset_index(drop=True)
        if self.pooling == "complete" or groups is None:
            self.level_keys_ = []
        else:
            self.level_keys_ = [
                {k: i for i, k in enumerate(pd.unique(self._path_codes(groups, l)))}
                for l in range(groups.shape[1])
            ]
        X, _ = self._design(a, groups)
        self.n_ = len(y)
        self.XtX_ = (X.T @ X).toarray()
        self.Xty_ = X.T @ y
        self.yty_ = y @ y

        L = len(self.level_keys_)
        lam_s = 1.0
        lams = [1e-4 if self.pooling == "none" else 10.0] * L
        tune_levels = self.pooling == "partial"
        for _ in range(self.n_sweeps):
            lam_s = min(self.lam_grid, key=lambda v: self._solve(v, lams)[0])
            if tune_levels:
                for l in range(L):
                    def crit(v, l=l):
                        trial = list(lams)
                        trial[l] = v
                        return self._solve(lam_s, trial)[0]
                    lams[l] = min(self.lam_grid, key=crit)
        self.lam_smooth_, self.lam_levels_ = lam_s, lams
        self.gcv_, self.coef_, chol, self.edf_ = self._solve(lam_s, lams)

        # sandwich (heteroscedasticity-robust) covariance of the coefficients
        r = y - X @ self.coef_
        Ainv = cho_solve(chol, np.eye(len(self.coef_)))
        meat = (X.T @ X.multiply(r[:, None] ** 2)).toarray()
        self.cov_ = Ainv @ meat @ Ainv
        return self

    # --------------------------------------------------------------- predict
    def curve(self, a_grid, path=None):
        """Curve (and pointwise s.e.) for a group path, e.g. ("north", "store_17").

        Unknown levels fall back to their parent: a brand-new store gets its
        region's curve; an unknown region gets the global curve.
        """
        a_grid = np.asarray(a_grid, float)
        B = _bspline_basis(a_grid, self.lo_, self.hi_, self.n_basis)
        K = self.n_basis
        sel = [np.arange(K)]
        offset = K
        for lvl, keys in enumerate(self.level_keys_):
            if path is not None and lvl < len(path):
                key = "/".join(map(str, path[: lvl + 1]))
                if key in keys:
                    sel.append(offset + keys[key] * K + np.arange(K))
            offset += len(keys) * K
        idx = np.concatenate(sel)
        M = np.zeros((len(a_grid), len(self.coef_)))
        for s in sel:
            M[:, s] += B
        est = M @ self.coef_
        se = np.sqrt(np.einsum("ij,jk,ik->i", M[:, idx], self.cov_[np.ix_(idx, idx)], M[:, idx]))
        return est, se
