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

    def __init__(self, n_basis=10, pooling="partial", lam_grid=None, n_sweeps=2, gcv_gamma=1.4,
                 cov_type="robust_bayes"):
        self.n_basis = n_basis
        self.pooling = pooling
        self.lam_grid = np.logspace(-2, 6, 9) if lam_grid is None else lam_grid
        self.n_sweeps = n_sweeps
        self.gcv_gamma = gcv_gamma
        self.cov_type = cov_type

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
    def fit(self, a, y, groups: pd.DataFrame | None = None, cluster=None, penalties=None):
        """Fit the pooled curves.

        groups    : DataFrame, one column per level (top -> bottom), or None.
        cluster   : optional array of cluster ids for the robust part of the
                    covariance (e.g. customer id when a customer appears many
                    times). Do not cluster at the level whose curves you
                    report: each group's own deviation is then estimated from a
                    single cluster.
        penalties : optional (lam_smooth, [lam_level, ...]) to skip GCV tuning.
        """
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
        if penalties is not None:
            lam_s, lams = penalties[0], list(penalties[1])[:L]
        else:
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

        # ---- covariances of the coefficients
        r = y - X @ self.coef_
        p = len(self.coef_)
        Ainv = cho_solve(chol, np.eye(p))
        if cluster is None:
            meat = (X.T @ X.multiply(r[:, None] ** 2)).toarray()
        else:
            codes = pd.factorize(np.asarray(cluster))[0]
            S = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
            U = (S @ X.multiply(r[:, None])).toarray()      # cluster score sums
            G = U.shape[0]
            meat = U.T @ U * G / max(G - 1, 1)
        P = self._penalty(lam_s, lams)
        self.sigma2_ = float(r @ r / max(self.n_ - self.edf_, 1.0))
        self.covs_ = {
            # heteroscedasticity-robust, treats the penalised fit as fixed
            "sandwich": Ainv @ meat @ Ainv,
            # posterior covariance of the Gaussian random-effects/smoothing model
            "bayes": self.sigma2_ * Ainv,
            # robust data term + prior term: the prior term is what lets the
            # band cover the smoothing/shrinkage bias (Nychka 1988; Wood 2006)
            "robust_bayes": Ainv @ meat @ Ainv + self.sigma2_ * Ainv @ P @ Ainv,
        }
        return self

    @property
    def cov_(self):
        return self.covs_[self.cov_type]

    # ---------------------------------------------------------- prediction rows
    def _rows(self, a_grid, path=None, centre=False):
        """Linear map M with curve = M @ coef_."""
        a_grid = np.asarray(a_grid, float)
        B = _bspline_basis(a_grid, self.lo_, self.hi_, self.n_basis)
        K = self.n_basis
        M = np.zeros((len(a_grid), len(self.coef_)))
        M[:, :K] += B
        offset = K
        for lvl, keys in enumerate(self.level_keys_):
            if path is not None and lvl < len(path):
                key = "/".join(map(str, path[: lvl + 1]))
                if key in keys:
                    j = offset + keys[key] * K
                    M[:, j:j + K] += B
            offset += len(keys) * K
        if centre:
            M = M - M.mean(0, keepdims=True)
        return M

    def _se(self, M, cov_type=None):
        cov = self.covs_[cov_type or self.cov_type]
        idx = np.flatnonzero(np.abs(M).sum(0) > 0)
        Mi = M[:, idx]
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Mi, cov[np.ix_(idx, idx)], Mi), 0)), idx

    # --------------------------------------------------------------- predict
    def curve(self, a_grid, path=None, centre=False, cov_type=None):
        """Curve (and pointwise s.e.) for a group path, e.g. ("north", "store_17").

        Unknown levels fall back to their parent: a brand-new store gets its
        region's curve; an unknown region gets the global curve.
        centre=True returns the curve minus its mean over a_grid (the shape),
        with standard errors for that centred curve.
        """
        M = self._rows(a_grid, path, centre)
        se, _ = self._se(M, cov_type)
        return M @ self.coef_, se

    def draws(self, a_grid, path=None, n_draws=4000, centre=False, cov_type=None, seed=0):
        """Curves drawn from N(coef_, cov): shape (n_draws, len(a_grid))."""
        M = self._rows(a_grid, path, centre)
        cov = self.covs_[cov_type or self.cov_type]
        idx = np.flatnonzero(np.abs(M).sum(0) > 0)
        C = M[:, idx] @ cov[np.ix_(idx, idx)] @ M[:, idx].T
        w, V = np.linalg.eigh((C + C.T) / 2)
        Lc = V * np.sqrt(np.maximum(w, 0))
        z = np.random.default_rng(seed).standard_normal((n_draws, len(w)))
        return M @ self.coef_ + z @ Lc.T

    def band(self, a_grid, path=None, level=0.95, simultaneous=True, centre=False,
             cov_type=None, n_draws=10000, seed=0):
        """Confidence band. simultaneous=True: the whole curve lies inside the
        band with probability `level` (sup-t critical value), which is what you
        need for statements like "the curve peaks between x and y".
        Returns (est, lo, hi, critical_value)."""
        M = self._rows(a_grid, path, centre)
        est = M @ self.coef_
        se, _ = self._se(M, cov_type)
        if simultaneous:
            d = self.draws(a_grid, path, n_draws, centre, cov_type, seed) - est
            crit = float(np.quantile(np.max(np.abs(d) / np.maximum(se, 1e-12), axis=1), level))
        else:
            from scipy.stats import norm
            crit = float(norm.ppf(0.5 + level / 2))
        return est, est - crit * se, est + crit * se, crit

    def contrast(self, a_from, a_to, path=None, path_to=None, cov_type=None):
        """theta_{path_to}(a_to) - theta_{path}(a_from), with its s.e.

        Same path: the effect of moving the dose from a_from to a_to.
        Different paths, same dose: how two groups differ at that dose.
        """
        path_to = path if path_to is None else path_to
        m = (self._rows([a_to], path_to) - self._rows([a_from], path))[0]
        cov = self.covs_[cov_type or self.cov_type]
        return float(m @ self.coef_), float(np.sqrt(m @ cov @ m))

    def optimal_dose(self, a_grid, path=None, level=0.95, n_draws=4000, maximize=True,
                     cov_type=None, seed=0):
        """Dose that maximises (or minimises) the curve on a_grid, with an
        interval from the spread of the argmax over posterior/sandwich draws.
        `at_edge` is the share of draws whose optimum is at the end of the
        grid: a large value means the optimum may lie outside the data."""
        a_grid = np.asarray(a_grid, float)
        d = self.draws(a_grid, path, n_draws, False, cov_type, seed)
        idx = (np.argmax if maximize else np.argmin)(d, axis=1)
        est, _ = self.curve(a_grid, path)
        best = a_grid[(np.argmax if maximize else np.argmin)(est)]
        lo, hi = np.quantile(a_grid[idx], [(1 - level) / 2, (1 + level) / 2])
        at_edge = float(np.mean((idx == 0) | (idx == len(a_grid) - 1)))
        return {"dose": float(best), "lo": float(lo), "hi": float(hi), "at_edge": at_edge,
                "draws": a_grid[idx]}
