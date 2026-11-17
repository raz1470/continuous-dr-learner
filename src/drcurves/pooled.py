"""Second stage: hierarchical penalised-spline regression of pseudo-outcomes on A.

Curve for a leaf group (e.g. store s in region r):

    theta_s(a) = B(a) @ (beta_global + b_r + b_s)

Penalty on every deviation block b (each group is shrunk towards its parent):

    lam_smooth * ||D2 b||^2                 smoothness
  + lam_level  * ||b - mean(b)||^2          shrinkage of the group's *shape*
  + lam_int    * K * mean(b)^2              (tiny) shrinkage of the group's *level*

B-splines sum to one, so mean(b) is the group's vertical shift. Penalising level
and shape separately stops large differences in group baselines from weakening
the pooling of shapes. The ridge on shapes is a Gaussian random-effects prior,
i.e. partial pooling. Penalties are tuned by GCV (coordinate search); GCV is
inflated by gamma = 1.4, the value proposed by Kim & Gu (2004) to curb GCV's
tendency to undersmooth, which matters here because pseudo-outcomes are noisy.

Small problems (p <= dense_max coefficients) are solved densely and exactly;
larger ones use a sparse LU factorisation and a Hutchinson estimate of the
effective degrees of freedom, so thousands of groups are feasible.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import splu

SEP = "\x1f"   # path separator for group keys (cannot clash with group names)


def _bspline_basis(x, lo, hi, n_basis, degree=3):
    n_int = n_basis - degree - 1
    inner = np.linspace(lo, hi, n_int + 2)
    t = np.r_[[lo] * degree, inner, [hi] * degree]
    x = np.clip(x, lo, hi - 1e-12 * (hi - lo))
    return BSpline.design_matrix(x, t, degree).toarray()


class _Factor:
    """Solve A x = b for symmetric positive-definite A, dense or sparse."""

    def __init__(self, A, dense):
        self.dense = dense
        if dense:
            self.c = cho_factor(A.toarray() if sp.issparse(A) else A)
        else:
            self.lu = splu(sp.csc_matrix(A))

    def solve(self, b):
        if self.dense:
            return cho_solve(self.c, b)
        b = b.toarray() if sp.issparse(b) else b
        return self.lu.solve(np.asarray(b, float))


class HierarchicalDoseResponse:
    """Partially pooled dose-response curves.

    The second stage of :func:`fit_dose_response`: regresses pseudo-outcomes on
    the dose with penalised B-splines. With groups, each group's spline
    coefficients are the sum of its ancestors' deviations, and ridge penalties
    on the deviations shrink small groups toward their parent (tuned by GCV).

    Parameters
    ----------
    n_basis : int, default 10
        Number of B-spline basis functions (reduced automatically when the
        dose takes only a few distinct values).
    pooling : {"partial", "none", "complete"}, default "partial"
        ``"partial"``: GCV-tuned shrinkage. ``"none"``: each group on its own,
        only smoothness penalised. ``"complete"``: one curve for everybody.
    lam_grid : array-like, optional
        Candidate penalties for GCV; default 9 values from 1e-2 to 1e6.
    n_sweeps : int, default 2
        Coordinate-descent sweeps over the penalties.
    gcv_gamma : float, default 1.4
        GCV inflation factor (Kim & Gu 2004); above 1 favours smoother fits,
        which matters because pseudo-outcomes are noisy.
    cov_type : {"bayes", "robust_bayes", "sandwich"}, default "bayes"
        Default covariance for intervals. ``"bayes"`` is the posterior
        covariance of the equivalent Bayesian model, as used by mgcv;
        ``"robust_bayes"`` uses a heteroscedasticity-robust data term plus the
        prior term; ``"sandwich"`` ignores penalty bias and is overconfident
        under shrinkage.
    lam_int : float, default 1e-4
        Small ridge on each group's overall level.
    dense_max : int, default 300
        Up to this many coefficients the fit is solved densely and exactly;
        above it, with a sparse LU factorisation.
    n_probes : int, default 40
        Probes for the Hutchinson estimate of the effective degrees of freedom
        in large models.
    """

    def __init__(self, n_basis=10, pooling="partial", lam_grid=None, n_sweeps=2, gcv_gamma=1.4,
                 cov_type="bayes", lam_int=1e-4, dense_max=300, n_probes=40):
        self.n_basis = n_basis
        self.pooling = pooling
        self.lam_grid = np.logspace(-2, 6, 9) if lam_grid is None else np.asarray(lam_grid, float)
        self.n_sweeps = n_sweeps
        self.gcv_gamma = gcv_gamma
        self.cov_type = cov_type
        self.lam_int = lam_int
        self.dense_max = dense_max
        self.n_probes = n_probes

    # ---------------------------------------------------------------- design
    def _basis(self, a):
        return _bspline_basis(a, self.lo_, self.hi_, self.K_, self.degree_)

    @staticmethod
    def _path_codes(groups, lvl):
        return groups.iloc[:, : lvl + 1].astype(str).agg(SEP.join, axis=1).to_numpy()

    def _design(self, a, groups):
        B = self._basis(a)
        n, K = B.shape
        blocks = [sp.csr_matrix(B)]
        for lvl, keys in enumerate(self.level_keys_):
            col = self._path_codes(groups, lvl)
            idx = np.array([keys.get(c, -1) for c in col])
            ok = idx >= 0
            rows = np.repeat(np.flatnonzero(ok), K)
            cols = (idx[ok][:, None] * K + np.arange(K)).ravel()
            blocks.append(sp.csr_matrix((B[ok].ravel(), (rows, cols)), shape=(n, len(keys) * K)))
        return sp.hstack(blocks).tocsr()

    def _design_rows(self, V, groups):
        """Like _design, but with each row's basis values replaced by V (n x K)."""
        n, K = V.shape
        blocks = [sp.csr_matrix(V)]
        for lvl, keys in enumerate(self.level_keys_):
            col = self._path_codes(groups, lvl)
            idx = np.array([keys.get(c, -1) for c in col])
            ok = idx >= 0
            rows = np.repeat(np.flatnonzero(ok), K)
            cols = (idx[ok][:, None] * K + np.arange(K)).ravel()
            blocks.append(sp.csr_matrix((V[ok].ravel(), (rows, cols)), shape=(n, len(keys) * K)))
        return sp.hstack(blocks).tocsr()

    def _penalty(self, lam_s, lams):
        K = self.K_
        order = min(2, K - 1)
        D = np.diff(np.eye(K), order, axis=0)
        S = D.T @ D
        J = np.ones((K, K)) / K
        blocks = [sp.csr_matrix(lam_s * S + 1e-8 * np.eye(K))]
        for lam, keys in zip(lams, self.level_keys_):
            blk = lam_s * S + lam * (np.eye(K) - J) + self.lam_int * K * J
            blocks.append(sp.kron(sp.identity(len(keys), format="csr"), sp.csr_matrix(blk)))
        return sp.block_diag(blocks, format="csc")

    # ----------------------------------------------------------------- solve
    def _solve(self, lam_s, lams, rng=None):
        P = self._penalty(lam_s, lams)
        dense = self.p_ <= self.dense_max
        A = self.XtX_ + P
        f = _Factor(A, dense)
        beta = f.solve(self.Xty_)
        rss = self.yty_ - 2 * beta @ self.Xty_ + beta @ (self.XtX_ @ beta)
        if dense:
            edf = float(np.trace(f.solve(self.XtX_.toarray())))
        else:   # Hutchinson: E[z' A^-1 XtX z] with Rademacher z
            rng = np.random.default_rng(0) if rng is None else rng
            Z = rng.choice([-1.0, 1.0], size=(self.p_, self.n_probes))
            edf = float(np.mean(np.sum(Z * f.solve(self.XtX_ @ Z), axis=0)))
        gcv = self.n_ * rss / max(self.n_ - self.gcv_gamma * edf, 1.0) ** 2
        return gcv, beta, f, edf, P

    # ------------------------------------------------------------------- fit
    def fit(self, a, y, groups=None, cluster=None, penalties=None, nuisance_term=None):
        """Fit the pooled curves.

        Parameters
        ----------
        a : array-like
            Dose for each row.
        y : array-like
            Pseudo-outcome (or outcome) for each row.
        groups : DataFrame or Series, optional
            One column per level, top to bottom; one column for flat groups; None
            for one curve.
        cluster : array-like, optional
            Cluster ids for the robust (``"sandwich"`` and ``"robust_bayes"``)
            covariance, e.g. customer id when customers repeat. Don't cluster at
            the level whose curves you report.
        penalties : tuple, optional
            ``(lam_smooth, [lam_level, ...])`` to skip GCV tuning.
        nuisance_term : tuple, optional
            ``(a_grid, q)`` from ``cross_fit_nuisances(..., nuisance_var=True)``.
            Adds the variance from estimating m(a) (Kennedy et al. 2017) to the
            "bayes" and "robust_bayes" covariances.

        Returns
        -------
        HierarchicalDoseResponse
            self.
        """
        a = np.asarray(a, float)
        y = np.asarray(y, float)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(y))):
            raise ValueError("a and y must be finite (found NaN or inf).")
        self.a_ = a
        if groups is not None:
            groups = (groups.to_frame() if isinstance(groups, pd.Series) else pd.DataFrame(groups))
            groups = groups.reset_index(drop=True)
            if len(groups) != len(a):
                raise ValueError("groups must have one row per observation.")
            if groups.isna().any().any():
                raise ValueError("groups contain missing values.")
        n_levels = len(np.unique(a))
        if n_levels < 2:
            raise ValueError("the dose takes a single value; no curve can be estimated.")
        self.K_ = int(min(self.n_basis, max(2, n_levels)))
        self.degree_ = int(min(3, self.K_ - 1))
        self.lo_, self.hi_ = np.quantile(a, [0.005, 0.995])
        if self.hi_ <= self.lo_:
            self.lo_, self.hi_ = a.min(), a.max()
        if self.pooling == "complete" or groups is None:
            self.level_keys_ = []
        else:
            self.level_keys_ = [
                {k: i for i, k in enumerate(pd.unique(self._path_codes(groups, l)))}
                for l in range(groups.shape[1])
            ]
        X = self._design(a, groups)
        self.level_counts_ = [np.bincount([keys[c] for c in self._path_codes(groups, l)], minlength=len(keys))
                              for l, keys in enumerate(self.level_keys_)]
        self.n_ = len(y)
        self.p_ = X.shape[1]
        self.XtX_ = (X.T @ X).tocsc()
        self.Xty_ = X.T @ y
        self.yty_ = float(y @ y)

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
            # Only the lower edge is a problem: the curve may be under-smoothed or groups not pooled
            # enough. The upper edge means a straight line (smoothness) or complete pooling (levels),
            # both legitimate fits, so it isn't worth a warning on every fit.
            low = self.lam_grid.min()
            tuned = [lam_s] + (lams if tune_levels else [])
            if any(v <= low for v in tuned):
                warnings.warn(f"A tuned penalty hit the lower edge of lam_grid ({low:g}): the curve may "
                              f"be under-smoothed (lam_smooth={lam_s:g}, levels={[float(v) for v in lams]}). "
                              "Consider a lam_grid with smaller values.", stacklevel=2)
        self.lam_smooth_, self.lam_levels_ = lam_s, lams
        self.gcv_, self.coef_, self._f, self.edf_, self._P = self._solve(lam_s, lams)

        r = y - X @ self.coef_
        if cluster is None:
            self._meat = (X.T @ X.multiply(r[:, None] ** 2)).tocsc()
        else:
            codes = pd.factorize(np.asarray(cluster))[0]
            S = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
            U = (S @ X.multiply(r[:, None])).tocsr()
            G = U.shape[0]
            self._meat = (U.T @ U * (G / max(G - 1, 1))).tocsc()
        self.sigma2_ = float(r @ r / max(self.n_ - self.edf_, 1.0))
        self._nmeat = None
        if nuisance_term is not None and nuisance_term[1] is not None:
            tg, q = np.asarray(nuisance_term[0], float), np.asarray(nuisance_term[1], float)
            wq = np.gradient(tg)                                   # trapezoid-like weights on the grid
            V = q @ (wq[:, None] * self._basis(tg))                # (n, K): integral of B(t)(mu - m)varpi
            Vx = self._design_rows(V, groups)                      # expanded into each row's blocks
            self._nmeat = (Vx.T @ Vx).tocsc()
        self._cov_cache = {}
        return self

    # ------------------------------------------------------------ covariance
    def _cov(self, idx, cov_type=None):
        """Covariance of coef_[idx] under the chosen covariance type."""
        ct = cov_type or self.cov_type
        key = (tuple(idx), ct)
        if key not in self._cov_cache:
            E = np.zeros((self.p_, len(idx)))
            E[idx, np.arange(len(idx))] = 1.0
            Z = self._f.solve(E)                       # columns idx of A^-1
            if ct == "bayes":
                C = self.sigma2_ * Z[idx, :]
            elif ct == "sandwich":
                C = Z.T @ (self._meat @ Z)
            elif ct == "robust_bayes":
                C = Z.T @ (self._meat @ Z) + self.sigma2_ * Z.T @ (self._P @ Z)
            else:
                raise ValueError(f"unknown cov_type {ct!r}")
            if self._nmeat is not None and ct in ("bayes", "robust_bayes"):
                C = C + Z.T @ (self._nmeat @ Z)       # variance from estimating m(a)
            self._cov_cache[key] = (C + C.T) / 2
        return self._cov_cache[key]

    # ---------------------------------------------------------- prediction rows
    def _rows(self, a_grid, path=None, centre=False):
        """Rows of the linear map from coefficients to the curve: (M_idx, idx)."""
        a_grid = np.atleast_1d(np.asarray(a_grid, float))
        out = (a_grid < self.lo_ - 1e-9) | (a_grid > self.hi_ + 1e-9)
        if out.any():
            warnings.warn(f"{out.sum()} dose value(s) lie outside the fitted range "
                          f"[{self.lo_:.4g}, {self.hi_:.4g}] (0.5%-99.5% of observed doses); "
                          "the curve is held flat there and is an extrapolation.", stacklevel=3)
        B = self._basis(a_grid)
        K = self.K_
        sel = [np.arange(K)]
        offset = K
        if path is not None and not isinstance(path, (tuple, list)):
            path = (path,)
        self._last_unseen = []
        for lvl, keys in enumerate(self.level_keys_):
            if path is not None and lvl < len(path):
                key = SEP.join(map(str, path[: lvl + 1]))
                if key in keys:
                    sel.append(offset + keys[key] * K + np.arange(K))
                else:
                    self._last_unseen.append(lvl)
            offset += len(keys) * K
        idx = np.concatenate(sel)
        M = np.hstack([B] * len(sel))
        if centre:
            M = M - M.mean(0, keepdims=True)
        return M, idx

    def _new_group_cov(self, a_grid, centre, unseen):
        """Extra covariance (len(a_grid) x len(a_grid)) for groups with no data at some level.

        A new group's deviation from its parent is unknown, so its band adds the spread of the
        deviations between groups at that level. The spread is estimated from the fitted
        deviations of the better-observed half of the groups (by row count), whose estimates
        are shrunk least; the penalty-implied prior alone was far too narrow in our checks."""
        if not unseen:
            return 0.0
        K = self.K_
        B = self._basis(np.atleast_1d(np.asarray(a_grid, float)))
        if centre:
            B = B - B.mean(0, keepdims=True)
        C = 0.0
        offset = K
        for lvl, keys in enumerate(self.level_keys_):
            if lvl in unseen:
                b = self.coef_[offset: offset + len(keys) * K].reshape(len(keys), K)
                n = self.level_counts_[lvl]
                big = n >= np.median(n)
                if big.sum() >= 3:
                    C = C + B @ np.cov(b[big], rowvar=False) @ B.T
            offset += len(keys) * K
        return C

    # --------------------------------------------------------------- predict
    def curve(self, a_grid, path=None, centre=False, cov_type=None, predictive=True):
        """Curve and pointwise standard errors for a group path.

        Parameters
        ----------
        a_grid : array-like
            Doses at which to evaluate the curve.
        path : tuple, optional
            Group path, top to bottom, e.g. ``("north", "store_17")``. None returns
            the global block. Under partial pooling that is a shrinkage-weighted
            "typical group" curve, not the population average E[Y(a)]; for the
            population curve fit with ``groups=None``. Unknown levels fall back to
            their parent (a new store gets its region's curve).
        centre : bool, default False
            Return the shape: the curve minus its mean over ``a_grid``.
        cov_type : str, optional
            Override the model's ``cov_type``.
        predictive : bool, default True
            For an unseen group, also include the spread between groups at that
            level, drawn from the fitted random-effects prior, so a new store's
            band is wider than its region's.

        Returns
        -------
        est : ndarray
            The curve at ``a_grid``.
        se : ndarray
            Pointwise standard errors.
        """
        M, idx = self._rows(a_grid, path, centre)
        est = M @ self.coef_[idx]
        var = np.einsum("ij,jk,ik->i", M, self._cov(idx, cov_type), M)
        extra = self._new_group_cov(a_grid, centre, self._last_unseen) if predictive else 0.0
        if np.ndim(extra):
            var = var + np.diag(extra)
            if not centre:
                warnings.warn("A new group's overall level is left almost free by the model, so its "
                              "uncentred band is very wide; use centre=True to compare shapes.", stacklevel=2)
        return est, np.sqrt(np.maximum(var, 0))

    def draws(self, a_grid, path=None, n_draws=4000, centre=False, cov_type=None, seed=0, predictive=True):
        """Curves drawn from N(coef, cov).

        Parameters
        ----------
        a_grid : array-like
            Doses at which to evaluate the curves.
        path : tuple, optional
            Group path; see :meth:`curve`.
        n_draws : int, default 4000
            Number of curves.
        centre : bool, default False
            Centre each curve over ``a_grid``.
        cov_type : str, optional
            Override the model's ``cov_type``.
        seed : int, default 0
            Random seed.
        predictive : bool, default True
            For unseen groups, add the between-group spread (see :meth:`curve`).

        Returns
        -------
        ndarray
            Array of shape ``(n_draws, len(a_grid))``.
        """
        M, idx = self._rows(a_grid, path, centre)
        C = M @ self._cov(idx, cov_type) @ M.T
        if predictive:
            C = C + self._new_group_cov(a_grid, centre, self._last_unseen)
        w, V = np.linalg.eigh((C + C.T) / 2)
        Lc = V * np.sqrt(np.maximum(w, 0))
        z = np.random.default_rng(seed).standard_normal((n_draws, len(w)))
        return M @ self.coef_[idx] + z @ Lc.T

    def band(self, a_grid, path=None, level=0.95, simultaneous=True, centre=False,
             cov_type=None, n_draws=10000, seed=0, critical=None):
        """Confidence band for a curve.

        Parameters
        ----------
        a_grid : array-like
            Doses at which to evaluate the band.
        path : tuple, optional
            Group path; see :meth:`curve`.
        level : float, default 0.95
            Coverage level.
        simultaneous : bool, default True
            Use a sup-t critical value (the 95th percentile of the largest
            standardised deviation across the grid), so the band holds for the
            whole curve at once. That's what you need for statements like "the
            curve peaks between x and y". False gives pointwise bands.
        centre : bool, default False
            Band for the shape (curve minus its mean over ``a_grid``).
        cov_type : str, optional
            Override the model's ``cov_type``.
        n_draws : int, default 10000
            Draws used for the sup-t critical value.
        seed : int, default 0
            Random seed.
        critical : float, optional
            Use this critical value instead, multiplying the standard errors
            (e.g. the calibrated value from :func:`coverage_check`).

        Returns
        -------
        est, lo, hi : ndarray
            The curve and the band's lower and upper limits.
        critical : float
            The critical value used.
        """
        est, se = self.curve(a_grid, path, centre, cov_type)
        if critical is not None:      # e.g. calibrated by coverage_check on your data
            return est, est - critical * se, est + critical * se, float(critical)
        if simultaneous:
            d = self.draws(a_grid, path, n_draws, centre, cov_type, seed) - est
            crit = float(np.quantile(np.max(np.abs(d) / np.maximum(se, 1e-12), axis=1), level))
        else:
            from scipy.stats import norm
            crit = float(norm.ppf(0.5 + level / 2))
        return est, est - crit * se, est + crit * se, crit

    def contrast(self, a_from, a_to, path=None, path_to=None, cov_type=None):
        """Difference between two points on the curves, with its standard error.

        Returns theta_path_to(a_to) - theta_path(a_from). On the same path this is
        the effect of moving the dose from ``a_from`` to ``a_to``. On different
        paths at the same dose, it's how two groups differ at that dose.

        Parameters
        ----------
        a_from, a_to : float
            The two doses.
        path : tuple, optional
            Group path for ``a_from``; see :meth:`curve`.
        path_to : tuple, optional
            Group path for ``a_to``; defaults to ``path``.
        cov_type : str, optional
            Override the model's ``cov_type``.

        Returns
        -------
        est : float
            The difference.
        se : float
            Its standard error.
        """
        path_to = path if path_to is None else path_to
        M1, i1 = self._rows([a_from], path)
        M2, i2 = self._rows([a_to], path_to)
        idx = np.union1d(i1, i2)
        m = np.zeros(len(idx))
        np.add.at(m, np.searchsorted(idx, i2), M2[0])
        np.add.at(m, np.searchsorted(idx, i1), -M1[0])
        return float(m @ self.coef_[idx]), float(np.sqrt(max(m @ self._cov(idx, cov_type) @ m, 0)))

    def plot(self, a_grid=None, paths=(None,), labels=None, band="pointwise", level=0.95, centre=False,
             truth=None, rug=True, ax=None, cov_type=None):
        """Plot one or more curves with their bands (needs matplotlib).

        Parameters
        ----------
        a_grid : array-like, optional
            Doses to plot; default the fitted dose range.
        paths : sequence of tuples, default (None,)
            Group paths to draw, e.g. ``[None]`` or ``[("north",), ("north", "s17")]``.
        labels : sequence of str, optional
            Legend labels, one per path.
        band : {"pointwise", "simultaneous", None}, default "pointwise"
            Which band to shade.
        level : float, default 0.95
            Band coverage level.
        centre : bool, default False
            Plot shapes (each curve minus its mean).
        truth : callable, optional
            ``a -> true curve``, drawn dashed (for simulations).
        rug : bool, default True
            Tick marks along the x axis at the observed doses.
        ax : matplotlib Axes, optional
            Axes to draw on.
        cov_type : str, optional
            Override the model's ``cov_type``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if a_grid is None:
            a_grid = np.linspace(self.lo_, self.hi_, 100)
        a_grid = np.asarray(a_grid, float)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.2))
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7c5cd6", "#d64a8a", "#5b6b7f", "#0f9fb8"]
        paths = list(paths)
        labels = labels or [("Estimated curve" if p is None else " / ".join(map(str, p))) for p in paths]
        for i, (p, lab) in enumerate(zip(paths, labels)):
            col = colors[i % len(colors)]
            if band is None:
                est, _ = self.curve(a_grid, p, centre, cov_type)
            else:
                est, lo, hi, _ = self.band(a_grid, p, level, simultaneous=(band == "simultaneous"),
                                           centre=centre, cov_type=cov_type)
                ax.fill_between(a_grid, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(a_grid, est, color=col, linewidth=2, label=lab)
        if truth is not None:
            t = np.asarray(truth(a_grid), float)
            ax.plot(a_grid, t - t.mean() if centre else t, color="#111827", linewidth=1.6,
                    linestyle=(0, (5, 4)), label="Truth")
        if rug and getattr(self, "a_", None) is not None:
            a = self.a_[(self.a_ >= a_grid.min()) & (self.a_ <= a_grid.max())]
            a = np.random.default_rng(0).choice(a, min(len(a), 2000), replace=False)
            ax.plot(a, np.full(len(a), 0.0), "|", color="#6b7280", alpha=0.25, markersize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_xlabel("Dose")
        ax.set_ylabel("Outcome (centred)" if centre else "Outcome")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(paths) > 1 or truth is not None:
            ax.legend(frameon=False)
        return ax

    def optimal_dose(self, a_grid, path=None, level=0.95, n_draws=4000, maximize=True,
                     cov_type=None, seed=0):
        """Dose that maximises (or minimises) the curve, with an interval.

        The interval comes from the argmax across curves drawn from the model.

        Parameters
        ----------
        a_grid : array-like
            Candidate doses.
        path : tuple, optional
            Group path; see :meth:`curve`.
        level : float, default 0.95
            Interval level.
        n_draws : int, default 4000
            Number of curve draws.
        maximize : bool, default True
            False finds the minimum.
        cov_type : str, optional
            Override the model's ``cov_type``.
        seed : int, default 0
            Random seed.

        Returns
        -------
        dict
            ``dose`` (the optimum of the fitted curve), ``lo`` and ``hi`` (the
            interval), ``at_edge`` (share of draws whose optimum is at an end of
            ``a_grid``; if large, the optimum may lie outside the data) and
            ``draws`` (the optimum in each draw).
        """
        a_grid = np.asarray(a_grid, float)
        d = self.draws(a_grid, path, n_draws, False, cov_type, seed)
        pick = np.argmax if maximize else np.argmin
        idx = pick(d, axis=1)
        est, _ = self.curve(a_grid, path)
        lo, hi = np.quantile(a_grid[idx], [(1 - level) / 2, (1 + level) / 2])
        at_edge = float(np.mean((idx == 0) | (idx == len(a_grid) - 1)))
        return {"dose": float(a_grid[pick(est)]), "lo": float(lo), "hi": float(hi),
                "at_edge": at_edge, "draws": a_grid[idx]}
