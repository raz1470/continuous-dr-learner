"""Balancing weights for a continuous dose (entropy balancing).

Under the target distribution p(x) p(a) the dose is independent of the
covariates. Entropy balancing finds the weights closest to uniform (in KL
divergence) such that, in the weighted sample,

  * the covariate basis h(X) keeps its sample mean,
  * the dose features g(A) keep their sample mean,
  * every product g(A) * h(X) has mean E[g(A)] E[h(X)]   (no association),

i.e. the dose is decorrelated from each covariate feature
(Tübbicke 2022; Vegetabile et al. 2021; Fong, Hazlett & Imai 2018 for the
related CBGPS). These weights estimate varpi(a) / pi(a | x) on the features
used, without a density model, and are stable when there are many covariates.
They are computed in-sample (not as a fitted function), so they are not
cross-fitted. Solved through the convex dual with a small ridge penalty, which
allows approximate balance when exact balance is impossible.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import SplineTransformer


class BalancingWeights:
    """Entropy-balancing weights for a continuous dose (Tübbicke 2022).

    Chooses weights close to uniform such that, after weighting, the dose is
    uncorrelated with functions of the confounders. An alternative to
    :class:`GPS` for a population curve (``groups=None``); it helped with many
    confounders in our simulations.

    Parameters
    ----------
    basis : {"quadratic", "linear", "spline"}, default "quadratic"
        Balance terms h(X): standardised X, plus X^2 (quadratic) or splines.
    dose_powers : tuple of int, default (1, 2)
        Powers of the standardised dose used in g(A).
    ridge : float, default 0.01
        Penalty on the dual variables; larger means smoother weights and
        looser balance.
    max_weight : float, default 50.0
        Cap on any single weight.
    scores : bool, default False
        Also balance cross-fitted predictions of the dose (and of the outcome,
        if given) and their squares: the directions along which confounding
        acts.
    score_model : sklearn regressor, optional
        Model for those predictions.
    n_knots : int, default 4
        Knots for ``basis="spline"``.
    """
    is_ratio = True
    in_sample = True

    def __init__(self, basis="quadratic", dose_powers=(1, 2), ridge=1e-2, max_weight=50.0,
                 scores=False, score_model=None, n_knots=4):
        self.basis = basis
        self.dose_powers = dose_powers
        self.ridge = ridge
        self.max_weight = max_weight
        self.scores = scores
        self.score_model = score_model
        self.n_knots = n_knots

    def _score_features(self, X, A, Y):
        """Cross-fitted predicted dose E[A|X] and predicted outcome E[Y|X]: the
        directions along which confounding acts. Balancing them (and their
        squares) targets the bias directly."""
        from .riesz import default_dose_model
        mdl = self.score_model if self.score_model is not None else default_dose_model()
        cv = KFold(5, shuffle=True, random_state=0)
        cols = [cross_val_predict(clone(mdl), X, A, cv=cv)]
        if Y is not None:
            cols.append(cross_val_predict(clone(mdl), X, Y, cv=cv))
        S = np.column_stack(cols)
        return np.hstack([S, S ** 2])

    def _constraints(self, X, A, Y=None):
        X = np.asarray(X, float)
        sd = X.std(0)
        keep = sd > 0
        Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
        H = [Z]
        if self.basis == "spline":
            many = np.array([len(np.unique(Z[:, j])) > 10 for j in range(Z.shape[1])])
            if many.any():
                St = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False)
                H.append(St.fit_transform(Z[:, many]))
        if self.scores:
            H.append(self._score_features(X, A, Y))
        if self.basis == "quadratic":
            Z2 = Z ** 2
            s2 = Z2.std(0)
            ok = s2 > 1e-8          # skip binary columns, whose square is collinear
            H.append((Z2[:, ok] - Z2[:, ok].mean(0)) / s2[ok])
        H = np.hstack(H)
        H = (H - H.mean(0)) / np.maximum(H.std(0), 1e-12)
        a = (A - A.mean()) / A.std()
        G = []
        for k in self.dose_powers:
            gk = a ** k
            G.append((gk - gk.mean()) / gk.std())
        G = np.column_stack(G)
        GH = (G[:, :, None] * H[:, None, :]).reshape(len(A), -1)
        # g and h are centred, so independence means E_w[g h] = 0 (no recentring!)
        C = np.hstack([H, G, GH])
        return C / np.maximum(C.std(0), 1e-12)

    def fit_weights(self, X, A, groups=None, Y=None):
        """Weights that make the weighted dose uncorrelated with the balance terms.

        Parameters
        ----------
        X : array-like
            Confounders (numeric).
        A : array-like
            Dose.
        groups : None
            Must be None: balancing weights support one population only.
        Y : array-like, optional
            Outcome, used only with ``scores=True``.

        Returns
        -------
        ndarray
            One weight per row.
        """
        if groups is not None:
            raise NotImplementedError("BalancingWeights supports a single population (groups=None); "
                                      "use GPS() for group curves.")
        A = np.asarray(A, float)
        C = self._constraints(X, A, Y if self.scores else None)
        n, k = C.shape

        def dual(lam):
            s = C @ lam
            lse = logsumexp(s)
            w = np.exp(s - lse)
            return lse + 0.5 * self.ridge * lam @ lam, C.T @ w + self.ridge * lam

        res = minimize(dual, np.zeros(k), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "gtol": 1e-8})
        s = C @ res.x
        w = np.exp(s - logsumexp(s)) * n          # mean weight 1
        self.dual_result_ = res
        self.max_imbalance_ = float(np.max(np.abs(C.T @ (w / n))))
        self.ess_share_ = float(w.sum() ** 2 / (w ** 2).sum() / n)
        return np.minimum(w, self.max_weight)
