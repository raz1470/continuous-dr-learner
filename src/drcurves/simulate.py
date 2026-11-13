"""Data-generating processes for the three simulations. Every DGP returns the
true dose-response function(s) so estimates can be scored against the truth."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------- 1
CURVES = {
    # rises then falls: e.g. discount depth vs margin, dose vs recovery
    "inverted_u": lambda a: 1.5 * a - 0.8 * a ** 2,
    # peak in the middle of the observed doses, so both sides of the hump show
    "inverted_u_centred": lambda a: 0.5 - 1.0 * a ** 2,
    # strong early effect that saturates: e.g. ad spend vs conversions
    "saturating": lambda a: 2.5 * (1 - np.exp(-1.5 * (a + 1.5))) - 2.0,
}


def sim_nonlinear(n=4000, curve="inverted_u", seed=0, shift=0.0):
    """Sim 1. Confounded continuous treatment with a nonlinear effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    m = 0.6 * X[:, 0] + 0.4 * np.tanh(2 * X[:, 1]) + 0.25 * X[:, 2] ** 2 - 0.25
    s = 0.45 + 0.35 * _sigmoid(X[:, 3])
    A = m + shift + s * rng.normal(size=n)
    f = CURVES[curve]
    Y = f(A) + 1.0 * X[:, 0] + 0.8 * np.sin(2 * X[:, 1]) + 0.5 * X[:, 2] ** 2 + rng.normal(size=n)
    return X, A, Y, f


# --------------------------------------------------------------------------- 2
def sim_double_robust(n=3000, effect="inverted_u", seed=0):
    """Sim 2. Strongly nonlinear confounding, so a linear nuisance model is
    badly misspecified while a boosted model is (approximately) correct."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    m = 0.5 * (0.9 * np.sin(1.5 * X[:, 0]) + 0.35 * X[:, 1] ** 2 - 0.4 + 0.3 * (X[:, 2] > 0))
    s = 0.8 + 0.3 * _sigmoid(2 * X[:, 0])   # enough noise for good overlap
    A = m + s * rng.normal(size=n)
    f = (lambda a: 1.0 * a) if effect == "linear" else CURVES["inverted_u"]
    g = 2.0 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] ** 2 + 0.5 * X[:, 3]
    Y = f(A) + g + 0.7 * rng.normal(size=n)
    return X, A, Y, f


# --------------------------------------------------------------------------- 3
def _random_shape(rng, sd_lin, sd_quad, sd_sin):
    c1, c2, c3 = rng.normal(0, sd_lin), rng.normal(0, sd_quad), rng.normal(0, sd_sin)
    ph = rng.uniform(0, 2 * np.pi)
    return lambda a: c1 * a + c2 * (a ** 2 - 0.6) + c3 * np.sin(2.0 * a + ph)


def sim_hierarchical(n_regions=8, stores_per_region=25, median_store_n=60, seed=0):
    """Sim 3. Region > store hierarchy. Each store has its own curve
    = global + region deviation + store deviation. Store sizes are heavy-tailed,
    so many stores have only a handful of observations."""
    rng = np.random.default_rng(seed)
    f0 = CURVES["inverted_u"]
    region_dev = [_random_shape(rng, 0.45, 0.30, 0.20) for _ in range(n_regions)]
    rows, truth = [], {}
    for r in range(n_regions):
        for j in range(stores_per_region):
            sid = f"r{r}_s{j:02d}"
            n_s = int(np.clip(np.exp(rng.normal(np.log(median_store_n), 0.9)), 8, 800))
            store_dev = _random_shape(rng, 0.25, 0.15, 0.10)
            alpha = rng.normal(0, 1.0)        # store baseline   -> affects Y
            delta = rng.normal(0, 0.25)       # store policy     -> affects A (confounding)
            X = rng.normal(size=(n_s, 3))
            m = 0.5 + delta + 0.6 * X[:, 0] + 0.3 * np.tanh(2 * X[:, 1]) + 0.4 * alpha * 0.5
            A = m + (0.45 + 0.2 * _sigmoid(X[:, 2])) * rng.normal(size=n_s)
            fs = (lambda a, rd=region_dev[r], sd=store_dev: f0(a) + rd(a) + sd(a))
            Y = fs(A) + alpha + 1.0 * X[:, 0] + 0.6 * np.sin(2 * X[:, 1]) + rng.normal(size=n_s)
            truth[(f"r{r}", sid)] = fs
            rows.append(pd.DataFrame({"region": f"r{r}", "store": sid, "x0": X[:, 0],
                                      "x1": X[:, 1], "x2": X[:, 2], "A": A, "Y": Y}))
    df = pd.concat(rows, ignore_index=True)
    return df, truth, f0


# --------------------------------------------------------------------------- 5
def sim_high_dim(n=3000, p=50, seed=0, weak_share=0.6, return_oracle=False, effect="inverted_u"):
    """Sim 2's design plus many weak confounders.

    Four strong nonlinear confounders as in sim_double_robust, plus p - 4 weak
    ones (correlated, dense, small linear effects) that push both the dose and
    the outcome the same way. The total size of the weak confounding is fixed
    as p grows, so a larger p spreads the same confounding over more variables,
    which is harder for a regularised model to learn.
    """
    rng = np.random.default_rng(seed)
    rho = 0.3
    Z = rng.normal(size=(n, p))
    X = np.sqrt(1 - rho) * Z + np.sqrt(rho) * rng.normal(size=(n, 1))  # equicorrelated
    X[:, :4] = 2 * (_sigmoid(1.7 * X[:, :4]) - 0.5) * 2                 # strong ones on ~(-2, 2)
    k = p - 4
    if k > 0:
        beta = np.random.default_rng(1234).choice([-1.0, 1.0], k) * np.random.default_rng(99).uniform(0.5, 1.5, k)
        weak = X[:, 4:] @ beta
        weak = weak_share * (weak - weak.mean()) / weak.std()
    else:
        weak = np.zeros(n)
    m = 0.5 * (0.9 * np.sin(1.5 * X[:, 0]) + 0.35 * X[:, 1] ** 2 - 0.4 + 0.3 * (X[:, 2] > 0)) + weak
    s = 0.8 + 0.3 * _sigmoid(2 * X[:, 0])
    A = m + s * rng.normal(size=n)
    f = (lambda a: 1.0 * a) if effect == "linear" else CURVES[effect]
    g = 2.0 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] ** 2 + 0.5 * X[:, 3] + 2.5 * weak
    Y = f(A) + g + 0.7 * rng.normal(size=n)
    if return_oracle:   # true weight varpi(A)/pi(A|X), for benchmarking only
        from scipy.stats import norm
        pi = norm.pdf(A, m, s)
        varpi = np.array([norm.pdf(a, m, s).mean() for a in A])
        return X, A, Y, f, varpi / pi
    return X, A, Y, f
