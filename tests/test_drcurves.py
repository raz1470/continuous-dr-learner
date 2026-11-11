import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from drcurves import GPS, HierarchicalDoseResponse, fit_dose_response


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 1.5 * A - 0.8 * A ** 2 + X[:, 0] + rng.normal(size=n)
    return X, A, Y


def test_gps_density_integrates_to_one():
    X, A, _ = _data()
    gps = GPS(mean_model=LinearRegression(), heteroscedastic=False).fit(X, A)
    grid = np.linspace(-8, 8, 4001)
    dens = gps.density(np.repeat(X[:1], len(grid), 0), grid)
    assert abs(np.trapezoid(dens, grid) - 1) < 0.05


def test_dr_curve_recovers_quadratic():
    X, A, Y = _data()
    om = make_pipeline(PolynomialFeatures(2), LinearRegression())
    gps = GPS(mean_model=LinearRegression(), heteroscedastic=False, residual="gaussian")
    m = fit_dose_response(X, A, Y, outcome_model=om, gps=gps)
    g = np.linspace(-1.5, 1.5, 30)
    est, _ = m.curve(g)
    truth = 1.5 * g - 0.8 * g ** 2
    assert np.sqrt(np.mean(((est - est.mean()) - (truth - truth.mean())) ** 2)) < 0.25


def test_partial_pooling_shrinks_small_groups():
    rng = np.random.default_rng(1)
    rows = []
    for gi in range(20):
        n = 400 if gi < 2 else 10
        a = rng.normal(size=n)
        rows.append(pd.DataFrame({"g": f"g{gi}", "a": a, "y": np.sin(a) + rng.normal(size=n)}))
    df = pd.concat(rows, ignore_index=True)
    grid = np.linspace(-1.5, 1.5, 20)
    err = {}
    for pooling in ["none", "partial"]:
        m = HierarchicalDoseResponse(pooling=pooling).fit(df.a, df.y, df[["g"]])
        e = [m.curve(grid, (f"g{gi}",))[0] for gi in range(2, 20)]
        err[pooling] = np.mean([np.sqrt(np.mean((c - np.sin(grid)) ** 2)) for c in e])
    assert err["partial"] < err["none"]


def test_unknown_group_falls_back_to_parent():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"r": rng.choice(["a", "b"], 500), "s": rng.choice(["x", "y"], 500),
                       "a": rng.normal(size=500)})
    df["y"] = df.a + rng.normal(size=500)
    m = HierarchicalDoseResponse().fit(df.a, df.y, df[["r", "s"]])
    g = np.linspace(-1, 1, 5)
    assert np.allclose(m.curve(g, ("a", "never_seen"))[0], m.curve(g, ("a",))[0])
