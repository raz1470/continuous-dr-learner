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


# ---------------------------------------------------------------- uncertainty
def _single_fit(seed=0):
    X, A, Y = _data(seed=seed)
    om = make_pipeline(PolynomialFeatures(2), LinearRegression())
    gps = GPS(mean_model=LinearRegression(), heteroscedastic=False, residual="gaussian")
    return fit_dose_response(X, A, Y, outcome_model=om, gps=gps)


def test_simultaneous_band_is_wider_than_pointwise():
    m = _single_fit()
    g = np.linspace(-1.5, 1.5, 40)
    _, lo_p, hi_p, z_p = m.band(g, simultaneous=False)
    _, lo_s, hi_s, z_s = m.band(g, simultaneous=True)
    assert z_s > z_p and np.all(hi_s - lo_s >= hi_p - lo_p)


def test_contrast_and_optimal_dose():
    m = _single_fit()
    est, se = m.contrast(0.0, 1.0)
    assert abs(est - 0.7) < 3 * se + 0.05          # truth: 1.5 - 0.8
    opt = m.optimal_dose(np.linspace(-1.5, 1.5, 61))
    assert opt["lo"] <= 0.9375 <= opt["hi"]         # truth: 1.5 / 1.6
    assert opt["at_edge"] < 0.05


def test_cov_types_and_cluster_run():
    rng = np.random.default_rng(3)
    a = rng.normal(size=600)
    y = np.sin(a) + rng.normal(size=600)
    cl = np.repeat(np.arange(60), 10)
    m = HierarchicalDoseResponse().fit(a, y, cluster=cl)
    for ct in ["sandwich", "bayes", "robust_bayes"]:
        _, se = m.curve(np.linspace(-1, 1, 5), cov_type=ct)
        assert np.all(se > 0)


def test_flat_groups_without_hierarchy():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"store": rng.choice([f"s{i}" for i in range(15)], 1500),
                       "a": rng.normal(size=1500)})
    df["y"] = np.sin(df.a) + rng.normal(size=1500)
    m = HierarchicalDoseResponse().fit(df.a, df.y, df[["store"]])
    est, se = m.curve(np.linspace(-1, 1, 5), ("s3",))
    assert np.all(np.isfinite(est)) and np.all(se > 0)
    assert np.allclose(m.curve([0.0], ("unseen",))[0], m.curve([0.0])[0])


def test_bootstrap_runs():
    from drcurves import bootstrap_curves
    X, A, Y = _data(n=800)
    gps = GPS(mean_model=LinearRegression(), heteroscedastic=False, residual="gaussian")
    g = np.linspace(-1, 1, 10)
    out = bootstrap_curves(X, A, Y, LinearRegression(), gps, g, n_boot=4, n_folds=2)
    assert out[None]["draws"].shape == (4, 10) and np.all(out[None]["lo"] <= out[None]["hi"])


# ---------------------------------------------------------------- real-data robustness
def test_discrete_dose_uses_discrete_gps_and_recovers_curve():
    from drcurves import DiscreteGPS
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 2))
    # notifications per week, 0..8, more for engaged users (confounded)
    A = np.clip(np.round(3 + 1.2 * X[:, 0] + rng.normal(size=n) * 1.5), 0, 8)
    f = lambda a: 1.2 * a - 0.15 * a ** 2                      # noqa: E731  (peak at 4)
    Y = f(A) + 1.5 * X[:, 0] + rng.normal(size=n)
    om = make_pipeline(PolynomialFeatures(2), LinearRegression())
    m = fit_dose_response(X, A, Y, outcome_model=om)
    assert isinstance(m.nuisances_, object)
    g = np.arange(1, 8, dtype=float)
    est, _ = m.curve(g)
    t = f(g)
    assert np.sqrt(np.mean(((est - est.mean()) - (t - t.mean())) ** 2)) < 0.3
    opt = m.optimal_dose(np.linspace(1, 7, 61))
    assert opt["lo"] <= 4.0 <= opt["hi"]


def test_dataframe_inputs_series_groups_and_odd_names():
    rng = np.random.default_rng(6)
    n = 1500
    df = pd.DataFrame({"x": rng.normal(size=n), "segment": rng.choice(["a", "b", "c"], n),
                       "store": rng.choice(["north/1", "north/2", "south/1"], n)})
    df["A"] = 0.4 * df.x + rng.normal(size=n)
    df["Y"] = np.sin(df.A) + df.x + rng.normal(size=n)
    m = fit_dose_response(df[["x", "segment"]], df.A, df.Y,
                          outcome_model=make_pipeline(PolynomialFeatures(2), LinearRegression()),
                          gps=GPS(mean_model=LinearRegression(), heteroscedastic=False),
                          groups=df["store"], fold_groups=rng.integers(0, 300, n))
    e1, _ = m.curve([0.0], ("north/1",))
    e2, _ = m.curve([0.0], ("north/2",))
    assert np.isfinite(e1).all() and np.isfinite(e2).all()
    d = m.diagnostics()
    assert d["groups"]["n_groups"] == 3 and 0 < d["weights"]["ess_share"] <= 1


def test_nan_outcome_raises():
    import pytest
    X, A, Y = _data(n=300)
    Y[3] = np.nan
    with pytest.raises(ValueError):
        fit_dose_response(X, A, Y, outcome_model=LinearRegression())


def test_many_groups_sparse_path():
    rng = np.random.default_rng(7)
    S = 1500
    st = rng.integers(0, S, S * 20)
    a = rng.normal(size=len(st))
    y = np.sin(a) + rng.normal(size=len(st))
    m = HierarchicalDoseResponse().fit(a, y, pd.DataFrame({"r": st % 15, "s": st}))
    assert m.p_ > m.dense_max
    est, se = m.curve(np.linspace(-1, 1, 5), (3, 18))
    assert np.all(np.isfinite(est)) and np.all(se > 0)


def test_riesz_weights_track_true_ratio_and_default():
    from scipy.stats import norm
    from drcurves import RieszWeights
    rng = np.random.default_rng(8)
    n = 4000
    X = rng.normal(size=(n, 3))
    m = 0.8 * X[:, 0] + 0.5 * np.sin(2 * X[:, 1])
    A = m + rng.normal(size=n)
    true = np.array([norm.pdf(a, m, 1).mean() for a in A[:500]]) / norm.pdf(A[:500], m[:500], 1)
    w = RieszWeights().fit(X[500:], A[500:]).weight(X[:500], A[:500])
    assert np.corrcoef(np.log(w + 1e-3), np.log(true))[0, 1] > 0.6
    # learned weights plug into the pipeline in place of a GPS
    Y = 1.5 * A - 0.8 * A ** 2 + X[:, 0] + rng.normal(size=n)
    mdl = fit_dose_response(X, A, Y, outcome_model=make_pipeline(PolynomialFeatures(2), LinearRegression()),
                            gps=RieszWeights())
    assert np.allclose(mdl.nuisances_.pi, 1.0)
    g = np.linspace(-1.5, 1.5, 30)
    est, _ = mdl.curve(g)
    t = 1.5 * g - 0.8 * g ** 2
    assert np.sqrt(np.mean(((est - est.mean()) - (t - t.mean())) ** 2)) < 0.25
