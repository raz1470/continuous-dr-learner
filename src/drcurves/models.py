"""Default nuisance learner: a stack of an additive spline model and gradient boosting.

The spline-ridge member fits smooth, additive structure (like a GAM); boosting picks up
interactions and thresholds. A non-negative linear blend, chosen by internal 3-fold CV,
decides how much of each to use. In our simulations this beat boosting alone by a wide margin,
mainly because boosted trees approximate smooth dose-confounder relationships with steps, and
those steps turn into extreme weights in the dose density.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def default_model(categorical_features=None, n_knots=5, max_iter=200, learning_rate=0.05, random_state=0):
    """Spline-ridge + boosting stack, usable for the outcome model and the dose mean/scale models.

    categorical_features : column indices to one-hot encode in the spline member and treat as
        categorical in boosting (e.g. group codes). Other columns get cubic B-splines.
    """
    cat = list(categorical_features) if categorical_features is not None else []
    if cat:
        pre = _SplineOneHot(cat, n_knots)
    else:
        pre = make_pipeline(StandardScaler(), SplineTransformer(n_knots=n_knots, degree=3))
    gam = make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))
    hgb = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, random_state=random_state,
                                        categorical_features=cat or None)
    return StackingRegressor([("gam", gam), ("hgb", hgb)], final_estimator=LinearRegression(positive=True), cv=3)


class _SplineOneHot(ColumnTransformer):
    """B-splines on numeric columns, one-hot on categorical ones; numeric columns found at fit time."""

    def __init__(self, cat, n_knots=5):
        self.cat = cat
        self.n_knots = n_knots
        super().__init__([])

    def fit_transform(self, X, y=None, **kw):
        self._set_transformers(X)
        return super().fit_transform(X, y, **kw)

    def fit(self, X, y=None, **kw):
        self._set_transformers(X)
        return super().fit(X, y, **kw)

    def _set_transformers(self, X):
        num = [j for j in range(np.asarray(X).shape[1]) if j not in self.cat]
        self.transformers = [
            ("spl", make_pipeline(StandardScaler(), SplineTransformer(n_knots=self.n_knots, degree=3)), num),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat)]
