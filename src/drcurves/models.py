"""Default nuisance learner: a stack of an additive spline model and gradient boosting.

The spline-ridge member fits smooth, additive structure (like a GAM); boosting picks up
interactions and thresholds. A non-negative linear blend, chosen by internal 3-fold CV,
decides how much of each to use. In our simulations this beat boosting alone by a wide margin,
mainly because boosted trees approximate smooth dose-confounder relationships with steps, and
those steps turn into extreme weights in the dose density.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def default_model(categorical_features=None, n_knots=5, max_iter=500, learning_rate=0.05, cv=2, n_jobs=None,
                  random_state=0):
    """Spline-ridge + boosting stack, usable for the outcome model and the dose mean/scale models.

    categorical_features : column indices to one-hot encode in the spline member and treat as
        categorical in boosting (e.g. group codes). Other columns get cubic B-splines.
    cv : folds used to choose the blend. 2 was as accurate as 3 in our simulations and a third faster.
    n_jobs : fit the two members in parallel (passed to StackingRegressor).
    Missing values: the spline member imputes the column mean; boosting handles NaN natively.
    Boosting stops early once a 10% validation split stops improving, so max_iter is a ceiling.
    """
    cat = list(categorical_features) if categorical_features is not None else []
    if cat:
        pre = _SplineOneHot(cat, n_knots)
    else:
        pre = make_pipeline(SimpleImputer(), StandardScaler(), SplineTransformer(n_knots=n_knots, degree=3))
    gam = make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))
    hgb = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, early_stopping=True,
                                        n_iter_no_change=10, validation_fraction=0.1, random_state=random_state,
                                        categorical_features=cat or None)
    return StackingRegressor([("gam", gam), ("hgb", hgb)], final_estimator=LinearRegression(positive=True),
                             cv=cv, n_jobs=n_jobs)


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
            ("spl", make_pipeline(SimpleImputer(), StandardScaler(), SplineTransformer(n_knots=self.n_knots, degree=3)), num),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat)]


def default_classifier(n_knots=5, max_iter=500, learning_rate=0.05, cv=2, n_jobs=None, random_state=0):
    """Spline logistic regression + boosting, blended by a logistic regression on their
    out-of-fold probabilities. Used by DiscreteGPS for the probability of each dose level."""
    logit = make_pipeline(SimpleImputer(), StandardScaler(), SplineTransformer(n_knots=n_knots, degree=3),
                          LogisticRegression(C=1.0, max_iter=2000))
    hgb = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, early_stopping=True,
                                         n_iter_no_change=10, validation_fraction=0.1, random_state=random_state)
    return StackingClassifier([("logit", logit), ("hgb", hgb)], final_estimator=LogisticRegression(max_iter=2000),
                              stack_method="predict_proba", cv=cv, n_jobs=n_jobs)
