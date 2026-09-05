"""Additive logistic GAM: per-feature cubic splines, exact attributions.

A Gaussian process would model the same non-linear surface and then make
feature contributions a kernel-similarity story — the opposite of what this
project claims. A GAM keeps the surface non-linear while staying strictly
additive: g(E[y]) = intercept + Σ f_i(x_i). The contribution of feature i
is f_i(x_i), and those terms plus the intercept reproduce the logit.

Implemented with scikit-learn splines so calibration can still wrap a
FrozenEstimator. InterpretML's EBM is the same additive idea with bagged
boosting; the spline GAM is the lighter estimator that does not fight the
isotonic + conformal pipeline on 569 rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler


class AdditiveGAM(ClassifierMixin, BaseEstimator):
    """Standardise, expand each column into a cubic spline, logistic link.

    Defaults (4 knots, C=0.1) leave enough bend for non-linear nuclei
    without saturating isotonic calibration so badly that qhat collapses to 0.
    """

    def __init__(
        self,
        n_knots: int = 4,
        degree: int = 3,
        C: float = 0.1,
        max_iter: int = 5000,
        random_state: int = 0,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        self.splines_ = ColumnTransformer(
            [
                (
                    f"s{i}",
                    SplineTransformer(
                        n_knots=self.n_knots,
                        degree=self.degree,
                        include_bias=False,
                    ),
                    [i],
                )
                for i in range(self.n_features_in_)
            ]
        ).fit(Xs)
        Z = self.splines_.transform(Xs)
        self._basis_slices = self._slices_from_transformer()
        self.lr_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        ).fit(Z, y)
        self.classes_ = self.lr_.classes_
        return self

    def _slices_from_transformer(self) -> list[slice]:
        weights = self.splines_.transformers_
        slices = []
        cursor = 0
        for _, trans, _ in weights:
            n_out = trans.n_features_out_
            slices.append(slice(cursor, cursor + n_out))
            cursor += n_out
        return slices

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.splines_.transform(self.scaler_.transform(X))

    def decision_function(self, X):
        return self.lr_.decision_function(self._design(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._design(X))

    def predict(self, X):
        return self.lr_.predict(self._design(X))

    @property
    def intercept_(self) -> float:
        return float(self.lr_.intercept_[0])

    def feature_contributions(self, X) -> np.ndarray:
        """Additive term per original feature. Rows sum with intercept to logit."""
        Z = self._design(X)
        coefs = self.lr_.coef_[0]
        out = np.zeros((Z.shape[0], self.n_features_in_))
        for i, sl in enumerate(self._basis_slices):
            out[:, i] = Z[:, sl] @ coefs[sl]
        return out


class LinearAdditive(ClassifierMixin, BaseEstimator):
    """Standardise, then a plain logistic link. The same interface as
    `AdditiveGAM`, without the splines.

    This is the default, and the splines are not, because the splines were
    measured and lost. Controlled A/B over 25 seeds, identical pipeline and
    calibration, WDBC:

        metric          linear      GAM
        AUC (uncal)     0.9942      0.9902
        Brier (uncal)   0.0216      0.0486
        ECE  (uncal)    0.0313      0.1114
        coverage        0.9396      0.9141
        abstention      4.6%        6.1%

    Linear wins on all eight. At n=569 with 30 features, per-feature cubic
    splines buy flexibility the data does not support and pay for it in
    calibration — ECE 3.6x worse, Brier more than doubled — and calibration
    is the one property this whole project rests on. A model that bends
    nicely and misstates its own confidence is the exact failure Guo et al.
    (2017) describe, arrived at by choosing the more expressive model.

    `AdditiveGAM` is kept rather than deleted: it is the honest record of
    what was tried, and re-running the comparison is how that table stays
    true rather than becoming folklore.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 5000, random_state: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        self.lr_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        ).fit(self.scaler_.transform(X), np.asarray(y))
        self.classes_ = self.lr_.classes_
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(np.atleast_2d(np.asarray(X, dtype=float)))

    def decision_function(self, X):
        return self.lr_.decision_function(self._design(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._design(X))

    def predict(self, X):
        return self.lr_.predict(self._design(X))

    @property
    def intercept_(self) -> float:
        return float(self.lr_.intercept_[0])

    def feature_contributions(self, X) -> np.ndarray:
        """coefficient x standardised value, per feature.

        Exact, not an approximation: these terms plus the intercept reproduce
        the model's logit, which is what lets the UI call the contribution
        chart the model's own arithmetic rather than a post-hoc explanation.
        """
        return self._design(X) * self.lr_.coef_[0]
