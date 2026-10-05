"""Preprocessing and estimator definitions for the four-variable ridge logistic model.

The class names and module path in this file are part of the serialized model
format: the packaged ``model.joblib`` refers to ``painpredict.modeling.IQRCapper``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260823
C_GRID = np.logspace(-3, 2, 8)


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clip each column to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data.

    Columns with zero IQR are left uncapped.
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, x, y=None):
        frame = pd.DataFrame(x).copy()
        self.columns_ = list(frame.columns)
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.multiplier * iqr).where(iqr.gt(0), -np.inf)
        self.upper_ = (q3 + self.multiplier * iqr).where(iqr.gt(0), np.inf)
        return self

    def transform(self, x):
        frame = pd.DataFrame(x, columns=self.columns_).copy()
        return frame.clip(self.lower_, self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_ if input_features is None else input_features, dtype=object)


def make_preprocessor(features: Iterable[str]) -> ColumnTransformer:
    """IQR capping, median imputation with missing indicators, then standardization."""

    numeric = Pipeline(
        [
            ("cap", IQRCapper()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, list(features))], remainder="drop", verbose_feature_names_out=False)


def make_search(features: Sequence[str], cv, *, n_jobs: int | None = None) -> GridSearchCV:
    """Ridge (L2) logistic regression with C tuned by log loss on ``cv``."""

    model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=20_000, random_state=RANDOM_STATE)
    return GridSearchCV(
        Pipeline([("preprocess", make_preprocessor(features)), ("model", model)]),
        {"model__C": C_GRID},
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
