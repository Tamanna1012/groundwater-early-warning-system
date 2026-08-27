"""Shared model-fitting logic for src/models/ml_forecast.py and
src/models/backtest_evaluation.py, so both scripts train identical models
and a backtest across multiple splits is actually comparable to the single
split first used in ml_forecast.py -- not two subtly different pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "depth_last_reading", "depth_same_quarter_last_year", "rolling_mean_4",
    "long_term_slope_m_per_year", "neighbor_mean_depth", "n_prior_readings", "lat", "lon",
]
CATEGORICAL_FEATURES = ["season", "site_type", "state"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "depth_to_water_m"


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("model", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def as_categorical_frame(X: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        if reference is not None:
            X[c] = X[c].astype("category").cat.set_categories(reference[c].cat.categories)
        else:
            X[c] = X[c].astype("category")
    return X


def fit_gbm(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[HistGradientBoostingRegressor, pd.DataFrame]:
    X_train_gbm = as_categorical_frame(X_train)
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=random_state)
    model.fit(X_train_gbm, y_train)
    return model, X_train_gbm


def predict_gbm(model: HistGradientBoostingRegressor, X_test: pd.DataFrame, X_train_gbm: pd.DataFrame):
    X_test_gbm = as_categorical_frame(X_test, reference=X_train_gbm)
    return model.predict(X_test_gbm)


def fit_and_predict_both(train: pd.DataFrame, test: pd.DataFrame, random_state: int = 42) -> dict:
    """Fit Ridge + gradient boosting on `train`, predict on `test`. Returns
    a dict of {'ridge': array, 'gbm': array} aligned to test's row order,
    plus the fitted models and the categorical-encoded train/test frames
    (needed by callers that want to run permutation importance afterward)."""
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test = test[FEATURE_COLS]

    ridge = fit_ridge(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    gbm, X_train_gbm = fit_gbm(X_train, y_train, random_state=random_state)
    X_test_gbm = as_categorical_frame(X_test, reference=X_train_gbm)
    gbm_pred = gbm.predict(X_test_gbm)

    return {
        "ridge": ridge_pred, "gbm": gbm_pred,
        "ridge_model": ridge, "gbm_model": gbm,
        "X_train_gbm": X_train_gbm, "X_test_gbm": X_test_gbm,
    }
