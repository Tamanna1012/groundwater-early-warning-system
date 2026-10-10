"""Tests for the prediction-interval machinery in src/models/_model_utils.py
and the way src/output/build_dashboard.py presents it.

The interval is only useful if it means what docs/UNCERTAINTY.md says it
means, so the things pinned here are the ones that would silently change
that: the bound ordering, and the fact that the interval measured during
evaluation is the same interval published in the output.
"""

import numpy as np
import pandas as pd
import pytest

from _model_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    QUANTILE_HIGH,
    QUANTILE_LOW,
    as_categorical_frame,
    predict_interval,
)


def synthetic_frame(n=240, seed=0):
    """A small frame with the real feature columns, enough signal for the
    quantile models to fit something non-degenerate."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(5, 40, n)
    data = {c: rng.normal(0, 1, n) for c in NUMERIC_FEATURES}
    data["depth_last_reading"] = base
    data["rolling_mean_4"] = base + rng.normal(0, 0.4, n)
    data["depth_same_quarter_last_year"] = base + rng.normal(0, 0.8, n)
    X = pd.DataFrame(data)
    X["season"] = rng.choice(["Jan", "May", "Aug", "Nov"], n)
    X["site_type"] = rng.choice(["Dug Well", "Bore Well"], n)
    X["state"] = rng.choice(["RJ", "PB", "HR"], n)
    y = pd.Series(base + rng.normal(0, 1.5, n), name="depth_to_water_m")
    return X[FEATURE_COLS], y


def test_quantile_bounds_bracket_the_nominal_interval():
    """The two quantiles must straddle the median and span the width the
    docs claim. If someone moves them to 0.05/0.95, every coverage number in
    docs/UNCERTAINTY.md silently becomes wrong."""
    assert 0 < QUANTILE_LOW < 0.5 < QUANTILE_HIGH < 1
    assert QUANTILE_HIGH - QUANTILE_LOW == pytest.approx(0.8)


def test_predict_interval_never_returns_a_negative_width():
    """The two quantile models are fit independently, so nothing in sklearn
    forces lower <= upper; where they cross, the helper has to order them
    rather than hand back an interval that is backwards."""
    X, y = synthetic_frame()
    X_gbm = as_categorical_frame(X)

    lower, upper = predict_interval(X_gbm, y, X_gbm)

    assert len(lower) == len(upper) == len(X)
    assert (upper >= lower).all(), "interval width must never be negative"


def test_predict_interval_is_deterministic_for_a_fixed_seed():
    """docs/UNCERTAINTY.md quotes exact coverage figures, so the same inputs
    and seed have to give the same interval."""
    X, y = synthetic_frame()
    X_gbm = as_categorical_frame(X)

    first = predict_interval(X_gbm, y, X_gbm, random_state=42)
    second = predict_interval(X_gbm, y, X_gbm, random_state=42)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_predict_interval_mostly_contains_the_training_target():
    """A sanity floor, not a calibration test (that is what the backtest in
    src/models/uncertainty.py is for): an 80% interval fit and evaluated on
    the same rows should contain most of them. If this drops near zero, the
    quantile loss is not doing what the pipeline assumes."""
    X, y = synthetic_frame()
    X_gbm = as_categorical_frame(X)

    lower, upper = predict_interval(X_gbm, y, X_gbm)
    inside = ((y.to_numpy() >= lower) & (y.to_numpy() <= upper)).mean()

    assert inside > 0.6


# --- how the dashboard presents it ---------------------------------------


def confidence_cells(rows_html: str) -> list[str]:
    import re

    return re.findall(r'<td data-sort="(\d)"><span class="(conf-[a-z]+)"[^>]*>([A-Za-z]+)</span></td>', rows_html)


def test_dashboard_confidence_sorts_by_reliability_not_alphabetically():
    """High/Low/Medium sorted as text puts the least reliable forecasts in
    the middle of the column. Each cell carries an explicit data-sort value
    so the column orders High -> Medium -> Low instead."""
    from build_dashboard import CONFIDENCE_ORDER, build_table_rows

    assert CONFIDENCE_ORDER["High"] < CONFIDENCE_ORDER["Medium"] < CONFIDENCE_ORDER["Low"]
    # The alphabetical order this guards against:
    assert sorted(CONFIDENCE_ORDER) == ["High", "Low", "Medium"]

    df = pd.DataFrame({
        "rank": [1, 2, 3],
        "state": ["RJ", "PB", "HR"],
        "district": ["Sikar", "Sangrur", "Karnal"],
        "n_wells": [70, 48, 125],
        "current_depth": [43.23, 27.24, 16.75],
        "measured_trend": [0.790, 0.853, 0.437],
        "risk_score": [86.3, 84.9, 87.9],
        "forecast_confidence": ["Low", "Medium", "High"],
        "forecast_interval_m": [6.6, 4.1, 2.5],
        "known_model_bias_flag": [True, True, False],
    })

    cells = confidence_cells(build_table_rows(df))
    assert [label for *_, label in cells] == ["Low", "Medium", "High"]
    assert [int(sort_value) for sort_value, _, _ in cells] == [2, 1, 0]
    assert [css for _, css, _ in cells] == ["conf-low", "conf-med", "conf-high"]


def test_dashboard_confidence_cell_shows_the_width_on_hover():
    from build_dashboard import build_table_rows

    df = pd.DataFrame({
        "rank": [1], "state": ["RJ"], "district": ["Sikar"], "n_wells": [70],
        "current_depth": [43.23], "measured_trend": [0.79], "risk_score": [86.3],
        "forecast_confidence": ["Low"], "forecast_interval_m": [6.64],
        "known_model_bias_flag": [True],
    })

    html = build_table_rows(df)
    assert 'title="80% prediction interval is 6.6 m wide"' in html


def test_dashboard_column_types_line_up_with_the_headers():
    """A missed entry here shifts every later column's sort behaviour."""
    from build_dashboard import TABLE_COLUMNS, TABLE_HEADERS, TABLE_TYPES

    assert len(TABLE_COLUMNS) == len(TABLE_HEADERS) == len(TABLE_TYPES)
    assert TABLE_TYPES[TABLE_HEADERS.index("Forecast confidence")] == "num"
