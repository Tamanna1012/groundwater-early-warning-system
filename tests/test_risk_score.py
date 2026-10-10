"""Tests for the serving-side feature construction in src/output/risk_score.py.

This is the train/serve boundary: the risk score forecasts one quarter past
each well's last real reading, so its features have to mean the same thing
they meant during training (src/features/build_features.py) -- measured over
the readings strictly before the row's own date. A mismatch here would not
raise an error anywhere, it would just quietly degrade the ranking the whole
project exists to produce.
"""

import numpy as np
import pandas as pd
import pytest

from risk_score import NEXT_QUARTER, next_quarter_date, percentile_rank, well_full_history_slope


def test_next_quarter_date_walks_the_cgwb_quarters():
    assert next_quarter_date(pd.Timestamp("2016-01-01")) == pd.Timestamp("2016-05-01")
    assert next_quarter_date(pd.Timestamp("2016-05-01")) == pd.Timestamp("2016-08-01")
    assert next_quarter_date(pd.Timestamp("2016-08-01")) == pd.Timestamp("2016-11-01")


def test_next_quarter_date_rolls_the_year_over_at_november():
    """Nov -> Jan is the only step that changes the year; getting this wrong
    would forecast into the past."""
    assert next_quarter_date(pd.Timestamp("2016-11-01")) == pd.Timestamp("2017-01-01")
    assert set(NEXT_QUARTER) == {"Jan", "May", "Aug", "Nov"}


def test_well_full_history_slope_is_metres_per_year():
    dates = pd.Series(pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01"]))
    depths = pd.Series([10.0, 11.0, 12.0, 13.0])  # +1 m/year

    assert well_full_history_slope(dates, depths) == pytest.approx(1.0, rel=1e-9)


def test_well_full_history_slope_needs_three_distinct_dates():
    dates = pd.Series(pd.to_datetime(["2010-01-01", "2010-05-01"]))
    depths = pd.Series([10.0, 11.0])

    assert np.isnan(well_full_history_slope(dates, depths))


def test_well_full_history_slope_is_negative_when_the_table_recovers():
    dates = pd.Series(pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01"]))
    depths = pd.Series([12.0, 11.0, 10.0])  # getting shallower = recovering

    assert well_full_history_slope(dates, depths) < 0


def test_percentile_rank_is_0_to_100_and_order_preserving():
    out = percentile_rank(pd.Series([1.0, 5.0, 3.0, 9.0]))

    assert out.min() > 0 and out.max() == pytest.approx(100.0)
    assert out.iloc[3] > out.iloc[1] > out.iloc[2] > out.iloc[0]
