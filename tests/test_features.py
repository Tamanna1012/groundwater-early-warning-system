"""Tests for src/features/build_features.py.

These pin the two properties that are easy to break and expensive to notice:
every per-well feature must stay inside its own well, and no feature may see
a reading at or after its own row's date (look-ahead bias). Both have
actually gone wrong in this file's history -- `long_term_slope_m_per_year`
was once a full-history slope copied onto every row, and `rolling_mean_4`
once rolled across well boundaries because a groupby's `.shift()` returns a
plain Series.
"""

import numpy as np
import pandas as pd
import pytest

from build_features import (
    MIN_POINTS_FOR_SLOPE,
    _expanding_slope_for_well,
    add_temporal_features,
    add_trend_feature,
)

QUARTER_MONTHS = [1, 5, 8, 11]


def quarterly_dates(n, start_year=2000):
    """n consecutive CGWB-style quarterly dates (Jan/May/Aug/Nov)."""
    out = []
    year, i = start_year, 0
    while len(out) < n:
        out.append(pd.Timestamp(year=year + i // 4, month=QUARTER_MONTHS[i % 4], day=1))
        i += 1
    return out


def make_long(wells):
    """Build a cleaned-long-format frame from {well_id: [depths]}."""
    rows = []
    for well_id, depths in wells.items():
        for date, depth in zip(quarterly_dates(len(depths)), depths):
            rows.append({"well_id": well_id, "date": date, "depth_to_water_m": depth})
    return (
        pd.DataFrame(rows)
        .sort_values(["well_id", "date"])
        .reset_index(drop=True)
    )


# --- rolling_mean_4 -------------------------------------------------------


def test_rolling_mean_4_does_not_leak_across_wells():
    """A well's first rows must not average in the previous well's readings.

    This is the regression test for the bug this file used to have: with the
    frame sorted by (well_id, date), rolling off an ungrouped Series made
    well B's first rows average well A's tail.
    """
    df = make_long({"A": [10.0, 10.0, 10.0, 10.0], "B": [1.0, 2.0, 3.0, 4.0]})
    out = add_temporal_features(df)
    b = out[out["well_id"] == "B"].reset_index(drop=True)

    assert np.isnan(b.loc[0, "rolling_mean_4"]), "first reading of a well has no prior readings"
    assert b.loc[1, "rolling_mean_4"] == pytest.approx(1.0)  # mean of B's [1.0]
    assert b.loc[2, "rolling_mean_4"] == pytest.approx(1.5)  # mean of B's [1.0, 2.0]
    assert b.loc[3, "rolling_mean_4"] == pytest.approx(2.0)  # mean of B's [1.0, 2.0, 3.0]
    # If the window leaked, every value above would be pulled toward A's 10.0.
    assert (b["rolling_mean_4"].dropna() < 10.0).all()


def test_rolling_mean_4_excludes_the_current_reading():
    """It is the mean of the previous up-to-4 readings, never including the
    row's own reading -- that would be leaking the target into a feature."""
    depths = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = add_temporal_features(make_long({"A": depths}))

    expected = [np.nan] + [float(np.mean(depths[max(0, i - 4):i])) for i in range(1, len(depths))]
    for i, want in enumerate(expected):
        got = out.loc[i, "rolling_mean_4"]
        if np.isnan(want):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(want), f"row {i}"


def test_rolling_mean_4_window_is_capped_at_four():
    out = add_temporal_features(make_long({"A": [1.0, 1.0, 1.0, 1.0, 100.0, 1.0]}))
    # Row 5 sees rows 1-4 ([1, 1, 1, 100]), NOT row 0 as well.
    assert out.loc[5, "rolling_mean_4"] == pytest.approx((1 + 1 + 1 + 100) / 4)


# --- depth_last_reading / depth_same_quarter_last_year --------------------


def test_depth_last_reading_is_the_wells_own_previous_reading():
    out = add_temporal_features(make_long({"A": [5.0, 6.0], "B": [50.0, 60.0]}))
    b = out[out["well_id"] == "B"].reset_index(drop=True)
    assert np.isnan(b.loc[0, "depth_last_reading"])
    assert b.loc[1, "depth_last_reading"] == pytest.approx(50.0)


def test_depth_same_quarter_last_year_is_exactly_one_year_back():
    depths = [1.0, 2.0, 3.0, 4.0, 9.0]  # 5th reading is Jan of the next year
    out = add_temporal_features(make_long({"A": depths}))
    # Row 4 is Jan 2001, so it looks back to row 0 (Jan 2000) -- the same
    # quarter, one year earlier -- not to the immediately preceding Nov.
    assert out.loc[4, "depth_same_quarter_last_year"] == pytest.approx(depths[0])
    # Rows with no reading exactly one year earlier stay missing.
    assert np.isnan(out.loc[0, "depth_same_quarter_last_year"])


# --- long_term_slope_m_per_year ------------------------------------------


def test_expanding_slope_uses_only_strictly_prior_readings():
    """The core no-look-ahead property: appending later readings must not
    change the slope already computed for an earlier row."""
    short = make_long({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    extended = make_long({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 99.0, 99.0, 99.0]})

    slope_short = add_trend_feature(short)["long_term_slope_m_per_year"]
    slope_extended = add_trend_feature(extended)["long_term_slope_m_per_year"]

    pd.testing.assert_series_equal(
        slope_short, slope_extended.iloc[: len(slope_short)], check_names=False
    )


def test_expanding_slope_is_nan_until_enough_prior_points():
    out = add_trend_feature(make_long({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    slope = out["long_term_slope_m_per_year"]
    assert slope.iloc[:MIN_POINTS_FOR_SLOPE].isna().all()
    assert slope.iloc[MIN_POINTS_FOR_SLOPE:].notna().all()


def test_expanding_slope_matches_an_explicit_fit_on_prior_rows():
    depths = [2.0, 2.5, 3.1, 3.9, 4.4, 5.2]
    df = make_long({"A": depths})
    got = _expanding_slope_for_well(df)

    x_all = (df["date"].dt.year + (df["date"].dt.month - 1) / 12).to_numpy()
    for i in range(MIN_POINTS_FOR_SLOPE, len(depths)):
        want = np.polyfit(x_all[:i], np.array(depths[:i]), 1)[0]
        assert got.iloc[i] == pytest.approx(want, rel=1e-9)


def test_expanding_slope_sign_convention_is_metres_per_year():
    """Depth-to-water grows as the water table falls, so a depleting well
    must come out with a POSITIVE slope of the right magnitude (see the sign
    convention note in clean_cgwb_data.py).

    Note the depths are derived from the actual dates rather than stepped by
    a constant per reading: CGWB's quarters are Jan/May/Aug/Nov, which are 4,
    3, 3 and 2 months apart, so "+0.25 m per reading" is not "+1 m/year".
    """
    dates = quarterly_dates(9)
    years = [d.year + (d.month - 1) / 12 for d in dates]
    depths = [10.0 + 1.0 * (y - years[0]) for y in years]  # exactly +1 m/year

    out = add_trend_feature(make_long({"A": depths}))
    assert out["long_term_slope_m_per_year"].iloc[-1] == pytest.approx(1.0, rel=1e-9)


def test_expanding_slope_is_per_well():
    """Two wells with opposite trends must not contaminate each other."""
    rising = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    falling = [9.0, 8.5, 8.0, 7.5, 7.0, 6.5]
    out = add_trend_feature(make_long({"A": rising, "B": falling}))

    assert out[out["well_id"] == "A"]["long_term_slope_m_per_year"].iloc[-1] > 0
    assert out[out["well_id"] == "B"]["long_term_slope_m_per_year"].iloc[-1] < 0
