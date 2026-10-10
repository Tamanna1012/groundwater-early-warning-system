"""Tests for src/data/clean_cgwb_data.py.

The cleaner is where the project's sign convention and its data-quality
filters live, so these tests pin the decisions that everything downstream
silently assumes: wide-to-long shape, quarter-column parsing, the
implausible-reading filter, and the blank-SITE_TYPE normalisation.
"""

import pandas as pd
import pytest

from clean_cgwb_data import MONTH_MAP, clean, parse_quarter_column

RAW_HEADER = ",STATE,DISTRICT,LAT,LON,SITE_TYPE,WLCODE,May 1996,Aug 1996,Nov 1996,Jan 1997\n"


def write_raw(tmp_path, rows):
    """Write a minimal CGWB-shaped wide CSV (unnamed index col + 4 quarters)."""
    path = tmp_path / "raw.csv"
    path.write_text(RAW_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_parse_quarter_column_maps_cgwb_quarters():
    assert parse_quarter_column("May 1996") == pd.Timestamp("1996-05-01")
    assert parse_quarter_column("Jan 2017") == pd.Timestamp("2017-01-01")
    assert set(MONTH_MAP) == {"Jan", "May", "Aug", "Nov"}


def test_clean_is_one_row_per_non_missing_reading(tmp_path):
    raw = write_raw(tmp_path, ["1,RJ,Sikar,27.6,75.1,Dug Well,W1,5.0,NA,6.0,NA\n"])
    out = clean(raw)

    assert len(out) == 2, "NA readings are dropped, not carried as rows"
    assert list(out["date"]) == [pd.Timestamp("1996-05-01"), pd.Timestamp("1996-11-01")]
    assert list(out["depth_to_water_m"]) == [5.0, 6.0]
    assert out["well_id"].tolist() == ["W1", "W1"]


def test_clean_drops_physically_implausible_readings(tmp_path):
    raw = write_raw(tmp_path, ["1,RJ,Sikar,27.6,75.1,Dug Well,W1,-3.0,250.0,0.0,200.0\n"])
    out = clean(raw)

    # -3 m (above ground) and 250 m (out of observation-well range) go;
    # the 0 m and 200 m boundary values are kept.
    assert sorted(out["depth_to_water_m"]) == [0.0, 200.0]


def test_clean_normalises_whitespace_only_site_type(tmp_path):
    raw = write_raw(
        tmp_path,
        [
            '1,RJ,Sikar,27.6,75.1,"   ",W1,5.0,NA,NA,NA\n',
            "2,PB,Sangrur,30.2,75.8,Bore Well,W2,7.0,NA,NA,NA\n",
        ],
    )
    out = clean(raw)

    site_types = dict(zip(out["well_id"], out["site_type"]))
    assert site_types["W1"] == "Unlabeled", "blank must not become its own invisible category"
    assert site_types["W2"] == "Bore Well"


def test_clean_coerces_non_numeric_readings_to_missing(tmp_path):
    raw = write_raw(tmp_path, ["1,RJ,Sikar,27.6,75.1,Dug Well,W1,abc,4.0,NA,NA\n"])
    out = clean(raw)

    assert list(out["depth_to_water_m"]) == [4.0]


def test_clean_output_is_sorted_by_well_then_date(tmp_path):
    raw = write_raw(
        tmp_path,
        [
            "1,PB,Sangrur,30.2,75.8,Bore Well,W2,7.0,8.0,NA,NA\n",
            "2,RJ,Sikar,27.6,75.1,Dug Well,W1,5.0,6.0,NA,NA\n",
        ],
    )
    out = clean(raw)

    # Every downstream per-well feature (shift, expanding slope) assumes this.
    assert out["well_id"].tolist() == ["W1", "W1", "W2", "W2"]
    assert out.groupby("well_id")["date"].is_monotonic_increasing.all()
    assert out.index.tolist() == [0, 1, 2, 3], "index is reset, not inherited from the wide frame"


def test_clean_exposes_the_expected_columns(tmp_path):
    raw = write_raw(tmp_path, ["1,RJ,Sikar,27.6,75.1,Dug Well,W1,5.0,NA,NA,NA\n"])
    out = clean(raw)

    assert list(out.columns) == [
        "well_id", "state", "district", "lat", "lon",
        "site_type", "date", "quarter", "depth_to_water_m",
    ]
    assert out["lat"].iloc[0] == pytest.approx(27.6)
