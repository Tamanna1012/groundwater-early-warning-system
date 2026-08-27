"""
Clean the raw CGWB quarterly groundwater-level mirror into a tidy long-format
table ready for EDA / forecasting.

Input:  data/raw/cgwb_mirror/cgwb_quarterly_1996_2017.csv
        (wide format: one row per well, one column per quarter, e.g. "May 1996")
Output: data/processed/groundwater_levels_long.csv
        (one row per well-quarter reading)
        columns: well_id, state, district, lat, lon, site_type,
                 date, quarter, depth_to_water_m

Depth-to-water is in metres below ground level, as reported by CGWB, so a
LARGER value means the water table is DEEPER / worse (this matters for
sign conventions in every downstream model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "cgwb_mirror" / "cgwb_quarterly_1996_2017.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "groundwater_levels_long.csv"

ID_COLS = ["STATE", "DISTRICT", "LAT", "LON", "SITE_TYPE", "WLCODE"]

# CGWB's quarterly readings are nominally taken in Jan / May / Aug / Nov.
MONTH_MAP = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}


def parse_quarter_column(col: str) -> pd.Timestamp:
    month_str, year_str = col.split()
    return pd.Timestamp(year=int(year_str), month=MONTH_MAP[month_str], day=1)


def clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, index_col=0)

    time_cols = [c for c in df.columns if c not in ID_COLS]
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A handful of rows have a whitespace-only SITE_TYPE (not truly missing,
    # just an unlabeled/blank category in the source export) -- normalize so
    # it doesn't silently become its own invisible category downstream.
    df["SITE_TYPE"] = df["SITE_TYPE"].fillna("").str.strip().replace("", "Unlabeled")

    long = df.melt(
        id_vars=ID_COLS,
        value_vars=time_cols,
        var_name="quarter",
        value_name="depth_to_water_m",
    )

    long = long.dropna(subset=["depth_to_water_m"]).copy()

    long["date"] = long["quarter"].map(parse_quarter_column)
    long = long.rename(
        columns={
            "STATE": "state",
            "DISTRICT": "district",
            "LAT": "lat",
            "LON": "lon",
            "SITE_TYPE": "site_type",
            "WLCODE": "well_id",
        }
    )

    # Drop physically implausible readings: negative depth (above ground) or
    # absurdly deep (>200 m is outside CGWB's normal observation-well range
    # and is almost always a data-entry error in this kind of export).
    before = len(long)
    long = long[(long["depth_to_water_m"] >= 0) & (long["depth_to_water_m"] <= 200)]
    dropped = before - len(long)
    if dropped:
        print(f"Dropped {dropped} out-of-range readings (<0m or >200m)")

    long = long.sort_values(["well_id", "date"]).reset_index(drop=True)
    long = long[["well_id", "state", "district", "lat", "lon", "site_type", "date", "quarter", "depth_to_water_m"]]
    return long


def main() -> None:
    long = clean()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(long):,} well-quarter readings for {long['well_id'].nunique():,} wells to {OUT_PATH}")


if __name__ == "__main__":
    main()
