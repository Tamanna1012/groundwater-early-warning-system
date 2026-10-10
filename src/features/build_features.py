"""
Build the model-ready feature table from the cleaned groundwater long-format
data. Every feature here traces back to a specific finding in
docs/EDA_FINDINGS.md / docs/DATA_AUDIT.md -- see docs/FEATURES.md for the
full explanation of each one.

Output: data/processed/features.csv (one row per well-quarter reading,
gitignored -- regenerate with this script, don't hand-edit)

`long_term_slope_m_per_year` is an EXPANDING-WINDOW trend: each row's value
uses only that well's readings strictly before its own date. An earlier
version of this script computed one full-history slope per well and reused
it on every row, which was look-ahead bias (a training feature that could
see the future). Fixed here -- see docs/FEATURES.md for the before/after.

Every per-well feature here is also grouped per well, which sounds obvious
but is easy to get wrong in pandas: a groupby's `.shift()` returns a plain
Series, so chaining `.rolling()` onto it silently rolls across well
boundaries. `rolling_mean_4` had exactly that bug and is now re-grouped
before rolling; tests/test_features.py pins the behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "groundwater_levels_long.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "FEATURES.md"

N_NEIGHBORS = 5


def load() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    return df.sort_values(["well_id", "date"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["season"] = df["date"].dt.strftime("%b")  # Jan / May / Aug / Nov

    g = df.groupby("well_id")["depth_to_water_m"]
    df["n_prior_readings"] = df.groupby("well_id").cumcount()
    df["depth_last_reading"] = g.shift(1)
    # The .groupby(df["well_id"]) in the middle is load-bearing: g.shift(1)
    # returns a plain Series (not a grouped object), so rolling straight off
    # it would slide the 4-reading window ACROSS well boundaries -- the first
    # rows of each well would average in the previous well's readings. Group
    # again before rolling so each window stays inside one well.
    df["rolling_mean_4"] = (
        g.shift(1)
        .groupby(df["well_id"])
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # depth_same_quarter_last_year: self-merge each reading against the same
    # well's reading exactly 1 year earlier (same season, different year).
    lookup = df[["well_id", "date", "depth_to_water_m"]].copy()
    lookup["date"] = lookup["date"] + pd.DateOffset(years=1)
    lookup = lookup.rename(columns={"depth_to_water_m": "depth_same_quarter_last_year"})
    df = df.merge(lookup, on=["well_id", "date"], how="left")
    return df


MIN_POINTS_FOR_SLOPE = 3

# The time axis is measured in years from this epoch rather than as raw
# calendar years. The running-sums fit below computes `n*Sxx - Sx^2`, which
# subtracts two large, nearly-equal numbers; with x ~ 2000 that cancellation
# costs about 8 significant digits. Shifting by a constant cannot change a
# slope, but it keeps x in 0..25 and the fit accurate to full precision.
SLOPE_EPOCH_YEAR = 1996  # first year in the CGWB series


def _year_axis(dates: pd.Series) -> np.ndarray:
    """Dates as fractional years since SLOPE_EPOCH_YEAR."""
    return (dates.dt.year + (dates.dt.month - 1) / 12 - SLOPE_EPOCH_YEAR).to_numpy()


def _expanding_slope_for_well(group: pd.DataFrame) -> pd.Series:
    """Per-row trend (m/year) using only readings strictly BEFORE that row's
    own date -- an expanding-window linear fit, not the well's full history.
    Computed via running sums (standard incremental least-squares) so it's
    O(n) per well instead of refitting from scratch at every row."""
    x = _year_axis(group["date"])
    y = group["depth_to_water_m"].to_numpy()
    n_total = len(x)

    # prior_sum_*[i] = sum over rows [0, i) -- excludes row i itself
    zero = np.array([0.0])
    prior_n = np.arange(n_total, dtype=float)
    prior_sum_x = np.concatenate([zero, np.cumsum(x)[:-1]]) if n_total > 1 else zero.copy()
    prior_sum_y = np.concatenate([zero, np.cumsum(y)[:-1]]) if n_total > 1 else zero.copy()
    prior_sum_xy = np.concatenate([zero, np.cumsum(x * y)[:-1]]) if n_total > 1 else zero.copy()
    prior_sum_xx = np.concatenate([zero, np.cumsum(x * x)[:-1]]) if n_total > 1 else zero.copy()

    denom = prior_n * prior_sum_xx - prior_sum_x**2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (prior_n * prior_sum_xy - prior_sum_x * prior_sum_y) / denom

    slope[(prior_n < MIN_POINTS_FOR_SLOPE) | (denom == 0)] = np.nan
    return pd.Series(slope, index=group.index)


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row expanding-window trend (m/year): each row's slope uses only
    that well's readings strictly before its own date, so it's a legitimate
    training feature (no look-ahead), unlike a single full-history slope
    per well would be.

    Built by concatenating per-well results explicitly rather than with
    `groupby(...).apply(...)`: when the applied function returns a Series,
    apply *infers* whether to stack the results or widen them into a frame,
    and that inference flips depending on how many groups there are and
    whether their indexes line up -- it returns a DataFrame for a single
    well, which then fails to assign to one column. Concatenating and
    reindexing is the same result, with no inference involved.
    """
    slopes = [_expanding_slope_for_well(group) for _, group in df.groupby("well_id", sort=False)]
    combined = pd.concat(slopes) if slopes else pd.Series(dtype="float64")
    df["long_term_slope_m_per_year"] = combined.reindex(df.index)
    return df


def add_spatial_feature(df: pd.DataFrame) -> pd.DataFrame:
    """neighbor_mean_depth: average depth among each well's N_NEIGHBORS
    nearest wells (by lat/lon), evaluated on the same date. Nearest-neighbor
    search uses plain lat/lon coordinates (not true great-circle distance)
    purely to pick *which* wells count as neighbors -- fine at India's
    scale, since it only needs to rank nearby wells, not measure distance
    precisely (we already validated real spatial correlation with proper
    haversine distances in spatial_analysis.py)."""
    wells = df[["well_id", "lat", "lon"]].drop_duplicates("well_id").reset_index(drop=True)
    coords = wells[["lat", "lon"]].to_numpy()
    tree = cKDTree(coords)
    # k+1 because the well's own point is always its own nearest neighbor
    _, idx = tree.query(coords, k=N_NEIGHBORS + 1)
    neighbor_ids = wells["well_id"].to_numpy()[idx[:, 1:]]  # drop self column

    neighbor_map = pd.DataFrame(neighbor_ids, index=wells["well_id"])
    neighbor_map = neighbor_map.reset_index().melt(id_vars="well_id", value_name="neighbor_id")[["well_id", "neighbor_id"]]

    depth_lookup = df[["well_id", "date", "depth_to_water_m"]].rename(
        columns={"well_id": "neighbor_id", "depth_to_water_m": "neighbor_depth"}
    )
    joined = neighbor_map.merge(depth_lookup, on="neighbor_id", how="left")
    neighbor_mean = joined.groupby(["well_id", "date"])["neighbor_depth"].mean().reset_index()
    neighbor_mean = neighbor_mean.rename(columns={"neighbor_depth": "neighbor_mean_depth"})

    return df.merge(neighbor_mean, on=["well_id", "date"], how="left")


def build() -> pd.DataFrame:
    df = load()
    df = add_temporal_features(df)
    df = add_trend_feature(df)
    df = add_spatial_feature(df)

    cols = [
        "well_id", "state", "district", "lat", "lon", "site_type", "date", "season",
        "depth_to_water_m",
        "depth_last_reading", "depth_same_quarter_last_year", "rolling_mean_4",
        "long_term_slope_m_per_year", "neighbor_mean_depth", "n_prior_readings",
    ]
    return df[cols]


def write_docs(features: pd.DataFrame) -> None:
    n_rows = len(features)
    coverage = {
        "depth_last_reading": features["depth_last_reading"].notna().mean(),
        "depth_same_quarter_last_year": features["depth_same_quarter_last_year"].notna().mean(),
        "rolling_mean_4": features["rolling_mean_4"].notna().mean(),
        "long_term_slope_m_per_year": features["long_term_slope_m_per_year"].notna().mean(),
        "neighbor_mean_depth": features["neighbor_mean_depth"].notna().mean(),
    }

    with DOCS_PATH.open("w") as f:
        f.write("# Feature Table\n\n")
        f.write(f"Generated by `src/features/build_features.py`. {n_rows:,} rows, one per well-quarter reading.\n\n")
        f.write("## Column reference\n\n")
        f.write("| Feature | What it is | EDA finding it's based on |\n|---|---|---|\n")
        f.write("| `season` | Measurement quarter (Jan/May/Aug/Nov) | Seasonality |\n")
        f.write("| `depth_same_quarter_last_year` | Same well's reading exactly 1 year earlier | Seasonality |\n")
        f.write("| `depth_last_reading` | Same well's immediately previous reading | Baseline signal |\n")
        f.write("| `rolling_mean_4` | Mean of the well's last 4 readings (excl. current) | Baseline signal |\n")
        f.write("| `long_term_slope_m_per_year` | Well's trend (m/year) using only readings before this row | Depletion / early-warning signal |\n")
        f.write("| `site_type` | Dug/bore/tube well etc. | Site-type effects |\n")
        f.write("| `neighbor_mean_depth` | Mean depth among 5 nearest wells, same date | Spatial patterns |\n")
        f.write("| `district`, `state` | Administrative region | Spatial patterns / output granularity |\n")
        f.write("| `n_prior_readings` | How many readings this well had before this one | Confidence / coverage |\n\n")
        f.write("## Missingness (fraction of rows with a real value)\n\n")
        f.write("| Feature | Coverage |\n|---|---|\n")
        for k, v in coverage.items():
            f.write(f"| `{k}` | {v:.1%} |\n")
        f.write(
            "\n`depth_same_quarter_last_year` is missing whenever a well doesn't have a reading "
            f"exactly 1 year before the current one (gaps in monitoring); `long_term_slope_m_per_year` "
            f"is missing whenever a row has fewer than {MIN_POINTS_FOR_SLOPE} prior readings for that "
            "well to fit a trend from yet.\n\n"
        )
        f.write(
            "`depth_last_reading` and `rolling_mean_4` should always have *identical* coverage, "
            "and it should be below 100%: both look only at readings before the current row, so "
            "both are missing on exactly one row per well -- its first. That equality is a cheap "
            "integrity check on this table, and it did not always hold. `rolling_mean_4` used to "
            "report 100% coverage, which is impossible for a feature that excludes the current "
            "reading; the cause was a window that rolled across well boundaries, so each well's "
            "first rows averaged the *previous* well's readings. See the note below and "
            "`tests/test_features.py`.\n\n"
        )
        f.write(
            "## Note: two data-leakage bugs this table used to have\n\n"
            "Both were found and fixed before any model was trained on it, and both are now "
            "pinned by `tests/test_features.py`. The first was `rolling_mean_4` rolling across "
            "well boundaries (above); the second is below.\n\n"
            "## Note: `long_term_slope_m_per_year` is an expanding-window feature\n\n"
            "Each row's value is fit using only that well's readings strictly BEFORE that row's own "
            "date -- never the row's own reading or any later one. This makes it safe to use as a "
            "training feature (no look-ahead). An earlier version of this script computed one "
            "full-history slope per well and copied it onto every row for that well, which *was* "
            "look-ahead bias (rows near the start of a well's record would have `seen` its future "
            "decline). That bug was caught and fixed before any model was trained on this table -- "
            "worth knowing about since it's the kind of mistake that silently inflates offline "
            "accuracy without showing up until deployment.\n"
        )

    print(f"Wrote {DOCS_PATH}")


def main() -> None:
    features = build()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote {len(features):,} rows to {FEATURES_PATH}")
    write_docs(features)


if __name__ == "__main__":
    main()
