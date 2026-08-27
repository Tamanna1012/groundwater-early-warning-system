"""
Build the model-ready feature table from the cleaned groundwater long-format
data. Every feature here traces back to a specific finding in
docs/EDA_FINDINGS.md / docs/DATA_AUDIT.md -- see docs/FEATURES.md for the
full explanation of each one.

Output: data/processed/features.csv (one row per well-quarter reading,
gitignored -- regenerate with this script, don't hand-edit)

IMPORTANT CAVEAT (read before using `long_term_slope` for anything): it is
computed once per well from that well's ENTIRE history (including readings
that come after many of the rows it's attached to). That's fine for a
"what does this well look like overall" snapshot / for ranking wells by
current risk, but it is look-ahead bias if used as a training feature to
predict individual historical rows -- the model would be peeking at the
future. The baseline/forecast stage must replace this with an
expanding-window version (recomputed using only data available up to each
row's date) before training anything on it.
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
    df["rolling_mean_4"] = g.shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)

    # depth_same_quarter_last_year: self-merge each reading against the same
    # well's reading exactly 1 year earlier (same season, different year).
    lookup = df[["well_id", "date", "depth_to_water_m"]].copy()
    lookup["date"] = lookup["date"] + pd.DateOffset(years=1)
    lookup = lookup.rename(columns={"depth_to_water_m": "depth_same_quarter_last_year"})
    df = df.merge(lookup, on=["well_id", "date"], how="left")
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Static per-well long-term slope (m/year), same method as the
    district-level ranking in DATA_AUDIT.md. See the module docstring for
    why this is look-ahead and needs replacing before model training."""

    def slope(group: pd.DataFrame) -> float:
        if group["date"].nunique() < 3:
            return np.nan
        x = group["date"].dt.year + (group["date"].dt.month - 1) / 12
        y = group["depth_to_water_m"]
        m, _ = np.polyfit(x, y, 1)
        return m

    slopes = df.groupby("well_id").apply(slope, include_groups=False)
    slopes.name = "long_term_slope_m_per_year"
    return df.merge(slopes, on="well_id", how="left")


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
        f.write("| `long_term_slope_m_per_year` | Well's full-history linear trend | Depletion / early-warning signal |\n")
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
            "exactly 1 year before the current one (gaps in monitoring); `long_term_slope_m_per_year` "
            "is missing for wells with fewer than 3 distinct dates in their whole history.\n\n"
        )
        f.write(
            "## Important caveat: look-ahead bias in `long_term_slope_m_per_year`\n\n"
            "This feature is computed once per well from **that well's entire history**, including "
            "readings that come after many of the rows it's attached to. That makes it useful as a "
            "*current well profile* (e.g. for ranking wells/districts by risk today), but it is "
            "look-ahead bias if used to train a model to predict individual historical rows -- the "
            "model would effectively see the future. Before this feature table is used for baseline "
            "or ML forecasting (the next pipeline stage), this column must be replaced with an "
            "expanding-window version that only uses data available up to each row's own date.\n"
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
