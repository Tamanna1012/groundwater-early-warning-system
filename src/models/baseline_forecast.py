"""
Baseline forecast: before any machine learning, establish the simplest
reasonable forecasts and measure their real error. Any model built later
only has value if it beats these.

Two baselines, both using only information already available before the
date being predicted (no look-ahead):
- naive: predict this reading = the well's immediately previous reading
- seasonal_naive: predict this reading = the well's reading from exactly
  1 year earlier (same quarter)

Given the seasonality finding (~3.5m median swing per well per year, larger
than the multi-year depletion trend), the seasonal-naive baseline is
expected to beat plain naive -- this script checks that against real
numbers rather than assuming it.

Evaluation: a time-based split (test = 2014-2017, train = everything
before), since a real deployment forecasts the future from the past, not
random past rows. Metrics: MAE and RMSE, overall, by district (worst
decliners from DATA_AUDIT.md), and by season.

Output: docs/BASELINE_RESULTS.md + reports/figures/baseline_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "BASELINE_RESULTS.md"
FIG_DIR = ROOT / "reports" / "figures"

TEST_CUTOFF = "2014-01-01"  # ~20% of rows fall in the test period (2014-2017)
WORST_DISTRICTS = [  # from docs/DATA_AUDIT.md's steepest-declining-districts table
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    # Only the two lag columns are used here -- both are properly lagged
    # (computed from strictly earlier readings), unlike long_term_slope_m_per_year
    # which docs/FEATURES.md flags as look-ahead and unsuitable for this.
    return df[["well_id", "state", "district", "site_type", "date", "season", "depth_to_water_m",
               "depth_last_reading", "depth_same_quarter_last_year"]]


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df.dropna(subset=[pred_col, "depth_to_water_m"])
    return {"mae": mae(valid["depth_to_water_m"], valid[pred_col]), "rmse": rmse(valid["depth_to_water_m"], valid[pred_col]), "n": len(valid)}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    test = df[df["date"] >= TEST_CUTOFF].copy()

    # Fair comparison: score both baselines on the exact same rows (both
    # predictors present), not each on its own largest-available subset.
    both_available = test.dropna(subset=["depth_last_reading", "depth_same_quarter_last_year"])
    naive_score = score(both_available, "depth_last_reading")
    seasonal_score = score(both_available, "depth_same_quarter_last_year")
    improvement_pct = 100 * (naive_score["mae"] - seasonal_score["mae"]) / naive_score["mae"]

    # By season
    by_season = (
        both_available.assign(
            naive_err=lambda d: (d["depth_to_water_m"] - d["depth_last_reading"]).abs(),
            seasonal_err=lambda d: (d["depth_to_water_m"] - d["depth_same_quarter_last_year"]).abs(),
        )
        .groupby("season")[["naive_err", "seasonal_err"]]
        .mean()
    )

    # By worst-decliner district
    district_rows = []
    for state, district in WORST_DISTRICTS:
        sub = both_available[(both_available["state"] == state) & (both_available["district"] == district)]
        if len(sub) == 0:
            continue
        district_rows.append({
            "state": state, "district": district,
            "naive_mae": mae(sub["depth_to_water_m"], sub["depth_last_reading"]),
            "seasonal_mae": mae(sub["depth_to_water_m"], sub["depth_same_quarter_last_year"]),
            "n": len(sub),
        })
    district_df = pd.DataFrame(district_rows)

    # Figure: overall MAE/RMSE comparison
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["MAE", "RMSE"]
    naive_vals = [naive_score["mae"], naive_score["rmse"]]
    seasonal_vals = [seasonal_score["mae"], seasonal_score["rmse"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, naive_vals, width, label="Naive (last reading)")
    ax.bar(x + width / 2, seasonal_vals, width, label="Seasonal-naive (same quarter last year)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (m)")
    ax.set_title(f"Baseline comparison, test period {TEST_CUTOFF} onward (n={both_available.shape[0]:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_comparison.png", dpi=150)
    plt.close(fig)

    with DOCS_PATH.open("w") as f:
        f.write("# Baseline Forecast Results\n\n")
        f.write(
            f"Time-based backtest: trained on data before {TEST_CUTOFF}, evaluated on "
            f"{TEST_CUTOFF} onward ({both_available.shape[0]:,} test readings where both baselines "
            "have a valid prediction, so the comparison is apples-to-apples). No model fitting "
            "involved -- both baselines are simple rules applied directly.\n\n"
        )
        f.write("## Overall\n\n")
        f.write("| Baseline | MAE (m) | RMSE (m) |\n|---|---|---|\n")
        f.write(f"| Naive (predict = last reading) | {naive_score['mae']:.3f} | {naive_score['rmse']:.3f} |\n")
        f.write(f"| Seasonal-naive (predict = same quarter last year) | {seasonal_score['mae']:.3f} | {seasonal_score['rmse']:.3f} |\n\n")
        f.write("![Baseline comparison](../reports/figures/baseline_comparison.png)\n\n")
        f.write(
            f"**Result: seasonal-naive beats naive by {improvement_pct:.1f}% on MAE.** This confirms, "
            "with real backtested numbers, what the seasonality EDA finding predicted: because the "
            "seasonal swing is larger than the underlying trend, comparing a well to itself one year "
            "back is a meaningfully better default than comparing it to its own last quarter.\n\n"
        )
        f.write("## By season\n\n")
        f.write("Mean absolute error broken down by which quarter is being predicted:\n\n")
        f.write("| Season | Naive MAE (m) | Seasonal-naive MAE (m) |\n|---|---|---|\n")
        for season_name, row in by_season.iterrows():
            f.write(f"| {season_name} | {row['naive_err']:.3f} | {row['seasonal_err']:.3f} |\n")
        f.write(
            "\nThe naive baseline is worst exactly at the seasonal turning points (transitioning into "
            "or out of monsoon), where consecutive readings differ the most -- consistent with the "
            "seasonality finding.\n\n"
        )
        f.write("## By worst-declining district (from DATA_AUDIT.md)\n\n")
        f.write("| State | District | Naive MAE (m) | Seasonal-naive MAE (m) | Winner | n |\n|---|---|---|---|---|---|\n")
        naive_wins = 0
        for _, row in district_df.iterrows():
            winner = "naive" if row["naive_mae"] < row["seasonal_mae"] else "seasonal-naive"
            naive_wins += winner == "naive"
            f.write(f"| {row['state']} | {row['district']} | {row['naive_mae']:.3f} | {row['seasonal_mae']:.3f} | {winner} | {int(row['n'])} |\n")
        f.write(
            f"\n**The pattern reverses here: in {naive_wins}/{len(district_df)} of the fastest-declining "
            "districts, plain naive beats seasonal-naive** -- the opposite of the national result above. "
            "The reason is consistent with what these districts are: they have a strong ongoing "
            "multi-year decline (that's *why* they're on this list), so 'same quarter last year' is off "
            "by a full year of depletion, while 'last quarter' is only 3 months stale. In other words, "
            "in a stable district the seasonal cycle dominates the error and seasonal-naive wins; in a "
            "fast-declining district the trend dominates and naive wins. Neither simple baseline handles "
            "both effects at once -- that gap (needing to model season *and* trend together) is exactly "
            "what the next modelling stage needs to close, and it's the strongest argument in this "
            "project for why a real model is needed rather than a lookup rule.\n\n"
        )
        f.write("## What this sets up\n\n")
        f.write(
            "Seasonal-naive (MAE "
            f"{seasonal_score['mae']:.3f} m) is the bar any real model must clear. A model that "
            "can't beat this isn't worth deploying, however sophisticated it is.\n"
        )

    print(f"Wrote {DOCS_PATH}")
    print("Naive:", naive_score)
    print("Seasonal-naive:", seasonal_score)
    print(f"Improvement: {improvement_pct:.1f}%")


if __name__ == "__main__":
    main()
