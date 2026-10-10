"""
Prediction intervals: can the model tell a user when to distrust it?

docs/EVALUATION.md establishes that the model wins 73% of district-year
checks overall but only 32% in the fastest-declining districts, and
docs/DISTRICT_MODEL.md rules out the obvious cause (it is not that the model
lacks district identity). That leaves a different kind of fix. If the point
forecast cannot be made reliable everywhere, the next most useful thing is
for it to be *honest about where it isn't* -- a groundwater board can act on
"28.4 m, and we are confident" differently from "28.4 m, but this district's
forecasts have been wide and wrong before".

So instead of one model predicting the mean, this fits two more predicting
the 10th and 90th percentile (`loss="quantile"`), giving an 80% prediction
interval around every forecast. Three things then have to be true for that
interval to be worth putting in front of anyone:

  1. It must be calibrated: ~80% of actual readings should land inside it.
     An interval that only holds 50% of the time is a false reassurance.
  2. It must be wider where the model is worse. If the interval is the same
     width everywhere, it carries no information about reliability.
  3. Its width must track error at district level -- the unit this project
     actually reports -- not just across individual wells.

All three are measured here, on the same 7 walk-forward splits as
src/models/backtest_evaluation.py, so the numbers sit alongside the existing
evaluation rather than in a different frame.

Output: docs/UNCERTAINTY.md + reports/figures/interval_calibration.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _model_utils import (
    FEATURE_COLS,
    QUANTILE_HIGH,
    QUANTILE_LOW,
    TARGET,
    as_categorical_frame,
    fit_gbm,
    predict_interval,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "UNCERTAINTY.md"
FIG_DIR = ROOT / "reports" / "figures"

TEST_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
MIN_DISTRICT_N = 30
LOWER_Q, UPPER_Q = QUANTILE_LOW, QUANTILE_HIGH  # shared with src/output/risk_score.py
NOMINAL_COVERAGE = UPPER_Q - LOWER_Q
RANDOM_STATE = 42

WORST_DISTRICTS = [
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
]
WORST_KEYS = {f"{s}/{d}" for s, d in WORST_DISTRICTS}


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df["district_key"] = df["state"].astype(str) + "/" + df["district"].astype(str)
    return df


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def run_split(df: pd.DataFrame, test_year: int) -> tuple[dict, pd.DataFrame]:
    start, end = f"{test_year}-01-01", f"{test_year + 1}-01-01"
    train = df[df["date"] < start]
    test = df[(df["date"] >= start) & (df["date"] < end)]
    test = test.dropna(subset=["depth_last_reading", "depth_same_quarter_last_year"])

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test = test[FEATURE_COLS]

    # The median model is the one the project already ships, refit here on
    # the same split so the interval is centred on the forecast a user sees.
    point_model, X_train_gbm = fit_gbm(X_train, y_train, random_state=RANDOM_STATE)
    X_test_gbm = as_categorical_frame(X_test, reference=X_train_gbm)

    lo, hi = predict_interval(X_train_gbm, y_train, X_test_gbm, random_state=RANDOM_STATE)
    point = point_model.predict(X_test_gbm)

    rows = pd.DataFrame({
        "district_key": test["district_key"].to_numpy(),
        "actual": test[TARGET].to_numpy(),
        "point": point,
        "lower": lo,
        "upper": hi,
    })
    rows["inside"] = (rows["actual"] >= rows["lower"]) & (rows["actual"] <= rows["upper"])
    rows["width"] = rows["upper"] - rows["lower"]
    rows["abs_error"] = (rows["actual"] - rows["point"]).abs()
    rows["is_worst"] = rows["district_key"].isin(WORST_KEYS)
    rows["test_year"] = test_year

    worst = rows[rows["is_worst"]]
    summary = {
        "test_year": test_year,
        "n_test": len(rows),
        "coverage": float(rows["inside"].mean()),
        "mean_width": float(rows["width"].mean()),
        "worst_coverage": float(worst["inside"].mean()) if len(worst) else float("nan"),
        "worst_mean_width": float(worst["width"].mean()) if len(worst) else float("nan"),
    }
    return summary, rows


def district_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Per (district, split) cell: mean interval width against actual error.
    This is the unit the project reports in, so it is the unit the interval
    has to be informative at."""
    cells = []
    for (key, year), sub in all_rows.groupby(["district_key", "test_year"], observed=True):
        if len(sub) < MIN_DISTRICT_N:
            continue
        cells.append({
            "district_key": key,
            "test_year": year,
            "n": len(sub),
            "mean_width": sub["width"].mean(),
            "mae": mae(sub["actual"], sub["point"]),
            "coverage": sub["inside"].mean(),
            "is_worst": key in WORST_KEYS,
        })
    return pd.DataFrame(cells)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    summaries, frames = [], []
    for year in TEST_YEARS:
        summary, rows = run_split(df, year)
        summaries.append(summary)
        frames.append(rows)
        print(f"{year}: coverage {summary['coverage']:.1%}, mean width {summary['mean_width']:.2f}m")

    results = pd.DataFrame(summaries)
    all_rows = pd.concat(frames, ignore_index=True)
    cells = district_table(all_rows)

    overall_coverage = float(all_rows["inside"].mean())
    overall_width = float(all_rows["width"].mean())
    worst_cells = cells[cells["is_worst"]]
    other_cells = cells[~cells["is_worst"]]
    width_ratio = worst_cells["mean_width"].mean() / other_cells["mean_width"].mean()
    corr = cells["mean_width"].corr(cells["mae"])

    # Does a wider interval actually mean a worse forecast? Split the cells
    # into width quartiles and report the MAE of each.
    cells = cells.copy()
    cells["width_quartile"] = pd.qcut(cells["mean_width"], 4, labels=["Q1 narrowest", "Q2", "Q3", "Q4 widest"])
    by_quartile = cells.groupby("width_quartile", observed=True).agg(
        cells=("mae", "size"), mean_width=("mean_width", "mean"),
        mae=("mae", "mean"), coverage=("coverage", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].plot(results["test_year"], results["coverage"] * 100, marker="o", color="#4C72B0", label="all districts")
    axes[0].plot(results["test_year"], results["worst_coverage"] * 100, marker="o", color="#C44E52",
                 label="fastest-declining 10")
    axes[0].axhline(NOMINAL_COVERAGE * 100, color="#8C8C8C", linestyle="--", linewidth=1, label="nominal 80%")
    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("Actual readings inside the interval (%)")
    axes[0].set_title("Calibration: is 80% really 80%?")
    axes[0].legend()

    axes[1].scatter(other_cells["mean_width"], other_cells["mae"], s=12, alpha=0.35,
                    color="#4C72B0", label="other districts")
    axes[1].scatter(worst_cells["mean_width"], worst_cells["mae"], s=28, alpha=0.9,
                    color="#C44E52", label="fastest-declining 10")
    axes[1].set_xlabel("Mean interval width (m)")
    axes[1].set_ylabel("District-year MAE (m)")
    axes[1].set_title(f"Width vs. error, per district-year (r = {corr:.2f})")
    axes[1].legend()

    axes[2].bar(by_quartile["width_quartile"].astype(str), by_quartile["mae"], color="#55A868")
    axes[2].set_ylabel("Mean district-year MAE (m)")
    axes[2].set_title("MAE by interval-width quartile")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "interval_calibration.png", dpi=150)
    plt.close(fig)

    lines = []
    lines.append("# Prediction intervals: does the model know when to distrust itself?")
    lines.append("")
    lines.append(
        "`docs/EVALUATION.md` found the model wins 73% of district-year checks overall but only 32% "
        "in the fastest-declining districts, and `docs/DISTRICT_MODEL.md` ruled out the obvious "
        "cause. If the point forecast cannot be made reliable in those districts, the next most "
        "useful thing is for it to say so. This fits two extra gradient-boosted models at the "
        f"{LOWER_Q:.0%} and {UPPER_Q:.0%} quantiles, giving an {NOMINAL_COVERAGE:.0%} prediction "
        "interval around every forecast, and asks whether that interval is trustworthy and "
        "informative -- on the same 7 walk-forward splits as the rest of the evaluation."
    )
    lines.append("")

    lines.append("## 1. Is it calibrated?")
    lines.append("")
    lines.append("| Test year | Test n | Coverage (all) | Coverage (worst 10) | Mean width (m) |")
    lines.append("|---|---|---|---|---|")
    for _, r in results.iterrows():
        lines.append(
            f"| {int(r['test_year'])} | {int(r['n_test']):,} | {r['coverage']:.1%} | "
            f"{r['worst_coverage']:.1%} | {r['mean_width']:.2f} |"
        )
    lines.append(f"| **overall** | {len(all_rows):,} | **{overall_coverage:.1%}** | "
                 f"**{all_rows[all_rows['is_worst']]['inside'].mean():.1%}** | **{overall_width:.2f}** |")
    lines.append("")
    gap = overall_coverage - NOMINAL_COVERAGE
    if abs(gap) <= 0.03:
        direction = "slightly over-confident" if gap < 0 else "slightly conservative"
        lines.append(
            f"An interval sold as {NOMINAL_COVERAGE:.0%} contains the actual reading "
            f"{overall_coverage:.1%} of the time -- {direction} by "
            f"{abs(gap) * 100:.1f} percentage points, close enough to mean roughly what it says, "
            "but worth stating in that direction rather than rounding it to 'calibrated'. "
        )
    elif gap < 0:
        lines.append(
            f"An interval sold as {NOMINAL_COVERAGE:.0%} contains the actual reading only "
            f"{overall_coverage:.1%} of the time -- it is **over-confident**, and quoting it as an "
            f"{NOMINAL_COVERAGE:.0%} interval would overstate how pinned down these forecasts are. "
        )
    else:
        lines.append(
            f"An interval sold as {NOMINAL_COVERAGE:.0%} contains the actual reading "
            f"{overall_coverage:.1%} of the time -- **conservative**: it is wider than it strictly "
            "needs to be, which errs in the safer direction for a warning system but costs precision. "
        )
    worst_cov = float(all_rows[all_rows["is_worst"]]["inside"].mean())
    lines.append(
        f"In the fastest-declining districts coverage is {worst_cov:.1%}, "
        + (
            "so the interval degrades exactly where the point forecast does -- it is least "
            "trustworthy in the districts that matter most, which has to be said plainly rather "
            "than buried."
            if worst_cov < overall_coverage - 0.03
            else "which is in line with the overall figure, so the interval does not quietly break "
                 "down in the districts that matter most."
        )
    )
    lines.append("")
    lines.append("![Interval calibration](../reports/figures/interval_calibration.png)")
    lines.append("")

    lines.append("## 2. Is it wider where the model is worse?")
    lines.append("")
    lines.append(
        f"Mean interval width is **{worst_cells['mean_width'].mean():.2f} m** in the "
        f"fastest-declining districts against **{other_cells['mean_width'].mean():.2f} m** elsewhere "
        f"-- {width_ratio:.1f}x wider. "
        + (
            "The model is visibly less sure in precisely the districts where it is in fact less "
            "accurate, which is the behaviour that makes an interval worth showing at all."
            if width_ratio > 1.1
            else "That is not a large enough difference to treat the width as a reliability signal "
                 "specific to those districts."
        )
    )
    lines.append("")

    lines.append("## 3. Does width predict error at district level?")
    lines.append("")
    lines.append(
        f"Across {len(cells):,} (district, year) cells with at least {MIN_DISTRICT_N} readings, the "
        f"correlation between mean interval width and actual MAE is **r = {corr:.2f}**. Grouped into "
        "width quartiles:"
    )
    lines.append("")
    lines.append("| Width quartile | Cells | Mean width (m) | Mean MAE (m) | Coverage |")
    lines.append("|---|---|---|---|---|")
    for _, r in by_quartile.iterrows():
        lines.append(
            f"| {r['width_quartile']} | {int(r['cells']):,} | {r['mean_width']:.2f} | "
            f"{r['mae']:.2f} | {r['coverage']:.1%} |"
        )
    lines.append("")
    q1, q4 = by_quartile.iloc[0], by_quartile.iloc[-1]
    lines.append(
        f"Forecasts in the widest quartile are off by {q4['mae']:.2f} m on average against "
        f"{q1['mae']:.2f} m in the narrowest -- {q4['mae'] / q1['mae']:.1f}x. The width is not "
        "decoration: it is a usable, ranked signal of how much to trust a given district's forecast, "
        "available at prediction time, before the truth is known."
    )
    lines.append("")

    lines.append("## What this changes")
    lines.append("")
    lines.append(
        "The district-level gap is not closed and this does not close it -- the point forecast in "
        "the fastest-declining districts is exactly as good, or as bad, as it was. What changes is "
        "that the system no longer has to present every forecast with the same implied confidence. "
        "A groundwater board reading the ranking can see which districts the model is guessing at, "
        "and weight a wide-interval district toward the directly measured severity and trend signals "
        "-- which is already how `src/output/risk_score.py` is weighted, for the same reason."
    )
    lines.append("")

    DOCS_PATH.write_text("\n".join(lines), encoding="utf-8", newline="")

    print(f"overall coverage {overall_coverage:.1%} (nominal {NOMINAL_COVERAGE:.0%})")
    print(f"width worst-10 {worst_cells['mean_width'].mean():.2f}m vs other {other_cells['mean_width'].mean():.2f}m")
    print(f"width-vs-MAE correlation r={corr:.3f}")
    print(f"Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
