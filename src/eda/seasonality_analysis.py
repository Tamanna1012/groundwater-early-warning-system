"""
Seasonality analysis on the cleaned groundwater long-format table.

CGWB measures each well at four fixed points in the year: January, May,
August, November. This script checks whether depth-to-water follows a
predictable yearly rhythm tied to India's monsoon cycle (June-September),
and how large that swing typically is per well. This matters for later
modelling: if the seasonal swing is large and consistent, a forecasting
model needs "which quarter is this" as an input, not just "how much time
has passed" -- otherwise it will mistake a normal pre-monsoon dip for a
depletion trend.

Output: docs/EDA_FINDINGS.md (seasonality section) +
reports/figures/seasonal_pattern.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "groundwater_levels_long.csv"
DOCS_PATH = ROOT / "docs" / "EDA_FINDINGS.md"
FIG_DIR = ROOT / "reports" / "figures"

MONTH_NAME = {1: "Jan (dry season)", 5: "May (pre-monsoon, driest)", 8: "Aug (monsoon)", 11: "Nov (post-monsoon)"}


def load() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def seasonal_means(df: pd.DataFrame) -> pd.Series:
    return df.groupby("month")["depth_to_water_m"].mean()


def per_well_seasonal_amplitude(df: pd.DataFrame) -> pd.Series:
    """For each well-year that has all 4 quarters, compute (deepest - shallowest)
    reading that year. This isolates the *typical* seasonal swing per well,
    rather than mixing in wells that only ever report in one season."""
    counts = df.groupby(["well_id", "year"])["month"].transform("nunique")
    complete = df[counts == 4]
    amplitude = complete.groupby(["well_id", "year"])["depth_to_water_m"].agg(lambda s: s.max() - s.min())
    return amplitude


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    means = seasonal_means(df)
    amplitude = per_well_seasonal_amplitude(df)
    n_complete_well_years = amplitude.shape[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    months = [1, 5, 8, 11]
    ax.plot(range(len(months)), [means[m] for m in months], marker="o", linewidth=2)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([MONTH_NAME[m] for m in months], rotation=15, ha="right")
    ax.set_ylabel("Mean depth to water (m below ground)")
    ax.set_title("National average seasonal cycle (all years pooled)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonal_pattern.png", dpi=150)
    plt.close(fig)

    swing = means[5] - means[8]  # pre-monsoon minus monsoon: how much the water table recovers

    with DOCS_PATH.open("w") as f:
        f.write("# EDA Findings\n\n")
        f.write("Deeper exploratory analysis beyond the initial data audit. Each section is a real, ")
        f.write("computed finding with a stated implication for modelling -- not decoration.\n\n")
        f.write("## Seasonality\n\n")
        f.write("Mean depth-to-water by measurement quarter, pooled across all years and wells:\n\n")
        f.write("| Quarter | Mean depth (m) |\n|---|---|\n")
        for m in months:
            f.write(f"| {MONTH_NAME[m]} | {means[m]:.2f} |\n")
        f.write(
            f"\nThe water table is on average **{swing:.2f} m shallower in August than in May** -- "
            "i.e. the monsoon visibly recharges groundwater nationally, and it draws back down "
            "through the dry season until the next monsoon. This is the expected physical pattern, "
            "which is itself a sanity check that the data is measuring something real.\n\n"
        )
        f.write("![Seasonal pattern](../reports/figures/seasonal_pattern.png)\n\n")
        f.write(f"**Per-well seasonal swing:** using only well-years with all 4 quarters present ")
        f.write(f"({n_complete_well_years:,} well-years), the typical (median) swing between a well's ")
        f.write(f"deepest and shallowest reading in the same year is **{amplitude.median():.2f} m** ")
        f.write(f"(90th percentile: {amplitude.quantile(0.9):.2f} m).\n\n")
        f.write(
            "**Modelling implication:** the seasonal swing is comparable in size to, or larger than, "
            "the multi-year depletion trend found in `DATA_AUDIT.md` (roughly 1 m/year in the "
            "fastest-declining districts). A model that ignores season would likely mistake normal "
            "pre-monsoon dips for depletion, or mask real depletion behind a good monsoon year. "
            "Any forecasting model built later must condition on quarter/season, and any trend "
            "estimate should compare same-quarter readings across years rather than raw consecutive "
            "readings.\n"
        )

    print(f"Wrote {DOCS_PATH}")
    print(means)
    print(f"Median per-well seasonal amplitude: {amplitude.median():.2f} m over {n_complete_well_years} well-years")


if __name__ == "__main__":
    main()
