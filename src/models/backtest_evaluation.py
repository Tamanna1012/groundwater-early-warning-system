"""
Multi-split backtest: docs/ML_MODEL_RESULTS.md tested exactly ONE train/test
split (train < 2014, test 2014-2017) and found gradient boosting beats the
best baseline nationally (11.6%) but only wins in 4/10 of the fastest-
declining districts. A single split can't tell us whether that's a real,
consistent pattern or just what happened to come out of one particular
cut of the data -- so this script repeats the same comparison across 7
independent splits (test = one full year at a time, 2010 through 2016,
each trained on everything strictly before that year) and checks whether
the earlier findings hold up.

Output: docs/EVALUATION.md + reports/figures/backtest_mae_by_year.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _model_utils import FEATURE_COLS, TARGET, fit_and_predict_both

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "EVALUATION.md"
FIG_DIR = ROOT / "reports" / "figures"

TEST_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016]  # walk-forward: train on everything strictly before
MIN_DISTRICT_N = 30  # per split, per district -- below this a MAE comparison is too noisy to trust

WORST_DISTRICTS = [  # same list used throughout the modelling stage, from docs/DATA_AUDIT.md
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
]


def load() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def run_split(df: pd.DataFrame, test_year: int) -> dict:
    cutoff_start = f"{test_year}-01-01"
    cutoff_end = f"{test_year + 1}-01-01"
    train = df[df["date"] < cutoff_start]
    test = df[(df["date"] >= cutoff_start) & (df["date"] < cutoff_end)]
    test = test.dropna(subset=["depth_last_reading", "depth_same_quarter_last_year"])

    naive_mae = mae(test[TARGET], test["depth_last_reading"])
    seasonal_mae = mae(test[TARGET], test["depth_same_quarter_last_year"])
    best_baseline = min(naive_mae, seasonal_mae)

    fitted = fit_and_predict_both(train, test)
    ridge_mae = mae(test[TARGET], fitted["ridge"])
    gbm_mae = mae(test[TARGET], fitted["gbm"])

    per_row = test[["state", "district"]].copy()
    per_row["actual"] = test[TARGET].to_numpy()
    per_row["naive"] = test["depth_last_reading"].to_numpy()
    per_row["seasonal_naive"] = test["depth_same_quarter_last_year"].to_numpy()
    per_row["gbm"] = fitted["gbm"]

    district_wins, district_total = 0, 0
    for state, district in WORST_DISTRICTS:
        sub = per_row[(per_row["state"] == state) & (per_row["district"] == district)]
        if len(sub) < MIN_DISTRICT_N:
            continue
        district_best_baseline = min(mae(sub["actual"], sub["naive"]), mae(sub["actual"], sub["seasonal_naive"]))
        district_gbm_mae = mae(sub["actual"], sub["gbm"])
        district_total += 1
        district_wins += district_gbm_mae < district_best_baseline

    # The same check over EVERY district, not just the fastest-declining ten.
    # Without it there is no way to read the number above: a low win rate in
    # those ten could mean the model is weak at district level generally, or
    # weak only there, and those have very different consequences for a
    # system whose whole purpose is ranking districts.
    all_wins, all_total = 0, 0
    for _, sub in per_row.groupby(["state", "district"], observed=True):
        if len(sub) < MIN_DISTRICT_N:
            continue
        sub_best_baseline = min(mae(sub["actual"], sub["naive"]), mae(sub["actual"], sub["seasonal_naive"]))
        all_total += 1
        all_wins += mae(sub["actual"], sub["gbm"]) < sub_best_baseline

    return {
        "test_year": test_year, "n_train": len(train), "n_test": len(test),
        "naive_mae": naive_mae, "seasonal_mae": seasonal_mae, "best_baseline_mae": best_baseline,
        "ridge_mae": ridge_mae, "gbm_mae": gbm_mae,
        "gbm_improvement_pct": 100 * (best_baseline - gbm_mae) / best_baseline,
        "district_wins": district_wins, "district_total": district_total,
        "all_district_wins": all_wins, "all_district_total": all_total,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    results = [run_split(df, year) for year in TEST_YEARS]
    results_df = pd.DataFrame(results)

    mean_improvement = results_df["gbm_improvement_pct"].mean()
    std_improvement = results_df["gbm_improvement_pct"].std()
    n_splits_gbm_wins = int((results_df["gbm_mae"] < results_df["best_baseline_mae"]).sum())
    total_district_wins = int(results_df["district_wins"].sum())
    total_district_checks = int(results_df["district_total"].sum())
    total_all_wins = int(results_df["all_district_wins"].sum())
    total_all_checks = int(results_df["all_district_total"].sum())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results_df["test_year"], results_df["naive_mae"], marker="o", label="Naive", color="#8C8C8C", linestyle="--")
    ax.plot(results_df["test_year"], results_df["seasonal_mae"], marker="o", label="Seasonal-naive", color="#8C8C8C")
    ax.plot(results_df["test_year"], results_df["ridge_mae"], marker="o", label="Ridge regression", color="#4C72B0")
    ax.plot(results_df["test_year"], results_df["gbm_mae"], marker="o", label="Gradient boosting", color="#55A868")
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAE (m)")
    ax.set_title("Backtest: MAE by method across 7 independent train/test splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "backtest_mae_by_year.png", dpi=150)
    plt.close(fig)

    with DOCS_PATH.open("w") as f:
        f.write("# Evaluation: Multi-Split Backtest\n\n")
        f.write(
            f"`docs/ML_MODEL_RESULTS.md` tested one train/test split. This repeats the same comparison "
            f"across {len(TEST_YEARS)} independent walk-forward splits (test = one full year, trained "
            "on everything strictly before it) to check whether that single result was representative "
            "or just what one particular split happened to show.\n\n"
        )
        f.write("## Results by split\n\n")
        f.write("| Test year | Train n | Test n | Naive MAE | Seasonal-naive MAE | Ridge MAE | GBM MAE | GBM vs best baseline |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, row in results_df.iterrows():
            f.write(
                f"| {int(row['test_year'])} | {int(row['n_train']):,} | {int(row['n_test']):,} | "
                f"{row['naive_mae']:.3f} | {row['seasonal_mae']:.3f} | {row['ridge_mae']:.3f} | "
                f"{row['gbm_mae']:.3f} | {row['gbm_improvement_pct']:+.1f}% |\n"
            )
        f.write("\n![Backtest MAE by year](../reports/figures/backtest_mae_by_year.png)\n\n")

        f.write("## Is gradient boosting's national win consistent?\n\n")
        f.write(
            f"**Gradient boosting beats the best baseline in {n_splits_gbm_wins}/{len(TEST_YEARS)} splits**, "
            f"with a mean improvement of {mean_improvement:+.1f}% (std dev {std_improvement:.1f} "
            "percentage points across splits). "
        )
        if n_splits_gbm_wins == len(TEST_YEARS):
            f.write(
                "It wins in every single split tested -- the single-split result in ML_MODEL_RESULTS.md "
                "was not a fluke; this is a consistent, real improvement over the baselines.\n\n"
            )
        else:
            f.write(
                f"It does not win in every split, so the single-split result should be read as "
                "representative of a real but imperfectly consistent effect, not a guarantee.\n\n"
            )

        f.write("## Is the fastest-declining-district gap consistent?\n\n")
        f.write(
            f"Across all splits, gradient boosting beat the best baseline in **{total_district_wins}/"
            f"{total_district_checks}** individual (district, year) checks with at least {MIN_DISTRICT_N} "
            "test readings (fewer checks than districts x splits, since many district-years fall below "
            "that threshold and are excluded as too small to trust). "
        )
        win_rate = total_district_wins / total_district_checks if total_district_checks else float("nan")
        f.write(
            f"That's a {win_rate:.0%} win rate in exactly the districts that matter most for an "
            "early-warning use case. "
        )
        if win_rate < 0.5:
            f.write(
                "This confirms, rather than resolves, the concern raised in ML_MODEL_RESULTS.md: the "
                "model's district-level performance in the highest-priority areas is genuinely weaker "
                "and less reliable than its strong national average suggests. Before any real "
                "early-warning output is built on this model, it needs either district-specific "
                "tuning, more training data per district, or an explicit uncertainty estimate so a "
                "user knows when to trust a given district's forecast less.\n\n"
            )
        else:
            f.write(
                "This is a meaningfully more reassuring result than the single split in "
                "ML_MODEL_RESULTS.md suggested on its own.\n\n"
            )

        all_rate = total_all_wins / total_all_checks if total_all_checks else float("nan")
        f.write("## Is it weak at district level generally, or only in those ten?\n\n")
        f.write(
            f"The number above on its own is easy to over-read. Run the identical check over "
            f"**every** district rather than the fastest-declining ten, and gradient boosting beats "
            f"the best baseline in **{total_all_wins:,}/{total_all_checks:,}** (district, year) "
            f"cells -- a {all_rate:.0%} win rate, against {win_rate:.0%} in those ten. "
        )
        f.write(
            "Both numbers are real and they do not contradict each other; they are different "
            "populations. The model is not weak at district level in general. It is weak "
            f"*specifically* where the water table is falling fastest, winning about "
            f"{100 * (all_rate - win_rate):.0f} percentage points less often there than in a typical "
            "district. That is the precise form of the limitation, and the precise form matters: the "
            "broad reading understates a model that is usually right, while the narrow reading is the "
            "one that actually threatens the use case, because those are exactly the districts this "
            "system exists to flag.\n\n"
        )
        f.write(
            "`docs/DISTRICT_MODEL.md` follows this up by testing the most obvious explanation -- that "
            "the model simply isn't told which district a well is in -- and rules it out.\n\n"
        )

        f.write("## Honest conclusion\n\n")
        f.write(
            f"Gradient boosting is a genuine, backtested improvement over both simple baselines "
            f"nationally ({mean_improvement:+.1f}% average across {len(TEST_YEARS)} independent splits), "
            f"and it holds up across districts in general ({all_rate:.0%} of all district-year cells). "
            "Its advantage in the specific districts an early-warning system cares about most is the "
            f"exception ({win_rate:.0%} win rate), and should be treated as an open limitation, not a "
            "solved problem, going into the explainability and actionable-output stages.\n"
        )

    print(results_df[["test_year", "naive_mae", "seasonal_mae", "ridge_mae", "gbm_mae", "gbm_improvement_pct"]])
    print(f"GBM wins {n_splits_gbm_wins}/{len(TEST_YEARS)} splits nationally")
    print(f"GBM wins {total_district_wins}/{total_district_checks} district-year checks")
    print(f"Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
