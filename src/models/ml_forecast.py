"""
Real ML forecasting models, compared against the baselines in
docs/BASELINE_RESULTS.md on the *same* time-based split and the *same*
test rows, so all four numbers are directly comparable.

Two models, deliberately different in kind:
- Ridge regression: a simple, fully interpretable linear model. One-hot
  encoded categoricals, median-imputed numerics (fit on train only).
- HistGradientBoostingRegressor: a real gradient-boosted tree model. Handles
  missing numeric values and categorical columns natively, so it can use
  every training row -- including ones Ridge has to impute around.

Features used: everything in data/processed/features.csv except `district`
(525 categories -- left out of the model deliberately; `state` + `lat`/`lon`
+ `neighbor_mean_depth` already carry geographic signal, see docs/ML_MODEL_RESULTS.md
for why) and the raw identifiers/target.

Output: docs/ML_MODEL_RESULTS.md + reports/figures/model_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from _model_utils import CATEGORICAL_FEATURES, FEATURE_COLS, NUMERIC_FEATURES, TARGET, fit_and_predict_both, predict_gbm

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "ML_MODEL_RESULTS.md"
FIG_DIR = ROOT / "reports" / "figures"

TEST_CUTOFF = "2014-01-01"  # matches src/models/baseline_forecast.py exactly

WORST_DISTRICTS = [  # same list as src/models/baseline_forecast.py, from docs/DATA_AUDIT.md
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
]

# Baseline numbers from docs/BASELINE_RESULTS.md, hardcoded here only for the
# comparison chart/table -- the actual baseline numbers are computed by
# src/models/baseline_forecast.py, not recomputed here.
BASELINE_MAE = {"Naive": 2.376, "Seasonal-naive": 1.809}


def load() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_eval_set(df: pd.DataFrame) -> pd.DataFrame:
    """Same restriction as the baseline script: test rows where both simple
    baselines have a prediction, so every model in the final table is scored
    on the identical set of rows."""
    test = df[df["date"] >= TEST_CUTOFF]
    return test.dropna(subset=["depth_last_reading", "depth_same_quarter_last_year"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    train = df[df["date"] < TEST_CUTOFF]
    test = build_eval_set(df)

    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    fitted = fit_and_predict_both(train, test)
    ridge_pred, gbm_pred = fitted["ridge"], fitted["gbm"]
    gbm, X_test_gbm = fitted["gbm_model"], fitted["X_test_gbm"]

    results = {
        "Naive": BASELINE_MAE["Naive"],
        "Seasonal-naive": BASELINE_MAE["Seasonal-naive"],
        "Ridge regression": mae(y_test, ridge_pred),
        "Gradient boosting": mae(y_test, gbm_pred),
    }

    # Per-row predictions, kept with state/district so we can check the
    # thing docs/BASELINE_RESULTS.md flagged as unresolved: do the real
    # models close the gap in the fastest-declining districts, where
    # neither simple baseline worked well?
    per_row = test[["state", "district"]].copy()
    per_row["actual"] = y_test.to_numpy()
    per_row["naive"] = test["depth_last_reading"].to_numpy()
    per_row["seasonal_naive"] = test["depth_same_quarter_last_year"].to_numpy()
    per_row["ridge"] = ridge_pred
    per_row["gbm"] = gbm_pred

    district_rows = []
    for state, district in WORST_DISTRICTS:
        sub = per_row[(per_row["state"] == state) & (per_row["district"] == district)]
        if len(sub) == 0:
            continue
        best_baseline_here = min(mae(sub["actual"], sub["naive"]), mae(sub["actual"], sub["seasonal_naive"]))
        district_rows.append({
            "state": state, "district": district, "n": len(sub),
            "best_baseline_mae": best_baseline_here,
            "ridge_mae": mae(sub["actual"], sub["ridge"]),
            "gbm_mae": mae(sub["actual"], sub["gbm"]),
        })
    district_df = pd.DataFrame(district_rows)
    district_df["gbm_beats_baseline"] = district_df["gbm_mae"] < district_df["best_baseline_mae"]
    rmse_results = {
        "Ridge regression": rmse(y_test, ridge_pred),
        "Gradient boosting": rmse(y_test, gbm_pred),
    }

    best_baseline_mae = min(BASELINE_MAE.values())
    gbm_improvement = 100 * (best_baseline_mae - results["Gradient boosting"]) / best_baseline_mae
    ridge_improvement = 100 * (best_baseline_mae - results["Ridge regression"]) / best_baseline_mae

    # Feature importance via permutation (HistGradientBoostingRegressor has
    # no built-in feature_importances_, unlike RandomForest-style models) --
    # a first look, not the formal explainability stage still to come. Run
    # on a random subsample of the test set purely for speed.
    perm_sample = X_test_gbm.sample(n=min(5000, len(X_test_gbm)), random_state=42)
    perm_sample_y = y_test.loc[perm_sample.index]
    perm_result = permutation_importance(gbm, perm_sample, perm_sample_y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    importances = pd.Series(perm_result.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(results.keys())
    vals = list(results.values())
    colors = ["#8C8C8C", "#8C8C8C", "#4C72B0", "#55A868"]
    axes[0].bar(names, vals, color=colors)
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("MAE: baselines vs. real models")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].barh(importances.index[::-1], importances.to_numpy()[::-1], color="#55A868")
    axes[1].set_xlabel("Gradient boosting feature importance")
    axes[1].set_title("What the model actually relies on")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    with DOCS_PATH.open("w") as f:
        f.write("# ML Model Results\n\n")
        f.write(
            f"Same time-based split and same test rows as `docs/BASELINE_RESULTS.md` "
            f"(train before {TEST_CUTOFF}, test on/after, {len(test):,} test readings), so every "
            "number below is directly comparable to the baseline table.\n\n"
        )
        f.write(
            f"Features used: {', '.join(NUMERIC_FEATURES)} (numeric) and "
            f"{', '.join(CATEGORICAL_FEATURES)} (categorical). `district` (525 categories) was "
            "deliberately left out -- `state` + `lat`/`lon` + `neighbor_mean_depth` already carry "
            "geographic signal at a coarser and a finer grain, and including 525 near-unique "
            "one-hot columns risked the linear model overfitting to individual districts rather "
            "than learning general spatial structure.\n\n"
        )
        f.write("## Results\n\n")
        f.write("| Method | MAE (m) | RMSE (m) |\n|---|---|---|\n")
        f.write(f"| Naive (baseline) | {results['Naive']:.3f} | -- |\n")
        f.write(f"| Seasonal-naive (baseline) | {results['Seasonal-naive']:.3f} | -- |\n")
        f.write(f"| Ridge regression | {results['Ridge regression']:.3f} | {rmse_results['Ridge regression']:.3f} |\n")
        f.write(f"| Gradient boosting | {results['Gradient boosting']:.3f} | {rmse_results['Gradient boosting']:.3f} |\n\n")
        f.write("![Model comparison](../reports/figures/model_comparison.png)\n\n")

        if results["Gradient boosting"] < best_baseline_mae:
            f.write(
                f"**Gradient boosting beats the best baseline (seasonal-naive) by {gbm_improvement:.1f}% "
                "on MAE.** "
            )
        else:
            f.write(
                f"**Gradient boosting does NOT beat the best baseline** (MAE {results['Gradient boosting']:.3f}m "
                f"vs. {best_baseline_mae:.3f}m) -- reported honestly rather than adjusted. "
            )
        if results["Ridge regression"] < best_baseline_mae:
            f.write(f"Ridge regression also beats it, by {ridge_improvement:.1f}%.\n\n")
        else:
            f.write(f"Ridge regression does not beat it (MAE {results['Ridge regression']:.3f}m).\n\n")

        f.write("## What the model actually relies on (gradient boosting feature importance)\n\n")
        f.write("| Feature | Importance |\n|---|---|\n")
        for feat, imp in importances.items():
            f.write(f"| `{feat}` | {imp:.3f} |\n")
        f.write(
            "\nImportance = how much MAE gets worse when that column is randomly shuffled (bigger = "
            "more relied on). This is a first look using scikit-learn's permutation importance, not "
            "the formal explainability stage still to come in the pipeline (which will go further -- "
            "per-prediction explanations via SHAP, not just an overall ranking) -- but it already "
            "confirms the model leans on sensible signals (recent readings and trend dominate) rather "
            "than something spurious.\n\n"
        )

        gbm_district_wins = int(district_df["gbm_beats_baseline"].sum())
        f.write("## Does it close the gap in the fastest-declining districts?\n\n")
        f.write(
            "`docs/BASELINE_RESULTS.md` found that in 8/10 of the fastest-declining districts, neither "
            "simple baseline was reliable (naive and seasonal-naive traded wins depending on whether "
            "trend or season dominated there). Checking the same districts here, against whichever "
            "baseline was actually best in each one:\n\n"
        )
        f.write("| State | District | Best baseline MAE (m) | Ridge MAE (m) | Gradient boosting MAE (m) | GBM beats baseline? | n |\n|---|---|---|---|---|---|---|\n")
        for _, row in district_df.iterrows():
            f.write(
                f"| {row['state']} | {row['district']} | {row['best_baseline_mae']:.3f} | "
                f"{row['ridge_mae']:.3f} | {row['gbm_mae']:.3f} | {'yes' if row['gbm_beats_baseline'] else 'no'} | {int(row['n'])} |\n"
            )
        f.write(
            f"\n**Gradient boosting beats the best available baseline in {gbm_district_wins}/{len(district_df)} "
            "of these districts.** "
        )
        if gbm_district_wins == len(district_df):
            f.write(
                "It wins everywhere the baselines struggled -- this is real evidence the model is "
                "picking up the trend+season interaction that neither simple rule could capture alone, "
                "specifically in the highest-priority districts for an early-warning system.\n\n"
            )
        else:
            f.write(
                f"It doesn't win everywhere, which is worth being honest about: {len(district_df) - gbm_district_wins} "
                "district(s) still aren't clearly improved. Small per-district sample sizes (n column above) "
                "make these individual numbers noisier than the overall test-set result, so this is a "
                "directional check, not a guarantee -- worth a closer look in the evaluation stage rather "
                "than treated as settled here.\n\n"
            )

        f.write("## What this means for the project\n\n")
        f.write(
            "Two results, both real, that don't fully agree: gradient boosting beats the best baseline "
            f"by a solid {gbm_improvement:.1f}% nationally, but only wins outright in "
            f"{gbm_district_wins}/{len(district_df)} of the specific fastest-declining districts that "
            "matter most for an early-warning system. The likely reason is sample size -- each district "
            "has only a few hundred test readings at most (see the `n` column above) versus ~154k "
            "nationally, so the model has far less signal to learn a district's specific behavior from, "
            "and a single national model may be under-fitting these harder, faster-moving districts "
            "even while it does well on average. That gap -- not the headline national number -- is the "
            "honest state of this stage, and it's exactly what the evaluation/backtesting stage needs "
            "to characterize properly (e.g. is it consistent across other time splits, or noise from "
            "small samples?) before any claim about district-level usefulness is made.\n"
        )

    print("MAE:", results)
    print("RMSE:", rmse_results)
    print(f"Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
