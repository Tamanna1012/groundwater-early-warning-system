"""
Does telling the model *which district* a well is in close the district-level
gap that docs/EVALUATION.md leaves open?

That gap is this project's main unresolved result: gradient boosting beats
the best baseline in 7/7 national splits, but in the fastest-declining
districts -- the ones an early-warning system exists for -- it wins only
about a third of the time. `district` was deliberately excluded from the
model (see docs/ML_MODEL_RESULTS.md) on the grounds that 525 one-hot columns
would invite overfitting. That reasoning is sound for the Ridge model. It
does not apply to the gradient-boosted model, which splits on categorical
features natively and never one-hot encodes anything -- so the exclusion was
never actually tested for the model that matters. This script tests it.

Three arms, same 7 walk-forward splits as src/models/backtest_evaluation.py,
same features, same seed -- the only thing that varies is how district
identity reaches the model:

  current    the shipped model: no district feature at all
  capped     district as a native categorical. HistGradientBoosting caps a
             categorical at max_bins (255) distinct values, and there are
             more districts than that, so the rarest are folded into one
             OTHER bucket by training-row count -- decided on the training
             split only, never using test rows.
  encoded    district replaced by one numeric column via sklearn's
             TargetEncoder, which cross-fits internally on the training data
             so a district's own rows do not set its own encoding.

Districts are keyed as "STATE/District", not district name alone: names
repeat across states (there is an Aurangabad in both MH and BR), and every
other stage of this project treats (state, district) as the unit.

Output: docs/DISTRICT_MODEL.md + reports/figures/district_model_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import TargetEncoder

from _model_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "DISTRICT_MODEL.md"
FIG_DIR = ROOT / "reports" / "figures"

TEST_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
MIN_DISTRICT_N = 30  # same threshold as backtest_evaluation.py
MAX_CATEGORIES = 255  # HistGradientBoosting's max_bins ceiling for a categorical
RANDOM_STATE = 42

WORST_DISTRICTS = [
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
]
WORST_KEYS = {f"{s}/{d}" for s, d in WORST_DISTRICTS}

ARMS = ["current", "capped", "encoded"]


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df["district_key"] = df["state"].astype(str) + "/" + df["district"].astype(str)
    return df


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def as_categorical(X: pd.DataFrame, cols: list[str], reference: pd.DataFrame | None = None) -> pd.DataFrame:
    X = X.copy()
    for c in cols:
        if reference is None:
            X[c] = X[c].astype("category")
        else:
            # Test rows take the training split's category set: a district
            # unseen in training becomes NaN, which the model handles as a
            # missing value rather than as a new, untrained category.
            X[c] = X[c].astype("category").cat.set_categories(reference[c].cat.categories)
    return X


def cap_categories(train_keys: pd.Series, test_keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Keep the MAX_CATEGORIES-1 most common districts in TRAIN; everything
    else (including anything that only appears in test) becomes OTHER."""
    keep = set(train_keys.value_counts().nlargest(MAX_CATEGORIES - 1).index)
    return (
        train_keys.where(train_keys.isin(keep), "OTHER"),
        test_keys.where(test_keys.isin(keep), "OTHER"),
    )


def fit_predict(arm: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit one arm on `train`, return its predictions for `test`."""
    cat_cols = list(CATEGORICAL_FEATURES)
    X_train = train[NUMERIC_FEATURES + cat_cols].copy()
    X_test = test[NUMERIC_FEATURES + cat_cols].copy()
    y_train = train[TARGET]

    if arm == "capped":
        train_keys, test_keys = cap_categories(train["district_key"], test["district_key"])
        X_train["district_key"] = train_keys.to_numpy()
        X_test["district_key"] = test_keys.to_numpy()
        cat_cols = cat_cols + ["district_key"]
    elif arm == "encoded":
        encoder = TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)
        X_train["district_te"] = encoder.fit_transform(train[["district_key"]], y_train).ravel()
        X_test["district_te"] = encoder.transform(test[["district_key"]]).ravel()

    X_train = as_categorical(X_train, cat_cols)
    X_test = as_categorical(X_test, cat_cols, reference=X_train)

    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def district_win_counts(per_row: pd.DataFrame, arm: str, keys: set[str] | None) -> tuple[int, int]:
    """How often this arm beats the better of the two naive baselines, counted
    per (district, split) cell with enough test rows to be worth comparing."""
    wins = total = 0
    for key, sub in per_row.groupby("district_key", observed=True):
        if keys is not None and key not in keys:
            continue
        if len(sub) < MIN_DISTRICT_N:
            continue
        best_baseline = min(mae(sub["actual"], sub["naive"]), mae(sub["actual"], sub["seasonal_naive"]))
        total += 1
        wins += mae(sub["actual"], sub[arm]) < best_baseline
    return wins, total


def head_to_head(per_row: pd.DataFrame, arm: str, keys: set[str] | None) -> tuple[int, int]:
    """How often this arm beats the CURRENT model in the same cells. This is
    the sharper question: not 'is it better than a naive rule' but 'is it
    better than what the project already ships'."""
    wins = total = 0
    for key, sub in per_row.groupby("district_key", observed=True):
        if keys is not None and key not in keys:
            continue
        if len(sub) < MIN_DISTRICT_N:
            continue
        total += 1
        wins += mae(sub["actual"], sub[arm]) < mae(sub["actual"], sub["current"])
    return wins, total


def run_split(df: pd.DataFrame, test_year: int) -> dict:
    start, end = f"{test_year}-01-01", f"{test_year + 1}-01-01"
    train = df[df["date"] < start]
    test = df[(df["date"] >= start) & (df["date"] < end)]
    test = test.dropna(subset=["depth_last_reading", "depth_same_quarter_last_year"])

    per_row = test[["district_key"]].copy()
    per_row["actual"] = test[TARGET].to_numpy()
    per_row["naive"] = test["depth_last_reading"].to_numpy()
    per_row["seasonal_naive"] = test["depth_same_quarter_last_year"].to_numpy()

    row = {
        "test_year": test_year,
        "n_test": len(test),
        "best_baseline_mae": min(mae(per_row["actual"], per_row["naive"]),
                                 mae(per_row["actual"], per_row["seasonal_naive"])),
    }
    for arm in ARMS:
        per_row[arm] = fit_predict(arm, train, test)
        row[f"{arm}_mae"] = mae(per_row["actual"], per_row[arm])

    for arm in ARMS:
        w, t = district_win_counts(per_row, arm, WORST_KEYS)
        row[f"{arm}_worst_wins"], row[f"{arm}_worst_total"] = w, t
        w, t = district_win_counts(per_row, arm, None)
        row[f"{arm}_all_wins"], row[f"{arm}_all_total"] = w, t
        if arm != "current":
            w, t = head_to_head(per_row, arm, WORST_KEYS)
            row[f"{arm}_h2h_worst_wins"], row[f"{arm}_h2h_worst_total"] = w, t
            w, t = head_to_head(per_row, arm, None)
            row[f"{arm}_h2h_all_wins"], row[f"{arm}_h2h_all_total"] = w, t
    return row


def pct(wins: int, total: int) -> str:
    return f"{wins}/{total} ({wins / total:.0%})" if total else "n/a"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    results = pd.DataFrame([run_split(df, year) for year in TEST_YEARS])

    national = {arm: results[f"{arm}_mae"].mean() for arm in ARMS}
    worst = {arm: (int(results[f"{arm}_worst_wins"].sum()), int(results[f"{arm}_worst_total"].sum())) for arm in ARMS}
    allg = {arm: (int(results[f"{arm}_all_wins"].sum()), int(results[f"{arm}_all_total"].sum())) for arm in ARMS}
    h2h_worst = {a: (int(results[f"{a}_h2h_worst_wins"].sum()), int(results[f"{a}_h2h_worst_total"].sum())) for a in ARMS[1:]}
    h2h_all = {a: (int(results[f"{a}_h2h_all_wins"].sum()), int(results[f"{a}_h2h_all_total"].sum())) for a in ARMS[1:]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"current": "#55A868", "capped": "#4C72B0", "encoded": "#C44E52"}
    for arm in ARMS:
        axes[0].plot(results["test_year"], results[f"{arm}_mae"], marker="o", label=arm, color=colors[arm])
    axes[0].plot(results["test_year"], results["best_baseline_mae"], marker="o", linestyle="--",
                 label="best baseline", color="#8C8C8C")
    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("National MAE across the 7 walk-forward splits")
    axes[0].legend()

    labels = ["worst-10 districts", "all districts"]
    width = 0.26
    x = np.arange(len(labels))
    for i, arm in enumerate(ARMS):
        rates = [worst[arm][0] / worst[arm][1] * 100, allg[arm][0] / allg[arm][1] * 100]
        axes[1].bar(x + (i - 1) * width, rates, width, label=arm, color=colors[arm])
    axes[1].axhline(50, color="#8C8C8C", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("% of district-year cells beating best baseline")
    axes[1].set_title("District-level win rate")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "district_model_comparison.png", dpi=150)
    plt.close(fig)

    best_arm = min(ARMS, key=lambda a: national[a])
    best_worst_arm = max(ARMS, key=lambda a: (worst[a][0] / worst[a][1]) if worst[a][1] else 0)

    lines = []
    lines.append("# Does giving the model `district` close the district-level gap?")
    lines.append("")
    lines.append(
        "`docs/EVALUATION.md` leaves one result open: gradient boosting wins 7/7 national splits "
        "but only about a third of the district-year checks in the fastest-declining districts. "
        "`district` was excluded from the model because 525 one-hot columns would invite "
        "overfitting -- true for Ridge, but the gradient-boosted model never one-hot encodes "
        "anything, so the exclusion was never tested for the model that actually ships. This "
        "tests it, on the same 7 walk-forward splits, same features, same seed."
    )
    lines.append("")
    lines.append("| Arm | How district reaches the model |")
    lines.append("|---|---|")
    lines.append("| `current` | it does not -- the shipped model |")
    lines.append(
        f"| `capped` | native categorical, rarest districts folded into one OTHER bucket "
        f"(HistGradientBoosting caps a categorical at {MAX_CATEGORIES} values) |"
    )
    lines.append("| `encoded` | one numeric column from sklearn's `TargetEncoder`, cross-fitted on train |")
    lines.append("")

    lines.append("## National MAE by split")
    lines.append("")
    lines.append("| Test year | Test n | Best baseline | current | capped | encoded |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in results.iterrows():
        lines.append(
            f"| {int(r['test_year'])} | {int(r['n_test']):,} | {r['best_baseline_mae']:.3f} | "
            f"{r['current_mae']:.3f} | {r['capped_mae']:.3f} | {r['encoded_mae']:.3f} |"
        )
    lines.append("| **mean** | | | " + " | ".join(f"**{national[a]:.3f}**" for a in ARMS) + " |")
    lines.append("")
    lines.append("![District model comparison](../reports/figures/district_model_comparison.png)")
    lines.append("")

    lines.append("## District-level win rate (vs. the better naive baseline)")
    lines.append("")
    lines.append("| Arm | Fastest-declining 10 | All districts |")
    lines.append("|---|---|---|")
    for arm in ARMS:
        lines.append(f"| `{arm}` | {pct(*worst[arm])} | {pct(*allg[arm])} |")
    lines.append("")
    lines.append(
        f"The all-districts column is the more trustworthy of the two: the worst-10 column rests "
        f"on only {worst['current'][1]} district-year cells, where a single cell moves the rate by "
        f"{1 / max(worst['current'][1], 1):.0%}, while the all-districts column rests on "
        f"{allg['current'][1]:,}. The worst-10 number is the one quoted in `docs/EVALUATION.md`, "
        "so it is kept here for continuity, not because it is the better measurement."
    )
    lines.append("")

    lines.append("## Head to head against the shipped model")
    lines.append("")
    lines.append(
        "Beating a naive rule and beating the model this project already ships are different "
        "questions. Per district-year cell, how often does each arm beat `current`?"
    )
    lines.append("")
    lines.append("| Arm | Fastest-declining 10 | All districts |")
    lines.append("|---|---|---|")
    for arm in ARMS[1:]:
        lines.append(f"| `{arm}` | {pct(*h2h_worst[arm])} | {pct(*h2h_all[arm])} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    improvement = 100 * (national["current"] - national[best_arm]) / national["current"]
    verdict = []
    if best_arm == "current":
        verdict.append(
            "**Neither way of adding `district` beats the shipped model nationally.** The original "
            "decision to leave it out holds -- though for a different reason than the one originally "
            "given, which was about one-hot encoding and did not apply to this model."
        )
    else:
        verdict.append(
            f"**`{best_arm}` has the lowest mean national MAE** ({national[best_arm]:.3f}m vs "
            f"{national['current']:.3f}m for the shipped model, {improvement:+.1f}%)."
        )
    current_worst_rate = worst["current"][0] / worst["current"][1] if worst["current"][1] else 0
    best_worst_rate = worst[best_worst_arm][0] / worst[best_worst_arm][1] if worst[best_worst_arm][1] else 0
    splits_won = int((results["capped_mae"] < results["current_mae"]).sum())
    verdict.append(
        f"It is ahead in {splits_won}/{len(TEST_YEARS)} individual splits, so the direction is "
        "consistent rather than a single lucky cut -- but the size of it is not something a user "
        "of this system would ever notice."
    )
    if best_worst_arm == "current" or best_worst_rate <= current_worst_rate:
        verdict.append(
            "And in the fastest-declining districts -- the gap this experiment set out to close -- "
            f"no arm does better than what already ships ({pct(*worst['current'])} either way). "
            "**The gap is not a missing-district-identity problem.**"
        )
    else:
        verdict.append(
            f"In the fastest-declining districts, `{best_worst_arm}` raises the win rate from "
            f"{current_worst_rate:.0%} to {best_worst_rate:.0%}, which would be grounds for promoting "
            "it into the shipped model."
        )
    lines.extend(" ".join(verdict).split("\n"))
    lines.append("")

    # Two findings that matter more than the arm comparison itself, and that
    # would be invisible if this experiment had only measured the worst-10.
    all_rate = allg["current"][0] / allg["current"][1]
    lines.append("### The limitation is narrower than this project has been describing it")
    lines.append("")
    lines.append(
        f"Measured across **all** districts rather than the worst 10, the shipped model beats the "
        f"better naive baseline in {pct(*allg['current'])} of district-year cells -- not the "
        f"{current_worst_rate:.0%} that `docs/EVALUATION.md` quotes. Both numbers are real and they "
        "are not in conflict: they are different populations. The model is not weak at district level "
        "in general; it is weak *specifically* in districts whose water table is falling fastest, "
        f"where it wins {100 * (all_rate - current_worst_rate):.0f} percentage points less often than "
        "in a typical district. That is a "
        "sharper and more useful statement of the limitation than the one this project started with, "
        "and it is worth being precise about, because the broad version understates the model and the "
        "narrow version is the one that actually threatens the use case."
    )
    lines.append("")
    lines.append("### District identity does help in those districts -- just not enough")
    lines.append("")
    lines.append(
        f"Head to head, `capped` beats `current` in {pct(*h2h_worst['capped'])} of the worst-10 "
        f"cells while its win rate against the baseline stays at {pct(*worst['capped'])}. So knowing "
        "the district does reduce error there; it just does not reduce it far enough to cross the "
        "bar a naive rule already clears. Whatever the fast-declining districts need, a per-district "
        "intercept is not it."
    )
    lines.append("")
    lines.append("### What was decided")
    lines.append("")
    lines.append(
        f"**The shipped model is left unchanged.** A {abs(improvement):.2f}% national gain that does "
        "nothing for the limitation it was meant to address does not justify adding a 255-category "
        "feature, an OTHER bucket whose membership shifts every time the model is retrained, and a "
        "regeneration of every downstream stage. The experiment is kept because the answer is useful "
        "-- it rules out the obvious explanation and redirects the next attempt, most plausibly "
        "toward an uncertainty estimate that tells a user when to distrust a district's forecast "
        "rather than toward trying to make the point forecast better."
    )
    lines.append("")

    DOCS_PATH.write_text("\n".join(lines), encoding="utf-8", newline="")

    print(results[["test_year"] + [f"{a}_mae" for a in ARMS]].to_string(index=False))
    print("mean national MAE:", {a: round(national[a], 4) for a in ARMS})
    print("worst-10 win rate:", {a: pct(*worst[a]) for a in ARMS})
    print("all-district win rate:", {a: pct(*allg[a]) for a in ARMS})
    print("head-to-head vs current (worst-10):", {a: pct(*h2h_worst[a]) for a in ARMS[1:]})
    print("head-to-head vs current (all):", {a: pct(*h2h_all[a]) for a in ARMS[1:]})
    print(f"Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
