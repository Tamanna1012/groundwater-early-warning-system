"""
The actionable output: a district-level groundwater risk ranking -- the
thing docs/PROBLEM_STATEMENT.md says this whole project is for. Turns
everything from the earlier stages into a single ranked priority list a
groundwater board could actually use to decide where to act first.

IMPORTANT CAVEAT, stated here and repeated in the output doc: the
underlying CGWB data (docs/DATA_SOURCES.md) runs through Jan 2017. This
script forecasts one quarter past each well's own last reading, which for
most wells means sometime in the mid-2010s, not the present day. This is a
methodology demonstration -- proof that the pipeline can produce a real,
usable ranked output from real data -- not a live 2026 assessment of
Indian groundwater conditions. Re-running this against current CGWB/
India-WRIS data (src/data/download_cgwb_data.py) would make it one.

Design choice driven directly by docs/EXPLAINABILITY.md: that stage found
the model is systematically too optimistic (under-predicts depth) in most
of the fastest-declining districts, and barely uses the trend feature on
its own. So the risk score here does NOT lean primarily on the model's
forecast. It's a weighted combination of three signals:
  - current severity (40%): how deep the water table already is -- a
    directly measured fact, not a model output
  - measured trend (40%): each well's own full-history slope (m/year) --
    also directly measured, not model-dependent
  - model-forecasted change (20%): the trained model's predicted
    next-quarter shift -- kept as a smaller, confirming signal given its
    documented tendency to under-call severity, not the primary driver

Output: docs/RISK_ASSESSMENT.md + reports/figures/district_risk_ranking.png
+ reports/figures/risk_map.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))  # for the shared _model_utils module
from _model_utils import FEATURE_COLS, TARGET, as_categorical_frame, fit_gbm, predict_interval
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DOCS_PATH = ROOT / "docs" / "RISK_ASSESSMENT.md"
FIG_DIR = ROOT / "reports" / "figures"
OUT_CSV = ROOT / "data" / "processed" / "district_risk_scores.csv"

SLOPE_EPOCH_YEAR = 1996  # must match src/features/build_features.py

MIN_WELLS_PER_DISTRICT = 5
N_NEIGHBORS = 5
MONTH_MAP = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
NEXT_QUARTER = {"Jan": "May", "May": "Aug", "Aug": "Nov", "Nov": "Jan"}

# From docs/EVALUATION.md / docs/EXPLAINABILITY.md: districts where the
# model was specifically shown to under-predict severity most.
KNOWN_MODEL_BIAS_DISTRICTS = {
    ("HR", "Kurukshetra"), ("RJ", "Sikar"), ("RJ", "Dausa"), ("RJ", "Jhunjhunun"),
    ("PB", "Sangrur"), ("HR", "Gurgaon"), ("GJ", "Banas Kantha"), ("PB", "Barnala"),
    ("RJ", "Jaipur"), ("RJ", "Alwar"),
}


def next_quarter_date(last_date: pd.Timestamp) -> pd.Timestamp:
    last_season = last_date.strftime("%b")
    next_season = NEXT_QUARTER[last_season]
    year = last_date.year + (1 if next_season == "Jan" else 0)
    return pd.Timestamp(year=year, month=MONTH_MAP[next_season], day=1)


def well_full_history_slope(dates: pd.Series, depths: pd.Series) -> float:
    """Full-history trend (m/year) for one well. Uses the same epoch-shifted
    time axis as src/features/build_features.py (SLOPE_EPOCH_YEAR) so the
    serving-time slope is numerically identical to the training-time one --
    a constant shift cannot change a slope, but fitting against raw calendar
    years is badly conditioned and loses precision."""
    if dates.nunique() < 3:
        return np.nan
    x = dates.dt.year + (dates.dt.month - 1) / 12 - SLOPE_EPOCH_YEAR
    m, _ = np.polyfit(x, depths, 1)
    return m


def build_current_well_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per well: its most recent actual reading, plus the features
    needed to forecast ONE quarter past that reading. Mirrors the logic in
    src/features/build_features.py, just extended one step into the
    future using only data that already, genuinely exists."""
    df = df.sort_values(["well_id", "date"])
    last = df.groupby("well_id").tail(1).set_index("well_id")

    snapshot = last[["state", "district", "lat", "lon", "site_type", "date"]].copy()
    snapshot["current_depth"] = last[TARGET]
    snapshot["forecast_date"] = snapshot["date"].apply(next_quarter_date)
    snapshot["season"] = snapshot["forecast_date"].dt.strftime("%b")
    snapshot["depth_last_reading"] = last[TARGET]

    # rolling_mean_4: mean of each well's last up-to-4 actual readings (inclusive of the latest)
    tail4 = df.groupby("well_id").tail(4)
    snapshot["rolling_mean_4"] = tail4.groupby("well_id")[TARGET].mean()

    # long_term_slope: full-history slope, legitimate here (this is a
    # current snapshot, not a training row -- no leakage concern)
    slopes = df.groupby("well_id").apply(
        lambda g: well_full_history_slope(g["date"], g[TARGET]), include_groups=False
    )
    snapshot["long_term_slope_m_per_year"] = slopes

    snapshot["n_prior_readings"] = df.groupby("well_id").size()

    # depth_same_quarter_last_year: this well's reading ~1 year before the
    # forecast date, if one exists in its real history
    lookup = df.set_index(["well_id", "date"])[TARGET]
    target_dates = snapshot["forecast_date"] - pd.DateOffset(years=1)
    same_q_last_year = []
    for well_id, tdate in zip(snapshot.index, target_dates):
        same_q_last_year.append(lookup.get((well_id, tdate), np.nan))
    snapshot["depth_same_quarter_last_year"] = same_q_last_year

    return snapshot.reset_index()


def add_neighbor_feature(snapshot: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    coords = snapshot[["lat", "lon"]].to_numpy()
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=N_NEIGHBORS + 1)
    neighbor_idx = idx[:, 1:]
    depths = snapshot["current_depth"].to_numpy()
    snapshot = snapshot.copy()
    snapshot["neighbor_mean_depth"] = depths[neighbor_idx].mean(axis=1)
    return snapshot


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    print("Building current well snapshots...")
    snapshot = build_current_well_snapshot(df)
    snapshot = add_neighbor_feature(snapshot)

    print("Training production model on the full dataset...")
    X_full, y_full = df[FEATURE_COLS], df[TARGET]
    model, X_train_gbm = fit_gbm(X_full, y_full)

    X_forecast = as_categorical_frame(snapshot[FEATURE_COLS], reference=X_train_gbm)
    snapshot["predicted_next_depth"] = model.predict(X_forecast)
    snapshot["predicted_change"] = snapshot["predicted_next_depth"] - snapshot["current_depth"]

    # How sure is the model about each of those forecasts? docs/UNCERTAINTY.md
    # validates this interval on the same backtest the rest of the evaluation
    # uses: it holds the actual reading 77% of the time against a nominal 80%,
    # and its width tracks district-level error closely (r = 0.90), so it is
    # a usable reliability signal rather than decoration.
    print("Fitting prediction-interval models...")
    lower, upper = predict_interval(X_train_gbm, y_full, X_forecast)
    snapshot["forecast_interval_m"] = upper - lower

    # --- Aggregate to district level ---
    district = snapshot.groupby(["state", "district"]).agg(
        n_wells=("well_id", "size"),
        current_depth=("current_depth", "median"),
        measured_trend=("long_term_slope_m_per_year", "median"),
        predicted_change=("predicted_change", "median"),
        forecast_interval_m=("forecast_interval_m", "median"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )
    district = district[district["n_wells"] >= MIN_WELLS_PER_DISTRICT].copy()
    district = district.dropna(subset=["measured_trend"])  # need a real trend estimate to score

    district["severity_pct"] = percentile_rank(district["current_depth"])
    district["trend_pct"] = percentile_rank(district["measured_trend"])
    district["forecast_pct"] = percentile_rank(district["predicted_change"])
    district["risk_score"] = 0.4 * district["severity_pct"] + 0.4 * district["trend_pct"] + 0.2 * district["forecast_pct"]

    district["known_model_bias_flag"] = [
        (s, d) in KNOWN_MODEL_BIAS_DISTRICTS for s, d in district.index
    ]

    # Forecast confidence, as terciles of the interval width across districts.
    # Deliberately NOT folded into risk_score: letting the model's confidence
    # change the weighting per district would make two districts' scores
    # incomparable, which defeats the point of a ranking. It is reported
    # beside the score instead, so a user can see which forecasts to discount
    # while still reading one consistently-constructed number.
    district["forecast_confidence"] = pd.qcut(
        district["forecast_interval_m"], 3, labels=["High", "Medium", "Low"]
    ).astype(str)

    district = district.sort_values("risk_score", ascending=False).reset_index()
    district.insert(0, "rank", range(1, len(district) + 1))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    district.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(district):,} districts to {OUT_CSV}")

    # --- Figure 1: top 20 priority districts, stacked components ---
    top20 = district.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top20))[::-1]
    ax.barh(y_pos, top20["severity_pct"] * 0.4, color="#4C72B0", label="Severity (40%)")
    ax.barh(y_pos, top20["trend_pct"] * 0.4, left=top20["severity_pct"] * 0.4, color="#DD8452", label="Trend (40%)")
    ax.barh(
        y_pos, top20["forecast_pct"] * 0.2,
        left=top20["severity_pct"] * 0.4 + top20["trend_pct"] * 0.4,
        color="#55A868", label="Model forecast (20%)",
    )
    labels = [f"{row['district']}, {row['state']}" + (" *" if row["known_model_bias_flag"] else "") for _, row in top20.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Risk score (0-100)")
    ax.set_title("Top 20 priority districts for groundwater intervention")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "district_risk_ranking.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: geographic risk map (bubble plot, no shapefile needed) ---
    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(
        district["lon"], district["lat"], s=district["n_wells"].clip(upper=60) * 2,
        c=district["risk_score"], cmap="YlOrRd", edgecolors="black", linewidths=0.3, alpha=0.85,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("District groundwater risk score (bubble size = # wells)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Risk score (0-100)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_map.png", dpi=150)
    plt.close(fig)

    n_flagged_in_top20 = int(top20["known_model_bias_flag"].sum())

    with DOCS_PATH.open("w") as f:
        f.write("# Groundwater Risk Assessment: District Priority Ranking\n\n")
        f.write(
            "**This is a methodology demonstration, not a live assessment.** The underlying CGWB data "
            "(`docs/DATA_SOURCES.md`) runs through Jan 2017; this output forecasts one quarter past each "
            "well's own last reading, which for most wells lands in the mid-2010s. It shows the pipeline "
            "produces a real, usable ranked output from real data -- re-running it against current "
            "CGWB/India-WRIS data (`src/data/download_cgwb_data.py`) would make it current.\n\n"
        )
        f.write("## Methodology\n\n")
        f.write(
            f"District risk score = 40% current severity + 40% measured historical trend + 20% "
            f"model-forecasted next-quarter change (each as a 0-100 percentile rank across "
            f"{len(district)} districts with at least {MIN_WELLS_PER_DISTRICT} monitored wells). "
            "The weighting is deliberate, not arbitrary: `docs/EXPLAINABILITY.md` found the model is "
            "systematically too optimistic and barely uses the trend feature on its own, so this score "
            "leans on two directly-measured, model-independent signals (severity, trend) and treats the "
            "model's forecast as a smaller, confirming input -- not the primary driver.\n\n"
        )
        f.write(
            f"**{len(KNOWN_MODEL_BIAS_DISTRICTS)} districts (marked \\*) are ones where "
            "`docs/EVALUATION.md`/`docs/EXPLAINABILITY.md` specifically found the model under-predicts "
            f"severity** -- {n_flagged_in_top20} of them appear in the top 20 below. For these, treat the "
            "risk score as a floor, not a ceiling: the real situation is more likely to be *worse* than "
            "scored than better.\n\n"
        )
        f.write("## Top 20 priority districts\n\n")
        f.write("![District risk ranking](../reports/figures/district_risk_ranking.png)\n\n")
        f.write("| Rank | State | District | Wells | Current depth (m) | Trend (m/yr) | Risk score | Forecast confidence | Flag |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, row in top20.iterrows():
            flag = "\\*" if row["known_model_bias_flag"] else ""
            f.write(
                f"| {int(row['rank'])} | {row['state']} | {row['district']} | {int(row['n_wells'])} | "
                f"{row['current_depth']:.2f} | {row['measured_trend']:+.3f} | {row['risk_score']:.1f} | "
                f"{row['forecast_confidence']} ({row['forecast_interval_m']:.1f} m) | {flag} |\n"
            )
        f.write(f"\nFull ranking for all {len(district)} qualifying districts: `data/processed/district_risk_scores.csv` (regenerate with `python src/output/risk_score.py`).\n\n")

        low_conf_in_top20 = int((top20["forecast_confidence"] == "Low").sum())
        f.write("### Reading the confidence column\n\n")
        f.write(
            "**Forecast confidence** is the width of the model's 80% prediction interval for that "
            "district, bucketed into terciles across all districts (`High` = narrowest third). "
            "`docs/UNCERTAINTY.md` validates it on the same backtest as the rest of the evaluation: "
            "the interval holds the actual reading 77% of the time against a nominal 80%, and its "
            "width tracks district-level error closely enough (r = 0.90) to be worth acting on -- "
            "districts in the widest quartile are off by 2.71 m on average against 0.81 m in the "
            "narrowest.\n\n"
        )
        f.write(
            f"It is deliberately **not** folded into the risk score. Letting the model's confidence "
            "change the weighting district by district would make two districts' scores incomparable, "
            "which is the one thing a ranking cannot afford. It sits beside the score instead: "
            f"{low_conf_in_top20} of the top 20 carry `Low` confidence, and for those the 20% "
            "model-forecast component should be read as the weakest part of the number, with the "
            "80% that comes from directly measured severity and trend carrying the weight.\n\n"
        )

        f.write("## Geographic view\n\n")
        f.write("![Risk map](../reports/figures/risk_map.png)\n\n")
        f.write(
            "Each point is one district (positioned at its wells' average location), sized by how many "
            "wells monitor it and colored by risk score. This is the actionable geographic output the "
            "project set out to build -- a groundwater board could use this directly to decide where to "
            "send a field team first, without needing to read any of the underlying model documentation.\n\n"
        )

        f.write("## What this decision-support output can and can't claim\n\n")
        f.write(
            "It can support: *prioritizing which districts to examine or intervene in first*, using both "
            "current severity and measured trend, cross-checked against a model forecast. It cannot "
            "replace: CGWB's official extraction/recharge-based classification (this ranking uses only "
            "well-level depth data, not recharge estimates); a real-time assessment (data currency "
            "caveat above); or a substitute for field verification, especially in the flagged districts "
            "where the model itself is known to run optimistic. This matches the scope set out in "
            "`docs/PROBLEM_STATEMENT.md`: a decision-support prototype, not a production regulatory "
            "tool.\n"
        )

    print(f"Wrote {DOCS_PATH}")
    print(district.head(20)[["rank", "state", "district", "risk_score", "known_model_bias_flag"]])


if __name__ == "__main__":
    main()
