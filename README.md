# Groundwater Depletion Early-Warning System

A data-driven early-warning system for groundwater depletion in India: forecasting
where the water table is falling fastest, using real monitoring-well data from the
Central Ground Water Board (CGWB), to support district-level decisions about where
to restrict extraction or invest in recharge.

## The problem

Large parts of north-western India are extracting groundwater faster than it is
being recharged. CGWB's own periodic dynamic groundwater resource assessments
classify hundreds of administrative blocks as "over-exploited" or "critical," a
crisis that is well documented in peer-reviewed hydrology (e.g. the
Indo-Gangetic basin groundwater-decline literature) and in national policy
discussion (NITI Aayog's composite water management index has repeatedly flagged
multiple states as facing severe water stress). Farmers dependent on borewells and
municipal water utilities are the ones who feel this first; state groundwater
boards are the ones responsible for deciding where to act, and today that
decision leans heavily on periodic manual assessments rather than a continuously
updated, forward-looking view of which specific blocks are deteriorating fastest.

**Stakeholders:** state/central groundwater boards and municipal water utilities
(deciding where to restrict extraction or fund recharge structures), and farmers
choosing whether a borewell investment in their area is viable.

**Decision this project supports:** *which districts should be prioritized for
groundwater intervention, based on measured decline trends and near-term
forecasts* — not a retrospective report, a forward-looking one.

See [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) for the full
write-up (who has this problem, what's currently done about it, its
limitations, and what this project adds).

## The data

Real CGWB monitoring-well data: **28,072 wells, 24 states, 525 districts,
963,072 quarterly depth-to-water readings from 1996–2017.** Full provenance,
credibility, known quality issues, and the official (non-mirror) data-collection
path are documented in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — read
that before trusting any number derived from it.

A first-pass audit of the cleaned data — well coverage, a national depth
trend, and the districts with the steepest declining trend — is in
[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md). It already surfaces a real,
useful signal: the steepest-declining districts in this data (Sikar, Dausa,
Jhunjhunun, Jaipur, Alwar in Rajasthan; Sangrur, Barnala in Punjab;
Kurukshetra, Gurgaon in Haryana; Banas Kantha in Gujarat) line up with the
belt of districts independently flagged as over-exploited/critical by CGWB
and NITI Aayog — a sanity check that the data captures the real crisis this
project is meant to help with, not an artifact of the cleaning process.

## Project status

This repo is being built incrementally, following the pipeline below. Nothing
past data collection/cleaning/first-pass audit is implemented yet — no
forecasting model exists, so no accuracy/error numbers should be trusted
until this section says otherwise.

- [x] Real problem definition, evidence, stakeholders — `docs/PROBLEM_STATEMENT.md`
- [x] Real data sourced, documented, and quality-audited — `docs/DATA_SOURCES.md`, `docs/DATA_AUDIT.md`
- [x] Data cleaning pipeline (`src/data/clean_cgwb_data.py`) — wide-to-long, type coercion, implausible-value filtering
- [x] First-pass EDA / sufficiency audit (`src/eda/data_audit.py`)
- [x] Deeper EDA & statistical analysis (`docs/EDA_FINDINGS.md`: seasonality, site-type effects, spatial autocorrelation)
- [x] Feature engineering (`src/features/build_features.py`, `docs/FEATURES.md`) -- note: `long_term_slope_m_per_year` has a known look-ahead-bias caveat to fix before training (see docs/FEATURES.md)
- [x] Baseline forecast (`src/models/baseline_forecast.py`, `docs/BASELINE_RESULTS.md`) -- seasonal-naive beats naive by 23.8% MAE nationally, but naive wins in 8/10 of the fastest-declining districts (trend dominates over season there); this is the bar (MAE 1.809m) any real model must clear
- [x] ML/time-series model(s) and comparison against baseline (`src/models/ml_forecast.py`, `docs/ML_MODEL_RESULTS.md`) -- gradient boosting beats the best baseline by 11.6% MAE nationally (1.600m vs 1.809m), but only wins outright in 4/10 of the fastest-declining districts, likely due to small per-district sample sizes; this national-vs-district gap is unresolved and flagged for the evaluation stage
- [x] Evaluation (`src/models/backtest_evaluation.py`, `docs/EVALUATION.md`) -- 7 independent walk-forward splits (2010-2016): gradient boosting beats the best baseline in **7/7** splits (mean +10.9%), confirming the national result is real, not a one-split fluke. But the district-level gap is also confirmed, not resolved: only a **38% win rate** (19/50) in the fastest-declining districts across all splits -- a genuine, documented limitation to carry into explainability/actionable-output, not solved yet
- [x] Explainability (`src/models/explainability.py`, `docs/EXPLAINABILITY.md`) -- caught SHAP TreeExplainer AND sklearn's default partial-dependence method both giving wrong results for this model (native-categorical-split incompatibility), switched to permutation importance + brute-force partial dependence instead; found the model is systematically too optimistic (under-predicts depth) in 7/10 fastest-declining districts, and traced it to a specific cause: `long_term_slope_m_per_year` has almost no marginal effect on predictions, so wells whose only warning sign is a steep trend (not yet reflected in recent absolute readings) are the cases it misses worst
- [x] Actionable output: district-level risk map + forecast (`src/output/risk_score.py`, `docs/RISK_ASSESSMENT.md`) -- ranks 499 districts by a deliberately-weighted risk score (40% measured severity + 40% measured trend + 20% model forecast, weighted this way *because* of the explainability findings above), with a top-20 priority chart and a geographic bubble map whose hotspot visibly matches the known NW-India over-exploited belt; 8/10 previously-flagged model-bias districts land in the top 20, a real consistency check
- [ ] Dashboard
- [ ] Deployment

## Repo layout

```
data/
  raw/cgwb_mirror/        real CGWB data, bootstrapped from a GitHub mirror (see docs/DATA_SOURCES.md)
  processed/               generated by src/data/clean_cgwb_data.py (gitignored — regenerate, don't hand-edit)
src/
  data/                    collection + cleaning scripts
  eda/                     analysis scripts
  features/                feature engineering
  models/                  baseline, ML model, backtest evaluation, explainability
  output/                  actionable output: district risk scoring
docs/
  PROBLEM_STATEMENT.md      the real-world problem, evidence, stakeholders, gap in existing work
  DATA_SOURCES.md           exact data provenance, credibility, quality issues
  DATA_AUDIT.md             generated: coverage stats, first trend check
  EDA_FINDINGS.md           generated: seasonality, site-type effects, spatial autocorrelation
  FEATURES.md               generated: feature table reference, coverage, look-ahead-bias caveat
  BASELINE_RESULTS.md       generated: naive vs seasonal-naive backtest, overall/season/district
  ML_MODEL_RESULTS.md       generated: Ridge + gradient boosting vs baselines, feature importance, district check
  EVALUATION.md             generated: 7-split walk-forward backtest, national + district-level consistency check
  EXPLAINABILITY.md         generated: permutation importance, partial dependence, district bias diagnostic, case study
  RISK_ASSESSMENT.md        generated: district risk ranking, priority chart, geographic risk map
reports/figures/            generated plots referenced from docs/
```

## Reproducing this

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/data/clean_cgwb_data.py       # writes data/processed/groundwater_levels_long.csv
python src/eda/data_audit.py             # writes docs/DATA_AUDIT.md + reports/figures/*.png
python src/eda/seasonality_analysis.py   # writes docs/EDA_FINDINGS.md (seasonality section)
python src/eda/site_type_analysis.py     # writes docs/EDA_FINDINGS.md (site-type section)
python src/eda/spatial_analysis.py       # writes docs/EDA_FINDINGS.md (spatial section)
python src/features/build_features.py    # writes data/processed/features.csv + docs/FEATURES.md
python src/models/baseline_forecast.py   # writes docs/BASELINE_RESULTS.md + reports/figures/baseline_comparison.png
python src/models/ml_forecast.py         # writes docs/ML_MODEL_RESULTS.md + reports/figures/model_comparison.png
python src/models/backtest_evaluation.py # writes docs/EVALUATION.md + reports/figures/backtest_mae_by_year.png
python src/models/explainability.py      # writes docs/EXPLAINABILITY.md + reports/figures/partial_dependence.png
python src/output/risk_score.py          # writes docs/RISK_ASSESSMENT.md + reports/figures/district_risk_ranking.png + risk_map.png
```
