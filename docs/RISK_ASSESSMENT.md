# Groundwater Risk Assessment: District Priority Ranking

**This is a methodology demonstration, not a live assessment.** The underlying CGWB data (`docs/DATA_SOURCES.md`) runs through Jan 2017; this output forecasts one quarter past each well's own last reading, which for most wells lands in the mid-2010s. It shows the pipeline produces a real, usable ranked output from real data -- re-running it against current CGWB/India-WRIS data (`src/data/download_cgwb_data.py`) would make it current.

## Methodology

District risk score = 40% current severity + 40% measured historical trend + 20% model-forecasted next-quarter change (each as a 0-100 percentile rank across 499 districts with at least 5 monitored wells). The weighting is deliberate, not arbitrary: `docs/EXPLAINABILITY.md` found the model is systematically too optimistic and barely uses the trend feature on its own, so this score leans on two directly-measured, model-independent signals (severity, trend) and treats the model's forecast as a smaller, confirming input -- not the primary driver.

**10 districts (marked \*) are ones where `docs/EVALUATION.md`/`docs/EXPLAINABILITY.md` specifically found the model under-predicts severity** -- 8 of them appear in the top 20 below. For these, treat the risk score as a floor, not a ceiling: the real situation is more likely to be *worse* than scored than better.

## Top 20 priority districts

![District risk ranking](../reports/figures/district_risk_ranking.png)

| Rank | State | District | Wells | Current depth (m) | Trend (m/yr) | Risk score | Flag |
|---|---|---|---|---|---|---|---|
| 1 | MP | Morena | 23 | 15.65 | +0.647 | 94.3 |  |
| 2 | GJ | Banas Kantha | 120 | 22.05 | +0.307 | 92.8 | \* |
| 3 | RJ | Jaipur | 118 | 33.65 | +0.920 | 92.3 | \* |
| 4 | HR | Karnal | 125 | 16.75 | +0.437 | 90.6 |  |
| 5 | RJ | Jhunjhunun | 40 | 53.15 | +0.713 | 90.5 | \* |
| 6 | MP | Bhind | 29 | 17.40 | +0.146 | 90.5 |  |
| 7 | RJ | Jodhpur | 100 | 28.15 | +0.143 | 89.8 |  |
| 8 | RJ | Sikar | 70 | 43.23 | +0.790 | 88.6 | \* |
| 9 | MP | Gwalior | 32 | 11.65 | +0.310 | 88.3 |  |
| 10 | RJ | Dausa | 45 | 23.26 | +0.935 | 88.1 | \* |
| 11 | WB | Purba Medinipur | 83 | 11.37 | +0.381 | 88.0 |  |
| 12 | GJ | Gandhinagar | 35 | 89.76 | +0.193 | 87.4 |  |
| 13 | TG | Medak | 109 | 11.31 | +0.115 | 87.1 |  |
| 14 | RJ | Alwar | 77 | 21.20 | +0.742 | 86.5 | \* |
| 15 | MP | Ratlam | 67 | 15.30 | +0.105 | 85.7 |  |
| 16 | HR | Mahendragarh | 37 | 26.02 | +0.696 | 84.3 |  |
| 17 | HR | Gurgaon | 49 | 21.75 | +0.631 | 84.1 | \* |
| 18 | PB | Patiala | 67 | 19.13 | +0.473 | 83.5 |  |
| 19 | PB | Sangrur | 48 | 27.24 | +0.853 | 83.2 | \* |
| 20 | TG | Nizamabad | 96 | 7.97 | +0.206 | 83.2 |  |

Full ranking for all 499 qualifying districts: `data/processed/district_risk_scores.csv` (regenerate with `python src/output/risk_score.py`).

## Geographic view

![Risk map](../reports/figures/risk_map.png)

Each point is one district (positioned at its wells' average location), sized by how many wells monitor it and colored by risk score. This is the actionable geographic output the project set out to build -- a groundwater board could use this directly to decide where to send a field team first, without needing to read any of the underlying model documentation.

## What this decision-support output can and can't claim

It can support: *prioritizing which districts to examine or intervene in first*, using both current severity and measured trend, cross-checked against a model forecast. It cannot replace: CGWB's official extraction/recharge-based classification (this ranking uses only well-level depth data, not recharge estimates); a real-time assessment (data currency caveat above); or a substitute for field verification, especially in the flagged districts where the model itself is known to run optimistic. This matches the scope set out in `docs/PROBLEM_STATEMENT.md`: a decision-support prototype, not a production regulatory tool.
