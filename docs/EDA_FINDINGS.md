# EDA Findings

Deeper exploratory analysis beyond the initial data audit. Each section is a real, computed finding with a stated implication for modelling -- not decoration.

## Seasonality

Mean depth-to-water by measurement quarter, pooled across all years and wells:

| Quarter | Mean depth (m) |
|---|---|
| Jan (dry season) | 8.00 |
| May (pre-monsoon, driest) | 9.68 |
| Aug (monsoon) | 6.87 |
| Nov (post-monsoon) | 7.18 |

The water table is on average **2.81 m shallower in August than in May** -- i.e. the monsoon visibly recharges groundwater nationally, and it draws back down through the dry season until the next monsoon. This is the expected physical pattern, which is itself a sanity check that the data is measuring something real.

![Seasonal pattern](../reports/figures/seasonal_pattern.png)

**Per-well seasonal swing:** using only well-years with all 4 quarters present (158,187 well-years), the typical (median) swing between a well's deepest and shallowest reading in the same year is **3.52 m** (90th percentile: 8.41 m).

**Modelling implication:** the seasonal swing is comparable in size to, or larger than, the multi-year depletion trend found in `DATA_AUDIT.md` (roughly 1 m/year in the fastest-declining districts). A model that ignores season would likely mistake normal pre-monsoon dips for depletion, or mask real depletion behind a good monsoon year. Any forecasting model built later must condition on quarter/season, and any trend estimate should compare same-quarter readings across years rather than raw consecutive readings.
