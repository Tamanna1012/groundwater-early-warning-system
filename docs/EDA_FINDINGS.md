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

## Site-type effects

CGWB well readings come from physically different well types. Categories with fewer than 500 readings are excluded below as too small to say anything reliable about.

| Site type | # wells | # readings | Mean depth (m) | Median depth (m) | Std dev (m) | Median seasonal swing (m) |
|---|---|---|---|---|---|---|
| Unlabeled | 581 | 17,904 | 5.71 | 4.18 | 7.72 | 3.78 |
| Dug Well | 19,671 | 771,125 | 6.53 | 4.92 | 6.81 | 3.43 |
| Bore Well | 3,975 | 97,297 | 10.71 | 8.12 | 9.52 | 5.07 |
| Tube Well | 3,799 | 75,172 | 19.22 | 11.35 | 22.09 | 3.04 |
| Dug Cum Bore Well | 44 | 1,529 | 14.58 | 12.62 | 9.68 | 4.70 |

![Site type comparison](../reports/figures/site_type_comparison.png)

**Findings:** dug wells (the large majority of the network, ~77% of readings) are shallow and relatively low-variance -- they track the near-surface water table closely. Bore and tube wells are markedly deeper, far noisier (tube wells have roughly 3x the standard deviation of dug wells), and swing more with the season, consistent with them tapping deeper, more heterogeneous aquifer layers rather than the shallow water table.

**Modelling implication:** site type is a real confound, not just metadata -- it should be included as a feature (or used to fit separate models per type) rather than pooling all wells together, since a 'depth of 15m' means something different for a dug well (unusually deep, possibly concerning) than for a tube well (unremarkable). The high variance in tube wells also means their readings need more caution before being used to flag anomalies -- a single-well spike is far more likely to be normal noise there.
