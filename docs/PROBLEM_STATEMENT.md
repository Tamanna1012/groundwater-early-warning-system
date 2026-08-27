# Problem Statement

## 1. Who currently faces this problem?

- **Farmers** in groundwater-dependent agricultural belts (Punjab, Haryana,
  Rajasthan, parts of Gujarat and Tamil Nadu) whose borewells are going
  deeper, and more expensive to run, year over year.
- **State groundwater boards and municipal water utilities**, who are
  statutorily responsible for regulating extraction (e.g. under state
  groundwater acts) and for planning recharge infrastructure, but who work
  from periodic manual assessments rather than a continuously updated,
  forward-looking view.
- **Urban water utilities** in cities whose supply partially depends on
  groundwater, where depletion translates into rising extraction costs and,
  eventually, supply risk.

## 2. Where does this problem occur?

Most acutely in north-western India — Punjab, Haryana, Rajasthan, parts of
Uttar Pradesh and Gujarat — but CGWB's dynamic groundwater resource
assessments classify "over-exploited" and "critical" blocks across many
states, not only that belt. This project's own first-pass trend analysis
(`docs/DATA_AUDIT.md`) independently reproduces that geography from raw well
data: the steepest-declining districts in the cleaned dataset are
concentrated in exactly this belt (Rajasthan, Punjab, Haryana, Gujarat).

## 3. Why is it important?

Groundwater supplies a large share of India's irrigation and a substantial
share of drinking water in both rural and urban areas. Once an aquifer is
over-exploited, recovery is slow (years to decades) even after extraction is
reduced — so *early* identification of accelerating decline, before a block
crosses from "safe" to "critical," is far more actionable than identifying it
after the fact.

## 4. What is currently being done about it?

- CGWB and state groundwater boards conduct periodic (typically multi-year)
  dynamic groundwater resource assessments that classify each administrative
  block as Safe / Semi-critical / Critical / Over-exploited, based on the
  ratio of extraction to recharge.
- CGWB operates a manual quarterly monitoring-well network (the source of
  this project's data) and is expanding a real-time DWLR telemetry network.
- Some states run their own regulatory schemes restricting new borewell
  permits in over-exploited blocks.

## 5. What limitations exist in current approaches?

- The block classification is a **periodic, backward-looking snapshot** —
  it tells you a block *is* critical, not which currently-safe blocks are on
  a trajectory to become critical soonest, so it is a weak input for
  proactive prioritization.
- Classification depends on a fixed extraction/recharge formula rather than
  the observed *trend* in the well data itself, so a block can be
  deteriorating quickly while nominally still "safe."
- Published academic work using CGWB/India-WRIS well data (see the GitHub
  mirror this project bootstraps from, and comparable public projects) is
  mostly retrospective visualization/EDA — it stops at "here's the trend,"
  not "here's a forecast per district you can act on before the crisis is
  official."

## 6. Who could potentially use my solution?

State/district groundwater board officials deciding where to prioritize
extraction restrictions or recharge-structure investment; agricultural
extension officers advising farmers on well-investment risk in their area;
researchers wanting a reproducible baseline forecast to compare against.

## 7. What real-world decision could this analysis/model help with?

*Which districts should be prioritized for intervention next*, ranked by a
combination of current severity and forecast trajectory — turning a
periodic backward-looking classification into a continuously updated,
forward-looking priority list.

## What this project will add over existing work (the gap)

1. A forecasting layer (per well / per district) on top of the trend
   analysis that most public CGWB-data projects stop at.
2. An explicit **early-warning framing**: flag districts whose trend is
   *accelerating*, not just currently bad — the "about to become critical"
   set that a static classification misses.
3. A documented, reproducible data pipeline from real government data
   (mirror + official-source script), including the honest data-quality
   caveats most portfolio projects skip.

This is deliberately scoped as a **decision-support prototype**, not a
production regulatory tool — CGWB's official classification process, which
incorporates recharge estimates this well-level dataset alone cannot
reproduce, remains the authoritative source. The claim this project can
support is narrower and testable: *does a trend/forecast model built on
public well data reproduce and anticipate the officially flagged
over-exploited geography earlier / at finer granularity than the periodic
assessment does* — which is exactly what the evaluation stage (backtesting
forecasts against subsequent CGWB classifications) is designed to check.
