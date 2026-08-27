# Evaluation: Multi-Split Backtest

`docs/ML_MODEL_RESULTS.md` tested one train/test split. This repeats the same comparison across 7 independent walk-forward splits (test = one full year, trained on everything strictly before it) to check whether that single result was representative or just what one particular split happened to show.

## Results by split

| Test year | Train n | Test n | Naive MAE | Seasonal-naive MAE | Ridge MAE | GBM MAE | GBM vs best baseline |
|---|---|---|---|---|---|---|---|
| 2010 | 606,950 | 37,462 | 2.345 | 1.713 | 1.706 | 1.538 | +10.2% |
| 2011 | 648,668 | 35,826 | 2.313 | 1.725 | 1.691 | 1.512 | +12.3% |
| 2012 | 689,733 | 26,955 | 2.511 | 1.731 | 1.763 | 1.607 | +7.2% |
| 2013 | 722,770 | 28,013 | 2.187 | 1.711 | 1.643 | 1.528 | +10.7% |
| 2014 | 774,680 | 43,718 | 2.336 | 1.798 | 1.773 | 1.581 | +12.1% |
| 2015 | 832,734 | 51,293 | 2.307 | 1.790 | 1.804 | 1.602 | +10.5% |
| 2016 | 896,673 | 52,494 | 2.468 | 1.837 | 1.777 | 1.591 | +13.4% |

![Backtest MAE by year](../reports/figures/backtest_mae_by_year.png)

## Is gradient boosting's national win consistent?

**Gradient boosting beats the best baseline in 7/7 splits**, with a mean improvement of +10.9% (std dev 2.0 percentage points across splits). It wins in every single split tested -- the single-split result in ML_MODEL_RESULTS.md was not a fluke; this is a consistent, real improvement over the baselines.

## Is the fastest-declining-district gap consistent?

Across all splits, gradient boosting beat the best baseline in **19/50** individual (district, year) checks with at least 30 test readings (fewer checks than districts x splits, since many district-years fall below that threshold and are excluded as too small to trust). That's a 38% win rate in exactly the districts that matter most for an early-warning use case. This confirms, rather than resolves, the concern raised in ML_MODEL_RESULTS.md: the model's district-level performance in the highest-priority areas is genuinely weaker and less reliable than its strong national average suggests. Before any real early-warning output is built on this model, it needs either district-specific tuning, more training data per district, or an explicit uncertainty estimate so a user knows when to trust a given district's forecast less.

## Honest conclusion

Gradient boosting is a genuine, backtested improvement over both simple baselines nationally (+10.9% average across 7 independent splits). Its advantage in the specific districts an early-warning system cares about most is real but weaker (38% win rate) and should be treated as an open limitation, not a solved problem, going into the explainability and actionable-output stages.
