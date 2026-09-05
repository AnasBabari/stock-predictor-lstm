# Market-context validation experiment

## Scope and artifacts

`scripts/run_market_context_comparison.py` consumes the preserved baseline run.
It verifies original cached-file hashes and train/validation index hashes,
reconstructs predictions from `ridge.json` and the large LSTM's
`selection_model.pt`, and checks MAE/RMSE/direction parity before proceeding.
The deployed/refitted model is never used for this reconstruction.

The output directory is new and cannot overwrite an existing experiment.
`validation_baseline.parquet` has one row per validation sample/horizon, including
sample index, stock, market, date, horizon, actual log return, Ridge/LSTM log-return
predictions, and zero-return persistence. There are seven records per origin.
No fitting, prediction, or scoring of the test partition is performed. The full
cache is still read to reconstruct the original dataset; this is not data sealing.

## Paired inference

For each horizon, errors use `100 * abs(exp(prediction) - exp(actual))`.
Reference error minus candidate error is averaged across stocks on each date.
The mean of this daily series is tested with a Bartlett/Newey-West intercept
standard error, using the n/(n-1) finite-sample correction and normal 95% intervals.
This is date-weighted evidence, which differs from the row-weighted pooled MAE
when daily stock counts differ. US and UK sensitivity reports are also produced.

HAC bandwidths are h-1, 2(h-1), and 3(h-1); day 1 uses 0, 1, and 5 instead of
three identical zeros. Lags count consecutive observed dates, not calendar days.
The covariance calculation follows the Bartlett HAC convention documented by
[statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).
The implementation is checked against an explicit Bartlett covariance matrix.

`baseline_hac.json` compares Ridge/LSTM with persistence and with one another.
`context_hac.json` compares each augmented model with its original model and
persistence. Positive loss improvement favors the candidate. Raw p-values and
conservative Holm-adjusted p-values over all reported tests in each file are
included. Zero standard-error cases have null p-values. These are exploratory
validation comparisons, not confirmatory tests of an independently selected model.

## Features and missingness

US and UK baskets are isolated using the existing `.L` ticker convention. No
same-date US prices enter a UK forecast. These are end-of-day historical features;
production use would still need provider availability/finalization timestamps.

The additional features are:

- Leave-one-out mean log return over 1/5/20 market-basket dates.
- Twenty-date sample standard deviation of those leave-one-out returns.
- Cross-sectional sample standard deviation of active stock returns.
- Stock cumulative log return minus the leave-one-out aggregate over 1/5/20 dates.
- Missing/stale flag, number of dates since a complete context vector, and peer coverage.

Only finite one-date returns with positive current volume contribute. A missing
close on the previous basket date invalidates the next return; a multi-date
return is never silently counted as a one-day return. At least 80% of registered
peers and at least two peers are required. The target's return is excluded from
its market mean. Dispersion follows the requested whole-basket definition.

Rolling metrics are calculated on raw valid returns. Missing rolling values
forward-fill from prior values only; initial gaps use zero placeholders plus the
missing flag. The stale counter measures time since all context features were
available together. These flags persist during recovery of the rolling windows.
Per-stock coverage counts are exported to `context_coverage.json`.

Features align to the original feature dates and sequence positions. Original
25 features, target values, sample IDs, and split indices stay unchanged. The
augmented matrix has 36 features. No stocks or validation rows are dropped.

## Fixed comparison and limitations

Ridge remains alpha 100 with latest-vector inputs; LSTM remains 128 units,
3 layers, maximum 20 epochs, patience 3, and the saved baseline settings.
Only input width changes. Each fits its own training-only scaler.

CPU normalization uses two-pass float64 moments in bounded chunks and scales
individual mini-batches, avoiding full-array normalized copies. Regression tests
check matching moments against the original full-array calculation. This memory
change does not change the feature definitions, loss, splits, or schedule.

The cached basket is a current selected universe, not point-in-time index
membership or an investable equal-weighted index. Mean log returns describe a
geometric basket proxy. Survivor selection and changing coverage limit inference.
Existing chronological partitions are per stock; their calendar boundaries can
differ across stocks. They are preserved for this ablation, not represented as
a globally synchronized temporal holdout. Model selection on this validation
set also limits p-value interpretation. News and sector classifications remain
outside this experiment. No result automatically deploys a model.
