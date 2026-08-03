# Model evaluation and promotion

Price forecasts are evaluated as direct, date-specific horizons (currently 1,
5 and 20 trading sessions in the offline benchmark). A model is not described
as “accurate” from a price-error score: the reporting layer keeps regression
error and directional classification separate.

## Reproducible benchmark

Run the operator-only benchmark with a frozen market-data snapshot identifier:

```powershell
uv run --project backend python backend/benchmark.py --ticker AAPL --output reports/aapl.json
```

Useful controls:

```text
--lookback 60
--horizons 1,5,20
--folds 5
--min-train-size 300
--validation-size 60
--target-type log_return
--feature-sets price,ohlcv,ohlcv_market,ohlcv_technical_market
```

The default target is the direct log return from the forecast origin to each
requested horizon. `price_level`, `simple_return`, and
`persistence_residual` are available for controlled comparisons. Every target
representation is converted back into price units before MAE/RMSE reporting.
Each evaluated baseline sees identical expanding walk-forward folds. Feature scaling is
fitted only on the training slice of each fold, a horizon-sized purge separates
training and validation, and the origin price is retained for every forecast.

The report provides MAE, MSE, RMSE, MAPE, bias, R², MASE, RMSSE and relative
MAE/RMSE against persistence, both per horizon and pooled across
origin–horizon pairs. Directional accuracy compares the predicted movement
with the price known at that specific origin; it does not use `diff` over a
flattened multi-horizon array.

The built-in baseline set is:

- `persistence`: forecast no price change from each origin.
- `drift`: extrapolate the average close-price change in the input window.
- `ridge`: direct multi-output regularized linear regression.
- `elastic_net`: direct multi-output linear regression with combined L1/L2 penalty (alpha 1.0, l1_ratio 0.5).
- `hist_gradient_boosting`: one deterministic tree regressor per horizon.

Feature sets are incremental and run without changing folds or model settings:
price only, OHLCV, OHLCV plus technical values, OHLCV plus market context,
OHLCV plus technical and market context, and all production market features.
The news set appears only after timestamp-safe news columns have been merged.

The benchmark CLI currently evaluates these five deterministic baselines. TensorFlow candidates use the opt-in offline training dependency group; the production browser path uses its own compact TensorFlow.js model and reports `browser_purged_holdout` metrics separately.

## Promotion gate

No model is promoted merely because it wins a single split. The default gate
requires at least a 5% pooled improvement in both MAE and RMSE over
persistence, MASE and RMSSE below one, wins in at least four folds, and no
fold worse than 1.25× persistence RMSE. A rejected report is valuable evidence
that the baseline should continue serving.

## Training lifecycle

Walk-forward folds provide published offline out-of-fold metrics. The Python trainer uses a purged tail to select its epoch count and can write a research artifact only when the opt-in `training` dependency group is enabled. Production does not load that artifact. Browser Quick and Balanced profiles report one untouched purged holdout as `browser_purged_holdout`. Browser Research performs five expanding, 60-session, train-only-scaled and purged folds, pools their untouched predictions as `browser_walk_forward_out_of_fold`, then fits the final local model. Browser and Python evidence are methodologically comparable only when the snapshot, schema, split, architecture, and metric source are disclosed; TensorFlow.js GPU weights are not expected to be bit-identical to Python TensorFlow.

The benchmark promotion decision is advisory and offline. It does not write a Render model directory, update a production endpoint, or change the browser model selected by a user.

## News features

Live Yahoo headlines are descriptive response context only. Historical news features require a licensed, timestamped archive. Records without a publication time are excluded; every session sees only earlier articles. The current feature builder creates exponentially decayed sentiment, effective article count, and sentiment-confidence series. They must enter a controlled ablation and pass the same promotion gate before being added to a future browser schema.

Offline operator pretraining is optional and is not part of Render deployment. Enable the `training` dependency group before preparing a research candidate:

```powershell
uv sync --project backend --frozen --group training --group dev
uv run --project backend python backend/pretrain.py --ticker AAPL --model-type gru
uv run --project backend python backend/pretrain.py --ticker AAPL --model-type bilstm_attention_regression
```

Preparing a candidate makes offline diagnostics available; it does not make the public price or direction endpoint select that candidate automatically. Render has no model directory.


## Target contract

The browser training path fixes `target_mode = "cumulative_log_return_v1"`.
Every price model regresses the cumulative log return from the forecast origin
to its horizon, $r_{t,h} = \ln(P_{t+h}/P_t)$, and forecasts are converted back
to price units as $\hat{P}_{t+h} = P_t \cdot \exp(\hat{r}_{t,h})$. The offline
experiments share this direct-horizon formulation; their default `log_return`
target is the same cumulative log return from each origin.

`backend/experiments/targets.py` defines four `TargetType` values, each with an
exact inverse used before error reporting:

| `TargetType` | Target $y_{t,h}$ | Price reconstruction |
| --- | --- | --- |
| `price_level` | $P_{t+h}$ | identity |
| `simple_return` | $P_{t+h}/P_t - 1$ | $P_t(1 + y)$ |
| `log_return` | $\ln(P_{t+h}/P_t)$ | $P_t \cdot \exp(y)$ |
| `persistence_residual` | $P_{t+h} - P_t$ | $P_t + y$ |

## Per-horizon metric table

`evaluate_forecast_horizons` in `backend/evaluation/metrics.py` emits one
`per_horizon` entry per forecast column plus a `pooled` entry across all
origin–horizon pairs. The report template below uses exactly the columns each
entry provides for persistence-relative comparison:

| horizon | sample_count | mae | rmse | relative_mae | relative_rmse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | … | … | … | … | … |
| 3 | … | … | … | … | … |
| 5 | … | … | … | … | … |
| 7 | … | … | … | … | … |
| 14 | … | … | … | … | … |
| 30 | … | … | … | … | … |
| pooled | … | … | … | … | … |

Rows reflect the horizons actually evaluated; the pooled row aggregates every
origin–horizon pair and is marked `metric_scope = "forecast_origin_horizon_pairs"`.
Relative values are ratios against the persistence forecast, so a value below
one means the candidate beats holding the origin price.

## Candidate interface contract

Two separate candidate interfaces exist and are not interchangeable:

- Backend experiments (`backend/experiments/candidates.py`, `NeuralCandidate`):
  `fit(features, targets, validation_data=None)`, `refit(features, targets)`
  (fresh rebuild and final fit for the selected epoch count),
  `predict(features)`, and `metadata()` returning architecture, target type,
  seed, epoch budget, and selected epoch.
- Research harness (`research/stock_autoresearch/candidates.py`, `Candidate`
  base): `fit(x, y)`, `predict(x)`, `describe()` returning a family/hyperparameter
  dictionary, and `parameter_count()` for resource budgeting.

## Statistical evidence

`backend/evaluation/evidence.py` provides the inference used before any
improvement claim:

- `moving_block_bootstrap_interval` resamples contiguous blocks and returns a
  percentile confidence interval of a mean. The block length is chosen at least
  as large as the horizon so overlapping multi-step targets keep their local
  time dependence.
- `paired_loss_evidence` compares aligned candidate and baseline losses
  (absolute or squared). It reports the mean paired improvement, the
  moving-block bootstrap interval, a Newey-West HAC Diebold–Mariano-style
  statistic, and its two-sided p-value. A positive mean improvement means the
  candidate beats the baseline.
- `relative_ratio_evidence` reports the candidate/baseline error ratio with a
  ratio-scale moving-block bootstrap confidence interval. Values below 1.0
  mean the candidate wins. With `metric="mae"` the errors are absolute and the
  ratio is the MAE ratio; with `metric="rmse"` the errors are squared and the
  ratio is square-rooted, matching the usual RMSE convention.
- `benjamini_hochberg` applies Benjamini–Hochberg FDR control to the full set
  of horizon × model comparisons, so repeated testing across horizons does not
  inflate false discoveries.

## Report extensions

Parallel implementation work adds the following additive report keys; existing
keys are unchanged:

- `seed_summary`: mean, median, standard deviation, best, worst, and
  `failure_count` of pooled relative MAE and RMSE across repeated seeds.
- `evidence_by_horizon`: paired-loss evidence reported separately for every
  horizon; each horizon entry carries absolute and squared paired-loss
  evidence plus relative-ratio evidence (MAE and RMSE ratios).
- `evidence_multiple_comparison`: top-level Benjamini–Hochberg FDR decisions
  over every per-horizon paired-loss p-value; emitted by default whenever
  per-horizon evidence is present.
- `quantile_diagnostics`: pinball loss at the band quantiles, quantile
  crossing rate, and band coverage for the quantile baseline; gated by
  `include_quantiles`.
- `intervals`: split-conformal per-horizon radii, empirical coverage, and
  interval width computed from pooled out-of-fold residuals.
- `drift`: PSI feature divergence between training and evaluation slices plus
  residual drift diagnostics.
- `blend`: persistence shrinkage strength α and constrained blend weights
  combining the candidate with the persistence forecast.

The optional blocks are gated by `ExperimentConfig` opt-in flags: `blend` by
`include_blends`, `quantile_diagnostics` by `include_quantiles`, and `drift`
by `include_drift`. All other extension keys are emitted by default.


## Reference result

The implementation QA run for AAPL used 734 rows from 2023-08-25 through
2026-07-30 and direct 1, 5 and 20-session horizons. Persistence produced pooled
MAE `8.5030` and RMSE `12.2591`. Drift, ridge, and histogram-gradient-boosting
variants did not pass the promotion policy on any tested feature group. The
`elastic_net` baseline was added after this QA run; its reference metrics are
not measured yet — re-run the benchmark to populate them. These
numbers are snapshot-specific evidence, not permanent model-performance
claims; rerun the command to evaluate current data.
