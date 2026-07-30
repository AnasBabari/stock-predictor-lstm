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
- `hist_gradient_boosting`: one deterministic tree regressor per horizon.

Feature sets are incremental and run without changing folds or model settings:
price only, OHLCV, OHLCV plus technical values, OHLCV plus market context,
OHLCV plus technical and market context, and all production market features.
The news set appears only after timestamp-safe news columns have been merged.

The benchmark CLI currently evaluates these four deterministic baselines; TensorFlow candidate architectures use the separate artifact-training workflow described below.

## Promotion gate

No model is promoted merely because it wins a single split. The default gate
requires at least a 5% pooled improvement in both MAE and RMSE over
persistence, MASE and RMSSE below one, wins in at least four folds, and no
fold worse than 1.25× persistence RMSE. A rejected report is valuable evidence
that the baseline should continue serving.

## Training lifecycle

Walk-forward folds provide published out-of-fold metrics. The final model uses
a purged tail only to select its epoch count, then a fresh model is refit on
all available labelled samples for that selected number of epochs. This keeps
the production artifact distinct from the selection model.

The benchmark promotion decision is advisory and offline. It does not write
model artifacts, update `current.json`, or change the model served by FastAPI.

## News features

Live Yahoo headlines are descriptive response context only. Historical news
features require a licensed, timestamped archive. Records without a publication
time are excluded; every session sees only earlier articles. The current
feature builder creates exponentially decayed sentiment, effective article
count, and sentiment-confidence series. They must enter a controlled ablation
and pass the same promotion gate before being added to a production model.

The default operator pretraining set remains `lstm` plus
`bilstm_attention_direction`. Candidate architectures can be prepared
explicitly:

```powershell
uv run --project backend python backend/pretrain.py --ticker AAPL --model-type gru
uv run --project backend python backend/pretrain.py --ticker AAPL --model-type bilstm_attention_regression
```

Preparing a candidate makes diagnostics available; it does not make the public
price or direction endpoint select that candidate automatically.

## Reference result

The implementation QA run for AAPL used 734 rows from 2023-08-25 through
2026-07-30 and direct 1, 5 and 20-session horizons. Persistence produced pooled
MAE `8.5030` and RMSE `12.2591`. Drift, ridge, and histogram-gradient-boosting
variants did not pass the promotion policy on any tested feature group. These
numbers are snapshot-specific evidence, not permanent model-performance
claims; rerun the command to evaluate current data.
