# Simplified volatility forecasting

The active forecasting path is intentionally small and inspectable. Render
downloads and validates market data, computes features through the latest
observed session, and serves a causal statistical volatility cone. It does
not train a model, write model files, or require a signed release to answer a
normal product request.

The historical `/api/v2/forecast` signed-release route is retained for
compatibility and archival reproduction. It is not the active product route
and continues to abstain when no verified release is configured.

## Target

For an origin session `t` and horizon `H`, the target is future annualised
realised volatility:

```text
RV(t,H) = sqrt(252 / H * sum(log(C[t+i] / C[t+i-1])² for i=1..H))
```

Every input feature is computed from rows through `t`; no feature reads the
future label window. The active API and offline benchmark use this
close-to-close target (the OHLC range and overnight proxies are research-only
alternatives). The implementation is in
`research/volatility_forecasting/simple_pipeline.py` and is also covered by
causality tests.

## Active API

```text
GET /api/v1/volatility/forecast?ticker=MSFT&horizon=7&model=har_rv
```

Supported horizons are 1, 3, 5, 7, 14, and 30 trading sessions. Supported
baseline models are:

| Model | Definition |
| --- | --- |
| `persistence` | Current 20-session close-to-close volatility carried forward |
| `rolling_mean` | Current 60-session rolling volatility |
| `ewma` | RiskMetrics-style exponentially weighted volatility |
| `har_rv` | Train-only log-HAR regression over 5/22/60-session volatility |

The response includes dated p05–p95 price paths derived from the selected
volatility, the annualised sigma, snapshot identity, feature schema, and:

```json
{
  "model_status": "baseline",
  "model_family": "statistical_baseline",
  "metric_source": "baseline_definition",
  "news_status": "not_used"
}
```

This vocabulary is deliberate: a volatility cone is not a point-price LSTM
forecast, and baseline output is never labelled as a learned model.

## Offline benchmark protocol

The research benchmark accepts one or more OHLCV CSV files:

```powershell
backend/.venv/Scripts/python.exe scripts/run_volatility_benchmark.py `
  --csv snapshot_MSFT.csv `
  --csv snapshot_QQQ.csv `
  --csv snapshot_SPY.csv `
  --horizon 5 `
  --include-lstm `
  --output reports/volatility.json
```

The default split is chronological 70% train, 15% validation, and 15% test.
The validation and test partitions start after an embargo of at least `H`
sessions, so a training label cannot overlap a later partition. Scalers are
fit only on the training sequences. The selected model is the minimum
validation QLIKE; test observations are not used for selection.

Reported metrics are MAE, MSE, RMSE, QLIKE, and R² on annualised volatility.
QLIKE is the primary volatility score because it evaluates the forecasted
variance and penalises both under- and over-estimation. Lower MAE, MSE, RMSE,
and QLIKE are better; higher R² is better. Scores are comparable only when
the target definition, horizon, snapshot, and partition are identical.

The benchmark also includes train-only-scaled Ridge and ElasticNet regressors
plus one deterministic gradient-boosting candidate. The optional
`--include-lstm` flag runs a compact offline PyTorch LSTM. It is lazy-imported,
uses a train-only standard scaler, predicts log volatility, uses
validation-only early stopping, and never writes weights. PyTorch is an offline
research dependency; it is not part of the Render API process.

## Data and news boundary

The current active target uses daily OHLCV-derived volatility. Live headlines
and historical news are deliberately excluded from the first simplified
benchmark. News can be added only after timestamped historical coverage is
available and its `available_at` time is enforced; otherwise publication
revisions and survivorship would leak future information. A future news
ablation must beat the same no-news baseline on the same origins and must
report coverage and QLIKE, not just directional accuracy.

## Interpreting the cone

The p50 path is anchored to the latest observed close because the active
baseline estimates uncertainty, not expected return. A wider cone means the
historical volatility estimate is higher; it does not imply a directional
price call. The dashboard therefore shows “Causal baseline” and “Annualized
Volatility” instead of “Certified” or “LSTM accuracy.”

## Migration boundary

Older v7–v11 protocol, certification, signed-bundle, and browser-training
modules remain in the repository for reproducibility and are not deleted as
part of this simplification. New product code must not import them to answer
`/api/v1/volatility/forecast`. A future learned model may replace a baseline
only after it is evaluated by the same target/split protocol and its response
metadata names the learned family and metric source explicitly.
