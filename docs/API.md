# API reference

Local backend: http://127.0.0.1:8000. Interactive OpenAPI: /docs; schema: /openapi.json.

Errors use {"detail":"..."} or a documented structured detail. 400 means an invalid ticker/horizon/model, 409 means the history cannot support the requested causal snapshot, 422 means valid input data is insufficient, 429 means rate limiting, and 503 means an upstream market-data or optional signed-model service is temporarily unavailable.

## Probes and discovery

- GET / — service metadata and documentation links.
- GET /health — O(1) liveness; does not load model files.
- GET /ready — market-data readiness and, when required/configured, verified global-release readiness.
- GET /models — active baseline status, plus signed global-volatility compatibility status and metric source.
- GET /api/v1/search?query=AAPL — bounded Yahoo suggestions.
- GET /api/v1/info?ticker=AAPL — fundamentals with a bounded cache.

## GET /api/v1/volatility/forecast

This is the active product endpoint. It computes a causal statistical
volatility baseline from the latest validated OHLCV history; it does not load
model files or require a signed release.

Parameters:

- `ticker`: validated `[A-Z0-9.\\-]{1,12}` symbol.
- `horizon`: one of `1, 3, 5, 7, 14, 30` trading sessions.
- `model`: `persistence`, `rolling_mean`, `ewma`, or `har_rv` (default).

Rate limit: 30 requests/minute/client. The request computes features through
the last observed session and returns p05–p95 price paths derived from the
selected annualised volatility estimate. It never accepts a client feature
matrix, model path, or weight payload.

Example response (abbreviated):

    {
      "ticker": "MSFT",
      "as_of": "2026-08-28",
      "horizon": 7,
      "current_price": 500.0,
      "forecast": {
        "future_dates": ["2026-08-31", "..."],
        "price_quantiles": {"p05": ["..."], "p50": ["..."], "p95": ["..."]},
        "expected_annualized_volatility": 0.21,
        "volatility_unit": "annualized_sigma",
        "model": "har_rv",
        "baseline": true
      },
      "evidence": {
        "model_status": "baseline",
        "model_family": "statistical_baseline",
        "metric_source": "baseline_definition",
        "target": "future_realized_volatility_close_to_close",
        "snapshot_id": "sha256...",
        "schema_version": "deployable_v5",
        "news_status": "not_used"
      }
    }

The p50 path is anchored to the unchanged latest close because this endpoint
forecasts uncertainty, not expected return. The response is intentionally not
called a certified or LSTM forecast. Offline model comparisons and their
70/15/15 test metrics are documented in
[VOLATILITY_FORECASTING.md](VOLATILITY_FORECASTING.md).

## GET /api/v2/forecast

Parameters:

- ticker: validated [A-Z0-9.\\-]{1,12} symbol.
- horizon: one of 1, 3, 5, 7, 14, 30 trading sessions.

Rate limit: 30 requests/minute/client. The route verifies a signed release before market-data work and before reading the response cache. Missing/stale/tampered releases and failed horizons return 503 with status abstain_no_certified_model.

Example response (abbreviated):

    {
      "ticker": "MSFT",
      "as_of": "2026-08-21",
      "horizon": 7,
      "current_price": 501.2,
      "forecast": {
        "future_dates": ["2026-08-24", "..."],
        "price_quantiles": {
          "p05": ["..."], "p50": ["..."], "p95": ["..."]
        },
        "probability_up": null,
        "expected_cumulative_variance": 0.0012,
        "expected_cumulative_return": 0.012,
        "return_distribution_variance": 0.0015,
        "return_distribution_family": "student_t",
        "expected_annualized_volatility": 0.208
      },
      "evidence": {
        "model_id": "global-tcn-v1",
        "snapshot_id": "sha256...",
        "metric_source": "locked_purged_walk_forward",
        "certified": true,
        "certified_heads": {
          "volatility": true,
          "return_distribution": true,
          "direction": false
        },
        "certified_head_horizons": {
          "volatility": [1, 3, 5, 7],
          "return_distribution": [7],
          "direction": []
        },
        "horizon_certification": {"7": {"decision": "pass"}}
      }
    }

The quantile bands are the learned/certified return distribution. For a
`certified_heads.return_distribution = true` release, the terminal p50 is the
certified Student-t return-location (median) path reconstructed from the latest
close; `expected_cumulative_return` and `return_distribution_variance` describe
that same terminal distribution. Intermediate daily points are transparent
linear interpolations of cumulative location and variance and are not separate
certification horizons. Direction remains uncertified (`probability_up = null`).

Legacy releases with `return_distribution = false` retain a zero-location
normal cone: their p50 is the unchanged-close distribution assumption and the
frontend intentionally does not render it as a learned price line.

When a signed release is news-certified, the evidence additionally reports `news_input` telemetry when the live news provider is enabled (`VOLATILITY_NEWS_PROVIDER_ENABLED=true`): `provider_cutoff_utc` (the origin-session close, 20:00 UTC), `eligible_article_count` (causally eligible articles), and `news_feature_count` (certified schema size). With the provider disabled, a news-certified release answers with a structured 503 abstention instead of a forecast.

## GET /models

The response contains:

- global_volatility.status: ready, unconfigured, unavailable, or integrity_failure.
- global_volatility.model_id and certified_horizons when ready.
- global_volatility.metric_source is the admitted signed release's declared
  source; failed V11.2 `sealed_holdout_once` evidence is never exposed as a
  ready model.
- volatility_forecasting.status = available and model_storage.required = false for the active train-free baseline contract.
- browser_training.status = disabled and server_models.status = disabled for the active production contract; the signed global route is a separate compatibility path.

## GET /ready

The data service can be ready without a model for local development. With `VOLATILITY_SERVING_REQUIRED=true`, the current 0/4 external-certification policy keeps readiness degraded even if a development release path is configured. A future production admission change must integrate authentic external evidence; editing a JSON status is insufficient. The liveness route remains 200 so the platform can report the actual cause.

## GET /api/v1/training-data

This bounded route is for diagnostics and offline research. It returns the validated Stationary Schema v4 matrix, historical closes, future calendar dates, and deterministic snapshot id. It accepts no client feature matrix and writes no files. It is not a public training service and is not used by the production global-volatility path.

V11.2 is an archived RTX research workflow. Its reserve was opened once and the candidate failed, so dataset preparation, certification, release assembly, and runtime loading now reject the generation. Its reports retain `sealed_holdout_once` only as historical methodology evidence; they cannot authorize an API response.

## Compatibility forecasts

GET /api/v1/predict and GET /api/v1/predict/direction remain temporarily for old clients. They return persistence/recent-base-rate outputs with metadata.engine.role = server_disabled_fallback, baseline_fallback = true, and metrics.metric_source = baseline_definition. A flat compatibility result must never be labelled an LSTM or global model.

The legacy GET /api/v1/server-forecasts/availability and GET /api/v1/server-forecasts/{ticker} routes are retained for migration only. Production keeps them disabled; they must not be used as the global-volatility contract.

## Diagnostics and telemetry

- GET /api/v1/prediction-status/{request_id} — short-lived telemetry for compatibility requests only.
- GET /api/v1/diagnostics/{ticker} — offline diagnostics when explicitly mounted; production Render normally has none.
- GET /api/v1/model-performance/{ticker} — discloses the global volatility family, signed-release status, locked metric source, and certified horizons. Direction is reported as not certified.

Compatibility forecast metadata may include typed timings_seconds (queue_wait, market_data, feature_preparation, artifact_load_validation, training, inference, and total) and execution mode (response_cache_hit, artifact_loaded, baseline_fallback, trained, or coalesced). It also carries artifact_state_before (fresh, missing, stale, or incompatible) and artifact_action (loaded, retrained, or not_applicable). Status stages include queued, downloading_market_data, preparing_features, checking_artifact, training, generating_forecast, completed, and failed. This caller-level semantics is short-lived: completed and failed views remain for up to 10 minutes and can be evicted under capacity pressure. These fields describe request telemetry only and never turn a baseline into a learned result.

## News

Live Yahoo headline sentiment is context-only. Historical GDELT news enters only the offline timestamped ablation pipeline. News coverage, provider gaps, exposure-map version, decay parameters, and snapshot id are recorded in the research report; no news feature is served until the ablation and certification gates pass.
