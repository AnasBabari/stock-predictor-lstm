# API reference

Local backend: http://127.0.0.1:8000. Interactive OpenAPI: /docs; schema: /openapi.json.

Errors use {"detail":"..."} or a documented structured detail. 400 means an invalid ticker/horizon or insufficient history, 409 means the history cannot support the requested causal snapshot, 429 means rate limiting, 502 means an upstream market-data failure, and 503 means the signed model is unavailable, incompatible, uncertified, or failed closed.

## Probes and discovery

- GET / — service metadata and documentation links.
- GET /health — O(1) liveness; does not load model files.
- GET /ready — market-data readiness and, when required/configured, verified global-release readiness.
- GET /models — signed global-volatility status, model id, certified horizons, disabled legacy/browser paths, and metric source.
- GET /api/v1/search?query=AAPL — bounded Yahoo suggestions.
- GET /api/v1/info?ticker=AAPL — fundamentals with a bounded cache.

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
          "p05": ["..."], "p50": [501.2], "p95": ["..."]
        },
        "probability_up": null,
        "expected_cumulative_variance": 0.0012,
        "expected_annualized_volatility": 0.208
      },
      "evidence": {
        "model_id": "global-tcn-v1",
        "snapshot_id": "sha256...",
        "metric_source": "locked_purged_walk_forward",
        "certified": true,
        "certified_heads": {
          "volatility": true,
          "return_distribution": false,
          "direction": false
        },
        "horizon_certification": {"7": {"decision": "pass"}}
      }
    }

The quantile bands are the learned/certified volatility head. The p50 line is an unchanged-close location baseline. The endpoint makes no claim that direction or expected price level was learned.

## GET /models

The response contains:

- global_volatility.status: ready, unconfigured, unavailable, or integrity_failure.
- global_volatility.model_id and certified_horizons when ready.
- global_volatility.metric_source = locked_purged_walk_forward.
- browser_training.status = disabled and server_models.status = disabled for the production contract.

## GET /ready

The data service can be ready without a model for local development. Set VOLATILITY_SERVING_REQUIRED=true in a production serving environment to make readiness fail closed until the signed release verifies. The liveness route remains 200 during release failures so the platform can report the actual cause.

## GET /api/v1/training-data

This bounded route is for diagnostics and offline research. It returns the validated Stationary Schema v4 matrix, historical closes, future calendar dates, and deterministic snapshot id. It accepts no client feature matrix and writes no files. It is not a public training service and is not used by the production global-volatility path.

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
