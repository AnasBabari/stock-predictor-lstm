# Global volatility models

This is the research, certification, and release contract for the production global model. It is deliberately separate from browser TFJS experiments and from Render request handling.

## Pipeline

    immutable panel snapshot + point-in-time news snapshot
            ↓
    causal Deployable Schema v5 + econometric baselines
            ↓
    paired market-only/news candidate evaluation on CUDA
            ↓
    calendar-aligned expanding folds with purge + embargo
            ↓
    horizon-specific QLIKE/coverage/calibration/promotion gates
            ↓
    one locked untouched certification holdout
            ↓
    CPU-parity ONNX members → signed immutable release
            ↓
    Render /api/v2/forecast → Vercel volatility cone

## Data provenance

Panel files are immutable and content-addressed. The builder validates positive prices, chronology, duplicates, finite values, suspicious adjustments, per-ticker checksums, and pooled snapshot identity. The initial universe is survivor-biased and must be labelled as such. Provider licenses are acknowledged before downloads; model redistribution rights are checked before signing.

The GDELT builder stores one verified daily part per archive, raw/aggregated checksums, coverage cutoffs, alias-map version, aggregation counts, and explicit missing_archive_dates. A provider HTTP 404 may be recorded only with the explicit --record-missing-404 operator flag; all other failures remain fail-closed. This prevents absent archives from becoming silent zero-news observations.

## Deployable Schema v5

The serving contract has 26 causal columns: return structure, overnight/open-close/range and drawdown terms, realized-volatility proxies, EWMA/HAR inputs, liquidity/illiquidity terms, and stale-price diagnostics. Research-only cross-sectional ranks and regime labels are not accepted by the single-ticker serving runtime.

## Candidates and metrics

Candidates include persistence, shrunk mean, Ridge/ElasticNet, DLinear, residual TCN, and GARCH-LSTM hybrids, with EWMA/HAR/GARCH/GJR as variance baselines. Volatility is scored primarily with QLIKE plus log-variance error, calibration, interval coverage, and width. Return-location and direction heads are evaluated separately and may be withheld even when volatility clears its gate.

The current development evidence selected a market-only residual TCN ensemble for short horizons. It is not production-certified until the locked holdout and release verification complete. News is retained as a paired ablation candidate; it can displace the market-only model only when it improves the same horizons on identical origins and survives the predeclared gate.

## Evaluation and promotion

All assets share calendar boundaries. Training windows expand through time; the forecast horizon is purged and an embargo prevents adjacent information leakage. Each fold fits its own scaler and preprocessing state. Candidate choices are made on development folds only. Bootstrap intervals, Diebold–Mariano tests with Holm correction, fold consistency, seed dispersion, and calibration are frozen before evaluation.

The locked certification holdout is consumed exactly once after the winner decision. The final refit is used to create the deployable artifact but never supplies reported metrics. A horizon that does not clear its guardrails is absent from certified_horizons and the API abstains.

## Release bundle

backend/release/bundle.py signs a directory containing ONNX members and manifest.json. The manifest binds runtime schema, feature order, window, horizon list, model id, member seeds/files, certified horizon decisions, certification metrics, and SHA-256 checksums. The serving runtime verifies Ed25519, checksums, paths, input/output names, feature schema, model size, and CPU inference before caching the runtime.

No model binary or private key belongs in Git. Release storage is immutable; promotion updates a pointer only after all checks pass. Response-cache keys include the signed model id so a new release cannot reuse a prior generation's forecast.

## News methodology

News features are derived from timestamped GDELT events before each market origin, aggregated through frozen topic exposures and bounded decay/coverage channels. The ablation report must disclose event counts, missing archive dates, coverage, feature version, and exact matched OOF indices. A news candidate is rejected when coverage is sparse, provider gaps overlap the evaluation advantage, or the paired model fails QLIKE/coverage gates. Live headlines remain context-only until a separately certified snapshot service exists.

## Operational constraints

Research runs on the local RTX machine with bounded CUDA memory, deterministic seeds where supported, subprocess isolation, and resumable run manifests. Render runs CPU-only ONNX inference with a small response cache and no training. /ready can require a verified release; /models exposes the actual status. Vercel uses the volatility client behind VITE_VOLATILITY_SERVING_ENABLED=true.

## Known limitations

- Daily OHLCV volatility is difficult to predict and simple baselines often remain competitive.
- The current production center line is a baseline, not a learned expected return.
- News archives have provider gaps and license/coverage constraints.
- The initial panel is survivor-biased and lacks options-implied, intraday, and fundamentals data.
- GPU results are not bit-for-bit identical across machines; snapshot, code, seed, fold, and runtime manifests are the reproducibility unit.
