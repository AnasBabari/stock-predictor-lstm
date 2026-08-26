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

The v6 development evidence selected a market-only residual TCN ensemble for short horizons, but its one-use locked certification failed overall. The 3-session asset-transfer check exceeded the preregistered NMM QLIKE guardrail. Strict rejection applies: no v6 weights or passing horizon may be materialized, signed, promoted, or served, and the consumed reserve cannot be reused. The v7 prospective cycle compares exactly two predeclared objectives on development data ending 2026-08-21; see [VOLATILITY_V7_PREREGISTRATION.md](VOLATILITY_V7_PREREGISTRATION.md).

News remains excluded after its matched point-in-time ablation failed to demonstrate incremental value. Reintroducing it requires a separately preregistered future cycle rather than an adjustment after seeing v7 results.

## Evaluation and promotion

All assets share calendar boundaries. Training windows expand through time; the forecast horizon is purged and an embargo prevents adjacent information leakage. Each fold fits its own scaler and preprocessing state. Candidate choices are made on development folds only. Bootstrap intervals, Diebold–Mariano tests with Holm correction, fold consistency, seed dispersion, and calibration are frozen before evaluation.

A locked certification holdout is consumed exactly once after the winner decision. The final refit never supplies reported metrics. If the overall locked result fails any required horizon or asset-transfer guardrail, the entire candidate is rejected: passing horizons are not carved out into a partial release. Materialization requires an overall evidence status of `passed`, and the API abstains when no verified release exists.

Because the v6 reserve is already consumed, v7 development uses no certification rows. A new reserve begins with observations on or after 2026-08-27 and cannot be complete until the required 252 holdout sessions plus the 30-session maximum target have matured. Development selection may create only an unsigned prospective candidate; it is not a release artifact.

## Release bundle

backend/release/bundle.py signs a directory containing ONNX members and manifest.json. The manifest binds runtime schema, feature order, window, horizon list, model id, member seeds/files, certified horizon decisions, certification metrics, and SHA-256 checksums. The serving runtime verifies Ed25519, checksums, paths, input/output names, feature schema, model size, and CPU inference before caching the runtime.

No model binary or private key belongs in Git. Release storage is immutable; promotion updates a pointer only after all checks pass. Response-cache keys include the signed model id so a new release cannot reuse a prior generation's forecast.

## News methodology

News features are derived from timestamped GDELT events before each market origin, aggregated through frozen topic exposures and bounded decay/coverage channels. The ablation report must disclose event counts, missing archive dates, coverage, feature version, and exact matched OOF indices. A news candidate is rejected when coverage is sparse, provider gaps overlap the evaluation advantage, or the paired model fails QLIKE/coverage gates. Live headlines remain context-only until a separately certified snapshot service exists.

## Operational constraints

Research runs on the local RTX machine with bounded CUDA memory, deterministic seeds where supported, subprocess isolation, and resumable run manifests. Render runs CPU-only ONNX inference with a small response cache and no training. /ready can require a verified release; /models exposes the actual status. Vercel uses the volatility client behind VITE_VOLATILITY_SERVING_ENABLED=true.

## Known limitations

- Daily OHLCV volatility is difficult to predict and simple baselines often remain competitive.
- The serving contract's center line is a baseline, not a learned expected return; no release is currently certified.
- News archives have provider gaps and license/coverage constraints.
- The initial panel is survivor-biased and lacks options-implied, intraday, and fundamentals data.
- GPU results are not bit-for-bit identical across machines; snapshot, code, seed, fold, and runtime manifests are the reproducibility unit.
