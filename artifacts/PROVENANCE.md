# Study provenance record — shared-tree hygiene

Branch: study/gpu-panel-volatility-v1  (from v1.0-ohlcv-negative-study, tag ef2d4d8)
Frozen study protocol: 4 closed negative hypotheses (absolute, context, macro news,
beta-neutral rank); 5 completed positive validations (B1/B2/B2/B1-YZ confirmation, G3 promotion, C null, rolling-origin replication, untouched test burn).
This provenance file exists only for hygiene; its presence in the shared-tree is expected, but its deletion is safe and preferred once another agent's work lands independently.

## Artifacts this session created (maintained, not garbage)

- `artifacts/STUDY_SUMMARY.md` — 4-null freeze, preserved caches.
- `artifacts/options_iv_study/PREREGISTRATION.md` — G3 preregistration, timing convention.
- `artifacts/DECISION.md` (if present) — G3 design decision.
- `artifacts/vol_structure_v1/` — frozen reference layer (range estimators, asymmetry, HAR wrapper, GJR/EGARCH, harness tests, A-D evidence).
- `artifacts/vol_structure_b/`, `..._c/`, `..._d/` — B1/B2/C/D panel results.
- `artifacts/vol_structure_study/` — study STATUS (A-D done, E/F not executed).
- `artifacts/gpu_rolling_origin_v1/` — rolling-origin replication (fold reports, pooled, promotion, test burn).
- `artifacts/gpu_vol_panel_v1/` — 286-stock pooled GPU panel (G0–G4, protocol, decision, QLIKE metrics, coverage); 237KB ONNX artifacts in `backend/volatility_models/` (tracked, parity-proven).
- `data/macro/market_dailies.parquet` — SPY + market series (reused by history endpoint).
- `data/macro/ftse_dailies.parquet` — native GBp FTSE reference.
- `scripts/probe_options_adapter.py` — plumbing-only probe.
- `scripts/package_g3_models.py` — ONNX packaging harness.
- `data/options/providers/{base,orats,yfinance_probe}.py`, `schemas.py`, `normalize.py`, `surface.py`, `features.py`, `validation.py`, `SAMPLES.md` — adapter framework.
- `front-end` new/updated: `App.jsx`, `styles.css`, `simpleForecastClient.js`, chart/hook/utils/test files.
- `backend`: history endpoint (`routes/market.py`), G3 serving (`services/g3_volatility.py` + `live_volatility.py`), revision gate (`news_archive.py`), artifacts (`forecast_artifacts.py`), config (`forecast_model_cache_dir`), collector fix (`scripts/collect_live_forecasts.py`), history + G3 tests.

## Shared-tree hygiene protocol observed

- No secrets committed (credential fragments verified absent).
- No `.pt` / `.ckpt` / `.h5` / `.keras` / `.pth` blobs committed (`*.parquet`/`*.jsonl` excluded by `.gitignore` except the explicitly tracked `data/news/alpaca/*.jsonl` archives with verified sha256 manifests).
- No training mutation during this session (frozen protocol).
- Strict index commits via `strict_commit_lib.ps1` — 13 files exactly per commit, 0 unexpected.
- Stand-down: commit/checkout/reset/clean/stash all deferred until the user confirms another agent finished.

If another agent's new artifacts overlap with the above lists, coordinate: keep this session's artifacts intact; don't overwrite them in a bare `git commit` or `git add -A`.
