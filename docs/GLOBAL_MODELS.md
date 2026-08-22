# Global Model Pipeline

This document describes the offline global-model training pipeline
(slices 4–10) and the browser-side serving path (slices 11–13). It covers
provenance, feature schema, candidate space, evaluation protocol,
selection gates, release flow, and known limitations.

## Architecture overview

```text
Offline panel data and research
        │
        ├── Econometric baselines (EWMA, GARCH, GJR, HAR)
        ├── Linear/tree baselines (Ridge, ElasticNet, DLinear)
        ├── Global LSTM/GRU (shared encoder across tickers)
        ├── GARCH-LSTM hybrid volatility model
        └── Global TCN candidates
                 │
        Purged walk-forward evaluation (calendar-aligned)
                 │
       Per-task/per-horizon champions + shrinkage blending
                 │
     Signed versioned TF.js release bundle
                 │
       Vercel/CDN static model delivery
                 │
Render feature snapshot → Browser inference
                 │
       Optional local calibration only
                 │
 Learned + baseline + blended + intervals
```

## Data provenance (slice 4)

Panel snapshots are immutable and content-addressed. The builder validates
every ticker's OHLCV history (duplicates, chronology, finite values,
positive prices), flags >20% single-day moves as suspicious adjustments,
computes per-ticker sha256 checksums, and produces a pooled panel id.
Files are write-once; corrections produce a new snapshot.

**License gate:** `PANEL_LICENSE_ACKNOWLEDGED=true` is required before any
provider-backed download. Review the provider's terms for offline model
training, public application use, derived weight distribution, caching,
and redistribution restrictions first.

**Survivorship limitation:** the initial universe (~500–1,000 currently
listed US equities) is survivor-biased. Results must be labelled as such
until a point-in-time universe containing delisted securities is available.

Run pipeline:
`python scripts/run_global_pipeline.py --run-dir runs/prod`
or build panel snapshot:
`python scripts/build_panel.py --run-dir runs/panel_snapshot --n-tickers 50 --n-sessions 500`

To evaluate and certify against the locked holdout:
`python scripts/run_global_pipeline.py --open-locked-certification-holdout --run-dir runs/certified`

## Feature schema (slice 5 & 9)

Features are split into two explicit contracts:

### Deployable Schema (`deployable_v5` — 26 features)
Single-ticker stationary indicators reproducible causal at browser/backend inference:
- **Return structure (13)**: Return_1D, Return_5D, Return_10D, Return_20D, Overnight_Return, OpenToClose_Return, HL_Range_Log, Downside_Semivar_20, Realized_Skew_20, Realized_Kurt_20, Drawdown_From_Peak, Up_Streak, Down_Streak.
- **Volatility structure (7)**: Vol_C2C_5, Vol_C2C_10, Vol_C2C_20, Vol_C2C_60, EWMA_Var (λ=0.94), Vol_Of_Vol_20, Vol_Percentile_252.
- **Liquidity (6)**: Log_Dollar_Volume, Dollar_Volume_Median_20, Volume_Surprise, Amihud_Illiquidity_20, Zero_Return_Fraction_20, Stale_Price_Flag.

### Research Schema (`research_v5`)
Extends `deployable_v5` with research-only regime labels (trend/vol/liquidity terciles) and causal same-date cross-sectional ranks (`*_XSRank`). Cross-sectional features require the whole panel and are excluded from single-ticker deployable champion models.

Every feature is causal: row t uses only information from rows ≤ t.
Ablation testing is required before promoting any feature group.

## Evaluation protocol (slices 6–7)

### Calendar-time separation

All assets share the SAME session grid and boundaries. Expanding training
windows advance in calendar time with a horizon purge plus embargo gap.
An asset-transfer holdout reserves entire tickers that never appear in
training.

### Baselines

Volatility baselines include EWMA/RiskMetrics, HAR-RV, GARCH(1,1)-t, and
GJR-GARCH. Return baselines include persistence/zero return, shrunk
rolling mean, Ridge/ElasticNet, and DLinear. Direction baselines include
the matched pre-evaluation base rate and market-direction classifier.

### Metrics

Return/price: log-return MAE/MSE/RMSE, relative MAE/RMSE vs persistence.
Direction: multiclass Brier + skill, macro balanced accuracy/F1, log loss,
calibration ECE. Volatility: QLIKE (primary), log-variance MAE/RMSE,
Mincer–Zarnowitz calibration. Distribution: pinball loss, CRPS, coverage.

No "price accuracy" percentages are used anywhere in this project.

### Promotion gates

Promotion occurs independently for every task/horizon. Gates require:
relative MAE and RMSE both < 1.0; bootstrap 95% CI upper bound < 1.0;
DM test significance after Holm correction; ≥4 of 5 folds beating the
baseline; no fold exceeding a 1.15 relative-RMSE ceiling; seed dispersion
within threshold; calibration checks passing.

These thresholds are promotion requirements, not hyperparameters to tune
against repeatedly.

### Shrinkage blending

Admissible candidates receive a convex blend weight α ∈ [0,1] regularized
toward zero. A marginal model produces a cautious near-baseline forecast;
α = 0 means the learned model supplied no usable edge at that horizon.

## Candidate space (slices 8–9)

| Family | Description | Dependencies |
|---|---|---|
| persistence | Zero cumulative excess return | none |
| ridge/enet_global | Regularized linear over flattened windows | scikit-learn |
| dlinear_global | Trend + residual decomposition linear | scikit-learn |
| global_lstm/gru | Shared-encoder recurrent with quantile+direction heads | TensorFlow (opt-in) |
| global_tcn | Residual causal dilated convolutions | TensorFlow (opt-in) |
| garch_lstm | Two-branch hybrid: econometric forecasts + LSTM residual | TensorFlow + scipy |

Neural candidates require the opt-in `training` dependency group and are
never promoted without evidence on frozen untouched data.

## Release flow (slice 11)

1. Offline training produces a trained Keras model.
2. TF.js conversion via `tensorflowjs_converter` (requires npm install).
3. `backend/release/bundle.py` assembles manifest.json with per-file
   sha256 checksums, signs with Ed25519 using the existing signing key.
4. Verification fails closed on missing key, invalid signature, or any
   checksum mismatch.
5. Release bundles are published as immutable GitHub Release assets;
   Vercel downloads them at build time after verifying checksums.

Raw model weights are never committed to Git and never placed on Render.

## Browser serving (slices 12–13)

The frontend loads a pinned catalog (`/models/global/catalog.json`),
verifies artifact checksums via WebCrypto, loads via tfjs, and runs
inference. Feature-flagged off by default (`VITE_GLOBAL_MODEL_ENABLED`).
When unavailable, the system transparently falls back to browser-local
training or labelled baselines — never presenting a fallback as a model.

Local calibration may adjust bias/scale/temperature/conformal intervals
using only pre-origin data. It cannot retrain the full production model.

## Known limitations

- Daily OHLCV carries limited predictive signal; simple baselines are
  often competitive.
- The initial research universe is survivor-biased.
- No intraday, options-implied, fundamental, or news data is included.
- Single-process backend by design.
- All claims require frozen untouched-data certification before production.
