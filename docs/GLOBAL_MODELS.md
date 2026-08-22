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

Run: `PANEL_LICENSE_ACKNOWLEDGED=true python scripts/build_panel.py --tickers-file universe.txt`

## Feature schema v5 (slice 5)

Schema v5 retains causal v4 groups and adds:

- **Return structure**: overnight return, open-to-close return, high-low
  log range, downside semivariance, realized skew/kurtosis, drawdown from
  rolling peak, positive/negative streaks.
- **Volatility structure**: close-to-close vol at 5/10/20/60, EWMA variance
  (λ=0.94), vol-of-vol, trailing-252 volatility percentile.
- **Liquidity**: log dollar volume, volume surprise, Amihud illiquidity,
  zero-return fraction, stale-price flag.
- **Regime labels**: trend/volatility/liquidity terciles from a trailing
  126-session window only.
- **Cross-sectional ranks**: same-date percentile ranks for momentum, vol,
  and liquidity columns — no future universe membership.

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
