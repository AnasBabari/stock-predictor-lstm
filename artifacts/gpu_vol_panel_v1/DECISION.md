# GPU volatility panel — candidate decision (frozen 2026-09-07)

Source: `artifacts/gpu_vol_panel_v1/` (286 stocks, ~466k common validation
origins, H = 5/10/20, QLIKE primary, HAC inference).

## Designations

- **PRIMARY VALIDATION CANDIDATE — G3**: global GPU XGBoost rolling-volatility
  correction (QLIKE objective, base_margin=log(rolling)). ΔQLIKE vs G0:
  +0.217 / +0.227 / +0.250, HAC p ~ 1e-9..1e-53, MAE agrees. Chosen over G2
  on preregistration (nominal candidate) plus structural properties
  (positivity by construction, interpretable correction, graceful fallback
  to rolling with δ=0) — NOT on the point estimates, where G2 is fractionally
  ahead without a paired significance test.
- **SECONDARY / SHADOW — G2**: direct log-variance XGBoost. Tracked, never
  promoted on validation point estimates alone.
- **NOT ADVANCED — G4**: market/SPY context adds no incremental value
  over G3 at any horizon.
- **G5 (neural)**: remains gated.

## Promotion rule (frozen before the rolling-origin study)

Advance G3 to untouched test evaluation only if, pooled over
chronological folds: pooled ΔQLIKE > 0 AND a clear majority of folds are
positive AND no major fold shows catastrophic degradation AND secondary
errors (MAE/RMSE) remain broadly supportive. No per-fold p < .05 required.

## Rolling-origin study (preregistered)

- Folds: calendar-year validation blocks 2019–2024, expanding train from
  2015, same purge semantics (labels may not cross the next boundary).
- Per fold × horizon: ΔQLIKE G3 vs G0, relative Δ, MAE/RMSE, HAC CI + p,
  % dates won, % tickers won, forecast/realized level ratio, tail QLIKE
  contribution, worst 1%/5%/10% dates.
- G3 configuration frozen (params, features, objective, 500 rounds).
  No tuning on fold results.
