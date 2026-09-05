# Fixed 25-stock news replication

Declared 2026-09-05 after inspecting MSFT, before examining the following replication results. MSFT is excluded. This is an exploratory replication using surviving securities, not an independently untouched certification sample.

Fixed roster: ADBE, AMD, CSCO, IBM, JPM, BAC, GS, AXP, JNJ, MRK, ABT, GILD, CAT, DE, HON, LMT, CVX, COP, EOG, KO, COST, MCD, HD, DIS, AEP.

The roster spans technology, financials, healthcare, industrials, energy, consumer businesses, communications and utilities. It is a deliberate convenience sample, not a sector-weighted or point-in-time representative universe. No replacements following missing coverage or disappointing results.

Use the existing MSFT implementation: per-security chronological 70/15/15 eligible-origin split, purged label boundaries; 5/10/20-session mean future squared log-return targets; the same 25 OHLCV and ten news features, six-hour revision exclusion and updated-time availability rules. Fit only training data. No tuning or test-driven refit. Per-security split dates may differ and must be reported.

XGBoost CUDA: 200 trees, depth 3, learning rate 0.03, lambda 10, minimum child weight 20, full row/column sampling, seed 42. Preserve OHLCV-only, activity and sentiment arms and both rolling references.

Primary replication contrast: C_sentiment minus A_ohlcv at five sessions, expressed as positive loss improvement A minus C. Ten/twenty sessions and activity are secondary. Report all assets, failures and missing coverage; no selected-winner summary.

Report equal-asset mean and median improvement, improvement counts, MAE/RMSE tradeoffs, and existing per-asset HAC intervals at H-1, 2(H-1), 3(H-1). For panel inference preserve common-date dependence: aggregate matched asset loss differences by date before HAC; do not treat the 25 assets as independent replications. Report sensitivity to changing date-level asset coverage. Apply Holm across the three sentiment horizons for panel inference. These are exploratory analyses, not guarantees of future skill.

MSFT artifacts are read-only inputs to this decision and must not be overwritten or retuned. No SEC, FinBERT, event, macro or architecture additions in this replication. No production promotion.
