# OHLCV Multi-Horizon Alpha Evaluation — Completed Negative Study

Frozen: 2026-09-05. Tag: `v1.0-ohlcv-negative-study`. Validation-only throughout:
no test scoring, no deployment refit, no production changes.

## Scope

- 286-stock tri-exchange universe (195 US, 91 UK), horizons 1–7 trading days.
- Frozen chronological splits with purged training tail
  ($t \le T_{\text{train}} - H$); Bartlett HAC inference with bandwidths
  scaling in horizon ($L \in \{h-1, 2(h-1), 3(h-1)\}$).
- Ridge $\alpha = 100.0$, training-only StandardScaler, fixed comparators
  (no-change persistence, training-majority direction).

## Recorded nulls

1. **Absolute returns (1–7d):** Ridge $\approx$ LSTM $\approx$ persistence
   (relative MAE $\approx 0.999$; direction $\le$ majority drift).
   `artifacts/price_validation_comparison_20260905_003731/`
2. **Cross-sectional context:** universe aggregates degraded MAE
   (2.7089% $\to$ 2.7126% Ridge, 2.7192% LSTM).
   `artifacts/market_context_comparison_20260905_010222/`
3. **Macro news:** decade-scale gated SPY sentiment (66,432 articles,
   6h revision gate, 0.28% discard) worsened full-US validation
   (MAE 2.7822% $\to$ 2.8218%, rel 1.011; direction 0.5260 $\to$ 0.5053),
   including the March 2023 SVB stress slice.
   `artifacts/macro_ridge_2026-09-05/`
4. **Beta-neutral residual ranks:** trailing-60d beta vs SPY/^FTSE
   (45-session gate, holiday-masked outcomes); mean |IC| < 0.015,
   $p_{\text{HAC}} \in [0.19, 0.37]$ across all scopes and horizons.
   `artifacts/residual_rank_2026-09-05/`

Related negative replications: 21-day SPY regime/XGBoost
(`artifacts/spy_21d_regime_2026-09-05/`, XGB direction 0.554 vs 0.552
base rate, HAC p=0.585); 5-ticker March-2023 news pilot and 21-ticker
stratified pull establishing 53.8% 1-day ticker-news sparsity.

## Preserved assets

- Revision-gating engine: `research/price_forecasting/news_archive.py`
  (`t_available`, 3-tier 6h policy, `build_macro_news_features`).
- Macro concat mode: `price_plus_macro` (US-only guard) in
  `research/price_forecasting/gpu_pipeline.py`.
- HAC harness: `research/price_forecasting/paired_validation.py`.
- Point-in-time caches: `data/news/macro_spy_full/`,
  `data/news/stratified_202303/`, `data/news/macro_spy_202303/`,
  `data/macro/market_dailies.parquet`, `data/macro/ftse_dailies.parquet`.
- Runners: `scripts/benchmark_macro_ridge.py`,
  `scripts/train_spy_21d_regime.py`, `scripts/run_residual_rank_experiment.py`,
  `scripts/run_rank_experiment.py`, `scripts/build_market_macro_cache.py`.

## Operational decision

Retire OHLCV technical features for short-horizon alpha generation on this
universe. New problem classes branch from this tag.
