# Empirical Volatility Forecasting Benchmark: Incremental News Signal Ablation (Phase 4)
**Date:** 2026-09-01T17:36:52Z | **Universe:** 44 Liquid Assets across 8 Sectors
**Base Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET` (25 features)
**Challenger Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET_PLUS_NEWS` (35 features)
**Target / Output Space:** `SOFTPLUS_VOLATILITY` (PyTorch LSTM + Regressors)

## Executive Summary & Core Hypothesis Test
- **Hypothesis Tested:** Does adding causal point-in-time financial news sentiment and intensity features provide statistically and practically meaningful incremental volatility forecasting skill beyond price, OHLC, and market context?
- **Causal Timestamp Safeguards:** Timezone-aware session market-close cutoff (16:00 America/New_York converted to UTC: 20:00 UTC during EDT, 21:00 UTC during EST). News features only consume articles published strictly prior to market close.
- **Experimental Discipline:** Strictly identical chronological 70/15/15 partitions, H-session purged boundary embargoes, and 44 assets across 8 market sectors.

## 0. News Corpus Coverage & Dataset Diagnostics
| Metric | Value | Note |
| :--- | :---: | :--- |
| **Upstream Provider Provenance** | Unknown / Unrecoverable | Upstream provider could not be reliably established from retained legacy artifacts |
| **Evaluated News Dataset** | Point-in-Time Headline Archive | Causal pre-filtered headline events with sentiment polarity scores |
| **Total Articles Evaluated** | ~232,000 | Corporate, earnings, regulatory & market headlines |
| **Assets with News Coverage** | 44 / 44 (100.0%) | 44 liquid assets across 8 market sectors |
| **Median Articles / Asset** | ~5,270 | Across 2,930 trading sessions (2015-01-02 to 2026-08-27) |
| **Median 1-Day Window Coverage** | 83.5% | Fraction of forecast origins with ≥1 headline in past 24h |
| **Median 3-Day Window Coverage** | 98.2% | Fraction of forecast origins with ≥1 headline in past 72h |
| **Median 7-Day Window Coverage** | 99.8% | Fraction of forecast origins with ≥1 headline in past 168h |
| **Date Range** | 2015-01-02 to 2026-08-27 | 11.6 years synchronized with market trading days |
| **Acquisition & Timestamp Filter** | Strict Causal Cutoff | Articles filtered strictly by published_at ≤ session_close_utc |
| **Exchange Session Calendar** | NYSE Calendar (mcal) | 16:00 ET (20:00/21:00 UTC); 13:00 ET (17:00/18:00 UTC early closes); fail-closed on non-sessions |
| **Sentiment Lexicon & Scoring** | VADER Financial Lexicon | Positive, negative, compound, dispersion, negative intensity |
| **Deduplication Method** | Exact Match Deterministic Filter | Duplicate records matching symbol, headline & timestamp removed |
| **Entity Matching Method** | Deterministic Universe Ticker Match | 100% of retained records matched valid target universe symbols |

## 1. Paired News Ablation Matrix (Base QLIKE vs +News QLIKE)
| Horizon | Model | Base QLIKE | +News QLIKE | Δ QLIKE | Rel Δ | Assets Improved | 95% Bootstrap CI | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Day | GARCH(1,1) | 1.8581 | 1.8581 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 1-Day | Rolling Mean (60d) | 1.8895 | 1.8895 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 1-Day | Gradient Boosting | 168.2886 | 382.2712 | -213.9826 | -127.15% | 24/44 (54.5%) | [-961.4006, +288.8341] | Degradation / Noise |
| 1-Day | Elastic Net | 5852189.9833 | 87623444.2972 | -81771254.3139 | -1397.28% | 1/44 (2.3%) | [-245313754.8609, -2.9633] | Degradation / Noise |
| 1-Day | PyTorch LSTM | 2.8679 | 2.8205 | +0.0474 | +1.65% | 19/44 (43.2%) | [-0.0273, +0.1655] | Non-Significant Gain |
| 5-Day | GARCH(1,1) | 0.7310 | 0.7310 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 5-Day | Rolling Mean (60d) | 0.7206 | 0.7206 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 5-Day | Gradient Boosting | 1.7886 | 1.7489 | +0.0398 | +2.22% | 26/44 (59.1%) | [-0.0124, +0.1273] | Non-Significant Gain |
| 5-Day | Elastic Net | 252.8856 | 576.9173 | -324.0316 | -128.13% | 8/44 (18.2%) | [-971.9959, -0.0361] | Degradation / Noise |
| 5-Day | PyTorch LSTM | 1.3349 | 1.3335 | +0.0014 | +0.11% | 22/44 (50.0%) | [-0.0196, +0.0219] | Non-Significant Gain |
| 10-Day | GARCH(1,1) | 0.5984 | 0.5984 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 10-Day | Rolling Mean (60d) | 0.5707 | 0.5707 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 10-Day | Gradient Boosting | 1.2117 | 1.2426 | -0.0309 | -2.55% | 23/44 (52.3%) | [-0.0999, +0.0071] | Degradation / Noise |
| 10-Day | Elastic Net | 246.1283 | 1240.2081 | -994.0798 | -403.89% | 9/44 (20.5%) | [-2982.1949, -0.0154] | Degradation / Noise |
| 10-Day | PyTorch LSTM | 1.2519 | 1.1397 | +0.1121 | +8.96% | 20/44 (45.5%) | [-0.0139, +0.3533] | Non-Significant Gain |
| 20-Day | GARCH(1,1) | 0.6592 | 0.6592 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 20-Day | Rolling Mean (60d) | 0.5776 | 0.5776 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 20-Day | Gradient Boosting | 0.8583 | 0.8909 | -0.0327 | -3.81% | 21/44 (47.7%) | [-0.1054, +0.0054] | Degradation / Noise |
| 20-Day | Elastic Net | 10.3576 | 12.9948 | -2.6372 | -25.46% | 13/44 (29.5%) | [-7.8881, -0.0067] | Degradation / Noise |
| 20-Day | PyTorch LSTM | 0.9792 | 0.8952 | +0.0840 | +8.58% | 20/44 (45.5%) | [-0.0075, +0.2541] | Non-Significant Gain |

## 2. Sector Breadth Breakdown (Count of Assets Improved by Horizon)
| Sector | Universe Assets | 1-Day Imprv | 5-Day Imprv | 10-Day Imprv | 20-Day Imprv |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mega-Cap Tech / Growth | 6 | 2/6 | 2/6 | 2/6 | 3/6 |
| Broad & Tech ETFs | 2 | 1/2 | 1/2 | 1/2 | 1/2 |
| Financials & Fintech | 3 | 0/3 | 0/3 | 0/3 | 1/3 |
| Industrials & Logistics | 7 | 4/7 | 3/7 | 3/7 | 3/7 |
| Healthcare & Biotech | 7 | 5/7 | 6/7 | 5/7 | 4/7 |
| Consumer Staples & Discretionary | 7 | 5/7 | 6/7 | 6/7 | 6/7 |
| Energy & Utilities | 6 | 1/6 | 0/6 | 0/6 | 0/6 |
| High-Beta / High-Vol | 6 | 1/6 | 4/6 | 3/6 | 2/6 |

## 3. Empirical Verdict & Scientific Conclusion
- **Scientific Finding:** **The tested causal news feature set did not add robust out-of-sample forecasting skill on this dataset.** Across multi-day horizons (5d, 10d, 20d), aggregate QLIKE improvements are non-significant, 95% asset-level bootstrap confidence intervals consistently include zero or negative territory, and asset-level win rates do not achieve a convincing majority.
- **Strategic Decision:** **REMOVE NEWS SIGNAL FROM ACTIVE PRODUCTION FORECASTING.**
- **Production Architecture Rationale:** Classical volatility structure with causal OHLC range estimators (Parkinson, Garman-Klass, Rogers-Satchell) and market context provides a parsimonious, robust, and empirically superior forecasting core without external news latency or feature noise.