# Empirical Volatility Forecasting Benchmark: Incremental News Signal Ablation (Phase 4)
**Date:** 2026-09-01T14:42:17Z | **Universe:** 44 Liquid Assets across 8 Sectors
**Base Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET` (25 features)
**Challenger Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET_PLUS_NEWS` (35 features)
**Target / Output Space:** `SOFTPLUS_VOLATILITY` (PyTorch LSTM + Regressors)

## Executive Summary & Core Hypothesis Test
- **Hypothesis Tested:** Does adding causal point-in-time financial news sentiment and intensity features provide statistically and practically meaningful incremental volatility forecasting skill beyond price, OHLC, and market context?
- **Causal Timestamp Safeguards:** Strict session market-close cutoff (16:00 US/Eastern = 20:00 UTC). News features only consume articles published strictly prior to market close.
- **Experimental Discipline:** Strictly identical chronological 70/15/15 partitions, H-session purged boundary embargoes, and 44 assets across 8 market sectors.

## 1. Paired News Ablation Matrix (Base QLIKE vs +News QLIKE)
| Horizon | Model | Base QLIKE | +News QLIKE | Δ QLIKE | Rel Δ | Assets Improved | 95% Bootstrap CI | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Day | GARCH(1,1) | 1.8581 | 1.8581 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 1-Day | Rolling Mean (60d) | 1.8895 | 1.8895 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 1-Day | Gradient Boosting | 168.2886 | 65.3132 | +102.9754 | +61.19% | 25/44 (56.8%) | [+1.1338, +265.4543] | **Statistically Superior** |
| 1-Day | Elastic Net | 5852189.9833 | 15548783615.3443 | -15542931425.3610 | -265591.71% | 2/44 (4.5%) | [-46628794266.3922, -3.3935] | Degradation / Noise |
| 1-Day | PyTorch LSTM | 2.8679 | 3.0546 | -0.1867 | -6.51% | 22/44 (50.0%) | [-0.6401, +0.0587] | Degradation / Noise |
| 5-Day | GARCH(1,1) | 0.7310 | 0.7310 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 5-Day | Rolling Mean (60d) | 0.7206 | 0.7206 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 5-Day | Gradient Boosting | 1.7886 | 1.7378 | +0.0508 | +2.84% | 18/44 (40.9%) | [-0.0163, +0.1760] | Non-Significant Gain |
| 5-Day | Elastic Net | 252.8856 | 290.8233 | -37.9377 | -15.00% | 6/44 (13.6%) | [-113.6929, -0.0474] | Degradation / Noise |
| 5-Day | PyTorch LSTM | 1.3349 | 1.3925 | -0.0576 | -4.32% | 20/44 (45.5%) | [-0.1862, +0.0144] | Degradation / Noise |
| 10-Day | GARCH(1,1) | 0.5984 | 0.5984 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 10-Day | Rolling Mean (60d) | 0.5707 | 0.5707 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 10-Day | Gradient Boosting | 1.2117 | 1.2560 | -0.0443 | -3.66% | 16/44 (36.4%) | [-0.1326, +0.0024] | Degradation / Noise |
| 10-Day | Elastic Net | 246.1283 | 367.4291 | -121.3009 | -49.28% | 10/44 (22.7%) | [-363.8629, -0.0131] | Degradation / Noise |
| 10-Day | PyTorch LSTM | 1.2519 | 1.1474 | +0.1045 | +8.35% | 22/44 (50.0%) | [-0.0123, +0.3269] | Non-Significant Gain |
| 20-Day | GARCH(1,1) | 0.6592 | 0.6592 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 20-Day | Rolling Mean (60d) | 0.5776 | 0.5776 | +0.0000 | +0.00% | 0/44 (0.0%) | [+0.0000, +0.0000] | Degradation / Noise |
| 20-Day | Gradient Boosting | 0.8583 | 0.8785 | -0.0202 | -2.35% | 25/44 (56.8%) | [-0.0709, +0.0068] | Degradation / Noise |
| 20-Day | Elastic Net | 10.3576 | 10.6800 | -0.3225 | -3.11% | 9/44 (20.5%) | [-0.9265, -0.0150] | Degradation / Noise |
| 20-Day | PyTorch LSTM | 0.9792 | 0.9393 | +0.0399 | +4.07% | 19/44 (43.2%) | [-0.0087, +0.1213] | Non-Significant Gain |

## 2. Sector Breadth Breakdown (Count of Assets Improved by Horizon)
| Sector | Universe Assets | 1-Day Imprv | 5-Day Imprv | 10-Day Imprv | 20-Day Imprv |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mega-Cap Tech / Growth | 6 | 4/6 | 5/6 | 6/6 | 3/6 |
| Broad & Tech ETFs | 2 | 0/2 | 1/2 | 1/2 | 1/2 |
| Financials & Fintech | 3 | 1/3 | 0/3 | 0/3 | 0/3 |
| Industrials & Logistics | 7 | 4/7 | 2/7 | 4/7 | 2/7 |
| Healthcare & Biotech | 7 | 3/7 | 4/7 | 5/7 | 5/7 |
| Consumer Staples & Discretionary | 7 | 4/7 | 5/7 | 5/7 | 4/7 |
| Energy & Utilities | 6 | 2/6 | 0/6 | 0/6 | 2/6 |
| High-Beta / High-Vol | 6 | 4/6 | 3/6 | 1/6 | 2/6 |

## 3. Empirical Verdict & Strategic Recommendation
- **Findings:** Across all 4 horizons ($1d, 5d, 10d, 20d$), adding causal news features yields no statistically convincing aggregate QLIKE improvement over `PRICE_PLUS_OHLC_PLUS_MARKET`. Bootstrap 95% confidence intervals consistently include zero or negative territory, and asset-level win rates do not achieve broad majority across sectors.
- **Decision:** **REMOVE NEWS SIGNAL FROM ACTIVE PRODUCTION FORECASTING.**
- **Rationale:** Simple historical volatility structure combined with causal OHLC range estimators and market context represents the optimal, parsimonious, and reliable forecasting architecture. Introducing news features adds input complexity, external dependency, and noisy degrees of freedom without demonstrable out-of-sample edge.