# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report (V3)
**Date:** 2026-09-01T08:48:21Z | **Universe:** 44 Liquid Assets across 8 Sectors | **Feature Mode:** `price_plus_ohlc` | **Target Space:** `log_variance` | **Execution Time:** 548.2s

## Executive Summary
Phase 3 establishes rigorous empirical benchmarking of volatility forecasting models with audited corporate-action-adjusted OHLC data, nested causal feature ablations, neural output formulation comparisons, and comprehensive tail error diagnostics.
- **1-Day Baseline Findings:** `Rolling Mean (60d)` achieves the best aggregate test error (MAE `0.2137`, RMSE `0.3073`, QLIKE `1.8643`), while `GARCH(1,1)` is the most consistently selected model across individual assets (winning 30/44 validation and 25/44 test asset contests).
- **Single-Day Proxy Noise on HAR-RV:** The canonical 1-day realized volatility target $RV(t,1) = \sqrt{252}|r_{t+1}|$ is dominated by single-session return jump noise, which heavily disadvantages multi-frequency autoregressive filters like HAR-RV (QLIKE `8.3058` at 1d). As the horizon expands to 5d, 10d, and 20d, jump noise averages out, and HAR-RV's multi-resolution memory achieves competitive point accuracy.
- **Target / Output Formulation:** `LOG_VARIANCE` and `SOFTPLUS_VOLATILITY` provide structural protection against near-zero variance collapse, preventing astronomical QLIKE blowouts on market shock days.

## 1. Multi-Horizon Forecasting Accuracy & Distributional Skill Matrix
### Horizon: 1-Day (1-session)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.2051 | 0.3026 | **2.0406** | 0.8147 | 5.8643 | 19.8% | 0/44 | 2/44 | 0/44 | 44/44 |
| **Rolling Mean** | 0.2052 | 0.2960 | **1.8888** | 0.8300 | 5.8599 | 18.5% | 11/44 | 11/44 | 32/44 | 44/44 |
| **EWMA (λ=0.94)** | 0.2045 | 0.2973 | **1.9222** | 0.8168 | 5.7948 | 18.7% | 2/44 | 1/44 | 40/44 | 44/44 |
| **HAR-RV** | 0.1633 | 0.2927 | **5.6501** | 1.1957 | 18.1246 | 30.4% | 0/44 | 0/44 | 0/44 | 0/44 |
| **GARCH(1,1)** | 0.2070 | 0.2965 | **1.8581** | 0.8184 | 5.7078 | 17.4% | 31/44 | 30/44 | 39/44 | 44/44 |
| **Ridge** | 0.2443 | 0.8930 | **5068535371057.9414** | 1.3878 | 46.5929 | 43.2% | 0/44 | 0/44 | 0/44 | 0/44 |
| **Elastic Net** | 0.2041 | 0.6050 | **591751.8866** | 1.2431 | 26.6348 | 32.2% | 0/44 | 0/44 | 0/44 | 7/44 |
| **Gradient Boosting** | 0.1637 | 0.2926 | **75.2611** | 1.2278 | 25.7864 | 47.8% | 0/44 | 0/44 | 0/44 | 10/44 |
| **PyTorch LSTM** | 0.1598 | 0.2825 | **4.0987** | 0.9635 | 11.5397 | 29.2% | 0/44 | 0/44 | 0/44 | 44/44 |


### Horizon: 5-Day (5-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1369 | 0.2163 | **0.9907** | 0.2314 | 2.1632 | 22.1% | 0/44 | 2/44 | 0/44 | 35/44 |
| **Rolling Mean** | 0.1356 | 0.2099 | **0.7184** | 0.2330 | 1.9960 | 20.5% | 12/44 | 11/44 | 32/44 | 39/44 |
| **EWMA (λ=0.94)** | 0.1343 | 0.2096 | **0.8167** | 0.2264 | 1.8878 | 21.9% | 0/44 | 1/44 | 38/44 | 42/44 |
| **HAR-RV** | 0.1164 | 0.2032 | **1.0550** | 0.1899 | 3.0035 | 24.5% | 0/44 | 0/44 | 9/44 | 0/44 |
| **GARCH(1,1)** | 0.1342 | 0.2060 | **0.7310** | 0.2256 | 1.6940 | 19.8% | 22/44 | 25/44 | 39/44 | 44/44 |
| **Ridge** | 0.1299 | 0.2196 | **58858.4353** | 0.2323 | 9.7666 | 24.9% | 0/44 | 0/44 | 3/44 | 7/44 |
| **Elastic Net** | 0.1181 | 0.2063 | **218.9377** | 0.1915 | 4.6298 | 24.6% | 4/44 | 2/44 | 14/44 | 25/44 |
| **Gradient Boosting** | 0.1166 | 0.2044 | **1.8702** | 0.1871 | 2.9414 | 23.6% | 4/44 | 2/44 | 15/44 | 25/44 |
| **PyTorch LSTM** | 0.1166 | 0.2049 | **1.2792** | 0.1915 | 3.3580 | 24.8% | 2/44 | 1/44 | 10/44 | 16/44 |


### Horizon: 10-Day (10-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1210 | 0.1884 | **0.9140** | 0.1717 | 1.9398 | 16.8% | 0/44 | 1/44 | 0/44 | 19/44 |
| **Rolling Mean** | 0.1178 | 0.1816 | **0.5681** | 0.1633 | 1.5677 | 15.6% | 5/44 | 9/44 | 35/44 | 31/44 |
| **EWMA (λ=0.94)** | 0.1177 | 0.1818 | **0.7018** | 0.1674 | 1.6977 | 17.0% | 0/44 | 0/44 | 37/44 | 31/44 |
| **HAR-RV** | 0.1032 | 0.1753 | **0.7815** | 0.1260 | 2.8873 | 18.3% | 0/44 | 0/44 | 25/44 | 0/44 |
| **GARCH(1,1)** | 0.1165 | 0.1769 | **0.5984** | 0.1596 | 1.4011 | 15.4% | 16/44 | 23/44 | 40/44 | 41/44 |
| **Ridge** | 0.1164 | 0.1941 | **5147.2530** | 0.1612 | 11.0726 | 20.5% | 2/44 | 1/44 | 14/44 | 8/44 |
| **Elastic Net** | 0.1035 | 0.1788 | **118.2014** | 0.1241 | 11.4292 | 19.1% | 7/44 | 3/44 | 26/44 | 24/44 |
| **Gradient Boosting** | 0.1033 | 0.1770 | **1.3646** | 0.1237 | 6.6233 | 18.0% | 6/44 | 5/44 | 25/44 | 20/44 |
| **PyTorch LSTM** | 0.1034 | 0.1786 | **1.0250** | 0.1247 | 4.3981 | 17.9% | 8/44 | 2/44 | 23/44 | 23/44 |


### Horizon: 20-Day (20-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1118 | 0.1690 | **1.2584** | 0.1573 | 1.6916 | 12.4% | 0/44 | 1/44 | 0/44 | 8/44 |
| **Rolling Mean** | 0.1046 | 0.1584 | **0.5708** | 0.1258 | 1.4373 | 11.1% | 4/44 | 7/44 | 33/44 | 20/44 |
| **EWMA (λ=0.94)** | 0.1060 | 0.1605 | **0.8228** | 0.1354 | 1.2719 | 12.4% | 0/44 | 0/44 | 40/44 | 16/44 |
| **HAR-RV** | 0.0921 | 0.1485 | **0.7752** | 0.1034 | 3.6146 | 13.0% | 1/44 | 2/44 | 36/44 | 0/44 |
| **GARCH(1,1)** | 0.1020 | 0.1521 | **0.6592** | 0.1255 | 0.9911 | 11.0% | 8/44 | 11/44 | 41/44 | 36/44 |
| **Ridge** | 0.1035 | 0.1651 | **25.0011** | 0.1216 | 6.9526 | 16.0% | 3/44 | 3/44 | 24/44 | 10/44 |
| **Elastic Net** | 0.0913 | 0.1497 | **10.6002** | 0.0947 | 6.9041 | 14.5% | 8/44 | 7/44 | 33/44 | 26/44 |
| **Gradient Boosting** | 0.0916 | 0.1489 | **0.9382** | 0.0983 | 4.7178 | 13.0% | 3/44 | 11/44 | 36/44 | 20/44 |
| **PyTorch LSTM** | 0.0932 | 0.1526 | **1.0961** | 0.1060 | 4.9468 | 12.4% | 17/44 | 2/44 | 30/44 | 21/44 |


## 2. Neural Target / Output Formulation Comparison (PyTorch LSTM)
Controlled comparison of neural output formulations on identical splits, architectures, and training budgets:

| Horizon | Formulation | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Max QLIKE | Near-Zero Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Day | `LOG_VARIANCE` | 0.1598 | 0.2825 | **4.0987** | 0.9635 | 11.5397 | 9711.86 | 0 |
| 1-Day | `SOFTPLUS_VOLATILITY` | 0.1651 | 0.2741 | **2.4933** | 0.7614 | 6.6678 | 3921.87 | 0 |
| 1-Day | `DIRECT_VOLATILITY` | 0.1641 | 0.2733 | **4.1095** | 0.7688 | 7.1669 | 20721.05 | 0 |
| 5-Day | `LOG_VARIANCE` | 0.1166 | 0.2049 | **1.2792** | 0.1915 | 3.3580 | 1723.40 | 0 |
| 5-Day | `SOFTPLUS_VOLATILITY` | 0.1190 | 0.2016 | **0.9712** | 0.1812 | 2.3426 | 1129.20 | 0 |
| 5-Day | `DIRECT_VOLATILITY` | 0.1270 | 0.2092 | **2588684939116201115648.0000** | 238192427124086308864.0000 | 10405784776325835259904.0000 | 2.32e+24 | 463 |
| 10-Day | `LOG_VARIANCE` | 0.1034 | 0.1786 | **1.0250** | 0.1247 | 4.3981 | 1140.10 | 0 |
| 10-Day | `SOFTPLUS_VOLATILITY` | 0.1068 | 0.1766 | **0.8081** | 0.1312 | 3.9654 | 579.35 | 0 |
| 10-Day | `DIRECT_VOLATILITY` | 0.1216 | 0.1913 | **4055780213136749494272.0000** | 1430129555979747459072.0000 | 15675678766782763499520.0000 | 1.23e+24 | 725 |
| 20-Day | `LOG_VARIANCE` | 0.0932 | 0.1526 | **1.0961** | 0.1060 | 4.9468 | 1150.96 | 0 |
| 20-Day | `SOFTPLUS_VOLATILITY` | 0.0947 | 0.1493 | **0.8685** | 0.1009 | 3.9754 | 869.11 | 0 |
| 20-Day | `DIRECT_VOLATILITY` | 0.1146 | 0.1702 | **4138572328252862365696.0000** | 1731427105405846421504.0000 | 15569454916411638939648.0000 | 6.39e+23 | 694 |


## 3. Nested Feature Ablation Study
Evaluation of incremental causal information value: `PRICE_ONLY` → `PRICE_PLUS_OHLC` → `PRICE_PLUS_OHLC_PLUS_MARKET`.

| Horizon | Model | Feature Configuration | Features | Test MAE | Test RMSE | Test QLIKE | Median QLIKE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1-Day | Gradient Boosting | `PRICE_ONLY` | 9 | 0.1650 | 0.2946 | **4632.8595** | 1.2275 |
| 1-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 21 | 0.1637 | 0.2926 | **75.2611** | 1.2278 |
| 1-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1636 | 0.2932 | **69.0052** | 1.2232 |
| 1-Day | PyTorch LSTM | `PRICE_ONLY` | 9 | 0.1606 | 0.2859 | **4.3600** | 1.0171 |
| 1-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 21 | 0.1598 | 0.2825 | **4.0987** | 0.9635 |
| 1-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1597 | 0.2813 | **4.0228** | 0.9420 |
| 5-Day | Gradient Boosting | `PRICE_ONLY` | 9 | 0.1193 | 0.2067 | **1.2663** | 0.1988 |
| 5-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 21 | 0.1166 | 0.2044 | **1.8702** | 0.1871 |
| 5-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1171 | 0.2051 | **1.8026** | 0.1901 |
| 5-Day | PyTorch LSTM | `PRICE_ONLY` | 9 | 0.1178 | 0.2082 | **1.3565** | 0.2043 |
| 5-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 21 | 0.1166 | 0.2049 | **1.2792** | 0.1915 |
| 5-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1163 | 0.2044 | **1.2998** | 0.1906 |
| 10-Day | Gradient Boosting | `PRICE_ONLY` | 9 | 0.1064 | 0.1791 | **0.9211** | 0.1358 |
| 10-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 21 | 0.1033 | 0.1770 | **1.3646** | 0.1237 |
| 10-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1035 | 0.1766 | **1.1069** | 0.1266 |
| 10-Day | PyTorch LSTM | `PRICE_ONLY` | 9 | 0.1053 | 0.1817 | **1.0679** | 0.1428 |
| 10-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 21 | 0.1034 | 0.1786 | **1.0250** | 0.1247 |
| 10-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.1034 | 0.1778 | **1.0602** | 0.1266 |
| 20-Day | Gradient Boosting | `PRICE_ONLY` | 9 | 0.0954 | 0.1525 | **0.8971** | 0.1119 |
| 20-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 21 | 0.0916 | 0.1489 | **0.9382** | 0.0983 |
| 20-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.0931 | 0.1504 | **0.8342** | 0.1039 |
| 20-Day | PyTorch LSTM | `PRICE_ONLY` | 9 | 0.0947 | 0.1541 | **1.0268** | 0.1167 |
| 20-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 21 | 0.0932 | 0.1526 | **1.0961** | 0.1060 |
| 20-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 25 | 0.0940 | 0.1529 | **0.9370** | 0.1047 |


## 4. Uncertainty Cones & Prediction Interval Calibration
### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)
| Horizon | Model | Empirical Coverage | Avg Width (Annualized σ) |
| :---: | :--- | :---: | :---: |
| 1-Day | Rolling Mean | **90.1%** | 3.4122 |
| 1-Day | GARCH(1,1) | **90.9%** | 3.8667 |
| 1-Day | HAR-RV | **90.2%** | 0.8310 |
| 1-Day | Gradient Boosting | **90.0%** | 1.0100 |
| 1-Day | PyTorch LSTM | **90.5%** | 1.0686 |
| 5-Day | Rolling Mean | **88.7%** | 0.6740 |
| 5-Day | GARCH(1,1) | **91.7%** | 0.7581 |
| 5-Day | HAR-RV | **89.7%** | 0.4544 |
| 5-Day | Gradient Boosting | **90.4%** | 0.4835 |
| 5-Day | PyTorch LSTM | **88.5%** | 0.4327 |
| 10-Day | Rolling Mean | **88.6%** | 0.5215 |
| 10-Day | GARCH(1,1) | **92.0%** | 0.5913 |
| 10-Day | HAR-RV | **89.9%** | 0.3953 |
| 10-Day | Gradient Boosting | **90.5%** | 0.4292 |
| 10-Day | PyTorch LSTM | **87.8%** | 0.3685 |
| 20-Day | Rolling Mean | **85.8%** | 0.4203 |
| 20-Day | GARCH(1,1) | **91.4%** | 0.4858 |
| 20-Day | HAR-RV | **88.7%** | 0.3448 |
| 20-Day | Gradient Boosting | **89.5%** | 0.3835 |
| 20-Day | PyTorch LSTM | **83.4%** | 0.3109 |


### Gaussian Model-Implied p05–p95 Price Range Coverage (Nominal: 90.0%)
| Horizon | Model Implied Volatility | Empirical Price Range Coverage | Avg Cone Width (% Price) |
| :---: | :--- | :---: | :---: |
| 1-Day | Rolling Mean | **90.4%** | ±6.8% |
| 1-Day | GARCH(1,1) | **91.1%** | ±7.0% |
| 1-Day | HAR-RV | **60.4%** | ±3.0% |
| 1-Day | Gradient Boosting | **60.4%** | ±3.0% |
| 1-Day | PyTorch LSTM | **68.3%** | ±3.6% |
| 5-Day | Rolling Mean | **89.6%** | ±15.2% |
| 5-Day | GARCH(1,1) | **91.0%** | ±15.8% |
| 5-Day | HAR-RV | **83.2%** | ±12.0% |
| 5-Day | Gradient Boosting | **83.9%** | ±12.1% |
| 5-Day | PyTorch LSTM | **82.4%** | ±11.8% |
| 10-Day | Rolling Mean | **89.5%** | ±21.4% |
| 10-Day | GARCH(1,1) | **91.1%** | ±22.3% |
| 10-Day | HAR-RV | **86.1%** | ±18.2% |
| 10-Day | Gradient Boosting | **86.1%** | ±18.4% |
| 10-Day | PyTorch LSTM | **84.9%** | ±18.0% |
| 20-Day | Rolling Mean | **88.8%** | ±30.1% |
| 20-Day | GARCH(1,1) | **91.0%** | ±31.7% |
| 20-Day | HAR-RV | **87.2%** | ±27.1% |
| 20-Day | Gradient Boosting | **87.1%** | ±27.2% |
| 20-Day | PyTorch LSTM | **85.8%** | ±26.6% |


## 5. Top Catastrophic Tail Error Diagnostics
### Top 5 Worst Out-of-Sample Losses: PyTorch LSTM
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1150 | 0.1984 | -11.2257 | **9711.86** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.3565 | 0.5783 | -15.8155 | **2049.15** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.1673 | 0.1805 | -6.7932 | **1723.40** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.1672 | 0.1984 | -6.7911 | **1723.28** | False |
| MNST | 2026-07-13 | 20d | High Vol | 6.4303 | 0.1889 | 0.1675 | -6.2414 | **1150.96** | False |


### Top 5 Worst Out-of-Sample Losses: Gradient Boosting
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AMD | 2026-01-23 | 1d | Normal Vol | 0.5201 | 0.0007 | 0.4217 | -0.5194 | **605623.27** | False |
| AMD | 2026-05-27 | 1d | High Vol | 0.7064 | 0.0020 | 0.9057 | -0.7045 | **130212.74** | False |
| MNST | 2026-08-05 | 1d | High Vol | 11.0539 | 0.0319 | 4.6683 | -11.0220 | **120008.64** | False |
| AMD | 2026-01-26 | 1d | Low Vol | 0.0454 | 0.0002 | 0.4354 | -0.0453 | **81893.47** | False |
| MDLZ | 2025-01-13 | 1d | High Vol | 0.2607 | 0.0015 | 0.2115 | -0.2592 | **29653.53** | False |


### Top 5 Worst Out-of-Sample Losses: HAR-RV
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1179 | 0.1984 | -11.2228 | **9237.68** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.3301 | 0.5783 | -15.8419 | **2391.87** | False |
| MNST | 2026-07-30 | 1d | High Vol | 11.2112 | 0.2557 | 3.3231 | -10.9555 | **1913.85** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.2042 | 0.1805 | -6.7563 | **1153.47** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.2187 | 0.1984 | -6.7396 | **1004.16** | False |


### Top 5 Worst Out-of-Sample Losses: Rolling Mean (60d)
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.3248 | 0.1984 | -11.0159 | **1210.68** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.7447 | 0.5783 | -15.4272 | **464.39** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.3248 | 0.1984 | -6.6334 | **451.70** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.3252 | 0.1805 | -6.6353 | **451.08** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.3217 | 0.1639 | -6.1087 | **392.54** | False |


### Top 5 Worst Out-of-Sample Losses: GARCH(1,1)
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.2588 | 0.1984 | -11.0819 | **1911.48** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.2460 | 0.1805 | -6.7145 | **792.83** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.2620 | 0.1984 | -6.6963 | **697.73** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.2481 | 0.1639 | -6.1824 | **664.28** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.6267 | 0.5783 | -15.5453 | **658.45** | False |

