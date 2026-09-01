# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report (V2)
**Date:** 2026-08-31T23:38:39Z | **Universe:** 44 Liquid Assets across 8 Sectors | **Target Space:** `log_variance` | **Execution Time:** 240.4s

## Executive Summary
This empirical study evaluates the predictive accuracy and uncertainty calibration of 9 volatility forecasting models across 4 horizons (1-day, 5-day, 10-day, and 20-day) using strict chronological 70/15/15 splits with horizon-length boundary embargoes.
Phase 2 incorporates canonical GARCH(1,1), target formulation analysis (Direct Volatility vs Log-Variance), metric numerical stabilization audits, and worst-error tail diagnostics.

## 1. Multi-Horizon Forecasting Accuracy & Skill Matrix
### Horizon: 1-Day (1-session)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins | Raw Min Pred | Near-Zero Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.2214 | 0.3189 | **2.2432** | — | +72.99% | 4/44 | 1/44 | 0.062263 | 0 |
| **Rolling Mean** | 0.2137 | 0.3073 | **1.8643** | +16.89% | +77.55% | 6/44 | 16/44 | 0.087747 | 0 |
| **EWMA (λ=0.94)** | 0.2186 | 0.3131 | **2.0291** | +9.54% | +75.57% | 4/44 | 2/44 | 0.073901 | 0 |
| **HAR-RV** | 0.1808 | 0.3192 | **8.3058** | -270.27% | — | 0/44 | 0/44 | 0.030926 | 0 |
| **GARCH(1,1)** | 0.2221 | 0.3138 | **1.9622** | +12.53% | +76.38% | 30/44 | 25/44 | 0.092974 | 0 |
| **Ridge** | 0.4747 | 1.3725 | **7812404604655297.0000** | -348276684996172160.00% | -94059877889388928.00% | 0/44 | 0/44 | 0.000000 | 6 |
| **Elastic Net** | 0.3387 | 0.9490 | **3632941921712.0093** | -161956405652972.16% | -43739935503477.41% | 0/44 | 0/44 | 0.000000 | 3 |
| **Gradient Boosting** | 0.1877 | 0.3243 | **1613.3562** | -71823.35% | -19324.50% | 0/44 | 0/44 | 0.000509 | 0 |
| **PyTorch LSTM** | 0.1777 | 0.3103 | **5.6461** | -151.70% | +32.02% | 0/44 | 0/44 | 0.054520 | 0 |


### Horizon: 5-Day (5-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins | Raw Min Pred | Near-Zero Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1437 | 0.2193 | **1.4050** | — | +47.69% | 1/44 | 1/44 | 0.062263 | 0 |
| **Rolling Mean** | 0.1361 | 0.2088 | **0.7562** | +46.17% | +71.84% | 4/44 | 12/44 | 0.087747 | 0 |
| **EWMA (λ=0.94)** | 0.1390 | 0.2122 | **1.0355** | +26.30% | +61.45% | 1/44 | 2/44 | 0.073901 | 0 |
| **HAR-RV** | 0.1311 | 0.2213 | **2.6858** | -91.17% | — | 1/44 | 1/44 | 0.064214 | 0 |
| **GARCH(1,1)** | 0.1394 | 0.2107 | **0.9429** | +32.89% | +64.89% | 12/44 | 25/44 | 0.097135 | 0 |
| **Ridge** | 0.2035 | 0.5835 | **25033210734.2527** | -1781768383889.03% | -932043671920.23% | 1/44 | 0/44 | 0.000001 | 2 |
| **Elastic Net** | 0.1380 | 0.2266 | **500716.3261** | -35638976.77% | -18642713.67% | 7/44 | 1/44 | 0.000019 | 2 |
| **Gradient Boosting** | 0.1383 | 0.2266 | **5.4132** | -285.29% | -101.55% | 5/44 | 0/44 | 0.068848 | 0 |
| **PyTorch LSTM** | 0.1368 | 0.2283 | **1.9849** | -41.28% | +26.10% | 12/44 | 2/44 | 0.091585 | 0 |


### Horizon: 10-Day (10-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins | Raw Min Pred | Near-Zero Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1239 | 0.1858 | **1.4806** | — | +47.10% | 2/44 | 0/44 | 0.062263 | 0 |
| **Rolling Mean** | 0.1150 | 0.1756 | **0.6604** | +55.40% | +76.41% | 4/44 | 12/44 | 0.087747 | 0 |
| **EWMA (λ=0.94)** | 0.1181 | 0.1785 | **1.0016** | +32.35% | +64.21% | 0/44 | 2/44 | 0.073901 | 0 |
| **HAR-RV** | 0.1169 | 0.1900 | **2.7989** | -89.04% | — | 3/44 | 3/44 | 0.074199 | 0 |
| **GARCH(1,1)** | 0.1180 | 0.1769 | **0.8789** | +40.64% | +68.60% | 6/44 | 21/44 | 0.120928 | 0 |
| **Ridge** | 0.1297 | 0.2027 | **1104973914.7632** | -74630769425.21% | -39478470502.35% | 0/44 | 2/44 | 0.000002 | 2 |
| **Elastic Net** | 0.1220 | 0.1952 | **1261627.7822** | -85211189.59% | -45075294.67% | 11/44 | 1/44 | 0.000008 | 1 |
| **Gradient Boosting** | 0.1258 | 0.1992 | **3.2104** | -116.83% | -14.70% | 3/44 | 2/44 | 0.082426 | 0 |
| **PyTorch LSTM** | 0.1254 | 0.2001 | **1.7283** | -16.73% | +38.25% | 15/44 | 1/44 | 0.104057 | 0 |


### Horizon: 20-Day (20-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins | Raw Min Pred | Near-Zero Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1109 | 0.1623 | **2.7020** | — | -36.06% | 1/44 | 1/44 | 0.062263 | 0 |
| **Rolling Mean** | 0.0981 | 0.1485 | **0.9125** | +66.23% | +54.05% | 4/44 | 8/44 | 0.087747 | 0 |
| **EWMA (λ=0.94)** | 0.1034 | 0.1542 | **1.6042** | +40.63% | +19.22% | 0/44 | 2/44 | 0.073901 | 0 |
| **HAR-RV** | 0.1025 | 0.1586 | **1.9859** | +26.51% | — | 2/44 | 5/44 | 0.085654 | 0 |
| **GARCH(1,1)** | 0.1010 | 0.1505 | **1.3438** | +50.27% | +32.33% | 4/44 | 18/44 | 0.131431 | 0 |
| **Ridge** | 0.1199 | 0.1776 | **10984.9406** | -406441.15% | -553057.62% | 0/44 | 4/44 | 0.000674 | 0 |
| **Elastic Net** | 0.1085 | 0.1645 | **9297.8042** | -344002.00% | -468100.18% | 7/44 | 3/44 | 0.000769 | 0 |
| **Gradient Boosting** | 0.1129 | 0.1697 | **2.0702** | +23.38% | -4.25% | 6/44 | 1/44 | 0.090224 | 0 |
| **PyTorch LSTM** | 0.1129 | 0.1694 | **1.9966** | +26.11% | -0.54% | 20/44 | 2/44 | 0.118772 | 0 |


## 2. Target Formulation Comparison: Direct Volatility vs Log-Variance vs Log-Volatility
Comparison of learned model performance when trained on levels of volatility vs log-variance vs log-volatility.

| Horizon | Model | Target Formulation | Test MAE | Test RMSE | Test QLIKE |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1-Day | Gradient Boosting | `log_variance` | 0.1877 | 0.3243 | **1613.3562** |
| 1-Day | Gradient Boosting | `direct_volatility` | 0.1905 | 0.3086 | **4.6607** |
| 1-Day | PyTorch LSTM | `log_variance` | 0.1777 | 0.3103 | **5.6461** |
| 1-Day | PyTorch LSTM | `direct_volatility` | 0.1787 | 0.2997 | **23660842361867116544.0000** |
| 5-Day | Gradient Boosting | `log_variance` | 0.1383 | 0.2266 | **5.4132** |
| 5-Day | Gradient Boosting | `direct_volatility` | 0.1408 | 0.2273 | **3.6546** |
| 5-Day | PyTorch LSTM | `log_variance` | 0.1368 | 0.2283 | **1.9849** |
| 5-Day | PyTorch LSTM | `direct_volatility` | 0.1342 | 0.2227 | **3288144904676307369984.0000** |
| 10-Day | Gradient Boosting | `log_variance` | 0.1258 | 0.1992 | **3.2104** |
| 10-Day | Gradient Boosting | `direct_volatility` | 0.1270 | 0.1999 | **2.6132** |
| 10-Day | PyTorch LSTM | `log_variance` | 0.1254 | 0.2001 | **1.7283** |
| 10-Day | PyTorch LSTM | `direct_volatility` | 0.1219 | 0.1936 | **22712421158249457254400.0000** |
| 20-Day | Gradient Boosting | `log_variance` | 0.1129 | 0.1697 | **2.0702** |
| 20-Day | Gradient Boosting | `direct_volatility` | 0.1131 | 0.1698 | **1.9785** |
| 20-Day | PyTorch LSTM | `log_variance` | 0.1129 | 0.1694 | **1.9966** |
| 20-Day | PyTorch LSTM | `direct_volatility` | 0.1109 | 0.1665 | **15172865463367885651968.0000** |


## 3. Uncertainty Cones & Prediction Interval Calibration
Evaluation of empirical coverage vs nominal 90% target coverage (p05 to p95 interval) on out-of-sample test partitions.

### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)
| Horizon | Model | Empirical Coverage (90% Nom.) | Avg Width (Annualized σ) | Low Vol Regime Cov | High Vol Regime Cov |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1-Day | Persistence | **90.8%** | 3.6145 | 72.8% | 99.6% |
| 1-Day | HAR-RV | **90.2%** | 0.8618 | 79.9% | 90.8% |
| 1-Day | GARCH(1,1) | **91.5%** | 4.1650 | 74.9% | 99.8% |
| 1-Day | Gradient Boosting | **90.0%** | 1.2000 | 83.3% | 88.9% |
| 1-Day | PyTorch LSTM | **91.3%** | 1.0870 | 78.3% | 95.6% |
| 5-Day | Persistence | **93.0%** | 0.7855 | 85.2% | 94.3% |
| 5-Day | HAR-RV | **90.6%** | 0.4806 | 91.8% | 80.3% |
| 5-Day | GARCH(1,1) | **94.5%** | 0.8351 | 86.8% | 96.9% |
| 5-Day | Gradient Boosting | **90.6%** | 0.5383 | 89.3% | 82.9% |
| 5-Day | PyTorch LSTM | **87.5%** | 0.4330 | 92.3% | 70.5% |
| 10-Day | Persistence | **91.4%** | 0.6128 | 85.5% | 89.5% |
| 10-Day | HAR-RV | **89.1%** | 0.3967 | 92.7% | 75.3% |
| 10-Day | GARCH(1,1) | **93.3%** | 0.5993 | 86.2% | 94.2% |
| 10-Day | Gradient Boosting | **89.3%** | 0.4621 | 90.7% | 79.0% |
| 10-Day | PyTorch LSTM | **83.6%** | 0.3405 | 94.2% | 59.5% |
| 20-Day | Persistence | **91.0%** | 0.5109 | 91.5% | 85.9% |
| 20-Day | HAR-RV | **88.8%** | 0.3592 | 95.1% | 73.3% |
| 20-Day | GARCH(1,1) | **94.0%** | 0.4745 | 91.0% | 91.5% |
| 20-Day | Gradient Boosting | **88.3%** | 0.4035 | 91.5% | 77.6% |
| 20-Day | PyTorch LSTM | **80.2%** | 0.3094 | 93.2% | 54.8% |


### Gaussian Model-Implied p05–p95 Price Range Coverage (Nominal: 90.0%)
| Horizon | Model Implied Volatility | Empirical Price Range Coverage | Avg Cone Width (% Price) |
| :---: | :--- | :---: | :---: |
| 1-Day | Persistence | **88.4%** | ±7.2% |
| 1-Day | HAR-RV | **57.6%** | ±3.0% |
| 1-Day | GARCH(1,1) | **90.3%** | ±7.4% |
| 1-Day | Gradient Boosting | **56.7%** | ±3.0% |
| 1-Day | PyTorch LSTM | **64.3%** | ±3.5% |
| 5-Day | Persistence | **87.9%** | ±16.1% |
| 5-Day | HAR-RV | **81.3%** | ±12.0% |
| 5-Day | GARCH(1,1) | **90.4%** | ±16.7% |
| 5-Day | Gradient Boosting | **81.3%** | ±12.4% |
| 5-Day | PyTorch LSTM | **77.9%** | ±11.3% |
| 10-Day | Persistence | **87.6%** | ±22.6% |
| 10-Day | HAR-RV | **83.7%** | ±18.3% |
| 10-Day | GARCH(1,1) | **90.4%** | ±23.6% |
| 10-Day | Gradient Boosting | **83.9%** | ±19.0% |
| 10-Day | PyTorch LSTM | **80.0%** | ±17.0% |
| 20-Day | Persistence | **87.4%** | ±30.9% |
| 20-Day | HAR-RV | **85.6%** | ±27.1% |
| 20-Day | GARCH(1,1) | **90.4%** | ±32.9% |
| 20-Day | Gradient Boosting | **85.5%** | ±27.9% |
| 20-Day | PyTorch LSTM | **83.0%** | ±25.7% |


## 4. Top Worst-Error QLIKE Diagnostics (Tail Error Analysis)
Inspection of top catastrophic QLIKE errors reveals why certain models achieve strong MAE but poor QLIKE:

### Top 5 Worst Out-of-Sample Losses: PyTorch LSTM
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error (Pred - Act) | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1169 | 0.1984 | -11.2238 | **9406.88** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.3152 | 0.5783 | -15.8567 | **2622.83** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.1884 | 0.1805 | -6.7721 | **1356.58** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.1890 | 0.1984 | -6.7693 | **1346.90** | False |
| MNST | 2026-07-17 | 20d | High Vol | 6.4301 | 0.2187 | 0.1984 | -6.2114 | **856.84** | False |


### Top 5 Worst Out-of-Sample Losses: Gradient Boosting
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error (Pred - Act) | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AMD | 2026-07-29 | 1d | High Vol | 1.9397 | 0.0014 | 0.7864 | -1.9384 | **1995939.32** | False |
| AMD | 2026-06-04 | 1d | High Vol | 1.8250 | 0.0015 | 0.8753 | -1.8234 | **1411257.50** | False |
| AMD | 2026-06-15 | 1d | High Vol | 1.2039 | 0.0014 | 0.7906 | -1.2025 | **717847.66** | False |
| AMD | 2026-06-18 | 1d | Normal Vol | 0.4158 | 0.0005 | 0.8227 | -0.4153 | **666155.24** | False |
| AMD | 2026-06-12 | 1d | High Vol | 1.0706 | 0.0014 | 0.7572 | -1.0691 | **545516.59** | False |


### Top 5 Worst Out-of-Sample Losses: HAR-RV
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error (Pred - Act) | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1194 | 0.1984 | -11.2213 | **9012.88** | False |
| MNST | 2026-07-30 | 1d | High Vol | 11.2112 | 0.1248 | 3.3231 | -11.0864 | **8056.02** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.2982 | 0.5783 | -15.8738 | **2932.29** | False |
| MNST | 2026-07-30 | 5d | High Vol | 8.4709 | 0.1991 | 3.3231 | -8.2717 | **1801.12** | False |
| MNST | 2026-07-22 | 1d | High Vol | 10.6493 | 0.2830 | 2.4261 | -10.3664 | **1408.07** | False |


### Top 5 Worst Out-of-Sample Losses: Rolling Mean (60d)
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error (Pred - Act) | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.3248 | 0.1984 | -11.0159 | **1210.68** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.7447 | 0.5783 | -15.4272 | **464.39** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.3248 | 0.1984 | -6.6334 | **451.70** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.3252 | 0.1805 | -6.6353 | **451.08** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.3217 | 0.1639 | -6.1087 | **392.54** | False |

