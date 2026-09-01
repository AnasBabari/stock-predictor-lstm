# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report
**Date:** 2026-08-31T23:17:21Z | **Universe:** 44 Liquid Assets across 8 Sectors | **Execution Time:** 220.8s

## Executive Summary
This empirical study evaluates the predictive accuracy and uncertainty calibration of 8 volatility forecasting models across 4 horizons (1-day, 5-day, 10-day, and 20-day) using strict chronological 70/15/15 splits with horizon-length boundary embargoes.

## 1. Multi-Horizon Forecasting Accuracy & Skill Matrix
### Horizon: 1-Day (1-session)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.2214 | 0.3189 | **2.2432** | — | +72.99% | 8/44 | 3/44 |
| **Rolling Mean** | 0.2137 | 0.3073 | **1.8643** | +16.89% | +77.55% | 14/44 | 25/44 |
| **EWMA (λ=0.94)** | 0.2186 | 0.3131 | **2.0291** | +9.54% | +75.57% | 22/44 | 16/44 |
| **HAR-RV** | 0.1808 | 0.3192 | **8.3058** | -270.27% | — | 0/44 | 0/44 |
| **Ridge** | 0.4747 | 1.3725 | **33424840934.1004** | -1490078072676.22% | -402428780260.44% | 0/44 | 0/44 |
| **Elastic Net** | 0.2736 | 0.7974 | **19316954116.4916** | -861149042271.15% | -232572483960.72% | 0/44 | 0/44 |
| **Gradient Boosting** | 0.1877 | 0.3244 | **1049.5676** | -46689.68% | -12536.60% | 0/44 | 0/44 |
| **PyTorch LSTM** | 0.1788 | 0.3128 | **6.2654** | -179.31% | +24.57% | 0/44 | 0/44 |


### Horizon: 5-Day (5-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1437 | 0.2193 | **1.4050** | — | +47.69% | 5/44 | 4/44 |
| **Rolling Mean** | 0.1361 | 0.2088 | **0.7562** | +46.17% | +71.84% | 7/44 | 25/44 |
| **EWMA (λ=0.94)** | 0.1390 | 0.2122 | **1.0355** | +26.30% | +61.45% | 3/44 | 7/44 |
| **HAR-RV** | 0.1311 | 0.2213 | **2.6858** | -91.17% | — | 2/44 | 1/44 |
| **Ridge** | 0.2035 | 0.5835 | **10688398001.2791** | -760759370175.77% | -397953495633.36% | 1/44 | 1/44 |
| **Elastic Net** | 0.1352 | 0.2242 | **50209.4781** | -3573618.99% | -1869313.67% | 11/44 | 4/44 |
| **Gradient Boosting** | 0.1383 | 0.2266 | **5.3675** | -282.04% | -99.85% | 8/44 | 1/44 |
| **PyTorch LSTM** | 0.1357 | 0.2261 | **2.1684** | -54.34% | +19.27% | 7/44 | 1/44 |


### Horizon: 10-Day (10-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1239 | 0.1858 | **1.4806** | — | +47.10% | 4/44 | 1/44 |
| **Rolling Mean** | 0.1150 | 0.1756 | **0.6604** | +55.40% | +76.41% | 4/44 | 18/44 |
| **EWMA (λ=0.94)** | 0.1181 | 0.1785 | **1.0016** | +32.35% | +64.21% | 0/44 | 8/44 |
| **HAR-RV** | 0.1169 | 0.1900 | **2.7989** | -89.04% | — | 4/44 | 4/44 |
| **Ridge** | 0.1297 | 0.2027 | **1104973914.7633** | -74630769425.22% | -39478470502.36% | 0/44 | 2/44 |
| **Elastic Net** | 0.1198 | 0.1933 | **1328166.3202** | -89705252.51% | -47452581.30% | 13/44 | 5/44 |
| **Gradient Boosting** | 0.1258 | 0.1991 | **3.2072** | -116.62% | -14.59% | 7/44 | 3/44 |
| **PyTorch LSTM** | 0.1239 | 0.1978 | **1.9282** | -30.23% | +31.11% | 12/44 | 3/44 |


### Horizon: 20-Day (20-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1109 | 0.1623 | **2.7020** | — | -36.06% | 2/44 | 1/44 |
| **Rolling Mean** | 0.0981 | 0.1485 | **0.9125** | +66.23% | +54.05% | 6/44 | 13/44 |
| **EWMA (λ=0.94)** | 0.1034 | 0.1542 | **1.6042** | +40.63% | +19.22% | 1/44 | 3/44 |
| **HAR-RV** | 0.1025 | 0.1586 | **1.9859** | +26.51% | — | 4/44 | 7/44 |
| **Ridge** | 0.1199 | 0.1776 | **10984.9406** | -406441.15% | -553057.62% | 1/44 | 5/44 |
| **Elastic Net** | 0.1063 | 0.1623 | **2461.4116** | -90994.27% | -123846.83% | 8/44 | 8/44 |
| **Gradient Boosting** | 0.1129 | 0.1696 | **2.0696** | +23.41% | -4.22% | 7/44 | 4/44 |
| **PyTorch LSTM** | 0.1111 | 0.1679 | **2.0846** | +22.85% | -4.97% | 15/44 | 3/44 |


## 2. Uncertainty Cones & Prediction Interval Calibration
Evaluation of empirical coverage vs nominal 90% target coverage (p05 to p95 interval) on out-of-sample test partitions.

### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)
| Horizon | Model | Empirical Coverage (90% Nom.) | Avg Width (Annualized σ) | Low Vol Regime Cov | High Vol Regime Cov |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1-Day | Persistence | **90.8%** | 3.6145 | 72.8% | 99.6% |
| 1-Day | HAR-RV | **90.2%** | 0.8618 | 79.9% | 90.8% |
| 1-Day | Ridge | **89.9%** | 4.0502 | 82.4% | 89.0% |
| 1-Day | Gradient Boosting | **90.0%** | 1.2106 | 83.5% | 88.8% |
| 1-Day | PyTorch LSTM | **90.4%** | 1.0004 | 77.7% | 93.7% |
| 5-Day | Persistence | **93.0%** | 0.7855 | 85.2% | 94.3% |
| 5-Day | HAR-RV | **90.6%** | 0.4806 | 91.8% | 80.3% |
| 5-Day | Ridge | **91.3%** | 0.7732 | 89.5% | 86.9% |
| 5-Day | Gradient Boosting | **90.5%** | 0.5363 | 89.3% | 82.7% |
| 5-Day | PyTorch LSTM | **88.5%** | 0.4568 | 90.3% | 75.6% |
| 10-Day | Persistence | **91.4%** | 0.6128 | 85.5% | 89.5% |
| 10-Day | HAR-RV | **89.1%** | 0.3967 | 92.7% | 75.3% |
| 10-Day | Ridge | **89.7%** | 0.5250 | 89.4% | 83.3% |
| 10-Day | Gradient Boosting | **89.2%** | 0.4626 | 90.6% | 78.9% |
| 10-Day | PyTorch LSTM | **85.5%** | 0.3822 | 89.8% | 69.3% |
| 20-Day | Persistence | **91.0%** | 0.5109 | 91.5% | 85.9% |
| 20-Day | HAR-RV | **88.8%** | 0.3592 | 95.1% | 73.3% |
| 20-Day | Ridge | **87.8%** | 0.4232 | 88.1% | 80.8% |
| 20-Day | Gradient Boosting | **88.2%** | 0.4029 | 91.4% | 77.6% |
| 20-Day | PyTorch LSTM | **83.1%** | 0.3454 | 89.7% | 66.4% |


### Price Diffusion Cone Coverage (Nominal Target: 90.0% p05-p95)
| Horizon | Model Implied Volatility | Price Cone Empirical Coverage | Avg Cone Width (% Price) |
| :---: | :--- | :---: | :---: |
| 1-Day | Persistence | **88.4%** | ±7.2% |
| 1-Day | HAR-RV | **57.6%** | ±3.0% |
| 1-Day | Gradient Boosting | **56.7%** | ±3.0% |
| 1-Day | PyTorch LSTM | **62.9%** | ±3.5% |
| 5-Day | Persistence | **87.9%** | ±16.1% |
| 5-Day | HAR-RV | **81.3%** | ±12.0% |
| 5-Day | Gradient Boosting | **81.4%** | ±12.4% |
| 5-Day | PyTorch LSTM | **80.2%** | ±12.0% |
| 10-Day | Persistence | **87.6%** | ±22.6% |
| 10-Day | HAR-RV | **83.7%** | ±18.3% |
| 10-Day | Gradient Boosting | **83.9%** | ±19.0% |
| 10-Day | PyTorch LSTM | **83.2%** | ±18.6% |
| 20-Day | Persistence | **87.4%** | ±30.9% |
| 20-Day | HAR-RV | **85.6%** | ±27.1% |
| 20-Day | Gradient Boosting | **85.5%** | ±27.9% |
| 20-Day | PyTorch LSTM | **85.2%** | ±28.1% |

