# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report (V3)
**Date:** 2026-09-01T11:47:13Z | **Universe:** 44 Liquid Assets across 8 Sectors | **Feature Mode:** `price_plus_ohlc` | **Target Space:** `log_variance` | **Execution Time:** 722.3s

## Executive Summary
Phase 3 establishes rigorous empirical benchmarking of volatility forecasting models with audited corporate-action-adjusted OHLC data, nested causal feature ablations, neural output formulation comparisons, and comprehensive tail error diagnostics.
- **1-Day Candidate Findings:** MAE: PyTorch LSTM (0.1606); RMSE: PyTorch LSTM (0.2824); QLIKE: GARCH(1,1) (1.8581). These are separate metric winners across all evaluated candidates, not one combined winner. GARCH(1,1) has the most validation selections (31/44).
- **1-Day Baseline Findings:** MAE: HAR-RV (0.1629); RMSE: HAR-RV (0.2933); QLIKE: GARCH(1,1) (1.8581). Baseline metric winners are reported separately from learned candidates.
- **Single-Day Proxy Noise on HAR-RV:** The canonical 1-day realized volatility target $RV(t,1) = \sqrt{252}|r_{t+1}|$ is dominated by single-session return jump noise, which can disadvantage multi-frequency autoregressive filters like HAR-RV (the current aggregate HAR-RV QLIKE is `5.8558`). As the horizon expands, jump noise averages out and HAR-RV's multi-resolution memory can become competitive.
- **Target / Output Formulation:** The neural link is horizon-dependent: Softplus has the lower held-out QLIKE at 1d, 5d, 20d; log-variance is lower at 10d. Both positive-output formulations prevent structural negative/near-zero predictions; neither is declared universally best.

### Phase 3.5 reporting and methodology audit
- Baseline winners are reported separately for MAE, RMSE, and QLIKE; no metric is silently substituted for another.
- The p05–p95 price display is a central 90% **raw Gaussian reference scenario** with zero expected log return. Its observed test coverage is diagnostic and is not used to claim calibration.
- Feature ablation comparisons below are paired by ticker and use deterministic asset-level bootstrap intervals; they do not select a winner after test access.

- **Next-experiment configuration freeze:** the table below records the complete-coverage aggregate validation-QLIKE winner for each horizon. The untouched test partition was not used, and this methodological freeze does not promote a model to production.

| Horizon | Feature configuration | Model | Validation QLIKE | Assets | Status |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1d | `price_plus_ohlc_plus_market` | PyTorch LSTM | 2.8992 | 44 | frozen_for_next_experiment |
| 5d | `price_plus_ohlc_plus_market` | PyTorch LSTM | 0.6353 | 44 | frozen_for_next_experiment |
| 10d | `price_plus_ohlc_plus_market` | PyTorch LSTM | 0.4061 | 44 | frozen_for_next_experiment |
| 20d | `price_only` | PyTorch LSTM | 0.2513 | 44 | frozen_for_next_experiment |

## 1. Multi-Horizon Forecasting Accuracy & Distributional Skill Matrix
### Horizon: 1-Day (1-session)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.2062 | 0.3050 | **2.0526** | 0.8192 | 5.9128 | 20.2% | 0/44 | 1/44 | 0/44 | 44/44 |
| **Rolling Mean (60d)** | 0.2055 | 0.2963 | **1.8895** | 0.8308 | 5.8619 | 18.5% | 11/44 | 11/44 | 33/44 | 44/44 |
| **EWMA (λ=0.94)** | 0.2045 | 0.2973 | **1.9222** | 0.8168 | 5.7948 | 18.7% | 2/44 | 2/44 | 41/44 | 44/44 |
| **HAR-RV** | 0.1629 | 0.2933 | **5.8558** | 1.2073 | 18.2586 | 30.2% | 0/44 | 0/44 | 0/44 | 0/44 |
| **GARCH(1,1)** | 0.2070 | 0.2965 | **1.8581** | 0.8184 | 5.7078 | 17.4% | 31/44 | 30/44 | 41/44 | 44/44 |
| **Ridge** | 0.2235 | 0.7347 | **5245897624880.8184** | 1.4487 | 44.9875 | 43.9% | 0/44 | 0/44 | 0/44 | 0/44 |
| **Elastic Net** | 0.2039 | 0.6087 | **1465722.4966** | 1.2557 | 25.7837 | 32.3% | 0/44 | 0/44 | 0/44 | 9/44 |
| **Gradient Boosting** | 0.1636 | 0.2928 | **162.4031** | 1.2276 | 31.5188 | 46.2% | 0/44 | 0/44 | 0/44 | 11/44 |
| **PyTorch LSTM** | 0.1606 | 0.2824 | **3.9446** | 0.9454 | 10.9296 | 29.5% | 0/44 | 0/44 | 0/44 | 44/44 |


### Horizon: 5-Day (5-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1384 | 0.2191 | **0.9972** | 0.2347 | 2.1840 | 22.4% | 0/44 | 1/44 | 0/44 | 44/44 |
| **Rolling Mean (60d)** | 0.1359 | 0.2102 | **0.7206** | 0.2340 | 1.9969 | 20.5% | 10/44 | 11/44 | 36/44 | 44/44 |
| **EWMA (λ=0.94)** | 0.1343 | 0.2096 | **0.8167** | 0.2264 | 1.8878 | 21.9% | 0/44 | 2/44 | 41/44 | 44/44 |
| **HAR-RV** | 0.1753 | 0.2663 | **6.0165** | 1.7289 | 17.6433 | 18.7% | 0/44 | 0/44 | 0/44 | 0/44 |
| **GARCH(1,1)** | 0.1342 | 0.2060 | **0.7310** | 0.2256 | 1.6940 | 19.8% | 23/44 | 26/44 | 40/44 | 44/44 |
| **Ridge** | 0.1291 | 0.2190 | **7236162.5961** | 0.2286 | 21.0922 | 25.5% | 0/44 | 0/44 | 6/44 | 43/44 |
| **Elastic Net** | 0.1180 | 0.2065 | **149.1543** | 0.1906 | 5.1656 | 24.6% | 5/44 | 0/44 | 17/44 | 43/44 |
| **Gradient Boosting** | 0.1166 | 0.2042 | **1.6935** | 0.1837 | 2.9206 | 23.5% | 3/44 | 3/44 | 16/44 | 44/44 |
| **PyTorch LSTM** | 0.1184 | 0.2070 | **1.2807** | 0.1922 | 3.2820 | 24.3% | 3/44 | 1/44 | 15/44 | 44/44 |


### Horizon: 10-Day (10-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1222 | 0.1908 | **0.9413** | 0.1766 | 2.0154 | 17.0% | 0/44 | 1/44 | 0/44 | 44/44 |
| **Rolling Mean (60d)** | 0.1181 | 0.1819 | **0.5707** | 0.1634 | 1.5614 | 15.5% | 8/44 | 9/44 | 37/44 | 44/44 |
| **EWMA (λ=0.94)** | 0.1177 | 0.1818 | **0.7018** | 0.1674 | 1.6977 | 17.0% | 0/44 | 1/44 | 40/44 | 44/44 |
| **HAR-RV** | 0.1949 | 0.2645 | **7.5067** | 2.7568 | 25.6507 | 12.1% | 0/44 | 0/44 | 0/44 | 0/44 |
| **GARCH(1,1)** | 0.1165 | 0.1769 | **0.5984** | 0.1596 | 1.4011 | 15.4% | 16/44 | 21/44 | 39/44 | 44/44 |
| **Ridge** | 0.1155 | 0.1931 | **35302.6747** | 0.1578 | 14.6101 | 20.6% | 0/44 | 1/44 | 15/44 | 43/44 |
| **Elastic Net** | 0.1033 | 0.1785 | **240.7076** | 0.1239 | 13.0882 | 19.2% | 7/44 | 4/44 | 30/44 | 43/44 |
| **Gradient Boosting** | 0.1034 | 0.1770 | **1.3710** | 0.1224 | 6.8252 | 17.9% | 7/44 | 5/44 | 29/44 | 44/44 |
| **PyTorch LSTM** | 0.1039 | 0.1785 | **1.0209** | 0.1252 | 4.6897 | 18.2% | 6/44 | 2/44 | 27/44 | 44/44 |


### Horizon: 20-Day (20-sessions)
*Evaluated across 44 liquid assets (Out-of-Sample Test Partition)*

| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.1126 | 0.1709 | **1.3152** | 0.1608 | 1.7651 | 12.9% | 0/44 | 1/44 | 0/44 | 44/44 |
| **Rolling Mean (60d)** | 0.1048 | 0.1587 | **0.5776** | 0.1252 | 1.4328 | 11.0% | 4/44 | 8/44 | 35/44 | 44/44 |
| **EWMA (λ=0.94)** | 0.1060 | 0.1605 | **0.8228** | 0.1354 | 1.2719 | 12.4% | 0/44 | 0/44 | 40/44 | 44/44 |
| **HAR-RV** | 0.2171 | 0.2664 | **11.4415** | 4.7892 | 37.7534 | 7.8% | 0/44 | 0/44 | 0/44 | 0/44 |
| **GARCH(1,1)** | 0.1020 | 0.1521 | **0.6592** | 0.1255 | 0.9911 | 11.0% | 11/44 | 13/44 | 41/44 | 44/44 |
| **Ridge** | 0.1022 | 0.1644 | **16.8155** | 0.1154 | 7.4152 | 16.2% | 1/44 | 2/44 | 23/44 | 43/44 |
| **Elastic Net** | 0.0912 | 0.1495 | **15.3455** | 0.0946 | 6.8580 | 14.6% | 8/44 | 8/44 | 37/44 | 43/44 |
| **Gradient Boosting** | 0.0915 | 0.1491 | **0.9647** | 0.0979 | 5.3026 | 13.2% | 5/44 | 8/44 | 36/44 | 44/44 |
| **PyTorch LSTM** | 0.0938 | 0.1525 | **1.0225** | 0.1032 | 5.0680 | 12.7% | 15/44 | 4/44 | 36/44 | 44/44 |


## 2. Neural Target / Output Formulation Comparison (PyTorch LSTM)
Controlled comparison of neural output formulations on identical splits, architectures, and training budgets:

| Horizon | Formulation | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Max QLIKE | Near-Zero Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Day | `LOG_VARIANCE` | 0.1606 | 0.2824 | **3.9446** | 0.9454 | 10.9296 | 8760.72 | 0 |
| 1-Day | `SOFTPLUS_VOLATILITY` | 0.1660 | 0.2761 | **2.6754** | 0.7684 | 6.7839 | 3291.28 | 0 |
| 1-Day | `DIRECT_VOLATILITY` | 0.1664 | 0.2776 | **1902476855572237385728.0000** | 17219168454732626.0000 | 5689214358916716036096.0000 | 3.85e+24 | 251 |
| 5-Day | `LOG_VARIANCE` | 0.1184 | 0.2070 | **1.2807** | 0.1922 | 3.2820 | 1855.76 | 0 |
| 5-Day | `SOFTPLUS_VOLATILITY` | 0.1202 | 0.2043 | **1.1901** | 0.1841 | 2.3737 | 1049.08 | 0 |
| 5-Day | `DIRECT_VOLATILITY` | 0.1250 | 0.2089 | **1742564840369299128320.0000** | 310276335674865942528.0000 | 7602171890481825316864.0000 | 1.40e+24 | 252 |
| 10-Day | `LOG_VARIANCE` | 0.1039 | 0.1785 | **1.0209** | 0.1252 | 4.6897 | 973.31 | 0 |
| 10-Day | `SOFTPLUS_VOLATILITY` | 0.1087 | 0.1793 | **1.0382** | 0.1377 | 5.9045 | 935.28 | 0 |
| 10-Day | `DIRECT_VOLATILITY` | 0.1153 | 0.1862 | **1621760029740949569536.0000** | 532576589293062586368.0000 | 8208653779705135104000.0000 | 8.54e+23 | 248 |
| 20-Day | `LOG_VARIANCE` | 0.0938 | 0.1525 | **1.0225** | 0.1032 | 5.0680 | 1052.34 | 0 |
| 20-Day | `SOFTPLUS_VOLATILITY` | 0.0962 | 0.1515 | **0.8378** | 0.1103 | 5.7624 | 740.35 | 0 |
| 20-Day | `DIRECT_VOLATILITY` | 0.1148 | 0.1703 | **4048702863838160617472.0000** | 1992805509629513039872.0000 | 15007145289924271407104.0000 | 5.95e+23 | 587 |


## 3. Nested Feature Ablation Study
Evaluation of incremental causal information value: `PRICE_ONLY` → `PRICE_PLUS_OHLC` → `PRICE_PLUS_OHLC_PLUS_MARKET`.

| Horizon | Model | Feature Configuration | Features | Test MAE | Test RMSE | Test QLIKE | Median QLIKE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1-Day | Gradient Boosting | `PRICE_ONLY` | 11 | 0.1648 | 0.2940 | **906.5767** | 1.2503 |
| 1-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 23 | 0.1636 | 0.2928 | **162.4031** | 1.2276 |
| 1-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1643 | 0.2943 | **168.2886** | 1.2669 |
| 1-Day | PyTorch LSTM | `PRICE_ONLY` | 11 | 0.1616 | 0.2866 | **4.4418** | 1.0121 |
| 1-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 23 | 0.1606 | 0.2824 | **3.9446** | 0.9454 |
| 1-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1595 | 0.2818 | **4.1469** | 0.9320 |
| 5-Day | Gradient Boosting | `PRICE_ONLY` | 11 | 0.1192 | 0.2066 | **1.2069** | 0.1987 |
| 5-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 23 | 0.1166 | 0.2042 | **1.6935** | 0.1837 |
| 5-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1168 | 0.2050 | **1.7886** | 0.1869 |
| 5-Day | PyTorch LSTM | `PRICE_ONLY` | 11 | 0.1198 | 0.2098 | **1.3298** | 0.2060 |
| 5-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 23 | 0.1184 | 0.2070 | **1.2807** | 0.1922 |
| 5-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1157 | 0.2039 | **1.3884** | 0.1855 |
| 10-Day | Gradient Boosting | `PRICE_ONLY` | 11 | 0.1063 | 0.1794 | **0.9740** | 0.1352 |
| 10-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 23 | 0.1034 | 0.1770 | **1.3710** | 0.1224 |
| 10-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1036 | 0.1771 | **1.2117** | 0.1277 |
| 10-Day | PyTorch LSTM | `PRICE_ONLY` | 11 | 0.1073 | 0.1829 | **1.0679** | 0.1417 |
| 10-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 23 | 0.1039 | 0.1785 | **1.0209** | 0.1252 |
| 10-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.1030 | 0.1779 | **1.1102** | 0.1248 |
| 20-Day | Gradient Boosting | `PRICE_ONLY` | 11 | 0.0953 | 0.1529 | **0.9490** | 0.1106 |
| 20-Day | Gradient Boosting | `PRICE_PLUS_OHLC` | 23 | 0.0915 | 0.1491 | **0.9647** | 0.0979 |
| 20-Day | Gradient Boosting | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.0936 | 0.1512 | **0.8583** | 0.1066 |
| 20-Day | PyTorch LSTM | `PRICE_ONLY` | 11 | 0.0977 | 0.1568 | **1.1112** | 0.1206 |
| 20-Day | PyTorch LSTM | `PRICE_PLUS_OHLC` | 23 | 0.0938 | 0.1525 | **1.0225** | 0.1032 |
| 20-Day | PyTorch LSTM | `PRICE_PLUS_OHLC_PLUS_MARKET` | 27 | 0.0926 | 0.1523 | **1.0487** | 0.1028 |


### Phase 3.5 Paired Ablation Breadth (all available models)
Positive deltas mean the added feature group reduced the per-asset test error. Bootstrap resamples assets, not individual overlapping origin rows; these summaries are descriptive and do not change model selection.

| Horizon | Model | Transition | Assets | Improved QLIKE | Δ QLIKE (mean) | QLIKE 95% CI | Improved MAE | Δ MAE (mean) | MAE 95% CI | Improved RMSE | Δ RMSE (mean) | RMSE 95% CI | Sectors >50% |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1d | Elastic Net | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 15/44 | -1465713.5407 | [-4397139.5695, -0.1319] | 16/44 | +0.0672 | [-0.0013, +0.2010] | 25/44 | +0.2472 | [-0.0000, +0.7391] | 3/8 |
| 1d | EWMA (λ=0.94) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | GARCH(1,1) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | Gradient Boosting | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 24/44 | +744.1736 | [-93.3535, +2223.3543] | 33/44 | +0.0012 | [+0.0003, +0.0020] | 33/44 | +0.0012 | [-0.0002, +0.0025] | 5/8 |
| 1d | HAR-RV | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | PyTorch LSTM | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 35/44 | +0.4972 | [+0.3213, +0.6929] | 30/44 | +0.0010 | [+0.0000, +0.0020] | 37/44 | +0.0043 | [+0.0028, +0.0057] | 8/8 |
| 1d | Persistence | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | Ridge | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 5/44 | -5245573362069.7373 | [-15736720085935.1016, +11.6845] | 3/44 | +0.0413 | [-0.0249, +0.1642] | 5/44 | +0.2037 | [-0.1198, +0.7691] | 0/8 |
| 1d | Rolling Mean (60d) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Elastic Net | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 28/44 | -146.2887 | [-438.9541, +0.0608] | 28/44 | +0.0660 | [+0.0000, +0.1967] | 28/44 | +0.4177 | [+0.0007, +1.2454] | 8/8 |
| 5d | EWMA (λ=0.94) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | GARCH(1,1) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Gradient Boosting | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 29/44 | -0.4866 | [-1.5111, +0.0470] | 33/44 | +0.0027 | [+0.0014, +0.0041] | 31/44 | +0.0024 | [+0.0002, +0.0048] | 7/8 |
| 5d | HAR-RV | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | PyTorch LSTM | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 30/44 | +0.0491 | [-0.0301, +0.1072] | 28/44 | +0.0014 | [-0.0012, +0.0036] | 32/44 | +0.0027 | [-0.0005, +0.0055] | 6/8 |
| 5d | Persistence | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Ridge | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 14/44 | -7234447.1585 | [-21703341.4189, +0.0071] | 7/44 | +0.0177 | [-0.0108, +0.0716] | 10/44 | +0.2473 | [-0.0120, +0.7617] | 2/8 |
| 5d | Rolling Mean (60d) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Elastic Net | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 27/44 | -239.4542 | [-718.4149, +0.0379] | 24/44 | +0.0263 | [-0.0011, +0.0792] | 21/44 | +0.2593 | [-0.0019, +0.7795] | 8/8 |
| 10d | EWMA (λ=0.94) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | GARCH(1,1) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Gradient Boosting | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 28/44 | -0.3970 | [-1.2335, +0.0327] | 32/44 | +0.0030 | [+0.0013, +0.0049] | 27/44 | +0.0023 | [-0.0003, +0.0050] | 6/8 |
| 10d | HAR-RV | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | PyTorch LSTM | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 31/44 | +0.0470 | [+0.0186, +0.0758] | 29/44 | +0.0034 | [+0.0014, +0.0056] | 31/44 | +0.0044 | [+0.0021, +0.0069] | 6/8 |
| 10d | Persistence | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Ridge | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 16/44 | -35274.7211 | [-105824.1208, -0.0003] | 5/44 | -0.0032 | [-0.0137, +0.0135] | 6/44 | +0.0687 | [-0.0174, +0.2357] | 3/8 |
| 10d | Rolling Mean (60d) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Elastic Net | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 29/44 | -14.3264 | [-43.0080, +0.0269] | 25/44 | +0.0001 | [-0.0027, +0.0025] | 23/44 | -0.0010 | [-0.0046, +0.0024] | 8/8 |
| 20d | EWMA (λ=0.94) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | GARCH(1,1) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Gradient Boosting | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 23/44 | -0.0157 | [-0.0842, +0.0330] | 29/44 | +0.0038 | [+0.0014, +0.0065] | 25/44 | +0.0038 | [+0.0006, +0.0072] | 5/8 |
| 20d | HAR-RV | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | PyTorch LSTM | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 30/44 | +0.0886 | [+0.0149, +0.2136] | 31/44 | +0.0039 | [+0.0017, +0.0061] | 28/44 | +0.0043 | [+0.0014, +0.0073] | 6/8 |
| 20d | Persistence | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Ridge | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 18/44 | -15.1750 | [-45.4449, -0.0206] | 9/44 | -0.0095 | [-0.0153, -0.0049] | 11/44 | -0.0137 | [-0.0208, -0.0074] | 4/8 |
| 20d | Rolling Mean (60d) | `PRICE_ONLY → PRICE_PLUS_OHLC` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | Elastic Net | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 5/44 | -4386467.4867 | [-13159400.6936, -0.6529] | 5/44 | +0.0123 | [-0.0105, +0.0528] | 7/44 | +0.1036 | [-0.0838, +0.4033] | 0/8 |
| 1d | EWMA (λ=0.94) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | GARCH(1,1) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | Gradient Boosting | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 22/44 | -5.8855 | [-175.7244, +162.3807] | 18/44 | -0.0007 | [-0.0017, +0.0002] | 11/44 | -0.0016 | [-0.0031, -0.0002] | 4/8 |
| 1d | HAR-RV | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | PyTorch LSTM | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 20/44 | -0.2024 | [-0.5980, +0.0441] | 22/44 | +0.0010 | [+0.0000, +0.0021] | 22/44 | +0.0006 | [-0.0009, +0.0021] | 4/8 |
| 1d | Persistence | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 1d | Ridge | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 3/44 | -1483118890195501.0000 | [-4449356670585792.0000, -8.0812] | 3/44 | -0.0096 | [-0.0185, +0.0015] | 4/44 | -0.0029 | [-0.0641, +0.0821] | 0/8 |
| 1d | Rolling Mean (60d) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Elastic Net | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 17/44 | -103.7313 | [-311.1941, +0.0170] | 17/44 | -0.0013 | [-0.0033, +0.0001] | 23/44 | -0.0022 | [-0.0059, +0.0004] | 2/8 |
| 5d | EWMA (λ=0.94) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | GARCH(1,1) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Gradient Boosting | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 17/44 | -0.0951 | [-0.2891, +0.0216] | 21/44 | -0.0002 | [-0.0014, +0.0009] | 19/44 | -0.0008 | [-0.0019, +0.0002] | 3/8 |
| 5d | HAR-RV | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | PyTorch LSTM | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 25/44 | -0.1077 | [-0.3250, +0.0171] | 28/44 | +0.0027 | [+0.0007, +0.0053] | 26/44 | +0.0032 | [+0.0002, +0.0066] | 6/8 |
| 5d | Persistence | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 5d | Ridge | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 11/44 | +4604520.6792 | [-0.2442, +13813562.3814] | 11/44 | -0.0097 | [-0.0203, -0.0029] | 11/44 | -0.0152 | [-0.0338, -0.0035] | 1/8 |
| 5d | Rolling Mean (60d) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Elastic Net | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 18/44 | -5.4207 | [-16.2401, -0.0032] | 16/44 | -0.0024 | [-0.0048, -0.0006] | 18/44 | -0.0029 | [-0.0067, -0.0004] | 3/8 |
| 10d | EWMA (λ=0.94) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | GARCH(1,1) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Gradient Boosting | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 16/44 | +0.1593 | [-0.0269, +0.5131] | 19/44 | -0.0002 | [-0.0016, +0.0011] | 19/44 | -0.0001 | [-0.0017, +0.0016] | 2/8 |
| 10d | HAR-RV | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | PyTorch LSTM | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 18/44 | -0.0893 | [-0.2616, +0.0059] | 21/44 | +0.0009 | [-0.0006, +0.0024] | 19/44 | +0.0006 | [-0.0015, +0.0029] | 3/8 |
| 10d | Persistence | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 10d | Ridge | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 12/44 | -261258.8820 | [-783776.4936, -0.0590] | 10/44 | -0.0079 | [-0.0156, -0.0026] | 12/44 | -0.0119 | [-0.0239, -0.0034] | 1/8 |
| 10d | Rolling Mean (60d) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Elastic Net | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 17/44 | +4.9879 | [-0.0278, +15.0036] | 18/44 | -0.0028 | [-0.0059, -0.0004] | 16/44 | -0.0041 | [-0.0078, -0.0010] | 3/8 |
| 20d | EWMA (λ=0.94) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | GARCH(1,1) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Gradient Boosting | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 15/44 | +0.1064 | [-0.0264, +0.3547] | 15/44 | -0.0021 | [-0.0038, -0.0008] | 16/44 | -0.0021 | [-0.0041, -0.0004] | 3/8 |
| 20d | HAR-RV | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | PyTorch LSTM | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 16/44 | -0.0262 | [-0.0532, -0.0019] | 18/44 | +0.0012 | [-0.0011, +0.0039] | 18/44 | +0.0002 | [-0.0028, +0.0036] | 3/8 |
| 20d | Persistence | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
| 20d | Ridge | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 9/44 | +12.2453 | [-0.0797, +36.8674] | 13/44 | -0.0081 | [-0.0159, -0.0028] | 12/44 | -0.0111 | [-0.0223, -0.0028] | 1/8 |
| 20d | Rolling Mean (60d) | `PRICE_PLUS_OHLC → PRICE_PLUS_OHLC_PLUS_MARKET` | 44 | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/44 | +0.0000 | [+0.0000, +0.0000] | 0/8 |
`Sectors >50%` is the number of sectors in which a majority of paired assets improved on QLIKE; the machine-readable JSON also records each sector's count and mean delta so concentration can be audited.


## 4. Uncertainty Cones & Prediction Interval Calibration
### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)
| Horizon | Model | Empirical Coverage | Avg Width (Annualized σ) |
| :---: | :--- | :---: | :---: |
| 1-Day | Rolling Mean (60d) | **90.0%** | 3.4094 |
| 1-Day | GARCH(1,1) | **90.9%** | 3.8667 |
| 1-Day | HAR-RV | **90.1%** | 0.8188 |
| 1-Day | Gradient Boosting | **90.0%** | 1.2640 |
| 1-Day | PyTorch LSTM | **90.5%** | 1.1225 |
| 5-Day | Rolling Mean (60d) | **88.6%** | 0.6749 |
| 5-Day | GARCH(1,1) | **91.7%** | 0.7581 |
| 5-Day | HAR-RV | **86.3%** | 0.4119 |
| 5-Day | Gradient Boosting | **89.9%** | 0.4803 |
| 5-Day | PyTorch LSTM | **88.4%** | 0.4480 |
| 10-Day | Rolling Mean (60d) | **88.5%** | 0.5224 |
| 10-Day | GARCH(1,1) | **92.0%** | 0.5913 |
| 10-Day | HAR-RV | **86.7%** | 0.4292 |
| 10-Day | Gradient Boosting | **90.6%** | 0.4303 |
| 10-Day | PyTorch LSTM | **88.2%** | 0.3838 |
| 20-Day | Rolling Mean (60d) | **85.9%** | 0.4224 |
| 20-Day | GARCH(1,1) | **91.4%** | 0.4858 |
| 20-Day | HAR-RV | **86.1%** | 0.4362 |
| 20-Day | Gradient Boosting | **89.4%** | 0.3782 |
| 20-Day | PyTorch LSTM | **85.4%** | 0.3258 |


### Raw Gaussian Model-Implied p05–p95 Price Scenario (Nominal Central Coverage: 90.0%)
These rows are **not calibrated prediction intervals**. They use a zero-location, Gaussian return assumption and the model-implied terminal variance; empirical coverage is a descriptive untouched-test diagnostic only. p05–p95 is central 90%, not 95%.

| Horizon | Model-Implied Volatility | Descriptive Test Coverage | Avg Scenario Width (% Price) |
| :---: | :--- | :---: | :---: |
| 1-Day | Rolling Mean (60d) | **90.5%** | ±6.8% |
| 1-Day | GARCH(1,1) | **91.1%** | ±7.0% |
| 1-Day | HAR-RV | **60.4%** | ±2.9% |
| 1-Day | Gradient Boosting | **60.4%** | ±3.0% |
| 1-Day | PyTorch LSTM | **69.4%** | ±3.7% |
| 5-Day | Rolling Mean (60d) | **89.6%** | ±15.2% |
| 5-Day | GARCH(1,1) | **91.0%** | ±15.8% |
| 5-Day | HAR-RV | **54.2%** | ±6.2% |
| 5-Day | Gradient Boosting | **84.2%** | ±12.2% |
| 5-Day | PyTorch LSTM | **83.2%** | ±12.1% |
| 10-Day | Rolling Mean (60d) | **89.5%** | ±21.4% |
| 10-Day | GARCH(1,1) | **91.1%** | ±22.3% |
| 10-Day | HAR-RV | **51.6%** | ±8.1% |
| 10-Day | Gradient Boosting | **86.2%** | ±18.5% |
| 10-Day | PyTorch LSTM | **85.5%** | ±18.2% |
| 20-Day | Rolling Mean (60d) | **88.9%** | ±30.1% |
| 20-Day | GARCH(1,1) | **91.0%** | ±31.7% |
| 20-Day | HAR-RV | **45.6%** | ±10.2% |
| 20-Day | Gradient Boosting | **87.2%** | ±27.3% |
| 20-Day | PyTorch LSTM | **86.4%** | ±27.0% |


## 5. Top Catastrophic Tail Error Diagnostics
### Top 5 Worst Out-of-Sample Losses: PyTorch LSTM
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1211 | 0.2009 | -11.2196 | **8760.72** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.3374 | 0.6047 | -15.8346 | **2289.14** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.1612 | 0.1841 | -6.7993 | **1855.76** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.1756 | 0.2009 | -6.7827 | **1562.62** | False |
| MNST | 2026-07-16 | 20d | High Vol | 6.4299 | 0.1975 | 0.1841 | -6.2324 | **1052.34** | False |


### Top 5 Worst Out-of-Sample Losses: Gradient Boosting
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AMD | 2026-01-23 | 1d | Normal Vol | 0.5201 | 0.0005 | 0.4260 | -0.5196 | **1151013.79** | False |
| FAST | 2026-01-16 | 1d | High Vol | 0.4118 | 0.0004 | 0.2187 | -0.4113 | **933956.52** | False |
| AMD | 2025-10-28 | 1d | Normal Vol | 0.3842 | 0.0008 | 0.9877 | -0.3834 | **229505.94** | False |
| EXC | 2025-06-02 | 1d | Normal Vol | 0.1201 | 0.0003 | 0.2224 | -0.1198 | **132249.94** | False |
| AMD | 2025-05-02 | 1d | Normal Vol | 0.2850 | 0.0009 | 1.0373 | -0.2842 | **105253.89** | False |


### Top 5 Worst Out-of-Sample Losses: HAR-RV
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.1175 | 0.2009 | -11.2232 | **9312.66** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.0776 | 0.1688 | -6.3529 | **6863.21** | False |
| MNST | 2026-07-13 | 20d | High Vol | 6.4303 | 0.0799 | 0.1665 | -6.3503 | **6465.41** | False |
| MNST | 2026-07-14 | 20d | High Vol | 6.4303 | 0.0800 | 0.1670 | -6.3503 | **6457.30** | False |
| MNST | 2026-07-16 | 20d | High Vol | 6.4299 | 0.0858 | 0.1841 | -6.3441 | **5605.61** | False |


### Top 5 Worst Out-of-Sample Losses: Rolling Mean (60d)
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.3202 | 0.2009 | -11.0205 | **1246.05** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.3202 | 0.2009 | -6.6381 | **465.00** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.3206 | 0.1841 | -6.6399 | **464.15** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.7471 | 0.6047 | -15.4248 | **461.40** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.3180 | 0.1688 | -6.1124 | **401.84** | False |


### Top 5 Worst Out-of-Sample Losses: GARCH(1,1)
| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MNST | 2026-07-17 | 1d | High Vol | 11.3407 | 0.2588 | 0.2009 | -11.0819 | **1911.48** | False |
| MNST | 2026-07-16 | 5d | High Vol | 6.9605 | 0.2460 | 0.1841 | -6.7145 | **792.83** | False |
| MNST | 2026-07-17 | 5d | High Vol | 6.9583 | 0.2620 | 0.2009 | -6.6963 | **697.73** | False |
| MNST | 2026-07-15 | 20d | High Vol | 6.4305 | 0.2481 | 0.1688 | -6.1824 | **664.28** | False |
| MRNA | 2026-08-18 | 1d | High Vol | 16.1720 | 0.6267 | 0.6047 | -15.5453 | **658.45** | False |

