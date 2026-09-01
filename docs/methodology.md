# Volatility Forecasting Methodology

This document outlines the mathematical foundation, feature engineering, experimental protocol, and empirical evaluation methodology behind the volatility forecasting platform.

---

## 1. Problem Formulation & Forecasting Target

The objective is to predict future realized equity volatility over a forward trading horizon of $H$ sessions ($H \in \{1, 5, 10, 20\}$ sessions) using only historical data available at forecast origin session $t$.

### Forward Realized Volatility Target
For daily close-to-close log-returns $r_{t+k} = \ln(S_{t+k} / S_{t+k-1})$, the annualized forward realized volatility target is defined as:

$$RV(t, H) = \sqrt{\frac{252}{H} \sum_{k=1}^H r_{t+k}^2}$$

- **Strict Causality:** The target evaluation window $[t+1, t+H]$ begins strictly in the session following the forecast origin $t$. Zero future returns participate in feature calculation.
- **Single-Day Jump Noise:** At $H=1$, the target is $RV(t, 1) = \sqrt{252} |r_{t+1}|$, which is dominated by single-session jump noise. As horizon $H$ expands ($H \ge 5$), the sum of squared daily returns averages out idiosyncratic daily jumps.

---

## 2. Leakage-Controlled Experimental Protocol

All empirical benchmarks and model evaluations enforce strict point-in-time chronological boundaries:

```text
|<──────── Train (70%) ────────>|<── H-Embargo ──>|<── Val (15%) ──>|<── H-Embargo ──>|<── Test (15%) ──>|
```

1. **Chronological Splitting:** Data is strictly partitioned chronologically (70% train, 15% validation, 15% out-of-sample test) spanning 2015 to 2026.
2. **$H$-Session Purged Boundary Embargo:** Because an $H$-day target spans $H$ future sessions, an $H$-session blackout embargo is placed between partitions to prevent target-overlap leakage into subsequent evaluation periods.
3. **Validation-Only Model Selection:** Production model selection is frozen strictly on validation QLIKE loss without inspecting test set performance.
4. **Exchange Calendar Synchronization:** Forecast origins and news cutoffs follow the official NYSE trading calendar via `pandas_market_calendars`, failing closed on weekends and exchange holidays.

---

## 3. Evaluated Models

### Statistical & Time-Series Baselines
- **Rolling Mean (60d):** Historical sample standard deviation of close-to-close returns over the preceding 60 trading sessions.
- **GARCH(1,1) MLE:** Causal maximum likelihood parameter estimation ($\omega, \alpha, \beta$) with Gaussian log-likelihood optimization via L-BFGS-B and multi-step geometric propagation:
  $$\sigma_{t+k}^2 = \bar{\sigma}^2 + (\alpha + \beta)^{k-1} (\sigma_{t+1}^2 - \bar{\sigma}^2)$$
- **EWMA ($\lambda=0.94$):** Exponentially weighted moving average with RiskMetrics decay parameter $\lambda = 0.94$.
- **HAR-RV:** Heterogeneous Autoregressive model of Realized Volatility capturing daily, weekly (5d), and monthly (22d) autoregressive volatility components.
- **Persistence (22d):** Naive benchmark predicting the upcoming volatility will equal the preceding 22-session historical realized volatility.

### Machine Learning & Neural Regressors
- **PyTorch LSTM (`SOFTPLUS_VOLATILITY`):** Sequence model with hidden dimension 64, dropout 0.15, and a softplus activation head ($\hat{\sigma} = \text{softplus}(z) + \epsilon$). This structural formulation guarantees strictly positive outputs without near-zero variance collapse on market shocks.
- **Gradient Boosting Regressor:** HistGradientBoosting with L2 regularization and feature sub-sampling.
- **Elastic Net & Ridge Regressors:** Regularized linear regressors with standardized causal feature matrices.

---

## 4. Evaluation Metrics

### Primary Metric: Quasi-Likelihood (QLIKE) Loss
QLIKE is the standard scale-free, asymmetric loss function for volatility and variance evaluation:

$$\text{QLIKE}(\sigma^2, \hat{\sigma}^2) = \frac{\sigma^2}{\hat{\sigma}^2} - \ln\left(\frac{\sigma^2}{\hat{\sigma}^2}\right) - 1$$

- **Properties:** Invariant to scale changes, strictly non-negative, reaches 0 if and only if $\hat{\sigma}^2 = \sigma^2$, and heavily penalizes under-forecasting volatility.
- **Numerical Positivity Floor:** Evaluated with numerical stabilization $\epsilon = 10^{-6}$ to prevent metric distortion while exposing model degradation.

### Secondary Metrics
- **Mean Absolute Error (MAE):** $\frac{1}{N}\sum |\hat{\sigma}_i - \sigma_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N}\sum (\hat{\sigma}_i - \sigma_i)^2}$
- **Paired Asset-Level Bootstrap CI (95%):** 1,000 paired bootstrap resamples of per-asset loss differentials to assess statistical significance across the universe.

---

## 5. Feature Ablation & Stopping Rule Findings

### Three-Way Nested Feature Ablation
1. **Stage A (`PRICE_ONLY` $\to$ `PRICE_PLUS_OHLC`):** Adding intraday range estimators (Parkinson, Garman-Klass, Rogers-Satchell) dramatically improved ML performance and eliminated gradient explosion.
2. **Stage B (`PRICE_PLUS_OHLC` $\to$ `PRICE_PLUS_OHLC_PLUS_MARKET`):** Adding leave-self-out SPY/QQQ market returns and market volatility reduced multi-day test loss across all sectors.
3. **Stage C (`+NEWS` Ablation):** Adding causal financial news sentiment (VADER lexicon, negative intensity, dispersion, volume z-scores) with session-close cutoffs **failed to provide statistically robust incremental predictive power** across the 44-asset universe (all 95% bootstrap CIs spanned zero).

### Strategic Architecture Decision
The production serving core relies on the empirically selected statistical winners (`GARCH(1,1)` at 1-day, `Rolling Mean 60d` at multi-day horizons), while the PyTorch LSTM remains available as an empirical ML comparison. News features were removed from the production forecasting path based on the empirical stopping rule.
