# Certification Protocol V1 (Historical Standards & Limitations)

This document formalizes the exact semantics, gates, and interpretations of **Certification Protocol V1** (`global-cert-v1`), as executed on the real historical 2025-08-21 to 2026-08-21 temporal holdout.

---

## 1. Historical Mandatory Gates

In Certification Protocol V1, a candidate forecast is marked `decision="pass"` at a horizon $h$ if and only if:

1. **`temporal_relative_rmse <= 1.00`**: The blended out-of-sample forecast on the 252-session future temporal holdout achieves RMSE less than or equal to the zero-drift persistence baseline.
2. **`temporal_relative_mae <= 1.00`**: The blended out-of-sample forecast achieves MAE less than or equal to the persistence baseline.

---

## 2. Descriptive-Only Metrics in V1

The following metrics were computed during V1 certification for diagnostic inspection, but were **not mandatory gates** for `decision="pass"`:

### A. Asset-Transfer Metrics
- **What was calculated**: `transfer_relative_rmse` and `transfer_relative_mae` across 53 whole tickers completely withheld from development fitting and selection.
- **Enforcement in V1**: In Protocol V1, `CertificationGateConfig.require_transfer_pass` defaulted to `False`. While all six horizons achieved $\text{transfer\_relative\_rmse} < 1.0000$, passing the asset-transfer threshold was **not enforced as a blocking gate** for the V1 pass decision.

### B. Direction Accuracy
- **What was calculated**: `temporal_direction_acc = mean(sign(pt) == sign(tgt))`.
- **Enforcement in V1**: Not a blocking gate (`min_direction_accuracy_delta` was unused in the decision logic).
- **Limitation**: For constant global drift estimators (such as `rolling_mean_shrunk`), the model predicts a single constant positive return $\hat{\mu} > 0$ for every sample. Consequently, `sign(pt) > 0` for 100% of observations, meaning the reported direction accuracy (e.g. 56.86% at 30d) **simply measures the empirical prevalence of positive returns** over that time horizon, rather than cross-sectional or stock-by-stock directional discrimination skill.

### C. Brier Score
- **What was calculated**: `temporal_brier = mean((1.0 - y_actual)^2 if is_up_pred else (0.0 - y_actual)^2)`.
- **Enforcement in V1**: Not a blocking gate (`max_brier_score` was unused in the decision logic).
- **Limitation**: Because `is_up_pred` is a hard binary decision ($\in \{0, 1\}$) rather than a calibrated probability $P(\text{Up}) \in [0, 1]$, the Brier score collapses to the raw binary misclassification rate:
  \[
  \text{Brier} = 1 - \text{Direction Accuracy} \approx 1 - 0.5686 = 0.4314
  \]
  This value is descriptive only and **must not be presented as evidence of probabilistic calibration**.

### D. QLIKE
- Declared in configuration structures but not implemented or evaluated for return forecasting tasks.

---

## 3. Development Selection vs Holdout Certification

| Property | Development Selection (Expanding Folds) | Locked Certification Holdout (V1) |
| :--- | :--- | :--- |
| **Data Scope** | 1,758 sessions (2018–2025) across 212 development tickers | 252 sessions (2025–2026) + 53 withheld tickers |
| **Statistical Gating** | Moving-block bootstrap ($R_{0.95} < 1.0$), Diebold-Mariano HAC test ($p < 0.05$), Holm family-wise correction | Point threshold check ($\text{rel-RMSE} \le 1.00$, $\text{rel-MAE} \le 1.00$) |
| **Fold Consistency** | Required $\ge 4 / 5$ fold wins, worst-fold ceiling $\le 1.15$ | Single forward-only untouched test window |
| **$\alpha$ Estimation** | Estimated via performance-proportional shrinkage | **Frozen** — applied immutably, zero retuning |

---

## 4. Scientifically Accurate Wording Standards

### Acceptable Statements
- *"The frozen baseline-anchored blend met the V1 temporal non-degradation threshold ($\text{rel-RMSE} \le 1.00$, $\text{rel-MAE} \le 1.00$) across all six horizons on the 252-session future holdout."*
- *"Descriptive asset-transfer relative-RMSE was also below persistence across 53 withheld assets, though transfer gating was not mandatory in Protocol V1."*
- *"The winning rolling-mean candidate is a constant global drift estimator, so its raw directional accuracy reflects positive return prevalence rather than stock-by-stock directional skill."*

### Prohibited / Misleading Statements
- ❌ *"All horizons passed mandatory temporal and asset-transfer certification gates."* (Transfer was descriptive in V1).
- ❌ *"The model achieved 56.86% directional prediction skill."* (It is positive-class prevalence from a constant positive drift).
- ❌ *"The Brier score of 0.431 proves calibrated probabilistic forecasting."* (It is binary error rate).
- ❌ *"The locked holdout proves statistically significant alpha over persistence."* (V1 evaluated point non-degradation, not a holdout DM test).
- ❌ *"The model beats the market."* / *"Proven alpha."*
