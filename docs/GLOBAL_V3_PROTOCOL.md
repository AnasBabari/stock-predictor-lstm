# Global Research & Certification Protocol V3: Cross-Sectional Ranking Skill

---

## 1. Executive Summary & Evolution from V1/V2

### What V1 and V2 Established
The **Phase 1 Global Pipeline (Protocols V1 & V2)** evaluated whether an offline global model could forecast future absolute cumulative stock log-returns ($r_{i, t \to t+h} = \log(P_{i, t+h} / P_{i, t})$) better than a persistence / martingale null hypothesis ($r = 0$).

The development and certification evidence across 265 assets and ~8 years demonstrated:
1. **Model Complexity Penalty**: High-capacity estimators (HistGradientBoost, DLinear, Ridge, Neural) suffered from parameter variance and underperformed persistence on expanding out-of-fold evaluations ($\text{rel-RMSE} > 1.0$).
2. **Shrinkage Dominance**: Simple exponential moving average shrinkage (`rolling_mean_shrunk`) was the only candidate to achieve out-of-fold $\text{rel-RMSE} < 1.0$ ($0.9999$ at 1d $\to 0.9963$ at 30d).
3. **Protocol V2 Holdout Results**: The selected frozen model achieved statistically significant, bit-for-bit reproducible error reductions vs. persistence on the locked 2025–2026 temporal holdout and unseen asset-transfer tickers.
4. **The Directional Reality**: Under Protocol V2's decomposed directional metrics, the apparent ~53%–57% directional accuracy was proven to be **100% market beta drift**. Directional lift over majority class prevalence was exactly **+0.00%**, and Balanced Accuracy was exactly **0.5000**.

### The New Scientific Question in Protocol V3
Protocol V3 does not attempt to rescue absolute directional forecasting or move post-hoc goalposts. Instead, V3 asks a fundamentally distinct, market-neutral question:

> **"Can causal cross-sectional information consistently rank which stocks will outperform other stocks at the same prediction origin?"**

The primary objective is **cross-sectional ranking skill**, evaluated via daily/session Spearman Rank Information Coefficient ($\text{IC}_t$) against leave-one-out market-neutralized relative returns.

---

## 2. Universe & Strict Asset-Transfer ($D/H$) Isolation

Let the total panel universe be partitioned deterministically into:
- **$D$ (Development / Reference Tickers)**: Used for feature empirical reference distributions, target benchmarks, model fitting, and expanding-window validation.
- **$H$ (Held-out Asset-Transfer Tickers)**: Withheld with zero model exposure.

### Isolation Rules
1. **Model Fitting**: $H$ assets never enter training batches or parameter estimation.
2. **Feature Reference Distributions**: For $i \in D$, features are ranked against valid $D_t$. For $k \in H$, feature ranks are computed as the percentile location against the empirical CDF of valid $D_t$. **Held-out assets never modify the rank of any development asset.**
3. **Target Benchmarks**: Development relative targets subtract the leave-one-out mean of valid $D_t \setminus \{i\}$. Held-out relative targets subtract the mean of valid $D_t$. **Held-out future returns NEVER participate in benchmark calculations.**

---

## 3. Causal Feature & Target Engineering

### Feature Contract: `cross_sectional_v3_rank_v1`
At prediction origin $t$, using only information $\le t$, stationary base features are computed:
- Return Structure: `Return_1D`, `Return_5D`, `Return_10D`, `Return_20D`, `Overnight_Return`, `OpenToClose_Return`
- Volatility Structure: `Vol_C2C_20`, `EWMA_Var`, `Vol_Percentile_252`
- Liquidity Structure: `Volume_Surprise`, `Log_Dollar_Volume`, `Amihud_Illiquidity_20`
- Interaction: `Return_20D_x_Vol_C2C_20_CS_Rank`

Ranks are mapped to $[-0.5, +0.5]$ using average ranks for ties. Ranks require at least $N_{\text{min}} = 30$ valid reference assets per session; otherwise marked $\text{NaN}$.

### Target Contract: `relative_forward_log_return_dev_loo_v1`
Forward cumulative log return:
$$r(i, t, h) = \log\left(\frac{P_{i, t+h}}{P_{i, t}}\right)$$

Relative return for $i \in D$:
$$y_{\text{rel}}(i, t, h) = r(i, t, h) - \frac{1}{|D_t \setminus \{i\}|} \sum_{j \in D_t \setminus \{i\}} r(j, t, h)$$

Relative return for $k \in H$:
$$y_{\text{rel}}(k, t, h) = r(k, t, h) - \frac{1}{|D_t|} \sum_{j \in D_t} r(j, t, h)$$

---

## 4. Evaluation Metric & Statistical Inference

### Session-Level Rank IC (Never Pooled Correlation)
For each origin session $t$:
$$\text{IC}_t = \text{Spearman}\left(\hat{s}_{\cdot, t}, y_{\text{rel}, \cdot, t}\right)$$

A daily IC is valid only when $\ge 30$ eligible assets exist and neither scores nor targets are constant.

### Overlapping Target Dependence: Newey-West HAC
Because forward return windows overlap ($t \to t+h$), daily IC observations exhibit serial autocorrelation.
- Hypothesis: $H_0: E[\text{IC}] \le 0 \quad \text{vs} \quad H_1: E[\text{IC}] > 0$
- Fixed Lag Policy: $\text{lag}(h) = h - 1$ ($1\text{d} \to 0, 3\text{d} \to 2, 5\text{d} \to 4, 7\text{d} \to 6, 14\text{d} \to 13, 30\text{d} \to 29$).
- Reports: HAC Standard Error, HAC $t$-statistic, one-sided $p$-value.

### Moving-Block Bootstrap
- Block length policy: $\text{block\_length}(h) = \max(5, h)$.
- 2000 resamples under seed 42.
- Empirical 95% confidence interval on mean IC: $[\text{mean\_ic\_ci\_lower\_95}, \text{mean\_ic\_ci\_upper\_95}]$.

### Multiple Testing: Holm-Bonferroni Correction
Holm step-down correction is applied across the complete family of all $\text{candidate} \times \text{horizon}$ hypotheses in development selection, strictly preserving $(h, \text{candidate\_name})$ pairing.

---

## 5. Development Selection Criteria

A candidate is eligible for selection at horizon $h$ only if ALL 7 pre-registered gates pass:
1. $\text{mean\_spearman\_ic} > 0$
2. $\text{holm\_adjusted\_p} \le 0.05$ (one-sided HAC)
3. $\text{mean\_ic\_ci\_lower\_95} > 0$ (moving-block bootstrap lower bound)
4. $\ge 4 \text{ of } 5 \text{ folds have positive mean IC}$ ($\ge 80\%$)
5. $\text{prediction\_row\_coverage} \ge 90\%$
6. $\text{valid\_ic\_session\_coverage} \ge 90\%$
7. $\text{median\_daily\_breadth} \ge 30 \text{ assets}$

If no candidate passes: `status = "abstain_no_robust_rank_signal"`.

---

## 6. Prospective Certification Protocol

### The Prospective Boundary
- Development Data Cutoff: $\le \text{2026-08-21}$.
- Prospective Certification Holdout: First 252 valid master-market sessions starting **2026-08-24**.
- Maturity Requirement: $252 + \max(\text{horizons}) = 282$ post-cutoff sessions.

### Anti-Peeking Lockout
Before full maturity:
- Certification returns `status = "locked_waiting_for_maturity"`.
- Reports non-performance metadata only (session count, maturity status).
- Zero performance metrics (IC, p-values, direction) are computed or leaked.

### Dual-Population Mandatory Gates
When mature and unlocked via `--open-locked-certification-holdout`, frozen models fit on dev data $\le \text{2026-08-21}$ must pass:
1. Temporal Mean IC $> 0$, Bootstrap Lower $95 > 0$, Holm HAC $p \le 0.05$, Breadth $\ge 30$, Coverage $\ge 90\%$.
2. Asset-Transfer Mean IC $> 0$, Bootstrap Lower $95 > 0$, Holm HAC $p \le 0.05$, Breadth $\ge 30$, Coverage $\ge 90\%$.

---

## 7. Claim Boundaries & Permitted Language

- **Permitted**: "Protocol V3 evaluates whether causal cross-sectional features provide statistically significant out-of-sample rank correlation against relative returns under expanding folds and unseen asset transfer."
- **Prohibited**: Claims of market outperformance, profitable trading strategies, guaranteed directional accuracy, or claiming prospective certification before holdout maturity.
