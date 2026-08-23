# Global Research Protocol V3: Pre-Registration Specification

**Protocol Identity**: `global-research-v3` / `global-cert-v3`  
**Development Cutoff Date**: `2026-08-21`  
**Prospective Certification Start**: First valid master session strictly after `2026-08-21` (e.g. `2026-08-24`)  
**Status**: Pre-registered (Frozen Development Specification; Prospective Certification Pending Holdout Maturity)

---

## 1. Primary Research Question & Hypotheses

### Primary Question
> "Do causal cross-sectional features produce stable out-of-sample rank information about future relative stock returns across expanding temporal folds and previously unseen assets?"

### Pre-Registered Feature Hypotheses
- **H1 (Medium-Horizon Momentum Rank)**: Cross-sectional ranks of 5d, 10d, and 20d returns contain positive rank information for relative forward returns at 5d, 7d, 14d, and 30d.
- **H2 (Short-Term Reversal Rank)**: 1d return, overnight return, and open-to-close return ranks mean-revert at short horizons (1d, 3d).
- **H3 (Volatility Conditioning)**: Interaction between return rank and volatility rank (`Return_20D_x_Vol_C2C_20_CS_Rank`) improves rank stability across market regimes.
- **H4 (Liquidity Conditioning)**: Volume surprise, dollar volume rank, and Amihud illiquidity rank provide incremental relative ordering information.

---

## 2. Mathematical Contracts

### 2.1 Cross-Sectional Features (`cross_sectional_v3_rank_v1`)
For development universe $D$ at date $t$:
$$z_{i, t} = \frac{\text{Rank}(x_{i, t}) - 1}{N_{D, t} - 1} - 0.5 \quad \in [-0.5, +0.5]$$
where $\text{Rank}$ uses average ranks for ties across valid development assets $N_{D, t} \ge 30$.

For held-out transfer universe $H$ at date $t$:
$$z_{k, t} = \frac{\sum_{j \in D_t} \mathbf{1}_{\{x_{j, t} < x_{k, t}\}} + 0.5 \sum_{j \in D_t} \mathbf{1}_{\{x_{j, t} = x_{k, t}\}}}{N_{D, t}} - 0.5$$
Held-out assets are mapped against the empirical distribution of $D_t$ without altering any $D_t$ ranks.

### 2.2 Relative Return Targets (`relative_forward_log_return_dev_loo_v1`)
Forward cumulative log-return:
$$r_{i, t, h} = \log P_{i, t+h} - \log P_{i, t}$$

For development asset $i \in D$:
$$y^{\text{rel}}_{i, t, h} = r_{i, t, h} - \frac{1}{|D_t \setminus \{i\}|} \sum_{j \in D_t \setminus \{i\}} r_{j, t, h}$$

For held-out asset $k \in H$:
$$y^{\text{rel}}_{k, t, h} = r_{k, t, h} - \frac{1}{|D_t|} \sum_{j \in D_t} r_{j, t, h}$$

---

## 3. Pre-Registered Evidence & Selection Thresholds

### 3.1 Session-Level Information Coefficient
For each session $t$ with valid asset count $N_t \ge 30$:
$$\text{IC}_t = \text{Spearman}\left(\hat{s}_{\cdot, t}, y^{\text{rel}}_{\cdot, t, h}\right)$$

### 3.2 Dependence-Aware Statistical Inference
- **HAC Lag Policy**: $\text{lag}(h) = h - 1$. One-sided $H_0: E[\text{IC}] \le 0 \quad \text{vs} \quad H_1: E[\text{IC}] > 0$.
- **Moving-Block Bootstrap**: $\text{block\_length}(h) = \max(5, h)$, 2000 resamples, seed 42.
- **Multiple Testing**: Holm-Bonferroni correction across all $(h, \text{candidate})$ pairs at familywise $\alpha = 0.05$.

### 3.3 Selection Criteria
A candidate is selected if and only if:
1. $\mu_{\text{IC}} > 0$
2. $\text{Holm-HAC } p \le 0.05$
3. $\text{Bootstrap } \text{CI}_{95, \text{lower}} > 0$
4. $\ge 4 / 5 \text{ expanding folds have } \mu_{\text{IC, fold}} > 0$
5. $\text{Prediction row coverage} \ge 0.90$
6. $\text{Valid IC session coverage} \ge 0.90$
7. $\text{Median daily breadth} \ge 30 \text{ assets}$

Otherwise, the horizon **abstains** (`status = "abstain_no_robust_rank_signal"`).

---

## 4. Prospective Certification Holdout Gates

| Gate Description | Threshold | Scope | Mandatory |
|---|---|---|---|
| Temporal Mean Rank IC | $\mu_{\text{IC, temporal}} > 0$ | $D$ assets | Yes |
| Temporal Bootstrap Lower 95% Bound | $\text{CI}_{95, \text{lower}} > 0$ | $D$ assets | Yes |
| Temporal Holm-adjusted HAC p-value | $p_{\text{HAC, Holm}} \le 0.05$ | $D$ assets | Yes |
| Temporal Prediction Coverage | $\ge 90\%$ | $D$ assets | Yes |
| Temporal Session Coverage | $\ge 90\%$ | $D$ assets | Yes |
| Temporal Median Breadth | $\ge 30$ assets | $D$ assets | Yes |
| Transfer Mean Rank IC | $\mu_{\text{IC, transfer}} > 0$ | $H$ assets | Yes |
| Transfer Bootstrap Lower 95% Bound | $\text{CI}_{95, \text{lower}} > 0$ | $H$ assets | Yes |
| Transfer Holm-adjusted HAC p-value | $p_{\text{HAC, Holm}} \le 0.05$ | $H$ assets | Yes |
| Transfer Prediction Coverage | $\ge 90\%$ | $H$ assets | Yes |
| Transfer Session Coverage | $\ge 90\%$ | $H$ assets | Yes |
| Transfer Median Breadth | $\ge 30$ assets | $H$ assets | Yes |

---

## 5. Holdout Immutability & Anti-Peeking Governance
1. Evaluation requires exactly 252 prospective origins + 30 subsequent sessions for maturity (282 sessions post-2026-08-21).
2. Certification evaluation requires explicit `--open-locked-certification-holdout`.
3. Once opened, `07_certification.json` cannot be rerun or overwritten within the same run directory.

---

## 6. Implementation Notes & Experimental Provenance

### Historical 252-Session Reserve in Fold Generation
The shared pipeline fold generator (`run_stage_folds`) inherited a 252-session historical reserve from the earlier protocol. Consequently, V3 development selection used 5 expanding out-of-fold validation windows over the first 1,758 trading sessions (the 5 validation folds ending on 15 August 2025; the development calendar containing 1,758 sessions before the historical 252-session reserve beginning 21 August 2025). 

This historical reserve was not inspected or used for candidate selection. Following development selection, the selected 3-day model (`short_term_reversal_rank`) was fit across all 212 development tickers through the full preregistered 21 August 2026 cutoff during Stage 2 model freezing. Prospective certification remains exclusively post-cutoff (evaluating data accumulating after 21 August 2026). The frozen V3 experiment will not be rerun or modified in response to this observation, preserving complete scientific transparency and provenance.
