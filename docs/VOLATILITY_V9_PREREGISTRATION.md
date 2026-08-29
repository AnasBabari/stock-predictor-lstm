# Volatility v9 preregistration — global equity volatility forecasting

Status: **frozen; all evaluation contracts, feature schemas, missing-data thresholds, candidate bounds, and promotion criteria immutable.**
Protocol Canonical SHA-256: `d205b6394cc39e0e63e6d5c5bf1f6d4a8ca20ceea8a4917f3963ed44f78523b1`

Machine-readable companion: [`configs/volatility_v9_protocol.json`](../configs/volatility_v9_protocol.json).
Where this document and that file disagree, **the JSON is authoritative** — it is the artifact that
code reads at runtime. This document supplies the reasoning, the definitions, and the commitments
that the JSON can only assert.

No protocol-conforming, certification-eligible v9 experiment has been run. Earlier pre-protocol
diagnostics are invalid, quarantined, and excluded from all decisions. This protocol is frozen
**before** any protocol-conforming v9 model is trained or selected. Its purpose is to make the
research question unchangeable after an inconvenient result appears. If the honest outcome is that
HAR beats every neural candidate, that is the reported result.

---

## 0. Data eligibility — read this first

v9 is preregistered against a **development-only** data estate. This is recorded in the protocol
under `data_eligibility` and it is not a formality:

| Input | Source identifier | Certification eligible |
|---|---|---|
| Point-in-time universe | `development_secondary_source_reviewed` | **No** |
| Market panel | `yfinance_development_cache` | **No** |

Consequences, enforced rather than aspirational:

- Every artifact produced while this is true carries `evidence_role=development_diagnostic_only`
  and `certification_eligible=false`.
- No v9 number may be published, shipped, or cited as evidence of forecasting skill on real markets.
- Stage 2 (attested point-in-time constituent source) and Stage 3 (licensed market panel permitting
  training and derived-model use) are **blocking prerequisites** for any certification claim.

The blocker is stated plainly in the protocol: *"A certified market model requires an attested
point-in-time constituent source and a licensed market panel permitting training and derived-model
use. Until then all results are development evidence only."*

---

## 1. Why v9 exists

v8 built substantial machinery — fail-closed provenance builders, purged splits, GPU training,
ONNX parity, signed packaging — and then could not certify, because the four-market panel was never
acquired. The result was a repository that is *implementation-ready* but *evidence-empty*.

v9 inverts the order. It does not add machinery. It freezes the **question** first, then refuses to
answer it until the inputs are entitled. The three specific failures v9 exists to prevent:

1. **A winner manufactured by the evaluator.** Closed in Phase 1 by hardening
   `select_numeric_champion` (reversed QLIKE arguments, horizon averaging, fake seed replication,
   GARCH cross-contamination, serializer fallthrough). Regression tests:
   `research/tests/test_v9_evaluation_safety.py` (25 tests).
2. **A fixture mistaken for market evidence.** Closed in Stage 0 by renaming the golden fixture to
   `synthetic_csco_like_golden_v1.csv`, pinning its SHA-256 in source code, marking it
   `contains_real_market_data: false`, dedicating it to the public domain (CC0-1.0), and printing a
   mandatory banner on every CLI run.
3. **A protocol rewritten after seeing the answer.** Closed here, in Stage 1.

---

## 2. Frozen v9 identities

| Item | Value |
|---|---|
| Protocol | `volatility-v9` |
| Task | Global equity volatility forecasting |
| Universe | Point-in-time Nasdaq-100, development reconstruction |
| Input window | 60 market sessions |
| Target contract | `future_annualized_realized_variance_log_v1` |
| Feature contract | `deployable_v5_numeric_v9` (26 numeric / 22 news / 48 total) |
| Horizons evaluated | 1, 3, 5, 7, 14, 30 |
| Horizons required for promotion | 1, 3, 5, 7 |
| Split | Chronological 70/15/15 by forecast origin |
| Embargo | 30 sessions |
| Development folds | 5, expanding window |
| Neural seeds | 41, 42, 43 |
| Deterministic seed | 0 |
| Primary metric | QLIKE |
| Family namespace | `global-volatility-v9` |

### Model family namespace

```
global-volatility-v9-<family>[-news]:<sha256-digest-prefix>
```

Reserved identities: `global-volatility-v9-numeric`, `global-volatility-v9-news-fusion`.

Rules:

- A family name describes the **actual architecture**, not the aspiration. This is why `dlinear` is
  conditional and `garch_lstm` is excluded (§11) — both names have been attached to implementations
  that did not match them.
- An identity is bound to a content digest. The same name may never refer to different weights.
- No v9 identity may reuse a v7 or v8 weight or identity.
- Development candidates carry `artifact_role=development_candidate` and are never named as
  certified models.

### Explicit non-primary outputs

v9 does **not** forecast exact future price, does **not** guarantee return, and does **not** treat
direction as a primary claim. The primary product is a volatility forecast and the expected price
range that follows from it. A model that predicts direction well and variance badly is a failure
under this protocol.

---

## 3. Target definition

```
r(i,t)   = log(adjusted_close(t) / adjusted_close(t-1))
RV(i,t,h) = sum over k=1..h of r(i,t+k)^2        # realized variance over the horizon
target    = log(epsilon + RV_annualized)
          = log(1e-8 + RV * 252 / h)
```

- Transformation: `log`
- Epsilon: `1e-8`
- Annualization basis: 252 sessions
- Realized-variance estimator: sum of squared log returns over the horizon

**Why log-variance.** Realized variance is strictly positive and heavily right-skewed. Modelling it
directly forces the network to spend capacity on a positivity constraint. Log-variance is
unconstrained on the real line, closer to Gaussian, and lets a residual model operate additively
without ad-hoc clamping. Forecasts are exponentiated back to variance space before scoring, so the
metric is computed where it matters rather than in the space the network happens to like.

**The argument order is frozen.** QLIKE is asymmetric. The canonical call is
`qlike_losses(forecast, realized)`. Swapping the arguments silently changes rankings and can
manufacture a winner. This is pinned by regression test, not by comment.

---

## 4. Horizons and the no-averaging rule

Evaluated: 1, 3, 5, 7, 14, 30 sessions.
Required for promotion: 1, 3, 5, 7.

> A candidate must show skill at **every** required horizon. A loss at any required horizon cannot
> be offset by gains elsewhere.

This is the single most important selection rule in v9. An average across horizons is a diagnostic
convenience and is labelled as such wherever it appears; it is never a promotion input. A model
that wins at h=1 and h=30 but loses at h=5 has not learned volatility — it has learned two different
things and is mediocre at one of them.

---

## 5. Feature schema

Contract `deployable_v5_numeric_v9`:

- **26 numeric features** — price/return/volatility statistics over the 60-session window
- **22 news features** — defined, but **not included in this cycle** (`news_included_in_this_cycle: false`)
- **48 total**

Rules, all enforced:

1. Transformations fitted on training data only.
2. Cross-sectional statistics use only securities observable at the origin.
3. Missingness indicators are explicit — never imputed silently.
4. Every feature carries an `available_at` timestamp.
5. Exact column order is frozen.
6. No raw future-scaled price level as a central input.

Counts are pinned by module-level constants (`V8_NEWS_ONLY_FEATURE_COUNT=22`,
`V8_TOTAL_FEATURE_COUNT=48`) and by `research/tests/test_repo_schema_consistency.py`, which also
scans the repository for stale numeric claims that contradict them.

---

## 6. Split, purge, embargo, sealing

Split contract `origin_chronological_70_15_15_v1`:

1. Build all valid forecast origins (60-session window + max horizon complete).
2. Sort by canonical origin timestamp.
3. Boundaries at 70% and 85% of sorted origins → train / validation / test.
4. **Purge** every label whose target window crosses a boundary.
5. **Embargo** 30 sessions after each boundary (≥ max required horizon).
6. Fit all scalers, encoders, vocabularies, and feature selection on train rows only.
7. Use validation for early stopping, hyperparameter choice, and candidate selection.
8. **Seal** the test partition. Not inspected before the candidate freeze.

- Split is by forecast origin, never random (`random_split_allowed: false`).
- Sealed target store is separate from development-readable metadata.
- `sealed_test_opened: false` — recorded in the protocol and enforced.

**Inner early-stopping splits must also be date-aligned, purged, and embargoed.** Row-based inner
splits are forbidden (`row_based_inner_split_allowed: false`). A row-based split inside a
chronological outer split reintroduces exactly the leakage the outer split was meant to remove, and
it is the failure mode that made earlier neural results uninterpretable.

### Periods

The realized train / validation / sealed-test date ranges are **not chosen independently**. They are
whatever the 70/15/15 split over sorted forecast origins produces once the licensed panel exists, so
they are recorded as `null` in the protocol until Stage 4 writes the split manifest.

Rules:

- Periods must be contiguous and non-overlapping in origin time.
- Purge and embargo mean the realized ranges may contain gaps at the boundaries. This is expected
  and must **not** be back-filled.
- Once the split manifest is written at Stage 4, the periods are immutable.
- The sealed test period is never used for training, early stopping, hyperparameter choice, or
  candidate selection.
- All three periods must be published in the certification record **even when the outcome is
  negative**. A negative result that does not state which period it was measured on is not
  inspectable.

Writing the periods as `null` rather than guessing them is deliberate: an invented date range would
be cited later as if it had been frozen.

### Asset-transfer policy

Unseen-asset transfer is an **evaluation surface, never a tuning surface**. An asset used to pick a
candidate is not an unseen asset.

- Holdout assets are predeclared before any training, chosen deterministically.
- They are excluded from training, validation, hyperparameter search, feature selection, scaler
  fitting, candidate selection, and blending.
- They are evaluated only after the candidate freeze.
- **NMM and MSFT** are eligible as transfer assets only after an explicit contamination audit
  confirms they were never used for feature selection, scaler fitting, or hyperparameter search in
  this or any prior cycle.

### Missing-data policy

`imputation: none_silent`.

1. Missing values are never imputed without an accompanying explicit indicator feature.
2. Missingness indicators are fitted on train only.
3. A security with insufficient history at an origin produces **no sample** at that origin rather
   than a padded sample.
4. No forward-fill across non-trading sessions.
5. Halts and missing sessions are preserved explicitly, not smoothed away.
6. Cross-sectional statistics degrade gracefully and record their effective sample size.
7. No forecast is emitted from a window whose missingness exceeds the frozen threshold.

At serving time, **if required inputs are missing the model abstains** rather than emitting a
forecast built on substituted values.

---

## 7. Folds

- **5 development folds**, expanding-window chronological.
- Terminal training state may be carried into validation (state continuation), but never from
  validation into test.
- Every fold must be present. A candidate evaluated on folds `{1,2,3}` is not evidence; it is an
  incomplete run, and `select_numeric_champion` rejects it.

---

## 8. Seed policy

- Neural families: seeds **41, 42, 43** — three genuine runs, reported as a distribution.
- Deterministic families: seed **0**, exactly one record per fold.

> Replicating deterministic families across the three neural seeds is **forbidden**. It fabricates
> evidence volume and corrupts every uncertainty estimate built on top of it.

This is the subtlest of the Phase 1 defects and the most tempting to reintroduce. A deterministic
model run three times produces the same number three times. Treating that as three samples shrinks
confidence intervals to zero and makes an unremarkable result look certain.

---

## 9. Metrics and statistical evidence

**Primary selection metric:** QLIKE, called as `qlike_losses(forecast, realized)`.

**Secondary metrics:** MAE/RMSE in volatility space, MAE/RMSE in log-variance space, SMAPE,
coverage, calibration, and relative skill vs HAR.

**Statistical evidence:**

- Paired row-level losses with weekly-clustered block bootstrap, ≥2000 resamples, 95% confidence.
- Diebold–Mariano where assumptions hold.
- Holm correction for multiple comparisons.

Weekly clustering is deliberate. Daily volatility losses are strongly autocorrelated; an i.i.d.
bootstrap over daily rows produces intervals that are far too narrow and makes noise look like
signal.

---

## 10. Promotion gates

### Per required horizon

- Relative skill ratio vs HAR below 1.0
- Bootstrap upper confidence bound below 1.0
- Coverage sufficient
- No non-finite outputs

### Robustness

- No catastrophic fold
- No single-year collapse
- No single-regime collapse
- No single-asset dependence
- Acceptable unseen-asset transfer
- Stable across seeds

### Integrity

- No target leakage
- No scaler fitted after the train boundary
- No sealed-test access
- Serialization and reload parity

> Every gate must pass. Selection is never based on an average that can hide a failing horizon.

If no learned candidate passes, the fallback is a qualifying baseline. If no baseline qualifies,
production retains explicit abstention. Abstention is a legitimate outcome, not a failure to be
engineered away.

---

## 11. Candidate families

**Baselines:** persistence, EWMA, HAR, GARCH, GJR-GARCH, Ridge, ElasticNet.

**Neural:** HAR-residual LSTM, HAR-residual GRU, TCN, Patch Transformer.

**Conditional — DLinear.** Included *only* if a mathematically genuine implementation exists. A
linear model relabelled as DLinear must not be evaluated. DLinear's contribution is its
series-decomposition preprocessing; a plain linear layer with the name DLinear attached is
mislabelling, and it was quarantined as such in Phase 1.

**Excluded — GARCH-LSTM.** Excluded until a genuinely trained recurrent residual hybrid exists. A
fixed GARCH/HAR blend is not a GARCH-LSTM.

### Residual parameterization

Neural candidates forecast a **residual** against the HAR baseline:

```
forecast_variance = baseline_variance * exp(bounded_log_residual)
```

Residual heads are **zero-initialized**, so an untrained model reproduces HAR exactly. This
guarantees the neural candidate can only ever be credited with the value it actually adds, and it
removes the possibility that a neural model "wins" merely by learning what HAR already knew.

### GARCH correctness

GARCH is fitted **per ticker**, never pooled. Fits are cached under a content key
`family:ticker:sha256(training_returns)`, so a cache hit can only occur when the training data is
byte-identical. This closes cross-ticker and cross-fold contamination through the cache.

---

## 12. Experiment budget and stopping rule

| Cycle | Count | Condition |
|---|---|---|
| Baseline | 1 | — |
| Primary neural | 1 | — |
| Corrective neural | 1 | Only for a **documented implementation defect**, not merely because it lost |

> Do not tune indefinitely against the development period. After the budget is spent, deploy the
> qualifying champion, else the qualifying baseline, else retain abstention.

The corrective cycle is the pressure valve that makes the rest of the budget credible, and it is
also the easiest thing to abuse. Losing is not a defect. A crash, a shape error, a mis-wired
residual head, or a scaler fitted on the wrong split is a defect. The distinction is recorded at
the time, not reconstructed afterwards.

---

## 13. Decision hierarchy

Selection proceeds top-down; the first satisfied outcome wins.

| Rank | Outcome | Condition |
|---|---|---|
| 1 | `certified_news_fusion` | News candidate passes standalone gates **and** adds incremental value over the frozen numeric companion **and** beats all negative controls |
| 2 | `certified_numeric_candidate` | Numeric candidate passes every promotion gate |
| 3 | `certified_baseline` | No learned candidate qualifies, but a baseline does |
| 4 | `retain_abstention` | No candidate meets operational and evidence gates |

---

## 14. News — deferred to Stage 11

News features exist in the schema (22 of 48) but are **not** part of this numeric cycle
(`news_included_in_this_cycle: false`). When Stage 11 runs, these ablations are mandatory:

1. Frozen numeric only
2. Numeric + news
3. Same-origin shuffled news
4. Timestamp-delayed news
5. Count-only news
6. Missingness-only
7. Source-quality-only

> News qualifies only if it improves over the frozen numeric companion **and** beats every negative
> control across all required horizons.

Shuffled and delayed controls exist because news features are extraordinarily good at smuggling in
information that was not available at the origin. If shuffling the news does not hurt, the model was
reading volume and cadence, not content — and content is the only thing a licensed news feed is
being paid for.

---

## 15. Integrity state

Recorded in the protocol and updated only by the stage that changes it:

```json
{
  "sealed_test_opened": false,
  "numeric_companion_frozen": false,
  "news_candidate_frozen": false,
  "certified_model": null,
  "v7_modified": false,
  "v8_sealed_test_opened": false
}
```

v7 remains a separate future-prospective experiment and is not modified. The v8 sealed test has not
been opened and is not opened by v9.

---

## 16. Terminal outcomes

Three outcomes are preregistered as acceptable, before any result exists:

- **A** — a learned numeric or news model passes and is deployed.
- **B** — no learned model passes; a certified baseline is deployed and the negative result is
  published.
- **C** — no candidate meets gates; production retains explicit abstention and the result is
  published.

B and C are outcomes, not setbacks. A credible negative result — that a well-implemented global
LSTM does not beat HAR out of sample — is more valuable than a fragile positive one, and it is the
result this protocol is most likely to produce.

---

## 17. Permitted and forbidden claims

**May say:** v9 is preregistered; the evaluation machinery is hardened; the fixture is synthetic and
labelled as such; development diagnostics exist and are labelled development-only.

**Must not say:** that v9 is certified; that any v9 number demonstrates forecasting skill on real
markets; that a baseline is a learned model; that a synthetic regression result is market evidence;
that v8 or v7 evidence has been superseded.

Production serving stays abstaining until a signed, certified v9 artifact passes the release gates.

---

## 18. Relationship to prior cycles

| Cycle | Status under v9 |
|---|---|
| v7 | Separate prospective experiment. Not modified. Not superseded. |
| v8 | Implementation-ready, evidence-empty. Sealed test remains closed. Not superseded. Builds on v8 machinery. |
| Pre-protocol diagnostics | Invalid and quarantined (see `artifacts/_quarantine_invalid_numeric_companion_v9/`). Excluded from all decisions. |
| CSCO fixture | Synthetic software regression fixture. Not market evidence. |
| Phase 1 hardening | Committed. Regression-tested. Prerequisite for v9. |

v9 builds on the v8 infrastructure and does not supersede v7 or v8. It does not inherit any numeric result from v8. It inherits the machinery and re-runs the question
from zero.
