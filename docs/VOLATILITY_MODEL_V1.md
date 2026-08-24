# Global Volatility Distribution Model v1

## Decision

The production application will expose one learned model family: a compact,
global, baseline-residual temporal convolutional network (TCN). It forecasts a
distribution of cumulative returns and future realized variance at 1, 3, 5,
7, 14, and 30 trading sessions. Training and certification run offline on the
local RTX machine. Production performs CPU inference from one promoted,
signed artifact. Browser TensorFlow.js training and WebGPU are migration-only
code until removed from the production path.

The primary learned claim is **volatility**, not an exact future closing
price. The UI may show a central price path, but must also show calibrated
intervals, expected move, and the benchmark. A zero or near-zero conditional
mean is not disguised as a failed price prediction; it is presented as the
centre of a non-zero forecast distribution.

## Why this replaces the browser LSTM

The current per-ticker browser LSTM has too little data, repeatedly relearns
similar structure, is hardware-dependent, and is selected using noisy single
asset evidence. It can regress to persistence, produce unstable endpoint
moves, or show a flat fallback while users believe an LSTM produced it.

The replacement pools many assets, learns stationary dynamics, predicts a
variance target whose persistence is empirically stronger than return-mean
predictability, and treats econometric forecasts as anchors rather than weak
competitors. It also separates three questions that the old output conflated:

1. What is the expected cumulative variance?
2. What return interval follows from that variance?
3. Is there statistically demonstrated evidence for a non-zero conditional
   return or directional tilt?

## Research basis and limitations

Ge et al., *Comparing Deep Learning Models for the Task of Volatility
Prediction Using Multivariate Data* (arXiv:2306.12446v2), report strong results
from baseline-residual TCN variants for implied volatility and strong
Transformer results for historical volatility. Their residual target,
exogenous inputs, and auxiliary direction task are adopted as candidate-design
ideas. Their single 70/15/15 split, five-asset universe, MSE-only headline
comparison, and pre-2019 endpoint are not adopted as production evidence.

Research on financial text provides a second candidate input, not a shortcut:
financial-domain embeddings can improve HAR realized-volatility forecasts;
topic-specific sentiment, news frequency, and unusualness have forecasted
volatility and drawdowns; and identified news days have materially higher
volatility. FinBERT also supports domain-specific sentiment inference. Those
results motivate the news ablation below, while point-in-time availability and
our own locked out-of-sample results determine whether it ships.

The locked benchmark therefore uses expanding walk-forward folds, purge and
embargo, asset-transfer holdouts, multiple seeds, QLIKE, probabilistic scores,
coverage, paired loss tests, and subgroup checks. Papers are sources for
candidates, not proof that a candidate works on this application.

## Frozen target contract

- Daily realized variance proxy: overnight squared return plus
  Rogers-Satchell intraday variance from adjusted OHLC.
- Origin: market close at session `t`.
- Cumulative variance target at horizon `h`: sum of proxy rows `t+1..t+h`.
- Cumulative return target: `log(Close[t+h] / Close[t])`.
- Direction target: Down, Neutral, or Up using a volatility-aware neutral band.
- Horizons: 1, 3, 5, 7, 14, and 30 trading sessions.
- Input window: 60 sessions ending at and including `t`.
- Features: ordered deployable stationary Schema v5 features; no future row,
  revised constituent list, or untimestamped news item may enter an origin.

## Matched baselines

Every origin receives a causal log-HAR cumulative variance forecast. At origin
`t`, its fit may use responses observed through `t`; recursive steps beyond
`t` use forecasts, not realized future values. EWMA, GARCH, GJR-GARCH, and a
naive rolling-volatility forecast remain benchmark competitors. The neural
model predicts a bounded log-variance correction around the best frozen
baseline rather than an unconstrained variance level.

For return location, the benchmark is zero cumulative return. A learned return
mean can be shrunk or rejected independently without suppressing a promoted
volatility forecast.

The realized-variance proxy and the conditional variance of cumulative close
returns are related but not numerically interchangeable. Each fold therefore
fits a bounded, per-horizon return-variance scale on its inner information set
by minimizing Gaussian CRPS. The candidate and matched baseline receive
separate calibration using the same pre-evaluation rows. QLIKE continues to
score the unscaled realized-variance forecast; CRPS and interval coverage use
the calibrated return-distribution variance. No outer fold or certification
return participates in calibration.

## Point-in-time financial-news channel

News is an optional exogenous feature group trained and evaluated separately
from the market-only model. Raw article text and embeddings are produced
offline; production receives only bounded numeric aggregates. Each record must
preserve provider ID, canonical URL hash, source, first-seen timestamp,
publication timestamp and its confidence, retrieval timestamp, language,
headline hash, entity links, and license class.

The free, reproducible first implementation combines two sources with distinct
roles. GDELT Event metadata supplies 15-minute first-seen geopolitical,
conflict, sanctions, commodity, and macro-event signals; its event date is
never substituted for `DATEADDED`. SEC submissions supply authoritative
issuer events using acceptance timestamps plus a conservative availability
delay, including 8-K, 10-Q, 10-K, 6-K, and 20-F metadata. SEC records with no
licensed text remain neutral in sentiment rather than receiving synthetic
polarity. Licensed headline/article embeddings, including FinBERT, are a
separate enhancement and cannot be trained only on currently available news
then claimed as a historical feature.

At a close-of-market origin, an article is eligible only when its reliable
publication/first-seen timestamp precedes the exchange-specific information
cutoff. Unknown or date-only timestamps are excluded from same-day features.
Corrections remain new versions; the pipeline never rewrites an older snapshot
with revised text. Duplicate syndications are clustered before aggregation.

Candidate news features include:

- ticker-, sector-, commodity-, country-, and market-level article intensity;
- positive, neutral, and negative FinBERT probabilities, disagreement, and
  absolute sentiment rather than only signed polarity;
- event taxonomy and severity for conflict, sanctions, regulation, monetary
  policy, earnings, guidance, litigation, supply disruption, accidents, and
  mergers;
- novelty/surprise versus trailing topic and entity coverage;
- source diversity, source reliability, event confidence, and lead prominence;
- exponentially decayed 1-hour, same-session, 1-day, 3-day, and 7-day counts;
- cross-asset exposure features, such as oil/shipping/defence sensitivity to a
  geopolitical event, derived from a frozen exposure map rather than hindsight.

News does not directly set a price direction. A geopolitical declaration, for
example, can increase uncertainty and propagate through oil, freight,
defence, airline, chemical, and consumer-input exposures, but the sign and
size are learned from prior point-in-time observations rather than encoded as
a hand-written trade. Direct issuer news, macro news, event severity,
commodity transmission, sector exposure, source quality, novelty, and missing
news are separate auditable channels. The model compares market-only TCN,
market-plus-news TCN, and matched baselines on the exact same origins. News
ships only if it improves QLIKE/CRPS, tail coverage, stress-regime results,
and asset-transfer results without relying on late, revised, duplicated, or
unverifiable stories.

## Candidate architecture

The first promoted candidate must be a causal residual TCN sized for a 6 GiB
RTX 2060 and CPU deployment:

- stationary market-feature projection;
- an optional separately projected news/event channel with missingness masks;
- five causal residual blocks with dilations 1, 2, 4, 8, and 16;
- compact shared temporal representation;
- multi-horizon log-variance residual head;
- volatility-scaled return-location head;
- three-class auxiliary direction head;
- dropout, weight decay, gradient clipping, mixed precision, early stopping,
  and deterministic recorded seeds.

The variance path is optimized with QLIKE plus zero-centred Gaussian CRPS,
both proper distribution scores, and a bounded residual penalty. The return
location uses a separate volatility-standardized robust loss, while direction
uses three-class cross entropy. This separation prevents a noisy auxiliary
price head from distorting the variance that will actually be served when the
mean is rejected. A TFT may be evaluated as a challenger because the cited
paper reports strong historical-volatility results, but it cannot become a
second production choice. Only the locked winner is exported.

## Evaluation and promotion

Development uses five expanding date-aligned outer folds. Each outer evidence
boundary is purged by the maximum label horizon and embargoed by 30 sessions.
Epoch selection uses a separate 63-session tail wholly inside each outer
training fold, with its own 30-session purge; the outer fold is never passed
to early stopping. Scalers fit only the inner fitting rows. At least 20% of
tickers are excluded from all fitting for asset-transfer evaluation. A final
252-session chronological interval is opened once for certification.

Primary evidence:

- QLIKE and QLIKE ratio versus the strongest matched variance baseline;
- variance-only Gaussian CRPS and negative log likelihood around the matched
  zero-return location, plus the full learned-location score as a diagnostic;
- 50%, 80%, and 95% interval coverage and width;
- log-variance MAE/RMSE;
- direction multiclass Brier score, log loss, macro F1, and balanced accuracy;
- return-location MAE/RMSE relative to zero return, reported separately;
- identical-origin market-only versus market-plus-news ablation results.

Statistical inference is clustered by exchange session. Candidate and matched
baseline losses are first averaged across all eligible assets for each origin
date, and the moving-block bootstrap plus Diebold-Mariano/Holm tests operate on
that session sequence. Treating ticker-date rows as independent observations
would pseudo-replicate common market shocks and make confidence intervals and
significance tests too optimistic. Descriptive pooled metrics may still use all
untouched rows, but they cannot replace the session-clustered promotion gate.

Realized volatility, return-distribution intervals, return location, and
direction are promoted independently. A weak interval, return-location, or
direction head cannot veto a statistically supported realized-variance
forecast and cannot ride into production on its verdict. Until an auxiliary
component clears its own matched gate, the UI may report promoted expected
volatility but must use an explicitly labelled baseline interval, a zero-return
location, and/or withheld direction. It must not describe that combination as
a fully promoted learned price distribution.

Conservative initial volatility-promotion defaults require median relative
QLIKE below 0.98, at least four of five folds beating the baseline, worst-fold
relative QLIKE no higher than 1.10, paired statistical evidence after
multiplicity correction, stable seeds, and no material failure by liquidity,
volatility regime, market regime, ticker size, or unseen-asset subgroup.
Return intervals separately require useful calibrated variance-only CRPS and
acceptable zero-centred 80% coverage. Return-location MAE and RMSE must
independently improve on zero return. Direction remains diagnostic until its
fold-specific, pre-evaluation prevalence baseline is available. These are
initial guardrails, not fixed universal requirements; any pre-certification
change creates a new protocol record.

## Production response

A seven-session response must contain:

- last observed close and origin date;
- one central path plus 50%, 80%, and 95% lower/upper paths;
- expected seven-day move in dollars and percent;
- predicted cumulative variance and annualized volatility;
- Down/Neutral/Up probabilities when calibrated;
- model, target, schema, snapshot, and artifact versions;
- metric source and locked certification summary;
- explicit promoted, blended, or baseline-only state;
- news availability and last eligible event timestamp, without implying that
  an article causally determined the forecast.

The central path may remain close to the last price while intervals widen. The
application must never manufacture a directional move merely to make a chart
look active, and must never label a fallback as learned output.

## Execution sequence

1. Validate the causal target and baseline builder with future-perturbation
   tests.
2. Implement the baseline-residual TCN and deterministic GPU trainer.
3. Acquire and freeze a license-reviewed multi-asset market panel; hold NMM and
   MSFT out of iterative tuning where feasible.
4. Acquire and freeze point-in-time news/event snapshots with a separately
   acknowledged provider license and strict timestamp-quality gates.
5. Run econometric baselines, market-only TCN, market-plus-news TCN, news-only
   diagnostic, TCN ablations, and the TFT challenger on the development folds.
6. Freeze the single candidate, configuration, code SHA, market/news digests,
   seeds, and preprocessing state.
7. Open the temporal and asset-transfer certification sets once.
8. Export the winner to signed ONNX and verify Python/ONNX parity.
9. Add one lightweight CPU inference endpoint and remove browser training,
   TensorFlow.js, WebGPU, profile selectors, and local model caches from the
   production frontend.
10. Restore the compact README screenshot layout, adding honest interval bands
    and volatility evidence without reintroducing research-dashboard clutter.
11. Pass unit, leakage, parity, resource, deployment, and visual gates; deploy;
    then run the frozen seven-day acceptance sequence on NMM followed by MSFT
    and capture both UI screenshots and response payloads.
