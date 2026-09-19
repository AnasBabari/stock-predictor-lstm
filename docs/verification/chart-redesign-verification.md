# Chart redesign: implementation and verification

## Scope

Stock selection requests history, a learned seven-session price forecast and
5/10/20-session volatility responses. The chart displays only a validated
midpoint path; unavailable or incompatible forecasts leave usable history
visible. Supporting information stays in tabs and expandable details.

## Correctness changes

- Seven positive finite lower/upper pairs and seven strictly increasing dates
  after the origin are required. Null, empty, boolean, inverted, truncated and
  missing-origin data cannot create a displayed estimate.
- Forecast and history ticker, origin date, price at display precision and
  supplied quote units must agree. Midpoints, dates and the Day 7 summary use
  the same adapter, including its explicit unavailable reason.
- Removed client-side hardcoded price/model/backtest substitution on HTTP 404.
  The learned API endpoint remains the only source of price forecasts.
- Removed fabricated news fallback. Empty or failed news remains empty or
  unavailable; verification headlines are explicitly labelled examples.
- Ledger reads use the configured API base, ticker+horizon+base cache key,
  cancellation, and stale-response protection. Public checks create no records.
- Expanded chart blocks background interaction, traps focus, supports Escape,
  restores focus, and keeps its axes inside the viewport. Supporting tabs
  support arrows/Home/End. Touch scrolling is not intercepted by chart panning.
- Missing volatility risk is unavailable, not implicitly moderate. Failed
  horizons are named while successful horizons remain visible. Source and
  fallback disclosures come from each response.

## Volatility serving investigation

The original saved five-session file was misnamed MSFT but contained **AAPL**:
origin 2026-09-09, close 315.3399963378906, cumulative variance
0.9895935637378549, annualized sigma 7.0622599507797705 (706.23%). The saved
twenty-session MSFT response used close 491.6499938964844 on 2026-09-09,
cumulative variance 1.0327584982543667 and annualized sigma
3.607319930087297 (360.73%). The originals remain in Git history at `f2dbeee`.

Two serving defects were demonstrated:

1. The old packaging script trained new squared-error regressors instead of
   exporting the recorded QLIKE-trained frozen boosters.
2. ONNX includes XGBoost's default intercept. Serving failed to remove it before
   applying the rolling base margin, and clipped the correction rather than
   the complete log variance.

Packaging now verifies the original booster hashes, feature order, objective
and freeze verification. It exports without training. The deployed feature
builder is tested against the research builder, including dirty-input cases.
There is no additional scaler in this 22-feature tree-model path; ordered
features are converted to float32 for both XGBoost and ONNX.

For rolling cumulative variance B, inference is:

```text
B = sample_variance(last 20 daily log returns) * horizon
V = exp(clip(ONNX(features) - graph_base_score + log(max(B, 1e-12)), -30, 10))
annualized_sigma = sqrt(V * 252 / horizon)
```

Clipping is the original model's log-variance rule, not a plausibility cap.
The ONNX bytes are checksum-verified; obsolete metadata or a corrupt graph
cannot silently load as the reviewed model.

Identical feature matrices were compared with the **original frozen booster**
using explicit base margins and with ONNX. The check covered 2,870 local MSFT
rows and base variances ranging from 1e-8 to 0.1. Maximum relative variance
error was 2.903e-5, below the predeclared export tolerance of 5e-5.

Latest local audit row: MSFT, 2026-08-27, close 505.05999755859375:

| Sessions | Original booster variance | Served ONNX variance | Annualized sigma |
| --- | ---: | ---: | ---: |
| 5 | 0.001513232922731 | 0.001513231071600 | 27.61645% |
| 10 | 0.003693817795939 | 0.003693825599726 | 30.50974% |
| 20 | 0.008561997588339 | 0.008561957992872 | 32.84519% |

**Limitations:** the saved September 9 responses did not preserve their full
input OHLCV/feature matrix. Their exact old input vectors cannot be recreated
from the response alone. The same-input parity comparison above uses the
available local August 27 dataset, not a claim of replaying the September 9
observations. Export parity is not model-skill validation or certification.
Evidence is labelled `validation_panel` and `learned_model`, not a claimed
untouched-test result. The statistical `model=auto` policy is unchanged.

## Verification executed

- Frontend: 171 tests across 26 files passed.
- Backend and research: 271 tests passed, including 13 G3 tests. Eleven
  existing Starlette/EGARCH warnings remain; no failures or skips in this run.
- Browser: 16 Playwright checks passed (8 fixture contracts and 8 rendered-flow
  checks), including invalid responses, partial failure, modal focus, tabs,
  blank first visit, and actual mobile touch scrolling.
- Production frontend build, Ruff check, Ruff format check and diff whitespace
  check passed.
- Export parity executed on the local frozen boosters, not mock predictors.

Windows test runs use a workspace-local pytest base directory because the
default system pytest temporary directory was inaccessible. Browser processes
required permission to launch. Neither workaround changes assertions.

## Screenshots and release boundary

`ui-success-desktop.png`, `ui-success-mobile.png`, `ui-expanded-desktop.png`,
`ui-partial-volatility-desktop.png` and `ui-partial-volatility-mobile.png`
are **fixture-backed local UI verification**, not live market results.
The three abridged API samples likewise identify `local_fixture`; the full
browser fixtures are in `frontend/e2e/uiFixtures.js` with distinct horizons.

The intended Vercel project is `stock-predictor-lstm`
(`prj_xVDiEtgqOpmZKeyzNEPC9tNSOOe4`), production domain
`stock-predictor-lstm-two.vercel.app`, repository
`AnasBabari/stock-predictor-lstm`, branch `main`.

Production publishing and live API/chart checks are pending at this local
verification checkpoint. They must not be inferred from the fixture images.
