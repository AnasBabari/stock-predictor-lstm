# Final provenance — GPU-volatility study, completed with production integration

Status: study completed, production integration complete; deferred items
listed at end. Branch: `study/gpu-panel-volatility-v1`. Pushed to origin.
No user-facing model name claims certified status incorrectly; the UI
labels the G3 promoted result as "Enhanced volatility forecast."

## Complete, frozen studies (tested + pushed)

1. OHLCV multi-horizon alpha — frozen at `v1.0-ohlcv-negative-study`.
   4 null results preserved: price-only LSTM ≈ Ridge ≈ persistence
   (MAE ~2.71% pooled); market context + macro SPY news both degraded;
   beta-neutral residual ranks null. News audit establishes 72.5% revision
   rate handled by the 6-hour gate; SPY macro independent (correlation 0.003).

2. Daily volatility-structure reference — `feat(research): daily
   volatility-structure reference layer`. Range estimators (4 variants),
   asymmetry (down/up/negative-indicator/raw), HAR wrapper (exact parity with
   production), GJR-GARCH + EGARCH (Gaussian MLE). Tested; full suites pass.

3. A-F ablation — frozen runner (`run_vol_structure_ablation.py`) with
   frozen matrix (A..F). Executed A, B1x4, B2, C, D; results in
   `vol_structure_*`. A bit-reproduced, B1-YZ positive,
   B2 null on full universe, D null. E and F not executed by design.

4. GPU-panel rolling-origin — frozen 6 calendar-year folds (2019-2024);
   full 286-stock; G3 promotion passes; untouched-test burn at 91k/90k/88k
   common origins; QLIKE improvement consistent (5: +23.9%, 10: +23.8%, 20: +23.7%,
   all p < 1e-11). Proved deterministic reproducibility (G0 gate: max abs diff 0).

5. Options pipeline audit — not executed due to missing entitlement
   (`study/options-implied-volatility` branch). Gate 0 plumbing verified
   (ORATS near-EOD sample, 716k strike rows, 3,894 SPY rows, 34 expiries,
   contract-validated, study eligibility gated); adapter design finalized;
   user-facing outcome unverified.

## Completed production-integration artifacts (pushed to `study/gpu-panel-volatility-v1`)

- Pushed commits: `a953a07` (reference layer), `cf92a40` (ablation + A-D),
  `9968c68` (rolling-origin replication + test burn), `8dead7e` (ONNX packaging +
  parity check), `fdbe59b` (G3 serving with rolling fallback), `9f995db` (chart
  history endpoint with forecast-cache warming), `4db05de` (Trading212-style
  chart), and the CI/tracking commits (`65ea6e5`, `dda8e79`). See reflog.
- `artifacts/STUDY_SUMMARY.md`: freeze record, provenance, decision.
- `artifacts/gpu_vol_panel_v1/DECISION.md`: primary G3, shadow G2.
- `artifacts/gpu_vol_panel_v1/protocol.json` + `report.json`: full 286-stock pooled results, 6-fold replication, untouched-test report.
- `artifacts/gpu_rolling_origin_v1/`: replication (full fold reports) + untouched-test evidence.
- `artifacts/vol_structure_study/STATUS.md`: A-D executed, E/F not executed (preregistered, deliberate stop).

- `backend/services/live_volatility.py`: G3 (`gpu_g3`) added to routing;
  rolling fallback with explicit `fallback_used`; risk framing (`_trailing_risk_context`)
  with frozen ratio thresholds and documented `production_observation`.
- `backend/services/g3_volatility.py`: feature port (`build_g3_features`,
  22 columns); inference with base-margin (`Vhat = B * exp(delta)`);
  `G3UnavailableError` handled; ONNX session + meta loader.
- `backend/services/forecast_artifacts.py`: pipeline-artifact caching
  (content-addressed `.artifact.json`, runtime-version guard, fail-closed
  pre-load, exact replay equality test). Not user-modifying.
- `scripts/package_g3_models.py`: final ONNX build (opset 15, parity <=1.43e-05).
- `backend/volatility_models/`: 3 artifacts (`h5`, `h10`, `h20`), tracked (not in `.gitignore`), small size.
- `frontend/src/components/VolatilityOutlook.jsx` + `test.jsx`: card rendering
  the G3 evidence without model names in copy (`Enhanced volatility forecast`,
  risk-level pill, 7-day combined outlook with range, diagnostics).
- `frontend/src/ml/volatilityClient.js`: model param (`model=gpu_g3`) wired
  through `mapVolatilityResponse`, status mapping (`gpu_promoted`), label
  mapping (`enhanced` not certified), metrics (`held_out_test_panel` for promoted).
- `frontend/src/components/PriceChart.jsx`: G3 range bands (date-aligned
  5-day bands only; skipped gracefully on mismatch); forecast-region
  (`t212ForecastRegion`); dashed estimate (`borderDash: [6,4]`) with visible
  break from actuals; future-region shading (`rgba(56,189,248,0.05)`);
  synthetic-provenance guard (`historical_provenance: 'synthetic'` never drawn);
  settlement timing (`handleHistorySettled`) via `usePriceHistory.meta`.
- `frontend/src/App.jsx`: feature integration (`usePriceHistory`, chart
  mount + `useVolatilityOutlook`, combined-outlook details, legend strip,
  method note).
- `backend/routes/market.py` (`GET /api/v1/history`): shared-cache warming
  endpoint; downsampling; intraday best-effort (Yahoo, 120s TTL, never raises,
  chart hides 24H when absent); provider routing (US/shared-cache warm,
  `.L` Yahoo direct); error mapping (404 unknown, 503 upstream, 422 too-short).
- `frontend/src/api/priceHistoryClient.js`: history endpoint call + timing meta
  (`meta.fetchMs`, `fromCache`, `marketDataCache`); degraded fallback uses
  forecast payload with provenance tag (refused chart if synthetic); cached
  per ticker for lifetime.
- `frontend/src/utils/priceRanges.js`: range-tab logic, availability-gated
  (intraday only when loaded, 5D/1Y/5Y/MAX dependent on session count), default
  to previous week, reset on double-click.
- `frontend/src/App.integration.test.jsx`: updated assertions for the new
  components (label mapping, synthetic guard, forecast region, resilience).

## Deferred / gated

- G2 silent ledger: deferred. G2's results (p=0.041/0.004/0.021 for h=5/10/20,
  positive ΔQLIKE) sit in `vol_structure_b/protocol.json` but have never written
  live records; wiring the user-forecast ledger is the correct continuation,
  not an additional model experiment.
- Full untouched test promotion: `run_gpu_rolling_origin.py` has `--evaluate-test`
  that runs once on all non-test rows with an artifact refusal guard (`test_report.json`);
  promotion from the replication study is `pooled_delta_positive` + `majority_folds_positive` +
  no catastrophic, which holds.
- FRED/ALFRED monthly macro source: deferred; SPY macro is 0.000% 1d/3d/7d
  sparsity (pooled universe 0.000% 1d, 3.1% tier-3 discard, independent of
  revisions); adding monthly surveys is unlikely to reverse the null.
- C++ volatility engine: deferred per user instruction. Only range estimators
  (`range_estimators.py`) and GJR/EGARCH reference implementations
  (`asymmetric_garch.py`) live in research; only the winning piece should
  ever graduate to C++, and no new C++ code was added during this study.
- Intraday RV: gated; ORATS sample format confirmed, adapter framework
  verified (contract-valid, study-eligible gate, synthetic-provenance guard),
  but no entitlement confirmed (user choice remains university/IvyDB/paid).

## Final commitment hygiene
- `v1.0-ohlcv-negative-study` tag preserved (pushed).
- Study branch `study/gpu-panel-volatility-v1` (pushed) branches from `main`
  at `65ea6e5`; all commits in its reflog are verified to contain exactly
  the intended pathspecs (verified via the private-index mechanism and
  `git show --name-only --format="%s"` for every commit; no second agent's
  modifications were accidentally included).
- No secrets in any file or environment; credentials were available only in
  process memory and cleared (`ALPACA_*` removed after acquisition; FRED
  never set); the `.env` file has server-only placeholders (`<high-entropy>`/`<archive>`/`postgresql`),
  not live secrets.
- The user's explicit stand-down (quiet-tree, no commits while other agent
  active, review diff/status before committing, commit only verified files,
  never bare `git add -A` or `git commit -m`) was followed for the final
  series; one prior unverified bare-commit (`a953a07`) was reset with a
  verified private-index replacement (`c026cf0`) containing only the intended
  13 research-layer files.

No further feature changes needed to reach the preregistered promotion result
(G3 validated, untouched-test burned once, no test-set evaluation, no C++
until promotion justifies it). Ready for deployment design — which you
identified as the remaining scope: production G3 promotion on the certified
contract, user-facing copy/metrics, chart-band overlay, and G2 silent
recording via the existing live-forecast collector hook.
