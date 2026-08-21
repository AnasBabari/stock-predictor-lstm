# Browser methodology gate

The browser evidence contract is pinned to a full check: reported numbers must
never outlive the method that produced them. Quick/Balanced metrics come from a
single untouched post-purge holdout of the selection model; Research metrics
come from untouched out-of-fold predictions pooled across five expanding
folds. Scalers fit only fitting observations. Direction baselines derive from
pre-evaluation labels. Direction evidence is reported per forecast day. The
served artifact is the final refit, and cached evidence is superseded whenever
the method changes.

recorded_sha: f3361d1
freeze_record_commit: f3361d113c41f9c30449fb8131bcf73ce06894ce


## Full check battery

Recorded evidence is valid only when every step below passes at `recorded_sha`
(the tree the battery was re-run on) on a clean worktree:

1. `npx vitest run` (frontend unit suite) — currently 152 tests across 23
   files for frontend units.
2. `npm run build` (frontend production build).
3. Contract e2e: `npx playwright test e2e/server-contract.spec.js e2e/fixtures.spec.js` — server contract and fixture contracts without real TensorFlow.js training.
4. Real-training e2e: `npx playwright test e2e/browser-real-training.spec.js --workers=1` — a real TensorFlow.js model trains in Chromium against the deterministic fixture for price and direction; the direction run must fall back to the majority-class baseline and render per-day direction evidence.
5. Temporal-isolation e2e: `npx playwright test e2e/browser-temporal-isolation.spec.js --workers=1` — distorting rows strictly beyond the final-refit boundary must leave metrics and the stored scaler bit-identical between clean and corrupted runs.

Guidance: `python scripts/check_methodology_gate.py` in CI fails when any
guard-path file changed since `recorded_sha` or when the battery listing
drifts; rerun the battery at the new HEAD, update `recorded_sha` to that
battery-verified tree, and commit the record (`freeze_record_commit`) together
with the methodology change.