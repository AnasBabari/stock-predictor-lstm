# Legacy browser methodology gate

This record guards the retained TFJS rollback/compatibility path. It is not
the production model certification: global volatility uses the locked CUDA
walk-forward and one-shot holdout protocol in GLOBAL_MODELS.md. Keeping this
gate explicit prevents old browser evidence from being mistaken for the
signed production release.

The browser evidence contract is pinned to a full check: reported numbers must
never outlive the method that produced them. Quick/Balanced metrics come from a
single untouched post-purge holdout of the selection model; Research metrics
come from untouched out-of-fold predictions pooled across five expanding
folds. Scalers fit only fitting observations. Direction evidence follows target
contract `cumulative_three_way_v2` (one Down/Neutral/Up softmax call per origin
on the cumulative horizon return); direction baselines are three-class
pre-evaluation base rates matched to each evaluation set — holdout trainCount,
per-fold trainEnd, and per-observation fold rows when pooling out-of-fold
predictions. The served artifact is the final refit, and cached evidence is
superseded whenever the method changes.

recorded_sha: 6374c4c
freeze_record_commit: cb17557e538bc58afbc969c1328848c45a99a9d8

**Field conventions.** `recorded_sha` identifies the battery-verified tree the
evidence was produced on (short or full SHA). `freeze_record_commit` is the
full 40-hex SHA of the commit that last *wrote this record*; it necessarily
descends from `recorded_sha`, so a two-commit dance is expected when evidence
is refreshed: run the battery at HEAD, then let a follow-up commit pin its own
hash here. The gate script validates both fields and the full chain:
existence of both SHAs; `recorded_sha` ancestor-of `freeze_record_commit`;
`freeze_record_commit` ancestor-of **HEAD** (a valid-looking freeze on a
sibling branch certifies nothing); the freeze commit actually modified this
document; staleness of guarded paths relative to `recorded_sha`; and any git
failure fails the check closed rather than skipping it.


## Full check battery

Recorded evidence is valid only when every step below passes at `recorded_sha`
(the tree the battery was re-run on) on a clean worktree:

1. `npx vitest run` (frontend unit suite) — currently 211 tests across 31
   files for frontend units.
2. `npm run build` (frontend production build).
3. Contract e2e: `npx playwright test e2e/server-contract.spec.js e2e/fixtures.spec.js` — server contract and fixture contracts without real TensorFlow.js training.
4. Real-training e2e: `npx playwright test e2e/browser-real-training.spec.js --workers=1` — a real TensorFlow.js model trains in Chromium against deterministic price and direction fixtures; the price run must prove that a reload uses the IndexedDB artifact, and the direction run must render the explicitly labelled matched pre-evaluation base-rate fallback under the cumulative three-way target contract.
5. Temporal-isolation e2e: `npx playwright test e2e/browser-temporal-isolation.spec.js --workers=1` — distorting rows strictly beyond the final-refit boundary must leave metrics and the stored scaler bit-identical between clean and corrupted runs.

Guidance: `python scripts/check_methodology_gate.py` in CI fails when any
guard-path file changed since `recorded_sha` or when the battery listing
drifts; rerun the battery at the new HEAD, update `recorded_sha` to that
battery-verified tree, and commit the record (`freeze_record_commit`) together
with the methodology change.
