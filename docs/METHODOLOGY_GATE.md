# Signed global-volatility serving methodology gate

This record guards the signed global volatility forecasting serving contract.
Legacy browser training has been completely retired; the frontend purely
interfaces with the certified server global volatility serving route (`/api/v2/forecast`).

The serving methodology contract is pinned to a full check: reported numbers
must never outlive the method that produced them. Production forecasts must come
from verified offline ONNX releases with valid Ed25519 signatures, exact feature
ordering, causal Deployable Schema v5 inputs, and explicit fail-closed abstentions
on uncertified horizons or unverified bundles.

recorded_sha: 39d30fff1b97cceec553d958a0619d86d035c24f
freeze_record_commit: 2122968ca5c225073d66057627eacf0a562593ab

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

1. `npm run test:run` (frontend unit test suite) — all Vitest unit and contract tests pass green.
2. `npm run build` (frontend production build) — clean Vite production bundle.
3. `npm run check:production-bundle` — production bundle verified TensorFlow.js-free.
4. Server volatility contract e2e: `npx playwright test e2e/server-contract.spec.js e2e/fixtures.spec.js` — verified server volatility forecasting contract, 503 abstentions, fail-closed integrity checks, and fixture schema consistency.

Guidance: `python scripts/check_methodology_gate.py` in CI fails when any
guard-path file changed since `recorded_sha` or when the battery listing
drifts; rerun the battery at the new HEAD, update `recorded_sha` to that
battery-verified tree, and commit the record (`freeze_record_commit`) together
with the methodology change.
