# QUARANTINED — INVALID DIAGNOSTIC ARTIFACT

**Do not sign, promote, deploy, or cite this artifact.**

## Why it is invalid

1. **Contradictory identity:** `winning_family` is `elasticnet` while
   `architecture.encoder_family` is `tcn` and `weights.pt` holds TCN weights.
   The artifact does not represent the model it names.
2. **Single-fold evidence:** generated from one development fold, presented
   as if it were a robust multi-fold result.
3. **Reversed QLIKE:** produced while `qlike_losses` arguments were passed
   realized-first, which can change rankings and relative skill.
4. **Incomplete provenance:** no binding to a full Git SHA, data snapshot,
   universe snapshot, or split manifest.

## Status

- `INVALID_DIAGNOSTIC_ONLY = true` in the freeze JSON
- Sealed test: never validly opened
- Nothing here is certified

Retained solely so the invalid result can be compared against future valid
runs. Remove only with explicit user authorization.
