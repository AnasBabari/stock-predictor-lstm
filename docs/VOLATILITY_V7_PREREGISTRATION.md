# Volatility v7 prospective retraining preregistration

Status: **development preregistered; no certified or releasable v7 model exists**.

This document freezes the next research cycle after the v6 locked reserve was
consumed. It must be read together with [GLOBAL_MODELS.md](GLOBAL_MODELS.md).
Code, data, objectives, selection rules, and future certification boundaries
are fixed here before the prospective period begins.

## Why a new cycle is required

The one-use v6 locked certification reported `status = failed`. Although the
candidate passed the 1-, 5-, and 7-session horizons, it failed the required
3-session asset-transfer guardrail: NMM relative QLIKE was
`1.055824243087597`, above the conservative initial maximum of `1.05`.

Strict rejection applies to the entire candidate:

- no v6 candidate weights may be materialized, signed, promoted, or served;
- passing horizons may not be released separately;
- the consumed v6 reserve may not be reopened, relabelled, or reused for
  selection, tuning, calibration, or certification;
- v6 locked results remain immutable historical evidence only;
- the API must abstain until a different candidate passes a genuinely future
  locked certification.

`scripts/materialize_certified_candidate.py` independently enforces this rule
by requiring the source evidence status to be exactly `passed`.

## Frozen time boundary

- Immutable development panel end: **2026-08-21**.
- Prospective observation period begins: **2026-08-27**.
- The development runner rejects any panel ticker ending after 2026-08-21 and
  requires at least one ticker to end exactly on that date.
- The prospective development fold plan contains no certification rows.
- NMM and MSFT remain deterministic asset-transfer holdouts and cannot enter
  development fitting or model selection.

The existing immutable 69-ticker panel ending on 2026-08-21 is the permitted
development snapshot. A later snapshot is a different dataset identity and
may only be used for the future certification described below.

## Frozen protocol identity

- Protocol: `global-volatility-distribution-v7-prospective`.
- Architecture: `baseline-residual-tcn-v3-objective-selection`.
- Target: `future-rv-total-v1`.
- Feature contract: Deployable Schema v5 in its exact ordered form.
- Input window: 60 market sessions.
- Model horizons: 1, 3, 5, 7, 14, and 30 sessions.
- Selection-required horizons: 1, 3, 5, and 7 sessions.
- Development folds: five calendar-aligned expanding folds.
- Per-fold early stopping: fit-only preprocessing followed by a disjoint
  63-session inner validation region.
- Outer validation: 126 sessions per fold.
- Embargo: 30 sessions.
- Minimum fitting history: 756 sessions.
- Seeds: 41, 42, and 43.
- Primary metric: QLIKE against the causal adaptive calibrated HAR/C2C
  baseline, with the existing CRPS, calibration, coverage, fold-consistency,
  bootstrap, Diebold-Mariano, and Holm controls retained.

## Finite objective comparison

Exactly two profiles may be compared. They use the same panel, examples,
folds, OOF origins, model architecture, seeds, optimizer settings, and
promotion gates.

### `multitask_v1` — incumbent control

| Loss term | Weight |
| --- | ---: |
| QLIKE | 0.60 |
| Variance-only Gaussian CRPS | 0.25 |
| Return location | 0.05 |
| Direction | 0.05 |
| Baseline regularization | 0.05 |

### `volatility_only_v1` — preregistered challenger

| Loss term | Weight |
| --- | ---: |
| QLIKE | 0.70 |
| Variance-only Gaussian CRPS | 0.25 |
| Return location | 0.00 |
| Direction | 0.00 |
| Baseline regularization | 0.05 |

The challenger is justified only by pre-certification v6 development
evidence: every auxiliary return-distribution head was rejected there. The
locked v6 reserve did not determine these weights or the selection rule.

No third profile, hyperparameter sweep, news variant, seed substitution, or
post-result objective adjustment is allowed inside this cycle. Such a change
requires a new protocol version and a new preregistration before execution.

## Development selection rule

A profile is eligible only when the volatility promotion decision is true for
every required horizon and every seed. If neither profile is eligible, the
cycle abstains and no candidate is frozen.

If exactly one profile is eligible, it is selected. If both are eligible, the
challenger displaces the incumbent only when both conditions hold:

1. the median, across required horizons, of challenger relative QLIKE divided
   by incumbent relative QLIKE is at most `0.995`; and
2. the worst required-horizon ratio is at most `1.01`.

Otherwise the incumbent is retained. Quick, partial, non-default, or
single-seed runs are diagnostic screens only and are never freeze-eligible.

Selection creates development evidence, not certification evidence. The
selected model may be refit and stored only as an **unsigned prospective
candidate**. It must not be described as production, certified, or releasable.

## Execution contract

The canonical runner is `scripts/run_prospective_volatility_research.py`.

The quick screen uses both profiles, seed 42, and at most three epochs. The
full comparison uses both profiles, seeds 41/42/43, 60 maximum epochs, batch
size 512, and CUDA mixed precision. Every run directory is append-only: the
runner refuses to write into a non-empty directory unless `--resume` is
supplied. Each report records the Git commit, panel checksum, protocol, fold
identities, OOF-index checksum, loss weights, runtime/GPU identity, promotion
decisions, and strict release policy.

`--resume` is a strict, fail-closed continuation for an interrupted full
comparison only. It accepts only the exact preregistered full comparison
configuration (both profiles, default epochs and batch size, no `--quick`).
Every existing seed record is revalidated before it is skipped:

- filename matches the requested profile and seed exactly,
- JSON parses and contains the required schema,
- protocol version, horizon coverage, and promotion rows are exact,
- OOF identity and per-fold boundaries match the recomputed panel-derived
  fold plan (the OOF SHA binds the ordered validation indices, tickers, and
  dates),
- training trace length satisfies the frozen early-stopping contract
  (`len == min(best_epoch + patience, maximum_epochs)`), so quick screens
  can never satisfy a full resume,
- per-file and per-decision finite-value and ordering invariants hold,
- no duplicate, unknown, or stray file is silently accepted — any unexpected
  file, directory, symlink, `.tmp`, or pre-existing final report aborts,
- a partial or truncated record aborts,
- when an embedded `run_manifest` is present, its panel checksum, objective
  loss weights, training config, device, resample count, protocol fingerprint,
  and architecture are checked exactly; records that predate embedded
  manifests are accepted only with an explicit `--accept-legacy-records`
  attestation after the operator audits the checkpoint, and even then every
  recomputable invariant is still enforced.

Resumed records are not merged blindly. At most the missing seeds are
evaluated; the final report is written atomically only after all six records
have been validated.

The full report is freeze-eligible only when it was generated with the exact
default profiles, seeds, epochs, and batch size and the frozen selection rule
returns `selected`. A resumed directory is freeze-eligible only when its
resume inventory was fully valid under the rules above.

## Future locked certification

Certification may start only after a new immutable market snapshot contains
genuine observations beginning on or after 2026-08-27. The reserve must be
constructed once, after the winner and its complete fitting procedure are
frozen.

The existing protocol requires 252 temporal holdout sessions and targets out
to 30 sessions. Therefore certification cannot be complete until at least 282
market sessions after the prospective start are available. Calendar time will
be longer because weekends and exchange holidays do not count.

Future certification must:

- use no row from the consumed v6 reserve as locked evidence;
- include the frozen NMM/MSFT asset-transfer checks;
- use the unchanged promotion and subgroup guardrails;
- evaluate the frozen objective, architecture, scaler, calibration, ensemble,
  and inference procedure without tuning;
- write one immutable outcome with status `passed` or `failed`;
- reject the entire candidate if any required horizon, seed-consensus rule,
  asset-transfer guardrail, integrity check, or CPU-parity check fails.

There is no partial release. Only a future result with overall status `passed`
may enter materialization, signing, release verification, and deployment.

## News boundary

The matched point-in-time GDELT ablation did not demonstrate incremental value
over the market-only candidate and remains excluded from v7. Live headlines
are context-only. Reintroducing news requires a separate preregistered cycle
with complete archive coverage, matched origins, causal timestamps, and an
incremental-value gate; it cannot be added after viewing v7 results.

## Permitted claims

Before future certification passes, the project may say that v7 is a
prospective development programme and may report clearly labelled OOF
development metrics. It must not claim that v7 is production-certified, that
the failed v6 model is deployable, or that any generated price path is a
certified learned expected return. The production API must continue to abstain
when no verified signed release is available.
