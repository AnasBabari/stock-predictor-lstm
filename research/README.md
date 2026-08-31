# Stock autoresearch harness

This is an isolated offline research harness inspired by the MIT-licensed `autoresearch-win-rtx` experiment loop. It does not run in Render, the browser, or the production request path.

The evaluator uses frozen snapshots, cumulative log-return targets, expanding time-ordered folds, horizon-dependent purge boundaries, persistence-relative MAE/RMSE, and append-only experiment records. Candidate families include persistence, ridge, elastic_net (plus the tuned `elastic_net_a*_l*` grid), compact_mlp, dlinear, and random_features_ridge.

> [!NOTE]
> Family rename (2026-08): the candidate formerly registered as `small_tcn` was renamed to `random_features_ridge`. Review showed it is a fixed random nonlinear projection with a Ridge readout — it contains no convolution, no learned kernels, and no dilation. Historical ledger rows keep the original `small_tcn` string; the name mapping (`LEGACY_FAMILY_ALIASES`) applies only at deserialization/reporting boundaries. The backend's `SmallTCNForecaster` is a genuine causal dilated TCN and is unrelated to the renamed research family.

Run from the repository root with `PYTHONPATH=research` after preparing a frozen CSV snapshot. The standalone simplified benchmark under `scripts/run_volatility_benchmark.py` bootstraps its repository import path itself. Transient snapshots, model outputs, credentials, and browser weights must not be committed; the frozen audit snapshots `snapshot.csv` and `snapshot_{SPY,QQQ,MSFT}.csv` are intentionally tracked so every ledger claim is reproducible.

## Audit trail

Every ledger record carries a `snapshot_id` equal to the SHA-256 content hash of the exact CSV bytes the subprocess evaluated (computed with the same parser/reserializer as the eval subprocess, so hashes are matchable in tests). A record with `snapshot_id` `"unknown"` means the subprocess failed to report one and must be treated as unaudited. Runs are tagged (e.g. `sweep-*`, `confirm2-*`, `holdout2-*`) in `research/results/experiments.jsonl`; `REPORT.md` and `experiments.tsv` are regenerated on every trial.

## Multi-window holdout

The final confirmation gate is `scripts/run_holdout.py`, which runs the 4-window block-bootstrap holdout on a frozen snapshot:

```powershell
$env:PYTHONPATH = "research"
python scripts/run_holdout.py snapshot_QQQ.csv --family elastic_net --horizon 20
```

A survivor requires pooled relative MAE/RMSE below one, 95% bootstrap-CI upper bounds below 1.0, and a majority of windows passing the 0.98 gate. As of the latest certified run the survivors are `elastic_net` QQQ h20, `random_features_ridge` SPY h20 (recorded under its former name `small_tcn`), and `random_features_ridge` QQQ h20 (likewise); `ridge` and all MSFT candidates were rejected. See "Certification status" below for the provenance caveats that apply to these records.

The local RTX 2060 has 6 GB VRAM, below the external fork's supported Turing floor. Future PyTorch candidates must use the stock harness's conservative runtime profile and official CUDA wheels.
## Certification status

Ledger schema v2 records carry auditable provenance (git commit, harness code
hash, snapshot content hash, window definitions, per-window metrics, bootstrap
CIs, decision reason, and the multiplicity policy). `scripts/run_holdout.py`
appends such a record for every completed holdout, and `REPORT.md` /
`experiments.tsv` are regenerated deterministically from the ledger.

All 19 currently kept records predate this schema: they carry
`snapshot_id: "unknown"` and no stored CIs, so the report labels every one of
them **LEGACY_UNAUDITED**. Per `program.md` rule 9 they must not be presented
as certified. Re-certifying any survivor requires one locked holdout run on a
newly frozen snapshot (new untouched data), which will append a schema-v2
record and update the generated report automatically.
