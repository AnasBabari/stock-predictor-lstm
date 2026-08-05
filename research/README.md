# Stock autoresearch harness

This is an isolated offline research harness inspired by the MIT-licensed `autoresearch-win-rtx` experiment loop. It does not run in Render, the browser, or the production request path.

The evaluator uses frozen snapshots, cumulative log-return targets, expanding time-ordered folds, horizon-dependent purge boundaries, persistence-relative MAE/RMSE, and append-only experiment records. Candidate families include persistence, ridge, elastic_net (plus the tuned `elastic_net_a*_l*` grid), compact_mlp, dlinear, and small_tcn.

Run from the repository root with `PYTHONPATH=research` after preparing a frozen CSV snapshot. Transient snapshots, model outputs, credentials, and browser weights must not be committed; the frozen audit snapshots `snapshot.csv` and `snapshot_{SPY,QQQ,MSFT}.csv` are intentionally tracked so every ledger claim is reproducible.

## Audit trail

Every ledger record carries a `snapshot_id` equal to the SHA-256 content hash of the exact CSV bytes the subprocess evaluated (computed with the same parser/reserializer as the eval subprocess, so hashes are matchable in tests). A record with `snapshot_id` `"unknown"` means the subprocess failed to report one and must be treated as unaudited. Runs are tagged (e.g. `sweep-*`, `confirm2-*`, `holdout2-*`) in `research/results/experiments.jsonl`; `REPORT.md` and `experiments.tsv` are regenerated on every trial.

## Multi-window holdout

The final confirmation gate is `scripts/run_holdout.py`, which runs the 4-window block-bootstrap holdout on a frozen snapshot:

```powershell
$env:PYTHONPATH = "research"
python scripts/run_holdout.py snapshot_QQQ.csv --family elastic_net --horizon 20
```

A survivor requires pooled relative MAE/RMSE below one, 95% bootstrap-CI upper bounds below 1.0, and a majority of windows passing the 0.98 gate. As of the latest certified run the survivors are `elastic_net` QQQ h20, `small_tcn` SPY h20, and `small_tcn` QQQ h20; `ridge` and all MSFT candidates were rejected.

The local RTX 2060 has 6 GB VRAM, below the external fork's supported Turing floor. Future PyTorch candidates must use the stock harness's conservative runtime profile and official CUDA wheels.