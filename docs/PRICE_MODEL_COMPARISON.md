# Price-model training correction and next comparison

The existing `artifacts/tri_exchange_gpu_v2` run is preserved. Its pooled test
MAE ratio versus no change is 1.00326, so it did not demonstrate an aggregate
MAE improvement. Its test metrics describe the validation-selected epoch-1
model, not the separately refitted 26-epoch deployment weights.

## Corrected procedure

- `selection.best_epoch` records the best validation checkpoint.
- `selection.completed_epochs` records work performed, including patience.
- `selection.epochs` remains available and now means the selected epoch count.
- Final refitting uses `best_epoch`, the same initialization seed, and the
  original cosine schedule length (`maximum_epochs`). More training rows still
  means more optimizer steps per epoch; refitting is not weight-identical.
- `selection_model.pt` saves the exact model/scalers used for evaluation.
- `model.pt` is a distinct all-data deployment refit. Test scores must not be
  represented as an evaluation of those refitted weights.
- Existing checkpoint/report files are never overwritten by a new run.

## Next bounded comparison

The implemented runner executes fixed-alpha Ridge first, followed by both GPU
candidates sequentially when `--gpu-comparison` is supplied:

```powershell
.\.venv\Scripts\python.exe scripts/train_ridge_baseline.py --gpu-comparison --output-dir artifacts/price_validation_comparison_v1
```

Use a new output folder for each run. `protocol.json` records cached-file hashes,
partition-index hashes, and settings. `ridge.json` records coefficients, scaler,
and validation results; each LSTM folder contains its selected checkpoint and
validation report. `comparison.json` is written only after requested runs finish.
Ridge is latest-vector only, alpha 100. All candidates use the same price-relative
percentage-point errors. Direction labels come from training per horizon, ignoring
zeros and breaking ties upward. Validation zeros count as incorrect for direction.
Undefined persistence-relative ratios are null, not artificial scores.

Use identical cached data and validation partitions for all candidates. Compare
the current 128-unit/3-layer architecture with a 32-unit/1-layer LSTM and a
Ridge regression on the latest feature vector. Include zero-return persistence
and a direction baseline based on training-only up/down prevalence. Select by
validation MAE, with per-horizon and per-stock breakdowns; do not pick winners
by their already observed test scores. Keep the input features and target
definition fixed. They already use relative features and cumulative log returns.

The validation-only mode does not predict or score the test partition, does not
refit on all data, and writes no deployment `model.pt`. Dataset construction
still loads the cached historical data; this is experiment separation, not a
sealed data-access mechanism.

Example small-model command (run separately from other GPU jobs):

```powershell
.\.venv\Scripts\python.exe scripts/train_gpu_price_model.py --universe broad_300 --hidden-size 32 --layers 1 --embed-dim 16 --batch-size 256 --epochs 20 --patience 3 --validation-only --output-dir artifacts/price_validation_small_v1
```

For the matched larger-model comparison, use hidden size 128 and 3 layers with
the same epoch limit, patience, and other settings in a different output folder.
These are new validation experiments, not a reproduction of the original
80-epoch schedule. Do not advertise test improvements from repeated tuning on
the original test period. A later independent evaluation needs data not used to
make these decisions.

News remains a separate experiment requiring timestamped historical coverage
matched to the stock-price history. Live headlines alone cannot train or
validate the historical news contribution.
