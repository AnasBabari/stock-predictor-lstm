# Bounded validation comparison

Completed on the existing 286-stock cache, with 412,801 training examples and
94,287 validation examples. The test partition was not scored. No deployment
refit or production update occurred.

| Candidate | MAE (%) | RMSE (%) | MAE / no-change | Direction correct |
|---|---:|---:|---:|---:|
| No-change reference | 2.711416 | 4.120749 | 1.000000 | Not a directional predictor |
| Ridge, latest features | 2.708864 | 4.115040 | 0.999059 | 51.49% |
| LSTM 32 units, 1 layer | 2.715109 | 4.136937 | 1.001362 | 51.73% |
| LSTM 128 units, 3 layers | 2.708572 | 4.122432 | 0.998951 | 52.93% |

Errors are percentage points relative to the origin price, using
`100 * (exp(predicted_log_return) - exp(actual_log_return))`.
The fixed training-majority direction achieves 53.36% on validation.
Zeros count as incorrect for directional scoring, and the majority label is
chosen separately per horizon from training only.

The larger LSTM has the lowest validation MAE, but only improves on no change
by approximately 0.105% relatively. Ridge has the lowest RMSE. No learned
candidate beats the pooled majority-direction baseline. These tiny aggregate
differences do not establish statistical significance or trading profitability.

Both LSTMs selected epoch 1 and stopped at epoch 4 (patience 3). Each selected
checkpoint is preserved; no all-data refit was performed. Do not label this
validation comparison as a new untouched test result or automatically deploy
the lowest-MAE model.

See `protocol.json` for data hashes/settings, `ridge.json` for the Ridge scaler,
coefficients and stock/horizon breakdowns, and each model's
`validation_report.json` for its corresponding evidence.
