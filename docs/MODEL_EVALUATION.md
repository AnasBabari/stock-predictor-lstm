# Model evaluation and promotion

Price forecasts are evaluated as direct, date-specific horizons (currently 1,
5 and 20 trading sessions in the offline benchmark). A model is not described
as “accurate” from a price-error score: the reporting layer keeps regression
error and directional classification separate.

## Reproducible benchmark

Run the operator-only benchmark with a frozen market-data snapshot identifier:

```powershell
uv run --project backend python backend/benchmark.py --ticker AAPL --output reports/aapl.json
```

Each candidate sees identical expanding walk-forward folds. Feature scaling is
fitted only on the training slice of each fold, a horizon-sized purge separates
training and validation, and the origin price is retained for every forecast.

The report provides MAE, MSE, RMSE, MAPE, bias, R², MASE, RMSSE and relative
MAE/RMSE against persistence, both per horizon and pooled across
origin–horizon pairs. Directional accuracy compares the predicted movement
with the price known at that specific origin; it does not use `diff` over a
flattened multi-horizon array.

## Promotion gate

No model is promoted merely because it wins a single split. The default gate
requires at least a 5% pooled improvement in both MAE and RMSE over
persistence, MASE and RMSSE below one, wins in at least four folds, and no
fold worse than 1.25× persistence RMSE. A rejected report is valuable evidence
that the baseline should continue serving.

## Training lifecycle

Walk-forward folds provide published out-of-fold metrics. The final model uses
a purged tail only to select its epoch count, then a fresh model is refit on
all available labelled samples for that selected number of epochs. This keeps
the production artifact distinct from the selection model.

## News features

Live Yahoo headlines are descriptive response context only. Historical news
features require a licensed, timestamped archive. Records without a publication
time are excluded; every session sees only earlier articles. The current
feature builder creates exponentially decayed sentiment, effective article count, and
sentiment-confidence series. They must enter a controlled ablation and pass
the same promotion gate before being added to a production model.
