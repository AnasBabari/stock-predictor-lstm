# Market-context validation comparison

Completed on 2026-09-05. Validation only: no test scoring, final refit, deployment, or checkpoint replacement.

## Outcome

The tested market-context features did not improve pooled validation accuracy. Neither augmented model beats persistence on pooled MAE. This is evidence against adopting this particular augmentation now, not proof that market context or sequence models cannot help.

| Model | MAE (%) | RMSE (%) | MAE / persistence |
|---|---:|---:|---:|
| Persistence | 2.711416 | 4.120749 | 1.000000 |
| Original Ridge | 2.708864 | 4.115040 | 0.999059 |
| Context Ridge | 2.712585 | 4.119521 | 1.000431 |
| Original large LSTM | 2.708572 | 4.122432 | 0.998951 |
| Context large LSTM | 2.719219 | 4.135859 | 1.002878 |

Errors are percentage-of-origin-price units, not percentage errors relative to the future price. The context LSTM selected epoch 1 and stopped after epoch 4, restoring the selected checkpoint. Direction accuracy was 51.86%, versus 53.36% for the training-selected majority label evaluated on validation.

## Paired evidence

Positive date-averaged loss difference favors the candidate. At horizon 7, with Bartlett HAC lag 6:

- Original LSTM versus persistence: improvement 0.02431 percentage points; 95% CI [-0.02568, 0.07429], two-sided p=0.341.
- Context Ridge versus original Ridge: improvement -0.01221; CI [-0.03402, 0.00959], p=0.272.
- Context LSTM versus original LSTM: improvement -0.00910; CI [-0.05788, 0.03969], p=0.715.
- Context LSTM versus persistence: improvement 0.01521; CI [-0.02313, 0.05355], p=0.437.

No context comparison survives Holm adjustment across the reported tests. Date-weighted paired estimates need not equal row-pooled MAE differences. Failure to reject is not proof of equivalence.

## Execution and artifacts

- 94,287 validation origins; 660,009 origin/horizon rows; 772 distinct dates.
- Original checkpoint predictions reproduced saved validation metrics before augmentation.
- Original sample identity, ordering, targets, and split indices were checked against the augmented export.
- 25 original features plus 11 context values/quality indicators. US and UK aggregation remain separate; market returns exclude the predicted stock.
- `validation_baseline.parquet` / `baseline_hac.json`: original-model evidence.
- `validation_context_comparison.parquet` / `context_hac.json`: augmented paired evidence.
- `ridge_context.json` and `lstm_context/validation_report.json`: metrics and model configuration.
- `lstm_context/selection_model.pt`: validation-selected checkpoint only.
- `completion.json`: completed, test_scored=false, deployment=false.

The earlier 005545 attempt was interrupted during memory-heavy preprocessing and is explicitly marked incomplete. This completed run uses tested chunked training-only scaler calculation and minibatch normalization, avoiding whole-dataset normalized copies.

Verification: 23 focused GPU-pipeline, Ridge-baseline, and market-context tests passed; git diff --check passed.

## Limitations and decision

These are exploratory, validation-selected results from one fixed neural run. Existing per-stock chronological splits are not globally synchronized calendar splits, and the current basket has survivorship limitations. Full cached history was loaded to reconstruct the dataset; only validation was scored. Do not describe this as an untouched test certification.

Keep the original models and production behavior unchanged. Do not add this feature set to production or begin further architecture tuning based on these results. Any next experiment should have a separate, explicit hypothesis and preserve these comparison artifacts.
