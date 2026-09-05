# Seven-session ranking experiment

## Assessment: share with caveats

The fixed Ridge model did not demonstrate useful ranking ability. Production remains unchanged. No GPU escalation, reserve evaluation, refit, commit, or deployment was performed.

## Design and data quality

- Source: same 286-stock cached parquet basket as the previous comparison; every source checksum verified against its preserved protocol.
- Target: centered within-market ranks of seven-session cumulative log returns. US and UK baskets evaluated separately, not pooled across closing times.
- Model: Ridge alpha=100, latest 25 causal price features, training-only StandardScaler. No tuning.
- Comparators: fixed positive 20-session momentum and equal scores (no ranking).
- Global unique-date 70/15/15 boundaries: validation starts 2023-03-30; reserved block starts 2024-12-10. Training and validation labels crossing their next boundary are purged.
- 375,611 training rows; 116,936 validation rows; 430 validation dates across the markets.
- 10,722 basket rows excluded because the market/date basket had fewer than 20 eligible assets or inconsistent seven-session outcome endpoints. No partial basket removal based on return magnitude.
- Positive origin volume, finite features/targets, unique ticker/date keys required. Historical membership and missing/delisted outcomes remain limitations; this is a current survivor basket.

## Results

Primary metric is date-level Spearman rank correlation (IC). Higher is better; zero means no observed rank association. Equal-score IC is defined as zero by convention, not a computed correlation of a constant.

| Scope | Ridge IC | Momentum IC |
|---|---:|---:|
| ALL | 0.001842 | -0.006653 |
| US | -0.000644 | -0.002953 |
| UK | 0.004148 | -0.016169 |

ALL first averages available market statistics within date, then equally averages dates. It is not stock-count weighted.

At HAC lag 6, Ridge versus the zero-IC reference: mean 0.001842, 95% interval [-0.024813, 0.028498], p=0.892. Ridge versus momentum: difference 0.008495, interval [-0.043146, 0.060136], p=0.747. Lags 12 and 18 are saved as sensitivity checks. These are exploratory, unadjusted comparisons, not evidence of equivalence or a confirmatory test.

Secondary rank MAE in report.json is descriptive only: constant median ranks can minimize absolute rank error without providing any ordering, so that metric alone must not decide whether a ranking model is useful.

## Verification and artifacts

- Two focused tests passed: chronological purge boundaries and market isolation/tie handling with known perfect/reversed rankings.
- Ruff check/format and git diff --check passed for the new implementation.
- protocol.json was written before model fitting and contains source hashes, boundaries, features, exclusions, and caveats.
- validation_predictions.parquet contains per-stock scores and outcomes; daily_scores.parquet contains independently inspectable date/market metrics.
- linear_model.json records fitted coefficients/scaler; report.json contains results and uncertainty.
- Executable implementation: scripts/run_rank_experiment.py; tests: research/tests/test_rank_experiment.py.

## Limitations and next decision

The cache has already been explored in earlier experiments. The reserved block was not scored here but must not be described as historically untouched. Sample identity differs from the preceding per-stock split experiment, so its MAE results are not directly comparable. Complete future outcomes are necessary for scoring and may exclude unavailable/delisted cases; no point-in-time universe claim is made. No costs, turnover, shortability, or executable portfolio returns were tested.

Do not deploy this candidate. This result provides no reason to scale up neural ranking on these features. A later experiment needs a separately specified information source or hypothesis; historical news or genuine historical sector metadata require a coverage and timestamp audit before fitting.
