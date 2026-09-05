# Historical news → realised volatility

This study is separate from price-direction experiments and the blocked options-data study. Production, live-ledger behaviour, and BP.L remain unchanged.

## First executable comparison

`scripts/run_historical_news_volatility.py` collects AAPL/MSFT/NVDA/GOOGL/AMZN from 2019-01-01 through 2026-09-04 into checksummed quarterly chunks. A completed chunk can be reused only when its saved bytes match the manifest. Credentials are supplied via process environment or masked prompts, never saved by this runner.

The first training comparison uses horizons 5, 10, and 20 sessions; the target is the mean of future squared daily log returns. QLIKE is evaluated on variance. MAE/RMSE describe annualized volatility, sqrt(252 × variance), in decimal units. The matched Ridge models fit log variance with alpha 100 and train-only standardization.

- Train: 2019–2022, with labels ending strictly before 2023.
- Validation: 2023–2024, with labels ending strictly before 2025.
- 2025 onward: collected but not scored or used for fitting.
- Arms: OHLCV only; identical model plus news activity; identical model plus activity and VADER sentiment.
- Rolling historical volatility is an additional comparator, not substituted for a learned output.
- Positive paired QLIKE differential favors news. Aggregate across tickers by date before Bartlett HAC inference at H−1, 2(H−1), and 3(H−1). Intervals are exploratory and unadjusted; no significance claim from isolated horizons.

## News semantics and limits

Only Alpaca history enters the first comparison. Missing update timestamps are excluded. Availability is max(publication, provider update); revisions beyond six hours are excluded. These rules rely on provider timestamp semantics and do not establish an independent revision history. Publication buckets plus normalized headlines identify exact-text syndication candidates; this is not semantic near-duplicate detection. The corpus covers current mega-cap survivors, not a representative point-in-time universe.

Each ticker must have at least 100 retained articles in each development partition and news within seven calendar days on at least 20% of sessions before the pilot fits. This is a minimum feasibility criterion, not evidence of comprehensive coverage. Zero-news features describe no retained articles in completed request coverage, not proven absence of real-world news.

`scripts/summarize_historical_news.py` produces yearly counts, source mix, canonical rows, and a corpus manifest. Raw/normalized JSONL files are ignored by Git; manifests contain no API credentials.

## Explicitly remaining after the first comparison

1. Full SEC submissions-file history and AMZN mapping support in the reusable collector. The separate current SEC pull is recent-submissions coverage only, and is not silently merged as complete historical filing history.
2. Source/relevance classification validation and full raw response/version retention beyond hashes. Canonical duplicates now retain their provider list.
3. Fixed event-feature, complete SEC, macro, and cached FinBERT ablations. The deterministic event labels in the pilot are metadata, not yet fitted features.
4. GARCH and nonlinear matched comparisons, then sequence training only if justified.
5. Expansion to 195 US stocks after pilot feasibility, with per-stock/month coverage and selection-bias disclosure.
6. Proven feature serving parity and live-news UI work only after historical evaluation; no automatic deployment or ledger writes.

## Pilot result (2026-09-05)

The completed `historical_news_volatility_pilot_v2` run passed coverage for all five stocks. Development sessions with retained trailing-seven-calendar-day news ranged from 99.0% to 100%. The test partition was not scored, and nothing was deployed.

| Horizon | OHLCV Ridge QLIKE | Activity + sentiment Ridge QLIKE |
| --- | ---: | ---: |
| 5 | 0.51546 | 0.50360 |
| 10 | 0.32685 | 0.32996 |
| 20 | 0.21002 | 0.21535 |

Lower is better. The five-session improvement has an exploratory date-aggregated HAC p-value of 0.033 at lag four, without multiple-comparison adjustment. It is not a general improvement: the longer horizons worsen, and rolling volatility remains better on QLIKE. Both a rolling second moment and the frozen policy's 60-observation sample variance are reported separately. No production promotion follows from this pilot.

The 195-US-stock collection has been initiated using the same quarterly cache. Its completion marker, not the existence of its output directory, determines whether collection finished. No broader-universe models have been fitted yet.

## Reproduce commands

Run from the repository root with the local Python environment:

```powershell
.\.venv\Scripts\python.exe scripts/run_historical_news_volatility.py --corpus-dir data/news/historical_v1 --output-dir artifacts/new_news_volatility_run
```

Use a new output directory. Completed quarterly downloads are reusable; training reports must not be overwritten. A failed or partial run is not a finished benchmark.
