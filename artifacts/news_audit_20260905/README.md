# Historical news audit — 5 September 2026

## Decision

Do not start price-plus-news training on the current archive. It is predominantly sparse SEC filing metadata, not a historical headline corpus. Production and source archives were left unchanged.

## Inventory and coverage

Audited all five JSONL archives under data/news/alpaca. The directory name does not describe the actual providers.

- 602 ticker-associated records: 546 sec_edgar, 56 yahoo, zero alpaca.
- AAPL 160; GOOGL 65; MSFT 99; NVDA 101; TSLA 177.
- Five of the 286 cached stock tickers have any archive; 281 have none. No UK ticker has an archive.
- Publication range: 22 July 2015 through 4 September 2026. This long range is supplied by SEC filings, not headline history.
- Yahoo headline range: 27 July through 4 September 2026.
- In the ranking validation window [30 March 2023, 10 December 2024), there are 115 SEC records and zero Yahoo/Alpaca headlines. Thus every headline feature would be absent throughout this validation window.
- Annual total record counts, 2015–2026: 8, 14, 18, 28, 40, 57, 63, 54, 60, 68, 75, 117. Absence of a record cannot distinguish no event from incomplete acquisition.

## Timestamp and duplicate checks

- Zero duplicates on (provider,id,ticker), and zero exact (ticker,headline,published_at) duplicates. This does not exclude syndicated or paraphrased duplicates.
- No null values in stored columns; publication timestamps parse, and none exceed collection time.
- Every record was collected on 4 September 2026. Median publication-to-collection lag is approximately 1,219 days.
- No first_seen_at, available_at, updated_at, original content version, or revision history is stored. Collection in 2026 does not disprove original publication, but cannot prove that today's retrieved text/version was available at the historical origin.
- SEC acceptance timestamps may support a separate filing-event experiment after document/version verification. Filing labels and VADER sentiment must not be represented as historical financial-news sentiment.

## Integrity and code findings

1. All five on-disk byte hashes disagree with their manifest hashes; all five match after universal-newline/LF normalization. merge_news_archive hashes LF text before Windows text-mode writing introduces CRLF. Fix future writes to hash the actual bytes written and version the hashing convention; preserve current archives rather than silently rewriting evidence.
2. build_causal_news_features selects by published_at only. It does not enforce available_at or revision timestamps. Historical use requires explicit provider/version assumptions, not an unqualified point-in-time claim.
3. _resolve_session_closes hardcodes NYSE and falls back to 16:00 New York time on calendar errors. This is unsuitable for UK tickers and can be wrong on early closes. Resolve the correct exchange and fail explicitly when calendar resolution fails.
4. validate_news_archive defaults to one article and checks count, not adequate historical coverage. Its is_valid flag cannot establish training readiness.
5. merge_news_archive keys solely by id across providers and overwrites existing records, including collection metadata. No collision was observed here, but provider-qualified identifiers and preserved first ingestion/version history are needed for reliable future acquisition.
6. SEC acquisition reads the recent submissions block only, not all historical submission files. Its date-range arguments do not prove complete historical coverage.

## Next bounded step

Audit access to a genuine historical headline endpoint, then acquire a small five-ticker pilot matching the existing train/validation dates. Keep SEC filing events separate. Preserve provider identifiers, publication/update timestamps, retrieval time, original responses, pagination completion, request coverage, and documented availability assumptions. Missing history must remain missing—not neutral sentiment.

Before fitting, correct byte hashing, provider-aware deduplication/version preservation, and exchange-specific cutoffs with offline tests. Then quantify ticker/month coverage, source outages, and historical-version uncertainty. Only a sufficiently covered pilot justifies a matched price-only versus price-plus-news experiment.

No paid subscription, collection job, credentials lookup, training, source mutation, commit, or deployment was performed during this audit. This audit establishes that the present archive is insufficient; it does not establish whether the user's provider account can supply the missing history.

## Reproduction

The companion audit.ipynb reads local archives only and reproduces counts, timestamps, duplicates, and both hashing conventions. Source inspection: research/price_forecasting/news_archive.py and scripts/collect_historical_news.py. No statistical inference or model outcome claims are made.
