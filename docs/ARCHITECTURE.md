# Architecture

## Request flow

The React SPA calls same-origin `/api/` paths. In Compose, Nginx proxies those paths to FastAPI. Fast cache hits return immediately. Cold prediction work is admitted to a bounded executor; identical in-flight identities share one future. The entire blocking flow—Yahoo downloads, feature work, artifact load/training, inference, exchange calendar, and news sentiment—runs outside the event loop.

```text
browser -> Nginx -> validation/rate limit/cache -> bounded coordinator
  -> one immutable target snapshot
  -> technical + calendar + versioned market-context features
  -> compatible versioned artifact OR bounded/coalesced training
  -> inference + exchange dates + response-shape validation
  -> identity cache -> browser
```

Prediction identity includes ticker, horizon, and type at the response/cache layer. Direction artifact identity deliberately uses a fixed 30-session output width; metadata and the Keras signature are checked before a shorter response is sliced.

## Data and ML boundaries

There are 22 ordered features: OHLCV (5), technical (9), market context (4), and cyclic calendar values (4). Market context schema v2 carries the last known close across a closed benchmark session before calculating its return. It never fills from the future. Any source that cannot be aligned from a prior observation fails closed with provenance rather than producing an indistinguishable zero.

Validation supports two real strategies:

- `expanding`: training starts at row zero and grows each fold.
- `rolling`: training uses exactly `min_train_size` rows before each gap.

Each fold has an exact `horizon` and `gap`. A scaler fits only on the fold's training rows. An inner tail of that training fold supplies early stopping; the subsequent evaluation fold is not passed to `fit`. Metrics pool predictions across every forecast output; the direction majority baseline is selected from the training fold. Per-fold diagnostic rows remain first-step views and are labelled separately. Only aggregated out-of-fold results are published. The final serving model may train on all sequences after evaluation.

Reproducibility metadata records Python/NumPy/scikit-learn/TensorFlow versions, seed, deterministic mode, feature schema, validation settings, input data range, snapshot hash, source provenance, and Git commit. Deterministic TensorFlow kernels remain platform-dependent; exact equivalence across different hardware/library builds is not promised.

## Persistence and concurrency

In-process and O_EXCL process locks prevent duplicate same-artifact training. A global semaphore bounds simultaneous training. A candidate version is written to a unique directory, hashed, then activated with atomic replacement of `current.json`; the prior version is not removed first. Readers resolve only activated versions and validate metadata, feature order/count, output width, scaler shape/finiteness, and SHA-256 hashes before Keras/scaler loading. Scalers are JSON, not pickle/joblib.

Count, byte, and free-space quotas evict old artifact roots. Readiness verifies the storage can create a file and retains a configured free-space floor.

## Calendars and fallbacks

Suffix mapping covers `.L`, `.SW`, `.TO`, `.AX`, and `.HK`; unsuffixed instruments use NYSE. `-USD`, `-GBP`, `-EUR`, and `-USDT` pairs use a 24/7 calendar. Unknown dotted suffixes use `NYSE_FALLBACK`, which is returned in response metadata so consumers can identify the assumption.

## Security boundaries

Ticker/model identities are allowlisted before path construction. Public forecast endpoints have per-IP rate limits plus bounded global capacity. CORS allows explicit origins and no credentials. Internal exceptions are not returned. External text renders through React text nodes, and export identity/length checks prevent cross-forecast ZIPs. CSV cells that begin with spreadsheet formula characters are neutralised. Model artifacts are trusted local server state and integrity checked; SHA-256 detects corruption but is not an authenticity signature against an attacker who can rewrite both data and hashes.
