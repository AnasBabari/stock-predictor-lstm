# Architecture

## Request flow

The React SPA calls same-origin `/api/` paths. In Compose, Nginx proxies those paths to FastAPI. A response-cache hit first revalidates its underlying artifact and is evicted with `503` if that artifact is no longer fresh. Cache misses enter a bounded, coalescing executor, but the worker first requires a fresh validated artifact. Missing or invalid artifacts fail with `503` before market data is fetched. TensorFlow training is reachable only from the operator CLI.

```mermaid
flowchart LR
    operator[Operator pretrain CLI] --> training[Bounded training]
    training --> artifacts[Validated atomic artifacts]
    browser[Browser / React SPA] --> nginx[Nginx]
    nginx --> api[FastAPI: validation, rate limit, cache]
    api -->|cache hit + fresh artifact| response[Response and diagnostics]
    api -->|cache hit + stale artifact| unavailable
    api -->|cache miss| coordinator[Bounded coordinator<br/>coalesced in-flight work]
    coordinator --> artifact{Fresh artifact?}
    artifact -->|no| unavailable[503 unavailable]
    artifact -->|yes| market[Market data and features]
    market --> inference[Inference]
    inference --> response
    response --> browser
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

Operator-controlled training uses in-process and O_EXCL process locks to prevent duplicate same-artifact work. A global semaphore bounds simultaneous CLI training; public HTTP requests cannot acquire it. A candidate version is written to a unique directory, hashed, then activated with atomic replacement of `current.json`; the prior version is not removed first. Readers resolve only activated versions and validate metadata, feature order/count, output width, scaler shape/finiteness, and SHA-256 hashes before Keras/scaler loading. Scalers are JSON, not pickle/joblib.

Count, byte, and free-space quotas evict old artifact roots. Readiness verifies the storage can create a file and retains a configured free-space floor.

## Calendars and fallbacks

Suffix mapping covers `.L`, `.SW`, `.TO`, `.AX`, and `.HK`; unsuffixed instruments use NYSE. `-USD`, `-GBP`, `-EUR`, and `-USDT` pairs use a 24/7 calendar. Unknown dotted suffixes use `NYSE_FALLBACK`, which is returned in response metadata so consumers can identify the assumption.

## Security boundaries

Ticker/model identities are allowlisted before path construction. Public forecast endpoints have per-client rate limits plus bounded global capacity. Forwarded addresses affect the limiter only when the direct peer is an exact configured trusted proxy; all other callers are keyed by their direct address. CORS allows explicit origins and no credentials. Internal exceptions are not returned. External text renders through React text nodes, and export identity/length checks prevent cross-forecast ZIPs. CSV cells that begin with spreadsheet formula characters are neutralised. Model artifacts are trusted local server state and integrity checked; SHA-256 detects corruption but is not an authenticity signature against an attacker who can rewrite both data and hashes.
