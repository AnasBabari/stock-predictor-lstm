# Learned forecast artifact reuse

`/api/v1/forecast` keeps its existing learned selection, fitting, clipping and response semantics. Set `FORECAST_MODEL_CACHE_DIR` to a private directory to enable fitted artifacts; unset means memory-only caching.

For sklearn candidates, artifacts include the fitted pipeline (including its scaler), final inference inputs, ordered feature schema, target clipping bounds, validation residual bounds, training row count, data cutoff, cache identity, and original evaluation/response metadata. On a cold load the service deserializes the model and performs prediction; it does not rebuild features, select candidates or train. It verifies exact agreement with the saved prices before returning the response.

The existing pretrained LSTM already has weights on disk. Its forecast artifact references the content-hashed checkpoint and records preprocessing/target scaling state and inference inputs; it does not duplicate the checkpoint weights. A missing or changed checkpoint invalidates reuse. The sklearn path has a real fresh-process golden test; checkpoint-path parity is checked at artifact creation/load but has not received equivalent subprocess fixture coverage.

Identity includes ticker, last observation, fingerprint of the full frame, selected model request, candidate configuration, feature/cache implementation versions, implementation bytes, runtime library versions and checkpoint content hashes. Changes invalidate reuse. No old response-only JSON cache is used by the new artifact path.

Writes publish a checksummed content-addressed blob before atomically replacing its manifest. Missing, oversized, corrupt, incompatible or non-equivalent artifacts cause a cache miss and normal learned training. No baseline substitution is introduced. Disk errors do not prevent serving the freshly trained response.

Security: joblib is pickle-based and **must only read service-owned files**. Never accept uploaded artifacts or use a directory writable by untrusted users. SHA256 detects accidental corruption; it is not authentication against an attacker able to replace both files.

Operational limits: files survive process restarts only while their filesystem survives. Render ephemeral storage is not cross-deploy/cross-instance persistence. Distributed coordination, shared durable artifact storage, automatic old-blob retention, expanded telemetry and UK cache unification remain separate work. The bounded in-worker lock remains unchanged. Concurrent processes may still duplicate training, although artifact publication is atomic.

Verification: cold-process serialized-response equivalence, fitted scaler/prediction round-trip, key/runtime/version mismatch, corrupt/missing blob, path traversal rejection, and the existing session/revised-data invalidation and concurrent-request tests. No production values or UI layouts are changed by this storage layer.
