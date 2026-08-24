"""Checksummed local cache for deterministic volatility-panel examples.

The cache is a research accelerator, not a model artifact or source of truth.
Its identity binds the immutable panel checksum and the complete protocol. A
corrupt or mismatched entry is rejected and rebuilt from the panel snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples

EXAMPLE_CACHE_VERSION = "volatility-panel-examples-v1"
_ARRAY_FIELDS = (
    "features",
    "baseline_variance",
    "realized_variance",
    "cumulative_returns",
    "direction_classes",
    "tickers",
    "origin_dates",
    "origin_closes",
)


class ExampleCacheError(ValueError):
    """A local derived-example cache failed identity or integrity checks."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_fingerprint(panel_dir: Path) -> str:
    manifest_path = panel_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum = str(manifest["pooled_checksum"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ExampleCacheError(
            "panel manifest does not contain a valid pooled checksum"
        ) from error
    if not checksum.startswith("sha256:") or len(checksum) != len("sha256:") + 64:
        raise ExampleCacheError("panel pooled checksum has an invalid format")
    return checksum


def example_cache_key(panel_checksum: str, protocol: VolatilityForecastProtocol) -> str:
    payload = {
        "cache_version": EXAMPLE_CACHE_VERSION,
        "panel_checksum": panel_checksum,
        "protocol": asdict(protocol),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def save_example_cache(
    cache_dir: Path,
    examples: VolatilityPanelExamples,
    *,
    panel_checksum: str,
    protocol: VolatilityForecastProtocol,
) -> None:
    """Atomically save checksummed arrays and their immutable identity."""
    target = cache_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        files: dict[str, dict[str, object]] = {}
        for field in _ARRAY_FIELDS:
            path = temporary / f"{field}.npy"
            np.save(path, getattr(examples, field), allow_pickle=False)
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        metadata = {
            "cache_version": EXAMPLE_CACHE_VERSION,
            "cache_key": example_cache_key(panel_checksum, protocol),
            "panel_checksum": panel_checksum,
            "protocol": asdict(protocol),
            "horizons": list(examples.horizons),
            "feature_names": list(examples.feature_names),
            "rows": len(examples.features),
            "files": files,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(temporary)
            return
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_example_cache(
    cache_dir: Path,
    *,
    panel_checksum: str,
    protocol: VolatilityForecastProtocol,
    mmap_mode: str | None = "r",
) -> VolatilityPanelExamples:
    """Load an entry only after protocol identity and file hashes verify."""
    root = cache_dir.resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ExampleCacheError("example cache metadata is unavailable or invalid") from error
    expected_key = example_cache_key(panel_checksum, protocol)
    if metadata.get("cache_version") != EXAMPLE_CACHE_VERSION:
        raise ExampleCacheError("example cache version does not match")
    if (
        metadata.get("cache_key") != expected_key
        or metadata.get("panel_checksum") != panel_checksum
    ):
        raise ExampleCacheError("example cache identity does not match the panel and protocol")

    arrays: dict[str, np.ndarray] = {}
    file_records = metadata.get("files")
    if not isinstance(file_records, dict):
        raise ExampleCacheError("example cache file manifest is missing")
    for field in _ARRAY_FIELDS:
        filename = f"{field}.npy"
        record = file_records.get(filename)
        path = root / filename
        if not isinstance(record, dict) or not path.is_file():
            raise ExampleCacheError(f"example cache is missing {filename}")
        if path.stat().st_size != record.get("bytes") or _sha256_file(path) != record.get("sha256"):
            raise ExampleCacheError(f"example cache checksum failed for {filename}")
        try:
            arrays[field] = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        except (OSError, ValueError) as error:
            raise ExampleCacheError(f"example cache array is invalid: {filename}") from error

    horizons = tuple(int(value) for value in metadata.get("horizons", ()))
    feature_names = tuple(str(value) for value in metadata.get("feature_names", ()))
    if horizons != protocol.horizons or feature_names != protocol.feature_names:
        raise ExampleCacheError("example cache feature or horizon contract does not match")
    return VolatilityPanelExamples(
        **arrays,
        horizons=horizons,
        feature_names=feature_names,
    )
