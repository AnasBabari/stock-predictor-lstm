"""Session-aware, non-pickle cache for normalized daily bars."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_data.base import MarketDataResult
from market_data.normalization import REQUIRED_OHLCV, normalize_daily_bars

MAX_CACHE_FILE_BYTES = 25 * 1024 * 1024


class MarketDataCache:
    """Atomic JSON cache keyed by provider and symbol."""

    def __init__(self, directory: Path | str | None, *, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = bool(enabled and self.directory)
        self._lock = threading.RLock()
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _identity(provider: str, symbol: str) -> str:
        raw = f"{provider.lower()}:{symbol.upper()}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _path(self, provider: str, symbol: str) -> Path:
        if self.directory is None:
            raise RuntimeError("market-data cache has no directory")
        return self.directory / f"{self._identity(provider, symbol)}.json"

    def load(self, provider: str, symbol: str, *, required_session: str) -> MarketDataResult | None:
        if not self.enabled:
            return None
        path = self._path(provider, symbol)
        with self._lock:
            try:
                if path.stat().st_size > MAX_CACHE_FILE_BYTES:
                    return None
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("provider") != provider or payload.get("symbol") != symbol.upper():
                    return None
                data_as_of = str(payload["data_as_of"])
                if data_as_of < required_session:
                    return None
                rows = payload["rows"]
                frame = pd.DataFrame(
                    [row[1:] for row in rows],
                    columns=list(REQUIRED_OHLCV),
                    index=[row[0] for row in rows],
                )
                normalized = normalize_daily_bars(frame, provider=provider, symbol=symbol)
            except (OSError, ValueError, KeyError, TypeError):
                return None
        return MarketDataResult(
            frame=normalized,
            provider=provider,
            data_as_of=data_as_of,
            cache_status="hit",
        )

    def save(self, symbol: str, result: MarketDataResult) -> None:
        if not self.enabled:
            return
        path = self._path(result.provider, symbol)
        payload = {
            "schema_version": 1,
            "provider": result.provider,
            "symbol": symbol.upper(),
            "data_as_of": result.data_as_of,
            "stored_at": datetime.now(UTC).isoformat(),
            "rows": [
                [index.date().isoformat(), *[float(value) for value in row]]
                for index, row in zip(
                    result.frame.index,
                    result.frame.loc[:, list(REQUIRED_OHLCV)].to_numpy(),
                    strict=True,
                )
            ],
        }
        with self._lock:
            fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_name)

    def fresh_entry_count(self, *, required_session: str) -> int:
        if not self.enabled or self.directory is None:
            return 0
        count = 0
        with self._lock:
            for path in self.directory.glob("*.json"):
                try:
                    if path.stat().st_size > MAX_CACHE_FILE_BYTES:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if str(payload.get("data_as_of", "")) >= required_session:
                        count += 1
                except (OSError, ValueError, TypeError):
                    continue
        return count
