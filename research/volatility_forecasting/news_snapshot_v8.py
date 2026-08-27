"""Historical news snapshot stub for v8.

The historical news lake does not yet exist as a licensed provider.
This module defines the *frozen* schema and the numeric-fallback gate so
that a v8 numeric candidate can be certified today with honest
``news_status=not_certified`` reporting, while the news branch remains a
separate, uncertified experiment.

When a licensed historical archive with stable ``published_at`` +
``first_seen_at`` is acquired, this stub will be replaced by a full
ingestion pipeline that:
- deduplicates by canonical_url + content_hash
- maps entities via point-in-time security master
- enforces ``available_at = max(published_at, first_seen_at) < origin``
- quarantines ambiguous timestamps
- records provider checksums and license_id
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

V8_NEWS_SNAPSHOT_SCHEMA = 1
V8_NEWS_TAXONOMY_VERSION = "v1"


@dataclass(frozen=True)
class NewsSnapshotProvenance:
    snapshot_id: str
    provider: str
    license_id: str
    retrieved_at: str
    article_count: int
    deduped_article_count: int
    coverage_start: str | None
    coverage_end: str | None
    missing_days: list[str]
    content_checksum: str
    news_enabled: bool
    news_status: str  # "not_certified" or "certified"


def news_snapshot_id(provider: str, content_checksum: str) -> str:
    raw = f"{provider}:{content_checksum}".encode()
    return f"news-{hashlib.sha256(raw).hexdigest()[:16]}"


def build_numeric_fallback_news_snapshot(
    *,
    provider: str = "none",
    license_id: str = "not_applicable_numeric_only",
) -> dict[str, Any]:
    """Build the honest numeric-only snapshot manifest (news not certified)."""
    import datetime

    empty_checksum = hashlib.sha256(b"no_articles").hexdigest()
    snap_id = news_snapshot_id(provider, empty_checksum)
    return {
        "schema_version": V8_NEWS_SNAPSHOT_SCHEMA,
        "snapshot_id": snap_id,
        "provider": provider,
        "license_id": license_id,
        "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "article_count": 0,
        "deduped_article_count": 0,
        "coverage_start": None,
        "coverage_end": None,
        "missing_days": [],
        "content_checksum": f"sha256:{empty_checksum}",
        "news_enabled": False,
        "news_status": "not_certified",
        "note": (
            "Historical news lake not yet acquired. This snapshot explicitly marks "
            "the v8 model as numeric-only. Do not claim news_enabled=true or "
            "certified news-enhanced performance until a real archive with "
            "published_at + first_seen_at and provider checksums exists."
        ),
    }


def write_numeric_fallback_news_snapshot(out_dir: Path, **kwargs: Any) -> Path:
    """Atomically write the numeric fallback manifest (for v8 numeric certification)."""
    import os
    import tempfile

    manifest = build_numeric_fallback_news_snapshot(**kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "news-v8-manifest.json"
    if target.exists():
        raise FileExistsError(f"news manifest already exists at {target}")
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=".news-v8-", dir=str(out_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as h:
            h.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            h.flush()
            os.fsync(h.fileno())
        tmp_path.replace(target)
        try:
            if hasattr(os, "O_DIRECTORY"):
                dir_fd = os.open(str(out_dir), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if tmp_path.exists():
            import contextlib as _ctx

            with _ctx.suppress(OSError):
                tmp_path.unlink()
    # Readback
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["snapshot_id"] == manifest["snapshot_id"]
    return target
