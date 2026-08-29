"""Resumable, bounded construction of historical GDELT daily snapshots."""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

from .gdelt import (
    GdeltDailyAggregationStats,
    GdeltV1DailyArchive,
    TickerAliasMatcher,
    aggregate_gdelt_v1_daily_lines,
    compile_ticker_aliases,
)
from .news import NewsEvent, NewsLicenseNotAcknowledged
from .news_snapshot import NewsSnapshotError, load_news_snapshot, save_news_snapshot

MAXIMUM_COMPRESSED_ARCHIVE_BYTES = 100_000_000
MAXIMUM_UNCOMPRESSED_ARCHIVE_BYTES = 750_000_000
MAXIMUM_ZIP_COMPRESSION_RATIO = 200.0


class GdeltArchiveError(RuntimeError):
    """A GDELT source archive failed bounded download or verification."""


class GdeltArchiveNotFound(GdeltArchiveError):
    """The official provider has no archive for the requested day."""


def load_ticker_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GdeltArchiveError("ticker alias file is missing or invalid JSON") from error
    if not isinstance(payload, dict):
        raise GdeltArchiveError("ticker aliases must be a JSON object")
    aliases: dict[str, tuple[str, ...]] = {}
    for ticker, values in payload.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise GdeltArchiveError("ticker alias keys must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise GdeltArchiveError("every ticker must have at least one alias")
        normalized = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
        if not normalized or any(len(value) < 3 for value in normalized):
            raise GdeltArchiveError("ticker aliases must contain at least three characters")
        symbol = ticker.strip().upper()
        if symbol in aliases:
            raise GdeltArchiveError("ticker aliases contain duplicate normalized symbols")
        aliases[symbol] = normalized
    return aliases


def download_gdelt_archive(
    archive: GdeltV1DailyArchive,
    destination: Path,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    """Download one generated official URL with a strict compressed-size cap."""
    parsed = urlparse(archive.url)
    if parsed.scheme != "https" or parsed.hostname != "data.gdeltproject.org":
        raise GdeltArchiveError("GDELT archive URL is outside the official HTTPS host")
    request = urllib.request.Request(
        archive.url,
        headers={"User-Agent": "StockLSTM research snapshot builder/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "data.gdeltproject.org":
                raise GdeltArchiveError("GDELT download redirected outside the official host")
            raw_length = response.headers.get("Content-Length")
            if raw_length and int(raw_length) > MAXIMUM_COMPRESSED_ARCHIVE_BYTES:
                raise GdeltArchiveError("GDELT archive exceeds the compressed-size limit")
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAXIMUM_COMPRESSED_ARCHIVE_BYTES:
                        raise GdeltArchiveError("GDELT archive exceeded the download byte limit")
                    handle.write(chunk)
        if written == 0:
            raise GdeltArchiveError("GDELT archive download was empty")
        os.replace(temporary, destination)
    except HTTPError as error:
        error_type = GdeltArchiveNotFound if error.code == 404 else GdeltArchiveError
        raise error_type(
            f"GDELT archive download failed for {archive.archive_date} (HTTP {error.code})"
        ) from error
    except (OSError, ValueError) as error:
        raise GdeltArchiveError("GDELT archive download failed") from error
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_downloaded_archive(
    path: Path,
    *,
    archive: GdeltV1DailyArchive,
    ticker_aliases: Mapping[str, tuple[str, ...]] | TickerAliasMatcher,
) -> tuple[tuple[NewsEvent, ...], GdeltDailyAggregationStats]:
    """Verify a ZIP container and stream its only TSV member into aggregation."""
    try:
        with zipfile.ZipFile(path) as container:
            entries = [entry for entry in container.infolist() if not entry.is_dir()]
            if len(entries) != 1:
                raise GdeltArchiveError("GDELT ZIP must contain exactly one data member")
            entry = entries[0]
            if Path(entry.filename).name != entry.filename or not entry.filename.endswith(
                ".export.CSV"
            ):
                raise GdeltArchiveError("GDELT ZIP member path or extension is invalid")
            if entry.file_size > MAXIMUM_UNCOMPRESSED_ARCHIVE_BYTES:
                raise GdeltArchiveError("GDELT ZIP exceeds the uncompressed-size limit")
            ratio = entry.file_size / max(entry.compress_size, 1)
            if ratio > MAXIMUM_ZIP_COMPRESSION_RATIO:
                raise GdeltArchiveError("GDELT ZIP compression ratio exceeds the guardrail")
            with (
                container.open(entry) as raw,
                io.TextIOWrapper(
                    raw,
                    encoding="utf-8",
                    errors="strict",
                    newline="",
                ) as text,
            ):
                return aggregate_gdelt_v1_daily_lines(
                    text,
                    archive=archive,
                    ticker_aliases=ticker_aliases,
                )
    except (OSError, UnicodeError, zipfile.BadZipFile) as error:
        raise GdeltArchiveError("GDELT archive container is corrupt") from error


def _validate_archive_sequence(archives: Sequence[GdeltV1DailyArchive]) -> None:
    if not archives:
        raise GdeltArchiveError("at least one GDELT archive is required")
    for previous, current in zip(archives, archives[1:], strict=False):
        if current.archive_date != previous.archive_date + timedelta(days=1):
            raise GdeltArchiveError("GDELT archive sequence contains a date gap")


def _load_verified_part(
    directory: Path,
    archive: GdeltV1DailyArchive,
) -> tuple[tuple[NewsEvent, ...], dict[str, object]] | None:
    if not directory.exists():
        return None
    try:
        events, manifest = load_news_snapshot(directory)
    except NewsSnapshotError:
        return None
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("archive_date") != str(
        archive.archive_date
    ):
        return None
    stats = provenance.get("aggregation_stats")
    required_stats = {
        "archive_date",
        "total_rows",
        "retained_rows",
        "invalid_rows",
        "output_events",
    }
    if not isinstance(stats, dict) or not required_stats.issubset(stats):
        return None
    if manifest.get("coverage_start") != archive.available_at.isoformat():
        return None
    return events, manifest


def _materialize_archive_part(
    archive: GdeltV1DailyArchive,
    *,
    part_dir: Path,
    download_path: Path,
    ticker_aliases: Mapping[str, tuple[str, ...]],
    downloader: Callable[[GdeltV1DailyArchive, Path], None],
) -> None:
    """Download, aggregate, and atomically save one independently keyed day."""
    try:
        downloader(archive, download_path)
        events, stats = aggregate_downloaded_archive(
            download_path,
            archive=archive,
            ticker_aliases=compile_ticker_aliases(ticker_aliases),
        )
        save_news_snapshot(
            part_dir,
            events,
            provider="GDELT 1.0 daily Event metadata",
            license_acknowledged=True,
            coverage_start=archive.available_at,
            coverage_end_exclusive=archive.available_at + timedelta(days=1),
            provenance={
                "archive_date": str(archive.archive_date),
                "archive_url": archive.url,
                "aggregation_stats": asdict(stats),
            },
        )
    finally:
        download_path.unlink(missing_ok=True)


def _materialize_archive_part_default(
    archive: GdeltV1DailyArchive,
    part_dir: Path,
    download_path: Path,
    ticker_aliases: dict[str, tuple[str, ...]],
    record_missing_404: bool,
) -> str | None:
    """Pickle-safe process worker using the production downloader."""
    try:
        _materialize_archive_part(
            archive,
            part_dir=part_dir,
            download_path=download_path,
            ticker_aliases=ticker_aliases,
            downloader=download_gdelt_archive,
        )
    except GdeltArchiveNotFound as error:
        if not record_missing_404:
            raise
        return str(error)
    return None


def _materialize_missing_archive_part(
    archive: GdeltV1DailyArchive,
    *,
    part_dir: Path,
    reason: str,
) -> None:
    """Persist an explicit provider gap; never silently convert it to no news."""
    save_news_snapshot(
        part_dir,
        (),
        provider="GDELT 1.0 daily Event metadata",
        license_acknowledged=True,
        coverage_start=archive.available_at,
        coverage_end_exclusive=archive.available_at + timedelta(days=1),
        provenance={
            "archive_date": str(archive.archive_date),
            "missing_archive": True,
            "missing_reason": reason,
            "aggregation_stats": {
                "archive_date": str(archive.archive_date),
                "total_rows": 0,
                "retained_rows": 0,
                "invalid_rows": 0,
                "output_events": 0,
            },
        },
    )


def build_gdelt_daily_snapshot(
    archives: Sequence[GdeltV1DailyArchive],
    *,
    output_dir: Path,
    work_dir: Path,
    ticker_aliases: Mapping[str, tuple[str, ...]],
    license_acknowledged: bool,
    downloader: Callable[[GdeltV1DailyArchive, Path], None] = download_gdelt_archive,
    workers: int = 1,
    missing_archive_dates: Sequence[date] = (),
    record_missing_404: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Resume verified daily parts and publish one bounded immutable snapshot."""
    if not license_acknowledged:
        raise NewsLicenseNotAcknowledged("GDELT data terms must be acknowledged explicitly")
    if workers < 1 or workers > 8:
        raise GdeltArchiveError("GDELT builder workers must be in [1, 8]")
    if workers > 1 and downloader is not download_gdelt_archive:
        raise GdeltArchiveError("parallel GDELT builds require the production downloader")
    _validate_archive_sequence(archives)
    allowed_gaps = set(missing_archive_dates)
    archive_dates = {archive.archive_date for archive in archives}
    if not allowed_gaps.issubset(archive_dates):
        raise GdeltArchiveError("missing archive dates must belong to the requested range")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise GdeltArchiveError("final GDELT snapshot directory must be empty")
    parts_root = work_dir / "parts"
    downloads_root = work_dir / "downloads"
    parts_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)
    missing: list[tuple[GdeltV1DailyArchive, Path, Path]] = []
    observed_missing_dates: set[date] = set(allowed_gaps)
    completed = 0
    for archive in archives:
        part_dir = parts_root / f"{archive.archive_date:%Y%m%d}"
        verified = _load_verified_part(part_dir, archive)
        if verified is not None:
            completed += 1
            continue
        if part_dir.exists():
            shutil.rmtree(part_dir)
        if archive.archive_date in allowed_gaps:
            _materialize_missing_archive_part(
                archive,
                part_dir=part_dir,
                reason="explicitly acknowledged provider archive gap",
            )
            completed += 1
            continue
        download_path = downloads_root / f"{archive.archive_date:%Y%m%d}.zip"
        missing.append((archive, part_dir, download_path))

    if progress is not None:
        progress(completed, len(archives), "verified")
    if workers == 1:
        for archive, part_dir, download_path in missing:
            try:
                _materialize_archive_part(
                    archive,
                    part_dir=part_dir,
                    download_path=download_path,
                    ticker_aliases=ticker_aliases,
                    downloader=downloader,
                )
            except GdeltArchiveNotFound as error:
                if not record_missing_404:
                    raise
                _materialize_missing_archive_part(
                    archive,
                    part_dir=part_dir,
                    reason=str(error),
                )
                observed_missing_dates.add(archive.archive_date)
            completed += 1
            if progress is not None:
                progress(completed, len(archives), str(archive.archive_date))
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _materialize_archive_part_default,
                    archive,
                    part_dir,
                    download_path,
                    dict(ticker_aliases),
                    record_missing_404,
                ): archive
                for archive, part_dir, download_path in missing
            }
            try:
                for future in as_completed(futures):
                    archive = futures[future]
                    missing_reason = future.result()
                    if missing_reason is not None:
                        _materialize_missing_archive_part(
                            archive,
                            part_dir=parts_root / f"{archive.archive_date:%Y%m%d}",
                            reason=missing_reason,
                        )
                        observed_missing_dates.add(archive.archive_date)
                    completed += 1
                    if progress is not None:
                        progress(completed, len(archives), str(archive.archive_date))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    # Re-read every part in archive order. This makes final event ordering and
    # provenance deterministic regardless of worker completion order and also
    # verifies that no worker claimed success without a valid snapshot.
    all_events: list[NewsEvent] = []
    source_stats: list[dict[str, object]] = []
    for archive in archives:
        verified = _load_verified_part(parts_root / f"{archive.archive_date:%Y%m%d}", archive)
        if verified is None:
            raise GdeltArchiveError(f"verified GDELT part is missing for {archive.archive_date}")
        events, manifest = verified
        all_events.extend(events)
        provenance = manifest["provenance"]
        source_stats.append(dict(provenance["aggregation_stats"]))

    totals = {
        name: sum(int(row[name]) for row in source_stats)
        for name in ("total_rows", "retained_rows", "invalid_rows", "output_events")
    }
    first = archives[0]
    last = archives[-1]
    return save_news_snapshot(
        output_dir,
        all_events,
        provider="GDELT 1.0 daily Event metadata",
        license_acknowledged=True,
        coverage_start=first.available_at,
        coverage_end_exclusive=last.available_at + timedelta(days=1),
        provenance={
            "source_archive_count": len(archives),
            "source_first_date": str(first.archive_date),
            "source_last_date": str(last.archive_date),
            "ticker_alias_count": len(ticker_aliases),
            "missing_archive_dates": [str(value) for value in sorted(observed_missing_dates)],
            **totals,
        },
    )
