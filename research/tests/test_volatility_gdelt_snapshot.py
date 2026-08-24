from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import pytest
from volatility_forecasting.gdelt import iter_gdelt_v1_daily_archives
from volatility_forecasting.gdelt_snapshot import (
    GdeltArchiveError,
    aggregate_downloaded_archive,
    build_gdelt_daily_snapshot,
    load_ticker_aliases,
)
from volatility_forecasting.news_snapshot import load_news_snapshot


def _line(archive_date: date) -> str:
    fields = [""] * 58
    fields[0] = f"{archive_date:%Y%m%d}01"
    fields[1] = f"{archive_date:%Y%m%d}"
    fields[6] = "MICROSOFT"
    fields[16] = "IRAN"
    fields[26] = "190"
    fields[27] = "190"
    fields[28] = "19"
    fields[29] = "4"
    fields[30] = "-9.0"
    fields[31] = "20"
    fields[32] = "5"
    fields[33] = "20"
    fields[34] = "-6.5"
    fields[56] = f"{archive_date:%Y%m%d}"
    fields[57] = "https://example.com/news/microsoft-iran"
    return "\t".join(fields)


def _fake_downloader(archive, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as container:
        container.writestr(
            f"{archive.archive_date:%Y%m%d}.export.CSV",
            _line(archive.archive_date) + "\n",
        )


def test_resumable_builder_reuses_verified_daily_parts(tmp_path: Path) -> None:
    archives = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 7))
    aliases = {"MSFT": ("Microsoft",)}
    work = tmp_path / "work"
    first_manifest = build_gdelt_daily_snapshot(
        archives,
        output_dir=tmp_path / "snapshot-one",
        work_dir=work,
        ticker_aliases=aliases,
        license_acknowledged=True,
        downloader=_fake_downloader,
    )
    assert first_manifest["event_count"] == 2
    assert first_manifest["provenance"]["source_archive_count"] == 2

    def fail_if_called(_archive, _destination: Path) -> None:
        raise AssertionError("verified daily parts should have been reused")

    build_gdelt_daily_snapshot(
        archives,
        output_dir=tmp_path / "snapshot-two",
        work_dir=work,
        ticker_aliases=aliases,
        license_acknowledged=True,
        downloader=fail_if_called,
    )
    events, manifest = load_news_snapshot(tmp_path / "snapshot-two")
    assert len(events) == 2
    assert manifest["snapshot_id"] == first_manifest["snapshot_id"]


def test_zip_member_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 6))[0]
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as container:
        container.writestr("../bad.export.CSV", _line(archive.archive_date))
    with pytest.raises(GdeltArchiveError, match="path"):
        aggregate_downloaded_archive(
            path,
            archive=archive,
            ticker_aliases={"MSFT": ("Microsoft",)},
        )


def test_alias_loader_rejects_short_and_normalizes_symbols(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"msft": ["Microsoft", "Microsoft Corp"]}), encoding="utf-8")
    assert load_ticker_aliases(path) == {"MSFT": ("Microsoft", "Microsoft Corp")}
    path.write_text(json.dumps({"C": ["C"]}), encoding="utf-8")
    with pytest.raises(GdeltArchiveError, match="three characters"):
        load_ticker_aliases(path)


def test_frozen_aliases_cover_the_complete_volatility_universe() -> None:
    root = Path(__file__).resolve().parents[2]
    aliases = load_ticker_aliases(root / "configs" / "news-ticker-aliases-v1.json")
    universe = {
        line.strip()
        for line in (root / "configs" / "volatility-universe-v1.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(aliases) == universe
