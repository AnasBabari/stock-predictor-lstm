#!/usr/bin/env python3
"""Materialize a development-only PIT64 snapshot and universe.

This command exists to exercise the frozen V11.2 CUDA development pipeline
while licensed, independently attested certification inputs are unavailable.
It deliberately stamps ``certification_eligible=false``.  The V11.2 preflight
and one-shot certifier must reject its output.

Market history comes from the repository's existing secondary NDX100 cache.
Stable identifiers are refreshed from SEC and OpenFIGI, while descriptive
industry metadata comes from Nasdaq.  None of those identity lookups upgrades
the cached OHLCV or secondary membership history into certification evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.panel.snapshots import build_snapshot, load_snapshot, write_snapshot  # noqa: E402
from research.ndx100.universe import (  # noqa: E402
    BASE_CONSTITUENTS_2021_12_31,
    get_membership_changes,
)
from research.volatility_forecasting.v11_2_protocol import V11_2_PROTOCOL_ID  # noqa: E402
from research.volatility_forecasting.v11_2_universe import (  # noqa: E402
    MembershipInterval,
    PITSecurity,
    TickerInterval,
    build_universe_manifest,
    save_universe_manifest,
)

PANEL_START = dt.date(2022, 1, 3)
DEVELOPMENT_SELECTION_METHOD = "development_secondary_source_reviewed_pit64"
USER_AGENT = "StockLSTM-development-research/11.2"

SECTOR_STRATA: dict[str, tuple[str, ...]] = {
    "semiconductors_hardware": (
        "NVDA", "AMD", "AVGO", "AMAT", "ADI", "LRCX", "KLAC", "QCOM",
    ),
    "software_cloud": (
        "MSFT", "ADBE", "ADSK", "CDNS", "SNPS", "INTU", "PANW", "FTNT",
    ),
    "communications_media": (
        "GOOG", "META", "NFLX", "CMCSA", "WBD", "BIDU", "NTES", "MELI",
    ),
    "consumer_discretionary": (
        "AMZN", "TSLA", "BKNG", "SBUX", "ROST", "MAR", "ORLY", "DASH",
    ),
    "consumer_staples_defensive": (
        "COST", "PEP", "MDLZ", "KDP", "KHC", "MNST", "DLTR", "CCEP",
    ),
    "healthcare_biotech": (
        "AMGN", "GILD", "REGN", "VRTX", "ISRG", "IDXX", "BIIB", "MRNA",
    ),
    "industrials_transportation": (
        "HON", "CSX", "ODFL", "CPRT", "CTAS", "PCAR", "FAST", "ROP",
    ),
    "energy_utilities_telecom": (
        "AEP", "EXC", "XEL", "CEG", "TMUS", "BKR", "FANG", "VRSK",
    ),
}


def _http_json(
    url: str,
    *,
    payload: object | None = None,
    user_agent: str = USER_AGENT,
    attempts: int = 4,
) -> Any:
    body = None
    headers = {"Accept": "application/json", "User-Agent": user_agent}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable HTTP retry state")


def _all_tickers() -> tuple[str, ...]:
    values = tuple(ticker for group in SECTOR_STRATA.values() for ticker in group)
    if len(values) != 64 or len(set(values)) != 64:
        raise RuntimeError("diagnostic PIT64 must contain exactly 64 unique tickers")
    if len(SECTOR_STRATA) != 8 or any(len(group) != 8 for group in SECTOR_STRATA.values()):
        raise RuntimeError("diagnostic PIT64 requires eight strata of eight securities")
    return values


def _load_frames(cache_dir: Path) -> tuple[dict[str, pd.DataFrame], dt.date]:
    frames: dict[str, pd.DataFrame] = {}
    common_end: dt.date | None = None
    for ticker in _all_tickers():
        path = cache_dir / f"{ticker}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached history for {ticker}: {path}")
        frame = pd.read_parquet(path)
        frame = frame.rename(columns={str(column).lower(): str(column).title() for column in frame.columns})
        required = ["Open", "High", "Low", "Close", "Volume"]
        if any(column not in frame for column in required):
            raise ValueError(f"{ticker}: cached frame lacks canonical OHLCV columns")
        frame = frame[required].copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        frame = frame.loc[frame.index.date >= PANEL_START]
        if frame.empty:
            raise ValueError(f"{ticker}: no cached rows on or after {PANEL_START}")
        ticker_end = pd.Timestamp(frame.index.max()).date()
        common_end = ticker_end if common_end is None else min(common_end, ticker_end)
        frames[ticker] = frame
    if common_end is None:
        raise RuntimeError("diagnostic cache produced no frames")
    for ticker in frames:
        frames[ticker] = frames[ticker].loc[frames[ticker].index.date <= common_end]
    # The cache stores Meta's full adjusted history under its current alias.
    # Duplicate the immutable bytes under FB solely for point-in-time alias
    # resolution before the 2022-06-09 ticker transition.
    frames["FB"] = frames["META"].copy()
    return frames, common_end


def _sec_ciks(tickers: Iterable[str]) -> dict[str, str]:
    agent = str(__import__("os").environ.get("STOCKLSTM_SEC_USER_AGENT", USER_AGENT))
    payload = _http_json("https://www.sec.gov/files/company_tickers.json", user_agent=agent)
    by_ticker = {
        str(item["ticker"]).upper(): f"{int(item['cik_str']):010d}"
        for item in payload.values()
        if isinstance(item, dict) and item.get("ticker") and item.get("cik_str") is not None
    }
    missing = sorted(set(tickers) - set(by_ticker))
    if missing:
        raise ValueError(f"SEC company ticker mapping missing: {', '.join(missing)}")
    return {ticker: by_ticker[ticker] for ticker in tickers}


def _openfigi(tickers: tuple[str, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for start in range(0, len(tickers), 10):
        batch = tickers[start : start + 10]
        jobs = [{"idType": "TICKER", "idValue": ticker, "exchCode": "US"} for ticker in batch]
        response = _http_json("https://api.openfigi.com/v3/mapping", payload=jobs)
        if not isinstance(response, list) or len(response) != len(batch):
            raise ValueError("OpenFIGI returned an unexpected batch response")
        for ticker, result in zip(batch, response, strict=True):
            candidates = result.get("data", []) if isinstance(result, dict) else []
            exact = [
                item
                for item in candidates
                if str(item.get("ticker", "")).upper() == ticker
                and str(item.get("marketSector", "")) == "Equity"
            ]
            if not exact:
                raise ValueError(f"OpenFIGI mapping missing for {ticker}")
            preferred = next(
                (item for item in exact if item.get("securityType2") == "Common Stock"), exact[0]
            )
            figi = str(preferred.get("compositeFIGI") or preferred.get("figi") or "")
            if not figi:
                raise ValueError(f"OpenFIGI response has no FIGI for {ticker}")
            output[ticker] = figi
        time.sleep(0.35)
    return output


def _nasdaq_profile(ticker: str) -> tuple[str, str]:
    url = f"https://api.nasdaq.com/api/company/{ticker}/company-profile"
    payload = _http_json(url, user_agent="Mozilla/5.0 StockLSTM-development-research")
    data = payload.get("data") if isinstance(payload, dict) else None
    industry = str((data.get("Industry") or {}).get("value") or "").strip() if isinstance(data, dict) else ""
    sector = str((data.get("Sector") or {}).get("value") or "").strip() if isinstance(data, dict) else ""
    if not industry or not sector:
        summary_url = f"https://api.nasdaq.com/api/quote/{ticker}/summary?assetclass=stocks"
        summary = _http_json(
            summary_url, user_agent="Mozilla/5.0 StockLSTM-development-research"
        )
        summary_data = ((summary.get("data") or {}).get("summaryData") or {})
        industry = str((summary_data.get("Industry") or {}).get("value") or "").strip()
        sector = str((summary_data.get("Sector") or {}).get("value") or "").strip()
    if not industry or not sector:
        raise ValueError(f"Nasdaq sector/industry metadata missing for {ticker}")
    return sector, industry


def _nasdaq_profiles(tickers: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(_nasdaq_profile, tickers))
    return {
        ticker: {"official_sector": sector, "industry": industry}
        for ticker, (sector, industry) in zip(tickers, values, strict=True)
    }


def _nasdaq_market_caps(tickers: tuple[str, ...]) -> dict[str, float]:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&offset=0&exchange=nasdaq"
    payload = _http_json(url, user_agent="Mozilla/5.0 StockLSTM-development-research")
    rows = (((payload or {}).get("data") or {}).get("table") or {}).get("rows") or []
    caps: dict[str, float] = {}
    for row in rows:
        ticker = str(row.get("symbol", "")).upper()
        raw = str(row.get("marketCap", "")).replace(",", "")
        if ticker in tickers:
            try:
                caps[ticker] = float(raw)
            except ValueError:
                continue
    missing = sorted(set(tickers) - set(caps))
    if missing:
        raise ValueError(f"Nasdaq screener market cap missing: {', '.join(missing)}")
    return caps


def _metadata(tickers: tuple[str, ...], cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("tickers") == list(tickers):
            return payload
    payload = {
        "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
        "tickers": list(tickers),
        "ciks": _sec_ciks(tickers),
        "figis": _openfigi(tickers),
        "profiles": _nasdaq_profiles(tickers),
        "market_caps": _nasdaq_market_caps(tickers),
        "sources": {
            "cik": "https://www.sec.gov/files/company_tickers.json",
            "figi": "https://api.openfigi.com/v3/mapping",
            "classification": "https://api.nasdaq.com/api/company/{ticker}/company-profile",
            "market_cap": "https://api.nasdaq.com/api/screener/stocks",
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _strata(values: dict[str, float], *, labels: tuple[str, str, str]) -> dict[str, str]:
    ordered = sorted(values, key=lambda ticker: (values[ticker], ticker))
    output: dict[str, str] = {}
    for index, ticker in enumerate(ordered):
        output[ticker] = labels[min(2, (index * 3) // len(ordered))]
    return output


def _membership_intervals(
    ticker: str,
    *,
    end_date: dt.date,
    source_digest: str,
) -> tuple[MembershipInterval, ...]:
    local_identity = f"ndx100-pit:{ticker}"
    changes = get_membership_changes()
    initial_alias = "FB" if ticker == "META" else ticker
    active = initial_alias in BASE_CONSTITUENTS_2021_12_31
    start = PANEL_START if active else None
    intervals: list[MembershipInterval] = []
    for row in changes.itertuples(index=False):
        effective = dt.date.fromisoformat(str(row.effective_date))
        if effective < PANEL_START or effective > end_date:
            continue
        if str(row.security_id) != local_identity:
            continue
        if row.action == "ADD" and not active:
            start = effective
            active = True
        elif row.action == "REMOVE" and active:
            intervals.append(
                MembershipInterval(
                    start_date=str(start),
                    end_date=(effective - dt.timedelta(days=1)).isoformat(),
                    source="development-secondary-ndx100-membership-history",
                    source_digest=source_digest,
                )
            )
            start = None
            active = False
    if active and start is not None:
        intervals.append(
            MembershipInterval(
                start_date=start.isoformat(),
                end_date=end_date.isoformat(),
                source="development-secondary-ndx100-membership-history",
                source_digest=source_digest,
            )
        )
    if not intervals:
        raise ValueError(f"{ticker}: no point-in-time development membership interval")
    return tuple(intervals)


def _ticker_intervals(
    ticker: str, membership: tuple[MembershipInterval, ...]
) -> tuple[TickerInterval, ...]:
    start = min(dt.date.fromisoformat(item.start_date) for item in membership)
    end = max(dt.date.fromisoformat(item.end_date) for item in membership)
    if ticker != "META":
        return (TickerInterval(ticker, start.isoformat(), end.isoformat()),)
    transition = dt.date(2022, 6, 9)
    return (
        TickerInterval("FB", start.isoformat(), (transition - dt.timedelta(days=1)).isoformat()),
        TickerInterval("META", transition.isoformat(), end.isoformat()),
    )


def _securities(
    frames: dict[str, pd.DataFrame],
    end_date: dt.date,
    metadata: dict[str, Any],
    membership_digest: str,
) -> list[PITSecurity]:
    tickers = _all_tickers()
    annualized_volatility = {
        ticker: float(np.std(np.diff(np.log(frames[ticker]["Close"])), ddof=1) * np.sqrt(252.0))
        for ticker in tickers
    }
    volatility_labels = _strata(
        annualized_volatility, labels=("lower_realized_vol", "middle_realized_vol", "higher_realized_vol")
    )
    cap_labels = _strata(
        {ticker: float(metadata["market_caps"][ticker]) for ticker in tickers},
        labels=("selected_lower_cap", "selected_middle_cap", "selected_upper_cap"),
    )
    sector_by_ticker = {
        ticker: sector for sector, group in SECTOR_STRATA.items() for ticker in group
    }
    output: list[PITSecurity] = []
    for ticker in tickers:
        membership = _membership_intervals(
            ticker, end_date=end_date, source_digest=membership_digest
        )
        frame = frames[ticker]
        output.append(
            PITSecurity(
                security_id=f"US.{metadata['figis'][ticker]}",
                cik=str(metadata["ciks"][ticker]),
                figi=str(metadata["figis"][ticker]),
                exchange_mic="XNAS",
                sector=sector_by_ticker[ticker],
                industry=str(metadata["profiles"][ticker]["industry"]),
                volatility_stratum=volatility_labels[ticker],
                market_cap_stratum=cap_labels[ticker],
                ticker_intervals=_ticker_intervals(ticker, membership),
                membership_intervals=membership,
                provider_aliases=tuple(item.ticker for item in _ticker_intervals(ticker, membership)),
                corporate_actions=(
                    ({"date": "2022-06-09", "type": "ticker_change", "from": "FB", "to": "META"},)
                    if ticker == "META"
                    else ()
                ),
                ohlcv_coverage={
                    "rows": int(len(frame)),
                    "start": pd.Timestamp(frame.index.min()).date().isoformat(),
                    "end": pd.Timestamp(frame.index.max()).date().isoformat(),
                },
                provenance={
                    "certification_eligible": False,
                    "market_data": "repository secondary NDX100 yfinance cache",
                    "membership": "development secondary history; not an official archive",
                    "official_sector": metadata["profiles"][ticker]["official_sector"],
                    "identity_sources": metadata["sources"],
                },
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "ndx100" / "cache")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts" / "v11_2_diagnostic_inputs"
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    frames, end_date = _load_frames(args.cache_dir.resolve())
    metadata = _metadata(_all_tickers(), output_root / "identity_metadata.json")
    membership_path = ROOT / "data" / "ndx100" / "membership_changes.csv"
    membership_digest = hashlib.sha256(membership_path.read_bytes()).hexdigest()
    securities = _securities(frames, end_date, metadata, membership_digest)
    universe = build_universe_manifest(
        securities,
        protocol_id=V11_2_PROTOCOL_ID,
        universe_version=f"v11.2-diagnostic-pit64-{end_date.isoformat()}-r1",
        membership_sources=(
            f"development-secondary:{membership_path.as_posix()}@sha256:{membership_digest}",
            "official-identity:https://www.sec.gov/files/company_tickers.json",
            "official-identity:https://api.openfigi.com/v3/mapping",
            "official-classification:https://api.nasdaq.com/api/company/{ticker}/company-profile",
        ),
        selection_method=DEVELOPMENT_SELECTION_METHOD,
        certification_eligible=False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    universe_path = output_root / "v11_2_diagnostic_universe.json"
    save_universe_manifest(universe, universe_path)

    snapshot_root = output_root / "snapshots"
    expected = build_snapshot(frames, license_acknowledged=True)
    snapshot_dir = snapshot_root / str(expected["panel_id"])
    if snapshot_dir.is_dir():
        load_snapshot(snapshot_dir)
    else:
        snapshot_dir = write_snapshot(snapshot_root, frames, license_acknowledged=True)

    print(
        json.dumps(
            {
                "certification_eligible": False,
                "diagnostic_only": True,
                "universe_manifest": str(universe_path),
                "universe_sha256": universe.manifest_sha256,
                "snapshot_dir": str(snapshot_dir),
                "snapshot_panel_id": expected["panel_id"],
                "security_count": universe.universe_size,
                "date_span": [PANEL_START.isoformat(), end_date.isoformat()],
                "next": "build panel, prepare encrypted diagnostic split, then run CUDA development",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
