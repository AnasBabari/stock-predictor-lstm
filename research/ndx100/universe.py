"""Point-in-time Nasdaq-100 universe reconstruction and calendar alignment.

Reconstructs the exact active constituent membership of the Nasdaq-100 index
for any date T >= 2022-01-01 by starting from the verified post-reconstitution
components of 2021-12-31 and applying historical additions and removals in
chronological order.

Guarantees survivorship-bias resistance: historical removals (e.g., PTON,
XLNX, OKTA, SPLK, ATVI, SIRI, WBA, DLTR) are included during their respective
active periods and excluded thereafter.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import exchange_calendars as ecals
import numpy as np
import pandas as pd

DEFAULT_CHANGES_CSV = (
    Path(__file__).resolve().parent.parent.parent / "data" / "ndx100" / "membership_changes.csv"
)
DEFAULT_MEMBERSHIP_MANIFEST = DEFAULT_CHANGES_CSV.with_name("membership_manifest.json")
MEMBERSHIP_ACTIONS = frozenset({"ADD", "REMOVE", "TICKER_CHANGE"})

# Verified post-reconstitution components as of 2021-12-31 (102 tickers / 100 companies)
BASE_CONSTITUENTS_2021_12_31: tuple[str, ...] = (
    "AAPL",
    "ABNB",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AEP",
    "ALGN",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "ANSS",
    "ASML",
    "ATVI",
    "AVGO",
    "BIDU",
    "BIIB",
    "BKNG",
    "CDNS",
    "CHTR",
    "CMCSA",
    "COST",
    "CPRT",
    "CRWD",
    "CSCO",
    "CSX",
    "CTAS",
    "CTSH",
    "DDOG",
    "DLTR",
    "DOCU",
    "DXCM",
    "EA",
    "EBAY",
    "EXC",
    "FAST",
    "FISV",
    "FOX",
    "FTNT",
    "FB",
    "GILD",
    "GOOG",
    "GOOGL",
    "HON",
    "IDXX",
    "ILMN",
    "INTC",
    "INTU",
    "ISRG",
    "JD",
    "KDP",
    "KHC",
    "KLAC",
    "LCID",
    "LRCX",
    "LULU",
    "MAR",
    "MCHP",
    "MDLZ",
    "MELI",
    "MNST",
    "MRNA",
    "MRVL",
    "MSFT",
    "MTCH",
    "MU",
    "NFLX",
    "NTES",
    "NVDA",
    "NXPI",
    "OKTA",
    "ORLY",
    "PANW",
    "PAYX",
    "PCAR",
    "PDD",
    "PEP",
    "PTON",
    "PYPL",
    "QCOM",
    "REGN",
    "ROST",
    "SBUX",
    "SGEN",
    "SIRI",
    "SNPS",
    "SPLK",
    "SWKS",
    "TEAM",
    "TMUS",
    "TSLA",
    "TXN",
    "VRSK",
    "VRSN",
    "VRTX",
    "WBA",
    "WDAY",
    "XEL",
    "XLNX",
    "ZM",
    "ZS",
)

HISTORICAL_REMOVALS: tuple[str, ...] = (
    "PTON",
    "XLNX",
    "OKTA",
    "SPLK",
    "ATVI",
    "SIRI",
    "WBA",
    "DLTR",
    "ILMN",
    "MRNA",
    "SMCI",
    "MDB",
    "ANSS",
    "BIIB",
    "CDW",
    "GFS",
    "LULU",
    "ON",
    "TTD",
    "AZN",
    "TEAM",
    "CSGP",
    "CHTR",
    "CTSH",
    "INSM",
)


def get_membership_changes(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the point-in-time membership changes log."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CHANGES_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Nasdaq-100 membership changes CSV not found: {path}")
    if csv_path is None:
        verify_membership_source(path)
    df = pd.read_csv(path)
    required_cols = {
        "effective_date",
        "action",
        "ticker",
        "security_id",
        "company_name",
        "source",
        "source_url",
        "source_snapshot_id",
        "verified",
        "notes",
        "related_ticker",
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"membership changes CSV missing required columns: {missing}")
    parsed_dates = pd.to_datetime(df["effective_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("membership changes contain an invalid effective date")
    df["effective_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["action"] = df["action"].str.strip().str.upper()
    df["related_ticker"] = df["related_ticker"].fillna("").astype(str).str.strip().str.upper()
    if (~df["action"].isin(MEMBERSHIP_ACTIONS)).any():
        raise ValueError("membership changes contain an unsupported action")
    if df["ticker"].eq("").any() or df["security_id"].fillna("").str.strip().eq("").any():
        raise ValueError("membership changes require ticker and stable security_id")
    if df[["source", "source_url", "source_snapshot_id"]].isna().any(axis=None):
        raise ValueError("membership changes require complete source provenance")
    verified = df["verified"].astype(str).str.lower().map({"true": True, "false": False})
    if verified.isna().any() or not verified.all():
        raise ValueError("membership changes must be explicitly verified before use")
    df["verified"] = verified
    ticker_changes = df["action"].eq("TICKER_CHANGE")
    if df.loc[ticker_changes, "related_ticker"].eq("").any():
        raise ValueError("ticker changes require related_ticker")
    if df.loc[~ticker_changes, "related_ticker"].ne("").any():
        raise ValueError("related_ticker is only valid for ticker changes")
    if df.duplicated(["effective_date", "action", "ticker"]).any():
        raise ValueError("membership changes contain duplicate transitions")
    return df.sort_values(["effective_date", "ticker"]).reset_index(drop=True)


def verify_membership_source(
    csv_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, object]:
    """Verify the immutable local source table before reconstructing membership."""
    source = Path(csv_path) if csv_path is not None else DEFAULT_CHANGES_CSV
    manifest_file = (
        Path(manifest_path) if manifest_path is not None else DEFAULT_MEMBERSHIP_MANIFEST
    )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("membership source manifest is missing or invalid") from error
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest.get("sha256") != digest:
        raise ValueError("membership source checksum differs from its immutable manifest")
    if manifest.get("evidence_status") != "development_secondary_source_reviewed":
        raise ValueError("membership source is not approved for development research")
    return manifest


def get_ndx100_constituents(
    as_of_date: str | datetime.date | pd.Timestamp,
    csv_path: Path | str | None = None,
    base_constituents: Sequence[str] = BASE_CONSTITUENTS_2021_12_31,
) -> list[str]:
    """Return the sorted list of active Nasdaq-100 tickers on a given date."""
    target_dt_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    if target_dt_str < "2022-01-01":
        raise ValueError(f"as_of_date must be on or after 2022-01-01, got {target_dt_str}")

    changes_df = get_membership_changes(csv_path)
    post_base = changes_df[
        (changes_df["effective_date"] > "2021-12-31")
        & (changes_df["effective_date"] <= target_dt_str)
    ].sort_values("effective_date")

    active: set[str] = set(base_constituents)
    for _, row in post_base.iterrows():
        action = row["action"]
        ticker = row["ticker"]
        if action == "ADD":
            active.add(ticker)
        elif action == "REMOVE":
            active.discard(ticker)
        elif action == "TICKER_CHANGE":
            active.discard(ticker)
            active.add(str(row["related_ticker"]))

    return sorted(active)


def get_ndx100_membership_timeline(
    csv_path: Path | str | None = None,
    base_constituents: Sequence[str] = BASE_CONSTITUENTS_2021_12_31,
) -> pd.DataFrame:
    """Generate an auditable turnover log for every transition from 2022 to 2026."""
    changes_df = get_membership_changes(csv_path)
    post_base = changes_df[changes_df["effective_date"] > "2021-12-31"].sort_values(
        "effective_date"
    )

    records: list[dict[str, object]] = []
    current: set[str] = set(base_constituents)

    for dt, group in post_base.groupby("effective_date"):
        adds = group[group["action"] == "ADD"]["ticker"].tolist()
        rems = group[group["action"] == "REMOVE"]["ticker"].tolist()
        ticker_changes = [
            {"from": row["ticker"], "to": row["related_ticker"]}
            for _, row in group[group["action"] == "TICKER_CHANGE"].iterrows()
        ]
        for a in adds:
            current.add(a)
        for r in rems:
            current.discard(r)
        for change in ticker_changes:
            current.discard(str(change["from"]))
            current.add(str(change["to"]))
        records.append(
            {
                "effective_date": dt,
                "additions": adds,
                "removals": rems,
                "ticker_changes": ticker_changes,
                "active_count": len(current),
                "active_constituents": sorted(current),
            }
        )
    return pd.DataFrame(records)


def get_ndx100_constituent_union(
    csv_path: Path | str | None = None,
    base_constituents: Sequence[str] = BASE_CONSTITUENTS_2021_12_31,
) -> tuple[str, ...]:
    """Return every symbol needed to reconstruct the development history."""
    changes = get_membership_changes(csv_path)
    symbols = set(str(value).upper() for value in base_constituents)
    symbols.update(changes["ticker"])
    symbols.update(value for value in changes["related_ticker"] if value)
    return tuple(sorted(symbols))


def point_in_time_membership_mask(
    tickers: Sequence[str] | np.ndarray,
    origin_dates: Sequence[object] | np.ndarray,
    csv_path: Path | str | None = None,
    base_constituents: Sequence[str] = BASE_CONSTITUENTS_2021_12_31,
) -> np.ndarray:
    """Return a row mask that excludes observations outside index membership."""
    ticker_values = np.asarray(tickers, dtype=str)
    dates = np.asarray(origin_dates, dtype="datetime64[D]")
    if ticker_values.ndim != 1 or dates.ndim != 1 or len(ticker_values) != len(dates):
        raise ValueError("membership mask requires matched one-dimensional row identities")
    if np.isnat(dates).any():
        raise ValueError("membership mask origin dates must be finite")
    changes = get_membership_changes(csv_path)
    change_dates = changes["effective_date"].to_numpy(dtype="datetime64[D]")
    active = set(str(value).upper() for value in base_constituents)
    cursor = 0
    active_by_date: dict[np.datetime64, set[str]] = {}
    for date in np.sort(np.unique(dates)):
        while cursor < len(changes) and change_dates[cursor] <= date:
            row = changes.iloc[cursor]
            ticker = str(row["ticker"])
            if row["action"] == "ADD":
                active.add(ticker)
            elif row["action"] == "REMOVE":
                active.discard(ticker)
            else:
                active.discard(ticker)
                active.add(str(row["related_ticker"]))
            cursor += 1
        active_by_date[date] = set(active)
    return np.asarray(
        [
            ticker.upper() in active_by_date[date]
            for ticker, date in zip(ticker_values, dates, strict=True)
        ],
        dtype=bool,
    )


def get_weekly_origins(
    start_date: str = "2022-01-01",
    end_date: str = "2026-08-28",
    calendar_name: str = "XNYS",
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    """Generate weekly forecast origins and their corresponding 5 forward sessions.

    Each origin is the final trading session of the week with exactly 5 forward
    trading sessions available within the range.
    """
    cal = ecals.get_calendar(calendar_name)
    sessions = [pd.Timestamp(s).normalize() for s in cal.sessions_in_range(start_date, end_date)]
    if not sessions:
        raise ValueError(
            f"No trading sessions found for {calendar_name} between {start_date} and {end_date}"
        )

    df_sessions = pd.DataFrame({"session": sessions})
    df_sessions["year_week"] = (
        df_sessions["session"].dt.isocalendar().year.astype(str)
        + "-"
        + df_sessions["session"].dt.isocalendar().week.astype(str).str.zfill(2)
    )

    origins: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    for _, group in df_sessions.groupby("year_week"):
        last_session = group["session"].iloc[-1]
        idx = sessions.index(last_session)
        if idx + 5 < len(sessions):
            target_sessions = sessions[idx + 1 : idx + 6]
            origins.append((last_session, target_sessions))

    return origins


def assert_survivorship_bias_resistant(
    constituents_by_origin: dict[pd.Timestamp, list[str]],
) -> None:
    """Assert that historical removals appear in early origins and are absent later."""
    if not constituents_by_origin:
        raise ValueError("constituents_by_origin mapping cannot be empty")

    all_observed_tickers: set[str] = set()
    for tickers in constituents_by_origin.values():
        all_observed_tickers.update(tickers)

    # Check that key historical removals appear
    found_removals = set(HISTORICAL_REMOVALS).intersection(all_observed_tickers)
    if not found_removals:
        raise AssertionError(
            f"No historical removals observed! Expected at least some of {HISTORICAL_REMOVALS}"
        )

    # Verify specific removals are present at start and absent after removal
    early_origin = min(constituents_by_origin.keys())
    if pd.Timestamp(early_origin) <= pd.Timestamp("2022-01-20"):
        early_tickers = set(constituents_by_origin[early_origin])
        assert "PTON" in early_tickers, "PTON must be present before 2022-01-24"
        assert "XLNX" in early_tickers, "XLNX must be present before 2022-02-22"
        assert "OKTA" in early_tickers, "OKTA must be present before 2022-11-21"

    late_origin = max(constituents_by_origin.keys())
    if pd.Timestamp(late_origin) >= pd.Timestamp("2026-07-01"):
        late_tickers = set(constituents_by_origin[late_origin])
        assert "PTON" not in late_tickers, "PTON must not be present in 2026"
        assert "XLNX" not in late_tickers, "XLNX must not be present in 2026"
        assert "OKTA" not in late_tickers, "OKTA must not be present in 2026"
        assert "SPLK" not in late_tickers, "SPLK must not be present in 2026"
        assert "SIRI" not in late_tickers, "SIRI must not be present in 2026"
        assert "WBA" not in late_tickers, "WBA must not be present in 2026"
        assert "DLTR" not in late_tickers, "DLTR must not be present in 2026"
