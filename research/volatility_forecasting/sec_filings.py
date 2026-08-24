"""Point-in-time SEC filing metadata as authoritative corporate event features."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .news import NewsEvent, NewsValidationError

_SUPPORTED_FORMS = frozenset({"8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "6-K"})
_HIGH_SEVERITY_8K_ITEMS = frozenset({"1.02", "1.03", "2.05", "2.06", "3.01", "4.01"})
_EARNINGS_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F"})


@dataclass(frozen=True)
class SecFilingRecord:
    ticker: str
    cik: str
    accession_number: str
    form: str
    accepted_at: pd.Timestamp
    filing_date: str
    items: tuple[str, ...]
    primary_document: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", self.ticker.upper().strip())
        object.__setattr__(self, "form", self.form.upper().strip())
        object.__setattr__(
            self, "items", tuple(sorted({item.strip() for item in self.items if item.strip()}))
        )
        accepted = pd.Timestamp(self.accepted_at)
        if accepted.tzinfo is None:
            accepted = accepted.tz_localize("America/New_York")
        object.__setattr__(self, "accepted_at", accepted.tz_convert("UTC"))
        if not self.ticker or not self.cik or not self.accession_number:
            raise NewsValidationError("SEC ticker, CIK, and accession number are required")
        if self.form not in _SUPPORTED_FORMS:
            raise NewsValidationError(f"unsupported SEC filing form: {self.form}")


def parse_sec_submissions_recent(
    payload: Mapping[str, object],
    *,
    ticker: str,
) -> tuple[SecFilingRecord, ...]:
    """Parse aligned arrays from data.sec.gov's company submissions response."""
    raw_cik = str(payload.get("cik", "")).strip()
    if not raw_cik.isdigit() or len(raw_cik) > 10:
        raise NewsValidationError("SEC submissions payload has an invalid CIK")
    cik = raw_cik.zfill(10)
    filings = payload.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if not isinstance(recent, Mapping):
        raise NewsValidationError("SEC submissions payload is missing filings.recent")
    required = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "items",
        "primaryDocument",
    )
    arrays: dict[str, Sequence[object]] = {}
    for name in required:
        values = recent.get(name)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise NewsValidationError(f"SEC submissions array is missing: {name}")
        arrays[name] = values
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise NewsValidationError("SEC submissions arrays are not aligned")

    records: list[SecFilingRecord] = []
    for index in range(next(iter(lengths), 0)):
        form = str(arrays["form"][index]).upper().strip()
        if form not in _SUPPORTED_FORMS:
            continue
        item_text = str(arrays["items"][index])
        items = tuple(part.strip() for part in item_text.split(",") if part.strip())
        records.append(
            SecFilingRecord(
                ticker=ticker,
                cik=cik,
                accession_number=str(arrays["accessionNumber"][index]),
                form=form,
                accepted_at=pd.Timestamp(str(arrays["acceptanceDateTime"][index])),
                filing_date=str(arrays["filingDate"][index]),
                items=items,
                primary_document=str(arrays["primaryDocument"][index]),
            )
        )
    return tuple(records)


def sec_filing_to_news_event(
    record: SecFilingRecord,
    *,
    conservative_availability_delay: timedelta = timedelta(minutes=15),
) -> NewsEvent:
    """Convert filing metadata without inferring sentiment from unavailable text."""
    if conservative_availability_delay < timedelta(0):
        raise ValueError("SEC availability delay cannot be negative")
    topics: set[str] = {"regulatory_filing"}
    if record.form in _EARNINGS_FORMS or "2.02" in record.items:
        topics.add("earnings")
    if "1.01" in record.items:
        topics.add("material_agreement")
    if "2.01" in record.items:
        topics.add("asset_transaction")
    if record.form == "6-K":
        topics.add("foreign_issuer_report")
    if "1.03" in record.items:
        topics.add("bankruptcy")
    if "5.02" in record.items:
        topics.add("management_change")
    severity = 0.8 if _HIGH_SEVERITY_8K_ITEMS.intersection(record.items) else 0.4
    if record.form in _EARNINGS_FORMS or "2.02" in record.items:
        severity = max(severity, 0.5)
    accession = record.accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(record.cik)}/{accession}/{record.primary_document}"
    return NewsEvent(
        event_id=f"sec:{record.accession_number}",
        cluster_id=f"sec:{record.accession_number}",
        source="sec.gov",
        first_seen_at=record.accepted_at + conservative_availability_delay,
        published_at=record.accepted_at,
        timestamp_quality="first_seen_only",
        tickers=(record.ticker,),
        topics=tuple(sorted(topics)),
        positive_probability=0.0,
        neutral_probability=1.0,
        negative_probability=0.0,
        novelty=0.5,
        severity=severity,
        confidence=1.0,
        source_reliability=1.0,
        canonical_url_hash=hashlib.sha256(url.encode("utf-8")).hexdigest(),
        license_class="sec_public_filing_metadata",
    )
