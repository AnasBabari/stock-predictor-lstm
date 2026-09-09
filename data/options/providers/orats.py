"""ORATS near-EOD strikes reader (sample + full archive share one format)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..schemas import NORMALIZED_COLUMNS, ORATS_STRIKES_COLUMNS, SOURCE_ORATS_NEAR_EOD
from .base import OptionsProvider


class OratsStrikesProvider(OptionsProvider):
    """Reads ORATS_SMV_Strikes_YYYYMMDD.(zip|csv), optionally ticker-filtered."""

    name = "orats_strikes"

    def read(
        self,
        path: str | Path,
        tickers: Iterable[str] | None = None,
        chunksize: int = 200_000,
    ) -> pd.DataFrame:
        path = Path(path)
        wanted = {t.strip().upper() for t in tickers} if tickers else None
        usecols = list(ORATS_STRIKES_COLUMNS)

        def _chunks():
            if path.suffix != ".zip":
                return pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
            return _read_zip_chunks(path, usecols, chunksize)

        parts = []
        for chunk in _chunks():
            if wanted is not None:
                chunk = chunk[chunk["ticker"].astype(str).str.upper().isin(wanted)]
            if not chunk.empty:
                parts.append(chunk)
        if not parts:
            raise ValueError(
                f"No ORATS rows for tickers={sorted(wanted) if wanted else 'ALL'} in {path}"
            )
        return normalize_orats(pd.concat(parts, ignore_index=True))


def _read_zip_chunks(path: Path, usecols: list[str], chunksize: int):
    import zipfile

    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {names}")
        with archive.open(names[0]) as fh:
            yield from pd.read_csv(fh, usecols=usecols, chunksize=chunksize, low_memory=False)


def normalize_orats(raw: pd.DataFrame) -> pd.DataFrame:
    """Map ORATS strikes grain to NORMALIZED_COLUMNS.

    Conventions pinned on sample 2024-01-03: `delta` is call-delta magnitude
    in [0,1] shared by both legs (25-delta put <=> call-delta 0.75);
    `yte` drives DTE; `stkPx` is the underlying reference (spot_px is NaN
    for ETFs); snapshot ~15:46 ET, strictly before the session close.
    """
    missing = set(ORATS_STRIKES_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"ORATS input missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "ticker": raw["ticker"].astype(str).str.upper(),
            "trade_date": pd.to_datetime(raw["trade_date"], format="mixed").dt.tz_localize(None),
            "expir_date": pd.to_datetime(raw["expirDate"], format="mixed").dt.tz_localize(None),
            "dte_days": pd.to_numeric(raw["yte"], errors="coerce") * 365.0,
            "strike": pd.to_numeric(raw["strike"], errors="coerce"),
            "underlying_px": pd.to_numeric(raw["stkPx"], errors="coerce"),
            "call_mid_iv": pd.to_numeric(raw["cMidIv"], errors="coerce"),
            "put_mid_iv": pd.to_numeric(raw["pMidIv"], errors="coerce"),
            "call_bid": pd.to_numeric(raw["cBidPx"], errors="coerce"),
            "put_bid": pd.to_numeric(raw["pBidPx"], errors="coerce"),
            "call_volume": pd.to_numeric(raw["cVolu"], errors="coerce").fillna(0),
            "put_volume": pd.to_numeric(raw["pVolu"], errors="coerce").fillna(0),
            "call_oi": pd.to_numeric(raw["cOi"], errors="coerce").fillna(0),
            "put_oi": pd.to_numeric(raw["pOi"], errors="coerce").fillna(0),
            "call_delta": pd.to_numeric(raw["delta"], errors="coerce"),
            "source": SOURCE_ORATS_NEAR_EOD,
            "source_timestamp": "15:46 ET "
            + pd.to_datetime(raw["trade_date"], format="mixed").dt.strftime("%Y-%m-%d"),
        }
    )
    return frame.loc[:, list(NORMALIZED_COLUMNS)]
