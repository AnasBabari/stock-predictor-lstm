"""Download unrevised market-daily macro series to a local parquet cache.

Universe: SPY (target proxy) + ^VIX, ^TNX, CL=F, DX-Y.NYB, HYG.
Market-settled closes are never revised, so no vintage logic is needed.
Monthly revised surveys (PAYEMS/INDPRO/CPIAUCSL/UNRATE) are EXCLUDED here;
see metadata flags. Re-runs never touch training code paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TICKERS = ["SPY", "^VIX", "^TNX", "CL=F", "DX-Y.NYB", "HYG"]
START = "2015-01-01"
END = "2025-06-01"
OUT_DIR = REPO_ROOT / "data" / "macro"


def main() -> int:
    import pandas as pd
    import yfinance as yf

    frame = yf.download(
        TICKERS, start=START, end=END, auto_adjust=False, progress=False, threads=True
    )
    if frame.empty:
        raise RuntimeError("yfinance returned no rows for macro universe")
    closes = frame["Close"].copy()
    closes.columns = [str(c).strip() for c in closes.columns]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.sort_index()
    closes = closes.loc[~closes.index.duplicated(keep="last")]
    # Total-return target needs dividend-adjusted SPY (21-day windows cross ex-div dates).
    adj = frame["Adj Close"]["SPY"].copy()
    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    closes["SPY_ADJ"] = adj
    # Align everything to SPY sessions; forward-fill stale prints (max 3 sessions).
    closes = closes.reindex(closes.index)
    spy_sessions = closes["SPY"].dropna().index
    aligned = closes.loc[spy_sessions].ffill(limit=3)
    if aligned.isna().any().any():
        bad = aligned.columns[aligned.isna().any()].tolist()
        raise RuntimeError(f"Unfillable gaps in market cache columns: {bad}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "market_dailies.parquet"
    aligned.to_parquet(parquet_path)
    meta = {
        "tickers": TICKERS,
        "start": START,
        "end": END,
        "sessions": int(len(aligned)),
        "first_session": aligned.index.min().date().isoformat(),
        "last_session": aligned.index.max().date().isoformat(),
        "macro_mode": "market_dailies_only",
        "monthly_revised_block": "excluded",
        "revision_risk": "none_by_construction_session_settled_closes",
        "source": "yfinance",
    }
    (OUT_DIR / "market_dailies.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {parquet_path} rows={len(aligned)} cols={list(aligned.columns)}", flush=True)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
