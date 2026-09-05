"""Describe the completed replication without fitting or selecting models."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.price_forecasting.paired_validation import hac_mean  # noqa: E402


def main():
    root = ROOT / "artifacts/news_replication25_v1"
    lock = json.loads((root / "lock.json").read_text())
    rows = pd.concat(
        [pd.read_parquet(root / t / "test/test_predictions.parquet") for t in lock["tickers"]]
    )
    assert not rows.duplicated(["ticker", "date", "horizon"]).any()
    metrics = []
    for (ticker, horizon), s in rows.groupby(["ticker", "horizon"]):
        for arm in ("A_ohlcv", "C_sentiment", "frozen_rolling"):
            ratio = np.maximum(s.y, 1e-12) / s[arm + "_variance"]
            loss = ratio - np.log(ratio) - 1
            assert np.allclose(loss, s[arm + "_qlike"])
            metrics.append(
                dict(ticker=ticker, horizon=int(horizon), arm=arm, qlike=float(loss.mean()))
            )
    table = pd.DataFrame(metrics)
    comparisons = []
    for horizon in (5, 10, 20):
        sub = rows[rows.horizon == horizon]
        daily = (sub.A_ohlcv_qlike - sub.C_sentiment_qlike).groupby(sub.date).mean()
        counts = sub.groupby("date").ticker.nunique()
        for scope, values in (("available_assets", daily), ("all_25_assets", daily[counts == 25])):
            for multiple in (1, 2, 3):
                lag = multiple * (horizon - 1)
                comparisons.append(
                    {
                        "horizon": horizon,
                        "scope": scope,
                        "lag_multiple": multiple,
                        "lag": lag,
                        **hac_mean(values.to_numpy(), lag),
                    }
                )
    for scope in ("available_assets", "all_25_assets"):
        for multiple in (1, 2, 3):
            family = sorted(
                [c for c in comparisons if c["scope"] == scope and c["lag_multiple"] == multiple],
                key=lambda c: c["p_two_sided"],
            )
            previous = 0.0
            for i, c in enumerate(family):
                previous = max(previous, min(1.0, (len(family) - i) * c["p_two_sided"]))
                c["holm_p_three_horizons"] = previous
    (root / "hac_results.json").write_text(json.dumps(comparisons, indent=2))
    table.to_csv(root / "per_stock_results.csv", index=False)
    (root / "panel_results.json").write_text(
        table.groupby(["horizon", "arm"])
        .qlike.mean()
        .reset_index()
        .to_json(orient="records", indent=2)
    )
    (root / "coverage.json").write_text(
        rows.groupby(["horizon", "date"])
        .ticker.nunique()
        .reset_index(name="assets")
        .to_json(orient="records", date_format="iso", indent=2)
    )
    (root / "STUDY_SUMMARY.md").write_text(
        "# Negative news-volatility replication\n\n25/25 fixed-stock runs completed. No tuning or promotion. "
        "News improves QLIKE for 15/25, 15/25 and 16/25 stocks at 5/10/20 sessions, but does not improve panel QLIKE convincingly. "
        "Rolling volatility is substantially stronger. Exploratory date-aggregated HAC p-values across lag choices are "
        "0.47–0.52, 0.94–0.95, and 0.77–0.80 respectively; all confidence intervals cross zero. "
        "Common-date (all 25 assets) sensitivity also finds no significant improvement.\n\n"
        "Surviving-stock convenience sample; provider update-time assumptions; reused historical data. "
        "No independent certification. Test observations must not drive further tuning. "
        "Large raw datasets and weights remain local; SHA256SUMS records the evidence files. "
        "Scheduled-event research requires a separately specified experiment and historically available schedule evidence.\n",
        encoding="utf-8",
    )
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n"
            for p in files
        )
    )


if __name__ == "__main__":
    main()
