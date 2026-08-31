from __future__ import annotations

import numpy as np
import pandas as pd

from panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from services import volatility_snapshot


def _market_frame(rows: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0003, 0.012, size=rows)
    close = 100 * np.exp(np.cumsum(returns))
    overnight = rng.normal(0, 0.003, size=rows)
    open_ = close * np.exp(overnight)
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, size=rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, size=rows))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, size=rows),
        },
        index=pd.bdate_range("2024-01-02", periods=rows),
    )


def test_volatility_snapshot_matches_frozen_deployable_contract(monkeypatch) -> None:
    frame = _market_frame()
    monkeypatch.setattr(volatility_snapshot, "_download_ohlcv", lambda _ticker: frame.copy())
    snapshot = volatility_snapshot.build_volatility_inference_snapshot("msft")
    assert snapshot.ticker == "MSFT"
    assert snapshot.origin_date == frame.index[-1].date().isoformat()
    assert snapshot.origin_close == frame["Close"].iloc[-1]
    assert snapshot.feature_names == DEPLOYABLE_FEATURE_COLUMNS_V5
    assert snapshot.features.shape == (60, len(DEPLOYABLE_FEATURE_COLUMNS_V5))
    assert snapshot.causal_har_variance.shape == (6,)
    assert np.all(snapshot.causal_har_variance > 0)
    assert {
        "causal_log_har",
        "riskmetrics_ewma_c2c",
        "rolling_c2c_5",
        "rolling_c2c_20",
        "rolling_c2c_60",
        "rolling_c2c_multiscale",
    }.issubset(snapshot.baseline_candidates)
    assert len(snapshot.snapshot_id) == 64


def test_volatility_snapshot_identity_changes_with_latest_observation(monkeypatch) -> None:
    frame = _market_frame()
    monkeypatch.setattr(volatility_snapshot, "_download_ohlcv", lambda _ticker: frame.copy())
    original = volatility_snapshot.build_volatility_inference_snapshot("NMM")
    changed = frame.copy()
    changed.loc[changed.index[-1], ["Open", "High", "Low", "Close"]] *= 1.01
    monkeypatch.setattr(volatility_snapshot, "_download_ohlcv", lambda _ticker: changed.copy())
    updated = volatility_snapshot.build_volatility_inference_snapshot("NMM")
    assert updated.snapshot_id != original.snapshot_id
    assert updated.origin_close != original.origin_close


def test_snapshot_har_baseline_uses_close_to_close_proxy(monkeypatch) -> None:
    frame = _market_frame()
    proxy_frame = pd.DataFrame(
        {
            "RV_C2C": np.full(len(frame), 0.01),
            "RV_Total": np.full(len(frame), 0.09),
        },
        index=frame.index,
    )
    captured: dict[str, np.ndarray] = {}

    def fake_har(rv_daily, horizons):
        captured["rv"] = np.asarray(rv_daily, dtype=float)
        return np.full((len(frame), len(horizons)), 0.02, dtype=float)

    monkeypatch.setattr(volatility_snapshot, "_download_ohlcv", lambda _ticker: frame.copy())
    monkeypatch.setattr(volatility_snapshot, "realized_variance_proxies", lambda _raw: proxy_frame)
    monkeypatch.setattr(volatility_snapshot, "causal_log_har_forecasts", fake_har)
    monkeypatch.setattr(
        volatility_snapshot,
        "future_trading_dates",
        lambda *_args, **_kwargs: (
            tuple(
                (frame.index[-1] + pd.Timedelta(days=offset)).date().isoformat()
                for offset in range(1, 31)
            ),
            "synthetic",
        ),
    )

    volatility_snapshot.build_volatility_inference_snapshot("MSFT")

    np.testing.assert_allclose(captured["rv"], proxy_frame["RV_C2C"].to_numpy())
