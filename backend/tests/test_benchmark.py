import numpy as np
import pandas as pd

from benchmark import _snapshot_features, build_parser
from snapshot import create_market_snapshot


def test_benchmark_parser_builds_reproducible_defaults():
    args = build_parser().parse_args([])
    assert args.ticker == "AAPL"
    assert args.lookback == 60
    assert args.horizons == (1, 5, 20)
    assert args.target_type == "log_return"
    assert "price" in args.feature_sets
    assert args.models == "baseline"
    assert args.snapshot is None


def test_benchmark_parser_accepts_verified_snapshot_and_neural_suite(tmp_path):
    args = build_parser().parse_args(
        [
            "--snapshot",
            str(tmp_path / "manifest.json"),
            "--models",
            "baseline,lstm,gru",
            "--seed",
            "314",
        ]
    )
    assert args.snapshot.name == "manifest.json"
    assert args.models == "baseline,lstm,gru"
    assert args.seed == 314


def test_benchmark_builds_features_only_from_verified_snapshot(tmp_path):
    index = pd.date_range("2024-01-01", periods=100, freq="B")
    close = np.arange(100.0, 200.0)

    def frame(_ticker, **_kwargs):
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 1_000,
            },
            index=index,
        )

    destination = tmp_path / "snapshot"
    create_market_snapshot(
        ["AAPL", "SPY", "QQQ", "^VIX", "^TNX"],
        start="2024-01-01",
        end="2025-01-01",
        output=destination,
        downloader=frame,
        benchmark_universe=["AAPL"],
    )

    features, prices, dates, metadata = _snapshot_features("AAPL", destination / "manifest.json")
    assert len(features) == len(prices) == len(dates)
    assert metadata["market_context"]["status"] == "complete"
    assert metadata["snapshot_id"]
