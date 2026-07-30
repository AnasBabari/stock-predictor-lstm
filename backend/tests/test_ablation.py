import pandas as pd

from experiments.ablation import feature_ablation_sets, run_feature_ablation
from experiments.runner import ExperimentConfig


def test_feature_sets_are_incremental_and_price_set_is_minimal():
    sets = feature_ablation_sets()
    assert sets["price"] == ["Close"]
    assert set(sets["ohlcv"]).issubset(sets["ohlcv_technical"])
    assert set(sets["ohlcv"]).issubset(sets["ohlcv_market"])
    assert set(sets["ohlcv_technical_market"]).issubset(sets["all_market_features"])


def test_ablation_uses_identical_dataset_boundaries(synthetic_feature_df):
    frame = pd.DataFrame(synthetic_feature_df)
    report = run_feature_ablation(
        frame,
        feature_sets=("price", "ohlcv"),
        config=ExperimentConfig(
            lookback=5,
            horizons=(1, 3),
            folds=2,
            min_train_size=50,
            validation_size=20,
            gap=3,
        ),
    )
    price_dataset = report["reports"]["price"]["dataset"]
    ohlcv_dataset = report["reports"]["ohlcv"]["dataset"]
    assert price_dataset["samples"] == ohlcv_dataset["samples"]
    assert price_dataset["first_origin_index"] == ohlcv_dataset["first_origin_index"]
    assert price_dataset["last_origin_index"] == ohlcv_dataset["last_origin_index"]
