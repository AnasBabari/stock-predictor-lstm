"""Unit tests for GlobalMultimodalTrainer under chronological 70/15/15 protocol."""

import numpy as np
import pandas as pd

from research.volatility_forecasting.global_multimodal_trainer_v11 import (
    GlobalMultimodalTrainer,
)


def test_global_multimodal_trainer_pipeline_and_certification():
    n_samples = 300
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="B").strftime("%Y-%m-%d").tolist()

    rng = np.random.default_rng(42)
    x_num = rng.normal(0.0, 1.0, size=(n_samples, 32))
    x_news = rng.normal(0.0, 1.0, size=(n_samples, 18))

    # Cumulative returns for 7 horizons
    y_rets = np.cumsum(rng.normal(0.0005, 0.015, size=(n_samples, 7)), axis=1)
    y_rv = np.abs(rng.normal(0.0003, 0.0001, size=(n_samples, 7)))

    model, cert = GlobalMultimodalTrainer.train_and_certify(
        dates=dates,
        x_numeric=x_num,
        x_news=x_news,
        y_returns=y_rets,
        y_rv=y_rv,
        epochs=5,
    )

    assert cert.winning_model in ["V2_NUMERIC", "V3_MULTIMODAL"]
    assert "V1_BASELINE" in cert.sealed_test_metrics
    assert "V2_NUMERIC" in cert.sealed_test_metrics
    assert "V3_MULTIMODAL" in cert.sealed_test_metrics
    assert cert.train_dates[1] < cert.val_dates[0] < cert.val_dates[1] < cert.test_dates[0]
