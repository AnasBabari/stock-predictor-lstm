from __future__ import annotations

import numpy as np

from research.volatility_forecasting.baselines_v8 import _qlike
from research.volatility_forecasting.metrics import qlike_losses


def test_v8_qlike_matches_canonical_repository_implementation() -> None:
    realized = np.array([[1.0, 4.0], [2.0, 8.0]], dtype=np.float64)
    forecast = np.array([[2.0, 2.0], [1.0, 16.0]], dtype=np.float64)

    assert np.isclose(_qlike(realized, forecast), np.mean(qlike_losses(forecast, realized)))


def test_v8_qlike_argument_reversal_is_detectable() -> None:
    realized = np.array([[1.0], [4.0]], dtype=np.float64)
    forecast = np.array([[2.0], [3.0]], dtype=np.float64)

    assert not np.isclose(_qlike(realized, forecast), _qlike(forecast, realized))
