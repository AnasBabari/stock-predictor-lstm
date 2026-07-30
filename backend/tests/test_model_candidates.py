import numpy as np

from config import FEATURES, WINDOW_SIZE
from model import build_gru_model


def test_gru_candidate_has_the_same_direct_horizon_contract_as_lstm():
    model = build_gru_model(forecast_days=3, num_features=len(FEATURES))
    prediction = model.predict(np.ones((1, WINDOW_SIZE, len(FEATURES))), verbose=0)

    assert prediction.shape == (1, 3)
