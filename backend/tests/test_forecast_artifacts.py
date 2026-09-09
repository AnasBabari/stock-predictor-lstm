from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from services import forecast_artifacts as store


def test_pipeline_round_trip_preserves_fitted_scaler_and_predictions(tmp_path):
    x = np.arange(80, dtype=float).reshape(20, 4)
    model = make_pipeline(RobustScaler(), Ridge()).fit(x, np.sin(x[:, :2]))
    path = tmp_path / "artifact.json"
    store.save(path, "key", {"model": model})
    loaded = store.load(path, "key")
    assert loaded is not None
    np.testing.assert_array_equal(model.predict(x), loaded["model"].predict(x))
    assert store.load(path, "different-key") is None


@pytest.mark.parametrize("fault", ["bytes", "runtime", "version", "path", "missing"])
def test_invalid_artifact_fails_closed_before_deserialization(tmp_path, monkeypatch, fault):
    path = tmp_path / "artifact.json"
    store.save(path, "key", {"model": "fixture"})
    manifest = json.loads(path.read_text())
    blob = tmp_path / (manifest["sha256"] + ".joblib")
    if fault == "bytes":
        blob.write_bytes(b"corrupt")
    elif fault == "missing":
        blob.unlink()
    else:
        field, value = {
            "runtime": ("runtime", {}),
            "version": ("version", -1),
            "path": ("sha256", "../outside"),
        }[fault]
        manifest[field] = value
        path.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid envelope must not be deserialized")

    monkeypatch.setattr(store.joblib, "load", forbidden)
    assert store.load(path, "key") is None
