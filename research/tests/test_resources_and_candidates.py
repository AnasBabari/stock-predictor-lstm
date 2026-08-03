from __future__ import annotations

import sys

import numpy as np
import pytest
from stock_autoresearch.candidates import (
    ELASTIC_NET_TUNING_GRID,
    ElasticNetCandidate,
    RidgeCandidate,
    elastic_net_family_factories,
    elastic_net_family_name,
)
from stock_autoresearch.resources import sample_cuda_memory


def test_resource_sample_is_safe_without_cuda() -> None:
    budget = type("Budget", (), {"vram_warning_mb": 5200, "vram_kill_mb": 5500})()
    sample = sample_cuda_memory(budget)
    assert sample.peak_vram_mb >= 0
    assert not sample.exceeded


def test_elastic_net_candidate_matches_ridge_contract() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 8, 3)).astype(np.float64)
    y = x[:, -1, 0] * 2.0 - x[:, -1, 1]

    candidate = ElasticNetCandidate(alpha=1e-8)
    fitted = candidate.fit(x, y)
    assert fitted is candidate

    prediction = candidate.predict(x[:5])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(candidate.predict(x), y, atol=1e-3)

    description = candidate.describe()
    assert description["family"] == "elastic_net"
    assert description["alpha"] == 1e-8
    assert description["l1_ratio"] == 0.5
    assert candidate.parameter_count() == x.shape[2] + 1


def test_elastic_net_candidate_is_deterministic_like_ridge() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, 6, 2)).astype(np.float64)
    y = rng.normal(size=20)

    ridge = RidgeCandidate(alpha=5.0).fit(x, y)
    elastic_net = ElasticNetCandidate(alpha=1.0, l1_ratio=0.5).fit(x, y)

    np.testing.assert_array_equal(
        ElasticNetCandidate(alpha=1.0, l1_ratio=0.5).fit(x, y).predict(x),
        elastic_net.predict(x),
    )
    assert ridge.parameter_count() == elastic_net.parameter_count()


def test_elastic_net_tuning_grid_variants_train_and_predict() -> None:
    factories = elastic_net_family_factories()
    assert len(factories) == len(ELASTIC_NET_TUNING_GRID) == 8
    # Family names are unique and follow elastic_net_a<alpha>_l<l1_ratio*100>.
    names = [elastic_net_family_name(alpha, l1) for alpha, l1 in ELASTIC_NET_TUNING_GRID]
    assert len(set(names)) == len(names)
    assert set(names) == set(factories)
    assert "elastic_net" not in factories  # baseline family stays separate

    rng = np.random.default_rng(11)
    x = rng.normal(size=(32, 4, 5)).astype(np.float64)
    y = x[:, -1, 0] - 0.5 * x[:, -1, 2] + 0.1 * rng.normal(size=32)

    for alpha, l1_ratio in ELASTIC_NET_TUNING_GRID:
        name = elastic_net_family_name(alpha, l1_ratio)
        candidate = factories[name](seed=0)
        fitted = candidate.fit(x, y)
        assert fitted is candidate
        prediction = candidate.predict(x[:5])
        assert prediction.shape == (5,)
        assert np.isfinite(prediction).all()

        description = candidate.describe()
        assert description["family"] == name
        assert description["alpha"] == alpha
        assert description["l1_ratio"] == l1_ratio
        assert candidate.parameter_count() == x.shape[2] + 1


def test_elastic_net_baseline_family_unchanged() -> None:
    candidate = ElasticNetCandidate()
    assert candidate.name == "elastic_net"
    assert candidate.alpha == 1.0
    assert candidate.l1_ratio == 0.5


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="PyTorch CUDA wheel is not compatible with Python 3.14"
)
def test_torch_lstm_candidate_has_finite_output() -> None:
    pytest.importorskip("torch")
    from stock_autoresearch.torch_candidates import TorchLSTMCandidate

    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 8, 2)).astype(np.float32)
    y = rng.normal(size=12).astype(np.float32)
    candidate = TorchLSTMCandidate(hidden_size=4, epochs=1, device="cpu").fit(x, y)
    prediction = candidate.predict(x[:3])
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
