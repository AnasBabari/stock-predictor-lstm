from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import pretrain
from config import FEATURES, MAX_FORECAST_DAYS, Settings


def _snapshot():
    index = pd.date_range("2024-01-01", periods=500, freq="B")
    frame = pd.DataFrame(
        {name: np.arange(len(index), dtype=float) + 1 for name in FEATURES}, index=index
    )
    return frame, frame["Close"].to_numpy(), index, {"snapshot_id": "one"}


def _arrays():
    return (
        np.zeros((2, 60, len(FEATURES))),
        np.zeros((1, 60, len(FEATURES))),
        np.zeros((2, MAX_FORECAST_DAYS)),
        np.zeros((1, MAX_FORECAST_DAYS)),
        MagicMock(),
        [],
        [],
    )


def test_cli_requires_and_validates_tickers():
    with pytest.raises(SystemExit) as missing:
        pretrain.main([])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as invalid:
        pretrain.main(["--ticker", "../AAPL"])
    assert invalid.value.code == 2
    assert pretrain.normalise_ticker(" aapl ") == "AAPL"


def test_pretrain_defaults_to_both_models_and_fetches_once(monkeypatch, capsys):
    snapshot = _snapshot()
    arrays = _arrays()
    fetch = MagicMock(return_value=snapshot)
    price = MagicMock(return_value=arrays)
    direction = MagicMock(return_value=arrays)
    load = MagicMock(return_value=(MagicMock(), arrays[4]))
    monkeypatch.setattr(pretrain, "fetch_data", fetch)
    monkeypatch.setitem(pretrain.PREPARERS, "lstm", price)
    monkeypatch.setitem(pretrain.PREPARERS, "bilstm_attention_direction", direction)
    monkeypatch.setattr(pretrain, "load_or_train", load)

    assert pretrain.main(["--ticker", "aapl", "--ticker", "AAPL"]) == 0
    assert fetch.call_count == 1
    assert price.call_count == direction.call_count == 1
    assert load.call_count == 2
    assert {call.args[6] for call in load.call_args_list} == set(pretrain.MODEL_TYPES)
    assert all(call.kwargs["allow_stale_fallback"] is False for call in load.call_args_list)
    assert "SUMMARY ready=2 failed=0" in capsys.readouterr().out


def test_pretrain_continues_after_model_failure(monkeypatch, capsys):
    monkeypatch.setattr(pretrain, "fetch_data", MagicMock(return_value=_snapshot()))
    monkeypatch.setitem(pretrain.PREPARERS, "lstm", MagicMock(return_value=_arrays()))
    monkeypatch.setitem(
        pretrain.PREPARERS, "bilstm_attention_direction", MagicMock(return_value=_arrays())
    )
    load = MagicMock(side_effect=[RuntimeError("failed"), (MagicMock(), MagicMock())])
    monkeypatch.setattr(pretrain, "load_or_train", load)

    code = pretrain.main(["--ticker", "AAPL"])
    captured = capsys.readouterr()
    assert code == 1
    assert load.call_count == 2
    assert "SUMMARY ready=1 failed=1" in captured.out
    assert "ERROR AAPL/lstm" in captured.err


def test_pretrain_does_not_count_stale_fallback_as_ready(monkeypatch, capsys):
    monkeypatch.setattr(pretrain, "fetch_data", MagicMock(return_value=_snapshot()))
    monkeypatch.setitem(pretrain.PREPARERS, "lstm", MagicMock(return_value=_arrays()))
    load = MagicMock(side_effect=RuntimeError("training failed"))
    monkeypatch.setattr(pretrain, "load_or_train", load)

    assert pretrain.main(["--ticker", "AAPL", "--model-type", "lstm"]) == 1
    assert load.call_args.kwargs["allow_stale_fallback"] is False
    captured = capsys.readouterr()
    assert "READY" not in captured.out
    assert "SUMMARY ready=0 failed=1" in captured.out


def test_trusted_proxy_configuration_accepts_exact_ips_only():
    settings = Settings(trusted_proxy_ips=["192.0.2.1", "2001:db8::1", "192.0.2.1"])
    assert settings.trusted_proxy_ips == ["192.0.2.1", "2001:db8::1"]
    with pytest.raises(ValueError):
        Settings(trusted_proxy_ips=["0.0.0.0/0"])
    with pytest.raises(ValueError):
        Settings(trusted_proxy_ips=["*"])
