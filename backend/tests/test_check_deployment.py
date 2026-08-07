import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_deployment.py"
spec = importlib.util.spec_from_file_location("check_deployment", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
sys.modules["check_deployment"] = smoke
spec.loader.exec_module(smoke)


def test_safe_error_sanitizes_multiline_message():
    assert "\n" not in smoke.safe_error(RuntimeError("bad\nsecret detail"))


def test_smoke_run_reports_json_failure_without_secret_headers(monkeypatch):
    def fail(*_args, **_kwargs):
        raise TimeoutError("timed out with token=hidden")

    monkeypatch.setattr(smoke, "get_json", fail)
    result = smoke.run(
        SimpleNamespace(
            base_url="https://example.invalid",
            ticker="MSFT",
            expected_commit=None,
            expected_environment=None,
            timeout=0.01,
            restart_window=0,
            cors_origin=None,
            training_mode="browser_only",
        )
    )
    assert result["status"] == "failed"
    assert result["checks"][0]["name"] == "request"


def test_smoke_run_accepts_expected_identity_and_contract(monkeypatch):
    def fake_get(_base, path, **_kwargs):
        if path == "/health":
            return {
                "status": "ok",
                "deployment": {"commit": "abcdef123456", "environment": "production"},
            }, {"access-control-allow-origin": "https://app.example"}
        if path == "/models":
            return {
                "server_models": {"status": "disabled", "training_mode": "browser_only"},
                "browser_training": {"status": "available", "storage": "indexeddb"},
            }, {}
        if path.startswith("/api/v1/training-data"):
            return {
                "schema_version": 4,
                "feature_names": smoke.EXPECTED_FEATURES,
                "window_size": 60,
                "output_width": 30,
                "dates": ["d"],
                "features": [[1.0] * len(smoke.EXPECTED_FEATURES)],
                "data_snapshot": {
                    "target_mode": smoke.EXPECTED_TARGET_MODE,
                    "quality": {"status": "clean", "checks": {}, "issues": []},
                },
            }, {}
        if path.startswith("/api/v1/predict/direction"):
            return {
                "metadata": {
                    "engine": {"baseline_fallback": True, "role": "server_disabled_fallback"}
                }
            }, {}
        return {
            "metadata": {
                "engine": {"baseline_fallback": True, "role": "server_disabled_fallback"},
                "execution": {"mode": "baseline_fallback"},
            }
        }, {}

    monkeypatch.setattr(smoke, "get_json", fake_get)
    result = smoke.run(
        argparse.Namespace(
            base_url="https://api.example",
            ticker="MSFT",
            expected_commit="abcdef1234567890",
            expected_environment="production",
            timeout=1,
            restart_window=0,
            cors_origin="https://app.example",
            training_mode="browser_only",
        )
    )
    assert result["status"] == "passed"


def test_smoke_run_accepts_server_pretrained_deployment(monkeypatch):
    def fake_get(_base, path, **_kwargs):
        if path == "/health":
            return {
                "status": "ok",
                "deployment": {"commit": "abcdef123456", "environment": "production"},
            }, {}
        if path == "/models":
            return {
                "server_models": {
                    "status": "configured",
                    "training_mode": "server_pretrained",
                },
                "browser_training": {"status": "available", "storage": "indexeddb"},
            }, {}
        if path.startswith("/api/v1/training-data"):
            return {
                "schema_version": 4,
                "feature_names": smoke.EXPECTED_FEATURES,
                "window_size": 60,
                "output_width": 30,
                "dates": ["d"],
                "features": [[1.0] * len(smoke.EXPECTED_FEATURES)],
                "data_snapshot": {
                    "target_mode": smoke.EXPECTED_TARGET_MODE,
                    "quality": {"status": "clean", "checks": {}, "issues": []},
                },
            }, {}
        return {"metadata": {"engine": {}}}, {}

    monkeypatch.setattr(smoke, "get_json", fake_get)
    result = smoke.run(
        argparse.Namespace(
            base_url="https://api.example",
            ticker="MSFT",
            expected_commit="abcdef1234567890",
            expected_environment="production",
            timeout=1,
            restart_window=0,
            cors_origin=None,
            training_mode="server_pretrained",
        )
    )
    assert result["status"] == "passed"
