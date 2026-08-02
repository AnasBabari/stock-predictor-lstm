import api
from config import settings


def test_deployment_identity_is_sanitised(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abcdef1234567890abcdef1234567890abcdef12")
    monkeypatch.setattr(settings, "deployment_provider", "render")
    monkeypatch.setattr(settings, "deployment_environment", "production")
    deployment = api._deployment_identity()
    assert deployment == {
        "provider": "render",
        "environment": "production",
        "commit": "abcdef123456",
        "preview": False,
    }
    assert "host" not in deployment
    assert "memory" not in deployment


def test_preview_cors_regex_is_preview_only(monkeypatch):
    monkeypatch.setattr(settings, "deployment_environment", "production")
    monkeypatch.setattr(settings, "preview_cors_origin_regex", r"https://[a-z0-9-]+\.vercel\.app")
    assert api._deployment_identity()["preview"] is False
    monkeypatch.setattr(settings, "deployment_environment", "preview")
    assert api._deployment_identity()["preview"] is True


def test_invalid_commit_is_omitted(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "not-a-sha-or-secret-like-value")
    monkeypatch.setattr(settings, "deployment_commit", None)
    assert api._deployment_commit() is None


def test_render_pull_request_overrides_production_environment(monkeypatch):
    monkeypatch.setattr(settings, "deployment_environment", "production")
    monkeypatch.setenv("IS_PULL_REQUEST", "true")
    assert api._deployment_environment() == "preview"
    assert api._deployment_identity()["preview"] is True
