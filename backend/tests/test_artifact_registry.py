import hashlib
from pathlib import Path

import pytest

from artifacts.registry import ArtifactRegistryError, LocalArtifactRegistry, PromotionManifest


def _candidate(tmp_path: Path, name: str, *, promoted: bool = True):
    source = tmp_path / f"source-{name}"
    source.mkdir()
    payload = f"model-{name}".encode()
    (source / "engine.bin").write_bytes(payload)
    manifest = PromotionManifest.create(
        ticker="AAPL",
        engine="lstm",
        version=name,
        benchmark_id="bench-1",
        snapshot_id="snapshot-1",
        promoted=promoted,
        evidence={"reasons": [] if promoted else ["failed gate"]},
        files={"engine.bin": hashlib.sha256(payload).hexdigest()},
    )
    return source, manifest


def test_registry_stages_and_promotes_approved_candidate(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "registry")
    source, manifest = _candidate(tmp_path, "v1")

    registry.stage(source, manifest)
    activated = registry.promote("AAPL", "lstm", "v1")

    assert activated.name == "v1"
    assert registry.resolve("AAPL", "lstm").name == "v1"
    assert registry.resolve("AAPL", "lstm", "eligible").name == "v1"


def test_registry_rejects_failed_evidence(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "registry")
    source, manifest = _candidate(tmp_path, "v1", promoted=False)
    registry.stage(source, manifest)

    with pytest.raises(ArtifactRegistryError, match="rejected"):
        registry.promote("AAPL", "lstm", "v1")

    assert registry.resolve("AAPL", "lstm") is None
    assert registry.resolve("AAPL", "lstm", "rejected").name == "v1"


def test_failed_probe_restores_previous_version(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "registry")
    source_one, manifest_one = _candidate(tmp_path, "v1")
    source_two, manifest_two = _candidate(tmp_path, "v2")
    registry.stage(source_one, manifest_one)
    registry.promote("AAPL", "lstm", "v1")
    registry.stage(source_two, manifest_two)

    with pytest.raises(ArtifactRegistryError, match="rolled back"):
        registry.promote(
            "AAPL",
            "lstm",
            "v2",
            probe=lambda _path: (_ for _ in ()).throw(ValueError("invalid inference")),
        )

    assert registry.resolve("AAPL", "lstm").name == "v1"
    assert registry.resolve("AAPL", "lstm", "previous").name == "v1"


def test_registry_detects_modified_file(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "registry")
    source, manifest = _candidate(tmp_path, "v1")
    registry.stage(source, manifest)
    staged = tmp_path / "registry" / "AAPL" / "lstm" / "versions" / "v1"
    (staged / "engine.bin").write_text("tampered")

    with pytest.raises(ArtifactRegistryError, match="integrity"):
        registry.resolve("AAPL", "lstm", "candidate")


def test_required_signature_fails_closed(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "registry", require_signature=True)
    source, manifest = _candidate(tmp_path, "v1")

    with pytest.raises(ArtifactRegistryError, match="signed"):
        registry.stage(source, manifest)
