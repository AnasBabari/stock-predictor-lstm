"""Artifact registry primitives for evidence-gated model activation."""

from artifacts.registry import (
    ArtifactRegistryError,
    LocalArtifactRegistry,
    PromotionManifest,
)
from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier

__all__ = [
    "ArtifactRegistryError",
    "Ed25519ManifestSigner",
    "Ed25519ManifestVerifier",
    "LocalArtifactRegistry",
    "PromotionManifest",
]
