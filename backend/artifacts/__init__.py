"""Artifact registry primitives for evidence-gated model activation."""

from .registry import (
    ArtifactRegistryError,
    LocalArtifactRegistry,
    PromotionManifest,
)
from .signing import Ed25519ManifestSigner, Ed25519ManifestVerifier

__all__ = [
    "ArtifactRegistryError",
    "Ed25519ManifestSigner",
    "Ed25519ManifestVerifier",
    "LocalArtifactRegistry",
    "PromotionManifest",
]
