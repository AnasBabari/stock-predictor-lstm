"""Ed25519 manifest signing without exposing private key material."""

from __future__ import annotations

import base64
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from artifacts.registry import ArtifactRegistryError


def _decode_signature(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ArtifactRegistryError("Promotion signature encoding is invalid.") from exc


class Ed25519ManifestSigner:
    """Callable signer loaded from a PEM private key."""

    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key

    @classmethod
    def from_pem_file(
        cls, path: str | Path, *, password: bytes | None = None
    ) -> Ed25519ManifestSigner:
        try:
            key = serialization.load_pem_private_key(
                Path(path).read_bytes(),
                password=password,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactRegistryError("Ed25519 private key could not be loaded.") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise ArtifactRegistryError("Promotion private key must use Ed25519.")
        return cls(key)

    def __call__(self, payload: bytes) -> str:
        return base64.b64encode(self._private_key.sign(payload)).decode("ascii")


class Ed25519ManifestVerifier:
    """Callable verifier loaded from a PEM public key."""

    def __init__(self, public_key: Ed25519PublicKey):
        self._public_key = public_key

    @classmethod
    def from_pem_file(cls, path: str | Path) -> Ed25519ManifestVerifier:
        try:
            key = serialization.load_pem_public_key(Path(path).read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactRegistryError("Ed25519 public key could not be loaded.") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise ArtifactRegistryError("Promotion public key must use Ed25519.")
        return cls(key)

    def __call__(self, payload: bytes, signature: str) -> bool:
        try:
            self._public_key.verify(_decode_signature(signature), payload)
        except (InvalidSignature, ArtifactRegistryError):
            return False
        return True
