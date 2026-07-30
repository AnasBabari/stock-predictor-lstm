from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from artifacts.registry import LocalArtifactRegistry
from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier
from tests.test_artifact_registry import _candidate


def test_signed_registry_accepts_valid_manifest(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    signer = Ed25519ManifestSigner.from_pem_file(private_path)
    verifier = Ed25519ManifestVerifier.from_pem_file(public_path)
    registry = LocalArtifactRegistry(
        tmp_path / "registry",
        require_signature=True,
        verify_signature=verifier,
    )
    source, manifest = _candidate(tmp_path, "signed-v1")

    signed = registry.stage(source, manifest, sign=signer)
    registry.promote("AAPL", "lstm", "signed-v1")

    assert signed.signature is not None
    assert registry.resolve("AAPL", "lstm").name == "signed-v1"


def test_verifier_rejects_tampered_payload():
    private = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private)
    verifier = Ed25519ManifestVerifier(private.public_key())
    signature = signer(b"approved")

    assert verifier(b"approved", signature)
    assert not verifier(b"rejected", signature)
    assert not verifier(b"approved", "not-base64")
