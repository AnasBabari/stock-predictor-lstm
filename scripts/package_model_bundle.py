"""Create and sign immutable model bundles for release/object storage publication."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _payload(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def build(source: Path, output: Path, manifest_path: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    entries = {str(path.relative_to(source)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = {"schema_version": 1, "source": str(source), "files": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(source))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_payload(manifest))


def sign(manifest_path: Path, signature_path: Path) -> None:
    encoded = os.environ.get("MODEL_BUNDLE_PRIVATE_KEY_B64")
    if not encoded:
        raise RuntimeError("MODEL_BUNDLE_PRIVATE_KEY_B64 is required for signing.")
    key = serialization.load_pem_private_key(base64.b64decode(encoded), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Bundle signing key must be Ed25519.")
    signature_path.write_text(base64.b64encode(key.sign(manifest_path.read_bytes())).decode(), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sign", type=Path)
    parser.add_argument("--signature", type=Path)
    args = parser.parse_args()
    if args.sign:
        sign(args.sign, args.signature)
    else:
        build(args.source, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())