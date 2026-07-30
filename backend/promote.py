"""Operator CLI for staging, promoting, and rolling back evidence-backed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts.registry import ArtifactRegistryError, LocalArtifactRegistry, PromotionManifest
from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--rollback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.rollback:
            if not args.manifest:
                raise ArtifactRegistryError("--manifest is required for rollback (JSON metadata).")
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            registry = LocalArtifactRegistry(args.registry)
            path = registry.rollback(payload["ticker"], payload["engine"])
            print(json.dumps({"status": "rolled_back", "path": str(path)}))
            return 0
        if args.source is None or args.manifest is None:
            raise ArtifactRegistryError("--source and --manifest are required for promotion.")
        manifest = PromotionManifest.from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        verifier = (
            Ed25519ManifestVerifier.from_pem_file(args.public_key) if args.public_key else None
        )
        registry = LocalArtifactRegistry(
            args.registry,
            require_signature=bool(args.public_key),
            verify_signature=verifier,
        )
        signer = Ed25519ManifestSigner.from_pem_file(args.private_key) if args.private_key else None
        registry.stage(args.source, manifest, sign=signer)
        path = registry.promote(manifest.ticker, manifest.engine, manifest.version)
        print(json.dumps({"status": "promoted", "path": str(path)}))
        return 0
    except (ArtifactRegistryError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"promotion failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
