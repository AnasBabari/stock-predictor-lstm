"""Foundation package for server-pretrained forecast artifacts.

Immutable, signed, schema-v4 artifacts with promotion pointers; additive to
the existing browser-training flow (all feature flags default to OFF).
"""

from server_models.compatibility import check_record_compatibility, is_fresh
from server_models.contracts import (
    FORECAST_LENGTH,
    CompatibilityReport,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
    git_commit_short,
    make_version_id,
)
from server_models.db import (
    SCHEMA_SQL,
    InMemoryRegistry,
    ModelRegistry,
    ModelRegistryError,
    PostgresRegistry,
)
from server_models.signing_manifests import (
    ServerArtifactIntegrityError,
    sha256_digest,
    sign_bundle,
    verify_bundle,
)
from server_models.storage import (
    InMemoryObjectStore,
    ObjectStore,
    ObjectStoreError,
    S3ObjectStore,
)

__all__ = [
    "FORECAST_LENGTH",
    "SCHEMA_SQL",
    "CompatibilityReport",
    "InMemoryObjectStore",
    "InMemoryRegistry",
    "ModelRegistry",
    "ModelRegistryError",
    "ObjectStore",
    "ObjectStoreError",
    "PostgresRegistry",
    "ReproducibilityMetadata",
    "RobustScalerParams",
    "S3ObjectStore",
    "ServerArtifactIntegrityError",
    "ServerArtifactKey",
    "ServerForecastBundle",
    "ServerModelRecord",
    "check_record_compatibility",
    "git_commit_short",
    "is_fresh",
    "make_version_id",
    "sha256_digest",
    "sign_bundle",
    "verify_bundle",
]
