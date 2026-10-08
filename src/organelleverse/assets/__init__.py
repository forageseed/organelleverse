"""Knowledge assets: reference libraries, models, and databases with identity (L2).

A scientific method (L3) *uses* an asset; it does not compute one. Until an asset
has a stable name and a content hash, a result cannot record which reference
library produced it, and a verification record cannot notice when that library
is swapped underneath it.

This layer supplies exactly three things:

- a stable locator, ``ov-asset:<namespace>/<name>@<version>``;
- a manifest declaring content identity plus provenance and license;
- pluggable resolution from locator to a directory on this machine.

The declaration slot already exists elsewhere: a method declares an asset with
``DependencySpec(kind=DependencyKind.DATABASE | DependencyKind.MODEL,
locator="ov-asset:...")``. This layer is what makes that locator mean something.

This layer holds no scientific judgement. What an asset is *for* belongs to the
method that uses it.
"""

from __future__ import annotations

from .identity import ASSET_LOCATOR_PATTERN, AssetLocator
from .manifest import (
    ASSET_MANIFEST_FILENAME,
    ASSET_MANIFEST_SCHEMA_VERSION,
    AssetContent,
    AssetManifest,
    AssetProvenance,
    compute_content,
)
from .resolvers import (
    AssetResolver,
    PackagedAssetResolver,
    content_hash,
    describe,
    list_assets,
    list_resolvers,
    locate,
    register_resolver,
    unregister_resolver,
    verify,
)

__all__ = [
    "ASSET_LOCATOR_PATTERN",
    "ASSET_MANIFEST_FILENAME",
    "ASSET_MANIFEST_SCHEMA_VERSION",
    "AssetContent",
    "AssetLocator",
    "AssetManifest",
    "AssetProvenance",
    "AssetResolver",
    "PackagedAssetResolver",
    "compute_content",
    "content_hash",
    "describe",
    "list_assets",
    "list_resolvers",
    "locate",
    "register_resolver",
    "unregister_resolver",
    "verify",
]
