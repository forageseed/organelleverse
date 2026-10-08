from __future__ import annotations

from uuid import uuid4

import pytest

from organelleverse.capabilities.assets import (
    AssetIdentity,
    register_asset_resolver,
    resolve_asset_identity,
)
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.frozen import FrozenMap


class _Resolver:
    def __init__(self, expected: str, content_hash: str | None) -> None:
        self.expected = expected
        self.resolved = content_hash

    def content_hash(self, locator: str) -> str | None:
        return self.resolved if locator == self.expected else None


def _reason(error: OrganelleDependencyError) -> object:
    assert isinstance(error.details, FrozenMap)
    return error.details["reason"]


def test_registered_scheme_resolver_returns_a_frozen_content_identity() -> None:
    scheme = f"test-{uuid4().hex}"
    locator = f"{scheme}:reference/v1"
    register_asset_resolver(scheme, _Resolver(locator, "sha256:" + "a" * 64))

    assert resolve_asset_identity(locator) == AssetIdentity(
        locator=locator, content_hash="sha256:" + "a" * 64
    )


def test_unknown_asset_scheme_fails_closed_with_the_required_public_code() -> None:
    locator = f"unknown-{uuid4().hex}:reference/v1"

    with pytest.raises(OrganelleDependencyError) as captured:
        resolve_asset_identity(locator)

    assert captured.value.code == "capability.asset_unresolvable"
    assert _reason(captured.value) == "asset.unresolvable"


def test_resolver_returning_none_fails_closed_instead_of_silently_skipping_asset() -> None:
    scheme = f"test-{uuid4().hex}"
    locator = f"{scheme}:missing/v1"
    register_asset_resolver(scheme, _Resolver(locator, None))

    with pytest.raises(OrganelleDependencyError) as captured:
        resolve_asset_identity(locator)

    assert captured.value.code == "capability.asset_unresolvable"
    assert _reason(captured.value) == "asset.unresolvable"


def test_asset_identity_rejects_a_non_content_hash() -> None:
    with pytest.raises(ValueError):
        AssetIdentity(locator="test:bad", content_hash="version-1")
