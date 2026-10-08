from __future__ import annotations

from typing import Any

import pytest

from organelleverse.assembly.environment_specs import PMAT_ENVIRONMENT
from organelleverse.assembly.environment_versions import (
    environment_spec_for_version,
    resolve_backend_version,
)
from organelleverse.core.errors import OrganelleDependencyError


class _ReleaseClient:
    def __init__(self, tag: str = "v2.1.6", commit: str = "a" * 40) -> None:
        self.tag = tag
        self.commit = commit
        self.calls: list[str] = []

    def release(self, selector: str) -> dict[str, Any]:
        self.calls.append(selector)
        return {
            "tag": self.tag,
            "commit": self.commit,
            "source_uri": f"https://github.com/aiPGAB/PMAT2/archive/{self.commit}.tar.gz",
            "source_sha256": "b" * 64,
            "release_metadata_uri": "https://api.github.com/repos/aiPGAB/PMAT2/releases/latest",
        }


def test_tested_pmat_version_is_offline_and_frozen() -> None:
    client = _ReleaseClient()
    resolved = resolve_backend_version("pmat", "tested", release_client=client)
    assert resolved.version == "2.1.5"
    assert resolved.tag == "v2.1.5"
    assert resolved.commit == "04534a2adf0c5309cb2e7478fd2bcc98ced3818c"
    assert (
        resolved.source_sha256 == "5564becd242ab254e379758133b514ce0c38431df55c77bee74836f1914bf872"
    )
    assert client.calls == []


def test_latest_is_explicit_and_freezes_release_identity() -> None:
    client = _ReleaseClient()
    resolved = resolve_backend_version("pmat", "latest", release_client=client)
    assert resolved.requested_selector == "latest"
    assert resolved.version == "2.1.6"
    assert resolved.commit == "a" * 40
    assert client.calls == ["latest"]


def test_resolved_release_derives_the_managed_source_spec() -> None:
    resolved = resolve_backend_version("pmat", "latest", release_client=_ReleaseClient())
    derived = environment_spec_for_version(PMAT_ENVIRONMENT, resolved)
    platform = derived.require_platform("linux-64")
    assert platform.package == "pmat-source=2.1.6"
    assert platform.source_build is not None
    assert platform.source_build.source_uri.endswith("/" + "a" * 40 + ".tar.gz")
    assert platform.source_build.source_sha256 == "b" * 64
    assert platform.source_build.archive_root == "PMAT2-" + "a" * 40
    assert PMAT_ENVIRONMENT.require_platform("linux-64").package == "pmat-source=2.1.5"


def test_exact_tested_version_does_not_need_network() -> None:
    client = _ReleaseClient()
    resolved = resolve_backend_version("pmat", "2.1.5", release_client=client)
    assert resolved.version == "2.1.5"
    assert client.calls == []


@pytest.mark.parametrize("selector", ["1.9.0", "2.1.5-rc1", "main"])
def test_pmat_rejects_unsupported_or_moving_selectors(selector: str) -> None:
    with pytest.raises((ValueError, OrganelleDependencyError)):
        resolve_backend_version("pmat", selector)


def test_latest_rejects_prerelease_from_release_client() -> None:
    with pytest.raises(OrganelleDependencyError):
        resolve_backend_version("pmat", "latest", release_client=_ReleaseClient("v2.2.0-rc1"))


def test_other_released_backends_keep_tested_pins() -> None:
    assert resolve_backend_version("oatk", "tested").version == "1.0.0"
    assert resolve_backend_version("himt", "tested").version == "1.1.3"


def test_pmat_routing_rejects_four_part_selector() -> None:
    """PMAT routing policy keeps accepting only stable 2.x.y releases. A
    four-part selector is rejected by the routing layer even though the shared
    :class:`ResolvedBackendVersion` model now legitimately represents real
    four-part software releases (e.g. GetOrganelle 1.7.7.1)."""
    with pytest.raises((ValueError, OrganelleDependencyError)):
        resolve_backend_version("pmat", "2.1.5.0")


def test_getorganelle_tested_version_resolves_offline() -> None:
    client = _ReleaseClient()
    resolved = resolve_backend_version("getorganelle", "tested", release_client=client)
    assert resolved.backend_id == "getorganelle"
    assert resolved.version == "1.7.7.1"
    assert resolved.requested_selector == "tested"
    # No network is touched for the tested pin.
    assert client.calls == []


def test_getorganelle_exact_four_part_version_resolves_offline() -> None:
    client = _ReleaseClient()
    resolved = resolve_backend_version("getorganelle", "1.7.7.1", release_client=client)
    assert resolved.version == "1.7.7.1"
    assert resolved.requested_selector == "1.7.7.1"
    assert client.calls == []


@pytest.mark.parametrize("selector", ["latest", "1.7.7.0", "1.7.7", "2.0.0", "main"])
def test_getorganelle_unsupported_selectors_fail_closed(selector: str) -> None:
    """GetOrganelle resolves only ``tested`` and the exact ``1.7.7.1`` pin;
    every other selector fails closed without network access."""
    with pytest.raises(OrganelleDependencyError):
        resolve_backend_version("getorganelle", selector)


def test_github_release_downloads_the_resolved_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag moving after lookup cannot change the selected archive."""
    import hashlib

    from organelleverse.assembly import environment_versions as versions

    commit = "c" * 40
    uris: list[str] = []
    monkeypatch.setattr(
        versions, "_read_json", lambda uri: {"object": {"type": "commit", "sha": commit}}
    )

    def read_bytes(uri: str) -> bytes:
        uris.append(uri)
        return b"downloaded commit archive"

    monkeypatch.setattr(versions, "_read_bytes", read_bytes)
    resolved = versions.resolve_backend_version("pmat", "2.1.6")
    assert uris == [f"https://github.com/aiPGAB/PMAT2/archive/{commit}.tar.gz"]
    assert resolved.source_sha256 == hashlib.sha256(b"downloaded commit archive").hexdigest()
    source = (
        versions.environment_spec_for_version(PMAT_ENVIRONMENT, resolved).platforms[0].source_build
    )
    assert source is not None
    assert source.archive_root == f"PMAT2-{commit}"


def test_tested_source_spec_and_resolver_agree() -> None:
    resolved = resolve_backend_version("pmat", "tested")
    assert environment_spec_for_version(PMAT_ENVIRONMENT, resolved) == PMAT_ENVIRONMENT


def test_orientation_patch_is_bound_to_reviewed_commit_only() -> None:
    tested = environment_spec_for_version(
        PMAT_ENVIRONMENT, resolve_backend_version("pmat", "tested")
    )
    other = environment_spec_for_version(
        PMAT_ENVIRONMENT, resolve_backend_version("pmat", "latest", release_client=_ReleaseClient())
    )
    assert tested.platforms[0].source_build.patches
    assert not other.platforms[0].source_build.patches


def test_tested_pmat_capability_requires_orientation_build_banner() -> None:
    from organelleverse.assembly.backends.runtime import pmat_runtime
    from organelleverse.assembly.environment_capabilities import runtime_capabilities
    from organelleverse.assembly.environment_specs import PMAT_ORIENTATION_BANNER

    tested = runtime_capabilities(pmat_runtime(), {}, resolve_backend_version("pmat", "tested"))
    other = runtime_capabilities(
        pmat_runtime(),
        {},
        resolve_backend_version("pmat", "latest", release_client=_ReleaseClient()),
    )
    assert tested.items[0].required_version_text == PMAT_ORIENTATION_BANNER
    assert other.items[0].required_version_text is None
