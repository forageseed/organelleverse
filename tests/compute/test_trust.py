from __future__ import annotations

import pytest

from organelleverse.compute.contracts import (
    ComputeProviderSpec,
    EnabledProvider,
    ProviderCandidate,
    TransportKind,
)
from organelleverse.compute.trust import (
    ComputeProviderTrustStore,
    canonical_declaration_digest,
)
from organelleverse.core.errors import OrganellePermissionError


def _candidate(provider_id="wsl", **overrides):
    base = dict(
        provider_id=provider_id,
        distribution_name="acme-wsl",
        distribution_version="1.0.0",
        entry_point_locator="acme_wsl:factory",
        distribution_record_digest="sha256:" + "a" * 64,
    )
    base.update(overrides)
    return ProviderCandidate(**base)


def _declaration(provider_id="wsl", **overrides):
    base = dict(
        provider_id=provider_id,
        display_name="WSL",
        declaration_version="1.0",
        transport_kind=TransportKind.WSL_MCP,
        supported_platforms=("linux-64",),
        supports_prepare=True,
        supports_cancel=True,
        artifact_transports=("wsl-cache",),
    )
    base.update(overrides)
    return ComputeProviderSpec(**base)


class _RecordingLoader:
    def __init__(self, declaration=None):
        self.calls = 0
        self._declaration = declaration

    def __call__(self, candidate):
        self.calls += 1
        return self._declaration or _declaration(candidate.provider_id)


def test_enable_loads_factory_exactly_once_and_persists_trust(tmp_path):
    loader = _RecordingLoader()
    store = ComputeProviderTrustStore(tmp_path / "t.json", factory_loader=loader)
    enabled = store.enable(_candidate())
    assert loader.calls == 1
    assert isinstance(enabled, EnabledProvider)
    assert enabled.declaration.provider_id == "wsl"
    assert enabled.declaration_digest == canonical_declaration_digest(enabled.declaration)
    assert store.trusted_provider_ids() == ("wsl",)
    assert store.is_current(_candidate())


def test_untrusted_provider_is_not_current(tmp_path):
    store = ComputeProviderTrustStore(tmp_path / "t.json", factory_loader=_RecordingLoader())
    assert not store.is_current(_candidate())
    with pytest.raises(OrganellePermissionError) as exc:
        store.require_trusted(_candidate())
    assert exc.value.code == "compute.provider_untrusted"


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("distribution_version", "1.0.1"),
        ("entry_point_locator", "acme_wsl:new_factory"),
        ("distribution_record_digest", "sha256:" + "b" * 64),
    ],
)
def test_static_identity_change_invalidates_trust(tmp_path, field, new_value):
    store = ComputeProviderTrustStore(tmp_path / "t.json", factory_loader=_RecordingLoader())
    store.enable(_candidate())
    stale = _candidate(**{field: new_value})
    assert not store.is_current(stale)


def test_declaration_digest_change_is_stale(tmp_path):
    loader = _RecordingLoader(_declaration())
    store = ComputeProviderTrustStore(tmp_path / "t.json", factory_loader=loader)
    store.enable(_candidate())
    # the installed package now ships a different declaration
    loader._declaration = _declaration(artifact_transports=("wsl-cache", "sftp"))
    fresh_digest = canonical_declaration_digest(loader._declaration)
    with pytest.raises(OrganellePermissionError) as exc:
        store.require_declaration_current(_candidate(), fresh_digest)
    assert exc.value.code == "compute.provider_trust_stale"


def test_forget_removes_trust(tmp_path):
    store = ComputeProviderTrustStore(tmp_path / "t.json", factory_loader=_RecordingLoader())
    store.enable(_candidate())
    assert store.forget("wsl") is True
    assert store.trusted_provider_ids() == ()
    assert store.forget("wsl") is False


def test_trust_persists_across_store_instances(tmp_path):
    path = tmp_path / "t.json"
    ComputeProviderTrustStore(path, factory_loader=_RecordingLoader()).enable(_candidate())
    reopened = ComputeProviderTrustStore(path, factory_loader=_RecordingLoader())
    assert reopened.is_current(_candidate())
    assert reopened.trusted_provider_ids() == ("wsl",)
