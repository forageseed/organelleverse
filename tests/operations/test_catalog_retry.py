"""A failed first catalog load must leave the default registry retryable."""

from __future__ import annotations

import pytest

from organelleverse.operations.registry import _ReleaseOperationRegistry


def test_failed_capability_attach_rolls_back_and_retry_reports_the_real_error(monkeypatch):
    registry = _ReleaseOperationRegistry()
    calls = {"n": 0}

    def flaky_attach(self):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise MemoryError("simulated while hashing capability bundles")

    monkeypatch.setattr(_ReleaseOperationRegistry, "_attach_default_capability_source", flaky_attach)

    with pytest.raises(MemoryError):
        registry._ensure_catalog()
    assert not registry._bindings
    assert not registry._authoritative_operation_ids
    # the retry sees the same real error, not "duplicate operation ID"
    with pytest.raises(MemoryError):
        registry._ensure_catalog()
    registry._ensure_catalog()
    assert registry._catalog_loaded
    assert "annotation.annotate" in registry._bindings
