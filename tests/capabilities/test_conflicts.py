from __future__ import annotations

from organelleverse.capabilities.index import CapabilityConflict, CapabilityConflictMember


def test_conflict_models_are_frozen_and_sort_members_by_origin() -> None:
    first = CapabilityConflictMember(
        content_hash="sha256:" + "1" * 64,
        origin_channel="local",
        source_path="/z/capability.toml",
    )
    second = CapabilityConflictMember(
        content_hash="sha256:" + "2" * 64,
        origin_channel="core",
        source_path="/a/capability.toml",
    )
    conflict = CapabilityConflict(capability_id="demo.conflict", members=(first, second))

    assert conflict.members == (second, first)
