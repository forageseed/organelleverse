from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest

from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT, OATKDB_VERSION
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedProfile,
)
from organelleverse.assembly.manifests import AssemblyComponentIdentity
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.quality_control.marker_profiles import resolve_marker_profiles


class _RecordingManager:
    def __init__(self, root: Path) -> None:
        self.calls: list[tuple[object, str, str]] = []
        self._profiles: dict[str, PreparedProfile] = {}
        for target in ("mitochondrion", "plastid"):
            path = root / f"{target}.fam"
            path.write_text(f"HMMER3/f\\nNAME {target}\\n")
            artifact = ArtifactRef.from_path(
                path,
                kind="hmm_profile",
                format="fam",
                media_type="text/plain",
            )
            self._profiles[target] = PreparedProfile(
                source="managed",
                target=cast(Literal["mitochondrion", "plastid"], target),
                path=path,
                artifact=artifact,
                component=AssemblyComponentIdentity(
                    category="profile",
                    name=f"oatkdb-{target}",
                    version=OATKDB_VERSION,
                    sha256=artifact.sha256,
                ),
            )

    def prepare_profile(
        self,
        spec: object,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        policy: Literal["ensure", "require"],
    ) -> PreparedProfile:
        self.calls.append((spec, organelle, policy))
        return self._profiles[organelle]


class _OfflineManager:
    def prepare_profile(self, *args: object, **kwargs: object) -> PreparedProfile:
        raise OSError("network unreachable")


def test_resolver_reuses_locked_oatkdb_profiles_for_both_targets(tmp_path: Path) -> None:
    manager = _RecordingManager(tmp_path)

    profiles = resolve_marker_profiles(manager=cast(EnvironmentManager, manager))

    assert manager.calls == [
        (OATK_ENVIRONMENT, "mitochondrion", "ensure"),
        (OATK_ENVIRONMENT, "plastid", "ensure"),
    ]
    assert tuple(item.profile_set_id for item in profiles) == (
        "oatkdb-embryophyta",
        "oatkdb-embryophyta",
    )
    assert tuple(item.version for item in profiles) == (OATKDB_VERSION, OATKDB_VERSION)
    assert tuple(item.target for item in profiles) == ("mitochondrion", "plastid")
    assert all(item.sha256 == item.hmm_artifact.sha256 for item in profiles)


def test_resolver_wraps_offline_download_as_dependency_unavailable() -> None:
    with pytest.raises(OrganelleDependencyError) as raised:
        resolve_marker_profiles(manager=cast(EnvironmentManager, _OfflineManager()))

    assert raised.value.code == "qc.marker_profiles_unavailable"
    assert "network unreachable" in raised.value.message
