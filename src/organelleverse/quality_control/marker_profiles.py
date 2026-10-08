"""Resolve the release-pinned plant-organelle marker profiles."""

from __future__ import annotations

from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT, OATKDB_VERSION
from organelleverse.assembly.environments import EnvironmentManager
from organelleverse.core.errors import OrganelleDependencyError

from .contracts import MarkerProfileSet


def resolve_marker_profiles(
    *,
    manager: EnvironmentManager | None = None,
) -> tuple[MarkerProfileSet, ...]:
    """Reuse or materialize the exact OatkDB embryophyta profile pair."""

    resolved = manager or EnvironmentManager()
    profiles: list[MarkerProfileSet] = []
    try:
        for target in ("mitochondrion", "plastid"):
            prepared = resolved.prepare_profile(
                OATK_ENVIRONMENT,
                organelle=target,
                policy="ensure",
            )
            profiles.append(
                MarkerProfileSet(
                    profile_set_id="oatkdb-embryophyta",
                    version=OATKDB_VERSION,
                    hmm_artifact=prepared.artifact,
                    sha256=prepared.artifact.sha256,
                    target=target,
                )
            )
    except OSError as error:
        raise OrganelleDependencyError(
            code="qc.marker_profiles_unavailable",
            message=f"plant-organelle marker profiles are unavailable: {error}",
            details={"profile_set_id": "oatkdb-embryophyta", "version": OATKDB_VERSION},
        ) from error
    return tuple(profiles)


__all__ = ["resolve_marker_profiles"]
