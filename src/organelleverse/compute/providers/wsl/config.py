"""WSL provider configuration types (spec §11).

Owner-configured, closed identities. The ``target_id`` is derived from the
distribution so a caller cannot name one target while selecting another. These
types carry no credentials, paths, commands, or grants (spec §5.3/§14).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from organelleverse.operations.spec import StrictSpecModel

__all__ = ["WslDiscoveredTarget", "WslTargetConfig"]


# A conservative distro-name shape: WSL distro names are alphanumeric with a few
# separators; this also rejects blank, control-character, and option-shaped
# ("-something") names that could not be a real `wsl.exe --distribution` value.
_DISTRO_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$"


class WslTargetConfig(StrictSpecModel):
    """An owner-configured WSL target: a distro and the required WSL version.

    ``target_id`` must equal ``wsl:<distribution>`` so the public id cannot
    diverge from the distro the provider actually launches.
    """

    target_id: str = Field(pattern=r"^wsl:[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
    distribution: str = Field(pattern=_DISTRO_PATTERN)
    expected_version: Literal[2] = 2

    @model_validator(mode="after")
    def _target_matches_distribution(self) -> WslTargetConfig:
        if self.target_id != f"wsl:{self.distribution}":
            raise ValueError("WSL target_id must be derived from distribution")
        return self


class WslDiscoveredTarget(StrictSpecModel):
    """A distro found by explicit ``wsl.exe --list --quiet`` (spec §5.1).

    Discovery returns names only; the subsequent protocol probe (Task 2)
    verifies WSL 2 and Linux identity. No environment is changed by discovery.
    """

    target_id: str = Field(pattern=r"^wsl:[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
    distribution: str = Field(pattern=_DISTRO_PATTERN)
