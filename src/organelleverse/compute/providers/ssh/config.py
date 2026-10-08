"""SSH provider configuration types (spec §12).

Owner-configured, closed identities. The connection target is a host *alias*
resolved through the owner's SSH config plus a dedicated least-privilege
account and an owner-held ``known_hosts`` file; no IP, port, credential, key,
or agent value lives here. ``workspace_id`` selects one administrator-created
workspace through a provider-owned mapping — it is never a filesystem path.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

__all__ = ["SshTargetConfig"]

_HOST_ALIAS = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$"
_ACCOUNT = r"^[a-z_][a-z0-9_-]{0,31}$"
_WORKSPACE = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class SshTargetConfig(StrictSpecModel):
    """An owner-configured SSH target: a host alias, a least-privilege account,
    an owner-held known-hosts file, and a workspace id.

    The model carries no private key, passphrase, agent socket, token, port, or
    IP — those stay inside the transport credential boundary (spec §12, §14).
    """

    target_id: str = Field(pattern=r"^ssh:[a-z0-9][a-z0-9._-]{0,127}$")
    host_alias: str = Field(pattern=_HOST_ALIAS)
    account: str = Field(pattern=_ACCOUNT)
    known_hosts_file: Path
    workspace_id: str = Field(pattern=_WORKSPACE)

    @model_validator(mode="after")
    def _target_matches_host_alias(self) -> SshTargetConfig:
        if self.target_id != f"ssh:{self.host_alias}":
            raise ValueError("SSH target_id must be derived from host_alias")
        return self

    @model_validator(mode="after")
    def _known_hosts_is_owner_held_absolute(self) -> SshTargetConfig:
        # known_hosts must be absolute, owner-configured, and free of symlink
        # components so a model/path cannot redirect host-key verification.
        path = self.known_hosts_file
        if not path.is_absolute():
            raise ValueError("known_hosts_file must be an absolute owner-held path")
        # reject any symlink component in the path (not just a trailing symlink)
        if any(part != path.name and Path(part).is_symlink() for part in path.parents):
            raise ValueError("known_hosts_file path must contain no symlink components")
        return self


def require_known_hosts_exists(config: SshTargetConfig) -> None:
    """Fail closed if the owner-held known-hosts file is missing or not a regular
    file. Called on the explicit discovery/probe path only, never on import."""
    path = config.known_hosts_file
    if not path.is_file() or path.is_symlink():
        raise OrganelleContractError(
            code="compute.ssh_known_hosts_missing",
            message=(
                "the owner-held known_hosts file is missing or is not a regular "
                "file; SSH host-key verification cannot proceed"
            ),
            details={"target_id": config.target_id},
        )
