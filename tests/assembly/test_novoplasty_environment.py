"""Declared NOVOPlasty banner exit and managed executable-to-role mapping."""

import json
from pathlib import Path

import pytest

from organelleverse.assembly.backends.runtime import RUNTIMES
from organelleverse.assembly.environment_capabilities import runtime_capabilities
from organelleverse.assembly.environment_specs import NOVOPLASTY_ENVIRONMENT
from organelleverse.assembly.environments import EnvironmentManager
from organelleverse.core.errors import OrganelleDependencyError


class Carrier:
    def __init__(self, code=2, banner="Version 4.3.5"):
        self.code = code
        self.banner = banner

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        (destination / "bin").mkdir(parents=True)
        for name in ("NOVOPlasty4.3.5.pl", "perl"):
            path = destination / "bin" / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        (destination / "conda-meta").mkdir()
        (destination / "conda-meta" / "novoplasty.json").write_text(
            json.dumps({"name": "novoplasty", "version": "4.3.5", "build": "test"})
        )

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        assert argv == ("-c", "")
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="empty configuration",
            details={"returncode": self.code, "stdout_tail": self.banner},
        )

    def probe_package_identity(self, prefix: Path):
        return (("novoplasty", "4.3.5", "test"),)


def test_declared_nonzero_version_banner_and_provider_role(tmp_path):
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", tool_root=tmp_path / "tools", carrier=Carrier()
    )
    env = manager.prepare(NOVOPLASTY_ENVIRONMENT, policy="ensure", platform="linux-64")
    assert env.version == "Version 4.3.5"
    contract = runtime_capabilities(RUNTIMES["novoplasty"], {})
    assert contract.items[0].version_exit_codes == (2,)
    assert contract.items[0].role == "novoplasty"
    registered = manager.register_prepared_provider(
        env, contract, requested_source="managed", resolved_version="4.3.5"
    )
    assert registered.version == "4.3.5"
    assert (
        manager.prepare(NOVOPLASTY_ENVIRONMENT, policy="require", platform="linux-64").prefix
        == env.prefix
    )


@pytest.mark.parametrize("code,banner", [(1, "Version 4.3.5"), (255, "Version 4.3.5"), (2, "")])
def test_wrong_exit_or_empty_banner_fails_closed(tmp_path, code, banner):
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", tool_root=tmp_path / "tools", carrier=Carrier(code, banner)
    )
    with pytest.raises(OrganelleDependencyError):
        manager.prepare(NOVOPLASTY_ENVIRONMENT, policy="ensure", platform="linux-64")
