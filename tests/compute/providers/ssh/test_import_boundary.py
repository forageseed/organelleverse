"""Import-boundary and registration tests for the SSH provider (spec §6/§14/§15).

Bare ``import organelleverse`` must not import ``mcp``, must not import the SSH
provider runtime, must not open an SSH connection, and must not read credentials.
"""

from __future__ import annotations

import pathlib

import organelleverse  # noqa: F401
from organelleverse.compute.providers import ssh as ssh_pkg


def test_bare_import_does_not_import_mcp_or_ssh_runtime():
    # The mcp SDK and SSH runtime modules must be lazily imported inside the
    # execution path, never at module scope. Verify the source keeps them out
    # of module-scope imports (a runtime sys.modules check is unreliable across
    # tests in the same process).
    root = pathlib.Path(__file__).resolve()
    for _ in range(6):
        root = root.parent
        if (root / "pyproject.toml").exists():
            break
    for rel in (
        "src/organelleverse/compute/providers/ssh/__init__.py",
        "src/organelleverse/compute/providers/ssh/launcher.py",
        "src/organelleverse/compute/providers/ssh/config.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith(("import mcp", "from mcp", "import paramiko", "from paramiko")):
                raise AssertionError(f"{rel} imports a transport dep at module scope:\n  {line}")


def test_pyproject_declares_the_ssh_entry_point():
    root = pathlib.Path(__file__).resolve()
    for _ in range(6):
        root = root.parent
        if (root / "pyproject.toml").exists():
            break
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ssh = "organelleverse.compute.providers.ssh:provider_factory"' in text


def test_provider_package_exposes_declaration_only():
    assert ssh_pkg.PROVIDER_ID == "ssh"
    assert ssh_pkg.provider_factory().provider_id == "ssh"
