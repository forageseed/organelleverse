"""Import-boundary and registration tests (spec §6/§14/§15).

The hardest safety guarantees: bare ``import organelleverse`` must not import
``mcp``, must not import the WSL provider runtime, and must not spawn any
process. Discovery reads entry-point metadata only and never loads the factory.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import organelleverse  # noqa: F401  — the bare import under test
from organelleverse.compute import discover_compute_providers
from organelleverse.compute.providers import wsl as wsl_pkg

# --- a fake entry point that mirrors the registered WSL locator ------------


class _FakeDist:
    name = "organelleverse"
    version = "0.0.1"

    @property
    def files(self):
        # a representative installed-file list so the RECORD digest is stable
        return [PurePosixPath(p) for p in ("organelleverse/__init__.py", "organelleverse/compute/providers/wsl/__init__.py")]

    def read_text(self, filename):
        # no RECORD metadata on this fake; discovery falls back to the path set
        return None


class _FakeEntryPoint:
    name = "wsl"
    value = "organelleverse.compute.providers.wsl:provider_factory"
    dist = _FakeDist()

    def load(self):  # pragma: no cover - discovery must never call this
        raise AssertionError("discovery must never call EntryPoint.load()")


def test_bare_import_does_not_import_mcp():
    # The mcp SDK must be lazily imported inside the execution path, never at
    # module top level. A runtime sys.modules check is unreliable across tests
    # in the same process, so verify the source keeps mcp out of module scope.
    import pathlib

    root = pathlib.Path(__file__).resolve()
    for _ in range(5):
        root = root.parent
        if (root / "pyproject.toml").exists():
            break
    for rel in (
        "src/organelleverse/compute/protocol.py",
        "src/organelleverse/compute/providers/wsl/__init__.py",
        "src/organelleverse/compute/__init__.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for line in text.splitlines():
            # a module-scope import starts at column 0; an indented import is
            # inside a function (lazy) and is allowed.
            if line.startswith(("import mcp", "from mcp")):
                raise AssertionError(f"{rel} imports mcp at module scope:\n  {line}")


def test_bare_import_does_not_import_wsl_provider_runtime():
    # the wsl worker module must not be imported by the package __init__
    import sys

    for forbidden in ("organelleverse.compute.providers.wsl.worker",):
        assert forbidden not in sys.modules, f"bare import pulled {forbidden}"


def test_candidate_discovery_reads_entry_point_without_loading_factory():
    # Inject the fake entry point; its load() raises, so if discovery loaded it
    # this test would fail. No real install is required.
    candidates = discover_compute_providers(entry_points=[_FakeEntryPoint()])
    wsl_candidates = [c for c in candidates if c.provider_id == "wsl"]
    assert len(wsl_candidates) == 1
    c = wsl_candidates[0]
    assert c.entry_point_locator == "organelleverse.compute.providers.wsl:provider_factory"
    assert c.distribution_record_digest.startswith("sha256:")
    assert c.distribution_name == "organelleverse"


def test_pyproject_declares_the_wsl_entry_point():
    # Static check that the entry point is declared in this worktree's pyproject,
    # independent of which distribution is currently installed.
    import pathlib
    root = pathlib.Path(__file__).resolve()
    # walk up to the repo root (tests/compute/providers/wsl -> root)
    for _ in range(5):
        root = root.parent
        if (root / "pyproject.toml").exists():
            break
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'organelleverse.compute_providers' in text
    assert 'wsl = "organelleverse.compute.providers.wsl:provider_factory"' in text


def test_provider_package_exposes_declaration_only():
    # __init__ exposes the declaration + factory id, not a constructed provider
    assert wsl_pkg.PROVIDER_ID == "wsl"
    assert wsl_pkg.provider_factory().provider_id == "wsl"
