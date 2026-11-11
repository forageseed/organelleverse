"""Shared test configuration for the OrganelleVerse suite.

Every managed-run operation publishes beneath ``ORGANELLEVERSE_CACHE_ROOT``.
Pinning that root to a per-test temporary directory keeps the suite hermetic:
no test writes into the real user cache or drops a ``.organelleverse`` tree in
the working directory, and independent calls inside one test reuse the same
managed run.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

_REAL_RELEASE_MARKERS = (
    "release_annotation",
    "release_assembly_oatk",
    "release_assembly_himt",
    "release_assembly_getorganelle",
    "release_assembly_pmat",
    "release_assembly_qc",
)


def _host_is_posix() -> bool:
    return os.name == "posix"


def _host_can_symlink() -> bool:
    # Windows refuses os.symlink without the "create symbolic links" privilege
    # (WinError 1314), so the answer has to be probed rather than read from the
    # platform name.
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "target"
        target.write_bytes(b"")
        try:
            os.symlink(target, Path(scratch) / "link")
        except (OSError, NotImplementedError):
            return False
    return True


def _module_installed(name: str):
    return lambda: importlib.util.find_spec(name) is not None


# Marker name -> (check that the precondition holds, reason shown when it does not).
# A marked test is skipped instead of failing when its host cannot meet the
# precondition; on a host that can, it runs unchanged.
_HOST_PRECONDITIONS = {
    "requires_posix": (
        _host_is_posix,
        "needs a POSIX host: the test relies on POSIX paths, file modes or Linux-only tools",
    ),
    "requires_symlink": (
        _host_can_symlink,
        "this account cannot create symbolic links",
    ),
    "requires_mappy": (_module_installed("mappy"), "mappy is not installed"),
    "requires_edlib": (_module_installed("edlib"), "edlib is not installed"),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    met: dict[str, bool] = {}
    for item in items:
        for marker, (check, reason) in _HOST_PRECONDITIONS.items():
            if item.get_closest_marker(marker) is None:
                continue
            if marker not in met:
                met[marker] = bool(check())
            if not met[marker]:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session", autouse=True)
def _verified_release_bundles(tmp_path_factory: pytest.TempPathFactory):
    """Verify the 16 bundle-served release operations once per session.

    The default registry serves the release surface from core bundles, and
    bundles become visible only with a content-addressed verification record.
    Records are generated once into an isolated ORGANELLEVERSE_HOME so the
    session reuses them - the owner-approved "verify once" model.
    """

    from tests.capabilities.release_bundles import verify_release_bundles

    home = tmp_path_factory.mktemp("organelleverse-home")
    previous = os.environ.get("ORGANELLEVERSE_HOME")
    os.environ["ORGANELLEVERSE_HOME"] = str(home)
    try:
        verify_release_bundles(home)
        yield
    finally:
        if previous is None:
            os.environ.pop("ORGANELLEVERSE_HOME", None)
        else:
            os.environ["ORGANELLEVERSE_HOME"] = previous


@pytest.fixture(autouse=True)
def _isolated_managed_cache_root(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real release gates deliberately reuse host/managed installations and
    # downloaded databases across cases. Giving each gate a fresh cache forces
    # every backend environment to be rebuilt and turns a scientific smoke test
    # into repeated installation work.
    if any(request.node.get_closest_marker(marker) for marker in _REAL_RELEASE_MARKERS):
        return
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
