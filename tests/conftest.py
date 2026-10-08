"""Shared test configuration for the OrganelleVerse suite.

Every managed-run operation publishes beneath ``ORGANELLEVERSE_CACHE_ROOT``.
Pinning that root to a per-test temporary directory keeps the suite hermetic:
no test writes into the real user cache or drops a ``.organelleverse`` tree in
the working directory, and independent calls inside one test reuse the same
managed run.
"""

from __future__ import annotations

import os
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
