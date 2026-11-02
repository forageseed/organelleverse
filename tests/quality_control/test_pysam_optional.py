"""Rust BAM evidence remains an optional dependency on every platform.

Without rubam: the package imports and rubam-free operations work; the
BAM-backed QC paths fail closed with a machine-readable reason and install
guidance — never a degraded result.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture()
def no_rubam(monkeypatch: pytest.MonkeyPatch):
    """Simulate an environment without rubam (e.g. Windows)."""
    monkeypatch.setitem(sys.modules, "rubam", None)


def test_package_imports_without_rubam(no_rubam: None) -> None:
    import importlib

    import organelleverse.quality_control.evidence as evidence
    import organelleverse.quality_control.graph as graph

    importlib.reload(evidence)
    importlib.reload(graph)


def test_qc_graph_evidence_fails_closed_without_rubam(no_rubam: None) -> None:
    from organelleverse.core.errors import OrganelleDependencyError
    from organelleverse.quality_control.graph import collect_graph_evidence

    with pytest.raises(OrganelleDependencyError) as raised:
        collect_graph_evidence(None, None, None)  # type: ignore[arg-type]  # gate fires first
    assert raised.value.code == "qc.rubam_missing"
    assert "rubam" in raised.value.message
    assert raised.value.details["extra"] == "qc"


def test_qc_mapping_evidence_fails_closed_without_rubam(no_rubam: None) -> None:
    from organelleverse.core.errors import OrganelleDependencyError
    from organelleverse.quality_control.evidence import collect_mapping_evidence

    with pytest.raises(OrganelleDependencyError) as raised:
        collect_mapping_evidence(None, None, workspace=".", threads=1, timeout_seconds=1)  # type: ignore[arg-type]
    assert raised.value.code == "qc.rubam_missing"


def test_core_dependencies_do_not_include_rubam() -> None:
    import tomllib

    from tests._paths import PROJECT_ROOT

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    core = pyproject["project"]["dependencies"]
    assert all("rubam" not in requirement for requirement in core)
    assert "rubam==0.3.13" in pyproject["project"]["optional-dependencies"]["qc"]
