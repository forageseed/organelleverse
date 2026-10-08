from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.quality_control.environment import (
    QcEnvironment,
    QcExecutable,
    mapper_preset,
    resolve_qc_environment,
)


def _fake_tool(path: Path, version_output: str) -> Path:
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version_output}'\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    ("technology", "quality_state", "preset"),
    [
        ("illumina", None, "sr"),
        ("pacbio_hifi", "ccs", "map-hifi"),
        ("pacbio_clr", "raw", "map-pb"),
        ("pacbio_clr", "corrected", "map-pb"),
        ("ont", "raw", "map-ont"),
        ("ont", "corrected", "map-ont"),
        ("ont", "hq", "lr:hq"),
        ("ont", "duplex", "lr:hq"),
    ],
)
def test_mapper_preset_is_manifest_driven(
    technology: str, quality_state: str | None, preset: str
) -> None:
    assert mapper_preset(technology, quality_state) == preset


@pytest.mark.parametrize(
    ("technology", "quality_state"),
    [
        ("unknown", None),
        ("pacbio_hifi", "raw"),
        ("illumina", "ccs"),
        ("ont", None),
    ],
)
def test_unknown_mapper_profile_fails_closed(technology: str, quality_state: str | None) -> None:
    with pytest.raises(OrganelleInputError) as captured:
        mapper_preset(technology, quality_state)
    assert captured.value.code == "qc.unsupported_technology"


def test_installed_environment_records_exact_component_identity(tmp_path: Path) -> None:
    minimap2 = _fake_tool(tmp_path / "minimap2", "2.30-r1287")
    samtools = _fake_tool(tmp_path / "samtools", "samtools 1.20")
    environment = resolve_qc_environment(installed={"minimap2": minimap2, "samtools": samtools})

    assert environment.source == "installed"
    assert environment.minimap2.path == minimap2.resolve()
    assert environment.minimap2.version == "2.30-r1287"
    assert environment.minimap2.sha256 == hashlib.sha256(minimap2.read_bytes()).hexdigest()
    assert environment.samtools.path == samtools.resolve()
    assert environment.samtools.version == "1.20"
    assert environment.meryl is None
    assert environment.environment_id.startswith("qc-environment:sha256:")


def test_installed_environment_precedes_managed_fallback(tmp_path: Path) -> None:
    minimap2 = _fake_tool(tmp_path / "minimap2", "2.30-r1287")
    samtools = _fake_tool(tmp_path / "samtools", "samtools 1.20")
    managed = QcEnvironment(
        environment_id="managed:test",
        source="managed",
        minimap2=QcExecutable(
            name="minimap2",
            path=Path("/managed/minimap2"),
            version="managed",
            sha256="a" * 64,
            source="managed",
        ),
        samtools=QcExecutable(
            name="samtools",
            path=Path("/managed/samtools"),
            version="managed",
            sha256="b" * 64,
            source="managed",
        ),
    )
    environment = resolve_qc_environment(
        installed={"minimap2": minimap2, "samtools": samtools},
        managed_resolver=lambda: managed,
    )

    assert environment.source == "installed"
    assert environment.environment_id != "managed:test"


def test_missing_required_tools_fails_with_typed_dependency_error(tmp_path: Path) -> None:
    with pytest.raises(OrganelleDependencyError) as captured:
        resolve_qc_environment(installed={}, registry_root=tmp_path)
    assert captured.value.code == "qc.dependency_unavailable"
