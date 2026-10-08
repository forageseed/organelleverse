"""The PMAT autoMito -> graphBuild continuation bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly.api import pmat_continue
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult

_RUN = (
    Path.home()
    / ".cache/organelleverse/runs/assembly.assemble"
    / "sha256-64c48a6793c8268e624aa2f33c3a11a86dde9083a3644176342cf862671fd86b"
    / "normalized"
)


def _artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        kind="file",
        uri=str(_RUN / name),
        format=name.rsplit(".", 1)[-1],
        sha256="0" * 64,
        size_bytes=1,
        validated=False,
    )


def _ok_result() -> OrganelleResult:
    return OrganelleResult(
        operation_id="assembly.assemble",
        operation_version="1.0",
        scope="mitochondrion",
        status="ok",
        summary_text="pmat autoMito ok",
        artifacts=(
            _artifact("assembly.fasta"),
            _artifact("pmat.subsample.fasta"),
            _artifact("pmat.all_contigs.fasta"),
            _artifact("pmat.contig_graph.txt"),
            _artifact("pmat_subsample.manifest.json"),
            _artifact("pmat_assembly_result.manifest.json"),
        ),
    )


def test_bridge_requires_a_successful_result() -> None:
    from organelleverse.core.result import ErrorDetail

    failed = _ok_result().model_copy(
        update={
            "status": "failed",
            "errors": (
                ErrorDetail(code="assembly.execution_failed", message="backend failed"),
            ),
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        pmat_continue(failed)
    assert raised.value.code == "assembly.invalid_continuation"
    assert raised.value.details == {"status": "failed"}


def test_bridge_rejects_results_without_pmat_artifacts() -> None:
    oatk_only = _ok_result().model_copy(
        update={"artifacts": (_artifact("assembly.fasta"), _artifact("assembly.gfa"))}
    )
    with pytest.raises(OrganelleInputError) as raised:
        pmat_continue(oatk_only)
    assert raised.value.code == "assembly.invalid_continuation"
    assert set(raised.value.details["missing_roles"]) == {
        "subsample_manifest",
        "assembly_result_manifest",
        "pmat_subsample",
        "pmat_all_contigs",
        "pmat_contig_graph",
    }


def test_bridge_infers_organelle_from_scope() -> None:
    result = _ok_result()
    assert result.scope == "mitochondrion"
    # The full graphBuild execution is exercised on real data in
    # docs/reports; here we only pin the payload assembly by monkeypatching
    # the API entry point this bridge delegates to.
    import organelleverse.assembly.api as api

    captured: dict[str, object] = {}

    def _capture(data, **kwargs):  # type: ignore[no-untyped-def]
        captured["data"] = data
        captured["kwargs"] = kwargs
        return "sentinel"  # type: ignore[no-any-return]

    original = api.pmat_graph_build
    api.pmat_graph_build = _capture  # type: ignore[assignment]
    try:
        outcome = pmat_continue(result, threads=16)
    finally:
        api.pmat_graph_build = original  # type: ignore[assignment]
    assert outcome == "sentinel"
    data = captured["data"]
    kwargs = captured["kwargs"]
    assert data.modality == "pmat_graph_input"
    assert data.payload["subsample_manifest_artifact"] == "subsample_manifest"
    assert data.payload["assembly_result_manifest_artifact"] == "assembly_result_manifest"
    assert list(data.payload["file_artifact_roles"]) == [
        "pmat_subsample",
        "pmat_all_contigs",
        "pmat_contig_graph",
    ]
    assert set(data.artifacts) == {
        "subsample_manifest",
        "assembly_result_manifest",
        "pmat_subsample",
        "pmat_all_contigs",
        "pmat_contig_graph",
    }
    assert data.artifacts["pmat_subsample"].uri.endswith("pmat.subsample.fasta")
    assert kwargs["organelle"] == "mitochondrion"
    assert kwargs["threads"] == 16


def test_bridge_accepts_explicit_organelle() -> None:
    import organelleverse.assembly.api as api

    captured: dict[str, object] = {}

    def _capture(data, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return "sentinel"  # type: ignore[no-any-return]

    original = api.pmat_graph_build
    api.pmat_graph_build = _capture  # type: ignore[assignment]
    try:
        pmat_continue(_ok_result(), organelle="plastid")
    finally:
        api.pmat_graph_build = original  # type: ignore[assignment]
    assert captured["organelle"] == "plastid"
