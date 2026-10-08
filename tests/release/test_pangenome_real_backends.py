"""Non-skippable real minigraph/PGGB release gate for pangenome graphs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.pangenome import service
from organelleverse.pangenome.install import check_backend

pytestmark = pytest.mark.integration


def _fixture_genomes(tmp_path: Path) -> list[OrganelleGenome]:
    recipe_path = Path(__file__).parent / "fixtures" / "pangenome" / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    rng = random.Random(recipe["seed"])
    reference = "".join(rng.choice("ACGT") for _ in range(recipe["length"]))
    alternate = list(reference)
    for index in range(0, len(alternate), recipe["mutation_stride"]):
        alternate[index] = rng.choice([base for base in "ACGT" if base != alternate[index]])
    genomes: list[OrganelleGenome] = []
    for name, sequence in zip(recipe["samples"], (reference, "".join(alternate)), strict=True):
        fasta = tmp_path / f"{name}.fa"
        fasta.write_text(f">{name}\n{sequence}\n", encoding="utf-8")
        genomes.append(
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=name, source="release-fixture"),
            )
        )
    return genomes


def _assert_real_graph_result(result: OrganelleResult, *, backend: str) -> ArtifactRef:
    graph = service.require_graph_result(result, backend=backend)
    version = next(a for a in result.artifacts if a.kind == "software_version")
    assert version.resolve().read_text().strip() == result.provenance.software_versions[backend]
    assert "fixture" not in result.provenance.software_versions[backend]
    return graph


def _require_backend(name: str) -> None:
    location = check_backend(name, scan_envs=True)
    assert location["installed"] is True, f"required real backend is unavailable: {location}"


def test_real_minigraph_builds_and_publishes_verified_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_backend("minigraph")
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = service.build_graph(_fixture_genomes(tmp_path), method="minigraph", threads=2)

    _assert_real_graph_result(result, backend="minigraph")


def test_real_pggb_builds_and_publishes_verified_final_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_backend("pggb")
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = service.build_graph(
        _fixture_genomes(tmp_path),
        method="pggb",
        threads=2,
        n_haplotypes=2,
        segment_length=500,
        identity=90,
    )

    graph = _assert_real_graph_result(result, backend="pggb")
    assert graph.uri.endswith(".smooth.final.gfa")
    # PGGB's invoked default pipeline actually produces these files; publication
    # must retain them, without manufacturing optional VCF or MAF outputs.
    assert {artifact.format for artifact in result.artifacts} >= {"gfa", "og", "paf"}
    assert len(result.metrics["related_artifact_validation"]) >= 2
    for artifact in result.artifacts:
        current = ArtifactRef.from_path(
            artifact.resolve(), kind=artifact.kind, format=artifact.format
        )
        assert current.sha256 == artifact.sha256


@pytest.mark.parametrize(
    "result",
    [
        OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="planned only",
            flags=("graph_planned",),
        ),
        OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="claimed built without artifact",
            flags=("graph_built",),
        ),
    ],
)
def test_release_assertion_rejects_historical_pseudo_success(result: OrganelleResult) -> None:
    with pytest.raises(OrganelleInputError, match="not publishable"):
        _assert_real_graph_result(result, backend="minigraph")
