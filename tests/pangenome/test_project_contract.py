"""AC-001/AC-002: PangenomeProject validates identity and stages PanSN FASTA."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse._bio import read_fasta
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.pangenome._contract import make_provenance
from organelleverse.pangenome.project import PangenomeProject


def _sequence(sha256: str) -> ArtifactRef:
    return ArtifactRef(
        kind="sequence",
        uri=f"{sha256[:8]}.fa",
        format="fasta",
        sha256=sha256,
        size_bytes=8,
    )


def _genome(*, accession: str = "", species: str = "", sha256: str = "a" * 64) -> OrganelleGenome:
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=_sequence(sha256),
        metadata=OrganelleMetadata(accession=accession, species=species),
    )


def _successful_graph_result(graph: Path, *, backend: str = "minigraph") -> OrganelleResult:
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version="1.0",
        scope="mitochondrion",
        status="ok",
        summary_text="synthetic graph",
        flags=("graph_built",),
        metrics={"output_path": str(graph)},
        artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"),),
        provenance=make_provenance(
            operation_id="pangenome.build_graph",
            parameters={},
            requested_backend=backend,
            actual_backend=backend,
            attempted_backends=(backend,),
        ),
    )


def test_two_valid_genomes_produce_ordered_unique_samples() -> None:
    first = _genome(accession="NC_001284", sha256="a" * 64)
    second = _genome(species="Example species", sha256="b" * 64)

    project = PangenomeProject.from_genomes([first, second])

    assert [sample.name for sample in project.samples] == ["NC_001284", "Example_species"]
    assert len({sample.pansn_path for sample in project.samples}) == 2

    reversed_project = PangenomeProject.from_genomes([second, first])
    assert [sample.name for sample in reversed_project.samples] == [
        "Example_species",
        "NC_001284",
    ]


def test_fewer_than_two_genomes_fail() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        PangenomeProject.from_genomes([_genome(accession="NC_001284")])
    with pytest.raises(ValidationError, match="at least 2"):
        PangenomeProject.from_genomes([])


def test_genome_without_sequence_fails() -> None:
    annotation_only = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef(
            kind="annotation",
            uri="genes.gff",
            format="gff3",
            sha256="c" * 64,
            size_bytes=16,
        ),
        metadata=OrganelleMetadata(accession="NC_001284"),
    )
    with pytest.raises(ValidationError, match="sequence"):
        PangenomeProject.from_genomes(
            [annotation_only, _genome(accession="NC_001285", sha256="b" * 64)]
        )


def test_duplicate_identity_fails() -> None:
    first = _genome(accession="NC_001284", sha256="a" * 64)
    second = _genome(accession="NC_001284", sha256="b" * 64)
    with pytest.raises(ValidationError, match="unique"):
        PangenomeProject.from_genomes([first, second])


def test_pansn_paths_are_stable_unique_and_well_formed() -> None:
    genomes = [
        _genome(accession="NC_001284", sha256="a" * 64),
        _genome(accession="NC_001285", sha256="b" * 64),
    ]

    project = PangenomeProject.from_genomes(genomes)
    rebuilt = PangenomeProject.from_genomes(genomes)

    paths = [sample.pansn_path for sample in project.samples]
    assert paths == [sample.pansn_path for sample in rebuilt.samples]
    assert len(set(paths)) == len(paths)
    for sample, path in zip(project.samples, paths, strict=True):
        segments = path.split("#")
        assert len(segments) == 3
        assert all(segments)
        assert segments[0] == sample.name


def test_materialize_stages_pansn_fasta_and_manifest(tmp_path: Path) -> None:
    first_path = tmp_path / "first.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second.fa"
    second_path.write_text(">molecule_a\nACGT\n>molecule_b\nTTTT\n", encoding="utf-8")
    first_ref = ArtifactRef.from_path(first_path, kind="sequence", format="fasta")
    second_ref = ArtifactRef.from_path(second_path, kind="sequence", format="fasta")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=first_ref,
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=second_ref,
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]
    project = PangenomeProject.from_genomes(genomes)
    staging_dir = tmp_path / "staging"

    manifest = project.materialize(staging_dir)

    fasta_records = read_fasta(Path(manifest.fasta_path))
    assert [header for header, _ in fasta_records] == [
        "NC_001284#1#1",
        "NC_001285#1#1",
        "NC_001285#1#2",
    ]
    assert [sequence for _, sequence in fasta_records] == ["ACGTACGT", "ACGT", "TTTT"]

    manifest_path = Path(manifest.manifest_path)
    assert manifest_path.parent == staging_dir
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["fasta_path"] == manifest.fasta_path
    assert [entry["sample"] for entry in payload["samples"]] == ["NC_001284", "NC_001285"]
    assert [entry["headers"] for entry in payload["samples"]] == [
        ["NC_001284#1#1"],
        ["NC_001285#1#1", "NC_001285#1#2"],
    ]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    for ref in (first_ref, second_ref):
        assert manifest_text.count(ref.sha256) == 1


def test_materialize_rejects_source_changed_after_artifact_capture(tmp_path: Path) -> None:
    first_path = tmp_path / "first-mutated.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second-stable.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    first_ref = ArtifactRef.from_path(first_path, kind="sequence", format="fasta")
    second_ref = ArtifactRef.from_path(second_path, kind="sequence", format="fasta")
    project = PangenomeProject.from_genomes(
        [
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=first_ref,
                metadata=OrganelleMetadata(accession="NC_001284"),
            ),
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=second_ref,
                metadata=OrganelleMetadata(accession="NC_001285"),
            ),
        ]
    )
    first_path.write_text(">chr1\nTTTTTTTT\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as captured:
        project.materialize(tmp_path / "mutated-staging")

    assert captured.value.code == "pangenome.source_digest_mismatch"


def test_managed_build_graph_hides_paths_and_uses_deterministic_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    signature = inspect.signature(service.build_graph)
    assert tuple(signature.parameters) == (
        "genomes",
        "method",
        "k",
        "pantools_memory_mb",
        "threads",
        "n_haplotypes",
        "segment_length",
        "identity",
        "recommendation",
        "auto_adopt_recommendation",
        "reference_index",
    )
    forbidden = {"output_dir", "workspace", "executor", "runner"}
    assert forbidden.isdisjoint(signature.parameters)
    assert "Callable" not in str(signature)

    first_path = tmp_path / "first-service.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second-service.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]
    expected = OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version="1.0",
        scope="mitochondrion",
        status="failed",
        summary_text="synthetic builder failure",
        errors=(ErrorDetail(code="test.synthetic", message="synthetic builder failure"),),
    )
    workspaces: list[Path] = []

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        staged = args[0]
        assert isinstance(staged, list)
        assert [genome.object_id for genome in staged] == [genome.object_id for genome in genomes]
        assert all(genome.sequence.resolve().parent.name == "inputs" for genome in staged)
        workspaces.append(Path(kwargs["output_dir"]))
        return expected

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)

    first = service.build_graph(genomes, method="minigraph", threads=2)
    second = service.build_graph(genomes, method="minigraph", threads=2)
    third = service.build_graph(genomes, method="minigraph", threads=3)

    assert first == expected
    assert second == expected
    assert third == expected
    assert len(workspaces) == 3
    assert workspaces[0] == workspaces[1]
    assert workspaces[2] != workspaces[0]
    assert workspaces[0].is_relative_to(tmp_path / "cache" / "runs" / "pangenome.build_graph")


def test_managed_build_graph_publishes_and_reuses_verified_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    first_path = tmp_path / "first-publish.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second-publish.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]
    calls = 0

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        nonlocal calls
        calls += 1
        graph = Path(kwargs["output_dir"]) / "pangenome.gfa"
        graph.write_text("H\tVN:Z:1.1\nS\tseg1\tACGT\n", encoding="utf-8")
        return _successful_graph_result(graph)

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)

    first = service.build_graph(genomes, method="minigraph")
    second = service.build_graph(genomes, method="minigraph")

    assert calls == 1
    assert first.object_id == second.object_id
    assert {artifact.kind for artifact in first.artifacts} == {
        "pangenome_graph",
        "pangenome_run_record",
    }
    assert all(Path(artifact.uri).is_file() for artifact in first.artifacts)
    assert "output_path" not in first.metrics, (
        "managed results expose ArtifactRef, not stale workspace"
    )
    assert first == second


def test_managed_build_graph_rejects_mutated_source_before_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    first_path = tmp_path / "first-cache-mutation.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second-cache-mutation.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        graph = Path(kwargs["output_dir"]) / "pangenome.gfa"
        graph.write_text("H\tVN:Z:1.1\nS\tseg1\tACGT\n", encoding="utf-8")
        return _successful_graph_result(graph)

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)
    service.build_graph(genomes)
    first_path.write_text(">chr1\nTTTTTTTT\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as captured:
        service.build_graph(genomes)

    assert captured.value.code == "pangenome.source_digest_mismatch"


def test_managed_build_uses_verified_input_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    first_path = tmp_path / "snapshot-first.fa"
    original = ">chr1\nACGTACGT\n"
    first_path.write_text(original, encoding="utf-8")
    second_path = tmp_path / "snapshot-second.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        first_path.write_text(">chr1\nTTTTTTTT\n", encoding="utf-8")
        staged = args[0]
        assert isinstance(staged, list)
        staged_path = staged[0].sequence.resolve()
        assert staged_path != first_path.resolve()
        assert staged_path.read_text(encoding="utf-8") == original
        graph = Path(kwargs["output_dir"]) / "pangenome.gfa"
        graph.write_text("H\tVN:Z:1.1\nS\tseg1\tACGT\n", encoding="utf-8")
        return _successful_graph_result(graph)

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)

    result = service.build_graph(genomes)

    assert result.status == "ok"


def test_managed_build_rejects_pseudo_success_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    first_path = tmp_path / "pseudo-first.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "pseudo-second.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        return OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="planned only",
            flags=("graph_planned",),
        )

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)

    with pytest.raises(OrganelleInputError) as captured:
        service.build_graph(genomes)

    assert captured.value.code == "pangenome.invalid_builder_result"


def test_managed_build_graph_rejects_unimplemented_backend_and_invalid_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.pangenome import service

    first_path = tmp_path / "first-validation.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "second-validation.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    with pytest.raises(OrganelleInputError) as unsupported:
        service.build_graph(genomes, method="unknown")  # type: ignore[arg-type]
    assert unsupported.value.code == "pangenome.unsupported_method"

    invalid_calls = (
        {"threads": 0},
        {"k": 0},
        {"segment_length": 0},
        {"identity": 101.0},
        {"reference_index": -1},
        {"reference_index": len(genomes)},
        {"n_haplotypes": 0},
    )
    for parameters in invalid_calls:
        with pytest.raises(OrganelleInputError) as invalid:
            service.build_graph(genomes, **parameters)
        assert invalid.value.code == "pangenome.invalid_parameter"
