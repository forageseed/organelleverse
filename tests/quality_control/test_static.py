from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    LongLibraryContract,
    ShortLibraryContract,
)
from organelleverse.assembly.manifests import (
    AssemblyComponentIdentity,
    AssemblyEnvironmentIdentity,
    AssemblyRunManifest,
    AssemblyRunParameters,
    AssemblyStageOutcome,
    ManifestArtifact,
    RoutingEvidence,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import save_contract
from organelleverse.quality_control.contracts import ResolvedAssemblyEvidence
from organelleverse.quality_control.static import (
    compute_assembly_statistics,
    resolve_assembly_evidence,
    summarize_fasta,
    validate_gfa,
)

_ENV_DIGEST = "sha256:" + "b" * 64
_OATK_SHA = "c" * 64
_OATKDB_SHA = "d" * 64
_INPUT_DATA_ID = "data:sha256:" + "1" * 64
_PARAMETERS_HASH = "f" * 64


@dataclass(frozen=True)
class _Published:
    result: OrganelleResult
    manifest: AssemblyRunManifest
    files: dict[str, Path]
    artifacts: dict[str, ArtifactRef]


def _fastq(path: Path) -> Path:
    path.write_text("@r\nACGTACGT\n+\nIIIIIIII\n")
    return path


def _input_payload(*, undeclared_long_role: bool = False) -> AssemblyInputPayload:
    long_libraries = [
        LongLibraryContract(
            technology="pacbio_hifi",
            quality_state="ccs",
            reads_artifact="long_reads",
        )
    ]
    if undeclared_long_role:
        long_libraries.append(
            LongLibraryContract(
                technology="ont",
                quality_state="raw",
                reads_artifact="ghost_reads",
            )
        )
    return AssemblyInputPayload(
        short_libraries=(
            ShortLibraryContract(
                technology="illumina",
                layout="paired_end",
                read1_artifact="pe_r1",
                read2_artifact="pe_r2",
                read_length=150,
                insert_size=300,
            ),
        ),
        long_libraries=tuple(long_libraries),
    )


def _publish_assembly_evidence(
    tmp_path: Path,
    *,
    taxon_group: str | None = "plant",
    include_payload: bool = True,
    undeclared_long_role: bool = False,
    result_scope: str | None = None,
    genome_organelle: str | None = None,
    source_manifest: Literal["correct", "wrong"] = "correct",
    primary_sequence: Literal["correct", "wrong"] = "correct",
    primary_output_uri: Literal["correct", "missing", "tampered"] = "correct",
    genome_sequence_uri: Literal["correct", "missing", "tampered"] = "correct",
    lineage_input_data: bool = True,
    lineage_param_hash: bool = True,
    lineage_operation_id: str = "assembly.assemble",
    lineage_operation_version: str = "1.0",
    extra_lineage: bool = False,
    lineage_extra_parent: bool = False,
    provenance_actual_backend: str = "oatk",
    provenance_operation_version: str = "1.0",
    provenance_input_object_ids: tuple[str, ...] = (_INPUT_DATA_ID,),
    result_operation_version: str = "1.0",
) -> _Published:
    work = tmp_path / "work"
    work.mkdir()
    files = {
        "long_reads": _fastq(work / "long_reads.fastq"),
        "pe_r1": _fastq(work / "pe_r1.fastq"),
        "pe_r2": _fastq(work / "pe_r2.fastq"),
        "hmm_profiles": work / "hmm.fam",
        "primary_fasta": work / "primary.fa",
        "assembly_graph": work / "assembly.gfa",
    }
    files["hmm_profiles"].write_text("profile-content")
    files["primary_fasta"].write_text(">ctg1\nACGTACGTACGTACGTNNACGTACGTACGTACGT\n")
    files["assembly_graph"].write_text("H\tVN:Z:1.0\nS\tctg1\tACGTACGT\nL\tctg1\t+\tctg1\t-\t0M\n")

    def _ref(role: str, path: Path, *, kind: str, fmt: str, media: str) -> ArtifactRef:
        return ArtifactRef.from_path(path, kind=kind, format=fmt, media_type=media)

    artifacts = {
        "long_reads": _ref(
            "long_reads",
            files["long_reads"],
            kind="long_read",
            fmt="fastq",
            media="application/x-fastq",
        ),
        "pe_r1": _ref(
            "pe_r1", files["pe_r1"], kind="short_read", fmt="fastq", media="application/x-fastq"
        ),
        "pe_r2": _ref(
            "pe_r2", files["pe_r2"], kind="short_read", fmt="fastq", media="application/x-fastq"
        ),
        "hmm_profiles": _ref(
            "hmm_profiles", files["hmm_profiles"], kind="hmm_profile", fmt="fam", media="text/plain"
        ),
        "primary_fasta": _ref(
            "primary_fasta",
            files["primary_fasta"],
            kind="sequence",
            fmt="fasta",
            media="text/x-fasta",
        ),
        "assembly_graph": _ref(
            "assembly_graph",
            files["assembly_graph"],
            kind="assembly_graph",
            fmt="gfa",
            media="text/x-gfa",
        ),
    }

    real_primary = artifacts["primary_fasta"]
    if primary_output_uri == "missing":
        # Manifest output points at a missing URI but carries the real hash.
        primary_output_artifact = real_primary.model_copy(
            update={"uri": str(work / "nonexistent.fa")}
        )
    elif primary_output_uri == "tampered":
        # Manifest output points at a real file with different content but the
        # real hash copied into its metadata.
        tampered_path = work / "tampered.fa"
        tampered_path.write_text(">ctg1\nTTTTGGGGCCCC\n")
        primary_output_artifact = ArtifactRef.from_path(
            tampered_path, kind="sequence", format="fasta", media_type="text/x-fasta"
        ).model_copy(update={"sha256": real_primary.sha256, "size_bytes": real_primary.size_bytes})
    else:
        primary_output_artifact = real_primary

    routing = (
        None
        if taxon_group is None
        else RoutingEvidence(rule_id="route.explicit_method", taxon_group=taxon_group)  # type: ignore[arg-type]
    )
    manifest = AssemblyRunManifest(
        input_data_id=_INPUT_DATA_ID,
        input_artifacts=(
            ManifestArtifact(role="long_reads", artifact=artifacts["long_reads"]),
            ManifestArtifact(role="pe_r1", artifact=artifacts["pe_r1"]),
            ManifestArtifact(role="pe_r2", artifact=artifacts["pe_r2"]),
            ManifestArtifact(role="hmm_profiles", artifact=artifacts["hmm_profiles"]),
        ),
        input_payload=_input_payload(undeclared_long_role=undeclared_long_role)
        if include_payload
        else None,
        organelle="mitochondrion",
        parameters=AssemblyRunParameters(
            threads=8,
            memory_gb=32,
            timeout_seconds=3600,
            environment_source="managed",
            backend_parameters={"backend": "oatk", "kmer_size": 1001},
        ),
        requested_method="auto",
        selected_backend="oatk",
        route_reason_code="route.explicit_method",
        routing_evidence=routing,
        environment=AssemblyEnvironmentIdentity(
            carrier="conda",
            digest=_ENV_DIGEST,
            platform="linux-64",
        ),
        components=(
            AssemblyComponentIdentity(
                category="software", name="oatk", version="1.0", sha256=_OATK_SHA
            ),
            AssemblyComponentIdentity(
                category="database", name="oatkdb", version="2026.07", sha256=_OATKDB_SHA
            ),
        ),
        stable_argv=(
            "oatk",
            "mito",
            "-i",
            "role://artifact/long_reads",
            "-o",
            "role://workspace/assembly",
        ),
        stages=(
            AssemblyStageOutcome(stage="prepare", status="ok"),
            AssemblyStageOutcome(
                stage="assemble",
                status="ok",
                process_started=True,
                exit_code=0,
                termination="exit",
                output_roles=("primary_fasta", "assembly_graph"),
            ),
        ),
        process_exit_code=0,
        primary_sequence_role="primary_fasta",
        outputs=(
            ManifestArtifact(role="primary_fasta", artifact=primary_output_artifact),
            ManifestArtifact(role="assembly_graph", artifact=artifacts["assembly_graph"]),
        ),
    )

    manifest_path = work / "assembly_run_manifest.json"
    manifest_path.write_bytes(manifest.canonical_bytes())
    manifest_artifact = ArtifactRef.from_path(
        manifest_path, kind="assembly_run_manifest", format="json", media_type="application/json"
    )
    record_path = work / "assembly_run_record.json"
    record_path.write_text(manifest.model_dump_json() + "\n")
    record_artifact = ArtifactRef.from_path(
        record_path, kind="assembly_run_record", format="json", media_type="application/json"
    )

    if source_manifest == "wrong":
        wrong_path = work / "wrong_manifest.json"
        wrong_path.write_bytes(b'{"not": "the canonical manifest"}')
        source_manifests: tuple[ArtifactRef, ...] = (
            ArtifactRef.from_path(
                wrong_path,
                kind="assembly_run_manifest",
                format="json",
                media_type="application/json",
            ),
        )
    else:
        source_manifests = (manifest_artifact,)

    if primary_sequence == "wrong":
        wrong_seq_path = work / "wrong_seq.fa"
        wrong_seq_path.write_text(">other\nTTTTGGGG\n")
        sequence_artifact = ArtifactRef.from_path(
            wrong_seq_path, kind="sequence", format="fasta", media_type="text/x-fasta"
        )
    else:
        sequence_artifact = artifacts["primary_fasta"]

    if genome_sequence_uri == "missing":
        # Genome sequence points at a missing URI but carries the real hash.
        sequence_artifact = sequence_artifact.model_copy(
            update={"uri": str(work / "missing_genome.fa")}
        )
    elif genome_sequence_uri == "tampered":
        # Genome sequence points at real, different content with the real hash
        # copied into its metadata.
        tampered_genome_path = work / "tampered_genome.fa"
        tampered_genome_path.write_text(">ctg1\nAAAACCCCgggg\n")
        sequence_artifact = ArtifactRef.from_path(
            tampered_genome_path,
            kind=sequence_artifact.kind,
            format="fasta",
            media_type="text/x-fasta",
        ).model_copy(
            update={
                "sha256": sequence_artifact.sha256,
                "size_bytes": sequence_artifact.size_bytes,
            }
        )

    lineage_hash = _PARAMETERS_HASH if lineage_param_hash else "a" * 64
    lineage_parent = (_INPUT_DATA_ID,) if lineage_input_data else ("data:sha256:" + "9" * 64,)
    if lineage_extra_parent:
        lineage_parent = (*lineage_parent, "data:sha256:" + "2" * 64)
    lineage_records = [
        LineageRecord(
            parent_object_ids=lineage_parent,
            operation_id=lineage_operation_id,
            operation_version=lineage_operation_version,
            parameters_hash=lineage_hash,
        )
    ]
    if extra_lineage:
        lineage_records.append(
            LineageRecord(
                parent_object_ids=lineage_parent,
                operation_id="assembly.assemble",
                operation_version="1.0",
                parameters_hash=lineage_hash,
            )
        )
    genome = OrganelleGenome(
        organelle=genome_organelle or "mitochondrion",
        sequence=sequence_artifact,
        lineage=tuple(lineage_records),
        source_manifests=source_manifests,
    )
    genome_path = work / "primary_genome.json"
    save_contract(genome, genome_path)
    genome_artifact = ArtifactRef.from_path(
        genome_path, kind="primary_genome_manifest", format="json", media_type="application/json"
    )

    provenance = ResultProvenance(
        operation_id="assembly.assemble",
        operation_version=provenance_operation_version,
        package_version="0.0.1",
        git_commit="0" * 40,
        input_object_ids=provenance_input_object_ids,
        parameters_hash=_PARAMETERS_HASH,
        actual_backend=provenance_actual_backend,
        attempted_backends=(provenance_actual_backend,),
        run_manifest_id=manifest.run_manifest_id,
    )
    result = OrganelleResult(
        operation_id="assembly.assemble",
        operation_version=result_operation_version,
        scope=result_scope or manifest.organelle,  # type: ignore[arg-type]
        status="ok",
        artifacts=(
            manifest_artifact,
            record_artifact,
            genome_artifact,
            artifacts["primary_fasta"],
            artifacts["assembly_graph"],
            artifacts["long_reads"],
            artifacts["pe_r1"],
            artifacts["pe_r2"],
            artifacts["hmm_profiles"],
        ),
        provenance=provenance,
    )
    return _Published(result=result, manifest=manifest, files=files, artifacts=artifacts)


# -- summarize_fasta -------------------------------------------------------


def test_summarize_fasta_preserves_ids_and_counts_gc(tmp_path: Path) -> None:
    fa = tmp_path / "seqs.fa"
    fa.write_text(">ctg1\nACGTACGTNN\n>ctg2\nGGGGCCCC\n")
    summaries = summarize_fasta(fa)
    assert [s.sequence_id for s in summaries] == ["ctg1", "ctg2"]
    assert summaries[0].length == 10
    assert summaries[0].gc_fraction == pytest.approx(0.4)
    assert summaries[0].ambiguous_bases == 2  # two N
    assert summaries[1].gc_fraction == 1.0
    assert summaries[1].ambiguous_bases == 0


def test_summarize_fasta_rejects_duplicate_ids(tmp_path: Path) -> None:
    fa = tmp_path / "dup.fa"
    fa.write_text(">ctg1\nACGT\n>ctg1\nGGGG\n")
    with pytest.raises(ValueError, match="unique"):
        summarize_fasta(fa)


def test_summarize_fasta_rejects_empty(tmp_path: Path) -> None:
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    with pytest.raises(ValueError, match="no records"):
        summarize_fasta(fa)


# -- compute_assembly_statistics (descriptive; N50/L50) --------------------


def test_compute_assembly_statistics_reports_correct_n50_and_l50(tmp_path: Path) -> None:
    # Lengths 500,400,300,200,100; total 1500; half 750. Sorted descending, the
    # cumulative length first reaches >=750 at the second contig (400): N50=400,
    # L50=2. This is the standard contig-N50/L50 definition.
    fa = tmp_path / "multi.fa"
    fa.write_text(
        ">a\n" + "A" * 500 + "\n"
        ">b\n" + "A" * 400 + "\n"
        ">c\n" + "A" * 300 + "\n"
        ">d\n" + "A" * 200 + "\n"
        ">e\n" + "A" * 100 + "\n"
    )
    stats = compute_assembly_statistics(fa)
    assert stats.sequence_count == 5
    assert stats.total_length == 1_500
    assert stats.largest_sequence_length == 500
    assert stats.n50 == 400
    assert stats.l50 == 2
    assert stats.overall_gc_fraction == 0.0
    assert stats.ambiguous_bases == 0


def test_compute_assembly_statistics_single_sequence_n50_equals_length(
    tmp_path: Path,
) -> None:
    fa = tmp_path / "single.fa"
    fa.write_text(">only\nACGTNN\n")
    stats = compute_assembly_statistics(fa)
    assert stats.sequence_count == 1
    assert stats.total_length == 6
    assert stats.largest_sequence_length == 6
    assert stats.n50 == 6
    assert stats.l50 == 1
    assert stats.overall_gc_fraction == pytest.approx(1 / 3)
    assert stats.ambiguous_bases == 2


# -- validate_gfa ----------------------------------------------------------


def test_validate_gfa_summarizes_a_valid_graph(tmp_path: Path) -> None:
    gfa = tmp_path / "g.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\n"
        "S\ta\tACGT\n"
        "S\tb\tTGCA\n"
        "S\tc\tGGGG\n"
        "L\ta\t+\tb\t+\t0M\n"
        "L\tb\t+\tc\t+\t0M\n"
        "P\tpath1\ta+,b+,c+\t0M\n"
    )
    summary = validate_gfa(gfa)
    assert summary.segment_count == 3
    assert summary.edge_count == 2
    assert summary.path_count == 1
    assert summary.path_names == ("path1",)


def test_validate_gfa_accepts_forward_segment_references(tmp_path: Path) -> None:
    """GFA permits links and paths before the referenced S records."""
    gfa = tmp_path / "forward.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\nS\t1\tA\nL\t1\t+\t2\t+\t0M\nP\tp1\t1+,2+\t*\nS\t2\tC\n",
        encoding="utf-8",
    )

    summary = validate_gfa(gfa)

    assert summary.segment_count == 2
    assert summary.edge_count == 1
    assert summary.path_names == ("p1",)
    assert summary.component_count == 1
    assert summary.parallel_edge_count == 0


def test_validate_gfa_counts_parallel_edges_without_calling_them_bubbles(
    tmp_path: Path,
) -> None:
    # Two links joining the same segment pair are parallel edges. The honest
    # metric name records exactly that; it does not claim a graph bubble.
    gfa = tmp_path / "parallel.gfa"
    gfa.write_text("H\tVN:Z:1.0\nS\ta\tACGT\nS\tb\tTGCA\nL\ta\t+\tb\t+\t0M\nL\ta\t-\tb\t-\t0M\n")
    summary = validate_gfa(gfa)
    assert summary.parallel_edge_count == 1
    assert summary.edge_count == 2


def test_validate_gfa_rejects_unknown_link_reference(tmp_path: Path) -> None:
    gfa = tmp_path / "bad.gfa"
    gfa.write_text("H\tVN:Z:1.0\nS\ta\tACGT\nL\ta\t+\tmissing\t+\t0M\n")
    with pytest.raises(OrganelleInputError, match="unknown"):
        validate_gfa(gfa)


def test_validate_gfa_rejects_unknown_path_reference_after_full_parse(tmp_path: Path) -> None:
    gfa = tmp_path / "unknown-path.gfa"
    gfa.write_text("S\ta\tACGT\nP\tp1\ta+,missing+\t*\n", encoding="utf-8")
    with pytest.raises(OrganelleInputError, match="unknown segment: missing"):
        validate_gfa(gfa)


def test_validate_gfa_rejects_duplicate_segments(tmp_path: Path) -> None:
    gfa = tmp_path / "dup.gfa"
    gfa.write_text("H\tVN:Z:1.0\nS\ta\tACGT\nS\ta\tGGGG\n")
    with pytest.raises(OrganelleInputError, match="duplicate"):
        validate_gfa(gfa)


# -- resolve_assembly_evidence: happy path + input verification ------------


def test_resolve_assembly_evidence_loads_the_full_contract_chain(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)

    evidence = resolve_assembly_evidence(published.result)

    assert isinstance(evidence, ResolvedAssemblyEvidence)
    assert evidence.run_manifest == published.manifest
    assert evidence.primary_fasta_path == published.files["primary_fasta"]
    assert evidence.assembly_graph_path == published.files["assembly_graph"]


def test_resolve_builds_input_libraries_only_from_payload(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    evidence = resolve_assembly_evidence(published.result)

    by_role = {lib.role: lib for lib in evidence.input_libraries}
    # Short paired-end library resolved exactly from the payload.
    short = by_role["pe_r1"]
    assert short.technology == "illumina"
    assert short.layout == "paired_end"
    assert short.read1 == published.artifacts["pe_r1"]
    assert short.read2 == published.artifacts["pe_r2"]
    assert short.read_length == 150
    assert short.insert_size == 300
    assert short.reads is None
    # Long library resolved exactly from the payload.
    long_lib = by_role["long_reads"]
    assert long_lib.technology == "pacbio_hifi"
    assert long_lib.quality_state == "ccs"
    assert long_lib.reads == published.artifacts["long_reads"]
    assert long_lib.read1 is None and long_lib.read2 is None
    # Managed HMM resource is verified but never represented as a read library.
    assert "hmm_profiles" not in by_role
    assert len(evidence.input_libraries) == 2


def test_resolve_requires_input_payload_else_unsupported_technology(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, include_payload=False)

    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.unsupported_technology"


def test_resolve_rejects_tampered_input_artifact(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    published.files["long_reads"].write_text("@r\nTTTTGGGG\n+\nIIIIIIII\n")

    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


def test_resolve_rejects_missing_input_artifact_file(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    published.files["pe_r1"].unlink()

    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_missing"


def test_resolve_rejects_library_role_not_declared_as_input(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, undeclared_long_role=True)

    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_missing"


def test_resolve_rejects_tampered_published_sequence(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    published.files["primary_fasta"].write_text(">ctg1\nTTTTGGGG\n")

    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


# -- resolve_assembly_evidence: taxon / operation / run identity -----------


def test_resolve_requires_plant_taxon(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, taxon_group="animal")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.unsupported_taxon"


def test_resolve_rejects_missing_plant_taxon(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, taxon_group=None)
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.unsupported_taxon"


def test_resolve_rejects_mismatched_run_identity(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    mismatched = published.result.model_copy(
        update={
            "provenance": published.result.provenance.model_copy(
                update={"run_manifest_id": "assembly-run:sha256:" + "e" * 64}
            )
        }
    )
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(mismatched)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_requires_assembly_operation(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    other = published.result.model_copy(update={"operation_id": "annotation.annotate"})
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(other)
    assert exc.value.code == "qc.input_contract_violation"


# -- resolve_assembly_evidence: result/manifest/genome identity chain ------


def test_resolve_rejects_scope_organelle_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, result_scope="plastid")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_genome_organelle_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, genome_organelle="plastid")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_source_manifest_not_canonical(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, source_manifest="wrong")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


def test_resolve_rejects_primary_sequence_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, primary_sequence="wrong")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


def test_resolve_rejects_lineage_input_data_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, lineage_input_data=False)
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_lineage_parameter_hash_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, lineage_param_hash=False)
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


# -- resolve_assembly_evidence: manifest primary output authority ------------


def test_resolve_rejects_missing_manifest_primary_output_uri(tmp_path: Path) -> None:
    # The manifest output URI is missing but carries the real hash; the Genome
    # sequence is valid. Today this can pass because only Genome.sequence is read.
    published = _publish_assembly_evidence(tmp_path, primary_output_uri="missing")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_missing"


def test_resolve_rejects_tampered_manifest_primary_output(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, primary_output_uri="tampered")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


def test_resolve_returns_manifest_primary_output_path(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path)
    evidence = resolve_assembly_evidence(published.result)
    # The manifest primary-sequence output is the authority for the path.
    assert evidence.primary_fasta_path == published.files["primary_fasta"]


# -- resolve_assembly_evidence: lineage exactness + provenance agreement -----


def test_resolve_rejects_multiple_lineage_records(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, extra_lineage=True)
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_forged_lineage_operation_id(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, lineage_operation_id="annotation.annotate")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_lineage_operation_version_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, lineage_operation_version="2.0")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_provenance_actual_backend_mismatch(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, provenance_actual_backend="getorganelle")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


# -- resolve_assembly_evidence: Genome sequence URI is reverified -------------


def test_resolve_rejects_missing_genome_sequence_uri(tmp_path: Path) -> None:
    # Manifest output is valid; Genome sequence URI is missing with copied hash.
    published = _publish_assembly_evidence(tmp_path, genome_sequence_uri="missing")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_missing"


def test_resolve_rejects_tampered_genome_sequence_uri(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, genome_sequence_uri="tampered")
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.artifact_digest_mismatch"


def test_resolve_returns_manifest_primary_output_path_even_after_genome_recheck(
    tmp_path: Path,
) -> None:
    published = _publish_assembly_evidence(tmp_path)
    evidence = resolve_assembly_evidence(published.result)
    assert evidence.primary_fasta_path == published.files["primary_fasta"]


# -- resolve_assembly_evidence: provenance / lineage exactness ----------------


def test_resolve_rejects_provenance_input_object_ids_mismatch(tmp_path: Path) -> None:
    # input_data_id is present, plus an extra entry: membership passes, exact
    # tuple match must not.
    published = _publish_assembly_evidence(
        tmp_path,
        provenance_input_object_ids=(_INPUT_DATA_ID, "data:sha256:" + "7" * 64),
    )
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_operation_version_not_equal_manifest_contract(tmp_path: Path) -> None:
    # result/provenance/lineage all agree with each other but not manifest.contract_version.
    published = _publish_assembly_evidence(
        tmp_path,
        result_operation_version="2.0",
        provenance_operation_version="2.0",
        lineage_operation_version="2.0",
    )
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"


def test_resolve_rejects_lineage_extra_parent(tmp_path: Path) -> None:
    published = _publish_assembly_evidence(tmp_path, lineage_extra_parent=True)
    with pytest.raises(OrganelleInputError) as exc:
        resolve_assembly_evidence(published.result)
    assert exc.value.code == "qc.input_contract_violation"
