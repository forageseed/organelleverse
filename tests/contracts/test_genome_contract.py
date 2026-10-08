from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata


class ExtendedMetadata(OrganelleMetadata):
    review_note: str = "must not cross the boundary"


def _genome(**changes: object) -> OrganelleGenome:
    values: dict[str, object] = {
        "organelle": "mitochondrion",
        "sequence": ArtifactRef(
            kind="sequence",
            uri="genome.fa",
            format="fasta",
            sha256="a" * 64,
            size_bytes=8,
        ),
        "metadata": OrganelleMetadata(species="Example species", source="test"),
    }
    values.update(changes)
    return OrganelleGenome.model_validate(values)


def _manifest_ref(*, uri: str, sha256: str) -> ArtifactRef:
    return ArtifactRef(
        kind="assembly_run_manifest",
        uri=uri,
        format="json",
        media_type="application/json",
        sha256=sha256,
        size_bytes=128,
    )


def test_artifact_ref_hashes_existing_file(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">g\nACGT\n", encoding="utf-8")
    artifact = ArtifactRef.from_path(fasta, kind="sequence", format="fasta")
    assert artifact.sha256
    assert artifact.size_bytes == fasta.stat().st_size
    assert artifact.validated is True


def test_artifact_ref_identity_excludes_local_uri(tmp_path: Path) -> None:
    first = tmp_path / "first.fa"
    second = tmp_path / "second.fa"
    first.write_text(">g\nACGT\n", encoding="utf-8")
    second.write_text(">g\nACGT\n", encoding="utf-8")

    first_ref = ArtifactRef.from_path(first, kind="sequence", format="fasta")
    second_ref = ArtifactRef.from_path(second, kind="sequence", format="fasta")

    assert first_ref.uri != second_ref.uri
    assert first_ref.object_id == second_ref.object_id


def test_genome_requires_sequence_or_annotation() -> None:
    with pytest.raises(ValidationError, match="sequence or annotation"):
        OrganelleGenome(organelle="mitochondrion")


def test_genome_uses_canonical_organelle_values(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">g\nACGT\n", encoding="utf-8")
    artifact = ArtifactRef.from_path(fasta, kind="sequence", format="fasta")
    genome = OrganelleGenome(
        organelle="plastid",
        sequence=artifact,
        metadata=OrganelleMetadata(species="Arabidopsis thaliana", plastid_type="chloroplast"),
    )
    assert genome.kind == "genome"
    assert genome.object_id.startswith("genome:sha256:")
    with pytest.raises(ValidationError):
        OrganelleGenome(organelle="chloro", sequence=artifact)  # type: ignore[arg-type]


def test_genome_identity_uses_artifact_object_ids_not_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.fa"
    second = tmp_path / "second.fa"
    first.write_text(">g\nACGT\n", encoding="utf-8")
    second.write_text(">g\nACGT\n", encoding="utf-8")

    first_genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(first, kind="sequence", format="fasta"),
    )
    second_genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(second, kind="sequence", format="fasta"),
    )

    assert first_genome.object_id == second_genome.object_id


def test_empty_evidence_fields_preserve_the_legacy_genome_identity() -> None:
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef(
            kind="sequence",
            uri="genome.fa",
            format="fasta",
            sha256="a" * 64,
            size_bytes=8,
        ),
        metadata=OrganelleMetadata(species="Example species", source="test"),
    )

    assert genome.lineage == ()
    assert genome.source_manifests == ()
    assert genome.object_id == (
        "genome:sha256:b5979b4fbefd2f87b7a55000305ca9dc245765fa6f308ace639292fcfd36604b"
    )


def test_nonempty_lineage_and_source_manifest_change_genome_identity() -> None:
    parent = OrganelleData(modality="sequencing_reads")
    manifest = _manifest_ref(uri="run.json", sha256="d" * 64)
    base = _genome()
    derived = base.evolve(
        lineage=(
            LineageRecord(
                parent_object_ids=(parent.object_id,),
                operation_id="assembly.assemble",
                operation_version="1.0",
                parameters_hash="e" * 64,
            ),
        ),
        source_manifests=(manifest,),
    )

    assert derived.object_id != base.object_id
    assert derived.source_manifests == (manifest,)


def test_source_manifests_are_unique_by_content_hash() -> None:
    first = _manifest_ref(uri="first/run.json", sha256="d" * 64)
    second = _manifest_ref(uri="second/run.json", sha256="d" * 64)

    with pytest.raises(ValidationError, match="unique by content SHA256"):
        _genome(source_manifests=(first, second))


def test_genome_evolve_returns_revalidated_new_object(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">g\nACGT\n", encoding="utf-8")
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
    )
    updated = genome.evolve(metadata=OrganelleMetadata(species="Test species"))
    assert updated is not genome
    assert genome.metadata.species == ""
    assert updated.metadata.species == "Test species"


def test_metadata_serialized_object_id_rejects_explicit_null_but_accepts_omission() -> None:
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    serialized = metadata.model_dump(mode="json")
    omitted = {key: value for key, value in serialized.items() if key != "object_id"}

    assert OrganelleMetadata.model_validate(omitted) == metadata
    assert OrganelleMetadata.model_validate(serialized) == metadata

    with pytest.raises(ValidationError, match="serialized object_id must be a string"):
        OrganelleMetadata.model_validate({**serialized, "object_id": None})

    with pytest.raises(ValidationError, match="does not match computed object_id"):
        OrganelleMetadata.model_validate({**serialized, "object_id": "metadata:sha256:" + "0" * 64})


def test_nested_metadata_instances_are_revalidated_at_core_boundaries() -> None:
    artifact = ArtifactRef(
        kind="sequence",
        uri="genome.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=8,
    )
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")

    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=artifact,
        metadata=metadata,
    )
    assert type(genome.metadata) is OrganelleMetadata
    assert genome.metadata == metadata

    with pytest.raises(ValidationError, match="genetic_code"):
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=artifact,
            metadata=OrganelleMetadata.model_construct(genetic_code=99),
        )

    with pytest.raises(ValidationError, match="review_note"):
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=artifact,
            metadata=ExtendedMetadata(species="Arabidopsis thaliana"),
        )
