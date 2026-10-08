"""ovasm molecules into annotation: topology, headers and the genome of an assembly result."""

from __future__ import annotations

import pytest

from organelleverse.annotation.contigs import apply_molecule_headers, molecule_headers
from organelleverse.annotation.models import AnnotationDocument, AnnotationRecord
from organelleverse.annotation.writer import _seq_records
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import save_contract
from organelleverse.io_genome import read_assembly_genome

_OVASM_FASTA = (
    ">linear.1 circular=true length=12 path=a,b\nACGTACGTACGT\n"
    ">linear.2 circular=false length=8 path=c\nTTGGCCAA\n"
)


def _document(*ids: str) -> AnnotationDocument:
    records = tuple(
        AnnotationRecord(
            seqid=i, name=i, description="definition", sequence="ACGT" * 4, features=()
        )
        for i in ids
    )
    return AnnotationDocument(
        backend="mitochondrion", requested_stages=(), completed_stages=(), records=records
    )


def test_ovasm_headers_give_id_description_and_topology(tmp_path):
    fasta = tmp_path / "molecules.fasta"
    fasta.write_text(_OVASM_FASTA)
    assert molecule_headers(fasta) == [
        ("linear.1", "circular=true length=12 path=a,b", True),
        ("linear.2", "circular=false length=8 path=c", False),
    ]


def test_a_plain_fasta_has_no_topology(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">NC_1.1 Arabidopsis thaliana mitochondrion, complete genome\nACGT\n")
    assert molecule_headers(fasta)[0][2] is None


def test_each_record_gets_its_own_locus_topology(tmp_path):
    fasta = tmp_path / "molecules.fasta"
    fasta.write_text(_OVASM_FASTA)
    document = apply_molecule_headers(_document("linear.1", "linear.2"), molecule_headers(fasta))
    assert [r.annotations["topology"] for r in _seq_records(document)] == ["circular", "linear"]
    assert document.records[0].description == "circular=true length=12 path=a,b"


def test_a_single_molecule_is_marked_even_when_the_pipeline_named_the_record(tmp_path):
    fasta = tmp_path / "molecules.fasta"
    fasta.write_text(">linear circular=true length=16 path=a\n" + "ACGT" * 4 + "\n")
    document = apply_molecule_headers(_document("Arabidopsis_thaliana"), molecule_headers(fasta))
    assert _seq_records(document)[0].annotations["topology"] == "circular"


def test_an_ordinary_fasta_leaves_the_document_untouched(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">only\n" + "ACGT" * 4 + "\n")
    document = _document("only")
    assert apply_molecule_headers(document, molecule_headers(fasta)) is document


def test_a_record_count_mismatch_leaves_the_document_untouched():
    document = _document("a", "b")
    assert apply_molecule_headers(document, [("x", "circular=true", True)]) is document


def test_without_record_topology_the_document_topology_applies():
    document = _document("a")
    assert _seq_records(document)[0].annotations["topology"] == "linear"


@pytest.fixture
def assembly_result(tmp_path):
    fasta = tmp_path / "assembly.fasta"
    fasta.write_text(_OVASM_FASTA)
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(
            fasta, kind="sequence", format="fasta", media_type="text/x-fasta"
        ),
    )
    manifest = save_contract(genome, tmp_path / "primary_genome.json")
    return OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
        artifacts=(
            ArtifactRef.from_path(
                manifest,
                kind="primary_genome_manifest",
                format="json",
                media_type="application/json",
            ),
        ),
    )


def test_the_genome_of_an_assembly_result_takes_the_species(assembly_result):
    genome = read_assembly_genome(assembly_result, species="Salvia miltiorrhiza")
    assert genome.organelle == "mitochondrion"
    assert genome.metadata.species == "Salvia miltiorrhiza"
    assert genome.sequence is not None


def test_a_blank_species_is_refused(assembly_result):
    with pytest.raises(OrganelleInputError):
        read_assembly_genome(assembly_result, species=" ")


def test_a_result_that_is_not_an_assembly_is_refused(assembly_result):
    other = assembly_result.evolve(operation_id="annotation.annotate")
    with pytest.raises(OrganelleInputError, match="assembly operation"):
        read_assembly_genome(other, species="x")


def test_a_result_without_the_genome_manifest_is_refused(assembly_result):
    empty = assembly_result.evolve(artifacts=())
    with pytest.raises(OrganelleInputError, match="primary genome manifest"):
        read_assembly_genome(empty, species="x")


def test_an_assembly_genome_gets_the_organelle_standard_genetic_code(assembly_result, tmp_path):
    mito = read_assembly_genome(assembly_result, species="Salvia miltiorrhiza")
    assert mito.metadata.genetic_code == 1

    fasta = tmp_path / "plastid.fasta"
    fasta.write_text(">p\n" + "ACGT" * 4 + "\n")
    plastid = OrganelleGenome(
        organelle="plastid",
        sequence=ArtifactRef.from_path(
            fasta, kind="sequence", format="fasta", media_type="text/x-fasta"
        ),
    )
    manifest = save_contract(plastid, tmp_path / "plastid_genome.json")
    result = assembly_result.evolve(
        scope="plastid",
        artifacts=(
            ArtifactRef.from_path(
                manifest,
                kind="primary_genome_manifest",
                format="json",
                media_type="application/json",
            ),
        ),
    )
    assert read_assembly_genome(result, species="x").metadata.genetic_code == 11


def test_a_stated_genetic_code_is_kept_for_an_assembly_genome(assembly_result):
    genome = read_assembly_genome(
        assembly_result, species="x", metadata=OrganelleMetadata(genetic_code=4)
    )
    assert genome.metadata.genetic_code == 4
