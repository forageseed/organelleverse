"""The two genome readers. Each populates the one artifact its consumer can use.

annotation.annotate passes genome.sequence straight through as input_fasta and
the pipeline parses it as FASTA, so a GenBank path cannot serve as a sequence
artifact. FASTA therefore feeds annotate; GenBank feeds extract.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from organelleverse import operations as op
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.io_genome import read_fasta_genome, read_genbank_genome

_FASTA = ">seq1\nACGTACGT\n"
_GENBANK = "LOCUS       X   8 bp    DNA     circular\nFEATURES\nORIGIN\n        1 acgtacgt\n//\n"
_SPECIES = "Arabidopsis thaliana"


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    path = tmp_path / "genome.fasta"
    path.write_text(_FASTA, encoding="utf-8")
    return path


@pytest.fixture
def genbank(tmp_path: Path) -> Path:
    path = tmp_path / "genome.gb"
    path.write_text(_GENBANK, encoding="utf-8")
    return path


def test_both_operations_are_registered() -> None:
    ids = {spec.operation_id for spec in op.list()}
    assert {"io.read_fasta_genome", "io.read_genbank_genome"} <= ids
    assert len(op.list()) == 20


def test_fasta_populates_sequence_only(fasta: Path) -> None:
    genome = read_fasta_genome(fasta, organelle="mitochondrion", species=_SPECIES)
    assert isinstance(genome, OrganelleGenome)
    assert genome.sequence is not None
    assert genome.sequence.format == "fasta"
    assert genome.annotation is None
    assert genome.metadata.species == _SPECIES


def test_genbank_populates_annotation_only(genbank: Path) -> None:
    genome = read_genbank_genome(genbank, organelle="plastid", species=_SPECIES)
    assert genome.annotation is not None
    assert genome.annotation.format == "genbank"
    assert genome.sequence is None


def test_blank_species_is_refused_before_the_file_is_opened(tmp_path: Path) -> None:
    """The released mitochondrial backend rejects blank species, so a genome with
    default metadata would be type-valid, reported reachable, and rejected at
    invocation. Refuse it here instead."""
    absent = tmp_path / "does-not-exist.fasta"
    with pytest.raises(OrganelleInputError) as excinfo:
        read_fasta_genome(absent, organelle="mitochondrion", species="   ")
    assert excinfo.value.code == "input.missing_species"


def test_contradictory_metadata_species_is_refused(fasta: Path) -> None:
    with pytest.raises(OrganelleInputError) as excinfo:
        read_fasta_genome(
            fasta,
            organelle="mitochondrion",
            species=_SPECIES,
            metadata=OrganelleMetadata(species="Zea mays"),
        )
    assert excinfo.value.code == "input.metadata_species_conflict"


def test_supplied_metadata_is_preserved_and_species_filled(fasta: Path) -> None:
    genome = read_fasta_genome(
        fasta,
        organelle="mitochondrion",
        species=_SPECIES,
        metadata=OrganelleMetadata(accession="NC_000932.1", genetic_code=1),
    )
    assert genome.metadata.species == _SPECIES
    assert genome.metadata.accession == "NC_000932.1"


def test_each_reader_refuses_the_other_format(fasta: Path, genbank: Path) -> None:
    with pytest.raises(OrganelleInputError) as fasta_error:
        read_genbank_genome(fasta, organelle="mitochondrion", species=_SPECIES)
    assert fasta_error.value.code == "input.invalid_genbank"
    with pytest.raises(OrganelleInputError) as genbank_error:
        read_fasta_genome(genbank, organelle="mitochondrion", species=_SPECIES)
    assert genbank_error.value.code == "input.invalid_fasta"


def test_the_fasta_genome_is_accepted_by_annotation_annotate(fasta: Path) -> None:
    """The whole point of the slice: the genome this produces must satisfy the
    released annotate contract's input checks."""
    from organelleverse.annotation.api import annotate

    genome = read_fasta_genome(fasta, organelle="mitochondrion", species=_SPECIES)
    schema = op.parameter_schema("annotation.annotate")
    assert isinstance(schema, dict)
    assert callable(annotate)
    assert genome.metadata.species.strip()
    assert genome.sequence is not None


def test_schema_requires_species_and_organelle() -> None:
    schema = op.parameter_schema("io.read_fasta_genome")
    required = cast(list[str], schema["required"])
    assert "species" in required
    assert "organelle" in required
    assert "path" in required


def test_the_three_invocation_paths_agree(fasta: Path) -> None:
    """Design §8.9. The encodings differ and that is correct: the parameter model
    is strict, so registry.invoke needs a real Path while invoke_json decodes the
    JSON string. They must agree on the resulting object."""
    from organelleverse.operations.adapters.json import invoke_json
    from organelleverse.operations.registry import registry

    direct = read_fasta_genome(fasta, organelle="mitochondrion", species=_SPECIES)
    through_registry = op.invoke(
        "io.read_fasta_genome",
        input=None,
        parameters={"path": fasta, "organelle": "mitochondrion", "species": _SPECIES},
    )
    response = invoke_json(
        {
            "operation_id": "io.read_fasta_genome",
            "input": None,
            "parameters": {
                "path": str(fasta),
                "organelle": "mitochondrion",
                "species": _SPECIES,
            },
        },
        registry=registry,
        granted_side_effects=["read_files"],
    )
    assert response["ok"] is True
    assert isinstance(through_registry, OrganelleGenome)
    assert through_registry.object_id == direct.object_id
    result = cast(dict[str, object], response["result"])
    assert result["object_id"] == direct.object_id


def test_read_files_must_be_granted(fasta: Path) -> None:
    from organelleverse.operations.adapters.json import invoke_json
    from organelleverse.operations.registry import registry

    response = invoke_json(
        {
            "operation_id": "io.read_fasta_genome",
            "input": None,
            "parameters": {
                "path": str(fasta),
                "organelle": "mitochondrion",
                "species": _SPECIES,
            },
        },
        registry=registry,
        granted_side_effects=[],
    )
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "permission.denied"


def test_plastid_type_on_a_non_plastid_genome_is_refused(fasta: Path) -> None:
    """The same rule the reader already applies to metadata.species: a supplied
    metadata field contradicting a required parameter is refused. The canonical
    form for a chloroplast is organelle=plastid with plastid_type=chloroplast, so
    plastid_type under mitochondrion asserts two incompatible things."""
    with pytest.raises(OrganelleInputError) as excinfo:
        read_fasta_genome(
            fasta,
            organelle="mitochondrion",
            species=_SPECIES,
            metadata=OrganelleMetadata(plastid_type="chloroplast"),
        )
    assert excinfo.value.code == "input.metadata_organelle_conflict"


def test_plastid_type_on_a_plastid_genome_is_kept(fasta: Path) -> None:
    genome = read_fasta_genome(
        fasta,
        organelle="plastid",
        species=_SPECIES,
        metadata=OrganelleMetadata(plastid_type="chloroplast"),
    )
    assert genome.metadata.plastid_type == "chloroplast"


_NCBI_GENBANK = """LOCUS       NC_000001                  9 bp    DNA     circular PLN 01-JAN-2020
DEFINITION  Test plastid, complete genome.
ACCESSION   NC_000001
VERSION     NC_000001.2
FEATURES             Location/Qualifiers
     source          1..9
     CDS             1..9
                     /gene="x"
                     /transl_table=11
ORIGIN
        1 atgaaatga
//
"""


@pytest.fixture
def ncbi_genbank(tmp_path: Path) -> Path:
    path = tmp_path / "ncbi.gb"
    path.write_text(_NCBI_GENBANK, encoding="utf-8")
    return path


def test_genbank_supplies_accession_and_transl_table(ncbi_genbank: Path) -> None:
    genome = read_genbank_genome(ncbi_genbank, organelle="plastid", species=_SPECIES)
    assert genome.metadata.accession == "NC_000001.2"
    assert genome.metadata.genetic_code == 11


def test_genbank_without_accession_line_keeps_accession_blank(genbank: Path) -> None:
    """A LOCUS name is not an accession; no CDS means the standard plastid table."""
    genome = read_genbank_genome(genbank, organelle="plastid", species=_SPECIES)
    assert genome.metadata.accession == ""
    assert genome.metadata.genetic_code == 11


def test_fasta_genetic_code_defaults_to_the_organelle_standard(fasta: Path) -> None:
    plastid = read_fasta_genome(fasta, organelle="plastid", species=_SPECIES)
    mito = read_fasta_genome(fasta, organelle="mitochondrion", species=_SPECIES)
    assert plastid.metadata.genetic_code == 11
    assert mito.metadata.genetic_code == 1


def test_stated_genetic_code_is_kept_for_fasta(fasta: Path) -> None:
    genome = read_fasta_genome(
        fasta,
        organelle="plastid",
        species=_SPECIES,
        metadata=OrganelleMetadata(genetic_code=1),
    )
    assert genome.metadata.genetic_code == 1


def test_unversioned_caller_accession_matches_the_file(ncbi_genbank: Path) -> None:
    genome = read_genbank_genome(
        ncbi_genbank,
        organelle="plastid",
        species=_SPECIES,
        metadata=OrganelleMetadata(accession="NC_000001", genetic_code=11),
    )
    assert genome.metadata.accession == "NC_000001.2"


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        (OrganelleMetadata(accession="NC_999999.1"), "input.metadata_accession_conflict"),
        (OrganelleMetadata(genetic_code=1), "input.metadata_genetic_code_conflict"),
    ],
)
def test_caller_metadata_contradicting_the_genbank_is_refused(
    ncbi_genbank: Path, metadata: OrganelleMetadata, code: str
) -> None:
    with pytest.raises(OrganelleInputError) as excinfo:
        read_genbank_genome(ncbi_genbank, organelle="plastid", species=_SPECIES, metadata=metadata)
    assert excinfo.value.code == code


def test_ov_read_fills_genbank_metadata(ncbi_genbank: Path) -> None:
    import organelleverse as ov

    genome = ov.read(ncbi_genbank, organelle="plastid", species=_SPECIES)
    assert isinstance(genome, OrganelleGenome)
    assert (genome.metadata.accession, genome.metadata.genetic_code) == ("NC_000001.2", 11)
