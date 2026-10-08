"""Scientific regression gates for the two legacy CDS-PAV entry points."""

from __future__ import annotations

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.pangenome.pangenome import gene_pav
from organelleverse.pangenome.pangenome_core import compute_gene_pav


def _genome(tmp_path, name, genes, *, unnamed=False):
    record = SeqRecord(Seq("ATG" * 30), id=name, name=name, description="PAV fixture")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(index * 3, index * 3 + 3), type="CDS", qualifiers={"gene": [gene]}
        )
        for index, gene in enumerate(genes)
    ]
    record.features.append(
        SeqFeature(FeatureLocation(60, 63), type="gene", qualifiers={"gene": ["gene_feature_only"]})
    )
    if unnamed:
        record.features.append(SeqFeature(FeatureLocation(63, 66), type="CDS"))
    source = tmp_path / f"{name}.gb"
    SeqIO.write([record], source, "genbank")
    return OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(source, kind="annotation", format="genbank"),
        metadata=OrganelleMetadata(accession=name, species=name),
    )


def test_three_sample_cds_classes_are_disjoint_and_entry_points_agree(tmp_path):
    genomes = [
        _genome(tmp_path, "a", ["CORE", "shared", "private", "private"]),
        _genome(tmp_path, "b", ["core", "shared"]),
        _genome(tmp_path, "c", ["core"]),
    ]
    metrics = compute_gene_pav(genomes)
    assert metrics["core"] == metrics["shell"] == metrics["cloud"] == 1
    assert sum(metrics[key] for key in ("core", "shell", "cloud")) == metrics["total_genes"] == 3
    assert metrics["classes"] == {"core": "core", "private": "cloud", "shared": "shell"}
    assert metrics["counts"] == {"core": 3, "private": 1, "shared": 2}
    assert metrics["matrix"] == {"core": [1, 1, 1], "private": [1, 0, 0], "shared": [1, 1, 0]}
    assert metrics["feature_type"] == "CDS"
    assert "gene_feature_only" not in metrics["matrix"]
    assert gene_pav(genomes).model_dump(mode="json")["metrics"] == metrics


@pytest.mark.parametrize("function", [gene_pav, compute_gene_pav])
def test_missing_annotation_is_not_a_gene_absence(tmp_path, function):
    present = _genome(tmp_path, "a", ["core"])
    sequence = tmp_path / "missing.fa"
    sequence.write_text(">missing\nATG\n")
    missing = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(sequence, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(accession="missing"),
    )
    with pytest.raises(OrganelleInputError) as error:
        function([present, missing])
    assert error.value.code == "pangenome.missing_annotation"


def test_supplied_empty_selected_cds_annotation_is_explicit_zero(tmp_path):
    present = _genome(tmp_path, "a", ["core"])
    empty = _genome(tmp_path, "b", [])
    metrics = compute_gene_pav([present, empty])
    assert metrics["matrix"] == {"core": [1, 0]}
    assert metrics["classes"] == {"core": "cloud"}
    assert metrics["shell"] == 0
    assert "not verified biological absence" in metrics["absence_interpretation"]


def test_one_genome_never_counts_core_as_cloud(tmp_path):
    metrics = compute_gene_pav([_genome(tmp_path, "a", ["only"])])
    assert (metrics["core"], metrics["shell"], metrics["cloud"]) == (1, 0, 0)


def test_unnamed_cds_and_zero_genomes_fail_explicitly(tmp_path):
    with pytest.raises(OrganelleInputError) as error:
        compute_gene_pav([_genome(tmp_path, "a", ["named"], unnamed=True)])
    assert error.value.code == "pangenome.unnamed_cds"
    with pytest.raises(OrganelleInputError) as error:
        compute_gene_pav([])
    assert error.value.code == "pangenome.empty_inputs"
