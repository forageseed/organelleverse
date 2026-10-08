from dataclasses import dataclass
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from organelleverse.annotation import api
from organelleverse.annotation.backends.base import AnnotationRequest, BackendRun
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.orf_annotation import add_orf_features
from organelleverse.annotation.service import run_annotation
from organelleverse.annotation.validation import validate_document
from organelleverse.annotation.writer import load_document_json, materialize_annotation
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import thaw_json
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata


def document(sequence="ATGAAATAACCCATGCCCTAGCCCTTATTTCAT", *, known=True, seqid="r1"):
    features = (
        (
            AnnotationFeature(
                feature_id=f"{seqid}:known",
                seqid=seqid,
                type="CDS",
                operator="single",
                parts=(LocationPart(start=0, end=9, strand=1),),
                parents=(),
                qualifiers=(
                    FeatureQualifier(name="gene", values=("nad1",)),
                    FeatureQualifier(name="translation", values=("MK",)),
                ),
            ),
        )
        if known
        else ()
    )
    return AnnotationDocument(
        backend="mitochondrion",
        requested_stages=("pcg",) if known else (),
        completed_stages=("pcg",) if known else (),
        records=(
            AnnotationRecord(
                seqid=seqid, name=seqid, description="fixture", sequence=sequence, features=features
            ),
        ),
        source_metadata={"species": "fixture"},
    )


def add(doc, tmp_path, **kwargs):
    return add_orf_features(
        doc,
        scratch=tmp_path,
        organelle="mitochondrion",
        genetic_code=kwargs.pop("genetic_code", 1),
        min_aa=kwargs.pop("min_aa", 2),
        circular=kwargs.pop("circular", False),
        **kwargs,
    )


def test_preserves_known_cds_and_adds_original_strand_candidates(tmp_path):
    before = document()
    after = add(before, tmp_path)
    assert after.records[0].features[0] == before.records[0].features[0]
    evidence = thaw_json(after.source_metadata)["orf_candidates"]
    assert evidence["exact_existing_cds_count"] == 1
    new = after.records[0].features[1:]
    assert any(f.parts[0].strand == -1 for f in new)
    assert any(f.extract(after.records[0].sequence) == "ATGCCCTAG" for f in new)
    for feature in new:
        assert feature.qualifier_values("product") == ("hypothetical protein",)
        assert not feature.qualifier_values("gene")
        assert "causality untested" in feature.qualifier_values("note")[0]
    assert validate_document(after, after.requested_stages).valid
    assert before.requested_stages == ("pcg",)


def test_zero_new_candidates_completes_and_materializes(tmp_path):
    after = add(document("ATGAAATAA"), tmp_path)
    assert after.source_metadata["orf_candidates"]["added_count"] == 0
    assert after.completed_stages == ("pcg", "orf")
    assert validate_document(after, after.requested_stages).valid
    materialize_annotation(after, tmp_path / "output")


@pytest.mark.parametrize("reverse", [False, True])
def test_circular_parts_roundtrip_genbank_gff_and_protein(tmp_path, reverse):
    sequence = "AAATAACCCATG"
    if reverse:
        sequence = str(Seq(sequence).reverse_complement())
    doc = add(document(sequence, known=False), tmp_path, circular=True)
    candidates = [f for f in doc.records[0].features if len(f.parts) == 2]
    target = next(f for f in candidates if f.parts[0].strand == (-1 if reverse else 1))
    assert target.extract(sequence) == "ATGAAATAA"
    assert target.qualifier_values("translation") == ("MK",)
    paths = materialize_annotation(doc, tmp_path / "output")
    record = SeqIO.read(paths["genbank"], "genbank")
    assert record.annotations["topology"] == "circular"
    exported = next(
        f
        for f in record.features
        if f.qualifiers.get("locus_tag") == list(target.qualifier_values("locus_tag"))
    )
    assert str(exported.extract(record.seq)) == "ATGAAATAA"
    rows = [
        line.split("\t")
        for line in paths["gff3"].read_text().splitlines()
        if not line.startswith("#") and f"ID={target.feature_id};" in line
    ]
    assert len(rows) == 2
    assert all("Parent=" not in row[8] for row in rows)
    assert all("Note=" in row[8] for row in rows)
    assert [row[7] for row in rows] == ["0", "0"]
    assert (
        dict((r.id, str(r.seq)) for r in SeqIO.parse(paths["protein_fasta"], "fasta"))[
            target.qualifier_values("locus_tag")[0]
        ]
        == "MK"
    )


def test_nonstandard_genetic_code_stop_is_honored(tmp_path):
    doc = add(document("ATGAAAAGA", known=False), tmp_path, genetic_code=2)
    assert any(f.qualifier_values("translation") == ("MK",) for f in doc.records[0].features)
    assert validate_document(doc, doc.requested_stages).valid


@dataclass
class Backend:
    doc: AnnotationDocument
    name: str = "mitochondrion"
    organelle_types: tuple = ("mitochondrion",)

    def run(self, genome, request, scratch):
        return BackendRun(
            document=self.doc, commands=(), software_versions={}, database_hashes={}, logs=()
        )


def test_public_annotation_service_writer_and_provenance(tmp_path, monkeypatch):
    doc = document()
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(f">r1\n{doc.records[0].sequence}\n")
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        api,
        "run_annotation",
        lambda g, req: run_annotation(g, req, adapters={"mitochondrion": Backend(doc)}),
    )
    result = api.annotate(genome, call_trna=False, call_rrna=False, call_orfs=True, orf_min_aa=2)
    assert result.status == "ok"
    assert "sequence_only_orf_candidates" in result.flags
    assert result.metrics["requested_stages"] == ("pcg", "orf")
    assert result.metrics["orf_candidates"]["added_count"] > 0
    assert result.provenance.software_versions["orfipy"] == "0.0.4"
    canonical = next(a for a in result.artifacts if a.kind == "annotation")
    assert load_document_json(canonical.resolve()).completed_stages == ("pcg", "orf")
    exported = api.write(result, output=tmp_path / "published")
    assert exported.status == "ok"
    assert (tmp_path / "published/annotation.gff3").is_file()
    assert (tmp_path / "published/proteins.fasta").is_file()


def test_candidate_orfs_cannot_satisfy_failed_known_gene_stage(tmp_path):
    doc = document(known=False)
    doc = doc.evolve(requested_stages=("pcg",), completed_stages=("pcg",))
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(f">r1\n{doc.records[0].sequence}\n")
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
    )
    request = AnnotationRequest(
        backend="mitochondrion",
        workspace=tmp_path / "work",
        threads=1,
        stages=("pcg",),
        call_orfs=True,
        orf_min_aa=2,
    )
    result = run_annotation(genome, request, adapters={"mitochondrion": Backend(doc)})
    assert result.status == "failed"
    assert result.errors[0].code == "annotation_validation_failed"
    assert not result.artifacts


@pytest.mark.parametrize("invalid", [0, True, 2.5])
def test_request_rejects_invalid_orf_minimum(tmp_path, invalid):
    with pytest.raises(ValueError, match="orf_min_aa"):
        AnnotationRequest(
            backend="mitochondrion",
            workspace=tmp_path,
            threads=1,
            stages=("pcg",),
            orf_min_aa=invalid,
        )


def test_real_genbank_orfs_preserve_annotation_and_recover_target(tmp_path):
    from organelleverse.annotation.genbank import parse_genbank

    data = Path(__file__).parents[1] / "phenotype/data"
    for accession, target in [("D14339.1", "orf79"), ("Z18896.1", "orf138")]:
        original = parse_genbank(data / f"{accession}.gb")
        # Mimic a known-gene backend that has not annotated the CMS ORF;
        # gene prediction itself is not benchmarked by this integration test.
        metadata = next(
            f
            for f in original.records[0].features
            if f.type == "CDS" and f.qualifier_values("product")[0].lower() == target
        )
        target_protein = metadata.qualifier_values("translation")[0]
        record = original.records[0]
        retained = tuple(f for f in record.features if f.feature_id != metadata.feature_id)
        before = original.evolve(records=(record.evolve(features=retained),))
        after = add(before, tmp_path, min_aa=30)
        assert after.records[0].features[: len(retained)] == retained
        candidate = next(
            f
            for f in after.records[0].features[len(retained) :]
            if f.qualifier_values("translation") == (target_protein,)
        )
        assert (
            str(Seq(candidate.extract(record.sequence)).translate(to_stop=True)) == target_protein
        )
        assert tuple((p.start, p.end, p.strand) for p in candidate.parts) == tuple(
            (p.start, p.end, p.strand) for p in metadata.parts
        )
