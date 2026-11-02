"""Seed provenance, database contents and whole-genome preservation."""

import importlib.util
import json
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from organelleverse._ovasm_seeddb import ROOT, bundled_seed


@pytest.mark.parametrize("organelle", ["mitochondrion", "plastid"])
def test_bundled_seed_catalog_matches_every_fasta_record(organelle):
    path, info = bundled_seed(organelle)
    database = json.loads((ROOT / "manifest.json").read_text())["databases"][organelle]
    records = list(SeqIO.parse(path, "fasta"))
    entries = {seed["id"]: seed for ref in database["references"] for seed in ref["seeds"]}
    assert len(records) == len(entries) == database["sequence_count"]
    assert {r.id for r in records} == set(entries)
    assert sum(len(r) for r in records) == database["total_bases"]
    assert info["reference_count"] == len(database["references"])
    for record in records:
        assert len(record) == entries[record.id]["length_bp"] >= 100
        assert set(str(record.seq)) <= set("ACGTRYSWKMBDHVN")
    for ref in database["references"]:
        assert "." in ref["accession"]
        assert "Embryophyta" in ref["taxonomy"]
        assert not ref["organism"].startswith(
            ("Arabidopsis thaliana", "Oryza sativa", "Salvia miltiorrhiza")
        )


def test_seed_builder_preserves_genomic_sequence_and_introns(tmp_path):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/assembly_benchmarks/build_ovasm_seeddb.py"
    )
    spec = importlib.util.spec_from_file_location("seed_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = SeqRecord(Seq("A" * 120 + "G" * 60 + "C" * 130), id="NC_999999.1")
    record.annotations.update(molecule_type="DNA", organism="Example plant")
    record.features = [
        SeqFeature(
            SimpleLocation(0, len(record)),
            type="source",
            qualifiers={"organelle": ["mitochondrion"]},
        ),
        SeqFeature(
            CompoundLocation(
                [SimpleLocation(0, 120, strand=-1), SimpleLocation(180, 310, strand=-1)]
            ),
            type="CDS",
            qualifiers={"gene": ["cox1"]},
        ),
    ]
    source = tmp_path / "source.gb"
    SeqIO.write(record, source, "genbank")
    output = tmp_path / "db"
    manifest = module.build_database({"mitochondrion": [source]}, output, "test")
    records = list(SeqIO.parse(output / manifest["databases"]["mitochondrion"]["file"], "fasta"))
    assert [str(r.seq) for r in records] == [str(record.seq)]
    entries = manifest["databases"]["mitochondrion"]["references"][0]["seeds"]
    assert [(s["start_1based"], s["end_1based"], s["strand"]) for s in entries] == [
        (1, 310, 1),
    ]


def test_bundled_gene_db_holds_genuine_genes_of_each_organelle():
    from organelleverse._ovasm_seeddb import bundled_gene_db

    path, info = bundled_gene_db()
    records = list(SeqIO.parse(path, "fasta"))
    assert len(records) == info["proteins"]
    organelles = {r.id.split("|")[0] for r in records}
    assert organelles == {"mitochondrion", "plastid"}
    genes = {o: {r.id.split("|")[1] for r in records if r.id.startswith(o)} for o in organelles}
    assert {"cox1", "nad5", "atp1", "rps12"} <= genes["mitochondrion"]
    assert {"psba", "rbcl", "atpa", "rps12"} <= genes["plastid"]
    # plastid genes annotated in plastid-derived stretches of mitochondria are left out
    assert not {"psba", "rbcl", "peta", "petg", "psbt"} & genes["mitochondrion"]
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert (
        manifest["shared_fraction_gap"]["kept_max"] < manifest["shared_fraction_gap"]["dropped_min"]
    )
    assert not any(
        s["organism"].split()[0] == "Arabidopsis" and not s["excluded"] for s in manifest["sources"]
    )
