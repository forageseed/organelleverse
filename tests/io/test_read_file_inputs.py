from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.inputs import Graph, Images, ReadAlignments, Tree, Variants, detect_kind
from organelleverse.read_facade import readable_modules
from organelleverse.sequences import Sequences, as_path

VCF = (
    "##fileformat=VCFv4.2\n"
    '##contig=<ID=chr1,length=100>\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
    "chr1\t10\t.\tA\tG\t.\t.\t.\tGT\t0/0\t1/1\n"
)


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    made = {
        "fasta": tmp_path / "x.fasta",
        "vcf": tmp_path / "x.vcf",
        "vcf_gz": tmp_path / "x.vcf.gz",
        "bam": tmp_path / "x.bam",
        "tree": tmp_path / "x.nwk",
        "nexus": tmp_path / "x.nex",
        "gfa": tmp_path / "x.gfa",
        "image": tmp_path / "x.tif",
        "image2": tmp_path / "y.tif",
    }
    made["fasta"].write_text(">a\nACGT\n>b\nACGA\n", encoding="utf-8")
    made["vcf"].write_text(VCF, encoding="utf-8")
    with gzip.open(made["vcf_gz"], "wt", encoding="utf-8") as handle:
        handle.write(VCF)
    made["bam"].write_bytes(gzip.compress(b"BAM\x01" + b"\x00" * 16))
    made["tree"].write_text("((a:1,b:1):1,c:2);\n", encoding="utf-8")
    made["nexus"].write_text("#NEXUS\nbegin trees;\ntree t = ((a,b),c);\nend;\n", encoding="utf-8")
    made["gfa"].write_text("H\tVN:Z:1.0\nS\t1\tACGT\n", encoding="utf-8")
    made["image"].write_bytes(b"II*\x00")
    made["image2"].write_bytes(b"II*\x00")
    return made


def test_every_readable_module_publishes_read() -> None:
    for name in readable_modules():
        module = importlib.import_module(f"organelleverse.{name}")
        assert callable(module.read), name
        assert "read" in module.__all__, name


def test_the_format_comes_from_the_file(files: dict[str, Path]) -> None:
    assert detect_kind(files["fasta"]) == "fasta"
    assert detect_kind(files["vcf_gz"]) == "vcf"
    assert detect_kind(files["bam"]) == "bam"
    assert detect_kind(files["tree"]) == "tree"
    assert detect_kind(files["nexus"]) == "tree"
    assert detect_kind(files["gfa"]) == "gfa"
    assert detect_kind(files["image"]) == "image"


def test_each_module_reads_its_own_formats(files: dict[str, Path]) -> None:
    variants = ov.population.read(files["vcf"])
    assert variants == Variants(files["vcf"], ("s1", "s2"))
    assert ov.population.read(files["vcf_gz"]).samples == ("s1", "s2")
    assert ov.population.read(files["bam"]) == ReadAlignments(files["bam"], "bam")
    assert type(ov.population.read(files["fasta"])) is Sequences

    assert ov.phylogeny.read(files["tree"]) == Tree(files["tree"], "newick")
    assert ov.phylogeny.read(files["nexus"]).format == "nexus"
    assert ov.visualization.read(files["gfa"]) == Graph(files["gfa"])
    assert ov.rna_editing.read(files["bam"]).format == "bam"

    images = ov.morphology.read([files["image"], files["image2"]])
    assert images == Images((files["image"], files["image2"]))
    assert ov.morphology.read(files["image"]).paths == (files["image"],)


def test_ov_read_reaches_every_new_module(files: dict[str, Path]) -> None:
    assert ov.read(files["vcf"], module="population") == ov.population.read(files["vcf"])
    assert ov.read(files["fasta"], module="composition") == ov.composition.read(files["fasta"])
    assert ov.read(files["gfa"], module="pan") == ov.pangenome.read(files["gfa"])


def test_a_module_refuses_a_format_it_does_not_take(files: dict[str, Path]) -> None:
    with pytest.raises(OrganelleInputError) as caught:
        ov.population.read(files["tree"])

    assert caught.value.code == "input.unsupported_read_format"
    assert caught.value.details["found"] == "tree"


def test_a_genome_module_reads_a_genome_with_organelle(files: dict[str, Path]) -> None:
    genome = ov.structure.read(files["fasta"], organelle="mito", species="Oryza sativa")
    assert type(genome) is OrganelleGenome
    assert type(ov.structure.read(files["fasta"])) is Sequences
    assert type(ov.ir_boundary.read(files["fasta"], organelle="chloro", species="x y")) is (
        OrganelleGenome
    )

    with pytest.raises(OrganelleInputError) as only_genome:
        ov.ir_boundary.read(files["fasta"])
    assert only_genome.value.code == "input.unsupported_read_format"

    with pytest.raises(OrganelleInputError) as options:
        ov.structure.read(files["fasta"], species="Oryza sativa")
    assert options.value.code == "input.genome_options_without_organelle"


def test_an_unrecognizable_file_is_named(tmp_path: Path) -> None:
    junk = tmp_path / "junk.dat"
    junk.write_text("not any biological format\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as caught:
        ov.population.read(junk)
    assert caught.value.code == "input.unknown_file_format"

    with pytest.raises(OrganelleInputError) as missing:
        ov.population.read(tmp_path / "absent.vcf")
    assert missing.value.code == "input.missing_artifact"


def test_a_vcf_without_a_header_line_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.vcf"
    bad.write_text("##fileformat=VCFv4.2\nchr1\t10\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as caught:
        ov.population.read(bad)
    assert caught.value.code == "input.invalid_variants"


def test_as_path_unwraps_read_results(files: dict[str, Path]) -> None:
    assert as_path(ov.population.read(files["vcf"])) == files["vcf"]
    assert as_path(ov.morphology.read(files["image"])) == files["image"]
    assert as_path(ov.morphology.read([files["image"], files["image2"]])) == [
        files["image"],
        files["image2"],
    ]
    assert as_path(None) is None
    assert as_path("a.fasta") == "a.fasta"


def test_functions_give_the_same_answer_for_read_results(files: dict[str, Path]) -> None:
    fasta = files["fasta"]
    from_path = ov.composition.gc_content(fasta)
    from_read = ov.composition.gc_content(ov.composition.read(fasta))
    assert from_read.status == from_path.status == "ok"
    assert from_read.metrics == from_path.metrics

    usage_path = ov.codon_composition.codon_usage(fasta)
    usage_read = ov.codon_composition.codon_usage(ov.codon_composition.read(fasta))
    assert usage_read.status == usage_path.status
    assert usage_read.metrics == usage_path.metrics

    populations = {"s1": "A", "s2": "B"}
    fst_path = ov.population.compute_fst(files["vcf"], populations)
    fst_read = ov.population.compute_fst(ov.population.read(files["vcf"]), populations)
    assert fst_read == fst_path

    other = files["tree"].with_name("other.nwk")
    other.write_text("((a:1,c:1):1,b:2);\n", encoding="utf-8")
    compare_path = ov.phylogeny.compare_trees(files["tree"], other)
    compare_read = ov.phylogeny.compare_trees(
        ov.phylogeny.read(files["tree"]), ov.phylogeny.read(other)
    )
    assert compare_read.status == compare_path.status == "ok"
    assert compare_read.metrics == compare_path.metrics
