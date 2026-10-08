"""Exercise the real command construction and both search parsers with fake runners."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from organelleverse import _losat
from organelleverse.annotation.mitochondrion import boundary, rrna, trans_splicing, trna
from organelleverse.annotation.mitochondrion.models.gene import ExonRecord, GeneAnnotation, Strand
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.core.errors import OrganelleDependencyError


@pytest.fixture(params=["losat", "ncbi"])
def search_backend(request, monkeypatch):
    """Resolve real environment overrides; emulate only the process boundary."""
    mode = request.param
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "/fake/LOSAT" if mode == "losat" else "ncbi")
    monkeypatch.setattr(
        _losat.shutil,
        "which",
        lambda name: (
            "/fake/LOSAT"
            if name == "/fake/LOSAT"
            else f"/fake/{name}"
            if mode == "ncbi" and name in {"blastn", "tblastn", "makeblastdb"}
            else None
        ),
    )
    calls = []
    raw_rows = []
    ncbi_rows = []

    def execute(argv, **kwargs):
        calls.append(tuple(argv))
        if Path(argv[0]).name == "LOSAT":
            assert "-s" in argv and "--outfmt" not in argv
            assert Path(argv[argv.index("-s") + 1]).is_file()
            return SimpleNamespace(stdout="\n".join("\t".join(r) for r in raw_rows))
        if "-outfmt" in argv:
            Path(argv[argv.index("-out") + 1]).write_text(
                "\n".join("\t".join(r) for r in ncbi_rows)
            )
        return SimpleNamespace(stdout="")

    for module in (_losat, boundary, trans_splicing, rrna, trna):
        monkeypatch.setattr(module, "run_external", execute)
    tools = (
        {"losat": "/fake/LOSAT"}
        if mode == "losat"
        else {name: f"/fake/{name}" for name in ("blastn", "tblastn", "makeblastdb")}
    )
    return SimpleNamespace(
        mode=mode,
        calls=calls,
        raw=raw_rows,
        ncbi=ncbi_rows,
        tools=tools,
        runner=SimpleNamespace(run=execute),
    )


def row(query, qstart, qend, sstart, send):
    return [
        query,
        "genome",
        "95",
        str(abs(qend - qstart) + 1),
        "0",
        "0",
        str(qstart),
        str(qend),
        str(sstart),
        str(send),
        "1e-30",
        "200",
    ]


def test_boundary_search(search_backend, tmp_path):
    b = search_backend
    (tmp_path / "atp1.Protein.fasta").write_text(">ref\n" + "M" * 100 + "\n")
    b.raw.append(row("ref", 1, 100, 100, 399))
    b.ncbi.append(["ref", "genome", "100", "399", "1e-30", "200", "95", "100"])
    genome = GenomeSequence(seqid="genome", sequence="A" * 1000)
    ann = GeneAnnotation(
        gene_name="atp1",
        gene_type="CDS",
        strand=Strand.PLUS,
        exons=[ExonRecord(start=110, end=389, strand=Strand.PLUS, number=1)],
    )
    result = boundary._refine_boundary_by_tblastn(
        ann,
        genome,
        SimpleNamespace(blast_ref_dir=tmp_path),
        tool_paths=b.tools,
        command_runner=b.runner,
    )
    assert (result.genomic_start, result.genomic_end) == (100, 399)
    assert result.source_method == "tblastn"
    assert len(b.calls) == (1 if b.mode == "losat" else 2)
    if b.mode == "losat":
        assert b.calls[0][1] == "tblastn"
        assert b.calls[0][b.calls[0].index("--db-gencode") + 1] == "1"


def test_short_exon_detection(search_backend, tmp_path):
    b = search_backend
    (tmp_path / "nad5.CDS.fasta").write_text(">ref\n" + "A" * 30 + "\n")
    b.raw.append(row("ref", 1, 30, 330, 301))
    b.ncbi.append(["ref", "genome", "330", "301", "1e-30", "200", "30", "95"])
    genome = GenomeSequence(seqid="genome", sequence="A" * 1000)
    ann = GeneAnnotation(
        gene_name="nad5",
        gene_type="CDS",
        strand=Strand.PLUS,
        exons=[ExonRecord(start=10, end=100, strand=Strand.PLUS, number=1)],
    )
    result = trans_splicing.detect_short_exons(
        genome, SimpleNamespace(exon_ref_dir=tmp_path), {"nad5": ann}
    )
    assert [(e.start, e.end, e.strand) for e in result["nad5"].exons] == [
        (10, 100, Strand.PLUS),
        (301, 330, Strand.MINUS),
    ]
    assert len(b.calls) == 1
    flag = "--task" if b.mode == "losat" else "-task"
    assert b.calls[0][b.calls[0].index(flag) + 1] == "blastn-short"
    if b.mode == "losat":
        assert b.calls[0][b.calls[0].index("--word-size") + 1] == "7"


def test_exon_search(search_backend, tmp_path):
    b = search_backend
    ref = tmp_path / "nad5.fasta"
    ref.write_text(">ref_nad5_3_30\n" + "A" * 30 + "\n")
    b.raw.append(row("ref_nad5_3_30", 1, 30, 330, 301))
    b.ncbi.append(["ref_nad5_3_30", "genome", "330", "301", "1e-30", "200", "30", "95"])
    result = trans_splicing.search_exons_blastn(
        "nad5",
        GenomeSequence(seqid="genome", sequence="A" * 1000),
        ref,
        b.tools.get("blastn"),
        tool_paths=b.tools,
        command_runner=b.runner,
    )
    assert result == {3: [(301, 330, Strand.MINUS, 95.0, 30, 30)]}
    assert len(b.calls) == 1
    flag = "--word-size" if b.mode == "losat" else "-word_size"
    assert b.calls[0][b.calls[0].index(flag) + 1] == "7"


def test_trans_spliced_entrypoint(search_backend, monkeypatch, tmp_path):
    b = search_backend
    ref = tmp_path / "nad5.CDS.Exons.Extent.fasta"
    ref.write_text(">ref_nad5_3_30\n" + "A" * 30 + "\n")
    monkeypatch.setattr(
        trans_splicing,
        "get_dynamic_trans_spliced_config",
        lambda n: {
            "nad5": {
                "expected_exons": 5,
                "min_total_length": 100,
                "max_genomic_span": 10000,
            }
        },
    )
    result = trans_splicing.annotate_trans_spliced_genes(
        GenomeSequence(seqid="genome", sequence="A" * 1000),
        SimpleNamespace(exon_ref_dir=tmp_path, blast_ref_dir=tmp_path),
        {},
        tool_paths=b.tools,
        command_runner=b.runner,
    )
    assert result == {}
    assert len(b.calls) == 1


@pytest.mark.parametrize("kind", ["rrna", "trna"])
def test_rna_search_coverage_and_reverse_strand(search_backend, tmp_path, kind):
    b = search_backend
    module = rrna if kind == "rrna" else trna
    query = "rrn5" if kind == "rrna" else "trnF-GAA"
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / f"{query}.fasta").write_text(f">{query}\n" + "A" * 100 + "\n")
    genome = tmp_path / "genome.fasta"
    genome.write_text(">genome\n" + "A" * 1000 + "\n")
    # Out-of-order disjoint HSPs jointly cover 60%; either HSP alone fails 50%.
    b.raw.extend([row(query, 61, 90, 390, 361), row(query, 1, 30, 300, 271)])
    b.ncbi.extend(
        [[query, "genome", r[6], r[7], r[8], r[9], r[10], r[11], r[3], r[2], "60"] for r in b.raw]
    )
    search = getattr(module, f"_{kind}_by_blastn")
    result = search(genome, tmp_path, SimpleNamespace(**{f"{kind}_ref_dir": refs}))
    assert [(h.start, h.end, h.strand) for h in result] == [(361, 390, -1), (271, 300, -1)]
    assert len(b.calls) == (1 if b.mode == "losat" else 2)


@pytest.mark.parametrize("kind", ["boundary", "short", "exons", "trans"])
def test_search_dependencies_fail_clearly(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    monkeypatch.setattr(_losat.shutil, "which", lambda name: None)
    (tmp_path / "rrn5.fasta").write_text(">rrn5\nAAAA\n")
    genome = GenomeSequence(seqid="genome", sequence="A" * 1000)
    db = SimpleNamespace(rrna_ref_dir=tmp_path, trna_ref_dir=tmp_path, blast_ref_dir=tmp_path)
    with pytest.raises(OrganelleDependencyError, match="Install LOSAT or NCBI BLAST"):
        if kind == "boundary":
            ann = GeneAnnotation(
                gene_name="atp1",
                gene_type="CDS",
                strand=Strand.PLUS,
                exons=[ExonRecord(start=1, end=99, strand=Strand.PLUS, number=1)],
            )
            boundary._refine_boundary_by_tblastn(ann, genome, db)
        elif kind == "short":
            trans_splicing.detect_short_exons(genome, db, {})
        elif kind == "exons":
            trans_splicing.search_exons_blastn("nad5", genome, tmp_path / "rrn5.fasta", None)
        else:
            trans_splicing.annotate_trans_spliced_genes(genome, db, {})


@pytest.mark.parametrize("kind", ["rrna", "trna"])
def test_rna_searches_fall_back_to_python_alignment_without_binaries(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    monkeypatch.setattr(_losat.shutil, "which", lambda name: None)
    module = rrna if kind == "rrna" else trna
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    sentinel = [object()]
    monkeypatch.setattr(module, f"_{kind}_python_fallback", lambda *args: sentinel)
    (tmp_path / "rrn5.fasta").write_text(">rrn5\nAAAA\n")
    db = SimpleNamespace(rrna_ref_dir=tmp_path, trna_ref_dir=tmp_path)
    result = getattr(module, f"_{kind}_by_blastn")(tmp_path / "g.fasta", tmp_path, db)
    assert result is sentinel
