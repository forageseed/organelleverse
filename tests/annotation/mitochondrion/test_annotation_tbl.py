"""Format tests for the NCBI five-column feature table.

Every case here is a way the table can be wrong while still loading: the file
parses, ``table2asn`` accepts it, and the deposited record says something other
than the annotation did. Reading such a file back does not reveal the error,
which is why each is pinned rather than checked by eye.
"""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.tbl import render_tbl
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence


def make_genome() -> GenomeSequence:
    return GenomeSequence(seqid="TEST01", sequence="ACGT" * 500)


def gene(name: str, exons: list[tuple[int, int]], strand: Strand, **kw) -> GeneAnnotation:
    return GeneAnnotation(
        gene_name=name,
        product=kw.pop("product", f"{name} product"),
        exons=[ExonRecord(start=a, end=b, strand=strand, number=n)
               for n, (a, b) in enumerate(exons, 1)],
        strand=strand,
        **kw,
    )


def read_blocks(text: str) -> list[list[str]]:
    """Split a table into feature blocks, each starting at a coordinate line."""
    blocks: list[list[str]] = []
    for line in text.splitlines()[1:]:
        if line and not line.startswith("\t"):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    return blocks


def test_minus_strand_coordinates_are_written_high_to_low(tmp_path) -> None:
    """A minus-strand feature is written 5'->3', so its start exceeds its end.

    Sorting the interval the usual way produces a table that loads and silently
    places the gene on the wrong strand.
    """
    out = tmp_path / "t.tbl"
    render_tbl([gene("cox1", [(100, 400)], Strand.MINUS)], [], [], make_genome(), out)

    coords = [b[0].split("\t")[:2] for b in read_blocks(out.read_text())]
    assert coords, "no features written"
    for start, end in coords:
        assert int(start) > int(end), f"minus strand written {start}..{end}"


def test_exon_order_follows_transcription_not_coordinates(tmp_path) -> None:
    """On the minus strand the last exon in the genome is the first transcribed."""
    out = tmp_path / "t.tbl"
    render_tbl(
        [gene("nad7", [(100, 200), (300, 400), (500, 600)], Strand.MINUS)],
        [], [], make_genome(), out,
    )

    cds = next(b for b in read_blocks(out.read_text()) if b[0].endswith("CDS"))
    spans = [ln.split("\t")[:2] for ln in cds if not ln.startswith("\t\t\t")]
    starts = [int(s) for s, _ in spans]
    assert starts == sorted(starts, reverse=True), f"exons out of transcription order: {starts}"


def test_partial_marks_attach_to_the_features_own_ends(tmp_path) -> None:
    """``<`` marks the 5' end, which on the minus strand is the higher coordinate."""
    out = tmp_path / "t.tbl"
    render_tbl(
        [gene("rps3", [(100, 400)], Strand.MINUS, is_partial_5prime=True)],
        [], [], make_genome(), out,
    )

    first = read_blocks(out.read_text())[0][0].split("\t")
    assert first[0].startswith("<"), f"5' mark missing or misplaced: {first}"
    assert not first[1].startswith(">")


def test_pseudogene_carries_pseudo_and_no_product(tmp_path) -> None:
    """Both a ``/pseudo`` and a ``/product`` on one CDS is a validator error."""
    out = tmp_path / "t.tbl"
    render_tbl(
        [gene("rps19", [(100, 300)], Strand.PLUS, is_pseudo=True)],
        [], [], make_genome(), out,
    )

    cds = next(b for b in read_blocks(out.read_text()) if b[0].endswith("CDS"))
    quals = [ln.strip().split("\t")[0] for ln in cds if ln.startswith("\t\t\t")]
    assert "pseudo" in quals
    assert "product" not in quals


def test_unedited_table_drops_rna_editing_exceptions(tmp_path) -> None:
    """The pair of tables states what the DNA says and what the transcript says.

    C-to-U editing creates initiators and terminators absent from the genome, so
    the unedited table must not claim the exception the edited one does.
    """
    genome = make_genome()
    edited, unedited = tmp_path / "e.tbl", tmp_path / "u.tbl"
    ann = gene("nad4", [(100, 400)], Strand.PLUS, exceptions=["RNA editing"])

    render_tbl([ann], [], [], genome, edited, unedited=False)
    render_tbl([ann], [], [], genome, unedited, unedited=True)

    assert "RNA editing" in edited.read_text()
    assert "RNA editing" not in unedited.read_text()
