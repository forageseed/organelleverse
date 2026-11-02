"""Short genes read through an edited stop; CDS remnants of lost genes become pseudo gene features, broken models of
real genes do not."""

from Bio.SeqFeature import FeatureLocation, SeqFeature

from organelleverse.annotation.plastome import pipeline
from organelleverse.annotation.plastome.cds_refine import refine_cds_boundaries_with_edits

OPEN = "GCT"  # open in frame 0


def test_short_gene_reads_through_an_edited_stop() -> None:
    # 38-codon gene (114 bp, like rpl36) with a genomic TAA at codon 12 that U-to-C editing removes.
    gene = "ATG" + OPEN * 10 + "TAA" + OPEN * 25 + "TAA"
    seq = "CCCCCCCCCC" + gene + "CCCCCCCCCC"
    start, end = 11, 10 + len(gene)
    ns, ne, edits = refine_cds_boundaries_with_edits(seq, start, end, 1, tolerate_edits=True)
    assert (ns, ne) == (start, end)
    assert len(edits) == 1


def cds(gene: str, start: int, end: int, **extra) -> SeqFeature:
    return SeqFeature(FeatureLocation(start, end, strand=1), type="CDS", qualifiers={"gene": [gene], **extra})


def genome() -> str:
    # "TAAC" repeats carry a stop in every frame; positions 1900-2300 are an open stretch (a real gene's transfer).
    g = list("TAAC" * 2500)
    g[1900:2300] = list("GCT" * 133 + "G")
    return "".join(g)


def run(feats, transferred, monkeypatch):
    # median reference CDS: ndhC 363 bp, rps12 372 bp
    monkeypatch.setattr(pipeline, "_reference_protein_map", lambda refs: {"ndhC": ["M" * 120], "rps12": ["M" * 123]})
    return pipeline._retire_remnants(feats, (), transferred, genome())


def test_lost_genes_become_pseudo_and_broken_real_genes_stay(monkeypatch) -> None:
    remnant = cds("ndhC", 1000, 1072)  # transferred 900-1134 has stops in every frame: a lost gene
    broken = cds("ndhC", 2000, 2072)  # transferred 1900-2263 is open: a real gene that refinement cut
    tiny = cds("ndhC", 5000, 5040)  # the transfer itself is a 40 bp scrap
    full = cds("ndhC", 3000, 3363)
    at_end = cds("ndhC", 9930, 9999)
    rps12 = cds("rps12", 6000, 6114, trans_splicing=[""])
    # full-length transfer with stops: a fern gene whose editing went undetected, not a lost gene
    stopped_full = cds("ndhC", 7000, 7072)
    # a 40 bp transfer inside an intact open gene (legume rps2): the full open frame beside it keeps it a CDS
    piece = cds("ndhC", 1950, 1990)
    feats = [remnant, broken, tiny, full, at_end, rps12, stopped_full, piece]
    transferred = {
        id(remnant): FeatureLocation(900, 1134, strand=1),
        id(broken): FeatureLocation(1900, 2263, strand=1),
        id(stopped_full): FeatureLocation(7000, 7363, strand=1),
    }
    assert run(feats, transferred, monkeypatch) == ["ndhC 72 bp", "ndhC 40 bp"]
    assert feats[0].type == "gene" and "pseudo" in feats[0].qualifiers
    assert (int(feats[0].location.start), int(feats[0].location.end)) == (900, 1134)
    assert feats[2].type == "gene"
    assert [f.type for f in (feats[1], *feats[3:])] == ["CDS"] * 6


def test_stops_are_no_evidence_in_an_editing_genome(monkeypatch) -> None:
    remnant = cds("ndhC", 1000, 1072)
    edited = cds("ndhC", 3000, 3363, note=["2 in-frame genomic stop codon(s) read through as U-to-C edited sites"])
    feats = [remnant, edited]
    assert run(feats, {id(remnant): FeatureLocation(900, 1134, strand=1)}, monkeypatch) == []
    assert feats[0].type == "CDS"
