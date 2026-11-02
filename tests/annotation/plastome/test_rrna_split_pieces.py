"""Native plastid rRNA: a lineage-specific insertion must not split one rRNA into pieces."""

from Bio import SeqIO

from organelleverse.annotation.plastome import rrna_native
from organelleverse.annotation.plastome.rrna_native import _join_split_pieces, _Piece, annotate_plastid_rrna


def test_welwitschia_5s_is_one_feature_per_copy():
    # Welwitschia 5S is 148 nt against 119-121 nt queries; nhmmer splits it at the insertion
    # (query 2-60 / 35-121). Before the fix each IR copy gave a 59 nt and an 87 nt fragment.
    gb = rrna_native.default_rrna_ref_dir().parent / "references" / "Welwitschia_mirabilis_chloroplast.gb"
    record = SeqIO.read(gb, "genbank")
    fives = [(h.start, h.end, h.strand) for h in annotate_plastid_rrna(str(record.seq)) if h.rrna_type == "5S"]
    # RefSeq NC_010654.1: rrn5 86911..87058 and complement(101225..101372)
    assert fives == [(86911, 87058, 1), (101225, 101372, -1)]


def piece(start, end, strand, q_from, q_to, score, index):
    return _Piece(start=start, end=end, strand=strand, q_from=q_from, q_to=q_to, q_length=121, scores={index: score})


def test_pieces_join_only_when_they_continue_each_other():
    first, second = piece(100, 160, 1, 2, 60, 30.0, 0), piece(161, 247, 1, 35, 121, 60.0, 1)
    (joined,) = _join_split_pieces([second, first])
    assert (joined.start, joined.end, joined.score) == (100, 247, 90.0)
    # minus strand: transcription order runs towards lower coordinates
    (joined,) = _join_split_pieces([piece(188, 247, -1, 2, 60, 30.0, 0), piece(100, 186, -1, 35, 121, 60.0, 1)])
    assert (joined.start, joined.end) == (100, 247)
    # two complete copies, a far gap, the wrong query order or a too long span stay apart
    assert len(_join_split_pieces([piece(100, 220, 1, 1, 121, 99.0, 0), piece(225, 345, 1, 1, 121, 99.0, 1)])) == 2
    assert len(_join_split_pieces([first, piece(300, 386, 1, 35, 121, 60.0, 1)])) == 2
    assert len(_join_split_pieces([piece(100, 186, 1, 35, 121, 60.0, 0), piece(187, 247, 1, 2, 60, 30.0, 1)])) == 2
    assert len(_join_split_pieces([piece(100, 160, 1, 2, 60, 30.0, 0), piece(161, 300, 1, 35, 121, 60.0, 1)])) == 2
