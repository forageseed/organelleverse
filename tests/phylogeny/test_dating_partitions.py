"""Partition serialization checked by Biopython's independent PHYLIP reader."""

import io

import pytest
from Bio import AlignIO

from organelleverse.core.errors import OrganelleInputError
from organelleverse.phylogeny.mcmctree import _partition_alignment


@pytest.mark.parametrize(
    "charsets, expected",
    [
        (r"charset pos12 = 1-12\3 2-12\3; charset pos3 = 3-12\3;", ["ACTAGTCG", "GCAA"]),
        (
            r"charset pos1 = 1-12\3; charset pos2 = 2-12\3; charset pos3 = 3-12\3;",
            ["ATGC", "CATG", "GCAA"],
        ),
        ("charset gene1 = 1-6; charset gene2 = 7-12;", ["ACGTAC", "GTACGA"]),
    ],
)
def test_codon_and_gene_blocks_roundtrip(tmp_path, charsets, expected):
    scheme = tmp_path / "scheme.nex"
    scheme.write_text("#NEXUS\nbegin sets; " + charsets + " end;")
    sequences = [("long_taxon_name", "ACGTACGTACGA"), ("other", "N-?RYMKSWBDH")]
    text, partitions = _partition_alignment(
        sequences, {"long_taxon_name": "T0001", "other": "T0002"}, scheme
    )
    blocks = list(AlignIO.parse(io.StringIO(text), "phylip-relaxed"))
    assert [str(block[0].seq) for block in blocks] == expected
    assert sum(p["n_sites"] for p in partitions) == 12
    assert all([r.id for r in block] == ["T0001", "T0002"] for block in blocks)
    # Ambiguities and gaps must also be retained, in their corresponding columns.
    assert sorted("".join(str(b[1].seq) for b in blocks)) == sorted(sequences[1][1])


@pytest.mark.parametrize(
    "charsets, match",
    [
        ("charset a = 1-3; charset b = 3-4;", "overlap"),
        ("charset a = 1-3;", "missing sites"),
        ("charset a = 0-4;", "outside"),
        ("charset a = 1-5;", "outside"),
        ("charset a = 1-2; charset a = 3-4;", "unique"),
        ("charset a = 3-1;", "nonempty"),
        (r"charset a = 1-4\0;", "Invalid partition"),
        ("charset a = 1-4; charpartition p = HKY:a;", "GTR"),
    ],
)
def test_invalid_partition_contract(tmp_path, charsets, match):
    scheme = tmp_path / "scheme.nex"
    scheme.write_text("#NEXUS\nbegin sets; " + charsets + " end;")
    with pytest.raises(OrganelleInputError, match=match):
        _partition_alignment([("a", "ACGT")], {"a": "T0001"}, scheme)
