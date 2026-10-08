"""Segment renumbering for ODGI, which reads segment names as integers."""

from pathlib import Path

import pytest

from organelleverse.pangenome.graph import load_gfa, path_sequences
from organelleverse.pangenome.odgi_names import numeric_gfa, write_name_table

NAMED = (
    "H\tVN:Z:1.0\tov:Z:organelleverse.graph.v1\n"
    "S\tS1.u0\tACGT\tLN:i:4\tdp:f:10.0\n"
    "S\tS1.u1\tAA\tLN:i:2\n"
    "S\tS1.u2\tTT\n"
    "L\tS1.u0\t+\tS1.u1\t+\t0M\tev:i:7\n"
    "L\tS1.u0\t+\tS1.u2\t-\t0M\n"
    "P\tS1#1#mt\tS1.u0+,S1.u1+\t0M\tTP:Z:circular\n"
    "W\tS1\t2\tmt\t0\t6\t>S1.u0<S1.u2\n"
)


def test_names_become_numbers_and_everything_else_stays(tmp_path: Path) -> None:
    src, dst = tmp_path / "named.gfa", tmp_path / "numeric.gfa"
    src.write_text(NAMED)
    table = numeric_gfa(src, dst)
    assert table == {"1": "S1.u0", "2": "S1.u1", "3": "S1.u2"}
    lines = dst.read_text().splitlines()
    assert lines[0] == "H\tVN:Z:1.0\tov:Z:organelleverse.graph.v1"
    assert lines[1] == "S\t1\tACGT\tLN:i:4\tdp:f:10.0"
    assert lines[4] == "L\t1\t+\t2\t+\t0M\tev:i:7"
    assert lines[5] == "L\t1\t+\t3\t-\t0M"
    assert lines[6] == "P\tS1#1#mt\t1+,2+\t0M\tTP:Z:circular"
    assert lines[7] == "W\tS1\t2\tmt\t0\t6\t>1<3"
    # the same graph: paths spell the same sequence
    assert path_sequences(dst) == path_sequences(src)
    assert set(load_gfa(dst).segments) == {"1", "2", "3"}
    write_name_table(table, tmp_path / "node_names.tsv")
    assert (tmp_path / "node_names.tsv").read_text().splitlines() == [
        "node\tsegment",
        "1\tS1.u0",
        "2\tS1.u1",
        "3\tS1.u2",
    ]


def test_numeric_names_are_left_alone(tmp_path: Path) -> None:
    src, dst = tmp_path / "numeric.gfa", tmp_path / "out.gfa"
    src.write_text("S\t1\tACGT\nS\t2\tAA\nL\t1\t+\t2\t+\t0M\n")
    assert numeric_gfa(src, dst) is None
    assert not dst.exists()


def test_a_link_to_an_unknown_segment_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "bad.gfa"
    src.write_text("S\ts1\tACGT\nL\ts1\t+\ts9\t+\t0M\n")
    with pytest.raises(ValueError, match="s9"):
        numeric_gfa(src, tmp_path / "out.gfa")
