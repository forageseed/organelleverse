"""Several GenBank files (or one multi-record file) drawn on one figure."""

from __future__ import annotations

import pytest
from Bio import SeqIO

import organelleverse as ov
from organelleverse.capabilities.adapters.visualization import _STRUCTURE_GENBANK
from organelleverse.visualization import plot_structure_map
from organelleverse.visualization.ogdraw import (
    _genbank_display_name,
    _load_genbank_records,
    plot_ogdraw_map,
)
from organelleverse.visualization.plots import plot_genome_map


@pytest.fixture
def two_files(tmp_path):
    first = tmp_path / "a.gb"
    first.write_text(_STRUCTURE_GENBANK.content)
    record = SeqIO.read(first, "genbank")
    second = record[:]
    second.id = second.name = "other"
    other = tmp_path / "b.gb"
    SeqIO.write([second], other, "genbank")
    return first, other


def test_records_of_several_files_are_loaded_in_order(two_files):
    loaded = _load_genbank_records(list(two_files))
    assert [name for name, _ in loaded] == ["fixture", "other"]


def test_a_repeated_record_name_is_prefixed_with_its_file_stem(two_files, tmp_path):
    first, _ = two_files
    copy = tmp_path / "copy.gb"
    copy.write_text(first.read_text())
    names = [name for name, _ in _load_genbank_records([first, copy])]
    assert names == ["a:fixture", "copy:fixture"]


def test_a_list_of_one_file_behaves_like_the_file(two_files):
    first, _ = two_files
    assert [n for n, _ in _load_genbank_records([first])] == ["fixture"]
    assert _genbank_display_name([first]) == "a"


def test_display_name_counts_several_files(two_files):
    assert _genbank_display_name(list(two_files)) == "2 GenBank files"


def test_an_ogdraw_map_of_two_files_is_one_png(two_files, tmp_path):
    plot = plot_ogdraw_map(list(two_files), dpi=50)
    out = tmp_path / "both.png"
    ov.write(plot, output=out)
    assert out.exists() and out.stat().st_size > 5000


def test_a_genome_map_of_two_files_is_one_png(two_files, tmp_path):
    plot = plot_genome_map(list(two_files), dpi=50)
    out = tmp_path / "genome.png"
    ov.write(plot, output=out)
    assert out.exists()


def test_gbdraw_refuses_several_files(two_files, tmp_path):
    plot = plot_genome_map(list(two_files), method="gbdraw")
    with pytest.raises(Exception, match="one GenBank file"):
        ov.write(plot, output=tmp_path / "x.png")


def test_a_structure_map_of_two_files_has_a_contig_per_file(two_files):
    plot = plot_structure_map(list(two_files), dpi=50)
    assert plot.metrics["contig_count"] == 2
    assert [c["name"] for c in plot.metrics["contigs"]] == ["fixture", "other"]
