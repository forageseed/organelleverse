"""Biological-coordinate and deferred-render contracts for OGDraw extensions."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation

import organelleverse as ov
from organelleverse.capabilities.adapters.visualization import _STRUCTURE_GENBANK
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.visualization import plot_structure_map
from organelleverse.visualization.ogdraw import (
    GENE_COLORS,
    Gene,
    _draw_gene_blocks,
    draw_mito_map,
    parse_genbank,
)
from organelleverse.visualization.structure_maps import _draw_exon_panel


@pytest.fixture
def genbank(tmp_path):
    path = tmp_path / "sample.gb"
    path.write_text(_STRUCTURE_GENBANK.content)
    return path


@pytest.fixture
def multi_genbank(tmp_path, genbank):
    record = SeqIO.read(genbank, "genbank")
    second = record[:]
    second.id = second.name = "ctgB"
    path = tmp_path / "multi.gb"
    SeqIO.write([record, second], path, "genbank")
    return path


def test_multi_record_draws_one_circle_per_contig_plus_standalone_figures(
    multi_genbank, tmp_path
):
    plot = plot_structure_map(multi_genbank, dpi=60)
    metrics = plot.metrics
    assert metrics["contig_count"] == 2
    assert [contig["name"] for contig in metrics["contigs"]] == ["fixture", "ctgB"]
    assert all("contig" in row for row in metrics["features"])
    # feature indices are renumbered globally so SVG gids stay unique.
    indexes = [row["feature_index"] for row in metrics["features"]]
    assert len(indexes) == len(set(indexes))
    result = ov.write(plot, tmp_path / "map.png")
    names = {Path(a.uri).name for a in result.artifacts}
    # combined figure, one standalone figure per contig, exon tables for both.
    assert {"map.png", "map.fixture.png", "map.ctgB.png", "map.exons.tsv"} <= names
    rows = list(csv.DictReader((tmp_path / "map.exons.tsv").open(), delimiter="\t"))
    assert rows[0]["contig"] == "fixture"
    assert {row["contig"] for row in rows} == {"fixture", "ctgB"}


def test_multi_record_rejects_ir_and_repeat_tracks(multi_genbank):
    with pytest.raises(OrganelleParameterError, match="single-record"):
        plot_structure_map(multi_genbank, show_ir=True)
    with pytest.raises(OrganelleParameterError, match="Repeat tracks"):
        plot_structure_map(multi_genbank, ssrs=[{"start": 1, "end": 10}])


def test_multi_record_serialization_keeps_the_whole_bundle(multi_genbank, tmp_path):
    from organelleverse.core.result import OrganelleResult

    original = plot_structure_map(multi_genbank, dpi=60)
    serialized = OrganelleResult.model_validate_json(original.model_dump_json())
    ov.write(original, tmp_path / "original.png")
    ov.write(serialized, tmp_path / "restored.png")
    assert (tmp_path / "original.png").read_bytes() == (tmp_path / "restored.png").read_bytes()
    assert (tmp_path / "original.fixture.png").read_bytes() == (
        tmp_path / "restored.fixture.png"
    ).read_bytes()


def test_coordinates_copies_strands_and_parent_qualifier(genbank):
    record = SeqIO.read(genbank, "genbank")
    record.features.append(
        SeqFeature(
            CompoundLocation(
                [SimpleLocation(700, 720, strand=-1), SimpleLocation(650, 660, strand=-1)]
            ),
            type="CDS",
            qualifiers={"gene": ["ycf3"]},
        )
    )
    SeqIO.write(record, genbank, "genbank")
    plot = plot_structure_map(genbank)
    features = plot.metrics["features"]
    assert len(features) == 4  # CDS supersedes gene span; both ycf3 copies survive.
    ycf = [r for r in features if r["gene"] == "ycf3"]
    assert len(ycf) == 2
    assert [(e["start"], e["end"]) for e in ycf[0]["exons"]] == [(101, 150), (201, 220), (251, 300)]
    trans = next(r for r in features if r["gene"] == "rps12")
    assert trans["splicing"] == "trans"  # qualifier is only on the parent gene
    assert [e["strand"] for e in trans["exons"]] == [-1, 1]
    assert plot.scope == "plastid"


@pytest.mark.parametrize("strand", [1, -1])
def test_circular_origin_join_is_not_an_intron(genbank, strand):
    record = SeqIO.read(genbank, "genbank")
    parts = [SimpleLocation(990, 1000, strand=strand), SimpleLocation(0, 10, strand=strand)]
    if strand == -1:
        parts.reverse()
    record.features.append(
        SeqFeature(CompoundLocation(parts), type="CDS", qualifiers={"gene": ["origin"]})
    )
    SeqIO.write(record, genbank, "genbank")
    row = plot_structure_map(genbank).metrics["features"][-1]
    assert row["splicing"] == "none"
    assert row["gaps"] == ()


def test_large_cis_intron_not_reclassified_by_distance(genbank):
    record = SeqIO.read(genbank, "genbank")
    record.seq = record.seq * 40
    record.features.append(
        SeqFeature(
            CompoundLocation(
                [SimpleLocation(1, 10, strand=1), SimpleLocation(35000, 35100, strand=1)]
            ),
            type="CDS",
            qualifiers={"gene": ["cis"]},
        )
    )
    SeqIO.write(record, genbank, "genbank")
    assert plot_structure_map(genbank).metrics["features"][-1]["splicing"] == "cis"


def test_actual_exon_rectangle_widths(genbank):
    row = plot_structure_map(genbank).metrics["features"][0]
    fig, ax = plt.subplots()
    try:
        _draw_exon_panel(ax, row)
        assert [p.get_width() for p in ax.patches] == [50, 20, 50]
        assert [p.get_x() for p in ax.patches] == [0, 100, 150]
        assert [p.get_gid() for p in ax.patches] == ["exon-2-1", "exon-2-2", "exon-2-3"]
    finally:
        plt.close(fig)


def test_mixed_strand_circle_blocks_have_exact_inclusive_lengths():
    gene = Gene(
        "rps12",
        101,
        401,
        -1,
        True,
        [(101, 110), (401, 401)],
        "ribo_SSU",
        trans_spliced=True,
        exon_strands=[-1, 1],
    )
    fig, ax = plt.subplots()
    try:
        _draw_gene_blocks(ax, gene, GENE_COLORS, 1000, 0.820, 0.857)
        assert [p.r for p in ax.patches] == [0.857, 0.90]
        assert [p.theta2 - p.theta1 for p in ax.patches] == pytest.approx([3.6, 0.36])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("suffix", ["svg", "png"])
def test_write_declares_image_and_exact_exon_table(genbank, tmp_path, suffix):
    before = set(tmp_path.iterdir())
    plot = plot_structure_map(
        genbank,
        ssrs=[{"start": 1, "end": 10}],
        tandem_repeats=[{"start": 20, "end": 40}],
        dispersed_repeats=[{"positions": [70, 900], "length": 20}],
        dpi=60,
    )
    assert set(tmp_path.iterdir()) == before
    assert plot.artifacts == ()
    out = tmp_path / f"map.{suffix}"
    result = ov.write(plot, out)
    assert {Path(a.uri).name for a in result.artifacts} == {out.name, "map.exons.tsv"}
    with (tmp_path / "map.exons.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    expected = [
        (str(r["feature_index"]), str(e["exon"]), str(e["start"]), str(e["end"]), str(e["strand"]))
        for r in plot.metrics["features"]
        for e in r["exons"]
    ]
    assert [
        (r["feature_index"], r["exon"], r["start"], r["end"], r["strand"]) for r in rows
    ] == expected
    if suffix == "svg":
        ids = {node.attrib.get("id") for node in ET.parse(out).iter()}
        assert {f"exon-{r[0]}-{r[1]}" for r in expected} <= ids
        assert "splice-4-1-2" in ids
    assert [len(t["intervals"]) for t in plot.metrics["tracks"]] == [1, 1, 2]


def test_explicit_ir_failure_and_invalid_tracks(genbank):
    # This tiny synthetic sequence cannot supply a quadripartite plastome.
    with pytest.raises(OrganelleInputError, match="quadripartite"):
        plot_structure_map(genbank, show_ir=True)
    with pytest.raises(OrganelleParameterError, match="1-based"):
        plot_structure_map(genbank, ssrs=[{"start": 0, "end": 3}])


def test_legacy_ogdraw_still_renders(genbank, tmp_path):
    parsed = parse_genbank(genbank)
    out = tmp_path / "legacy.svg"
    assert draw_mito_map(parsed, output_file=out, dpi=60) == out
    assert out.stat().st_size > 1000


def test_serialized_result_keeps_render_data_after_source_is_removed(genbank, tmp_path):
    from organelleverse.core.result import OrganelleResult

    original = plot_structure_map(genbank, dpi=60)
    serialized = OrganelleResult.model_validate_json(original.model_dump_json())
    genbank.unlink()
    ov.write(original, tmp_path / "original.png")
    ov.write(serialized, tmp_path / "restored.png")
    assert (tmp_path / "original.png").read_bytes() == (tmp_path / "restored.png").read_bytes()
    assert (tmp_path / "original.exons.tsv").read_bytes() == (
        tmp_path / "restored.exons.tsv"
    ).read_bytes()
