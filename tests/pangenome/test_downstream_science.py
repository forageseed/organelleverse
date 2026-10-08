from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.pangenome.annotation_projection import (
    Annotation,
    annotation_tables,
    normalize_annotations,
    project_annotations,
    read_bed,
    read_genbank,
    read_gff,
)
from organelleverse.pangenome.phylogeny import jaccard_distances, node_pav_tree
from organelleverse.pangenome.visualization import write_report


def test_jaccard_exact_empty_sets_and_nonbinary_rejection():
    distances = jaccard_distances(["a", "b", "c", "d"], [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert distances == [[0, 0.5, 1, 1], [0.5, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    with pytest.raises(ValueError, match="binary"):
        jaccard_distances(["a", "b"], [[2, 1]])
    with pytest.raises(ValueError, match="uniquely"):
        jaccard_distances(["a", "a"], [[1, 1]])
    with pytest.raises(ValueError, match="nonempty"):
        jaccard_distances(["a", "b"], [])


def test_upgma_exact_distances_branch_lengths_and_quoted_labels():
    paths = ["a one", "b#1", "c", "d"]
    matrix = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    result = node_pav_tree(paths, matrix)
    tree = Phylo.read(io.StringIO(result["newick"]), "newick")
    assert {tip.name for tip in tree.get_terminals()} == set(paths)
    assert tree.distance("a one", "b#1") == 0
    assert tree.distance("a one", "c") == pytest.approx(0.8)
    assert sorted(clade.branch_length for clade in tree.root.clades) == [0.4, 0.4]
    assert {tuple(row["paths"]) for row in result["clades"]} == {("a one", "b#1"), ("c", "d")}
    assert all(row["support"] is None for row in result["clades"])


def test_bootstrap_resamples_node_rows_reproducibly_and_records_fraction():
    matrix = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    result = node_pav_tree(["a", "b", "c", "d"], matrix, bootstrap_replicates=100, seed=13)
    assert result == node_pav_tree(["a", "b", "c", "d"], matrix, bootstrap_replicates=100, seed=13)
    assert result["resampling_unit"] == "graph_node"
    assert all(row["support"] == row["replicate_count"] / 100 for row in result["clades"])
    # Check replicate counts against independently re-running non-bootstrap trees.
    rng = np.random.default_rng(13)
    expected = {tuple(row["paths"]): 0 for row in result["clades"]}
    for _ in range(100):
        sampled = [matrix[index] for index in rng.integers(0, 5, size=5)]
        clades = {
            tuple(row["paths"]) for row in node_pav_tree(["a", "b", "c", "d"], sampled)["clades"]
        }
        for key in expected:
            expected[key] += int(key in clades)
    assert {tuple(row["paths"]): row["replicate_count"] for row in result["clades"]} == expected
    with pytest.raises(ValueError, match="nonnegative integer"):
        node_pav_tree(["a", "b"], [[1, 0]], bootstrap_replicates=-1)


def test_exact_projection_reverse_node_and_compound_feature():
    annotation = Annotation("s#1#mt", "nad1", "locus1", ((8, 14),), "+")
    steps = {
        "s#1#mt": [
            {"node": "n1", "orientation": "+", "start": 0, "end": 10, "node_length": 10},
            {"node": "n2", "orientation": "-", "start": 10, "end": 16, "node_length": 6},
        ]
    }
    fragments = project_annotations([annotation], steps)
    assert [
        (row["node"], row["node_start"], row["node_end"], row["strand"]) for row in fragments
    ] == [("n1", 8, 10, "+"), ("n2", 2, 6, "-")]
    compound = Annotation("s#1#mt", "nad2", "locus2", ((0, 3), (13, 16)), "-")
    tables = annotation_tables([annotation, compound], {"s#1#mt": (0, 16)}, bin_size=10)
    assert tables["copy_matrix"] == [[1], [1]]
    assert len(tables["gene_arrows"]) == 3
    assert tables["bin_coverage"] == [
        {"path": "s#1#mt", "start": 0, "end": 10, "annotated_bp": 5, "annotated_fraction": 0.5},
        {"path": "s#1#mt", "start": 10, "end": 16, "annotated_bp": 6, "annotated_fraction": 1.0},
    ]


def test_projection_overlap_spans_and_missing_coordinates_fail():
    steps = {
        "p": [
            {"node": "a", "orientation": "+", "start": 0, "end": 10},
            {"node": "b", "orientation": "+", "start": 8, "end": 18},
        ]
    }
    fragments = project_annotations([Annotation("p", "g", "l", ((9, 11),))], steps)
    assert [(row["node_start"], row["node_end"]) for row in fragments] == [(9, 10), (1, 3)]
    with pytest.raises(ValueError, match="unrepresented"):
        project_annotations([Annotation("p", "g", "l", ((17, 20),))], steps)
    with pytest.raises(ValueError, match="absent"):
        project_annotations([Annotation("other", "g", "l", ((0, 1),))], steps)


def test_bed_copy_numbers_group_molecules_and_synonyms(tmp_path):
    bed = tmp_path / "genes.bed"
    bed.write_text("p1\t0\t4\tNAD1\t0\t+\np2\t0\t4\tnad1\t0\t-\n")
    annotations = normalize_annotations(read_bed(bed), gene_synonyms={"NAD1": "nad1"})
    result = annotation_tables(
        annotations,
        {"p1": (0, 5), "p2": (0, 5), "p3": (0, 5)},
        path_samples={"p1": "s1", "p2": "s1", "p3": "s2"},
    )
    assert result["copy_matrix"] == [[2, 0]]
    assert result["pav_matrix"] == [[1, 0]]
    assert "not validated biological absence" in result["interpretation"]


def test_gff_one_based_and_multipart_locus(tmp_path):
    path = tmp_path / "genes.gff3"
    path.write_text(
        "##gff-version 3\np\tx\tgene\t1\t3\t.\t-\t.\tID=g1;Name=nad%201\n"
        "p\tx\tgene\t9\t10\t.\t-\t.\tID=g1;Name=nad%201\n"
    )
    annotations = read_gff(path)
    assert annotations == [Annotation("p", "nad 1", "g1", ((0, 3), (8, 10)), "-")]
    assert annotation_tables(annotations, {"p": (0, 10)})["copy_matrix"] == [[1]]


def test_genbank_origin_spanning_feature_preserves_parts(tmp_path):
    record = SeqRecord(Seq("A" * 20), id="mt", name="mt", description="fixture")
    record.annotations["molecule_type"] = "DNA"
    location = CompoundLocation(
        [FeatureLocation(16, 20, strand=1), FeatureLocation(0, 4, strand=1)]
    )
    record.features = [
        SeqFeature(location, type="gene", qualifiers={"gene": ["nad1"], "locus_tag": ["g1"]})
    ]
    path = tmp_path / "origin.gb"
    SeqIO.write([record], path, "genbank")
    result = read_genbank(path)
    assert result == [
        Annotation("mt", "nad1", "genbank:0", ((16, 20), (0, 4)), "+", source_locus_tag="g1")
    ]
    assert (
        annotation_tables(result, {"mt": (0, 20)}, bin_size=20)["bin_coverage"][0]["annotated_bp"]
        == 8
    )


def test_publication_svgs_have_complete_exact_source_tables(tmp_path):
    pav = {
        "nodes": ["n1", "n2", "n3"],
        "paths": ["s1", "s2"],
        "matrix": [[1, 1], [1, 0], [0, 1]],
        "frequencies": [1.0, 0.5, 0.5],
        "classes": ["core", "shell", "shell"],
    }
    tree = node_pav_tree(pav["paths"], pav["matrix"])
    annotation = annotation_tables(
        [Annotation("s1", "g", "l1", ((0, 4),))], {"s1": (0, 10), "s2": (0, 10)}, bin_size=5
    )
    annotation["path_bounds"] = {"s1": (0, 10), "s2": (0, 10)}
    annotation["projection"] = []
    files = write_report(tmp_path, pav, tree, annotation)
    assert len(files) == len(set(files))
    assert all(path.is_file() for path in files)
    for path in files:
        if path.suffix == ".svg":
            assert ET.parse(path).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    with (tmp_path / "node_pav.tsv").open() as handle:
        assert list(csv.reader(handle, delimiter="\t")) == [
            ["feature", "s1", "s2"],
            ["n1", "1", "1"],
            ["n2", "1", "0"],
            ["n3", "0", "1"],
        ]
    with (tmp_path / "node_classes.tsv").open() as handle:
        assert list(csv.reader(handle, delimiter="\t")) == [
            ["class", "node_count"],
            ["core", "1"],
            ["shell", "2"],
            ["cloud", "0"],
            ["unobserved", "0"],
        ]
    assert "Node-PAV Jaccard" in (tmp_path / "node_pav_tree.svg").read_text()


def test_annotate_graph_matches_pansn_and_walk_native_coordinates(tmp_path):
    from organelleverse.pangenome.annotation_projection import annotate_graph

    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "H\tVN:Z:1.1\nS\ta\tAAAA\nS\tb\tCCCC\n"
        "L\ta\t+\tb\t-\t0M\n"
        "P\ts1#1#mt\ta+,b-\t0M\n"
        "W\ts2\t1\tmt\t100\t108\t>a<b\n"
    )
    bed = tmp_path / "annotation.bed"
    bed.write_text("s1#1#mt\t2\t6\tnad1\t0\t+\ns2#1#mt:100-108\t102\t106\tnad1\t0\t+\n")
    result = annotate_graph(graph, bed, bin_size=4)
    assert result["samples"] == ["s1", "s2"]
    assert result["copy_matrix"] == [[1, 1]]
    assert result["path_bounds"]["s2#1#mt:100-108"] == (100, 108)
    assert [
        (row["node"], row["node_start"], row["node_end"], row["strand"])
        for row in result["projection"]
    ] == [
        ("a", 2, 4, "+"),
        ("b", 2, 4, "-"),
        ("a", 2, 4, "+"),
        ("b", 2, 4, "-"),
    ]
    graph.write_text("H\tVN:Z:1.1\nS\ta\tAAAA\nW\ts\t1\tmt\t*\t*\t>a\n")
    bed.write_text("s#1#mt:*-*\t0\t2\tnad1\t0\t+\n")
    with pytest.raises(ValueError, match="declared GFA walk start"):
        annotate_graph(graph, bed)


def test_graph_report_retains_unobserved_nodes(tmp_path):
    from organelleverse.pangenome.graph import node_pav

    graph = tmp_path / "unobserved.gfa"
    graph.write_text("S\ta\tAAAA\nS\tb\tCCCC\nS\tu\tGGGG\nP\ts1\ta+\t*\nP\ts2\tb+\t*\n")
    pav = node_pav(graph)
    assert pav["classes"][-1] == "unobserved"
    files = write_report(tmp_path / "report", pav)
    assert all(path.exists() for path in files)
    assert "unobserved\t1" in (tmp_path / "report" / "node_classes.tsv").read_text()


def test_report_frequency_axis_preserves_sample_denominator(tmp_path):
    pav = {
        "nodes": ["n1"],
        "paths": ["s1", "s2"],
        "matrix": [[1, 1]],
        "frequencies": [1.0],
        "classes": ["core"],
        "frequency_denominator": "samples",
    }
    write_report(tmp_path, pav)
    assert "Fraction of samples containing node" in (tmp_path / "node_frequency.svg").read_text()
    assert "Graph-node presence across samples" in (tmp_path / "node_pav.svg").read_text()


def _project_genbank_fixture(tmp_path):
    def write_records(name, records):
        destination = tmp_path / f"{name}.gb"
        SeqIO.write(records, destination, "genbank")
        return str(destination)

    def record(identifier, sequence, gene):
        item = SeqRecord(Seq(sequence), id=identifier, name=identifier, description="fixture")
        item.annotations["molecule_type"] = "DNA"
        item.features = [
            SeqFeature(
                FeatureLocation(0, 3, strand=1),
                type="gene",
                qualifiers={"gene": [gene], "locus_tag": ["same_locus_tag"]},
            )
        ]
        return item

    # Both samples use "mt" and the same locus tag; sample 1 has two molecules.
    s1 = [record("mt", "AAAAAA", "NAD1"), record("sub", "CCCC", "nad1")]
    s1[0].features[0].location = CompoundLocation(
        [FeatureLocation(0, 2, strand=1), FeatureLocation(4, 6, strand=1)]
    )
    s2 = [record("mt", "AAAAAA", "nad1")]
    manifest = {
        "samples": [
            {
                "name": "s1",
                "annotation": write_records("s1", s1),
                "molecules": [
                    {"source_id": "mt", "path": "s1#1#1", "length": 6, "topology": "circular"},
                    {"source_id": "sub", "path": "s1#1#2", "length": 4, "topology": "linear"},
                ],
            },
            {
                "name": "s2",
                "annotation": write_records("s2", s2),
                "molecules": [
                    {"source_id": "mt", "path": "s2#1#1", "length": 6, "topology": "linear"}
                ],
            },
            {
                "name": "s3",
                "annotation": None,
                "molecules": [
                    {"source_id": "mt", "path": "s3#1#1", "length": 6, "topology": "linear"}
                ],
            },
        ]
    }
    graph = tmp_path / "project.gfa"
    graph.write_text(
        "S\ta\tAAAAAA\nS\tb\tCCCC\n"
        "P\ts1#1#1\ta+\t*\nP\ts1#1#2\tb+\t*\n"
        "P\ts2#1#1\ta+\t*\nP\ts3#1#1\ta+\t*\n"
    )
    return graph, manifest


def test_project_genbank_per_sample_maps_preserve_compound_loci_and_copy_counts(tmp_path):
    from organelleverse.pangenome.annotation_projection import annotate_project_graph

    graph, manifest = _project_genbank_fixture(tmp_path)
    result = annotate_project_graph(graph, manifest, tmp_path, gene_synonyms={"NAD1": "nad1"})
    assert result["genes"] == ["nad1"]
    assert result["samples"] == ["s1", "s2", "s3"]
    assert result["copy_matrix"] == [[2, 1, 0]]
    assert result["annotation_count"] == 3
    assert len(result["gene_arrows"]) == 4
    assert result["sample_annotation_counts"] == {"s1": 2, "s2": 1, "s3": 0}
    assert [
        (record["sample"], record["source_id"], record["path"])
        for record in result["source_annotation_records"]
    ] == [("s1", "mt", "s1#1#1"), ("s1", "sub", "s1#1#2"), ("s2", "mt", "s2#1#1")]
    assert all(
        record["sequence_coordinate_check"] == "exact_sequence_match"
        for record in result["source_annotation_records"]
    )
    # Aliases must be explicitly requested.
    raw = annotate_project_graph(graph, manifest, tmp_path)
    assert raw["genes"] == ["NAD1", "nad1"]


def test_project_genbank_rejects_missing_paths_and_changed_coordinate_sequences(tmp_path):
    from organelleverse.pangenome.annotation_projection import annotate_project_graph

    graph, manifest = _project_genbank_fixture(tmp_path)
    original = graph.read_text()
    graph.write_text(original.replace("P\ts2#1#1\ta+\t*\n", ""))
    with pytest.raises(ValueError, match="Mapped annotation path absent"):
        annotate_project_graph(graph, manifest, tmp_path)
    graph.write_text(original.replace("S\ta\tAAAAAA", "S\ta\tAAAAAC"))
    with pytest.raises(ValueError, match="sequence differs"):
        annotate_project_graph(graph, manifest, tmp_path)
    graph.write_text(original)
    manifest["samples"][0]["molecules"][1]["source_id"] = "mt"
    with pytest.raises(ValueError, match="ambiguous molecule identities"):
        annotate_project_graph(graph, manifest, tmp_path)


def test_report_exports_true_pdf_png_and_svg_with_source_manifest(tmp_path):
    from PIL import Image

    pav = {
        "nodes": ["n1", "n2"],
        "paths": ["s1", "s2"],
        "matrix": [[1, 1], [1, 0]],
        "frequencies": [1, 0.5],
        "classes": ["core", "shell"],
    }
    tree = node_pav_tree(pav["paths"], pav["matrix"])
    annotation = annotation_tables(
        [Annotation("s1", "g", "l1", ((0, 4),))], {"s1": (0, 10), "s2": (0, 10)}, bin_size=5
    )
    annotation["path_bounds"] = {"s1": (0, 10), "s2": (0, 10)}
    annotation["projection"] = []
    files = write_report(tmp_path, pav, tree, annotation, formats=("svg", "pdf", "png"))
    panels = {
        "node_frequency",
        "node_classes",
        "node_pav",
        "node_pav_tree",
        "gene_copy_number",
        "gene_pav",
        "gene_arrows",
        "bin_coverage",
    }
    for stem in panels:
        assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        png = tmp_path / f"{stem}.png"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(png) as decoded:
            assert decoded.width >= 1500 and decoded.height >= 900
            assert np.ptp(np.asarray(decoded)) > 0
            assert decoded.info["dpi"][0] == pytest.approx(300, abs=0.1)
        assert ET.parse(tmp_path / f"{stem}.svg").getroot().tag.endswith("svg")
    with (tmp_path / "figure_sources.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == len(panels) * 3
    for row in rows:
        assert tmp_path / row["figure"] in files
        assert all((tmp_path / source).exists() for source in row["source_files"].split(";"))
    with pytest.raises(ValueError, match="publication formats"):
        write_report(tmp_path, pav, formats=("jpg",))


def test_genbank_repeated_locus_tags_and_mixed_strand_trans_splicing(tmp_path):
    record = SeqRecord(Seq("A" * 100), id="mt", name="mt", description="real-case regression")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(0, 4, strand=1),
            type="gene",
            qualifiers={"gene": ["trnS"], "locus_tag": ["trnS"]},
        ),
        SeqFeature(
            FeatureLocation(10, 14, strand=-1),
            type="gene",
            qualifiers={"gene": ["trnS"], "locus_tag": ["trnS"]},
        ),
        SeqFeature(
            CompoundLocation(
                [FeatureLocation(20, 24, strand=-1), FeatureLocation(80, 84, strand=1)]
            ),
            type="gene",
            qualifiers={"gene": ["rps12"], "locus_tag": ["rps12"], "trans_splicing": [""]},
        ),
    ]
    source = tmp_path / "mixed.gb"
    SeqIO.write([record], source, "genbank")
    annotations = read_genbank(source)
    assert [annotation.locus_id for annotation in annotations] == [
        "genbank:0",
        "genbank:1",
        "genbank:2",
    ]
    assert [annotation.source_locus_tag for annotation in annotations] == ["trnS", "trnS", "rps12"]
    assert annotations[2].part_strands == ("-", "+")
    tables = annotation_tables(annotations, {"mt": (0, 100)}, bin_size=100)
    assert tables["genes"] == ["rps12", "trnS"]
    assert tables["copy_matrix"] == [[1], [2]]
    arrows = [row for row in tables["gene_arrows"] if row["gene"] == "rps12"]
    assert [(row["start"], row["end"], row["strand"]) for row in arrows] == [
        (20, 24, "-"),
        (80, 84, "+"),
    ]
    projection = project_annotations(
        annotations,
        {
            "mt": [
                {"node": "a", "orientation": "+", "start": 0, "end": 50},
                {"node": "b", "orientation": "-", "start": 50, "end": 100},
            ]
        },
    )
    trans_spliced = [row for row in projection if row["gene"] == "rps12"]
    assert [
        (row["node"], row["node_start"], row["node_end"], row["strand"]) for row in trans_spliced
    ] == [("a", 20, 24, "-"), ("b", 16, 20, "-")]
    assert all(row["source_locus_tag"] == "rps12" for row in trans_spliced)
