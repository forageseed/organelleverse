import pytest
from Bio.Seq import Seq

from organelleverse.pangenome.annotation_audit import (
    audit_annotations,
    untangle_annotations,
    validate_gene_synonyms,
)
from organelleverse.pangenome.annotation_projection import Annotation
from organelleverse.pangenome.normalization import MoleculeTransform


def extracted(annotation, sequence):
    return "".join(
        str(Seq(sequence[a:b]).reverse_complement())
        if annotation.strand_for_part(i) == "-"
        else sequence[a:b]
        for i, (a, b) in enumerate(annotation.parts)
    )


@pytest.mark.parametrize("orientation", ["+", "-"])
@pytest.mark.parametrize("origin", range(8))
def test_normalization_inverse_and_trans_spliced_annotation_sequence(orientation, origin):
    sequence = "ACGTTGCA"
    transform = MoleculeTransform("original", "normalized", 8, "circular", orientation, origin)
    output = transform.sequence(sequence)
    assert transform.inverse().sequence(output) == sequence
    feature = Annotation(
        "original", "rps12", "locus", ((0, 3), (5, 8)), ".", part_strands=("+", "-")
    )
    changed = transform.annotation(feature)
    assert extracted(changed, output) == extracted(feature, sequence)
    assert extracted(transform.inverse().annotation(changed), sequence) == extracted(
        feature, sequence
    )
    for row in transform.coordinate_rows():
        source = sequence[row["source_start"] : row["source_end"]]
        if orientation == "-":
            source = str(Seq(source).reverse_complement())
        assert source == output[row["output_start"] : row["output_end"]]


def test_rotation_requires_explicit_circular_topology_and_identity_retained():
    with pytest.raises(ValueError, match="circular"):
        MoleculeTransform("a", "a", 10, "unknown", "+", 3)
    transform = MoleculeTransform("a", "a", 4)
    assert transform.sequence("AcgT") == "AcgT"
    assert transform.coordinate_rows()[0]["source_path"] == "a"


def test_annotation_audit_does_not_equate_missing_input_with_gene_absence():
    feature = Annotation("a#1#1", "alias", "x", ((0, 2),))
    result = audit_annotations(
        [feature],
        path_samples={"a#1#1": "a", "b#1#1": "b", "c#1#1": "c"},
        annotated_samples=["a", "b"],
        expected_genes=["gene", "other"],
        gene_synonyms={"alias": "gene"},
    )
    a, b, c = result["rows"]
    assert a["observed_fraction"] == 0.5
    assert b["feature_count"] == 0 and b["expected_not_observed"] == ["gene", "other"]
    assert c["feature_count"] is None and c["expected_not_observed"] is None
    with pytest.raises(ValueError, match="chains/cycles"):
        validate_gene_synonyms({"a": "b", "b": "c"})


def test_exact_untangle_preserves_reverse_duplicate_target_occurrences():
    def step(orientation, start):
        return {
            "node": "1",
            "orientation": orientation,
            "start": start,
            "end": start + 10,
            "node_length": 10,
        }

    result = untangle_annotations(
        [Annotation("source", "g", "l", ((2, 5),))],
        {"source": [step("+", 0)], "target": [step("-", 0), step("+", 10)]},
        reference_paths=["target"],
    )
    assert [
        (row["target_start"], row["target_end"], row["target_strand"]) for row in result["rows"]
    ] == [(5, 8, "-"), (12, 15, "+")]
    assert [row["target_step"] for row in result["rows"]] == [0, 1]


def test_project_annotation_uses_declared_transform_and_audits_empty_features(tmp_path):
    from Bio import SeqIO
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    from organelleverse.pangenome.annotation_projection import annotate_project_graph

    sequence = "ACGTTGCA"
    transform = MoleculeTransform("mt", "a#1#1", 8, "circular", "-", 3)
    samples = []
    for sample in ("a", "b"):
        record = SeqRecord(
            Seq(sequence), id="mt", annotations={"molecule_type": "DNA", "topology": "circular"}
        )
        if sample == "a":
            record.features = [
                SeqFeature(
                    SimpleLocation(0, 4, strand=1), type="gene", qualifiers={"gene": ["alias"]}
                )
            ]
        path = tmp_path / f"{sample}.gb"
        SeqIO.write([record], path, "genbank")
        samples.append(
            {
                "name": sample,
                "annotation": str(path),
                "molecules": [
                    {
                        "source_id": "mt",
                        "path": f"{sample}#1#1",
                        "length": 8,
                        "topology": "circular",
                        "normalization": {"orientation": "-", "origin": 3} if sample == "a" else {},
                    }
                ],
            }
        )
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{transform.sequence(sequence)}\nS\t2\t{sequence}\nP\ta#1#1\t1+\t*\nP\tb#1#1\t2+\t*\n"
    )
    result = annotate_project_graph(
        graph,
        {"samples": samples},
        tmp_path,
        gene_synonyms={"alias": "g"},
        expected_genes=["g"],
        reference_paths=["a#1#1"],
    )
    assert result["annotation_audit"]["rows"][0]["observed_fraction"] == 1
    assert result["annotation_audit"]["rows"][1]["observed_fraction"] == 0
    assert result["source_annotation_records"][0]["normalization"] == {
        "orientation": "-",
        "origin": 3,
    }
    assert result["untangle"]["rows"]
    arrows = result["gene_arrows"]
    changed = Annotation(
        "a#1#1", "g", "x", tuple((row["start"], row["end"]) for row in arrows), "-"
    )
    assert extracted(changed, transform.sequence(sequence)) == sequence[:4]
