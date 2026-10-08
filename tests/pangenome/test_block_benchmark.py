import pytest

from organelleverse.pangenome.block_benchmark import BlockBenchmarkTruth, score_blocks


def make_truth(ref, query, strand="+"):
    return BlockBenchmarkTruth.model_validate(
        {
            "schema_version": "organelleverse.pangenome.block-truth.v1",
            "benchmark_id": "test.block",
            "benchmark_version": "1.0.0",
            "case_id": "fixture",
            "source_description": "Explicit test blocks",
            "reference": "ref",
            "sequences": {"ref": ref, "query": query},
            "blocks": [
                {
                    "query": "query",
                    "query_start": 0,
                    "query_end": len(query),
                    "reference_start": 0,
                    "reference_end": len(ref),
                    "strand": strand,
                    "source_row": 3,
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "strand,category", [("-", "consistent_unique"), ("+", "inconsistent_unique")]
)
def test_inversion_direction_is_scored_independently_of_sequence_reconstruction(
    tmp_path, strand, category
):
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "H\tVN:Z:1.0\nS\t1\tAAA\nS\t2\tCCC\nL\t1\t+\t2\t+\t0M\nP\tref\t1+,2+\t*\nP\tquery\t2-,1-\t*\n"
    )
    result = score_blocks(graph, make_truth("AAACCC", "GGGTTT", strand))
    assert result["counts"][category] == 6
    assert sum(result["counts"].values()) == 6
    assert result["metrics"]["exact_input_paths"] == 1


def test_repeated_reference_placements_are_ambiguous_not_cherry_picked(tmp_path):
    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tAAA\nL\t1\t+\t1\t+\t0M\nP\tref\t1+,1+\t*\nP\tquery\t1+\t*\n")
    result = score_blocks(graph, make_truth("AAAAAA", "AAA"))
    assert result["counts"]["ambiguous"] == 3
    assert result["counts"]["consistent_unique"] == 0


def test_unshared_nodes_are_unmapped_and_wrong_reference_interval_is_inconsistent(tmp_path):
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "S\t1\tAAA\nS\t2\tCCC\nS\t3\tAAA\nL\t1\t+\t2\t+\t0M\nL\t3\t+\t2\t+\t0M\nP\tref\t1+,2+\t*\nP\tquery\t3+,2+\t*\n"
    )
    truth = make_truth("AAACCC", "AAACCC")
    truth = truth.model_copy(
        update={"blocks": (truth.blocks[0].model_copy(update={"reference_end": 3}),)}
    )
    result = score_blocks(graph, truth)
    assert result["counts"]["unmapped"] == result["counts"]["inconsistent_unique"] == 3
    assert sum(result["metrics"]["block_" + c + "_fraction"] for c in result["counts"]) == 1


def test_bounds_and_sequence_mismatch_are_not_repaired(tmp_path):
    data = make_truth("AAA", "AAA").model_dump()
    data["blocks"][0]["query_end"] = 4
    with pytest.raises(ValueError, match="outside"):
        BlockBenchmarkTruth.model_validate(data)
    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tAAA\nP\tref\t1+\t*\nP\tquery\t1-\t*\n")
    with pytest.raises(ValueError, match="exactly reconstruct"):
        score_blocks(graph, make_truth("AAA", "AAA"))
