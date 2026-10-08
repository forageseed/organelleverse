from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.heteroplasmy_structural import estimate_structural_fractions


def _gaf_row(query: str, path: str, *, mapq: int = 60, qstart: int = 0, qend: int = 100) -> str:
    path_nodes = path.count(">") + path.count("<")
    return "\t".join(
        [query, "100", str(qstart), str(qend), "+", path, str(path_nodes * 100),
         "0", str(path_nodes * 100), "95", "100", str(mapq)]
    )


def _inputs(tmp_path: Path, rows: list[str]) -> tuple[Path, Path]:
    configs = tmp_path / "configs.tsv"
    configs.write_text(
        "branch_id\tconfiguration_id\tpath_signature\n"
        "b1\tA\t>L>choiceA>R\n"
        "b1\tB\t>L<choiceB>R\n",
        encoding="utf-8",
    )
    gaf = tmp_path / "reads.gaf"
    gaf.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return gaf, configs


def test_competing_oriented_paths_use_unique_read_denominator(tmp_path: Path) -> None:
    gaf, configs = _inputs(
        tmp_path,
        [
            _gaf_row("a1", ">L>choiceA>R"),
            _gaf_row("a1", ">L>choiceA>R"),  # duplicate record for same molecule
            _gaf_row("a2", ">X>L>choiceA>R>Y"),
            _gaf_row("b1", ">L<choiceB>R"),
            _gaf_row("ambiguous", ">L>choiceA>R"),
            _gaf_row("ambiguous", ">L<choiceB>R"),
            _gaf_row("partial", ">L>choiceA>R", qstart=1),
            _gaf_row("unknown_mapq", ">L<choiceB>R", mapq=255),
        ],
    )
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "ok", result.errors
    branch = result.metrics["branches"][0]
    assert branch["denominator_unique_informative_reads"] == 3
    assert branch["ambiguous_reads_excluded"] == 1
    rows = {row["configuration_id"]: row for row in branch["configurations"]}
    assert rows["A"]["supporting_reads"] == 2
    assert rows["A"]["fraction"] == 0.666667
    assert rows["B"]["supporting_reads"] == 1
    assert rows["B"]["fraction"] == 0.333333
    assert rows["A"]["ci95_lower"] < rows["A"]["fraction"] < rows["A"]["ci95_upper"]


def test_insufficient_graph_spanning_reads_report_no_estimate(tmp_path: Path) -> None:
    gaf, configs = _inputs(tmp_path, [_gaf_row("short", ">L>choiceA>R", qend=99)])
    branch = estimate_structural_fractions(gaf, configs).metrics["branches"][0]
    assert branch["denominator_unique_informative_reads"] == 0
    assert branch["status"] == "no_informative_reads"
    assert all(row["fraction"] is None for row in branch["configurations"])


def test_competing_signatures_require_a_shared_oriented_anchor(tmp_path: Path) -> None:
    configs = tmp_path / "configs.tsv"
    configs.write_text(
        "branch_id\tconfiguration_id\tpath_signature\n"
        "b1\tA\t>L>A>R\n"
        "b1\tB\t>Z>B<Y\n",
        encoding="utf-8",
    )
    gaf = tmp_path / "reads.gaf"
    gaf.write_text("", encoding="utf-8")
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.invalid_configurations"


@pytest.mark.parametrize(
    ("signatures", "walks"),
    [
        ((">repeat>left", ">repeat>right"), (">repeat>left", "<right<repeat")),
        ((">left>repeat", ">right>repeat"), (">left>repeat", "<repeat<right")),
    ],
)
def test_binary_graph_junctions_accept_shared_start_or_end_anchor(
    tmp_path: Path,
    signatures: tuple[str, str],
    walks: tuple[str, str],
) -> None:
    configs = tmp_path / "configs.tsv"
    configs.write_text(
        "branch_id\tconfiguration_id\tpath_signature\n"
        f"b1\tA\t{signatures[0]}\n"
        f"b1\tB\t{signatures[1]}\n",
        encoding="utf-8",
    )
    gaf = tmp_path / "reads.gaf"
    gaf.write_text(
        "\n".join([_gaf_row("forward", walks[0]), _gaf_row("reverse", walks[1])]) + "\n",
        encoding="utf-8",
    )
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "ok", result.errors
    rows = {
        row["configuration_id"]: row
        for row in result.metrics["branches"][0]["configurations"]
    }
    assert rows["A"]["supporting_reads"] == 1
    assert rows["B"]["supporting_reads"] == 1


def test_reverse_traversal_counts_unique_molecules_and_excludes_ambiguity(tmp_path: Path) -> None:
    gaf, configs = _inputs(tmp_path, [
        _gaf_row("a_forward", ">L>choiceA>R"),
        _gaf_row("a_forward", "<R<choiceA<L"),  # same molecule in both directions
        _gaf_row("a_reverse", ">X<R<choiceA<L<Y"),
        _gaf_row("b_reverse", "<R>choiceB<L"),
        _gaf_row("ambiguous", "<R<choiceA<L"),
        _gaf_row("ambiguous", ">L<choiceB>R"),
        _gaf_row("wrong_orientation", ">R>choiceA>L"),
        _gaf_row("partial_reverse", "<R>choiceB<L", qstart=1),
    ])
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "ok", result.errors
    branch = result.metrics["branches"][0]
    assert branch["denominator_unique_informative_reads"] == 3
    assert branch["ambiguous_reads_excluded"] == 1
    rows = {row["configuration_id"]: row for row in branch["configurations"]}
    assert rows["A"]["supporting_reads"] == 2
    assert rows["B"]["supporting_reads"] == 1


@pytest.mark.parametrize("walk", [">single", "<single"])
def test_valid_single_node_alignment_is_uninformative(tmp_path: Path, walk: str) -> None:
    gaf, configs = _inputs(tmp_path, [_gaf_row("one_node", walk)])
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "ok", result.errors
    assert result.metrics["branches"][0]["status"] == "no_informative_reads"


def test_single_node_configuration_is_rejected(tmp_path: Path) -> None:
    gaf, configs = _inputs(tmp_path, [])
    configs.write_text(
        "branch_id\tconfiguration_id\tpath_signature\n"
        "b1\tA\t>L\nb1\tB\t>R\n", encoding="utf-8"
    )
    result = estimate_structural_fractions(gaf, configs)
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.invalid_configurations"
