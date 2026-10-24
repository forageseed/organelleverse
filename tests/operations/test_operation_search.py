"""Operation visibility and the Agent-facing operation search (ov.operations.search)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import organelleverse.operations as ops
from organelleverse.operations import OperationSpec
from organelleverse.operations.operation_search import SearchEntry, rank

FIXTURES = Path(__file__).parent / "fixtures"
RELEASE_ORACLE = Path(__file__).parents[1] / "capabilities" / "fixtures" / "release_catalog_v0.json"


@pytest.fixture(scope="module")
def specs() -> tuple[OperationSpec, ...]:
    return ops.list()


def _recall(specs, path: Path) -> tuple[float, float]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    at1 = at5 = 0
    for case in cases:
        got = [hit.operation_id for hit in ops.search(case["q"], limit=5, specs=specs)]
        at1 += bool(got) and got[0] in case["ok"]
        at5 += bool(set(got) & set(case["ok"]))
    return at1 / len(cases), at5 / len(cases)


def test_superseded_operations_point_at_an_agent_reachable_operation(specs) -> None:
    by_id = {spec.operation_id: spec for spec in specs}
    superseded = [spec for spec in specs if spec.superseded_by]
    assert len(superseded) >= 18
    for spec in superseded:
        assert spec.visibility == "internal", spec.operation_id
        assert by_id[spec.superseded_by].visibility != "internal", spec.operation_id


def test_contracts_match_the_visibility_ledger(specs) -> None:
    import tomllib

    ledger = tomllib.loads((Path(__file__).parents[2] / "docs/operations/agent-visibility.toml").read_text("utf-8"))
    expected = {op: ("internal", target) for op, target in ledger["superseded"].items()}
    expected |= {op: ("internal", None) for op in ledger["internal"]["ids"]}
    expected |= {op: ("step", None) for op in ledger["step"]["ids"]}
    actual = {spec.operation_id: (spec.visibility, spec.superseded_by) for spec in specs}
    assert {op: actual.get(op) for op in expected} == expected
    assert {op for op, (level, _) in actual.items() if level != "primary"} == set(expected)


def test_released_contracts_stay_primary_and_byte_identical(specs) -> None:
    released = set(json.loads(RELEASE_ORACLE.read_text(encoding="utf-8")))
    for spec in specs:
        if spec.operation_id in released:
            assert spec.visibility == "primary"
            assert "visibility" not in spec.model_dump(mode="json")
            assert "superseded_by" not in spec.model_dump(mode="json")


def test_superseded_by_requires_internal_visibility(specs) -> None:
    document = next(spec for spec in specs if spec.operation_id == "composition.gc_content").model_dump(mode="json")
    with pytest.raises(ValueError, match="superseded_by requires visibility 'internal'"):
        OperationSpec.model_validate({**document, "superseded_by": "composition.compute_gc_content"})


def test_internal_operations_are_never_offered(specs) -> None:
    internal = {spec.operation_id for spec in specs if spec.visibility == "internal"}
    assert {"localization.install_hint", "composition.compute_gc_content", "selection.check_kaks_calculator"} <= internal
    for query in ("install hint", "check backend", "gc content", "kaks calculator", "compute gc content dict"):
        assert not {hit.operation_id for hit in ops.search(query, limit=50, specs=specs)} & internal, query


def test_synonym_resolves_to_the_result_returning_operation(specs) -> None:
    assert ops.search("GC content", specs=specs)[0].operation_id == "composition.gc_content"
    assert ops.search("kaks", specs=specs)[0].operation_id == "selection.kaks"


def test_steps_rank_below_and_can_be_excluded(specs) -> None:
    hits = ops.search("kaks", limit=20, specs=specs)
    assert hits[0].visibility == "primary"
    assert all(hit.visibility == "primary" for hit in ops.search("kaks", limit=20, specs=specs, include_steps=False))


def test_chinese_request_reaches_the_analysis(specs) -> None:
    assert ops.search("计算线粒体基因组的GC含量", specs=specs)[0].operation_id == "composition.gc_content"
    assert ops.search("注释叶绿体基因组", specs=specs)[0].operation_id == "annotation.annotate"


def test_organelle_in_the_query_filters_declared_operations() -> None:
    entries = [
        SearchEntry("x.plastid_only", title="Edit sites", description="Edit sites.", organelle_types=("plastid",)),
        SearchEntry("x.any", title="Edit sites", description="Edit sites."),
    ]
    assert [hit.operation_id for hit in rank(entries, "mitochondrial edit sites")] == ["x.any"]
    assert {hit.operation_id for hit in rank(entries, "chloroplast edit sites")} == {"x.any", "x.plastid_only"}


def test_figure_operations_wait_for_a_plot_request(specs) -> None:
    assert not ops.search("nucleotide diversity", specs=specs)[0].operation_id.startswith("visualization.")
    assert ops.search("plot nucleotide diversity", specs=specs)[0].operation_id.startswith("visualization.")


def test_recall_on_the_tuning_set(specs) -> None:
    at1, at5 = _recall(specs, FIXTURES / "agent_search_eval.json")
    assert at1 >= 0.90 and at5 >= 0.97, (at1, at5)


def test_recall_floor_on_held_out_requests(specs) -> None:
    # Held-out cases were never used to tune; this is a floor against regressions, not a target.
    at1, at5 = _recall(specs, FIXTURES / "agent_search_heldout.json")
    assert at1 >= 0.55 and at5 >= 0.80, (at1, at5)
