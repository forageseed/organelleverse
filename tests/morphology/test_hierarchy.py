"""T-F1: the object hierarchy — tree rules, aggregation levels, delete policy."""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology.hierarchy import HierarchyNode, ObjectHierarchy


def _tree() -> ObjectHierarchy:
    return ObjectHierarchy(
        [
            HierarchyNode(id="cell-1", kind="region", class_name=None),
            HierarchyNode(id="cell-2", kind="region", class_name=None),
            HierarchyNode(
                id="cp-1", kind="organelle", parent_id="cell-1", class_name="Chloroplast"
            ),
            HierarchyNode(
                id="cp-2", kind="organelle", parent_id="cell-1", class_name="Chloroplast"
            ),
            HierarchyNode(
                id="mt-1", kind="organelle", parent_id="cell-2", class_name="Mitochondria"
            ),
            HierarchyNode(id="mt-2", kind="organelle", class_name="Mitochondria"),  # top-level
        ]
    )


def test_only_regions_may_parent() -> None:
    tree = _tree()
    with pytest.raises(OrganelleInputError) as raised:
        tree.add(HierarchyNode(id="x", kind="organelle", parent_id="cp-1"))
    assert raised.value.code == "morphology.hierarchy_invalid_parent"


def test_missing_parent_and_duplicate_id_fail_closed() -> None:
    tree = _tree()
    with pytest.raises(OrganelleInputError):
        tree.add(HierarchyNode(id="y", kind="organelle", parent_id="ghost"))
    with pytest.raises(OrganelleInputError):
        tree.add(HierarchyNode(id="cp-1", kind="organelle"))


def test_reparent_moves_and_cycle_is_refused() -> None:
    tree = _tree()
    moved = tree.reparent("cp-2", "cell-2")
    assert moved.parent_id == "cell-2"
    assert {n.id for n in tree.children_of("cell-2")} == {"mt-1", "cp-2"}
    # regions can nest; a cycle through region->region is refused
    tree.reparent("cell-2", "cell-1")  # cell-2 now inside cell-1
    with pytest.raises(OrganelleInputError) as raised:
        tree.reparent("cell-1", "cell-2")  # would cycle
    assert raised.value.code == "morphology.hierarchy_cycle"
    # organelles can never parent, so that class of cycle cannot form
    with pytest.raises(OrganelleInputError):
        tree.reparent("cell-1", "cp-1")


def test_delete_cascade_removes_descendants() -> None:
    tree = _tree()
    removed = tree.delete("cell-1", policy="cascade")
    assert set(removed) == {"cell-1", "cp-1", "cp-2"}
    with pytest.raises(OrganelleInputError):
        tree.get("cp-1")


def test_delete_promote_lifts_children_to_the_parent_level() -> None:
    tree = _tree()
    removed = tree.delete("cell-1", policy="promote")
    assert removed == ("cell-1",)
    assert tree.get("cp-1").parent_id is None  # promoted to top level
    assert tree.get("cp-2").parent_id is None


def test_delete_never_silently_drops_children() -> None:
    tree = _tree()
    with pytest.raises(OrganelleInputError) as raised:
        tree.delete("cell-1", policy="forget")  # type: ignore[arg-type]
    assert raised.value.code == "morphology.hierarchy_unknown_policy"
    assert tree.get("cp-1")  # untouched


def test_aggregation_by_region_and_class() -> None:
    tree = _tree()
    rows = [
        {"object_id": "cp-1", "area_px": 100.0},
        {"object_id": "cp-2", "area_px": 200.0},
        {"object_id": "mt-1", "area_px": 50.0},
        {"object_id": "mt-2", "area_px": 70.0},
    ]
    whole = tree.aggregate(rows)
    assert whole["n"] == 4 and whole["sum"] == 420.0
    cell1 = tree.aggregate(rows, region_id="cell-1")
    assert cell1["n"] == 2 and cell1["sum"] == 300.0
    mito = tree.aggregate(rows, class_name="Mitochondria")
    assert mito["n"] == 2 and mito["sum"] == 120.0
    cell2_mito = tree.aggregate(rows, region_id="cell-2", class_name="Mitochondria")
    assert cell2_mito["n"] == 1 and cell2_mito["sum"] == 50.0
