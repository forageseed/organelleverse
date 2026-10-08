"""Object hierarchy for the morphology suite (T-F1).

QuPath's core model is an object tree — annotation objects contain detection
objects. This project's object list was flat; this module adds the tree with
the project's own rules (fail-closed, provenance-bearing), driven by the
behavioral spec, not by transcompiled Java.

Rules (task brief T-F1):

- ``kind``: ``region`` (user ROI / cell outline) vs ``organelle`` (segmented
  object). Regions contain organelles; organelles never contain anything.
- Aggregation must work at any level: whole image / one region / one class.
- A hierarchy change is a correction — it goes through the T4a record shape
  (reparent is recorded, never silent).
- Deleting a parent has an explicit, declared child policy:
  ``cascade`` (children deleted with it) or ``promote`` (children become
  top-level). Never silently drop descendants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.errors import OrganelleInputError

__all__ = ["HierarchyNode", "ObjectHierarchy"]

ObjectKind = Literal["region", "organelle"]
DeletePolicy = Literal["cascade", "promote"]


@dataclass(frozen=True)
class HierarchyNode:
    """One object in the tree. ``parent_id=None`` means top-level."""

    id: str
    kind: ObjectKind
    parent_id: str | None = None
    class_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class ObjectHierarchy:
    """A fail-closed object tree for one image."""

    def __init__(self, nodes: list[HierarchyNode] | None = None) -> None:
        self._nodes: dict[str, HierarchyNode] = {}
        for node in nodes or []:
            self.add(node)

    def add(self, node: HierarchyNode) -> None:
        if node.id in self._nodes:
            raise OrganelleInputError(
                code="morphology.hierarchy_duplicate_id",
                message=f"duplicate object id: {node.id}",
                details={"id": node.id},
            )
        if node.parent_id is not None:
            parent = self._nodes.get(node.parent_id)
            if parent is None:
                raise OrganelleInputError(
                    code="morphology.hierarchy_missing_parent",
                    message=f"parent {node.parent_id!r} does not exist",
                    details={"id": node.id, "parent_id": node.parent_id},
                )
            if parent.kind != "region":
                raise OrganelleInputError(
                    code="morphology.hierarchy_invalid_parent",
                    message="only a region may contain objects",
                    details={"id": node.id, "parent_id": node.parent_id},
                )
        self._nodes[node.id] = node

    def get(self, object_id: str) -> HierarchyNode:
        node = self._nodes.get(object_id)
        if node is None:
            raise OrganelleInputError(
                code="morphology.hierarchy_unknown_object",
                message=f"unknown object: {object_id}",
                details={"id": object_id},
            )
        return node

    def children_of(self, parent_id: str) -> tuple[HierarchyNode, ...]:
        self.get(parent_id)
        return tuple(n for n in self._nodes.values() if n.parent_id == parent_id)

    def descendants_of(self, parent_id: str) -> tuple[HierarchyNode, ...]:
        out: list[HierarchyNode] = []
        frontier = list(self.children_of(parent_id))
        while frontier:
            node = frontier.pop()
            out.append(node)
            frontier.extend(self.children_of(node.id))
        return tuple(out)

    def reparent(self, object_id: str, new_parent_id: str | None) -> HierarchyNode:
        """Move one object; the move is returned for the caller's T4a record."""
        node = self.get(object_id)
        if new_parent_id == object_id:
            raise OrganelleInputError(
                code="morphology.hierarchy_self_parent",
                message="an object cannot parent itself",
                details={"id": object_id},
            )
        if new_parent_id is not None:
            parent = self.get(new_parent_id)
            if parent.kind != "region":
                raise OrganelleInputError(
                    code="morphology.hierarchy_invalid_parent",
                    message="only a region may contain objects",
                    details={"id": object_id, "parent_id": new_parent_id},
                )
            if any(d.id == new_parent_id for d in self.descendants_of(object_id)):
                raise OrganelleInputError(
                    code="morphology.hierarchy_cycle",
                    message="reparenting would create a cycle",
                    details={"id": object_id, "parent_id": new_parent_id},
                )
        moved = HierarchyNode(
            id=node.id,
            kind=node.kind,
            parent_id=new_parent_id,
            class_name=node.class_name,
            payload=node.payload,
        )
        self._nodes[object_id] = moved
        return moved

    def delete(self, object_id: str, *, policy: DeletePolicy) -> tuple[str, ...]:
        """Delete one object under the declared child policy; returns removed ids."""
        node = self.get(object_id)
        descendants = self.descendants_of(object_id)
        removed = [object_id]
        if policy == "cascade":
            removed.extend(d.id for d in descendants)
            for desc in descendants:
                del self._nodes[desc.id]
        elif policy == "promote":
            for child in self.children_of(object_id):
                self.reparent(child.id, node.parent_id)
        else:
            raise OrganelleInputError(
                code="morphology.hierarchy_unknown_policy",
                message=f"unknown delete policy: {policy!r}",
                details={"policy": policy},
            )
        del self._nodes[object_id]
        return tuple(removed)

    def aggregate(
        self,
        rows: list[dict[str, Any]],
        *,
        region_id: str | None = None,
        class_name: str | None = None,
        value_key: str = "area_px",
    ) -> dict[str, Any]:
        """Aggregate measurement rows at any level: whole image / one region / one class.

        ``rows`` are measure()-style records keyed by object ``id`` in their
        ``object_id``/``label`` payload — the caller aligns ids with nodes.
        """
        allowed: set[str] | None = None
        if region_id is not None:
            allowed = {d.id for d in self.descendants_of(region_id)}
        values: list[float] = []
        for row in rows:
            object_id = str(row.get("object_id"))
            if allowed is not None and object_id not in allowed:
                continue
            node = self._nodes.get(object_id)
            if class_name is not None and (node is None or node.class_name != class_name):
                continue
            value = row.get(value_key)
            if value is None:
                continue
            values.append(float(value))
        return {
            "region_id": region_id,
            "class_name": class_name,
            "n": len(values),
            "sum": sum(values),
            "mean": (sum(values) / len(values)) if values else None,
        }

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "id": n.id,
                "kind": n.kind,
                "parent_id": n.parent_id,
                "class_name": n.class_name,
                "payload": n.payload,
            }
            for n in self._nodes.values()
        ]
