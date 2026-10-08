"""Class management for the morphology suite (T-F3).

The four OrgSegNet classes are built-in defaults; users may add custom
classes. Rules from the brief:

- custom-class colors come from the token categorical palette — never a
  user-picked arbitrary color (theme + colorblind safety are contractual)
- classes toggle visibility (hide one class to see the rest)
- custom classes enter provenance: a downstream consumer seeing a
  non-built-in class must be able to trace its definition
- rename/delete handle existing objects explicitly (migrate target or
  unclassify), never silently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.errors import OrganelleInputError

__all__ = ["BUILTIN_CLASSES", "ClassDefinition", "ClassRegistry"]

#: The OrgSegNet four (Nat. Plants 2023) — built-in defaults, never removable.
BUILTIN_CLASSES: tuple[str, ...] = ("Chloroplast", "Mitochondria", "Vacuole", "Nucleus")

#: Token-palette slots for custom classes (seriesColor indices 5..7 are the
#: slots the four defaults don't use); custom class colors come from here.
CUSTOM_PALETTE_SLOTS = (5, 6, 7, 0)


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    builtin: bool
    palette_index: int
    visible: bool = True
    defined_by: str = "builtin"  # operator id for custom classes (provenance)
    created_at: str = ""


class ClassRegistry:
    """Fail-closed class registry for one workbench session."""

    def __init__(self) -> None:
        self._classes: dict[str, ClassDefinition] = {
            name: ClassDefinition(name=name, builtin=True, palette_index=i + 1)
            for i, name in enumerate(BUILTIN_CLASSES)
        }
        self._custom_slots = iter(CUSTOM_PALETTE_SLOTS * 16)

    def add_custom(self, name: str, *, defined_by: str, created_at: str) -> ClassDefinition:
        clean = name.strip()
        if not clean:
            raise OrganelleInputError(
                code="morphology.class_blank_name",
                message="class name must not be blank",
                details={},
            )
        if clean in self._classes:
            raise OrganelleInputError(
                code="morphology.class_duplicate",
                message=f"class already exists: {clean}",
                details={"name": clean},
            )
        if not defined_by.strip():
            raise OrganelleInputError(
                code="morphology.class_missing_operator",
                message="a custom class needs its defining operator for provenance",
                details={"name": clean},
            )
        definition = ClassDefinition(
            name=clean,
            builtin=False,
            palette_index=next(self._custom_slots),
            defined_by=defined_by,
            created_at=created_at,
        )
        self._classes[clean] = definition
        return definition

    def set_visible(self, name: str, visible: bool) -> ClassDefinition:
        current = self._require(name)
        updated = ClassDefinition(
            name=current.name,
            builtin=current.builtin,
            palette_index=current.palette_index,
            visible=visible,
            defined_by=current.defined_by,
            created_at=current.created_at,
        )
        self._classes[name] = updated
        return updated

    def rename(self, name: str, new_name: str, *, migrate_objects: Literal["rename"]) -> ClassDefinition:
        """Rename a class; object membership migrates with the name (declared)."""
        current = self._require(name)
        if current.builtin:
            raise OrganelleInputError(
                code="morphology.class_builtin_immutable",
                message="built-in classes cannot be renamed",
                details={"name": name},
            )
        clean = new_name.strip()
        if not clean or clean in self._classes:
            raise OrganelleInputError(
                code="morphology.class_duplicate",
                message=f"invalid or duplicate class name: {new_name!r}",
                details={"name": new_name},
            )
        if migrate_objects != "rename":
            raise OrganelleInputError(
                code="morphology.class_rename_policy_unknown",
                message="rename must declare object handling",
                details={"name": name},
            )
        updated = ClassDefinition(
            name=clean,
            builtin=False,
            palette_index=current.palette_index,
            visible=current.visible,
            defined_by=current.defined_by,
            created_at=current.created_at,
        )
        del self._classes[name]
        self._classes[clean] = updated
        return updated

    def delete(self, name: str, *, objects: Literal["migrate", "unclassify"], migrate_to: str | None = None) -> None:
        """Delete a custom class; existing objects migrate or unclassify — declared."""
        current = self._require(name)
        if current.builtin:
            raise OrganelleInputError(
                code="morphology.class_builtin_immutable",
                message="built-in classes cannot be deleted",
                details={"name": name},
            )
        if objects == "migrate":
            if not migrate_to or migrate_to not in self._classes or migrate_to == name:
                raise OrganelleInputError(
                    code="morphology.class_migrate_target_invalid",
                    message="migrate deletes need a valid target class",
                    details={"name": name, "migrate_to": migrate_to},
                )
        elif objects != "unclassify":
            raise OrganelleInputError(
                code="morphology.class_delete_policy_unknown",
                message="delete must declare object handling: migrate | unclassify",
                details={"name": name, "objects": objects},
            )
        del self._classes[name]

    def get(self, name: str) -> ClassDefinition | None:
        return self._classes.get(name)

    def list(self) -> tuple[ClassDefinition, ...]:
        return tuple(self._classes.values())

    def _require(self, name: str) -> ClassDefinition:
        current = self._classes.get(name)
        if current is None:
            raise OrganelleInputError(
                code="morphology.class_unknown",
                message=f"unknown class: {name}",
                details={"name": name},
            )
        return current
