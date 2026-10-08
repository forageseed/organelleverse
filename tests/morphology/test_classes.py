"""T-F3: class registry — palette-restricted custom classes, explicit lifecycle."""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology.classes import BUILTIN_CLASSES, ClassRegistry


def test_builtin_four_present_with_palette_slots() -> None:
    registry = ClassRegistry()
    names = [c.name for c in registry.list()]
    assert names == list(BUILTIN_CLASSES)
    assert all(c.builtin for c in registry.list())


def test_custom_class_gets_a_palette_color_and_provenance() -> None:
    registry = ClassRegistry()
    custom = registry.add_custom("LipidBody", defined_by="jane", created_at="2026-08-18T00:00:00Z")
    assert custom.palette_index in (5, 6, 7, 0)  # token palette slots only
    assert custom.defined_by == "jane"  # traceable definition
    assert not custom.builtin


def test_blank_duplicate_and_anonymous_custom_classes_fail_closed() -> None:
    registry = ClassRegistry()
    with pytest.raises(OrganelleInputError):
        registry.add_custom("  ", defined_by="jane", created_at="")
    with pytest.raises(OrganelleInputError):
        registry.add_custom("Chloroplast", defined_by="jane", created_at="")
    with pytest.raises(OrganelleInputError):
        registry.add_custom("NewClass", defined_by=" ", created_at="")


def test_builtin_classes_are_immutable() -> None:
    registry = ClassRegistry()
    with pytest.raises(OrganelleInputError):
        registry.delete("Chloroplast", objects="unclassify")
    with pytest.raises(OrganelleInputError):
        registry.rename("Vacuole", "V", migrate_objects="rename")


def test_delete_requires_an_explicit_object_policy() -> None:
    registry = ClassRegistry()
    registry.add_custom("LipidBody", defined_by="jane", created_at="")
    with pytest.raises(OrganelleInputError):
        registry.delete("LipidBody", objects="forget")  # type: ignore[arg-type]
    assert registry.get("LipidBody") is not None  # untouched


def test_delete_with_migrate_needs_a_valid_target() -> None:
    registry = ClassRegistry()
    registry.add_custom("LipidBody", defined_by="jane", created_at="")
    with pytest.raises(OrganelleInputError):
        registry.delete("LipidBody", objects="migrate", migrate_to="ghost")
    registry.delete("LipidBody", objects="migrate", migrate_to="Vacuole")
    assert registry.get("LipidBody") is None


def test_visibility_toggle() -> None:
    registry = ClassRegistry()
    hidden = registry.set_visible("Nucleus", False)
    assert hidden.visible is False
    assert registry.set_visible("Nucleus", True).visible is True
