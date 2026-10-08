"""Every suggestion emitted by released code must name a released operation.

This is the static half of the class C detector's regression coverage. The
runtime half arrives in Task 4.
"""

import ast
from pathlib import Path

from organelleverse import operations as op

SRC = Path(__file__).resolve().parents[2] / "src" / "organelleverse"


def _suggested_operation_ids() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "OperationSuggestion":
                continue
            for keyword in node.keywords:
                if keyword.arg == "operation_id" and isinstance(keyword.value, ast.Constant):
                    found.append((str(path.relative_to(SRC)), str(keyword.value.value)))
    return found


def test_released_code_emits_no_suggestion_outside_the_catalog() -> None:
    catalog = {spec.operation_id for spec in op.list()}
    offenders = [entry for entry in _suggested_operation_ids() if entry[1] not in catalog]
    assert offenders == []


def test_assembly_and_qc_specs_have_no_suggestion_literals() -> None:
    emitted = {entry[0] for entry in _suggested_operation_ids()}
    assert "assembly/service.py" not in emitted
    assert "quality_control/service.py" not in emitted
