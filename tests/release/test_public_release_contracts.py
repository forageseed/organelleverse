"""Static release checks for public compute-then-write examples and gates."""

from __future__ import annotations

import ast
import inspect
import json
import re
import runpy
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

import organelleverse as ov
from organelleverse import operations
from organelleverse.assembly.contracts import AssemblyRequest

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_GATE = ROOT / "scripts" / "annotation_benchmarks" / "run_annotation_release_gate.py"
HIMT_GATE = ROOT / "tests" / "release" / "test_assembly_himt_release_gate.py"
OATK_GATE = ROOT / "tests" / "release" / "test_assembly_oatk_release_gate.py"


def _literal_key(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_value(node: ast.Dict, key: str) -> ast.expr | None:
    for candidate, value in zip(node.keys, node.values, strict=True):
        if _literal_key(candidate) == key:
            return value
    return None


def _python_trees(path: Path) -> Iterator[ast.Module]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    yield tree
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if "annotation.annotate" not in node.value and "annotation.extract" not in node.value:
            continue
        try:
            yield ast.parse(textwrap.dedent(node.value), filename=f"{path}:embedded")
        except SyntaxError:
            continue


def _agent_invocations(tree: ast.AST) -> Iterator[tuple[str, ast.Dict]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        operation = _dict_value(node, "operation_id")
        parameters = _dict_value(node, "parameters")
        if (
            isinstance(operation, ast.Constant)
            and isinstance(operation.value, str)
            and isinstance(parameters, ast.Dict)
        ):
            yield operation.value, parameters


def _parameter_keys(parameters: ast.Dict) -> set[str]:
    return {key for node in parameters.keys if (key := _literal_key(node)) is not None}


def test_annotation_release_gate_calls_match_current_public_signatures() -> None:
    trees = list(_python_trees(ANNOTATION_GATE))
    gate = runpy.run_path(str(ANNOTATION_GATE))
    clean_wheel_probe = cast(object, gate["_clean_wheel_probe"])
    assert callable(clean_wheel_probe)
    trees.append(ast.parse(clean_wheel_probe(ROOT / "src"), filename="clean-wheel-probe"))

    public_signatures = {
        "annotate": inspect.signature(ov.annotation.annotate),
        "extract": inspect.signature(ov.annotation.extract),
    }
    stale: list[str] = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            signature = public_signatures.get(node.func.attr)
            if signature is None:
                continue
            allowed = set(signature.parameters)
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in allowed:
                    stale.append(f"{node.func.attr}:{keyword.arg}")

    assert stale == []


def test_release_gate_agent_parameters_match_invocation_schemas() -> None:
    gate = runpy.run_path(str(ANNOTATION_GATE))
    clean_wheel_probe = cast(object, gate["_clean_wheel_probe"])
    assert callable(clean_wheel_probe)
    trees = [
        *_python_trees(ANNOTATION_GATE),
        ast.parse(clean_wheel_probe(ROOT / "src"), filename="clean-wheel-probe"),
        ast.parse(HIMT_GATE.read_text(encoding="utf-8"), filename=str(HIMT_GATE)),
        ast.parse(OATK_GATE.read_text(encoding="utf-8"), filename=str(OATK_GATE)),
    ]
    checked = 0
    stale: list[str] = []
    for tree in trees:
        for operation_id, parameters in _agent_invocations(tree):
            if operation_id not in {"annotation.annotate", "assembly.assemble"}:
                continue
            schema = operations.registry.invocation_schema(operation_id)
            parameter_schema = schema["properties"]["parameters"]
            allowed = set(parameter_schema["properties"])
            extras = _parameter_keys(parameters) - allowed
            if extras:
                stale.append(f"{operation_id}:{sorted(extras)}")
            checked += 1

    assert checked >= 5
    assert stale == []


def test_himt_release_gate_uses_compute_then_write_boundary() -> None:
    tree = ast.parse(HIMT_GATE.read_text(encoding="utf-8"), filename=str(HIMT_GATE))
    request_parameters = set(inspect.signature(AssemblyRequest).parameters)
    stale_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "output_dir"
        or (
            isinstance(node.func, ast.Name)
            and node.func.id == "AssemblyRequest"
            and keyword.arg is not None
            and keyword.arg not in request_parameters
        )
    ]

    assert stale_lines == []


def test_oatk_release_gate_selects_oatk_explicitly() -> None:
    tree = ast.parse(OATK_GATE.read_text(encoding="utf-8"), filename=str(OATK_GATE))
    stale_lines: list[int] = []
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name == "AssemblyRequest":
                methods = [keyword.value for keyword in node.keywords if keyword.arg == "method"]
                if not (
                    len(methods) == 1
                    and isinstance(methods[0], ast.Constant)
                    and methods[0].value == "oatk"
                ):
                    stale_lines.append(node.lineno)
                checked += 1
        if isinstance(node, ast.Dict):
            operation = _dict_value(node, "operation_id")
            parameters = _dict_value(node, "parameters")
            if not (
                isinstance(operation, ast.Constant)
                and operation.value == "assembly.assemble"
                and isinstance(parameters, ast.Dict)
            ):
                continue
            method = _dict_value(parameters, "method")
            if not (isinstance(method, ast.Constant) and method.value == "oatk"):
                stale_lines.append(node.lineno)
            checked += 1

    assert checked >= 4
    assert stale_lines == []


def test_readme_agent_json_examples_validate_against_invocation_schema() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```json\n(.*?)\n```", readme, flags=re.DOTALL)

    assert examples
    for example in examples:
        invocation = json.loads(example)
        operation_id = invocation["operation_id"]
        schema = operations.registry.invocation_schema(operation_id)
        Draft202012Validator(schema).validate(invocation)
