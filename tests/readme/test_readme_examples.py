"""Every ``ov.<module>.<function>(...)`` call shown in the READMEs must be real and well-formed."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import organelleverse as ov

ROOT = Path(__file__).resolve().parents[2]
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _ov_calls(readme: str) -> list[tuple[str, ast.Call]]:
    calls: list[tuple[str, ast.Call]] = []
    for block in _BLOCK.findall(readme):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call):
                continue
            parts: list[str] = []
            target: ast.expr = node.func
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name) and target.id == "ov" and parts:
                calls.append((".".join(reversed(parts)), node))
    return calls


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_examples_call_real_functions_with_valid_arguments(name: str) -> None:
    calls = _ov_calls((ROOT / name).read_text(encoding="utf-8"))
    assert len(calls) >= 30, "the quick start is expected to show the whole feature set"

    problems: list[str] = []
    for dotted, node in calls:
        target: object = ov
        try:
            for part in dotted.split("."):
                target = getattr(target, part)
            keywords = {kw.arg: None for kw in node.keywords if kw.arg is not None}
            inspect.signature(target).bind(*[None] * len(node.args), **keywords)  # type: ignore[arg-type]
        except (AttributeError, TypeError) as error:
            problems.append(f"ov.{dotted}: {error}")

    assert problems == []


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_example_data_files_exist(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    referenced = set(re.findall(r"examples/data/[\w.\-]+", text))
    assert referenced, "the quick start is expected to use the shipped example data"
    assert [p for p in sorted(referenced) if not (ROOT / p).is_file()] == []


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_lists_the_23_analysis_modules(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    listed = re.findall(r"^\| `(\w+)` \|", text, re.M)

    assert len(listed) == 23 and len(set(listed)) == 23
    assert [m for m in listed if not hasattr(ov, m)] == []


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_quick_start_has_a_section_per_module(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    table = re.findall(r"^\| `(\w+)` \|", text, re.M)
    sections = re.findall(r"^### `ov\.(\w+)`", text, re.M)

    assert sorted(sections) == sorted(table)
