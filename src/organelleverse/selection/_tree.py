"""Shared Newick parsing and exact tip selection for codeml and HyPhy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.errors import OrganelleInputError

_FOREGROUND = "Foreground"


@dataclass
class _Node:
    name: str = ""
    length: str | None = None
    children: list[_Node] = field(default_factory=list)
    marks: set[str] = field(default_factory=set)
    clade_marks: set[str] = field(default_factory=set)


def _mark_name(number: str) -> str:
    return _FOREGROUND if number == "1" else f"Group{number}"


def _parse_newick(text: str) -> _Node:
    """Parse Newick with codeml (``#1``/``$1``) and HyPhy (``{Tag}``) branch marks."""
    text = text.strip()
    if not text.endswith(";"):
        text += ";"
    pos = 0

    def error(message: str) -> OrganelleInputError:
        return OrganelleInputError(
            code="selection.hyphy_tree_parse_error",
            message=f"cannot parse Newick tree at character {pos}: {message}",
            details={"position": pos},
        )

    def skip_ws() -> None:
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1

    def read_annotations(node: _Node) -> None:
        nonlocal pos
        while True:
            skip_ws()
            if pos >= len(text):
                return
            char = text[pos]
            if char == ":":
                pos += 1
                skip_ws()
                match = re.match(r"[-+0-9.eE]+", text[pos:])
                if match is None:
                    raise error("branch length expected")
                node.length = match.group(0)
                pos += len(match.group(0))
            elif char in "#$":
                match = re.match(r"[#$](\d+)", text[pos:])
                if match is None:
                    raise error("codeml mark must be #<n> or $<n>")
                target = node.marks if char == "#" else node.clade_marks
                target.add(_mark_name(match.group(1)))
                pos += len(match.group(0))
            elif char == "{":
                end = text.find("}", pos)
                if end < 0:
                    raise error("unterminated {tag}")
                tag = text[pos + 1 : end].strip()
                if tag:
                    node.marks.add(tag)
                pos = end + 1
            elif char == "[":
                end = text.find("]", pos)
                if end < 0:
                    raise error("unterminated [comment]")
                pos = end + 1
            else:
                return

    def read_label() -> str:
        nonlocal pos
        skip_ws()
        if pos < len(text) and text[pos] == "'":
            end = text.find("'", pos + 1)
            if end < 0:
                raise error("unterminated quoted label")
            label = text[pos + 1 : end]
            pos = end + 1
            return label
        match = re.match(r"[^\s(),:;\[\]{}#$']+", text[pos:])
        if match is None:
            return ""
        pos += len(match.group(0))
        return match.group(0)

    def parse_node() -> _Node:
        nonlocal pos
        skip_ws()
        node = _Node()
        if pos < len(text) and text[pos] == "(":
            pos += 1
            while True:
                node.children.append(parse_node())
                skip_ws()
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == ")":
                    pos += 1
                    break
                raise error("expected ',' or ')'")
        node.name = read_label()
        read_annotations(node)
        return node

    root = parse_node()
    skip_ws()
    if pos >= len(text) or text[pos] != ";":
        raise error("trailing characters after tree")
    return root


def _tips(node: _Node) -> list[_Node]:
    if not node.children:
        return [node]
    return [tip for child in node.children for tip in _tips(child)]


def _propagate_clade_marks(node: _Node, inherited: frozenset[str] = frozenset()) -> None:
    active = inherited | frozenset(node.clade_marks)
    node.marks |= active
    for child in node.children:
        _propagate_clade_marks(child, active)


def _smallest_clade(node: _Node, targets: set[str]) -> _Node | None:
    names = {tip.name for tip in _tips(node)}
    if not targets <= names:
        return None
    for child in node.children:
        found = _smallest_clade(child, targets)
        if found is not None:
            return found
    return node


def _label_tips(root: _Node, labels: dict[str, set[str]], *, error_prefix: str) -> list[str]:
    """Validate and label exact leaf names for both selection backends."""
    tips = _tips(root)
    tip_names = [tip.name for tip in tips]
    if len(set(tip_names)) != len(tip_names) or any(not name for name in tip_names):
        raise OrganelleInputError(
            code=f"selection.{error_prefix}_tree_tip_names",
            message="tree tips must have unique, non-empty names",
            details={"tips": tip_names},
        )
    missing = sorted(set(labels) - set(tip_names))
    if missing:
        raise OrganelleInputError(
            code=f"selection.{error_prefix}_foreground_not_in_tree",
            message=f"foreground labels not found among tree tips: {missing}",
            details={"missing": missing, "tips": tip_names},
        )
    by_name = {tip.name: tip for tip in tips}
    for label, marks in labels.items():
        by_name[label].marks.update(marks)
    return tip_names


def _serialize_codeml(node: _Node) -> str:
    """Keep topology, node names, lengths and PAML branch/clade annotations."""
    text = ""
    if node.children:
        text = "(" + ",".join(_serialize_codeml(child) for child in node.children) + ")"
    name = node.name
    if any(c.isspace() or c in "(),:;[]{}#$" for c in name):
        name = f"'{name}'"
    text += name
    for prefix, marks in (("#", node.marks), ("$", node.clade_marks)):
        for mark in sorted(marks):
            number = "1" if mark == _FOREGROUND else mark.removeprefix("Group")
            if not number.isdigit():
                raise OrganelleInputError(
                    code="selection.codeml_tree_mark",
                    message=f"codeml requires numeric branch marks, got {mark!r}",
                )
            text += f" {prefix}{number}"
    if node.length is not None:
        text += ":" + node.length
    return text
