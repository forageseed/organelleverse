"""Triage the unmigrated restoration-ledger capabilities (P2, Decision 004).

Diffs ``docs/operations/restored-capabilities.toml`` (253 records) against the
checked-in bundle ``callable_locator``s (166: 150 restored + 16 release
operations) and classifies the 103 unmigrated records with transparent,
mechanical rules. Every emitted row carries a *suggested* verdict and the
signals that produced it; the verdict itself is the owner's to rule
(Decision 004: every ledger item gets publish / merge / internal / reject).

Verdict buckets emitted:

- ``publish``: science function with a working precedent (canonical shape,
  matplotlib artifact plot, or annotated JSON-safe result). Needs an adapter
  override and fixtures.
- ``publish-after-fix``: same, but the signature is not fully annotated, so
  binding cannot be derived until the suite code is annotated.
- ``internal-l6``: writer (``write_*`` / ``save_*``). The owner has ruled that
  writers belong to the L6 publication path (``materialize_result``), not to
  the published science-capability surface.
- ``internal``: private-module surfaces, path/config getters, and other suite
  implementation details.
- ``blocked-item-4``: external-tool execution; the bundle contract cannot yet
  express executable-probe verification for package code (Decision 004
  item 4, same lane as ``annotation.annotate``).
- ``reject``: UI shells, input parsers, and stale inventory records whose
  locator no longer resolves.
- ``needs-ruling``: custom-shaped science functions where a codec design
  decision is required before an override can be written.

Usage::

    python scripts/capabilities/triage_unmigrated_capabilities.py \
        --output docs/operations/unmigrated-capability-triage.toml
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_LEDGER = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CAPABILITIES = PROJECT_ROOT / "src" / "organelleverse" / "capabilities"

WRITER_PREFIXES = ("write_", "save_")
UI_NAMES = {"build_app", "serve"}
PARSER_PREFIXES = ("parse_", "read_")
PATH_GETTER_SUFFIXES = ("_dir", "_path")

#: Mirror of OperationSpec.operation_id's pattern (one dot, lowercase segments).
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Signals:
    """AST-level facts about the record's target callable."""

    resolves: bool
    reason: str  # "ok", "missing-module", "not-found-in-source"
    annotated: bool
    is_alias: bool


@dataclass(frozen=True)
class TriageRow:
    capability_id: str
    domain: str
    python_locator: str
    execution_class: str
    result_shape: str
    dependencies: tuple[str, ...]
    signals: tuple[str, ...]
    suggested_verdict: str
    reason: str


def _analyze(locator: str) -> Signals:
    module_name, function_name = locator.split(":")
    path = PROJECT_ROOT / "src" / (module_name.replace(".", "/") + ".py")
    if not path.exists():
        return Signals(False, "missing-module", annotated=False, is_alias=False)
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            unannotated = any(
                arg.annotation is None
                for arg in (*node.args.args, *node.args.kwonlyargs)
                if arg.arg not in ("self", "cls")
            )
            return Signals(
                True,
                "ok",
                annotated=not (unannotated or node.returns is None),
                is_alias=False,
            )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == function_name
            for target in node.targets
        ):
            return Signals(True, "ok", annotated=True, is_alias=True)
    return Signals(False, "not-found-in-source", annotated=False, is_alias=False)


def _classify(record: dict, signals: Signals) -> tuple[str, str, tuple[str, ...]]:
    """Return (suggested_verdict, reason, extra_signal_tags)."""

    name = record["public_name"]
    shape = record["result_shape"]
    deps = tuple(record["dependencies"])

    if not signals.resolves:
        return "reject", f"stale inventory record: locator {signals.reason}", ()
    if signals.is_alias:
        return "reject", "module-level alias assignment, not a definition", ("alias",)

    module_name = record["python_locator"].split(":")[0]
    if any(part.startswith("_") for part in module_name.split(".")[1:]):
        reason = "private-module implementation surface"
        if not _ID_PATTERN.fullmatch(record["id"]):
            reason += "; the id also violates the one-dot capability id grammar"
        return "internal", reason, ("private-module",)

    if name in UI_NAMES or "gradio" in deps:
        return "reject", "UI shell (gradio/serve), not a science capability", ("ui",)

    if name.startswith(PARSER_PREFIXES) and ({"Bio", "gfapy"} & set(deps) or shape == "custom"):
        return (
            "reject",
            "input parser; core inputs should arrive through the io suites",
            ("parser",),
        )

    if record["execution_class"] == "tool":
        return (
            "blocked-item-4",
            "external-tool execution; bundle contract cannot yet express "
            "executable-probe verification for package code",
            ("external-tool",),
        )

    if name.endswith(PATH_GETTER_SUFFIXES) and shape == "custom" and not deps:
        return "internal", "path/config getter, a suite implementation detail", ("path-getter",)

    if "matplotlib" in deps and shape == "artifact":
        return (
            "publish",
            "matplotlib artifact plot; precedent: 10 admitted visualization bundles",
            ("artifact-plot",),
        )

    if name.startswith(WRITER_PREFIXES):
        return (
            "internal-l6",
            "writer; the L6 publication path (materialize_result) owns file emission",
            ("writer",),
        )

    if not signals.annotated:
        return (
            "publish-after-fix",
            "signature is not fully annotated; binding cannot be derived until it is",
            ("unannotated",),
        )

    if shape == "canonical":
        return "publish", "canonical result shape; needs an adapter override", ()
    if shape == "artifact":
        return "publish", "artifact result shape; artifact codec precedent exists", ()
    if shape == "json":
        return "publish", "annotated JSON-safe result; json codec", ()

    return (
        "needs-ruling",
        f"custom result shape on a science function ({name}); codec design decision required",
        ("custom-shape",),
    )


def triage() -> list[TriageRow]:
    records = tomllib.loads(_LEDGER.read_text(encoding="utf-8"))["capability"]
    bundled = {
        tomllib.loads(path.read_text(encoding="utf-8"))["contract"]["callable_locator"]
        for path in _CAPABILITIES.rglob("capability.toml")
    }
    rows: list[TriageRow] = []
    for record in records:
        if record["python_locator"] in bundled:
            continue
        signals = _analyze(record["python_locator"])
        verdict, reason, tags = _classify(record, signals)
        signal_tags = list(tags)
        if signals.annotated and signals.resolves:
            signal_tags.append("annotated")
        rows.append(
            TriageRow(
                capability_id=record["id"],
                domain=record["domain"],
                python_locator=record["python_locator"],
                execution_class=record["execution_class"],
                result_shape=record["result_shape"],
                dependencies=tuple(record["dependencies"]),
                signals=tuple(signal_tags),
                suggested_verdict=verdict,
                reason=reason,
            )
        )
    return rows


def render(rows: list[TriageRow]) -> str:
    lines = [
        "# Generated by scripts/capabilities/triage_unmigrated_capabilities.py",
        "# Suggested verdicts are mechanical; the owner rules every item (Decision 004).",
        "",
    ]
    for row in sorted(rows, key=lambda item: item.capability_id):
        lines.append("[[capability]]")
        lines.append(f'id = "{row.capability_id}"')
        lines.append(f'domain = "{row.domain}"')
        lines.append(f'python_locator = "{row.python_locator}"')
        lines.append(f'execution_class = "{row.execution_class}"')
        lines.append(f'result_shape = "{row.result_shape}"')
        deps = ", ".join(f'"{dep}"' for dep in row.dependencies)
        lines.append(f"dependencies = [{deps}]")
        signals = ", ".join(f'"{signal}"' for signal in row.signals)
        lines.append(f"signals = [{signals}]")
        lines.append(f'suggested_verdict = "{row.suggested_verdict}"')
        lines.append(f'reason = "{row.reason}"')
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = triage()
    if len(rows) != 103:
        print(f"error: expected 103 unmigrated records, triaged {len(rows)}", file=sys.stderr)
        return 1
    args.output.write_text(render(rows), encoding="utf-8", newline="\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.suggested_verdict] = counts.get(row.suggested_verdict, 0) + 1
    for verdict in sorted(counts):
        print(f"{verdict}: {counts[verdict]}")
    print(f"total: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
