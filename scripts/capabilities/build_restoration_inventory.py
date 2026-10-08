#!/usr/bin/env python3
"""Generate the frozen 253-capability restoration inventory from the archive.

Reads only the archive's file layout and AST (never imports archived code),
resolves every archived capability against the *current* source tree, and
writes a deterministic, portable TOML ledger. Exits non-zero, without writing
any output, if the archive's counts do not match the independently verified
246 base + 7 phenotype/CMS total, or if any archived capability cannot be
resolved to exactly one current callable.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BASE_SUITES = (
    "barcode",
    "codon_composition",
    "coevolution",
    "comparative",
    "composition",
    "diversity",
    "format_conversion",
    "hgt",
    "ir_boundary",
    "localization",
    "morphology",
    "pangenome",
    "phylogeny",
    "population",
    "rna_editing",
    "selection",
    "structure",
    "trans_splicing",
    "transfer",
    "variation",
    "visualization",
)
PHENOTYPE_SUITE = "phenotype"
ALL_SUITES = (*BASE_SUITES, PHENOTYPE_SUITE)

EXPECTED_BASE_TOTAL = 246
EXPECTED_PHENOTYPE_TOTAL = 7

EXPECTED_DOMAIN_COUNTS = {
    "barcode": 4,
    "codon_composition": 8,
    "coevolution": 26,
    "comparative": 13,
    "composition": 3,
    "diversity": 6,
    "format_conversion": 4,
    "hgt": 4,
    "ir_boundary": 3,
    "localization": 10,
    "morphology": 14,
    "pangenome": 9,
    "phylogeny": 20,
    "population": 9,
    "rna_editing": 14,
    "selection": 32,
    "structure": 10,
    "trans_splicing": 2,
    "transfer": 10,
    "variation": 5,
    "visualization": 40,
    "phenotype": 7,
}

EXECUTION_CLASSES = ("pure", "tool", "model", "network")
RESULT_SHAPES = ("canonical", "legacy_result", "json", "artifact", "custom")

_SUBPROCESS_ATTRS = {"run", "call", "check_call", "check_output", "Popen"}
_MODEL_PACKAGES = {"sklearn", "torch", "tensorflow", "skimage", "cv2", "statsmodels", "keras"}
_NETWORK_PACKAGES = {"requests", "urllib", "http", "httpx", "urllib3"}


class InventoryValidationError(Exception):
    """Raised when the archive cannot be turned into a valid, exact ledger."""


@dataclass(frozen=True)
class _RawCapability:
    domain: str
    archive_relpath: Path
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    bindings: dict[str, str]


def source_paths(root: Path, suite: str) -> tuple[Path, ...]:
    if suite == "visualization":
        return tuple(sorted((root / suite).glob("*.py")))
    return tuple(sorted((root / suite).rglob("*.py")))


def public_definitions(path: Path) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    )


def _import_bindings(path: Path) -> dict[str, str]:
    """Map each locally-bound import name to its top-level external package.

    Relative imports (``from . import x``) and imports of ``organelleverse``
    itself are intra-package, not external dependencies, and are excluded.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                local = alias.asname or alias.name.split(".")[0]
                if top != "organelleverse":
                    bindings[local] = top
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            if top == "organelleverse":
                continue
            for alias in node.names:
                bindings[alias.asname or alias.name] = top
    return bindings


def _function_dependencies(node: ast.FunctionDef | ast.AsyncFunctionDef, bindings: dict[str, str]) -> tuple[str, ...]:
    used = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    deps = {bindings[name] for name in used if name in bindings}
    deps -= set(sys.stdlib_module_names)
    return tuple(sorted(deps))


def _uses_subprocess(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "subprocess" and func.attr in _SUBPROCESS_ATTRS:
                return True
            if func.value.id == "os" and func.attr == "system":
                return True
    return False


def _execution_class(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str, dependencies: tuple[str, ...]) -> str:
    if name.startswith("run_") or _uses_subprocess(node):
        return "tool"
    dependency_set = set(dependencies)
    if dependency_set & _MODEL_PACKAGES:
        return "model"
    if dependency_set & _NETWORK_PACKAGES:
        return "network"
    return "pure"


def _annotation_name(annotation: ast.expr | None) -> str:
    if annotation is None:
        return ""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        return _annotation_name(annotation.value)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ""


def _result_shape(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str, domain: str) -> str:
    annotation_name = _annotation_name(node.returns)
    if annotation_name.endswith("Result"):
        return "canonical"
    if annotation_name in {"dict", "Dict"}:
        return "json"
    if domain == "visualization" or name.startswith(("plot_", "render_", "draw_")):
        return "artifact"
    if not annotation_name and name.startswith("run_"):
        return "legacy_result"
    return "custom"


def _collect_raw_capabilities(archive_root: Path) -> list[_RawCapability]:
    raw: list[_RawCapability] = []
    for suite in ALL_SUITES:
        for path in source_paths(archive_root, suite):
            bindings = _import_bindings(path)
            for node in public_definitions(path):
                raw.append(
                    _RawCapability(
                        domain=suite,
                        archive_relpath=path.relative_to(archive_root),
                        name=node.name,
                        node=node,
                        bindings=bindings,
                    )
                )
    return raw


def _check_counts(raw: list[_RawCapability]) -> None:
    base_total = sum(1 for item in raw if item.domain != PHENOTYPE_SUITE)
    phenotype_total = sum(1 for item in raw if item.domain == PHENOTYPE_SUITE)
    actual_domain_counts = dict(Counter(item.domain for item in raw))
    mismatches = {
        domain: (actual_domain_counts.get(domain, 0), expected)
        for domain, expected in EXPECTED_DOMAIN_COUNTS.items()
        if actual_domain_counts.get(domain, 0) != expected
    }

    if base_total != EXPECTED_BASE_TOTAL or phenotype_total != EXPECTED_PHENOTYPE_TOTAL or mismatches:
        raise InventoryValidationError(
            f"expected {EXPECTED_BASE_TOTAL} base + {EXPECTED_PHENOTYPE_TOTAL} phenotype "
            f"capabilities, got {base_total} base + {phenotype_total} phenotype; "
            f"per-domain mismatches (domain: (actual, expected)): {mismatches}"
        )


def _find_current_definition(path: Path, name: str) -> bool:
    """Whether ``name`` is defined as a top-level function in ``path``.

    Deliberately not restricted to public (non-underscore) names: a locator
    only needs the exact name to still exist at module level to stay
    reachable via ``getattr``, regardless of naming convention.
    """
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in tree.body
    )


def _resolve_current_relpath(current_root: Path, domain: str, archive_relpath: Path, name: str) -> Path:
    """Find where ``name`` lives in current source; fail closed if it cannot be found.

    Checks the archived relative path first (the common case: this suite's
    relative layout was preserved verbatim during restoration). Only if that
    fails does it search the rest of the domain, so a genuine rename/move is
    still resolved to its canonical current locator rather than silently
    dropped, and a capability with no current counterpart at all — or more
    than one candidate — stops the generator instead of guessing.
    """
    candidate = current_root / archive_relpath
    if _find_current_definition(candidate, name):
        return archive_relpath

    domain_root = current_root / domain
    matches: list[Path] = []
    if domain_root.is_dir():
        for path in sorted(domain_root.rglob("*.py")):
            if _find_current_definition(path, name):
                matches.append(path.relative_to(current_root))

    if len(matches) == 1:
        print(
            f"NOTE: {domain}.{name} archived at {archive_relpath} now lives at {matches[0]}",
            file=sys.stderr,
        )
        return matches[0]
    if not matches:
        raise InventoryValidationError(
            f"unresolvable archived capability: {domain}.{name} (archived at "
            f"{archive_relpath}) has no matching callable anywhere under current "
            f"src/organelleverse/{domain}/"
        )
    raise InventoryValidationError(
        f"ambiguous archived capability: {domain}.{name} (archived at "
        f"{archive_relpath}) matches multiple current locations: "
        f"{[str(match) for match in matches]}"
    )


def _build_locator(source_relpath: Path, name: str) -> str:
    parts = list(source_relpath.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    module = ".".join(parts)
    return f"organelleverse.{module}:{name}"


def _assign_ids(records: list[dict]) -> None:
    counts = Counter((record["domain"], record["public_name"]) for record in records)
    for record in records:
        key = (record["domain"], record["public_name"])
        if counts[key] > 1:
            module_stem = Path(record["source_relpath"]).with_suffix("").stem
            record["id"] = f"{record['domain']}.{module_stem}.{record['public_name']}"
        else:
            record["id"] = f"{record['domain']}.{record['public_name']}"


def _check_no_duplicates(records: list[dict]) -> None:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise InventoryValidationError(f"duplicate capability ids after assignment: {duplicates}")

    locators = [record["python_locator"] for record in records]
    if len(locators) != len(set(locators)):
        duplicates = sorted({item for item in locators if locators.count(item) > 1})
        raise InventoryValidationError(f"duplicate python_locator values: {duplicates}")


def build_inventory_records(archive_root: Path, current_root: Path) -> list[dict]:
    raw = _collect_raw_capabilities(archive_root)
    _check_counts(raw)

    records: list[dict] = []
    for item in raw:
        source_relpath = _resolve_current_relpath(current_root, item.domain, item.archive_relpath, item.name)
        dependencies = _function_dependencies(item.node, item.bindings)
        records.append(
            {
                "domain": item.domain,
                "public_name": item.name,
                "python_locator": _build_locator(source_relpath, item.name),
                "source_relpath": source_relpath.as_posix(),
                "execution_class": _execution_class(item.node, item.name, dependencies),
                "result_shape": _result_shape(item.node, item.name, item.domain),
                "dependencies": list(dependencies),
            }
        )

    _assign_ids(records)
    _check_no_duplicates(records)
    records.sort(key=lambda record: (record["domain"], record["source_relpath"], record["public_name"]))
    return records


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_string(value: str) -> str:
    return f'"{_toml_escape(value)}"'


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def render_toml(records: list[dict]) -> str:
    lines = [
        "# Generated by scripts/capabilities/build_restoration_inventory.py",
        "# Do not edit by hand: regenerate from the archive root.",
        "",
    ]
    for record in records:
        lines.append("[[capability]]")
        lines.append(f"id = {_toml_string(record['id'])}")
        lines.append(f"domain = {_toml_string(record['domain'])}")
        lines.append(f"public_name = {_toml_string(record['public_name'])}")
        lines.append(f"python_locator = {_toml_string(record['python_locator'])}")
        lines.append(f"source_relpath = {_toml_string(record['source_relpath'])}")
        lines.append(f"execution_class = {_toml_string(record['execution_class'])}")
        lines.append(f"result_shape = {_toml_string(record['result_shape'])}")
        lines.append(f"dependencies = {_toml_array(record['dependencies'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _default_current_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "organelleverse"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    archive_root = args.archive_root
    if not archive_root.is_dir():
        print(f"error: --archive-root does not exist or is not a directory: {archive_root}", file=sys.stderr)
        return 1

    try:
        records = build_inventory_records(archive_root, _default_current_root())
        rendered = render_toml(records)
        import tomllib

        round_tripped = tomllib.loads(rendered)
        if len(round_tripped.get("capability", [])) != len(records):
            raise InventoryValidationError(
                "rendered TOML did not round-trip to the same record count "
                f"({len(round_tripped.get('capability', []))} != {len(records)})"
            )
    except InventoryValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")

    base_total = sum(1 for record in records if record["domain"] != PHENOTYPE_SUITE)
    phenotype_total = sum(1 for record in records if record["domain"] == PHENOTYPE_SUITE)
    print(f"base archived capabilities: {base_total}")
    print(f"phenotype/CMS capabilities: {phenotype_total}")
    print(f"total restoration capabilities: {base_total + phenotype_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
