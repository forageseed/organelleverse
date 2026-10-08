#!/usr/bin/env python3
"""Generate core capability bundles for one domain of the restoration ledger.

Reads ``docs/operations/restored-capabilities.toml`` (frozen by
``build_restoration_inventory.py``) and, for one requested domain, emits one
``capability.toml`` bundle directory per ledger record under
``src/organelleverse/capabilities/`` - the one root ``discovery.py`` labels
origin ``"core"`` (``discovery.py:93``, ``SearchRoot("core", package_root /
"capabilities")``).

**AST-only, never an import.** Every fact this generator derives about a
restored capability's real Python signature and docstring comes from
``ast.parse`` on its source file, exactly like ``build_restoration_inventory.py``
already does for the ledger itself. Module-level type aliases (e.g.
``visualization/gbdraw.py``'s ``PathInput = str | Path``) are resolved from
that same already-parsed tree - reading the alias's definition site, never
importing it - so a ``PathInput``-annotated parameter validates for the
``path`` codec exactly as a bare ``str | Path`` would. It never imports
``organelleverse.<domain>.<module>`` to introspect a live function - doing so
would run arbitrary scientific code (heavy optional dependencies, network
calls, subprocess spawns) merely to generate a bundle description. The one
import this script does perform is ``organelleverse.capabilities.adapters.<domain>``:
a small, hand-authored, dependency-free data module (see
``adapters/__init__.py``), not scientific code, and
``organelleverse.capabilities.parser.parse_capability_bundle`` to
self-validate what it just wrote (mirroring ``scaffold.py``'s own
dogfooding) - parsing never imports ``contract.callable_locator`` either.

**Two binding shapes, one generic and one adapter-gated.** A capability whose
first parameter is typed exactly ``OrganelleGenome``/``OrganelleData``/
``OrganelleResult`` (by bare name or a resolved import alias) and whose
return annotation is one of the same three is recognized automatically as
``canonical_core`` - the same signature-derivation path the 20 released
operations already use, requiring no adapter input at all. Every other
capability requires an explicit ``NamedParameterOverride`` from that domain's
adapter module; absent one, the generator FAILS CLOSED for that capability -
skips it, with a structured diagnostic on stderr - rather than guessing a
parameter codec or a result codec it cannot honestly justify.

**Deterministic.** Ledger records are processed in the ledger's own sorted
order; every derived value (title, description, keywords) is a pure function
of the ledger record and the AST it derives from - no timestamps, no
filesystem iteration order, no randomness. Re-running against an unchanged
ledger and source tree reproduces byte-identical ``capability.toml`` files.

Runtime never runs this generator: generated bundles are checked in, exactly
like the restoration inventory itself.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from organelleverse.capabilities.adapters import FixtureCase
from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.capabilities.verification import strip_run_provenance
from organelleverse.operations.spec import CoreKind, OperationStage, ParameterCodec, SideEffect

_MIN_DESCRIPTION_LENGTH = 20
_CORE_KIND_BY_TYPE_NAME = {
    "OrganelleGenome": CoreKind.GENOME,
    "OrganelleData": CoreKind.DATA,
    "OrganelleResult": CoreKind.RESULT,
}
_STAGE_BY_KIND_PAIR: dict[tuple[CoreKind, CoreKind], OperationStage] = {
    (CoreKind.NONE, CoreKind.GENOME): OperationStage.READ,
    (CoreKind.NONE, CoreKind.DATA): OperationStage.READ,
    (CoreKind.GENOME, CoreKind.GENOME): OperationStage.TRANSFORM,
    (CoreKind.GENOME, CoreKind.DATA): OperationStage.TRANSFORM,
    (CoreKind.DATA, CoreKind.GENOME): OperationStage.TRANSFORM,
    (CoreKind.DATA, CoreKind.DATA): OperationStage.TRANSFORM,
    (CoreKind.NONE, CoreKind.RESULT): OperationStage.ANALYZE,
    (CoreKind.GENOME, CoreKind.RESULT): OperationStage.ANALYZE,
    (CoreKind.DATA, CoreKind.RESULT): OperationStage.ANALYZE,
    (CoreKind.RESULT, CoreKind.RESULT): OperationStage.CONSUME,
}
_JSON_SAFE_NAMES = frozenset({"str", "int", "float", "bool"})
_JSON_SAFE_CONTAINERS = frozenset({"list", "dict", "tuple"})
# ``collections.abc``/``typing`` aliases for the two container shapes the
# real runtime binder's ``_JSON_SEQUENCE_ORIGINS``/``_JSON_MAPPING_ORIGINS``
# (``operations/signature.py:50-51``) resolve to the exact same origin as
# their bare counterparts (``get_origin(Sequence[X]) is collections.abc
# .Sequence``, which is exactly what that frozenset contains - same for
# ``Mapping``/``dict``). Both the bare spellings and the aliases get real,
# element-wise checking - required to keep rejecting ``Mapping[str, Any]``/
# ``Sequence[Mapping[str, Any]]`` (``Any`` is independently unsafe at the
# real runtime binder), and the same recursion is what keeps ``Path`` out of
# the JSON-safe set at any nesting depth, matching the strict codec-wall
# variant of ``_is_supported_json_annotation`` (``allow_path=False``).
_JSON_SAFE_SEQUENCE_ALIASES = frozenset({"Sequence"})
_JSON_SAFE_MAPPING_ALIASES = frozenset({"Mapping"})


class BundleGenerationError(Exception):
    """Raised when the requested domain cannot be generated at all."""


@dataclass(frozen=True)
class PlannedParameter:
    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class RenderedFixture:
    """One ``[[fixture]]`` block, fully resolved and ready to render.

    Produced only by ``_capture_fixtures`` - never hand-constructed - after
    it has actually run the capability's real implementation and frozen its
    real return value. ``parameters`` and ``expect`` are already
    bundle-root-relative strings (or literal JSON values), exactly the shape
    ``organelleverse.capabilities.models.FixtureSpec`` expects.
    """

    case: str
    parameters: dict[str, object]
    expect: str
    equivalence: str
    tolerance: float | None = None


@dataclass(frozen=True)
class PlannedBundle:
    """Every field a ``capability.toml`` needs, fully resolved and ready to render."""

    capability_id: str
    title: str
    description: str
    keywords: tuple[str, ...]
    stage: OperationStage
    input_kind: CoreKind
    output_kind: CoreKind
    callable_locator: str
    argument_mode: str
    parameters: tuple[PlannedParameter, ...]
    result_codec: str
    result_key: str | None
    side_effects: tuple[SideEffect, ...]
    dependencies: tuple[str, ...]
    input_sequence: bool = False
    deterministic: bool = True
    deterministic_reason: str = ""
    extra_keywords: tuple[str, ...] = ()
    fixtures: tuple[RenderedFixture, ...] = ()


@dataclass(frozen=True)
class SkippedCapability:
    """One ledger record this run deliberately did not turn into a bundle."""

    capability_id: str
    code: str
    message: str


@dataclass(frozen=True)
class GeneratedBundle:
    capability_id: str
    path: Path


@dataclass(frozen=True)
class BuildReport:
    domain: str
    generated: tuple[GeneratedBundle, ...]
    skipped: tuple[SkippedCapability, ...]


# --- ledger -------------------------------------------------------------


def _load_domain_records(ledger_path: Path, domain: str) -> list[dict[str, Any]]:
    payload = tomllib.loads(ledger_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = [
        record for record in payload["capability"] if record["domain"] == domain
    ]
    records.sort(key=lambda record: record["id"])
    return records


# --- AST introspection (never an import) ---------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _description(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    docstring = ast.get_docstring(node, clean=True)
    if not docstring:
        return None
    first_paragraph = docstring.split("\n\n", 1)[0]
    collapsed = " ".join(first_paragraph.split())
    if len(collapsed) < _MIN_DESCRIPTION_LENGTH:
        return None
    return collapsed


def _title(public_name: str) -> str:
    return " ".join(word.capitalize() for word in public_name.split("_") if word)


def _keywords(
    domain: str,
    public_name: str,
    *,
    stage: OperationStage,
    input_kind: CoreKind,
    output_kind: CoreKind,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    pool = {token for token in public_name.split("_") if token}
    pool.add(domain)
    pool.add("restored")
    pool.add(stage.value)
    if input_kind is not CoreKind.NONE:
        pool.add(input_kind.value)
    pool.add(output_kind.value)
    pool.update(extra)
    ordered = sorted(pool)
    return tuple(ordered[:8])


def _core_type_import_map(tree: ast.Module) -> dict[str, str]:
    """Map each locally-bound import name to one of the three core type names.

    Only ``from ... import OrganelleGenome`` (and its aliased/renamed form)
    is tracked - a plain ``import organelleverse.core.genome`` is never used
    for these types anywhere in the restored source, so it is deliberately
    not handled here.
    """
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _CORE_KIND_BY_TYPE_NAME:
                    mapping[alias.asname or alias.name] = alias.name
    return mapping


def _plain_core_type_name(expr: ast.expr | None, import_map: dict[str, str]) -> str | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Name):
        if expr.id in import_map:
            return import_map[expr.id]
        if expr.id in _CORE_KIND_BY_TYPE_NAME:
            return expr.id
        return None
    if isinstance(expr, ast.Attribute) and expr.attr in _CORE_KIND_BY_TYPE_NAME:
        return expr.attr
    return None


def _sequence_core_type_name(expr: ast.expr | None, import_map: dict[str, str]) -> str | None:
    """Ruling 3: recognize list[T]/Sequence[T]/tuple[T, ...] of one core type.

    A one-argument ``list``/``Sequence``/``tuple`` subscript whose element is
    a plain core type resolves to that core name; anything else (multi-arg,
    non-core element, nested sequences) returns ``None`` so the caller falls
    through to the domain adapter exactly as before.
    """
    if not isinstance(expr, ast.Subscript) or not isinstance(expr.value, ast.Name):
        return None
    if expr.value.id not in {"list", "Sequence", "tuple"}:
        return None
    if not isinstance(expr.slice, ast.expr) or isinstance(expr.slice, ast.Tuple):
        return None
    return _plain_core_type_name(expr.slice, import_map)


def _detect_canonical_core(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    import_map: dict[str, str],
) -> tuple[CoreKind, CoreKind, OperationStage] | None:
    """Recognize the framework's generic ``canonical_core`` shape, or refuse to guess.

    Mirrors what ``organelleverse.operations.signature.derive_operation_signature``
    itself requires at bind time: a required first positional parameter typed
    exactly one of the three core types, and a return annotation of the same
    closed set, in one of the four combinations ``OperationSpec`` accepts for
    a stage. Anything else - a list of core objects, a union, an untyped
    ``list`` - returns ``None`` so the caller falls through to the domain
    adapter instead of fabricating a binding that would fail the first time
    it was actually verified.
    """
    positional = [*node.args.posonlyargs, *node.args.args]
    if not positional:
        return None
    defaults_start = len(positional) - len(node.args.defaults)
    if defaults_start <= 0:
        return None  # the first positional parameter carries a default
    input_sequence = _sequence_core_type_name(positional[0].annotation, import_map) is not None
    input_name = (
        _sequence_core_type_name(positional[0].annotation, import_map)
        if input_sequence
        else _plain_core_type_name(positional[0].annotation, import_map)
    )
    if input_name is None:
        return None
    output_name = _plain_core_type_name(node.returns, import_map)
    if output_name is None:
        return None
    input_kind = _CORE_KIND_BY_TYPE_NAME[input_name]
    output_kind = _CORE_KIND_BY_TYPE_NAME[output_name]
    stage = _STAGE_BY_KIND_PAIR.get((input_kind, output_kind))
    if stage is None:
        return None
    return input_kind, output_kind, stage, input_sequence


def _looks_like_path_annotation(
    expr: ast.expr | None,
    aliases: dict[str, ast.expr] | None = None,
    _seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expr, ast.Name):
        if expr.id in {"str", "Path"}:
            return True
        # resolve a module-level type alias (e.g. PathInput = str | Path) from
        # the module's own AST - never an import; guard against alias cycles
        if aliases is not None and expr.id in aliases and expr.id not in _seen:
            return _looks_like_path_annotation(aliases[expr.id], aliases, _seen | {expr.id})
        return False
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return _looks_like_path_annotation(expr.left, aliases, _seen) or (
            _looks_like_path_annotation(expr.right, aliases, _seen)
        )
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
        # a one-argument list/Sequence of path-likes (the PATH codec's list
        # form, e.g. ``list[str | Path]``) - mirrors the runtime binder's
        # ``_resolve_path_parameter_plan`` list branch
        return expr.value.id in {"list", "Sequence"} and _looks_like_path_annotation(
            expr.slice, aliases, _seen
        )
    return False


def _looks_like_directory_annotation(expr: ast.expr | None) -> bool:
    """True for a scalar ``str``/``Path`` annotation - never a list.

    Mirrors ``_looks_like_path_annotation``'s scalar branches (a bare name or
    a union naming one of ``str``/``Path``, an unrelated arm such as ``None``
    simply not disqualifying it) but deliberately drops the list/Sequence
    branch: the real runtime binder's ``_resolve_directory_parameter_plan``
    (``operations/python_binding.py``) has no list form at all - a directory
    parameter names exactly one tree or destination, unlike PATH's list of
    files.
    """
    if isinstance(expr, ast.Name):
        return expr.id in {"str", "Path"}
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return _looks_like_directory_annotation(expr.left) or _looks_like_directory_annotation(
            expr.right
        )
    return False


def _looks_json_safe(expr: ast.expr | None) -> bool:
    if expr is None:
        return False
    if isinstance(expr, ast.Constant) and expr.value is None:
        return True
    if isinstance(expr, ast.Name):
        return expr.id in _JSON_SAFE_NAMES
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return _looks_json_safe(expr.left) and _looks_json_safe(expr.right)
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
        container = expr.value.id
        if container in _JSON_SAFE_CONTAINERS:
            # Fully recursive, matching the strict codec-wall variant of the
            # runtime binder (``allow_path=False``): the outer container name
            # alone is never enough - every type argument must itself be
            # JSON-safe, which is what withdraws ``Path`` at any nesting
            # depth (``list[str | Path]``) and rejects a bare, unparameterized
            # container element (``list[dict]`` - a bare ``Name`` outside
            # ``_JSON_SAFE_NAMES``) at generation time too.
            elements = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else (expr.slice,)
            if container == "dict":
                return (
                    len(elements) == 2
                    and isinstance(elements[0], ast.Name)
                    and elements[0].id == "str"
                    and _looks_json_safe(elements[1])
                )
            if container == "tuple":
                if (
                    len(elements) == 2
                    and isinstance(elements[1], ast.Constant)
                    and elements[1].value is Ellipsis
                ):
                    return _looks_json_safe(elements[0])
                return bool(elements) and all(_looks_json_safe(element) for element in elements)
            return len(elements) == 1 and _looks_json_safe(elements[0])
        if container in _JSON_SAFE_SEQUENCE_ALIASES:
            return _looks_json_safe(expr.slice)
        if container in _JSON_SAFE_MAPPING_ALIASES:
            if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
                key, value = expr.slice.elts
                return isinstance(key, ast.Name) and key.id == "str" and _looks_json_safe(value)
            return False
    return False


def _module_type_aliases(tree: ast.Module) -> dict[str, ast.expr]:
    """Collect the module's top-level simple type aliases, AST-only.

    ``PathInput = str | Path`` at module scope (``visualization/gbdraw.py``)
    is a plain ``ast.Assign`` with one ``Name`` target; reading its value
    expression from the already-parsed tree resolves the alias without ever
    importing the module - the same discipline every other fact in this
    generator follows. Only single-name assignments with a type-shaped value
    (union / name / subscript) are collected; anything else is not an alias.
    """
    aliases: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        if isinstance(stmt.value, (ast.BinOp, ast.Name, ast.Subscript)):
            aliases[stmt.targets[0].id] = stmt.value
    return aliases


def _validate_named_parameter_override(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    override: Any,
    aliases: dict[str, ast.expr] | None = None,
) -> str | None:
    """Cross-check one domain adapter override against the real AST signature.

    Returns ``None`` when the override is honest (every declared parameter
    really exists, every uncovered parameter has a default, every declared
    codec is compatible with that parameter's real annotation); otherwise a
    human-readable reason the generator must fail closed for this
    capability rather than emit a bundle that would only fail later, at
    verify/bind time.
    """
    positional = [*node.args.posonlyargs, *node.args.args]
    kwonly = node.args.kwonlyargs
    by_name = {parameter.arg: parameter for parameter in (*positional, *kwonly)}
    declared_names = {parameter.name for parameter in override.parameters}
    missing = declared_names - set(by_name)
    if missing:
        return f"declared parameters not found in the implementation: {sorted(missing)}"

    defaults_start = len(positional) - len(node.args.defaults)
    uncovered_required: list[str] = []
    for index, parameter in enumerate(positional):
        if parameter.arg in declared_names:
            continue
        if index < defaults_start:
            uncovered_required.append(parameter.arg)
    for index, parameter in enumerate(kwonly):
        if parameter.arg in declared_names:
            continue
        if node.args.kw_defaults[index] is None:
            uncovered_required.append(parameter.arg)
    if uncovered_required:
        return (
            "implementation has required parameters the override does not cover: "
            f"{sorted(uncovered_required)}"
        )

    for parameter in override.parameters:
        annotation = by_name[parameter.name].annotation
        if parameter.codec is ParameterCodec.PATH and not _looks_like_path_annotation(
            annotation, aliases
        ):
            return (
                f"parameter {parameter.name!r} uses codec 'path' but its annotation is not str/Path"
            )
        if parameter.codec is ParameterCodec.LEGACY_RESULT:
            # Ruling 2 (Decision 004 approval 2026-08-14): a core-native
            # capability may take the L1 result itself as an input. The
            # implementation annotation must actually name OrganelleResult
            # (bare or in a union) - the codec never binds a parameter the
            # implementation does not declare as accepting a result.
            ann = annotation
            names = (
                {ann.id}
                if isinstance(ann, ast.Name)
                else {
                    leaf.id
                    for part in ([ann.left, ann.right] if isinstance(ann, ast.BinOp) else [])
                    for leaf in ([part] if isinstance(part, ast.Name) else [])
                }
            )
            if "OrganelleResult" not in names:
                return (
                    f"parameter {parameter.name!r} uses codec 'legacy_result' but its "
                    "annotation does not name OrganelleResult"
                )
        if (
            parameter.codec is ParameterCodec.PATH
            and parameter.path_role == "output"
            and SideEffect.WRITE_FILES not in override.side_effects
        ):
            return (
                f"parameter {parameter.name!r} declares an output-file destination but "
                "the override does not declare the write_files side effect"
            )
        if parameter.codec is ParameterCodec.DIRECTORY:
            if not _looks_like_directory_annotation(annotation):
                return (
                    f"parameter {parameter.name!r} uses codec 'directory' but its annotation "
                    "is not str/Path"
                )
            if parameter.path_role not in ("input", "output"):
                return (
                    f"parameter {parameter.name!r} uses codec 'directory' but declares no "
                    "path_role ('input' for a pre-existing directory tree, 'output' for a "
                    "write destination)"
                )
        if parameter.codec is ParameterCodec.JSON and not _looks_json_safe(annotation):
            return f"parameter {parameter.name!r} uses codec 'json' but its annotation is not JSON-safe"
    return None


# --- planning --------------------------------------------------------------


def _plan_bundle(
    record: dict[str, Any],
    current_root: Path,
    overrides: dict[str, Any],
) -> PlannedBundle | SkippedCapability:
    capability_id = record["id"]
    source_path = current_root / record["source_relpath"]
    tree = _parse_module(source_path)
    node = _find_function(tree, record["public_name"])
    if node is None:
        return SkippedCapability(
            capability_id,
            "capability.source_function_missing",
            f"{record['public_name']} is not a top-level function in {source_path}",
        )

    description = _description(node)
    if description is None:
        return SkippedCapability(
            capability_id,
            "capability.description_undecidable",
            "docstring is missing or shorter than "
            f"{_MIN_DESCRIPTION_LENGTH} characters after normalization; a fabricated "
            "description is not an honest substitute",
        )

    try:
        adapter_module = importlib.import_module(
            f"organelleverse.capabilities.adapters.{record['domain']}"
        )
    except ModuleNotFoundError:
        adapter_module = None
    excluded = getattr(adapter_module, "EXCLUDE", frozenset())
    if capability_id in excluded:
        return SkippedCapability(
            capability_id,
            "capability.excluded_by_domain_adapter",
            "the domain adapter explicitly excludes this capability "
            "(see its module docstring for the precise reason)",
        )
    import_map = _core_type_import_map(tree)
    canonical = _detect_canonical_core(node, import_map)
    if canonical is not None:
        input_kind, output_kind, stage, input_sequence = canonical
        inferred_side_effects = (
            (SideEffect.READ_FILES,) if input_kind in {CoreKind.GENOME, CoreKind.DATA} else ()
        )
        canonical_side_effects = getattr(adapter_module, "CANONICAL_SIDE_EFFECTS", {})
        if not isinstance(canonical_side_effects, dict):
            raise BundleGenerationError(
                f"adapters.{record['domain']} declares a non-dict CANONICAL_SIDE_EFFECTS table"
            )
        side_effects = canonical_side_effects.get(capability_id, inferred_side_effects)
        if not isinstance(side_effects, tuple) or not all(
            isinstance(effect, SideEffect) for effect in side_effects
        ):
            raise BundleGenerationError(
                f"adapters.{record['domain']} declares invalid canonical side effects "
                f"for {capability_id}"
            )
        return PlannedBundle(
            capability_id=capability_id,
            title=_title(record["public_name"]),
            description=description,
            keywords=_keywords(
                record["domain"],
                record["public_name"],
                stage=stage,
                input_kind=input_kind,
                output_kind=output_kind,
            ),
            stage=stage,
            input_kind=input_kind,
            output_kind=output_kind,
            input_sequence=input_sequence,
            callable_locator=record["python_locator"],
            argument_mode="canonical_core",
            parameters=(),
            result_codec="canonical",
            result_key=None,
            side_effects=side_effects,
            dependencies=tuple(record["dependencies"]),
        )

    override = overrides.get(capability_id)
    if override is None:
        return SkippedCapability(
            capability_id,
            "capability.no_adapter_override",
            "signature does not match the generic canonical_core shape, and no domain "
            "adapter override is registered for it; the generator refuses to invent a "
            "parameter or result codec",
        )
    validation_error = _validate_named_parameter_override(
        node, override, _module_type_aliases(tree)
    )
    if validation_error is not None:
        return SkippedCapability(
            capability_id, "capability.adapter_override_invalid", validation_error
        )

    stage = OperationStage.ANALYZE
    input_kind = CoreKind.NONE
    output_kind = CoreKind.RESULT
    return PlannedBundle(
        capability_id=capability_id,
        title=_title(record["public_name"]),
        description=description,
        keywords=_keywords(
            record["domain"],
            record["public_name"],
            stage=stage,
            input_kind=input_kind,
            output_kind=output_kind,
            extra=tuple(getattr(override, "extra_keywords", ())),
        ),
        stage=stage,
        input_kind=input_kind,
        output_kind=output_kind,
        callable_locator=record["python_locator"],
        argument_mode="named_parameters",
        parameters=tuple(
            PlannedParameter(
                name=parameter.name, codec=parameter.codec, path_role=parameter.path_role
            )
            for parameter in override.parameters
        ),
        result_codec=override.result_codec.value,
        result_key=override.result_key,
        side_effects=tuple(override.side_effects),
        dependencies=tuple(record["dependencies"]),
        deterministic=getattr(override, "deterministic", True),
        deterministic_reason=getattr(override, "deterministic_reason", ""),
        extra_keywords=tuple(getattr(override, "extra_keywords", ())),
    )


def _slug(capability_id: str) -> str:
    return capability_id.replace(".", "-").replace("_", "-")


# --- rendering ---------------------------------------------------------

_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    """Render one TOML table key: bare when safe, JSON-quoted otherwise.

    A JSON-quoted TOML key uses the identical basic-string syntax
    ``json.dumps`` already produces - the same equivalence this module's
    string *values* have always relied on (every other ``json.dumps(...)``
    call in ``render_capability_toml`` below).
    """
    if _TOML_BARE_KEY_RE.fullmatch(key):
        return key
    return json.dumps(key)


def _toml_inline(value: object) -> str:
    """Render one JSON-safe Python value as a TOML inline literal.

    Handles exactly the shapes a fixture's ``input``/``parameters`` can
    carry (``FixtureSpec`` requires finite JSON: str/int/float/bool/list/
    dict, recursively - see ``models.py``'s ``_assert_finite_json``). TOML
    has no ``null``, so a JSON ``None`` is refused here rather than silently
    dropped or mis-rendered; no fixture in this generator ever needs one.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{_toml_key(str(key))} = {_toml_inline(item)}" for key, item in value.items()
        )
        return "{ " + pairs + " }"
    raise BundleGenerationError(f"fixture value is not TOML-inline-safe: {value!r}")


def render_capability_toml(plan: PlannedBundle) -> str:
    keywords_toml = ", ".join(json.dumps(keyword) for keyword in plan.keywords)
    lines = [
        'schema = "organelleverse.capability.v1"',
        "",
        "[capability]",
        f"id = {json.dumps(plan.capability_id)}",
        'bundle_version = "1.0.0"',
        'implementation = "native"',
        "",
        "[contract]",
        'contract_version = "1.0"',
        f"title = {json.dumps(plan.title)}",
        f"description = {json.dumps(plan.description)}",
        f"keywords = [{keywords_toml}]",
        'execution_mode = "inline"',
        f"stage = {json.dumps(plan.stage.value)}",
        f"input_kind = {json.dumps(plan.input_kind.value)}",
        *(["input_sequence = true"] if getattr(plan, "input_sequence", False) else []),
        f"output_kind = {json.dumps(plan.output_kind.value)}",
        f"callable_locator = {json.dumps(plan.callable_locator)}",
    ]
    if plan.side_effects:
        rendered = ", ".join(json.dumps(effect.value) for effect in plan.side_effects)
        lines.append(f"side_effects = [{rendered}]")
    lines.extend(
        [
            f"deterministic = {str(plan.deterministic).lower()}",
        ]
    )
    if plan.deterministic_reason:
        lines.append(f"deterministic_reason = {json.dumps(plan.deterministic_reason)}")
    lines.extend(
        [
            "idempotent = true",
            "cacheable = false",
        ]
    )
    for dependency in plan.dependencies:
        lines.extend(
            [
                "",
                "[[contract.dependencies]]",
                'kind = "python"',
                f"name = {json.dumps(dependency)}",
            ]
        )
    lines.extend(
        [
            "",
            "[contract.binding]",
            f"argument_mode = {json.dumps(plan.argument_mode)}",
            f"result_codec = {json.dumps(plan.result_codec)}",
        ]
    )
    if plan.result_key is not None:
        lines.append(f"result_key = {json.dumps(plan.result_key)}")
    for parameter in plan.parameters:
        lines.extend(
            [
                "",
                "[[contract.binding.parameters]]",
                f"name = {json.dumps(parameter.name)}",
                f"codec = {json.dumps(parameter.codec.value)}",
            ]
        )
        if parameter.path_role is not None:
            lines.append(f"path_role = {json.dumps(parameter.path_role)}")
    for fixture in plan.fixtures:
        lines.extend(
            [
                "",
                "[[fixture]]",
                f"case = {json.dumps(fixture.case)}",
                f"input = {_toml_inline({'kind': 'none'})}",
            ]
        )
        if fixture.parameters:
            lines.append(f"parameters = {_toml_inline(fixture.parameters)}")
        lines.append(f"expect = {json.dumps(fixture.expect)}")
        lines.append(f"equivalence = {json.dumps(fixture.equivalence)}")
        if fixture.tolerance is not None:
            lines.append(f"tolerance = {fixture.tolerance}")
    return "\n".join(lines) + "\n"


# --- fixture capture: the ONE deliberate, non-AST step --------------------
#
# Everything above this point (ledger loading, AST introspection, planning,
# rendering) derives every fact from ``ast.parse`` and small hand-authored
# adapter *data* modules - never an import of a capability's own
# implementation, per this file's module docstring. ``_capture_fixtures`` is
# the one exception, and it is scoped as tightly as the rest of this file is
# strict: it only ever runs for a capability a domain adapter's ``FIXTURES``
# table names explicitly, it imports exactly that one capability's own
# module (never a whole domain), and it calls the real implementation
# through the exact same ``bind_python_capability`` dispatch
# ``LocalVerificationEnvironment.evaluate_fixture``'s own ``_invoke_fixture``
# uses to *re*-run a fixture later - so what gets frozen here as ``expect``
# is the same kind of real, live output re-verification will recompute and
# compare against, never a value reasoned about from the source text.


def _capture_fixtures(
    *,
    bundle_dir: Path,
    bundle_toml_path: Path,
    fixture_cases: tuple[FixtureCase, ...],
) -> tuple[RenderedFixture, ...]:
    """Actually run one capability's real implementation, once per case.

    Parses *bundle_toml_path* (the base bundle, already written to disk
    without fixtures) for real, binds it to its real callable via
    ``bind_python_capability`` - the exact real binding contract this
    bundle ships, not a hand-approximated stand-in - and invokes it once per
    *fixture_cases* entry with real, on-disk input files. The produced
    ``OrganelleResult`` is frozen verbatim as that case's ``expect`` file.
    """
    from organelleverse.operations.python_binding import bind_python_capability

    bundle = parse_capability_bundle(bundle_toml_path)
    locator = bundle.contract.callable_locator
    if locator is None:
        raise BundleGenerationError(
            f"fixture capture: bundle has no callable_locator: {bundle_toml_path}"
        )
    module_name, attribute_name = locator.split(":", 1)
    module = importlib.import_module(module_name)
    implementation = getattr(module, attribute_name, None)
    if not callable(implementation):
        raise BundleGenerationError(
            f"fixture capture: callable locator does not resolve: {locator}"
        )
    bound = bind_python_capability(bundle, implementation, frozen_schema=None)

    parameter_codec_by_name = {
        parameter.name: parameter.codec for parameter in bundle.contract.binding.parameters
    }

    rendered: list[RenderedFixture] = []
    for case in fixture_cases:
        case_dir = bundle_dir / "fixtures" / case.case
        input_dir = case_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        file_paths: dict[str, Path] = {}
        for one_file in case.files:
            file_path = input_dir / one_file.relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(one_file.content, encoding="utf-8", newline="\n")
            file_paths[one_file.relative_path] = file_path

        invoke_kwargs: dict[str, object] = {}
        toml_parameters: dict[str, object] = {}
        for name, value in case.parameters.items():
            codec = parameter_codec_by_name.get(name)
            if codec is None:
                raise BundleGenerationError(
                    f"fixture case {case.case!r} sets undeclared parameter {name!r} "
                    f"for {bundle.capability.id}"
                )
            if codec is ParameterCodec.PATH:
                scalar_path = isinstance(value, str)
                if scalar_path:
                    value = [value]
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item in file_paths for item in value)
                ):
                    raise BundleGenerationError(
                        f"fixture case {case.case!r} parameter {name!r} must name one of "
                        f"its own declared input files for {bundle.capability.id}"
                    )
                real_paths = [file_paths[item] for item in value]
                if scalar_path:
                    invoke_kwargs[name] = str(real_paths[0])
                    toml_parameters[name] = real_paths[0].relative_to(bundle_dir).as_posix()
                else:
                    invoke_kwargs[name] = [str(path) for path in real_paths]
                    toml_parameters[name] = [
                        path.relative_to(bundle_dir).as_posix() for path in real_paths
                    ]
            else:
                invoke_kwargs[name] = value
                toml_parameters[name] = value

        # THE deliberate, real execution: the actual capability, actually run.
        result = bound.invoke(None, invoke_kwargs)
        produced = result.model_dump(mode="json")
        # Do not freeze this checkout's own run/machine identity as if it
        # were part of the expected science - object_id, and provenance's
        # parameters_hash/package_version/git_commit, are stripped the same
        # way LocalVerificationEnvironment.evaluate_fixture strips them
        # before comparing (see verification.py's strip_run_provenance and
        # its own docstring for the full reasoning and field list). A
        # PATH-codec parameter's real, resolved absolute path - this
        # checkout's own filesystem location - is exactly what makes
        # parameters_hash/object_id non-portable; freezing them here would
        # commit a fixture that only ever verifies on this machine.
        portable = cast(dict[str, object], strip_run_provenance(produced))

        expected_path = case_dir / "expected.json"
        expected_path.write_text(
            json.dumps(portable, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        rendered.append(
            RenderedFixture(
                case=case.case,
                parameters=toml_parameters,
                expect=expected_path.relative_to(bundle_dir).as_posix(),
                equivalence=case.equivalence,
                tolerance=case.tolerance,
            )
        )
    return tuple(rendered)


# --- orchestration -------------------------------------------------------


def _load_overrides(domain: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(f"organelleverse.capabilities.adapters.{domain}")
    except ModuleNotFoundError:
        return {}
    overrides = getattr(module, "OVERRIDES", None)
    if not isinstance(overrides, dict):
        raise BundleGenerationError(f"adapters.{domain} does not declare a dict OVERRIDES table")
    return overrides


def _load_fixture_cases(domain: str) -> dict[str, tuple[FixtureCase, ...]]:
    try:
        module = importlib.import_module(f"organelleverse.capabilities.adapters.{domain}")
    except ModuleNotFoundError:
        return {}
    fixtures = getattr(module, "FIXTURES", None)
    if fixtures is None:
        return {}
    if not isinstance(fixtures, dict):
        raise BundleGenerationError(f"adapters.{domain} declares a non-dict FIXTURES table")
    return fixtures


def build_domain(
    *,
    domain: str,
    ledger_path: Path,
    current_root: Path,
    output_root: Path,
) -> BuildReport:
    records = _load_domain_records(ledger_path, domain)
    if not records:
        raise BundleGenerationError(f"no ledger records for domain {domain!r}")
    overrides = _load_overrides(domain)
    fixture_cases_by_id = _load_fixture_cases(domain)

    generated: list[GeneratedBundle] = []
    skipped: list[SkippedCapability] = []
    for record in records:
        planned = _plan_bundle(record, current_root, overrides)
        if isinstance(planned, SkippedCapability):
            skipped.append(planned)
            continue
        toml_text = render_capability_toml(planned)
        bundle_dir = output_root / _slug(planned.capability_id)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        toml_path = bundle_dir / "capability.toml"
        toml_path.write_text(toml_text, encoding="utf-8", newline="\n")
        # Dogfood the real, side-effect-free parser immediately, exactly as
        # scaffold.py does: catch a malformed bundle here, not on the first
        # caller's discover_capabilities().
        parse_capability_bundle(toml_path)

        fixture_cases = fixture_cases_by_id.get(planned.capability_id, ())
        if fixture_cases:
            # The one deliberate, non-AST step - see _capture_fixtures's own
            # docstring. Runs only because this capability's domain adapter
            # named it explicitly in FIXTURES.
            rendered_fixtures = _capture_fixtures(
                bundle_dir=bundle_dir,
                bundle_toml_path=toml_path,
                fixture_cases=fixture_cases,
            )
            planned = replace(planned, fixtures=rendered_fixtures)
            toml_text = render_capability_toml(planned)
            toml_path.write_text(toml_text, encoding="utf-8", newline="\n")
            # Re-dogfood the final bundle, fixtures included.
            parse_capability_bundle(toml_path)

        generated.append(GeneratedBundle(planned.capability_id, toml_path))

    return BuildReport(domain=domain, generated=tuple(generated), skipped=tuple(skipped))


def _default_ledger() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "docs" / "operations" / "restored-capabilities.toml"
    )


def _default_current_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "organelleverse"


def _default_output_root() -> Path:
    return _default_current_root() / "capabilities"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, help="restoration ledger domain to generate")
    parser.add_argument("--ledger", type=Path, default=_default_ledger())
    parser.add_argument("--current-root", type=Path, default=_default_current_root())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    args = parser.parse_args(argv)

    try:
        report = build_domain(
            domain=args.domain,
            ledger_path=args.ledger,
            current_root=args.current_root,
            output_root=args.output_root,
        )
    except BundleGenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for skip in sorted(report.skipped, key=lambda item: item.capability_id):
        print(f"SKIP {skip.capability_id}: {skip.code}: {skip.message}", file=sys.stderr)
    for item in sorted(report.generated, key=lambda item: item.capability_id):
        print(f"OK   {item.capability_id}: {item.path}")
    print(f"domain: {report.domain}")
    print(f"generated: {len(report.generated)}")
    print(f"skipped: {len(report.skipped)}")
    return 0 if report.generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
