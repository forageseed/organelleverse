"""The one authoritative reachability computation.

Reachability is a fixpoint seeded only by operations that need no core input.
A set-membership test over declared outputs is NOT reachability: two mutually
feeding transforms, or a single genome -> genome transform, would appear to
produce their own inputs.

This computation proves TYPE reachability over the released catalog only. It
does not prove that a reachable chain produces a scientifically useful result,
and operations registered into a local OperationRegistry are deliberately
invisible to it.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import organelleverse.operations as op
from organelleverse.operations import CoreKind, OperationSpec
from organelleverse.operations.registry import registry

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "organelleverse"
_BASELINE_DIR = Path(__file__).resolve().parent / "reachability_baseline"
_INTENT_KEYS = frozenset({"tracked_by", "first_recorded", "derived"})


@dataclass(frozen=True)
class UnreachableOperation:
    operation_id: str
    hole_class: str  # "A" (input kind) or "B" (input modalities)
    missing: tuple[str, ...]


@dataclass(frozen=True)
class ReachabilityReport:
    invocable: tuple[str, ...] = ()
    unreachable: tuple[UnreachableOperation, ...] = ()
    producible_kinds: tuple[str, ...] = ()
    producible_modalities: tuple[str, ...] = ()
    unsatisfiable_declarations: tuple[tuple[str, str], ...] = ()
    declared_only_modalities: tuple[str, ...] = ()
    reachable_unvalidated: tuple[str, ...] = field(default=())


def compute_reachability(
    specs: Iterable[OperationSpec],
    *,
    contract_modalities: frozenset[str],
) -> ReachabilityReport:
    catalog = tuple(specs)
    producible_kinds: set[str] = set()
    producible_modalities: set[str] = set()
    invocable: set[str] = set()

    changed = True
    while changed:
        changed = False
        for spec in catalog:
            if spec.operation_id in invocable:
                continue
            if not _can_invoke(spec, producible_kinds, producible_modalities):
                continue
            invocable.add(spec.operation_id)
            changed = True
            if spec.output_kind is not CoreKind.NONE:
                producible_kinds.add(spec.output_kind.value)
            producible_modalities.update(spec.output_modalities)

    unreachable: list[UnreachableOperation] = []
    unsatisfiable: list[tuple[str, str]] = []
    for spec in catalog:
        unmet = tuple(
            modality for modality in spec.input_modalities if modality not in producible_modalities
        )
        if spec.operation_id in invocable:
            unsatisfiable.extend((spec.operation_id, modality) for modality in unmet)
            continue
        if spec.input_kind.value not in producible_kinds and spec.input_kind is not CoreKind.NONE:
            unreachable.append(
                UnreachableOperation(spec.operation_id, "A", (spec.input_kind.value,))
            )
        else:
            unreachable.append(UnreachableOperation(spec.operation_id, "B", unmet))

    declared_only = tuple(sorted(producible_modalities - contract_modalities))
    reachable_unvalidated = tuple(
        sorted(
            spec.operation_id
            for spec in catalog
            if spec.operation_id in invocable
            and spec.input_modalities
            and all(modality in declared_only for modality in spec.input_modalities)
        )
    )

    return ReachabilityReport(
        invocable=tuple(sorted(invocable)),
        unreachable=tuple(sorted(unreachable, key=lambda entry: entry.operation_id)),
        producible_kinds=tuple(sorted(producible_kinds)),
        producible_modalities=tuple(sorted(producible_modalities)),
        unsatisfiable_declarations=tuple(sorted(unsatisfiable)),
        declared_only_modalities=declared_only,
        reachable_unvalidated=reachable_unvalidated,
    )


def _can_invoke(
    spec: OperationSpec,
    producible_kinds: set[str],
    producible_modalities: set[str],
) -> bool:
    if spec.input_kind is CoreKind.NONE:
        return True
    if spec.input_kind.value not in producible_kinds:
        return False
    if spec.input_kind is CoreKind.DATA and spec.input_modalities:
        return any(modality in producible_modalities for modality in spec.input_modalities)
    return True


@dataclass(frozen=True)
class SuggestionFinding:
    source: str
    lineno: int
    operation_id: str | None
    non_literal: bool


def _relative_source(path: Path, root: Path) -> str:
    """Render a scanned path relative to the scan root.

    ``SuggestionFinding.source`` must never carry an absolute path: it is
    rendered into ``docs/operations/reachability.md``, which is byte-compared in
    CI, so an absolute prefix would make the report machine-specific and break
    both determinism and portability. Class C is empty today, so nothing
    currently exercises this - which is exactly why it is enforced here rather
    than left to the caller.
    """
    base = root.parent if root.is_file() else root
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def scan_suggestions(root: Path) -> tuple[SuggestionFinding, ...]:
    """Find every OperationSuggestion construction under *root*.

    The static scan is the STRONGER mechanism for this hole class: the runtime
    check resolves against the invoking registry, so it is defeatable by a local
    or mock registration, while this scan cannot be masked. Its limitation is
    that it sees only constructions in the scanned tree.

    operation_id must be a string literal or a module-level constant, so the
    scan cannot silently lose coverage when someone writes
    ``operation_id=f"io.{name}"``. This enforcement is required because
    OperationSuggestion.operation_id is a bare str with no pattern
    (core/result.py:72), so the model gives no help and an f-string would
    silently drop coverage.
    """
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    findings: list[SuggestionFinding] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name) and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "OperationSuggestion":
                continue
            for keyword in node.keywords:
                if keyword.arg != "operation_id":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    resolved, non_literal = value.value, False
                elif isinstance(value, ast.Name) and value.id in constants:
                    resolved, non_literal = constants[value.id], False
                else:
                    resolved, non_literal = None, True
                findings.append(
                    SuggestionFinding(
                        _relative_source(path, root), node.lineno, resolved, non_literal
                    )
                )
    return tuple(findings)


def class_c_holes(
    specs: Iterable[OperationSpec], findings: Iterable[SuggestionFinding]
) -> tuple[tuple[str, str, str], ...]:
    """Return sorted (source, operation_id, reason) tuples for class C holes.

    ``reason`` is ``"non_literal_operation_id"`` when the AST could not resolve
    the operation_id to a literal or module-level constant, or
    ``"not_in_catalog"`` when the resolved id names no released operation.

    Two of the three class C rules are NOT checkable from the AST: kind
    compatibility and ``parameter_changes`` satisfiability both depend on the
    emitting spec, which is unknown here. Task 4's runtime check covers those
    two rules; this scan covers only operation_id resolvability against the
    catalog.
    """
    by_id = {spec.operation_id: spec for spec in specs}
    holes: list[tuple[str, str, str]] = []
    for finding in findings:
        if finding.non_literal:
            holes.append((finding.source, "", "non_literal_operation_id"))
            continue
        assert finding.operation_id is not None
        target = by_id.get(finding.operation_id)
        if target is None:
            holes.append((finding.source, finding.operation_id, "not_in_catalog"))
    return tuple(sorted(holes))


def contract_modalities() -> frozenset[str]:
    """Every DataContract modality resolvable in the released registry.

    Derived from the registrations, not hardcoded: every resolvable contract
    appears on some ``BoundOperation`` (input side on ``data_contracts``, output
    side on ``output_data_contracts``), so the union over all bindings is exactly
    the resolvable set and it stays correct when someone adds a contract. Today
    this is ``{organelle_records, nuclear_assemblies, sequencing_reads,
    pmat_graph_input}``; class D falls out as ``declared_outputs - resolvable``.
    """
    found: set[str] = set()
    for spec in op.list():
        binding = registry.require(spec.operation_id)
        for contract in (*binding.data_contracts, *binding.output_data_contracts):
            found.add(contract.modality)
    return frozenset(found)


def released_inventory() -> dict[str, list[dict[str, object]]]:
    """Group the released catalog's holes by the MISSING thing, not the blocked op.

    One hole therefore carries one ``tracked_by`` owner instead of repeating it
    per blocked operation. Class A/B rows carry ``missing`` and ``blocks``;
    class C rows carry ``source``/``operation_id``/``reason``; class D rows carry
    ``modality``. This is the COMPUTED shape - it carries no intent fields, which
    is what stops regeneration from inventing ``tracked_by`` (see
    ``merge_baseline_intent`` and ``assert_baseline_writable``).
    """
    specs = op.list()
    resolvable = contract_modalities()
    report = compute_reachability(specs, contract_modalities=resolvable)

    grouped: dict[str, dict[str, list[str]]] = {"A": {}, "B": {}}
    for entry in report.unreachable:
        for missing in entry.missing:
            grouped[entry.hole_class].setdefault(missing, []).append(entry.operation_id)

    def rows(hole_class: str) -> list[dict[str, object]]:
        return [
            {"missing": missing, "blocks": sorted(blocked)}
            for missing, blocked in sorted(grouped[hole_class].items())
        ]

    declared = {modality for spec in specs for modality in spec.output_modalities}
    return {
        "A": rows("A"),
        "B": rows("B"),
        "C": [
            {"source": source, "operation_id": operation_id, "reason": reason}
            for source, operation_id, reason in class_c_holes(specs, scan_suggestions(_SRC_ROOT))
        ],
        "D": [{"modality": modality} for modality in sorted(declared - resolvable)],
    }


def _load_class(hole_class: str) -> list[dict[str, object]]:
    path = _BASELINE_DIR / f"{hole_class.lower()}.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return cast(list[dict[str, object]], data)


def load_baseline() -> dict[str, list[dict[str, object]]]:
    """Return the committed baseline in the computation's shape.

    Intent fields (``tracked_by``, ``first_recorded``, ``derived``) are stripped
    so the ratchet compares the computed inventory field-for-field against the
    baseline in both directions. Use ``load_baseline_entries`` to inspect the
    intent fields themselves.
    """
    return {
        hole_class: [
            {k: v for k, v in entry.items() if k not in _INTENT_KEYS}
            for entry in _load_class(hole_class)
        ]
        for hole_class in ("A", "B", "C", "D")
    }


def load_baseline_entries() -> list[dict[str, object]]:
    """Every baseline entry across all four classes, intent fields included.

    Each entry is tagged with its ``hole_class`` so a reviewer (and the
    regeneration refusal) can name it. This returns ALL entries, including the
    derived one - the derived exemption is explicit in the checks, never a hole.
    """
    flattened: list[dict[str, object]] = []
    for hole_class in ("A", "B", "C", "D"):
        for entry in _load_class(hole_class):
            tagged = dict(entry)
            tagged["hole_class"] = hole_class
            flattened.append(tagged)
    return flattened


#: Prepended verbatim to every rendered report. Design §6 requires both
#: boundaries and both suggestion-mechanism limitations to travel WITH the
#: inventory: without them a reader can over-read the report as proving that a
#: reachable chain produces a useful result, which is exactly the overclaim the
#: design guards against.
_REPORT_HEADER: tuple[str, ...] = (
    "# Operation reachability inventory",
    "",
    "Generated by `scripts/reachability_audit.py --write`. Do not edit by hand.",
    "",
    "## What this report does and does not prove",
    "",
    "- **Scope is the released catalog only.** Operations registered into a local",
    "  `OperationRegistry` are invisible here, and correctly so: a locally registered",
    "  producer does **not** make a released consumer reachable, so registering a",
    "  producer in a test can never be mistaken for closing a hole.",
    "- **It proves type reachability, not scientific usefulness.** An operation may",
    "  return its declared modality with an empty or degenerate payload and satisfy",
    "  every invariant. Whether a reachable chain produces a meaningful result is",
    "  outside this audit's remit and must not be claimed on its evidence.",
    "",
    "## Limits of the two suggestion-integrity mechanisms",
    "",
    "- The **static AST scan** is the stronger one for class C, because it cannot be",
    "  masked. Its limitation is that it sees only constructions in the scanned tree.",
    "- The **runtime check** fires only when the code path actually executes - the",
    "  assembly success path needs a real backend, so CI does not exercise it - and it",
    "  resolves against the invoking registry, so a local or mock registration masks",
    "  the defect.",
    "",
)


def render_report(inventory: dict[str, list[dict[str, object]]]) -> str:
    """Render the inventory as a deterministic markdown report.

    ``json.dumps(..., sort_keys=True, ...)`` gives canonical key ordering,
    ``"\\n".join`` gives LF endings, and no timestamp or absolute path is ever
    emitted - confirmed by the byte-compare test in CI. The report is rendered
    from the COMPUTED inventory, so it carries no intent fields; those live only
    in the baseline.
    """
    lines = [*_REPORT_HEADER]
    for hole_class in ("A", "B", "C", "D"):
        rows = inventory[hole_class]
        lines.append(f"## Class {hole_class} ({len(rows)})")
        lines.append("")
        for row in rows:
            lines.append(
                "- " + json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            )
        lines.append("")
    return "\n".join(lines)


def _entry_identity(hole_class: str, entry: dict[str, object]) -> object:
    """The stable key that ties a computed entry to its baseline entry."""
    if hole_class in ("A", "B"):
        return entry["missing"]
    if hole_class == "D":
        return entry["modality"]
    return (entry["source"], entry["operation_id"], entry["reason"])


def merge_baseline_intent(
    computed: dict[str, list[dict[str, object]]],
    baseline_entries: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Copy intent fields from the baseline onto the computed inventory.

    Matching is by entry identity (``missing`` for A/B, ``modality`` for D, the
    ``(source, operation_id, reason)`` triple for C). A computed entry with no
    baseline match carries no intent - that is the seam that stops regeneration
    from inventing ``tracked_by``: a brand-new hole has no baseline entry to
    inherit intent from, so ``assert_baseline_writable`` refuses it until a human
    names an owner.
    """
    intent: dict[str, dict[object, dict[str, object]]] = {
        hole_class: {} for hole_class in ("A", "B", "C", "D")
    }
    for entry in baseline_entries:
        hole_class = entry.get("hole_class")
        if not isinstance(hole_class, str):
            continue
        identity = _entry_identity(hole_class, entry)
        intent[hole_class][identity] = {key: entry[key] for key in _INTENT_KEYS if key in entry}
    return {
        hole_class: [
            {**entry, **intent[hole_class].get(_entry_identity(hole_class, entry), {})}
            for entry in computed[hole_class]
        ]
        for hole_class in ("A", "B", "C", "D")
    }


class UntrackedHoleError(ValueError):
    """A computed hole lacks ``tracked_by`` and is not derived."""


def _describe_entry(hole_class: str, entry: dict[str, object]) -> str:
    if hole_class in ("A", "B"):
        return str(entry["missing"])
    if hole_class == "D":
        return str(entry["modality"])
    return str(entry["operation_id"])


def assert_baseline_writable(merged: dict[str, list[dict[str, object]]]) -> None:
    """Refuse to write any non-derived entry that lacks ``tracked_by``.

    The derived result hole (class A, ``missing='result'``) is the ONLY entry
    exempt from ``tracked_by``, and the exemption is explicit: it must carry
    ``'derived': True``. Anything else without ``tracked_by`` is a brand-new hole
    the computation discovered and the regeneration tool cannot own - a human
    must name the owning follow-up slice before it can enter the baseline. This
    is the whole mechanism that stops reflexive regeneration, so the exemption is
    a checked condition here rather than a gap in the check.
    """
    offenders: list[str] = []
    for hole_class in ("A", "B", "C", "D"):
        for entry in merged[hole_class]:
            if entry.get("derived"):
                continue
            if not entry.get("tracked_by"):
                offenders.append(f"class {hole_class} {_describe_entry(hole_class, entry)!r}")
    if offenders:
        raise UntrackedHoleError(
            "refusing to write reachability baseline entries without an owner; "
            "add tracked_by (or mark derived) for: " + ", ".join(offenders)
        )
