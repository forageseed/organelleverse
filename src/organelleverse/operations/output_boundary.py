"""Public output-boundary ratchets for the unified compute-then-write contract.

OrganelleVerse has one output rule: a scientific operation computes and returns
a typed value; it never accepts a user-selected final write target. Only an
explicit writer/materializer accepts a destination. This module ships two
scanners that make that rule mechanically enforceable:

* :func:`public_surface_final_write_violations` walks the canonical ``ov.*``
  facades and reports forbidden final-write parameters on scientific callables.
  Writers and reader/fetch/install/persistence primitives are classified
  explicitly so a destination on ``ov.write`` or ``ov.fetch.merge_with_baseline``
  does not register as a scientific violation.
* :func:`released_scientific_specs_with_final_write` and
  :func:`released_writer_specs_missing_destination` derive the same facts from
  each released ``OperationSpec``'s registered JSON Schema (what an Agent sees).

The registry scanners classify a writer from its consume/write contract (a
CONSUME Result->Result stage whose operation id declares writer intent), not
from a parameter spelling or a bare name suffix.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast

from .registry import OperationRegistry
from .spec import CoreKind, OperationSpec, OperationStage

# The exact spelling list from the unified-output-boundary design. A public
# scientific callable may not declare any of these names.
FORBIDDEN_FINAL_WRITE_PARAMS: frozenset[str] = frozenset(
    {
        "output",
        "output_dir",
        "out_dir",
        "destination",
        "workspace",
        "work_dir",
        "output_path",
        "output_prefix",
        "save_path",
        "export_path",
    }
)

# Writer/materializer callables: explicitly named ``write``/``save``/
# ``export``/``materialize``. These are the only public callables that may
# accept a user destination, so they are excluded from the scientific report.
WRITER_CALLABLE_PREFIXES: tuple[str, ...] = (
    "write",
    "save",
    "export",
    "materialize",
)

# Facades whose callables are reader/fetch/install/persistence primitives. Their
# destination-shaped parameters are declared side effects on managed snapshots or
# caller-supplied input paths, not scientific outputs.
PRIMITIVE_FACADES: frozenset[str] = frozenset({"io", "fetch", "environments"})

# OperationSpec writer classification. The consume/write contract is a CONSUME
# Result->Result stage. The operation id must also declare writer intent; either
# signal alone is insufficient. ``qc.assembly`` has the stage contract but no
# writer intent (and exposes ``workspace``, not a destination), so it stays
# scientific; ``annotation.write``/``qc.write`` declare intent and expose
# ``output``, so they are writers.
WRITER_OPERATION_SUFFIXES: tuple[str, ...] = (".write", ".save", ".export", ".materialize")
WRITER_DESTINATION_PARAMS: frozenset[str] = frozenset(
    {
        "output",
        "output_dir",
        "out_dir",
        "destination",
        "output_path",
        "output_prefix",
        "save_path",
        "export_path",
    }
)


def _public_callables(module: ModuleType) -> list[tuple[str, Callable[..., object]]]:
    """Names and callables a facade exports, via ``__all__`` when it is declared."""
    declared = getattr(module, "__all__", None)
    names: tuple[str, ...]
    if declared is None:
        names = tuple(name for name in dir(module) if not name.startswith("_"))
    else:
        names = tuple(declared)
    callables: list[tuple[str, Callable[..., object]]] = []
    for name in names:
        attribute = getattr(module, name)
        if (
            inspect.isfunction(attribute)
            or inspect.ismethod(attribute)
            or inspect.isbuiltin(attribute)
        ):
            callables.append((name, attribute))
    return callables


def _classify_public_callable(facade: str, name: str) -> str:
    """Return ``writer`` | ``primitive`` | ``scientific`` for one exported callable."""
    if name in WRITER_CALLABLE_PREFIXES or name.startswith(
        tuple(f"{prefix}_" for prefix in WRITER_CALLABLE_PREFIXES)
    ):
        return "writer"
    if facade in PRIMITIVE_FACADES or name == "read" or name.startswith("read_"):
        return "primitive"
    return "scientific"


def _public_facades() -> tuple[str, ...]:
    """Canonical lazy ``ov.*`` facade names from the package export map."""
    import organelleverse

    public = cast("frozenset[str] | None", getattr(organelleverse, "_PUBLIC_MODULES", None))
    if not isinstance(public, frozenset) or not public:
        raise RuntimeError(
            "organelleverse canonical facade set (_PUBLIC_MODULES) is unavailable; the "
            "output-boundary scanner cannot enumerate the public surface"
        )
    return tuple(sorted(public))


def public_surface_final_write_violations() -> dict[str, tuple[str, ...]]:
    """Forbidden final-write parameters on every public scientific callable.

    The scanner inspects each callable exported by the canonical ``ov.*`` facades.
    Writers/materializers and reader/fetch/install/persistence primitives are
    classified explicitly and excluded, so only scientific callables that smuggle
    in a destination appear here. The result is sorted by dotted name.
    """
    import organelleverse

    violations: dict[str, tuple[str, ...]] = {}
    for facade_name in _public_facades():
        module = getattr(organelleverse, facade_name)
        for callable_name, callable_obj in _public_callables(module):
            if _classify_public_callable(facade_name, callable_name) != "scientific":
                continue
            signature = inspect.signature(callable_obj)
            forbidden = tuple(
                sorted(
                    parameter_name
                    for parameter_name in signature.parameters
                    if parameter_name in FORBIDDEN_FINAL_WRITE_PARAMS
                )
            )
            if forbidden:
                violations[f"{facade_name}.{callable_name}"] = forbidden
    return dict(sorted(violations.items()))


def _operation_parameter_names(registry: OperationRegistry, operation_id: str) -> frozenset[str]:
    """Parameter names declared on a released spec's registered JSON Schema."""
    schema = registry.parameter_schema(operation_id)
    properties = cast("Mapping[str, object] | None", schema.get("properties"))
    if not isinstance(properties, Mapping):
        return frozenset()
    return frozenset(properties.keys())


def capability_final_write_parameters(
    spec: OperationSpec,
    frozen_schema: Mapping[str, object],
) -> frozenset[str]:
    """Forbidden final-write names on one controlled-worker capability schema.

    This predicate deliberately has no READ, primitive, or writer exemption:
    bundle-local controlled workers are compute-first only, while writers and
    other destination-owning contracts require a later execution provider.
    """

    properties = frozen_schema.get("properties")
    if not isinstance(properties, Mapping):
        return frozenset()
    typed_properties = cast("Mapping[str, object]", properties)
    return frozenset(typed_properties) & FORBIDDEN_FINAL_WRITE_PARAMS


def _declares_writer_destination(registry: OperationRegistry, spec: OperationSpec) -> bool:
    return bool(_operation_parameter_names(registry, spec.operation_id) & WRITER_DESTINATION_PARAMS)


def is_writer_spec(spec: OperationSpec) -> bool:
    """Whether a spec satisfies the consume/write contract and declares writer intent.

    A writer is a CONSUME Result->Result operation whose id declares writer
    intent (``.write``/``.save``/``.export``/``.materialize``). Both signals are
    required: ``qc.assembly`` has the stage contract but no writer intent (and
    exposes ``workspace``, not a destination), so it is scientific, while
    ``annotation.write``/``qc.write`` declare intent and expose ``output``.
    """
    return (
        spec.stage is OperationStage.CONSUME
        and spec.input_kind is CoreKind.RESULT
        and spec.output_kind is CoreKind.RESULT
        and spec.operation_id.endswith(WRITER_OPERATION_SUFFIXES)
    )


def released_scientific_specs_with_final_write(
    registry: OperationRegistry,
) -> frozenset[str]:
    """Released scientific OperationSpecs that expose a final-write field.

    A scientific spec is any released spec that is not a writer. Writers are
    identified by the consume/write contract, not by parameter spelling, so a
    destination field on ``annotation.write`` does not register here while a
    managed ``workspace`` on ``qc.assembly`` does.
    """
    return frozenset(
        spec.operation_id
        for spec in registry.list()
        if spec.stage is not OperationStage.READ
        and not is_writer_spec(spec)
        and bool(
            _operation_parameter_names(registry, spec.operation_id) & FORBIDDEN_FINAL_WRITE_PARAMS
        )
    )


def released_writer_specs_missing_destination(
    registry: OperationRegistry,
) -> frozenset[str]:
    """Released writer specs that fail to expose a closed destination field.

    A writer is identified by the consume/write contract (CONSUME Result->Result
    stage plus writer intent in the operation id). Once a spec is a writer, its
    registered schema must expose a destination parameter; otherwise the writer
    cannot accept the destination that defines it.
    """
    return frozenset(
        spec.operation_id
        for spec in registry.list()
        if is_writer_spec(spec) and not _declares_writer_destination(registry, spec)
    )


__all__ = [
    "FORBIDDEN_FINAL_WRITE_PARAMS",
    "PRIMITIVE_FACADES",
    "WRITER_CALLABLE_PREFIXES",
    "WRITER_DESTINATION_PARAMS",
    "WRITER_OPERATION_SUFFIXES",
    "capability_final_write_parameters",
    "is_writer_spec",
    "public_surface_final_write_violations",
    "released_scientific_specs_with_final_write",
    "released_writer_specs_missing_destination",
]
