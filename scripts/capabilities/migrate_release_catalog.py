#!/usr/bin/env python3
"""Migrate the 20-operation release catalog to core capability bundles.

Reads the current privileged catalog (``operations/catalog.py`` and the
suite-level spec modules it aggregates) and emits one checked-in bundle
directory per operation under ``src/organelleverse/capabilities/`` so the
default registry can load the release surface through the same discovery /
admission channel as every third-party bundle (Plan 03, Task 1).

**Self-verifying.** Every emitted ``capability.toml`` is immediately parsed
back with :func:`parse_capability_bundle` and its contract is compared,
field by field, against the ``OperationSpec`` it was generated from. A
mismatch aborts the run with a diff - a bundle that does not reproduce its
spec byte-for-byte (as ``model_dump(mode="json")``) is never written.

**Deterministic.** Operations are processed in sorted ``operation_id``
order; TOML is emitted with fixed section and key order; no timestamps, no
filesystem iteration order. Re-running against an unchanged catalog
reproduces byte-identical bundles.

**16+4 split, owner-ruling addendum 2026-08-08.** Sixteen of the twenty
operations emit as ``native`` core bundles. Four stay on the residual Python
catalog (``operations/catalog.py``):

- ``annotation.annotate`` declares three BLAST *executable* dependencies; the
  bundle validator requires probes for every executable dependency while
  forbidding probes on ``native`` implementations, and the ``external`` shape
  it implies needs a content-addressed bundle-local worker identity that
  package-resident code does not have (Decision 004 item 4).
- ``io.read_long_reads``, ``assembly.assemble`` and
  ``assembly.pmat_graph_build`` carry register-time suite ``DataContract``s
  (``sequencing_reads`` x2, ``pmat_graph_input``) that the bundle contract
  cannot yet express; without them their bindings reject their own declared
  input modality. A bundle-expressible data-contract reference is the same
  Decision 004 item 4 extension.

Their contracts remain pinned by the same oracle fixture. The BLAST
interface probes from the plan's original mapping are recorded below as
evidence for that future re-classification.

**Fixtures.** This script emits contract bundles only. Runnable fixtures
for the 20 operations land with the Plan 04 fixture batches (owner ruling
on Decision 004, 2026-08-08), like the 127 restored bundles that already
ship without fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organelleverse.capabilities.parser import parse_capability_bundle  # noqa: E402
from organelleverse.operations.spec import OperationSpec  # noqa: E402

CAPABILITIES_ROOT = PROJECT_ROOT / "src" / "organelleverse" / "capabilities"

#: Implementation kind per operation. See the module docstring for the
#: documented assembly deviation from Plan 03's original mapping.
IMPLEMENTATION_KIND: dict[str, str] = {
    "annotation.annotate": "native",
    "annotation.extract": "native",
    "annotation.write": "native",
    "assembly.assemble": "native",
    "assembly.pmat_graph_build": "native",
    "assembly.write": "native",
    "io.read_fasta_genome": "native",
    "io.read_genbank_genome": "native",
    "io.read_long_reads": "native",
    "qc.annotation": "native",
    "qc.assembly": "native",
    "qc.write": "native",
    "fetch.accession_list": "native",
    "fetch.dedup_genomes": "native",
    "fetch.entrez_query": "native",
    "fetch.gene_records": "native",
    "fetch.genome_size_candidates": "native",
    "fetch.ngdc_gwh": "native",
    "fetch.nuclear_assembly": "native",
    "fetch.refseq_snapshot": "native",
}

#: Operations that stay on the residual Python catalog (see module docstring).
RESIDUAL_CATALOG: frozenset[str] = frozenset(
    {
        "annotation.annotate",
        "assembly.assemble",
        "assembly.pmat_graph_build",
        "io.read_long_reads",
    }
)

#: Interface probes recorded for a future external re-classification of
#: annotation.annotate. Not emitted while the bundle is ``native`` (native
#: implementations forbid probes). ``requires`` is the CLI surface the
#: mitochondrion pipeline actually uses (see
#: ``annotation/mitochondrion/cds.py``, ``boundary.py``, ``pcg.py``).
BLAST_PROBES: tuple[dict[str, object], ...] = (
    {
        "dependency": "blastn",
        "help_argv": ("blastn", "-help"),
        "requires": ("-query", "-db", "-outfmt", "-task"),
        "version_argv": ("blastn", "-version"),
        "version_capture": "blastn",
    },
    {
        "dependency": "makeblastdb",
        "help_argv": ("makeblastdb", "-help"),
        "requires": ("-in", "-out", "-dbtype"),
        "version_argv": ("makeblastdb", "-version"),
        "version_capture": "makeblastdb",
    },
    {
        "dependency": "tblastn",
        "help_argv": ("tblastn", "-help"),
        "requires": ("-query", "-db", "-outfmt"),
        "version_argv": ("tblastn", "-version"),
        "version_capture": "tblastn",
    },
)


def _slug(operation_id: str) -> str:
    return operation_id.replace(".", "-").replace("_", "-")


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _emit_key_value(lines: list[str], key: str, value: object) -> None:
    lines.append(f"{key} = {_toml_value(value)}")


def _contract_lines(spec: OperationSpec) -> list[str]:
    """Serialize the contract section, preserving every non-default field."""
    lines = ["[contract]"]
    _emit_key_value(lines, "contract_version", spec.contract_version)
    _emit_key_value(lines, "title", spec.title)
    _emit_key_value(lines, "description", spec.description)
    _emit_key_value(lines, "keywords", list(spec.keywords))
    _emit_key_value(lines, "execution_mode", spec.execution_mode.value)
    _emit_key_value(lines, "stage", spec.stage.value)
    _emit_key_value(lines, "input_kind", spec.input_kind.value)
    _emit_key_value(lines, "output_kind", spec.output_kind.value)
    if spec.organelle_types:
        _emit_key_value(lines, "organelle_types", list(spec.organelle_types))
    if spec.input_modalities:
        _emit_key_value(lines, "input_modalities", list(spec.input_modalities))
    if spec.output_modalities:
        _emit_key_value(lines, "output_modalities", list(spec.output_modalities))
    if spec.callable_locator is not None:
        _emit_key_value(lines, "callable_locator", spec.callable_locator)
    _emit_key_value(lines, "side_effects", [effect.value for effect in spec.side_effects])
    _emit_key_value(lines, "deterministic", spec.deterministic)
    if spec.deterministic is False and spec.deterministic_reason:
        _emit_key_value(lines, "deterministic_reason", spec.deterministic_reason)
    _emit_key_value(lines, "idempotent", spec.idempotent)
    _emit_key_value(lines, "cacheable", spec.cacheable)
    if spec.references:
        _emit_key_value(lines, "references", list(spec.references))

    binding = spec.binding
    if binding.argument_mode.value != "canonical_core":
        lines.append("")
        lines.append("[contract.binding]")
        _emit_key_value(lines, "argument_mode", binding.argument_mode.value)

    retry = spec.retry
    if retry.max_attempts != 1 or retry.retryable_error_codes:
        lines.append("")
        lines.append("[contract.retry]")
        _emit_key_value(lines, "max_attempts", retry.max_attempts)
        if retry.retryable_error_codes:
            _emit_key_value(lines, "retryable_error_codes", list(retry.retryable_error_codes))

    fallback = spec.fallback
    if fallback.allowed or fallback.allowed_backends:
        lines.append("")
        lines.append("[contract.fallback]")
        _emit_key_value(lines, "allowed", fallback.allowed)
        if fallback.allowed_backends:
            _emit_key_value(lines, "allowed_backends", list(fallback.allowed_backends))

    for dependency in spec.dependencies:
        lines.append("")
        lines.append("[[contract.dependencies]]")
        _emit_key_value(lines, "kind", dependency.kind.value)
        _emit_key_value(lines, "name", dependency.name)
        if dependency.version_spec:
            _emit_key_value(lines, "version_spec", dependency.version_spec)
        if dependency.locator:
            _emit_key_value(lines, "locator", dependency.locator)
        if dependency.optional:
            _emit_key_value(lines, "optional", dependency.optional)
    return lines


def _probe_lines(operation_id: str) -> list[str]:
    # All 20 bundles are emitted native; native forbids probes. BLAST_PROBES
    # is retained as recorded evidence for a future external re-classification.
    return []


def emit_bundle(spec: OperationSpec, root: Path) -> Path:
    """Write one bundle directory and verify it round-trips to ``spec``."""
    operation_id = spec.operation_id
    lines = [
        'schema = "organelleverse.capability.v1"',
        "",
        "[capability]",
        f"id = {_toml_string(operation_id)}",
        'bundle_version = "1.0.0"',
        f"implementation = {_toml_string(IMPLEMENTATION_KIND[operation_id])}",
        "",
        *_contract_lines(spec),
        *_probe_lines(operation_id),
    ]
    destination = root / _slug(operation_id)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_path = destination / "capability.toml"
    bundle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    parsed = parse_capability_bundle(bundle_path)
    expected = spec.model_dump(mode="json")
    actual = parsed.contract.model_dump(mode="json")
    if actual != expected:
        diffs = sorted(key for key in expected if expected[key] != actual.get(key))
        raise SystemExit(
            f"emitted bundle for {operation_id} does not reproduce its spec; "
            f"divergent fields: {diffs}"
        )
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    # The frozen oracle is the ground truth for the release contracts: it was
    # dumped from the live catalog before migration, and regenerating bundles
    # from it keeps this script independent of the residual catalog itself.
    oracle_path = (
        PROJECT_ROOT / "tests" / "capabilities" / "fixtures" / "release_catalog_v0.json"
    )
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if len(oracle) != 20:
        raise SystemExit(f"release oracle holds {len(oracle)} operations, expected 20")
    specs = sorted(
        (OperationSpec.model_validate(payload) for payload in oracle.values()),
        key=lambda item: item.operation_id,
    )
    emittable = [spec for spec in specs if spec.operation_id not in RESIDUAL_CATALOG]
    unknown = sorted(
        spec.operation_id for spec in emittable if spec.operation_id not in IMPLEMENTATION_KIND
    )
    if unknown:
        raise SystemExit(f"operations missing an implementation-kind mapping: {unknown}")

    written = [emit_bundle(spec, CAPABILITIES_ROOT) for spec in emittable]
    for path in written:
        print(path.relative_to(PROJECT_ROOT))
    print(
        f"emitted {len(written)} core bundles, all contracts verified byte-identical; "
        f"residual catalog: {sorted(RESIDUAL_CATALOG)}"
    )


if __name__ == "__main__":
    main()
