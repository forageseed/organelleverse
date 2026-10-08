"""Adapter decisions for the ``format_conversion`` domain's restored capabilities.

Chosen as Capability Plan 03 Task 3's proof-of-mechanism domain: it is the
smallest restored domain (4 capabilities, ``docs/operations/restored-
capabilities.toml``) that both declares a ``custom`` ``result_shape`` - the
shape ``build_restored_bundles.py`` refuses to resolve generically - and
actually yields at least one bindable capability under that shape.

``ir_boundary`` (3 capabilities) is nominally smaller and also declares one
``custom`` entry, but every one of its 3 capabilities takes an
``OrganelleGenome``/``OrganelleResult``-shaped parameter that no current codec
can decode: ``ir_boundary()`` and ``compute_ir_boundary()`` both take a
(``list[OrganelleGenome]`` / bare ``list``) parameter, and
``write_boundary_svg()`` takes ``OrganelleResult | Mapping[str, Any]``. The
``LEGACY_GENOME``/``LEGACY_DATA``/``LEGACY_RESULT`` parameter codecs exist in
the schema for exactly this shape but are explicitly unimplemented
(``organelleverse.operations.python_binding._UNIMPLEMENTED_PARAMETER_CODECS``),
and a bare, untyped ``list`` fails the JSON codec's own annotation check
(``_is_supported_json_annotation`` requires a *parameterized* sequence). All
three would fail closed - a domain adapter cannot honestly fix that without
either restoring the forbidden legacy shim or fabricating a decoder, so
``ir_boundary`` was rejected as this slice's domain and ``format_conversion``
was chosen instead: still the smallest domain where the mechanism actually
admits a bundle under the ``custom`` shape.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``format_conversion.convert`` - ``canonical`` result_shape, first parameter
  typed ``OrganelleGenome``, returns ``OrganelleResult``: matches the
  generator's generic ``canonical_core`` shape detector on its own. No entry
  needed here.
* ``format_conversion.write_conversion`` - ``json`` result_shape, but its
  first parameter is typed ``OrganelleResult | OrganelleGenome``: a union no
  codec can decode (the JSON codec requires a JSON-safe annotation; both
  legacy codecs are unimplemented, same as above). No entry here either - the
  generator fails closed for it with ``capability.no_adapter_override``,
  honestly, rather than silently dropping it or fabricating a decoder.
* ``format_conversion.convert_genbank_to_gff3`` and
  ``.convert_genbank_to_fasta`` - ``custom`` result_shape, single
  ``genbank_path: str`` parameter, return a finite-JSON value (``list[str]``
  / ``list[tuple[str, str]]``) that the archived code never wrapped in an
  ``OrganelleResult``. Both resolved below: ``genbank_path`` is a real
  filesystem path the implementation reads, so it gets the ``path`` codec
  (declaring ``side_effects = ["read_files"]``, honestly, rather than the
  weaker ``json`` codec that would accept any string without checking the
  file exists); the ``custom`` return value is already recursively finite
  JSON, so it resolves to ``json_metric`` under a descriptive metric key.

Capability Plan 04 Task 1 adds ``FIXTURES`` for both. ``parse_genbank`` is a
pure, deterministic parser (no timestamps, no network); the sample record
below is the same minimal, valid GenBank flat file
``tests/annotation/test_genbank.py`` already uses for its own parser
fixtures (one 12 bp locus, one ``CDS`` split across two segments).
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_GENBANK_FASTA = FixtureFile(
    relative_path="record.gb",
    content=(
        "LOCUS       R1                        12 bp    DNA     linear   PLN 01-JAN-2025\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             join(1..3,10..12)\n"
        '                     /gene="nad5"\n'
        "ORIGIN\n"
        "        1 aaacccgggttt\n"
        "//\n"
    ),
)


@dataclass(frozen=True)
class ParameterOverride:
    """One named, Agent-facing parameter the generator could not derive alone."""

    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class NamedParameterOverride:
    """The full ``named_parameters`` binding plan for one restored capability.

    Every field here is exactly what ``build_restored_bundles.py`` cannot
    honestly infer from the ledger and an AST-only read of the source: which
    parameters are Agent-facing, what each one's codec is, which closed
    ``ResultCodec`` its ``custom`` (or otherwise ambiguous) return value
    resolves to, and - for ``json_metric`` - the metric key it is filed
    under.
    """

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    "format_conversion.convert_genbank_to_gff3": NamedParameterOverride(
        parameters=(ParameterOverride(name="genbank_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="gff3_lines",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "format_conversion.convert_genbank_to_fasta": NamedParameterOverride(
        parameters=(ParameterOverride(name="genbank_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="fasta_records",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""Every restored id this domain resolves outside the generic ``canonical_core``
path. ``format_conversion.write_conversion`` is deliberately absent: no codec
can decode its ``OrganelleResult | OrganelleGenome`` parameter honestly, so it
is not this adapter's job to invent one - the generator fails closed for it.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    capability_id: (
        FixtureCase(
            case="basic",
            files=(_GENBANK_FASTA,),
            parameters={"genbank_path": _GENBANK_FASTA.relative_path},
        ),
    )
    for capability_id in (
        "format_conversion.convert_genbank_to_gff3",
        "format_conversion.convert_genbank_to_fasta",
    )
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
