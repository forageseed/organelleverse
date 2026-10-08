"""Adapter decisions for the ``variation`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``variation.snp`` / ``variation.snp_density`` - ``canonical`` result_shape,
  first parameter ``alignment_fasta: str | Path`` (not a bare
  ``OrganelleGenome``), so neither matches ``canonical_core``. Resolved
  below: ``alignment_fasta`` gets the ``path`` codec (a real file
  ``read_fasta`` opens); both already return a real ``OrganelleResult`` they
  built themselves, so ``result_codec = "canonical"``. ``reference_index``
  (both) and ``window_size``/``step`` (``snp_density`` only) keep their own
  defaults.
* ``variation.compute_snp`` - the ``json`` twin (``variation/snp_core.py``,
  no ``organelleverse.core`` types): same ``alignment_fasta: str | Path``
  first parameter gets ``path``; its ``dict`` return is recursively finite
  JSON at every real call, so ``json_metric`` under ``"snp_metrics"``.
* ``variation.write_snp`` / ``.write_snp_density`` - both take
  ``result: OrganelleResult | Mapping[str, Any]`` as their required first
  parameter. No implemented ``ParameterCodec`` decodes an ``OrganelleResult``
  as an Agent-facing JSON argument (``LEGACY_RESULT`` is declared but
  unimplemented - ``python_binding._UNIMPLEMENTED_PARAMETER_CODECS``), so
  neither has an entry here - the generator fails closed for both with
  ``capability.no_adapter_override``.

Capability Plan 04 Task 1 adds ``FIXTURES`` for all three bindable
capabilities, sharing one 3-sequence, 41 bp alignment
(``_ALIGNMENT_FASTA``) - the reference (``reference_index=0`` default) plus
two sequences each carrying two real substitutions (one transition, one
transversion relative to the reference at each differing position),
confirmed interactively to produce 4 total SNPs / 1 transition / 3
transversions. No timestamp, no randomness: ``snp.py``'s own ``_provenance``
never sets ``started_at``/``finished_at``.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_REFERENCE_SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


def _with_substitutions(*positions_and_bases: tuple[int, str]) -> str:
    chars = list(_REFERENCE_SEQ)
    for position, base in positions_and_bases:
        chars[position] = base
    return "".join(chars)


_ALIGNMENT_FASTA = FixtureFile(
    relative_path="alignment.fasta",
    content=(
        f">ref\n{_REFERENCE_SEQ}\n"
        f">s2\n{_with_substitutions((3, 'G'), (10, 'T'))}\n"
        f">s3\n{_with_substitutions((3, 'A'), (20, 'G'))}\n"
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
    """The full ``named_parameters`` binding plan for one restored capability."""

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    "variation.snp": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "variation.snp_density": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "variation.compute_snp": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="snp_metrics",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``variation.write_snp`` and ``.write_snp_density`` are deliberately
absent - see the module docstring for why no codec can decode their
``OrganelleResult | Mapping`` parameter honestly.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    capability_id: (
        FixtureCase(
            case="basic",
            files=(_ALIGNMENT_FASTA,),
            parameters={"alignment_fasta": _ALIGNMENT_FASTA.relative_path},
        ),
    )
    for capability_id in ("variation.snp", "variation.snp_density", "variation.compute_snp")
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
