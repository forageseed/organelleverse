"""Adapter decisions for the ``hgt`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``hgt.detect_hgt`` - ``canonical`` result_shape, ``donor_fasta``/
  ``recipient_fasta: str | Path`` (not a bare ``OrganelleGenome``), so it
  does not match ``canonical_core``. Resolved below: both get ``path`` (both
  are really opened via ``read_fasta`` inside ``_alignment_metrics``, which
  ``detect_hgt`` always calls); the implementation already returns a real
  ``OrganelleResult`` it built itself, so ``result_codec = "canonical"``.
  Every other parameter keeps its own default. ``side_effects =
  ["read_files"]``.
* ``hgt.compute_hgt`` - the ``json`` twin of ``detect_hgt``
  (``hgt/hgt_core.py``): same two required parameters get ``path``; its
  ``dict`` return is recursively finite JSON at every real call (either the
  full metrics dict, or the ``{"status": "failed", ...}`` degraded shape),
  so ``json_metric`` under ``"hgt_candidates"``. ``side_effects =
  ["read_files"]``.
* ``hgt.run_hgt_alignments`` / ``.run_blastn_confirmation`` - **excluded**:
  a new reason, distinct from every gap seen in prior domains - their return
  values (``tuple[str, list[HGTAlignment]]`` and ``list[HGTBlastHit]``
  respectively, both defined in ``hgt/align.py``) are plain ``@dataclass``
  instances, not ``dict``/``OrganelleResult``/``Path``/NumPy data. No
  implemented ``ResultCodec`` can honestly represent them:
  ``json_metric``'s own runtime check (``codecs._is_finite_json``) rejects
  any object that is not ``None``/``bool``/``str``/``int``/``float``/
  ``list``/``tuple``/``dict`` - a bare dataclass instance always fails it,
  deterministically, at every call (not merely a data-dependent edge case,
  unlike the ``diversity`` adapter's NaN caveat). Both dataclasses do carry
  an ``as_dict()`` method the *legacy* pre-v1 suite used to call before
  returning - but inventing a codec that silently calls an unrelated method
  the function itself never calls is exactly the fabrication this generator
  refuses to do. No entry here for either.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``run_blastn_confirmation``/``run_hgt_alignments`` are INTERNAL
 (subprocess runner helpers of the published detect_hgt pipeline; Decision
 004 item 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect


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
    "hgt.detect_hgt": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="donor_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="recipient_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "hgt.compute_hgt": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="donor_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="recipient_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hgt_candidates",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``hgt.run_hgt_alignments`` and ``.run_blastn_confirmation`` are
deliberately absent - see the module docstring for why neither's return
value has an honest codec.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
