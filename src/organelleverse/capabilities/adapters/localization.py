"""Adapter decisions for the ``localization`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``localization.check_backend`` / ``.check_all_backends`` - resolved
  below: ``name`` (only on ``check_backend``) gets ``json``; both
  nested-dict returns are recursively finite JSON at every real call, so
  ``json_metric`` under ``"backend_status"`` for both - both really scan
  conda/micromamba environments (same pattern as the ``morphology``/
  ``population``/``pangenome`` precedents), so ``side_effects =
  ["subprocess"]`` for both.
* ``localization.install_hint`` - resolved below: ``name: str`` gets
  ``json``; the bare ``str`` return is finite JSON, so ``json_metric``
  under ``"hint_text"``.
* ``localization.predict`` - ``canonical`` result_shape, first parameter
  ``fasta: str | Path`` (not a bare ``OrganelleGenome``), so it does not
  match ``canonical_core``. Resolved below: ``fasta`` gets ``path`` (really
  opened via ``read_fasta``); the implementation already returns a real
  ``OrganelleResult`` it built itself, so ``canonical``. Every other
  parameter (including ``executor``, never Agent-exposed - a
  ``Callable[[list[str]], Any]`` cannot be a JSON argument, so an external
  ``deeploc``/``targetp`` backend, even when auto-selected, only ever plans)
  keeps its own default. ``side_effects = ["read_files", "subprocess"]`` -
  the backend auto-selection itself scans conda/micromamba environments via
  ``install.check_backend(..., scan_envs=True)``, independent of whether an
  external tool is ultimately run.
* ``localization.score_nls`` / ``.score_pts`` / ``.score_signal_peptide`` /
  ``.score_transit_peptides`` / ``.score_transmembrane`` - resolved below:
  all five take a required ``seq: str`` (plain scalar) as their only
  required parameter, so it gets ``json``; every ``dict[str, object]``
  return is recursively finite JSON at every real call (these are pure
  sequence-motif heuristics, no I/O), so ``json_metric`` under
  ``"signal_peptide"`` / ``"transit_peptides"`` / ``"nls"`` / ``"pts"`` /
  ``"transmembrane"`` respectively. ``minscore``/``organism`` (signal
  peptide), ``window`` (transit peptides/transmembrane), and ``threshold``
  (transmembrane) keep their own defaults. No side effects for any of the
  five.
* ``localization.predict_heuristic`` - **excluded**: returns
  ``localize_core.LocalizationScores``, a plain ``@dataclass`` (with a
  computed ``probabilities`` property), not JSON-safe and not an
  ``OrganelleResult`` - the same shape already documented in the ``hgt``
  adapter for ``HGTAlignment``/``HGTBlastHit``. No entry here.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``predict_heuristic`` is INTERNAL (LocalizationScores local-model
 return).
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
    "localization.check_backend": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "localization.check_all_backends": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "localization.install_hint": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hint_text",
        side_effects=(),
    ),
    "localization.predict": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "localization.score_nls": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="nls",
        side_effects=(),
    ),
    "localization.score_pts": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="pts",
        side_effects=(),
    ),
    "localization.score_signal_peptide": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="signal_peptide",
        side_effects=(),
    ),
    "localization.score_transit_peptides": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="transit_peptides",
        side_effects=(),
    ),
    "localization.score_transmembrane": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="transmembrane",
        side_effects=(),
    ),
}
"""``localization.predict_heuristic`` is deliberately absent - see the
module docstring for why its ``LocalizationScores`` return has no honest
codec.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
