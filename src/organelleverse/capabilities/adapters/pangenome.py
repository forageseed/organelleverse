"""Adapter decisions for the ``pangenome`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``pangenome.check_backend`` / ``.check_all_backends`` - resolved below:
  ``name`` (only on ``check_backend``) gets ``json``; both nested-dict
  returns are recursively finite JSON at every real call, so
  ``json_metric`` under ``"backend_status"`` for both - both really scan
  conda/micromamba environments (``assembly.install._list_conda_envs``,
  same pattern as the ``morphology``/``population`` precedents), so
  ``side_effects = ["subprocess"]`` for both.
* ``pangenome.install_hint`` - resolved below: ``name: str`` gets ``json``;
  the bare ``str`` return is finite JSON, so ``json_metric`` under
  ``"hint_text"``.
* ``pangenome.build_graph`` / ``.classify_sequences`` / ``.gene_pav`` /
  ``.pan_repeats`` bind through the reviewed sequence-of-core contract.
  ``build_graph`` points at the managed service, so its Agent schema exposes
  scientific parameters but no destination path or execution callback.
* ``pangenome.compute_gene_pav`` - **excluded**: its sole parameter,
  ``genomes``, carries *no annotation at all*
  (``pangenome/pangenome_core.py:8``: ``def compute_gene_pav(genomes) ->
  dict``). No codec can honestly bind an unannotated parameter - the same
  shape already documented for ``trans_splicing.compute_trans_splicing``.
  No entry here.
* ``pangenome.write_pav`` - **excluded**: required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim. No entry here.

**Ruling 3 follow-up (Decision 004 approval 2026-08-14):** of the four
sequence capabilities above, ``classify_sequences``/``gene_pav``/
``pan_repeats`` bind via the sequence-of-core extension. ``build_graph`` now
binds to ``pangenome.service:build_graph``: that managed public contract has
no destination or Callable parameter, while the legacy explicitly-placed
Python surface remains separate.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``compute_gene_pav`` is INTERNAL (unannotated genomes,
 sequence-non-core-return class).
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

EXCLUDE: frozenset[str] = frozenset()

#: Canonical signatures normally infer read-only effects.  This managed
#: builder additionally writes its governed run and starts a real backend.
CANONICAL_SIDE_EFFECTS: dict[str, tuple[SideEffect, ...]] = {
    "pangenome.build_graph": (
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
    )
}


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
    "pangenome.check_backend": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "pangenome.check_all_backends": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "pangenome.install_hint": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hint_text",
        side_effects=(),
    ),
}
"""Only ``compute_gene_pav`` and ``write_pav`` remain deliberately absent;
see the module docstring for their fail-closed rationale.
"""

__all__ = [
    "CANONICAL_SIDE_EFFECTS",
    "OVERRIDES",
    "NamedParameterOverride",
    "ParameterOverride",
]
