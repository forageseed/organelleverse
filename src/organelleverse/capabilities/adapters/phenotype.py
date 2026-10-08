"""Adapter decisions for the ``phenotype`` domain's restored capabilities.

**A new gap discovered in this domain: a directory-valued *result* is just
as unbindable as a directory-valued *parameter*, and for the same reason.**
``phenotype.cms_data_dir``/``.cms_model_dir`` (``phenotype/cms/resources.py``)
take zero parameters and return a bare ``Path`` - which
``ResultCodec.ARTIFACT`` accepts directly (``codecs._encode_artifact``) -
but both real return values are *directories*
(``phenotype/cms/data/`` and its ``models/`` subdirectory), and
``codecs._encode_path_artifacts`` requires ``path.is_file()`` on every
artifact path before it will materialize one - the same ``is_file()``
preflight the charter's directory-*parameter* gap already names, just
applied on the result side instead of the parameter side. Both are excluded
below. By contrast, ``cms_reference_json``/``cms_protein_fasta`` return real
*files* that ship inside the package
(``phenotype/cms/data/curated/cms_reference.json``,
``phenotype/cms/data/curated/cms_proteins.fasta``, both present on disk in this
checkout) - the artifact codec's preflight passes for both, so both
resolve.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``phenotype.cms`` - ``canonical`` result_shape, first parameter
  ``genome_fasta: str | Path`` (not a bare ``OrganelleGenome``), so it does
  not match ``canonical_core``. Resolved below: ``genome_fasta`` gets
  ``path`` (really opened via ``read_fasta``); the implementation already
  returns a real ``OrganelleResult`` it built itself, so ``result_codec =
  "canonical"``. ``min_orf_aa``/``tm_window``/``tm_threshold`` keep their
  own defaults. ``side_effects = ["read_files"]``.
* ``phenotype.cms_data_dir`` / ``.cms_model_dir`` - **excluded**: both
  return a directory ``Path`` - see the module docstring above. No entry
  here for either.
* ``phenotype.cms_protein_fasta`` / ``.cms_reference_json`` - resolved
  below: both take zero parameters (mirrors
  ``morphology.check_all_backends``); both return a real, pre-existing
  packaged file, so ``result_codec = "artifact"`` (``codecs._encode_artifact``
  accepts a bare ``Path`` directly). ``side_effects = ["read_files"]``
  (both resolve a path under the installed package, no computation).
* ``phenotype.compute_cms_candidates`` - the ``json`` twin of ``cms``
  (``phenotype/cms/core.py``, no ``organelleverse.core`` types): its first
  parameter is ``fasta_path`` (not ``genome_fasta``, though the same shape)
  and gets ``path``; its ``dict`` return is recursively finite JSON at
  every real call, so ``json_metric`` under ``"cms_candidates"``.
  ``side_effects = ["read_files"]``.
* ``phenotype.download_cms_sequences`` - exposes ``email`` explicitly,
  downloads only the pinned accession-checked seed, checks protein ID, length
  and full sequence, and atomically publishes beneath the managed cache.
  The reviewed package reference is never overwritten. Its result is the
  cached FASTA artifact, with network/read/write side effects and no BLAST
  database subprocess. Tests substitute Entrez responses and validate that
  failed downloads preserve earlier cache and packaged files.

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
    "phenotype.cms": NamedParameterOverride(
        parameters=(ParameterOverride(name="genome_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phenotype.cms_protein_fasta": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phenotype.cms_reference_json": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phenotype.compute_cms_candidates": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="cms_candidates",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phenotype.download_cms_sequences": NamedParameterOverride(
        parameters=(ParameterOverride(name="email", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.NETWORK, SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    ),
}
"""``phenotype.cms_data_dir`` and ``.cms_model_dir`` are deliberately
absent - see the module docstring for why a directory-valued result is
unbindable by ``ResultCodec.ARTIFACT``, same as a directory-valued
parameter is unbindable by ``ParameterCodec.PATH``.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
