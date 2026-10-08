"""Adapter decisions for the ``transfer`` domain's restored capabilities.

**A second genuinely ungated real external-tool call, beyond
``phylogeny.align``.** ``transfer.detect_transfers_blast``
(``transfer/transfer.py:684-724``, delegating to
``_detect_transfers_blast_core``) has no ``executor`` parameter at all: it
resolves ``blastn``/``makeblastdb`` via ``shutil.which`` and, when both are
found, unconditionally runs ``subprocess.run([makeblastdb, ...])`` then
``subprocess.run([blastn, ...])`` - a real BLAST database build and search,
not a plan. This environment has real ``blastn``/``makeblastdb`` on
``PATH``, so the invoke test exercises the actual subprocess pipeline.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``transfer.compute_transfer`` - the ``json`` twin of the domain's k-mer
  detector (``transfer/transfer_core.py``, no ``organelleverse.core``
  types): both required ``target_fasta``/``source_fasta: str | Path`` get
  ``path`` (both really opened via ``read_fasta``); its ``dict`` return is
  recursively finite JSON at every real call, so ``json_metric`` under
  ``"transfer_fragments"``. ``side_effects = ["read_files"]``.
* ``transfer.detect_mtpt`` - ``canonical`` result_shape, both required
  parameters ``str | Path`` (not a bare ``OrganelleGenome``), so it does
  not match ``canonical_core``. Resolved below: ``mito_fasta``/``cp_fasta``
  both get ``path`` (delegates to the shared ``_detect_transfer``, which
  reads both files); the implementation already returns a real
  ``OrganelleResult`` it built itself, so ``canonical``. ``k``/``min_len``
  keep their own defaults. ``side_effects = ["read_files"]``.
* ``transfer.detect_transfers_blast`` - resolved below:
  ``nuclear_fasta``/``organelle_fasta`` both get ``path`` (both are only
  ever passed by ``str()`` to ``makeblastdb``/``blastn`` argv - never opened
  directly in Python, so no ``read_files`` claim); ``canonical`` result.
  Every other parameter keeps its own default. ``side_effects =
  ["subprocess"]`` - see the module docstring above; this really runs.
* ``transfer.detect_transfers_evidence`` - ``canonical`` result_shape,
  first parameter ``nuclear_fasta: str | Path | OrganelleData`` (a 3-way
  union, not a bare core type), so it does not match ``canonical_core``.
  Resolved below: ``nuclear_fasta`` gets ``path`` - the same single-arm
  reasoning as the ``morphology`` adapter's ``measure``/``segment``
  (``_resolve_path_parameter_plan`` only needs *one* union member to be
  ``str``/``Path``, and the generator's own ``_looks_like_path_annotation``
  recurses left-to-right through the ``BinOp`` chain and finds ``str``
  first) - the ``OrganelleData`` calling convention is simply not exposed
  to the Agent. Every other parameter, including ``organelle_fasta``
  (``str | Path | OrganelleGenome | OrganelleData | None``), has a default
  and is left uncovered. The implementation already returns a real
  ``OrganelleResult``, so ``canonical``. ``side_effects = ["read_files",
  "subprocess"]`` - the pipeline (layer 1 BLASTN, optional depth/long-read
  validation) genuinely reads and can spawn ``blastn``/``minimap2`` when
  the relevant optional inputs are given; with only ``nuclear_fasta``
  supplied it degrades gracefully rather than raising.
* ``transfer.annotate_nuclear_locus`` / ``.annotate_organelle_genes`` -
  **excluded**: both take a required ``candidates: list[TransferCandidate]``
  parameter and return ``list[TransferCandidate]``.
  ``transfer.TransferCandidate`` is a plain ``@dataclass``, not JSON-safe
  under ``operations.signature._is_supported_json_annotation`` (same shape
  as the ``hgt`` adapter's ``HGTAlignment``/``HGTBlastHit`` finding, on both
  the parameter and the return side here) and not a core type
  ``canonical_core`` recognizes. No entry here for either.
* ``transfer.detect`` - **excluded**: every parameter is explicitly typed
  ``Any`` (``transfer/transfer.py:119-127`` - ``nuclear_fasta: Any``,
  ``organelle_fasta: Any``, etc.), which both ``ParameterCodec.PATH``
  (``_looks_like_path_annotation`` only recognizes ``str``/``Path``
  names or unions naming one) and ``ParameterCodec.JSON``
  (``_is_supported_json_annotation`` explicitly returns ``False`` for
  ``Any``) refuse outright. No entry here.
* ``transfer.validate_transfers_depth`` / ``.validate_transfers_longread`` -
  **excluded**: both take a required ``candidates: list[TransferCandidate]
  | list[dict]`` parameter. Neither union arm is JSON-safe:
  ``list[TransferCandidate]`` for the same dataclass reason as
  ``annotate_nuclear_locus`` above, and the bare, unparameterized
  ``list[dict]`` for the same reason already documented in the
  ``phylogeny`` adapter (``get_origin(dict) is None``). A ``Union``
  requires *every* arm to be JSON-safe
  (``_is_supported_json_annotation``'s ``Union``/``UnionType`` branch), so
  both failing independently is doubly conclusive, not merely additive. No
  entry here for either.
* ``transfer.write_fragments`` - **excluded**: required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim. No entry here.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``detect`` is INTERNAL (a ``**kwargs`` facade of the published
 ``detect_transfers_evidence``); ``annotate_nuclear_locus``/
 ``annotate_organelle_genes`` are INTERNAL (TransferCandidate local-model
 returns).
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
    # Zero-out batch: candidate rows carry the TransferCandidate field names
    # as JSON primitives; the model arm stays runtime-accepted (isinstance).
    "transfer.validate_transfers_depth": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="candidates", codec=ParameterCodec.JSON),
            ParameterOverride(name="bam_path", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "transfer.validate_transfers_longread": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="nuclear_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="candidates", codec=ParameterCodec.JSON),
            ParameterOverride(name="hifi_reads", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "transfer.compute_transfer": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="target_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="source_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="transfer_fragments",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "transfer.detect_mtpt": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="mito_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="cp_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "transfer.detect_transfers_blast": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="nuclear_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="organelle_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "transfer.detect_transfers_evidence": NamedParameterOverride(
        parameters=(ParameterOverride(name="nuclear_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
}
"""``transfer.annotate_nuclear_locus``, ``.annotate_organelle_genes``,
``.detect``, ``.validate_transfers_depth``, ``.validate_transfers_longread``,
and ``.write_fragments`` are deliberately absent - see the module docstring
for why each fails closed honestly.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
