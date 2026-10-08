"""Adapter decisions for the ``rna_editing`` domain's restored capabilities.

``rna_editing.predict_edits``, ``.predict_edits_deepred``,
``.predict_edits_plantc2u``, and ``.predict_edits_prep`` all match the
generator's generic ``canonical_core`` shape on their own (``genome:
OrganelleGenome`` first parameter, ``-> OrganelleResult``) - no entries
needed here for any of the four.

**A fourth blocker class, new in this domain and structural rather than
signature-shaped: two ledger ids are not valid ``operation_id`` values at
all.** ``rna_editing._deepred.score_cytidines`` and
``rna_editing._plantc2u.score_cytidines`` each embed their private,
underscore-prefixed submodule name as a second path segment
(``_deepred``/``_plantc2u``). ``CapabilityBundle``'s schema requires
``contract.operation_id`` to match ``^[a-z][a-z0-9_]*\\.[a-z][a-z0-9_]*$`` -
*exactly one* dot, and every segment must start with a lowercase letter (a
leading ``_`` never matches). This was tried directly: rendering a
``capability.toml`` for ``rna_editing._plantc2u.score_cytidines`` (even
with an otherwise well-formed ``named_parameters`` override) makes
``parser.parse_capability_bundle`` raise ``OrganelleContractError`` with
``pydantic``'s ``string_pattern_mismatch`` on ``contract.operation_id`` -
and because the generator's own ``build_domain`` loop does not catch that
error, it is fatal to the *whole domain's* generation run, not just a
per-capability skip. Across the entire 253-record ledger these are the
*only* two ids that fail this pattern (checked directly against every
``id``), so no entry for either exists below - both fail closed with
``capability.no_adapter_override``, which is honest and does not require
touching the generator itself. (Their return shapes are separately
unrepresentable too - see below - but the id-pattern violation would block
both regardless.)

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``rna_editing.encode_matrix`` - resolved below: ``seqs: list[str]`` gets
  ``json``; the bare ``np.ndarray`` return is accepted directly by
  ``ResultCodec.ARTIFACT`` (``codecs._try_encode_numpy_artifact``), the
  same pattern as ``morphology.overlay``. Pure NumPy one-hot encoding, no
  model load - invoked for real in the test suite. ``seq_len`` keeps its
  own default.
* ``rna_editing.change_n`` - resolved below: ``seq: str`` gets ``json``;
  the bare ``str`` return is finite JSON, so ``json_metric`` under
  ``"sequence"``.
* ``rna_editing.extract_window`` - resolved below: all three required
  parameters (``sequence: str``, ``position_1based: int``, ``strand: int``)
  get ``json``; the bare ``str`` return is finite JSON, so ``json_metric``
  under ``"window"``. ``flank`` keeps its own default.
* ``rna_editing.extract_windows`` - resolved below: ``fin: str | Path``
  gets ``path`` (really opened via ``Path(fin).read_text()`` in
  ``_read_fasta``); the ``dict[str, str]`` return is recursively finite
  JSON at every real call, so ``json_metric`` under ``"windows"``.
  ``side_effects = ["read_files"]``.
* ``rna_editing.plantc2u_model_path`` - resolved below: zero parameters;
  returns a real, pre-existing packaged file
  (``rna_editing/data/plantc2u_flank90.hdf5``, confirmed present in this
  checkout), so ``result_codec = "artifact"`` (bare ``Path`` accepted
  directly). ``side_effects = ["read_files"]``.
* ``rna_editing.deepredmt_model_path`` - **excluded**: returns
  ``rna_editing/data/deepredmt_210520.tf`` - a TensorFlow *SavedModel
  directory* (confirmed present as a directory, not a file, in this
  checkout), the same directory-valued-result gap already documented in
  the ``phenotype`` adapter for ``cms_data_dir``/``cms_model_dir``. No
  entry here.
* ``rna_editing._deepred.score_cytidines`` / ``._plantc2u.score_cytidines``
  - **excluded**: invalid ``operation_id`` (see above). Independently,
  neither return shape has an honest codec either:
  ``_deepred.score_cytidines`` returns ``tuple[list[str], np.ndarray]`` - a
  tuple *mixing* a plain list with a NumPy array, which
  ``ResultCodec.ARTIFACT`` does not accept (only a bare ``np.ndarray``, a
  bare ``Path``, a sequence of ``Path``, or an ``OrganellePlot``) and
  ``JSON_METRIC`` rejects outright (the ``np.ndarray`` member is not one of
  ``_is_finite_json``'s recognized types); ``_plantc2u.score_cytidines``
  returns a bare ``np.ndarray``, which *would* resolve via ``artifact`` on
  its own (same as ``encode_matrix`` above) were its id valid. No entry
  here for either.
* ``rna_editing.validate_edits`` - **excluded**: required
  ``predicted_sites: list[dict]`` - the bare-``dict``-in-``list`` gap
  first documented in the ``phylogeny`` adapter, since Capability Plan 03
  Task 4 rejected by ``_looks_json_safe`` itself at generation time too
  (matching the real runtime ``_is_supported_json_annotation``, which has
  always rejected it since a bare ``dict`` has no ``get_origin``). No entry
  here.
* ``rna_editing.write_sites`` - **excluded**: required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim. No entry here.
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
    "rna_editing.detect_editing_sites": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="bam_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="annotation_genbank", codec=ParameterCodec.PATH),
            ParameterOverride(name="dna_bam_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="scope", codec=ParameterCodec.JSON),
            ParameterOverride(name="library_type", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_mapping_quality", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_base_quality", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_depth", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_edited_reads", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_editing_fraction", codec=ParameterCodec.JSON),
            ParameterOverride(name="trim_read_ends", codec=ParameterCodec.JSON),
            ParameterOverride(name="exclude_duplicates", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_dna_depth", codec=ParameterCodec.JSON),
            ParameterOverride(name="max_dna_nonref_fraction", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    # Zero-out batch: site rows are primitive mappings; both evidence
    # sources are optional staged-input paths (exactly one is required by
    # the implementation itself, which fails clearly when both are absent).
    "rna_editing.validate_edits": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="predicted_sites", codec=ParameterCodec.JSON),
            ParameterOverride(name="bam_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="pileup_tsv", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "rna_editing.encode_matrix": NamedParameterOverride(
        parameters=(ParameterOverride(name="seqs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "rna_editing.change_n": NamedParameterOverride(
        parameters=(ParameterOverride(name="seq", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="sequence",
        side_effects=(),
    ),
    "rna_editing.extract_window": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="sequence", codec=ParameterCodec.JSON),
            ParameterOverride(name="position_1based", codec=ParameterCodec.JSON),
            ParameterOverride(name="strand", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="window",
        side_effects=(),
    ),
    "rna_editing.extract_windows": NamedParameterOverride(
        parameters=(ParameterOverride(name="fin", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="windows",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "rna_editing.plantc2u_model_path": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``rna_editing._deepred.score_cytidines``, ``._plantc2u.score_cytidines``,
``.deepredmt_model_path``, ``.validate_edits``, and ``.write_sites`` are
deliberately absent - see the module docstring for why each fails closed
honestly.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
