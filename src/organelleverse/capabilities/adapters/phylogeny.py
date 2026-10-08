"""Adapter decisions for the ``phylogeny`` domain's restored capabilities.

Chosen as Capability Plan 03 Task 3's next domain per the charter: none of
this domain's 20 ledger records matches the generator's generic
``canonical_core`` shape (every first parameter is a plain ``str | Path``,
``list[...]``, or scalar - never a bare ``OrganelleGenome``/``OrganelleData``/
``OrganelleResult``), so every one of the 10 resolved below required an
explicit override, and ``phylogeny.align`` is this slice's proof of the
genuinely ungated real external-tool call the charter asked for.

**``phylogeny.align`` really does spawn ``mafft``.** Unlike every ``tool``
record examined in the ``morphology`` adapter's survey, ``align()``
(``phylogeny/phylo.py:33-120``) has no ``executor: Callable[...] | None``
gate at all: when ``method`` is ``"auto"`` or ``"mafft"`` and the self
-contained Rust port is unavailable, it calls ``shutil.which("mafft")`` and,
if found, unconditionally runs ``subprocess.run([mafft_bin, "--auto", "-"],
...)`` with the real FASTA on stdin - not behind an injectable callable, not
merely a probe. This environment has a real ``mafft`` (``v7.526``) on
``PATH``, so the invoke test below runs the actual binary and asserts on the
real alignment it returns (not the ``concat_placeholder`` degraded path).

**Two new gap shapes beyond the charter's two named ones, both of the same
underlying kind: this generator's own AST-level pre-validation
(``_looks_json_safe``/``_looks_like_path_annotation``) is more lenient than
the real runtime binder (``operations.signature._is_supported_json_annotation``),
so an override that passes bundle *generation* can still fail bundle
*binding* - caught here empirically, not just reasoned about, exactly the
same shape the ``morphology`` adapter already documented for ``Any``-typed
containers.**

1. **A bare, unparameterized ``dict`` inside a ``list[...]`` is not
   JSON-safe at runtime.** ``build_mjn``, ``build_msn``, ``build_tcs_network``,
   and ``pairwise_distances`` all take a required ``haplotypes: list[dict]``
   parameter. At the time this was first tried, ``_looks_json_safe`` only
   inspected a ``Subscript``'s outer name (``"list"`` is JSON-safe, full
   stop - it never looked at what was inside), so declaring
   ``codec = "json"`` for ``haplotypes`` passed generation while
   ``_is_supported_json_annotation`` requires a *mapping* annotation to
   carry ``get_origin() in {dict, Mapping}`` with exactly two type arguments
   (key ``str``, value JSON-safe) - a bare, un-subscripted ``dict`` has
   ``get_origin(dict) is None``, so it fell through every branch and
   returned ``False``. This was tried and verified: binding
   ``phylogeny.build_mjn`` at ``verify_capability`` time raised
   ``OrganelleContractError: parameter 'haplotypes' has no JSON-safe
   annotation for codec 'json'``. **Capability Plan 03 Task 4 closed this
   generator/runtime disagreement**: ``_looks_json_safe`` now also rejects a
   bare, unparameterized container (``dict``, ``list``, ``tuple``,
   ``Sequence``, ``Mapping``) used directly as a type argument, so
   ``list[dict]`` now fails closed at *generation* time too, the same
   outcome, just caught earlier. All four capabilities stay excluded below
   - none has a second parameter that could carry the binding alone.
2. **An *output* path is unbindable by either codec, for two different
   reasons that only cancel out by accident.**
   ``python_binding._prepare_path_parameters``
   (``operations/python_binding.py:936-940``) requires every PATH-codec
   value to already satisfy ``candidate.is_file()`` *before* the
   implementation runs - it is exclusively an input-artifact codec
   (existence check + content hash), never a write-destination one.
   ``render_network``'s required ``output: str | Path`` parameter is a
   *destination* the function itself creates (``output.parent.mkdir(...)``
   then ``fig.savefig(...)`` or the ``_write_dot`` fallback) - a real,
   pre-existing file at that path would be nonsensical to demand, so
   ``PATH`` is out. ``ParameterCodec.JSON`` looks like the honest
   alternative - at the real runtime binder,
   ``_is_supported_json_annotation`` accepts ``str | Path`` under its
   registry-admission default (``annotation is Path`` short-circuits true,
   same as every member of a ``Union``) - but the bundle codec wall's
   strict variant (``allow_path=False``, in force since the bare-``Path``
   JSON hole closed) rejects it, and this generator's own
   ``_looks_json_safe`` only recognizes ``{str, int, float, bool}`` as
   JSON-safe *names*; it has no entry for ``Path`` at all, so declaring
   ``codec = "json"`` for ``output`` fails closed at *generation* time
   (verified: ``capability.adapter_override_invalid``, "annotation is not
   JSON-safe"). Here the generator's own check is the wall, the opposite
   direction from gap 1 above - but the practical outcome is identical: no
   entry, ``render_network`` is excluded.

**A third codec choice, revised when the bare-``Path`` JSON hole closed
(decision 004 §5): ``genbank_paths: list[str | Path]`` now takes
``ParameterCodec.PATH``, not JSON.** The JSON route this domain originally
used was withdrawn: ``Path`` is no longer JSON-safe at the bundle codec
wall (``_is_supported_json_annotation(..., allow_path=False)`` - the
registry-admission default keeps admitting it for the released operations,
a separate, deliberate wall), so a ``list[str | Path]`` JSON parameter -
decoded with no existence check and no content hashing - is no longer
bindable. The PATH codec learned lists for exactly this case
(``_resolve_path_parameter_plan`` accepts a one-argument
``list[...]``/``Sequence[...]`` of path-likes): every element must be an
existing file (``is_file()`` preflight, else ``input.missing_artifact``)
whose content hash lands in ``input_artifact_hashes`` before the
implementation ever runs, and the ``str`` union arm means the
implementation receives the same list of strings it always did. This is the
honest codec for the parameter, verified empirically in the test suite (a
real ``genbank_paths`` list of on-disk files is submitted and the
implementation's own ``str(gbk)`` calls handle them exactly as it always
has).

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``phylogeny.align`` - resolved below: ``input_fasta`` gets ``path``
  (a real file ``read_fasta`` opens); implementation already returns a real
  ``OrganelleResult`` it built itself, so ``result_codec = "canonical"``.
  ``method`` keeps its own default. ``side_effects = ["read_files",
  "subprocess"]`` - both genuinely happen.
* ``phylogeny.trim_alignment`` / ``.build_tree`` - resolved below:
  ``alignment_fasta`` gets ``path``; ``canonical`` result (both already
  build a real ``OrganelleResult``). Neither function ever opens
  ``alignment_fasta`` itself (only ``str()``-embeds it into a planned
  ``argv``), and ``executor`` is never Agent-exposed (a
  ``Callable[[list[str]], Any]`` cannot be a JSON argument), so every real
  call takes the plan-only branch - no ``read_files``/``subprocess``/
  ``write_files`` side effect is honestly claimable, so ``side_effects = ()``
  for both (contrast with ``align``, which really does call
  ``read_fasta``/spawn a subprocess).
* ``phylogeny.run_build_tree`` / ``.run_trim_alignment`` - resolved below:
  ``alignment_fasta`` gets ``path`` (same reasoning as ``build_tree`` /
  ``trim_alignment``, never opened - only ``str()``-embedded). The public
  service allocates managed output; Agent calls omit ``executor`` and do
  not create empty run directories. Both return a canonical result.
  Verified by real invocation:
  ``_build_tree_impl``/``_trim_alignment_impl`` hardcode their own
  ``"build_tree"``/``"trim_alignment"`` op names regardless of caller, so
  the returned ``OrganelleResult`` carries ``operation_id =
  "phylogeny.build_tree"``/``"phylogeny.trim_alignment"``, not
  ``"phylogeny.run_build_tree"``/``"phylogeny.run_trim_alignment"`` - the
  same cross-function ``operation_id`` quirk already documented for
  ``population.detect_numt``, left exactly as found.
* ``phylogeny.compute_alignment`` - the ``json`` twin of ``align``
  (``phylo_core.py``, no ``organelleverse.core`` types): ``input_fasta``
  gets ``path``; its ``{"sequences": [...], "n_sequences": int}`` return is
  recursively finite JSON at every real call, so ``json_metric`` under
  ``"alignment"``.
* ``phylogeny.plan_tree_build`` - resolved below: ``alignment_fasta`` gets
  ``path``; ``{"argv": [...], "method": str}`` return is finite JSON, so
  ``json_metric`` under ``"tree_build_plan"``. ``method``/``bootstrap``/
  ``seed``/``threads`` keep their own defaults.
* ``phylogeny.collapse_haplotypes`` - resolved below: ``seqs: list[tuple[str,
  str]]`` gets ``json`` (fully parameterized, so - unlike ``list[dict]``
  above - honestly JSON-safe at runtime too); its 4-tuple return is
  recursively finite JSON, so ``json_metric`` under ``"haplotype_collapse"``.
  ``population_map`` keeps its own default.
* ``phylogeny.pdistance`` - resolved below: ``a``/``b`` (plain ``str``) get
  ``json``; the bare ``int`` return is finite JSON, so ``json_metric`` under
  ``"p_distance"``.
* ``phylogeny.pairwise_distances`` / ``.build_msn`` / ``.build_tcs_network``
  / ``.build_mjn`` - **excluded**: each has a required ``haplotypes:
  list[dict]`` parameter - the bare-``dict``-in-``list`` gap described
  above. No entry here.
* ``phylogeny.tcs_connection_limit`` - resolved below: ``n_haplotypes: int``
  gets ``json``; the bare ``int`` return is finite JSON, so ``json_metric``
  under ``"tcs_connection_limit"``. ``sequence_length``/``confidence`` keep
  their own defaults.
* ``phylogeny.haplotype_network`` - resolved below: ``alignment_fasta`` gets
  ``path`` (really opened via ``read_fasta``); implementation already
  returns a real ``OrganelleResult``, so ``canonical``. Every other
  parameter (``method``, ``connection_limit``, ``confidence``, ``epsilon``,
  ``population_map``, ``title``, ``backend``) keeps its own default -
  ``backend`` is never Agent-exposed, so the external ``hapnet`` package
  path is never taken; every real call runs the in-tree TCS/MJN/MSN
  algorithm. ``side_effects = ["read_files"]``.
* ``phylogeny.extract_shared_genes`` - resolved below: ``genbank_paths:
  list[str | Path]`` gets ``path`` (the PATH codec's list form - see the
  module docstring above for why the JSON route was withdrawn when the
  bare-``Path`` JSON hole closed). ``feature_type`` keeps its own
  default. The implementation already returns a real ``OrganelleResult``, so
  ``canonical``. ``side_effects = ["read_files"]`` (each GenBank path is
  really opened via Biopython's ``SeqIO.parse``, when biopython is present -
  the implementation degrades to a ``failed`` ``OrganelleResult`` otherwise,
  which is still a real, valid ``canonical`` result).
* ``phylogeny.render_network`` - **excluded**: every parameter except
  ``output: str | Path`` would resolve cleanly (``haplotypes``/``members``/
  ``freq``/``edges`` are all honestly ``json``, and the bare ``Path`` return
  would have been accepted directly by ``ResultCodec.ARTIFACT``), but
  ``output`` itself is unbindable by either implemented codec - see gap 2
  above. No entry here.
* ``phylogeny.write_alignment`` - **excluded**: required first parameter
  ``result: OrganelleResult`` (bare, not even the ``| Mapping`` union seen
  elsewhere). No implemented ``ParameterCodec`` decodes an ``OrganelleResult``
  from Agent-facing JSON (``LEGACY_RESULT`` is declared but unimplemented -
  ``python_binding._UNIMPLEMENTED_PARAMETER_CODECS`` - and a bare
  ``OrganelleResult`` is not a JSON-safe annotation under
  ``_is_supported_json_annotation`` either, since it is not one of the
  registered ``OperationParameterModel``/``OrganelleMetadata`` types). Same
  root cause as the charter's ``LEGACY_RESULT`` gap, just without the
  ``| Mapping`` union spelling. No entry here.
* ``phylogeny.write_haplotype_network`` - **excluded**: required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim - *and* a required ``output_dir: str |
  Path`` directory parameter. No entry here.
* ``phylogeny.write_shared_genes`` - **excluded**: required
  ``result: OrganelleResult`` (the same bare-type variant as
  ``write_alignment``) *and* a required ``output_dir: str | Path`` directory
  parameter. No entry here.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
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
    "phylogeny.date_tree": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="tree_newick", codec=ParameterCodec.PATH),
            ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="calibrations", codec=ParameterCodec.JSON),
            ParameterOverride(name="backend", codec=ParameterCodec.JSON),
            ParameterOverride(name="mcmctree_options", codec=ParameterCodec.JSON),
            ParameterOverride(name="outgroup", codec=ParameterCodec.JSON),
            ParameterOverride(name="model", codec=ParameterCodec.JSON),
            ParameterOverride(name="partition_nexus", codec=ParameterCodec.PATH),
            ParameterOverride(name="ci_replicates", codec=ParameterCodec.JSON),
            ParameterOverride(name="clock_sd", codec=ParameterCodec.JSON),
            ParameterOverride(name="seed", codec=ParameterCodec.JSON),
            ParameterOverride(name="threads", codec=ParameterCodec.JSON),
            ParameterOverride(name="dry_run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    # Zero-out batch (needs-ruling rulings 2026-08-14): the network builders
    # take primitive haplotype rows and return finite-JSON tuples - bound as
    # json_metric. predict_heuristic stays internal (local-model return).
    "phylogeny.pairwise_distances": NamedParameterOverride(
        parameters=(ParameterOverride(name="haplotypes", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="pairwise_distances",
        side_effects=(),
    ),
    "phylogeny.build_msn": NamedParameterOverride(
        parameters=(ParameterOverride(name="haplotypes", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="msn_network",
        side_effects=(),
    ),
    "phylogeny.build_tcs_network": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="haplotypes", codec=ParameterCodec.JSON),
            ParameterOverride(name="members", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="tcs_network",
        side_effects=(),
    ),
    "phylogeny.build_mjn": NamedParameterOverride(
        parameters=(ParameterOverride(name="haplotypes", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="mjn_network",
        side_effects=(),
    ),
    # The renderer prepares a deferred plot; the artifact codec materializes it.
    "phylogeny.render_network": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="haplotypes", codec=ParameterCodec.JSON),
            ParameterOverride(name="members", codec=ParameterCodec.JSON),
            ParameterOverride(name="freq", codec=ParameterCodec.JSON),
            ParameterOverride(name="edges", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "phylogeny.align": NamedParameterOverride(
        parameters=(ParameterOverride(name="input_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "phylogeny.trim_alignment": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "phylogeny.build_tree": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "phylogeny.compute_alignment": NamedParameterOverride(
        parameters=(ParameterOverride(name="input_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="alignment",
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "phylogeny.plan_tree_build": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="tree_build_plan",
        side_effects=(),
    ),
    "phylogeny.collapse_haplotypes": NamedParameterOverride(
        parameters=(ParameterOverride(name="seqs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="haplotype_collapse",
        side_effects=(),
    ),
    "phylogeny.pdistance": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="a", codec=ParameterCodec.JSON),
            ParameterOverride(name="b", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="p_distance",
        side_effects=(),
    ),
    "phylogeny.tcs_connection_limit": NamedParameterOverride(
        parameters=(ParameterOverride(name="n_haplotypes", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="tcs_connection_limit",
        side_effects=(),
    ),
    "phylogeny.haplotype_network": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phylogeny.extract_shared_genes": NamedParameterOverride(
        parameters=(ParameterOverride(name="genbank_paths", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "phylogeny.run_build_tree": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "phylogeny.run_trim_alignment": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
}
"""``phylogeny.build_mjn``, ``.build_msn``, ``.build_tcs_network``,
``.pairwise_distances``, ``.render_network``, ``.write_alignment``,
``.write_haplotype_network``, and ``.write_shared_genes`` are deliberately
absent - see the module docstring for why each fails closed honestly.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]

# Discrete ancestral traits use JSON tables and explicit rooted tree artifacts.

OVERRIDES["phylogeny.reconstruct_ancestral_states"] = NamedParameterOverride(
    parameters=(
        ParameterOverride(name="tree_newick", codec=ParameterCodec.PATH),
        ParameterOverride(name="traits", codec=ParameterCodec.JSON),
        ParameterOverride(name="state_space", codec=ParameterCodec.JSON),
        ParameterOverride(name="rate", codec=ParameterCodec.JSON),
        ParameterOverride(name="rate_bounds", codec=ParameterCodec.JSON),
    ),
    result_codec=ResultCodec.CANONICAL, result_key=None, side_effects=(SideEffect.READ_FILES,),
)
OVERRIDES["phylogeny.gene_presence_traits"] = NamedParameterOverride(
    parameters=(
        ParameterOverride(name="genbank_paths", codec=ParameterCodec.PATH),
        ParameterOverride(name="genes", codec=ParameterCodec.JSON),
        ParameterOverride(name="sample_names", codec=ParameterCodec.JSON),
        ParameterOverride(name="include_pseudogenes", codec=ParameterCodec.JSON),
    ),
    result_codec=ResultCodec.CANONICAL, result_key=None, side_effects=(SideEffect.READ_FILES,),
)
FIXTURES = {
    "phylogeny.reconstruct_ancestral_states": (
        FixtureCase(case="binary", files=(FixtureFile("tree.nwk", "((A:0.1,B:0.1):0.5,C:0.2);\n"),),
                    parameters={"tree_newick": "tree.nwk", "traits": {"A": {"gene": 0}, "B": {"gene": 0}, "C": {"gene": 1}}, "rate": 1.0},
                    equivalence="numeric_tolerance", tolerance=1e-10),
    ),
    "phylogeny.gene_presence_traits": (
        FixtureCase(case="presence", files=(FixtureFile("sample.gb", '''LOCUS       sample                    30 bp    DNA              UNK 01-JAN-1980
DEFINITION  annotation presence fixture.
ACCESSION   sample
VERSION     sample
KEYWORDS    .
SOURCE      .
  ORGANISM  .
            .
FEATURES             Location/Qualifiers
     CDS             1..30
                     /gene="accD"
ORIGIN
        1 atgatgatga tgatgatgat gatgatgatg
//
'''),), parameters={"genbank_paths": ["sample.gb"], "genes": ["accD", "ycf1"]}),
    ),
}
