"""Adapter decisions for the ``visualization`` domain's restored capabilities.

**A fifth blocker class, dominant in this domain by capability count: every
one of these 21 capabilities' primary parameter is poisoned by ``Any``
somewhere inside a ``Sequence``/``Mapping`` shape, which the real runtime
binder rejects outright.** Until Capability Plan 03 Task 4,
``_looks_json_safe``'s ``Subscript`` branch checked only
``expr.value.id in {"list", "dict", "tuple"}`` - it did not recognize
``typing``/``collections.abc`` aliases like ``Sequence``/``Mapping`` at
all, so declaring ``codec = "json"`` for any such parameter failed at
*generation* time with ``capability.adapter_override_invalid`` before the
question of ``Any`` poisoning the real runtime check
(``_is_supported_json_annotation`` also explicitly rejects ``Any``) ever
arose. Task 4 closed that generator/runtime disagreement: ``_looks_json_safe``
now recognizes ``Sequence``/``Mapping`` and recurses into their type
arguments exactly as the real runtime binder does - verified directly,
``_looks_json_safe`` on ``plot_erc_distribution``'s ``pairs:
Sequence[Mapping[str, Any]]`` still returns ``False``, now because it
correctly walks into the ``Mapping[str, Any]`` value and rejects the bare
``Any``, the same reason the runtime always rejected it. The outcome for
revised: eleven of the 21 below are now RESOLVED by source signature narrowing, but the paragraph
for the true reason (``Any``) rather than an unrelated recognition gap.
This was, and remains, the single largest source of exclusions in this
domain: the overwhelming majority of ``plot_*`` functions take their
primary data as ``Sequence[Mapping[str, Any]]`` (or a ``Mapping[str, Any]``
alone) rather than a bare ``list[dict]``/``dict``, and none of them
resolve. Listed once here rather than repeated per capability below:
``ideogram``,
``nuclear_transfer_ideogram``, ``plot_erc_distribution``,
``plot_erc_group_ridges``, ``plot_erc_network``, ``plot_erc_pair_scatter``,
``plot_erc_significance``, ``plot_localization_bars``,
``plot_matrix_heatmap``, ``plot_network``, ``plot_qc_dashboard``,
``plot_rna_editing_summary``, ``plot_selection_summary``,
``plot_splicing_schematic``, ``plot_track_density``,
``plot_transfer_schematic``, ``plot_transfer_tracks``,
``plot_synteny_matrix``, ``summarize_synteny_matrix``,
``write_erc_visualization_report``, ``write_visualization_preview_report``.

**Signature narrowing (Plan 03 follow-up):** the eleven row-driven suite plots
(``plot_track_density``, ``plot_network``, ``plot_selection_summary``,
``plot_transfer_schematic``, ``plot_transfer_tracks``,
``plot_rna_editing_summary``, ``plot_erc_pair_scatter``,
``plot_erc_distribution``, ``plot_erc_group_ridges``, ``plot_erc_network``,
``plot_erc_significance``) had their row parameters narrowed in the source to
``Sequence[Mapping[str, str | int | float | bool | None]]``. That is not a
convenience cast: every helper in each render chain was inspected and reads
only string labels, numeric metrics coerced via ``_float``/``float``, and
optional ``None`` - all JSON primitives - so the narrowed type states what
the code actually consumes. The genuinely-nested shapes are NOT in this
batch: ``plot_qc_dashboard`` (a ``flags`` list inside a metrics mapping),
``plot_splicing_schematic`` (exon lists inside gene rows), and
``plot_localization_bars`` (nested probability mappings) stay excluded until
their row contracts can be expressed below; the ideogram pair stays out for
its own documented reasons. The nested five WERE subsequently expressible at
their real depth (a ``flags`` list[str] inside qc metrics; exon lists of
``{start, end}`` numerics inside splicing gene rows; a ``Mapping[str, int |
float]`` of probabilities inside localization rows; heatmap's matrix of
numerics; summarize_synteny's two JSON arms and its all-finite-JSON dict
return) and are landed. ``plot_synteny_matrix`` itself stays excluded: its
required ``output: str | Path`` is a single output FILE, and today's contract
expresses output destinations only as directories (``codec = "directory"`` +
``path_role = "output"``) while ``codec = "path"`` means staged input files -
an output-file binding needs a contract extension (Decision 004 ruling).
The same output-FILE wall excludes both figures-writers
(``write_erc_visualization_report``, ``write_visualization_preview_report``):
their ``figures`` rows are plain primitives and would bind as json, but each
takes a required ``output_html: str | Path``. The ideogram pair IS landed:
``karyotype``/``chromosomes`` narrow to
``Sequence[Mapping[str, str | int | float | None]] | Mapping[str, int |
float]`` (every ``_numt_*``/``_ideogram_*`` helper reads chromosome names as
str and lengths/positions/coords coerced via ``_float``), and
``nuclear_transfer_ideogram``'s required ``fragments`` narrows to its two
JSON arms - the row sequence and the ``{candidates: [rows]}``-style mapping
wrapper - dropping the ``OrganelleResult`` arm from the ANNOTATION only;
runtime keeps accepting results (the ``isinstance`` path), but the capability
binding is the JSON surface. Both return ``OrganelleResult`` -> canonical.

**The core-model six (Plan 03 close-out, verified against current code):**
``parse_genbank`` (``gb_file: str | Path`` binds as path, but returns the
LOCAL ``MitoGenome`` class - no ResultCodec represents a local model),
``read_gfa_graph`` (same: path in, LOCAL ``GfaGraph`` out),
``draw_mito_map`` (first param IS the local ``MitoGenome`` - not a
registered parameter model, not a path, not json),
``save_plot`` (TWO walls: ``plot: OrganelleResult | Mapping[str, Any]`` -
``OrganelleResult`` is json-unsafe as a param and ``legacy_result`` exists
in ParameterCodec but requires the external/probe bundle shape native
packages cannot carry (Decision 004 item 4, same ruling that keeps
``annotation.annotate`` on the residual catalog) - plus a required
``output: str | Path`` single output FILE, the same output-file wall as
above), and ``write_ideogram`` / ``write_nuclear_transfer_ideogram`` (the
same two walls as ``save_plot``). Unblocking any of the four param-walled
ones needs one contract extension: EITHER registering the core models as
bindable ``legacy_result`` inputs for native bundles, OR an output-file
path_role; both are Decision 004 rulings, not adapter work. The two
local-model parsers additionally need either a representable result kind
for local models or a narrow-to-core return refactor.

**The single-arm ``str | Path`` union reasoning (first used in
``morphology``/``transfer``) resolves several capabilities whose primary
parameter is a wide union, confirmed by direct inspection of both the AST
check and the real runtime type** (``get_type_hints`` +
``_is_supported_json_annotation``/``_resolve_path_parameter_plan``):
``plot_gfa_graph``'s ``gfa: str | Path | OrganelleData`` and
``plot_rscu_usage``'s ``rscu_rows: Sequence[Mapping[str, Any]] |
Mapping[str, Any] | str | Path`` (a four-way union with two ``Sequence``/
``Mapping`` arms that would each independently fail) both get ``path`` -
``_looks_like_path_annotation`` only needs *one* ``or``-chained leaf to
match ``str``/``Path``, regardless of how many other arms are unresolvable
container aliases.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``visualization.classify_gene`` - resolved below: ``name: str`` gets
  ``json``; the bare ``str`` return is finite JSON, so ``json_metric``
  under ``"gene_class"``.
* ``visualization.spread_labels`` - resolved below: ``gene_mids: list[float]``
  and ``genome_length: float`` both get ``json``; the ``(list[float],
  list[bool])`` 2-tuple return is finite JSON, so ``json_metric`` under
  ``"spread_labels"``.
* ``visualization.plot_genome_map`` / ``.plot_ogdraw_map`` - resolved
  below: ``genbank_path`` gets ``path`` (really opened to parse GenBank);
  the ``OrganellePlot`` return is accepted directly by
  ``ResultCodec.ARTIFACT`` (``codecs._encode_plot_artifact``). Both default
  to the self-contained matplotlib OGDraw port (no external tool). Every
  other parameter keeps its own default. ``side_effects = ["read_files"]``.
* ``visualization.plot_genome_identity`` - resolved below: the
  ``comparative.compute_genome_identity`` JSON rows bind as ``windows`` /
  ``features`` (each ``Sequence[Mapping[str, str | int | float | bool |
  None]]``, the same narrowed row shape as the eleven landed row plots),
  with the optional ``reference_length`` / ``region`` scalars as ``json``;
  ``artifact`` (deferred-render ``OrganellePlot``). No file I/O at prepare
  time, so ``side_effects = []``.
* ``visualization.plot_pan_circular`` - resolved below: ``reference_gbk``
  and the optional VCF/table/GFA inputs get ``path`` (really opened);
  ``canonical`` (every drawn number is in ``metrics``; ``ov.write``
  redraws a serialized Result through the writer). The centre graph is
  drawn by OrganelleVerse, no external tool; Bandage only when
  ``graph_renderer="bandage"`` is asked for. A real fixture (Citrus
  mitochondrial PGGB subset, 1-30 kb of NC_057143.1) admits it.
  ``side_effects =
  ["read_files"]``.
* ``visualization.plot_tree`` - resolved below: ``newick_path`` gets
  ``path`` (really opened); ``artifact``. ``method="auto"`` is the
  Bio.Phylo + OrganelleVerse matplotlib drawing on every machine;
  ``method="toytree"`` uses toytree when installed (no subprocess either).
  ``side_effects = ["read_files"]``.
* ``visualization.plot_gfa_graph`` - resolved below: ``gfa: str | Path |
  OrganelleData`` gets ``path`` (single-arm reasoning, see above; really
  opened to parse the graph); ``canonical`` (a real ``OrganelleResult`` -
  its own docstring: "the Bandage subprocess runs when the plot is
  materialized [...] A missing executable returns a failed result at
  compute time" - so calling this capability alone never spawns Bandage,
  regardless of whether it is installed). Every other parameter keeps its
  own default. ``side_effects = ["read_files"]``.
* ``visualization.plot_heatmap`` - resolved below: ``data: dict[str,
  dict[str, float]]`` (fully parameterized, unlike the bare-``dict``
  gap documented in ``phylogeny``) gets ``json``; ``artifact``
  (``OrganellePlot``). Pure in-memory plotting, no I/O.
* ``visualization.plot_scatter`` - resolved below: ``x: list[float]`` and
  ``y: list[float]`` both get ``json``; ``artifact``. Pure in-memory
  plotting, no I/O.
* ``visualization.plot_rscu_usage`` - resolved below: ``rscu_rows`` gets
  ``path`` (single-arm reasoning, see above - a real RSCU table file, one
  of its four union arms); ``artifact``. Every other parameter keeps its
  own default. ``side_effects = ["read_files"]``.
* ``visualization.draw_mito_map`` - **excluded**: required
  ``parsed: MitoGenome`` - a custom dataclass
  (``visualization/ogdraw.py:184``, nested ``list[Gene]``/``list[dict]``
  fields), not JSON-safe, not a core type, not path-shaped. No entry here.
* ``visualization.parse_genbank`` - **excluded**: returns ``MitoGenome`` -
  the same custom dataclass, not representable by any implemented
  ``ResultCodec``. No entry here.
* ``visualization.read_gfa_graph`` - **excluded**: returns ``GfaGraph`` -
  a custom dataclass (``visualization/gfa_graph.py:44``, nested
  ``tuple[GfaSegment, ...]``/``tuple[GfaEdge, ...]``), not representable by
  any implemented ``ResultCodec``. No entry here.
* ``visualization.plot_collinearity`` / ``.plot_gbdraw`` /
  ``.plot_gene_structure`` - **resolved** (previously excluded): all three
  take ``inputs: PathInput | Sequence[PathInput] | OrganelleGenome |
  OrganelleData`` as their required first parameter. The earlier exclusion
  was purely an alias-recognition gap: ``PathInput`` is a module-level type
  alias (``gbdraw.py``'s ``PathInput = str | Path``), and the generator's
  path check only recognized bare ``str``/``Path`` names. The generator now
  resolves module-level type aliases from the already-parsed module AST
  (still no import), so the union's ``PathInput`` arm resolves to
  ``str | Path`` and the single-arm path reasoning applies exactly as it
  already did for ``plot_gfa_graph``'s ``str | Path | OrganelleData``:
  ``inputs`` gets ``path``, result is ``artifact``, side effects read the
  fixture file. Verified through the real pipeline (the domain test asserts
  all three reach admitted).
* ``visualization.plot_synteny_matrix`` / ``.summarize_synteny_matrix`` -
  **excluded**: both take a required ``blocks: Sequence[Mapping[str, Any]]
  | Mapping[str, Any] | Any`` - unlike ``plot_rscu_usage`` above, this
  union has no ``str``/``Path`` leaf at all (its third arm is bare ``Any``,
  not a path type), so the single-arm rescue does not apply. No entry here
  for either.
* ``visualization.save_plot`` / ``.write_ideogram`` /
  ``.write_nuclear_transfer_ideogram`` - **excluded**: all three take
  ``OrganelleResult | Mapping[str, Any]`` (or, for ``save_plot``, the
  equivalent ``plot`` parameter) as a required parameter - the charter's
  ``LEGACY_RESULT`` gap verbatim. No entry here for any of the three.
* ``visualization.ideogram``, ``.nuclear_transfer_ideogram``,
  ``.plot_erc_distribution``, ``.plot_erc_group_ridges``,
  ``.plot_erc_network``, ``.plot_erc_pair_scatter``,
  ``.plot_erc_significance``, ``.plot_localization_bars``,
  ``.plot_matrix_heatmap``, ``.plot_network``, ``.plot_qc_dashboard``,
  ``.plot_rna_editing_summary``, ``.plot_selection_summary``,
  ``.plot_splicing_schematic``, ``.plot_track_density``,
  ``.plot_transfer_schematic``, ``.plot_transfer_tracks``,
  ``.write_erc_visualization_report``, ``.write_visualization_preview_report``
  - **excluded**: the ``Sequence``/``Mapping``-alias gap described at the
  top of this module docstring. No entry here for any of these nineteen.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``draw_mito_map`` is INTERNAL (MitoGenome local-model parameter;
 the published plot_genome_map/plot_ogdraw_map cover the surface).
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

from . import FixtureCase, FixtureFile


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
    "visualization.plot_assembly_graph": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gfa", codec=ParameterCodec.PATH),
            ParameterOverride(name="reads", codec=ParameterCodec.PATH),
            ParameterOverride(name="read_type", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="threads", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "visualization.plot_structure_map": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="genbank_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="genome_name", codec=ParameterCodec.JSON),
            ParameterOverride(name="show_ir", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
            ParameterOverride(name="ssrs", codec=ParameterCodec.JSON),
            ParameterOverride(name="tandem_repeats", codec=ParameterCodec.JSON),
            ParameterOverride(name="dispersed_repeats", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_transfer_circos": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="transfers", codec=ParameterCodec.JSON),
            ParameterOverride(name="recipient", codec=ParameterCodec.PATH),
            ParameterOverride(name="donor", codec=ParameterCodec.PATH),
            ParameterOverride(name="transfer_kind", codec=ParameterCodec.JSON),
            ParameterOverride(name="color_by", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="show_genes", codec=ParameterCodec.JSON),
            ParameterOverride(name="gene_labels", codec=ParameterCodec.JSON),
            ParameterOverride(name="minus_labels", codec=ParameterCodec.JSON),
            ParameterOverride(name="gene_label_min_fraction", codec=ParameterCodec.JSON),
            ParameterOverride(name="show_ir", codec=ParameterCodec.JSON),
            ParameterOverride(name="order", codec=ParameterCodec.JSON),
            ParameterOverride(name="zoom", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.classify_gene": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="gene_class",
        side_effects=(),
    ),
    "visualization.spread_labels": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_mids", codec=ParameterCodec.JSON),
            ParameterOverride(name="genome_length", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="spread_labels",
        side_effects=(),
    ),
    "visualization.plot_genome_map": NamedParameterOverride(
        parameters=(ParameterOverride(name="genbank_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_genome_identity": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="windows", codec=ParameterCodec.JSON),
            ParameterOverride(name="features", codec=ParameterCodec.JSON),
            ParameterOverride(name="reference_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="region", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_ogdraw_map": NamedParameterOverride(
        parameters=(ParameterOverride(name="genbank_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_pan_circular": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="reference_gbk", codec=ParameterCodec.PATH),
            ParameterOverride(name="variants", codec=ParameterCodec.PATH),
            ParameterOverride(name="variant_matrix", codec=ParameterCodec.PATH),
            ParameterOverride(name="diversity", codec=ParameterCodec.PATH),
            ParameterOverride(name="sv_variants", codec=ParameterCodec.PATH),
            ParameterOverride(name="pan_segments", codec=ParameterCodec.PATH),
            ParameterOverride(name="pangenome_gfa", codec=ParameterCodec.PATH),
            ParameterOverride(name="metadata", codec=ParameterCodec.JSON),
            ParameterOverride(name="sample_labels", codec=ParameterCodec.JSON),
            ParameterOverride(name="bins", codec=ParameterCodec.JSON),
            ParameterOverride(name="window_size", codec=ParameterCodec.JSON),
            ParameterOverride(name="graph_renderer", codec=ParameterCodec.JSON),
            ParameterOverride(name="graph_max_blocks", codec=ParameterCodec.JSON),
            ParameterOverride(name="gc", codec=ParameterCodec.JSON),
            ParameterOverride(name="show_ir", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_tree": NamedParameterOverride(
        parameters=(ParameterOverride(name="newick_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_gfa_graph": NamedParameterOverride(
        parameters=(ParameterOverride(name="gfa", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_gbdraw": NamedParameterOverride(
        parameters=(ParameterOverride(name="inputs", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_collinearity": NamedParameterOverride(
        parameters=(ParameterOverride(name="inputs", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_gene_structure": NamedParameterOverride(
        parameters=(ParameterOverride(name="inputs", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_heatmap": NamedParameterOverride(
        parameters=(ParameterOverride(name="data", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    # The eleven row-driven suite plots below were previously excluded because
    # their row parameters were annotated Sequence[Mapping[str, Any]] - the
    # Any-poisoning blocker. Their signatures are now narrowed to
    # Sequence[Mapping[str, str | int | float | bool | None]], which matches
    # what every helper in their render chains actually reads (str() labels,
    # _float() metrics, optional None) - verified per cluster, and the
    # genuinely-nested row shapes (qc_dashboard's flags list, splicing's exon
    # lists, localization's nested probability mappings) are NOT in this batch.
    "visualization.plot_track_density": NamedParameterOverride(
        parameters=(ParameterOverride(name="tracks", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_network": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="nodes", codec=ParameterCodec.JSON),
            ParameterOverride(name="edges", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_selection_summary": NamedParameterOverride(
        parameters=(ParameterOverride(name="records", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_transfer_schematic": NamedParameterOverride(
        parameters=(ParameterOverride(name="events", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_transfer_tracks": NamedParameterOverride(
        parameters=(ParameterOverride(name="candidates", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_rna_editing_summary": NamedParameterOverride(
        parameters=(ParameterOverride(name="editing_sites", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_erc_pair_scatter": NamedParameterOverride(
        parameters=(ParameterOverride(name="points", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_erc_distribution": NamedParameterOverride(
        parameters=(ParameterOverride(name="pairs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_erc_group_ridges": NamedParameterOverride(
        parameters=(ParameterOverride(name="pairs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_erc_network": NamedParameterOverride(
        parameters=(ParameterOverride(name="pairs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_erc_significance": NamedParameterOverride(
        parameters=(ParameterOverride(name="pairs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    # The nested-row batch (Plan 03 follow-up 2): five capabilities whose row
    # shapes are genuinely nested, narrowed at their real depth after per-helper
    # inspection - see the module docstring's narrowing note for the method.
    "visualization.plot_qc_dashboard": NamedParameterOverride(
        parameters=(ParameterOverride(name="metrics", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_matrix_heatmap": NamedParameterOverride(
        parameters=(ParameterOverride(name="matrix", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_splicing_schematic": NamedParameterOverride(
        parameters=(ParameterOverride(name="genes", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_localization_bars": NamedParameterOverride(
        parameters=(ParameterOverride(name="predictions", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    # The output-file batch (Ruling 1, Decision 004 approval 2026-08-14):
    # codec=path with path_role="output" names a single destination file
    # validated under the same containment rules as directory outputs
    # (no traversal, no symlink component, leaf is a regular file or absent,
    # never the managed run store); write_files side effect is mandatory.
    # The legacy_result batch (Ruling 2, Decision 004 approval 2026-08-14):
    # these three take the L1 result itself plus an output file (Ruling 1).
    # The agent-facing field IS OrganelleResult, so Pydantic reconstructs the
    # frozen result before the callable runs; the destination is validated by
    # the file-output containment rules.
    "visualization.save_plot": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="plot", codec=ParameterCodec.LEGACY_RESULT),
            ParameterOverride(
                name="output", codec=ParameterCodec.PATH, path_role="output"
            ),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.write_ideogram": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="result", codec=ParameterCodec.LEGACY_RESULT),
            ParameterOverride(
                name="output", codec=ParameterCodec.PATH, path_role="output"
            ),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.write_nuclear_transfer_ideogram": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="result", codec=ParameterCodec.LEGACY_RESULT),
            ParameterOverride(
                name="output", codec=ParameterCodec.PATH, path_role="output"
            ),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.plot_synteny_matrix": NamedParameterOverride(
        parameters=(ParameterOverride(name="blocks", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.write_erc_visualization_report": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="figures", codec=ParameterCodec.JSON),
            ParameterOverride(
                name="output_html", codec=ParameterCodec.PATH, path_role="output"
            ),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.write_visualization_preview_report": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="figures", codec=ParameterCodec.JSON),
            ParameterOverride(
                name="output_html", codec=ParameterCodec.PATH, path_role="output"
            ),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
    "visualization.ideogram": NamedParameterOverride(
        parameters=(ParameterOverride(name="karyotype", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "visualization.nuclear_transfer_ideogram": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="chromosomes", codec=ParameterCodec.JSON),
            ParameterOverride(name="fragments", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "visualization.summarize_synteny_matrix": NamedParameterOverride(
        parameters=(ParameterOverride(name="blocks", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="synteny_summary",
        side_effects=(),
    ),
    "visualization.plot_scatter": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="x", codec=ParameterCodec.JSON),
            ParameterOverride(name="y", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(),
    ),
    "visualization.plot_rscu_usage": NamedParameterOverride(
        parameters=(ParameterOverride(name="rscu_rows", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_fst_scan": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="vcf", codec=ParameterCodec.PATH),
            ParameterOverride(name="groups", codec=ParameterCodec.JSON),
            ParameterOverride(name="pair", codec=ParameterCodec.JSON),
            ParameterOverride(name="estimator", codec=ParameterCodec.JSON),
            ParameterOverride(name="window_size", codec=ParameterCodec.JSON),
            ParameterOverride(name="step", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_snps", codec=ParameterCodec.JSON),
            ParameterOverride(name="contig", codec=ParameterCodec.JSON),
            ParameterOverride(name="genome_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="reference_sample", codec=ParameterCodec.JSON),
            ParameterOverride(name="highlight", codec=ParameterCodec.JSON),
            ParameterOverride(name="top_fraction", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_colors", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="height", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_pi_ratio": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="groups", codec=ParameterCodec.JSON),
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="vcf", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference", codec=ParameterCodec.JSON),
            ParameterOverride(name="pair", codec=ParameterCodec.JSON),
            ParameterOverride(name="window_size", codec=ParameterCodec.JSON),
            ParameterOverride(name="step", codec=ParameterCodec.JSON),
            ParameterOverride(name="gap_mode", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_sites", codec=ParameterCodec.JSON),
            ParameterOverride(name="pseudocount", codec=ParameterCodec.JSON),
            ParameterOverride(name="contig", codec=ParameterCodec.JSON),
            ParameterOverride(name="reference_sample", codec=ParameterCodec.JSON),
            ParameterOverride(name="genbank", codec=ParameterCodec.PATH),
            ParameterOverride(name="top_fraction", codec=ParameterCodec.JSON),
            ParameterOverride(name="label_top", codec=ParameterCodec.JSON),
            ParameterOverride(name="log_scale", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_colors", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="height", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_sv_schematic": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gfa", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference", codec=ParameterCodec.JSON),
            ParameterOverride(name="start", codec=ParameterCodec.JSON),
            ParameterOverride(name="end", codec=ParameterCodec.JSON),
            ParameterOverride(name="samples", codec=ParameterCodec.JSON),
            ParameterOverride(name="labels", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_sv_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="homology_fraction", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_span", codec=ParameterCodec.JSON),
            ParameterOverride(name="max_insert", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="row_height", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_sv_frequency": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="svs", codec=ParameterCodec.JSON),
            ParameterOverride(name="groups", codec=ParameterCodec.JSON),
            ParameterOverride(name="gfa", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference", codec=ParameterCodec.JSON),
            ParameterOverride(name="units", codec=ParameterCodec.JSON),
            ParameterOverride(name="collapse_identical", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_order", codec=ParameterCodec.JSON),
            ParameterOverride(name="test", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="max_svs", codec=ParameterCodec.JSON),
            ParameterOverride(name="sort", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_colors", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="height", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "visualization.plot_presence_tracks": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gfa", codec=ParameterCodec.PATH),
            ParameterOverride(name="paf", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference", codec=ParameterCodec.JSON),
            ParameterOverride(name="start", codec=ParameterCodec.JSON),
            ParameterOverride(name="end", codec=ParameterCodec.JSON),
            ParameterOverride(name="groups", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_order", codec=ParameterCodec.JSON),
            ParameterOverride(name="annotations", codec=ParameterCodec.JSON),
            ParameterOverride(name="annotation_colors", codec=ParameterCodec.JSON),
            ParameterOverride(name="sample_labels", codec=ParameterCodec.JSON),
            ParameterOverride(name="svs", codec=ParameterCodec.JSON),
            ParameterOverride(name="min_sv_length", codec=ParameterCodec.JSON),
            ParameterOverride(name="genbank", codec=ParameterCodec.PATH),
            ParameterOverride(name="min_block", codec=ParameterCodec.JSON),
            ParameterOverride(name="group_colors", codec=ParameterCodec.JSON),
            ParameterOverride(name="title", codec=ParameterCodec.JSON),
            ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
            ParameterOverride(name="width", codec=ParameterCodec.JSON),
            ParameterOverride(name="row_height", codec=ParameterCodec.JSON),
            ParameterOverride(name="dpi", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""Every restored id not listed above is deliberately absent - see the
module docstring for the three exclusion families (custom-dataclass
params/returns, unresolvable ``PathInput``-alias unions, the
``Sequence``/``Mapping``-alias gap, and the charter's ``LEGACY_RESULT``
gap) and which specific capability falls into each.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]

#: Toy population set of tests/visualization/popgen_fixture.py (popgen figure bundles).
_POPGEN_VCF = FixtureFile(
    relative_path="toy.vcf",
    content="""##fileformat=VCFv4.2
##contig=<ID=REF,length=2000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	W1	W2	W3	W4	C1	C2	C3	C4
REF	51	.	C	T	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	101	.	A	G	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	151	.	G	A	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	201	.	A	G	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	251	.	C	T	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	301	.	T	C	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	351	.	T	C	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	401	.	T	C	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	451	.	T	C	60	PASS	.	GT	0	0	0	0	1	1	1	1
REF	1001	.	A	G	60	PASS	.	GT	1	1	0	0	0	0	0	0
REF	1051	.	C	T	60	PASS	.	GT	0	1	1	0	0	0	0	0
REF	1101	.	C	T	60	PASS	.	GT	0	0	1	1	0	0	0	0
REF	1151	.	T	C	60	PASS	.	GT	1	0	0	1	0	0	0	0
REF	1201	.	T	C	60	PASS	.	GT	1	1	0	0	0	0	0	0
REF	1251	.	A	G	60	PASS	.	GT	0	1	1	0	0	0	0	0
REF	1301	.	A	G	60	PASS	.	GT	0	0	1	1	0	0	0	0
REF	1351	.	G	A	60	PASS	.	GT	1	0	0	1	0	0	0	0
REF	1401	.	C	T	60	PASS	.	GT	1	1	0	0	0	0	0	0
REF	1451	.	G	A	60	PASS	.	GT	0	1	1	0	0	0	0	0
REF	1601	.	C	T	60	PASS	.	GT	1	0	0	0	1	0	0	0
REF	1700	.	AAGC	A	60	PASS	.	GT	0	0	0	0	0	1	0	0
REF	1701	.	A	G	60	PASS	.	GT	0	1	0	0	0	1	0	0
REF	1801	.	A	G	60	PASS	.	GT	0	0	1	0	0	0	1	0
REF	1901	.	A	G	60	PASS	.	GT	0	0	0	1	0	0	0	1
""",
)

_POPGEN_REF_FASTA = FixtureFile(
    relative_path="ref.fasta",
    content=""">REF
TGTGGAAGTAAGCCTAACCCCGTGACGTGCGCGACTATGGGGGTCCAACACTCGGGGGCTGTGAAGTACGGCTATGCATTAGCGTGATCTGGAATTTTGCATGGACGGGACATGAGGGAGTTCTGCTCGATTGTTGGCATTCCTACACGAGCGCAAGCAAGCCGTCAGGTTGTAGCTTGTGTCGTACTTCCAGGTTGGTTATGCATTACGACTACTTAGACACAGTACGACCTGGCTTGTAGGAAGTACGCCGCACAACATTGCTTCATAGAAGGGTTGAAAGCCGGTACACTCTGTTGCTGTAATCTCACTCCGTTTGTCAAGGGCATTAATGAGGGGCTAAGGACAGATCCGGATACGCGGGAGTAAATAAGATCCGGGAGTAAAGCACTCCTAGGTCTATCACGGAATTCACAAACGGGCGATATATCCAAGAGTGGTTTATATTTATAGCTTTTGCGTCGAACGAACCGCGCAAAGATCTACAACCTTTATCCATTCCACAGTAACACGGCAACTAGCCAGGATTGACGTTTAATTAGCCGTCGGTCTTGCGTCAGGGGTAAGCGGGTTGAAGATTTCGTCTGTTAATAAAGTATCGCAACCACCCATCATCTTCTGAGACCTAGTAAGCAGAACAACCAGTGAGTAGAATACATAAAGAGAGCGAATTACTGTTCTCCGATTACCCCCTCTGAACAAGGGGATTCACATGCGATCAGTGACGTCGGACGGGGCGTGAGCTCAGGTATAATGGCCGTATTACGACACAACTCGCCCGTATTGCAAAACTTCGACGTACAAGTACTATCTCAGGGACCCGGGCCGTGGGCAAAGGGGCCGGTAATATATTGAATGAGACGAACAATTGGGCGTGTGATGTTTGGCGGGGTCTACAAATCAGGTATGTAATGCGACATATGTCGTATTACCTTACGGTCTCTCTGTTATTGGCTATTCTCACCAGAGGTTCGGCTCAGAAGTCTTGGCGGGCACGTTAAACGTTCAGCAAGGCGCCGGAACCCACTCTGACACCATCGGCAGCACCCTCGAGGGGATCTAGTACATAAGGATGAATCAAGGACAGGCCAATATCTGAACTACAAGCGCTTCAATCTGGCATCACTTCAGTATATTATGACACGACCGATGTAAATATCTGTGAGTATATGTCAGAGAATTTTAGTACTTGAGAGCGCCTTGAGTCCGTGCCTCGGACGCGATTTGATACTTTGGTCGACAAAAGTGCTACTCGATTTGCGTCAAACACCCAACGCACAGGGATGTGTGATTCGAATTAACACGGATGCCGTTGCAACTCATGAGTATTCCCAAAGCGGACTAGGGAAGGAGAGAAGCAGACCACCACGCATCGCTTGGAGAGTTGTTTTTCATTGCTTCCCGGTACATTGGAGCATAACGCCAAATCTTGACAGCAACACGGCCATGAGACATAGGTCAAGAAGCTGGGATACTCCGGTACCTGAGAGGTCTCGAGAAACAGCATTACGGCGGCACGGGAGGGGGGTGAGGCAATCAAAAGCCGCCTGGTAACATTTTTTCTATGTTTGAACATACCAGTAACGGTCTGGACCCCTAGCCCGACTGACATCTGGAACGGTATCCGGATATCAGTCGTAGTACACGTTTTGAGACTGTCAATTAGTTACTGTGTTGCTTGATGCGTTTTATGGTACCTAAGCTATACCAGAGTGTATAAGCCGAAAGATTTTGAACGTGAAGCTTTTATTCTTGAATAATCGACATGGATAGTCAATACGATGCACAGAGGAACTGAGTAGATTCACAAGCAACTCGCGCGTAACCTGAAAAGCTGTTTTTATTCATGCGCCTTGCTCATGACTTTCCGGTGCACCGGCGCCCATGGTGCGAGTAATGTACGCCAGTGACGTATGGACGGCGCGACGACTCGCCCTGGCCGCCAAGTAACGGATCTACAGGGCGACTCATCTTGAACCTAAACGTGTATTCGTGGACAC
""",
)

_POPGEN_GENBANK = FixtureFile(
    relative_path="ref.gb",
    content="""LOCUS       REF                     2000 bp    DNA     circular PLN 11-OCT-2026
DEFINITION  Toy plant mitochondrion, complete genome.
ACCESSION   REF
VERSION     REF
SOURCE      mitochondrion Toy plant
  ORGANISM  Toy plant
FEATURES             Location/Qualifiers
     source          1..2000
                     /organism="Toy plant"
                     /organelle="mitochondrion"
     gene            101..400
                     /gene="geneA"
     CDS             101..400
                     /gene="geneA"
     gene            complement(1001..1300)
                     /gene="geneB"
     CDS             complement(1001..1300)
                     /gene="geneB"
     gene            1501..1572
                     /gene="trnX"
     tRNA            1501..1572
                     /gene="trnX"
ORIGIN
        1 tgtggaagta agcctaaccc cgtgacgtgc gcgactatgg gggtccaaca ctcgggggct
       61 gtgaagtacg gctatgcatt agcgtgatct ggaattttgc atggacggga catgagggag
      121 ttctgctcga ttgttggcat tcctacacga gcgcaagcaa gccgtcaggt tgtagcttgt
      181 gtcgtacttc caggttggtt atgcattacg actacttaga cacagtacga cctggcttgt
      241 aggaagtacg ccgcacaaca ttgcttcata gaagggttga aagccggtac actctgttgc
      301 tgtaatctca ctccgtttgt caagggcatt aatgaggggc taaggacaga tccggatacg
      361 cgggagtaaa taagatccgg gagtaaagca ctcctaggtc tatcacggaa ttcacaaacg
      421 ggcgatatat ccaagagtgg tttatattta tagcttttgc gtcgaacgaa ccgcgcaaag
      481 atctacaacc tttatccatt ccacagtaac acggcaacta gccaggattg acgtttaatt
      541 agccgtcggt cttgcgtcag gggtaagcgg gttgaagatt tcgtctgtta ataaagtatc
      601 gcaaccaccc atcatcttct gagacctagt aagcagaaca accagtgagt agaatacata
      661 aagagagcga attactgttc tccgattacc ccctctgaac aaggggattc acatgcgatc
      721 agtgacgtcg gacggggcgt gagctcaggt ataatggccg tattacgaca caactcgccc
      781 gtattgcaaa acttcgacgt acaagtacta tctcagggac ccgggccgtg ggcaaagggg
      841 ccggtaatat attgaatgag acgaacaatt gggcgtgtga tgtttggcgg ggtctacaaa
      901 tcaggtatgt aatgcgacat atgtcgtatt accttacggt ctctctgtta ttggctattc
      961 tcaccagagg ttcggctcag aagtcttggc gggcacgtta aacgttcagc aaggcgccgg
     1021 aacccactct gacaccatcg gcagcaccct cgaggggatc tagtacataa ggatgaatca
     1081 aggacaggcc aatatctgaa ctacaagcgc ttcaatctgg catcacttca gtatattatg
     1141 acacgaccga tgtaaatatc tgtgagtata tgtcagagaa ttttagtact tgagagcgcc
     1201 ttgagtccgt gcctcggacg cgatttgata ctttggtcga caaaagtgct actcgatttg
     1261 cgtcaaacac ccaacgcaca gggatgtgtg attcgaatta acacggatgc cgttgcaact
     1321 catgagtatt cccaaagcgg actagggaag gagagaagca gaccaccacg catcgcttgg
     1381 agagttgttt ttcattgctt cccggtacat tggagcataa cgccaaatct tgacagcaac
     1441 acggccatga gacataggtc aagaagctgg gatactccgg tacctgagag gtctcgagaa
     1501 acagcattac ggcggcacgg gaggggggtg aggcaatcaa aagccgcctg gtaacatttt
     1561 ttctatgttt gaacatacca gtaacggtct ggacccctag cccgactgac atctggaacg
     1621 gtatccggat atcagtcgta gtacacgttt tgagactgtc aattagttac tgtgttgctt
     1681 gatgcgtttt atggtaccta agctatacca gagtgtataa gccgaaagat tttgaacgtg
     1741 aagcttttat tcttgaataa tcgacatgga tagtcaatac gatgcacaga ggaactgagt
     1801 agattcacaa gcaactcgcg cgtaacctga aaagctgttt ttattcatgc gccttgctca
     1861 tgactttccg gtgcaccggc gcccatggtg cgagtaatgt acgccagtga cgtatggacg
     1921 gcgcgacgac tcgccctggc cgccaagtaa cggatctaca gggcgactca tcttgaacct
     1981 aaacgtgtat tcgtggacac
//
""",
)

_POPGEN_GFA = FixtureFile(
    relative_path="toy.gfa",
    content="""H	VN:Z:1.0
S	n1	TGTGGAAGTAAGCCTAACCCCGTGACGTGCGCGACTATGGGGGTCCAACACTCGGGGGCTGTGAAGTACGGCTATGCATTAGCGTGATCTGGAATTTTGCATGGACGGGACATGAGGGAGTTCTGCTCGATTGTTGGCATTCCTACACGAGCGCAAGCAAGCCGTCAGGTTGTAGCTTGTGTCGTACTTCCAGGTTGGTTATGCATTACGACTACTTAGACACAGTACGACCTGGCTTGTAGGAAGTACGCCGCACAACATTGCTTCATAGAAGGGTTGAAAGCCGGTACACTCTGTTGCTGTAATCTCACTCCGTTTGTCAAGGGCATTAATGAGGGGCTAAGGACAGATCCGGATACGCGGGAGTAAATAAGATCCGGGAGTAAAGCACTCCTAGGTCTATCACGGAATTCACAAACGGGCGATATATCCAAGAGTGGTTTATATTTATAGCTTTTGCGTCGAACGAACCGCGCAAAGATCTACAACCTTTATCCATTCCACAGTAACACGGCAACTAGCCAGGATTGACGTTTAATTAGCCGTCGGTCTTGCGTCAGGGGTAAGCGGGTTGAAGATTTCGTCTGTTAATAAAGTATC
S	n2	GCAACCACCCATCATCTTCTGAGACCTAGTAAGCAGAACAACCAGTGAGTAGAATACATAAAGAGAGCGAATTACTGTTCTCCGATTACCCCCTCTGAACAAGGGGATTCACATGCGATCAGTGACGTCGGACGGGGCGTGAGCTCAGGTATAATGGCCGTATTACGACACAACTCGCCCGTATTGCAAAACTTCGACGTACAAGTACTATCTCAGGGACCCGGGCCGTGGGCAAAGGGGCCGGTAATATATTGAATGAGACGAACAATTGGGCGTGTGATGTTTGGCGGGGTCTACAAA
S	n3	TCAGGTATGTAATGCGACATATGTCGTATTACCTTACGGTCTCTCTGTTATTGGCTATTCTCACCAGAGGTTCGGCTCAGAAGTCTTGGCGGGCACGTTAAACGTTCAGCAAGGCGCCGGAACCCACTCTGACACCATCGGCAGCACCCTCGAGGGGATCTAGTACATAAGGATGAATCAAGGACAGGCCAATATCTGAACTACAAGCGCTTCAATCTGGCATCACTTCAGTATATTATGACACGACCGATGTAAATATCTGTGAGTATATGTCAGAGAATTTTAGTACTTGAGAGCGCCTTGAGTCCGTGCCTCGGACGCGATTTGATACTTTGGTCGACAAAAGTGCTACTCGATTTGCGTCAAACACCCAACGCACAGGGATGTGTGATTCGAATTAACACGGATGCCGTTGCAACTCATGAGTATTCCCAAAGCGGACTAGGGAAGGAGAGAAGCAGACCACCACGCATCGCTTGGAGAGTTGTTTTTCATTGCTT
S	n5	CCCGGTACATTGGAGCATAACGCCAAATCTTGACAGCAACACGGCCATGAGACATAGGTCAAGAAGCTGGGATACTCCGGTACCTGAGAGGTCTCGAGAAACAGCATTACGGCGGCACGGGAGGGGGGTGAGGCAATCAAAAGCCGCCTGGTAACATTTTTTCTATGTTTGAACATACCAGTAACGGTCTGGACCCCTAGCCCGACTGACATCTGGAACGGTATCCGGATATCAGTCGTAGTACACGTTTTGAGACTGTCAATTAGTTACTGTGTTGCTTGATGCGTTTTATGGTACCTAAGCTATACCAGAGTGTATAAGCCGAAAGATTTTGAACGTGAAGCTTTTATTCTTGAATAATCGACATGGATAGTCAATACGATGCACAGAGGAACTGAGTAGATTCACAAGCAACTCGCGCGTAACCTGAAAAGCTGTTTTTATTCATGCGCCTTGCTCATGACTTTCCGGTGCACCGGCGCCCATGGTGCGAGTAATGTACGCCAGTGACGTATGGACGGCGCGACGACTCGCCCTGGCCGCCAAGTAACGGATCTACAGGGCGACTCATCTTGAACCTAAACGTGTATTCGTGGACAC
S	n4	GCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCGCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGACTGGCATTTTTATTACACTCAGAAACAGAACTCGGGTAATTTTGACAGGTCACGCAGAGGCGCGCCCTCCTGAAGTGCGTG
L	n1	+	n2	+	0M
L	n1	+	n3	+	0M
L	n2	+	n3	+	0M
L	n3	+	n4	+	0M
L	n3	+	n5	+	0M
L	n4	+	n5	+	0M
P	REF#1#1	n1+,n2+,n3+,n5+	*
P	W1#1#1	n1+,n2+,n3+,n5+	*
P	W2#1#1	n1+,n2+,n3+,n5+	*
P	W3#1#1	n1+,n2+,n3+,n5+	*
P	W4#1#1	n1+,n3+,n5+	*
P	C1#1#1	n1+,n3+,n4+,n5+	*
P	C2#1#1	n1+,n3+,n4+,n5+	*
P	C3#1#1	n1+,n3+,n4+,n5+	*
P	C4#1#1	n1+,n2+,n3+,n5+	*
""",
)


_STRUCTURE_GENBANK = FixtureFile(
    relative_path="structure.gb",
    content="""LOCUS       fixture                 1000 bp    DNA     circular UNK 01-JAN-1980
DEFINITION  structure map verification.
ACCESSION   fixture
VERSION     fixture.1
KEYWORDS    .
SOURCE      .
  ORGANISM  .
            .
FEATURES             Location/Qualifiers
     source          1..1000
                     /organelle="plastid:chloroplast"
     gene            101..300
                     /gene="ycf3"
     CDS             join(101..150,201..220,251..300)
                     /gene="ycf3"
     gene            join(complement(401..440),801..870)
                     /gene="rps12"
                     /trans_splicing=""
     CDS             join(complement(401..440),801..870)
                     /gene="rps12"
     tRNA            complement(join(551..580,601..630))
                     /gene="trnK"
ORIGIN
        1 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
       61 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      121 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      181 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      241 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      301 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      361 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      421 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      481 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      541 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      601 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      661 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      721 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      781 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      841 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      901 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
      961 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
//
""",
)

_CIRCOS_MITO_GENBANK = FixtureFile(
    relative_path="mito.gb",
    content="""LOCUS       MTFIX                   3800 bp    DNA     circular UNK 01-JAN-1980
DEFINITION  Fixture plant mitochondrion, complete genome.
ACCESSION   MTFIX
VERSION     MTFIX.1
KEYWORDS    .
SOURCE      .
  ORGANISM  Fixture plant
            .
FEATURES             Location/Qualifiers
     source          1..3800
                     /organelle="mitochondrion"
                     /organism="Fixture plant"
     gene            join(101..300,3301..3500)
                     /gene="nad1"
     CDS             join(101..300,3301..3500)
                     /gene="nad1"
     gene            1401..1472
                     /gene="trnN-GUU"
     tRNA            1401..1472
                     /gene="trnN-GUU"
ORIGIN
        1 tattgttaag ttacagtaag actagcatga attcgggcct gccggcatgc aagttacagg
       61 tggcgcattt agttctgaac tccactgtgc agaggaaggt agagctaaaa tcgcgctgta
      121 gaggtctcta attttgtaac caccgggaat atatcgaaag ttcttctcta accattatat
      181 tacctgagga cttcgaagtc gtcttgcatg atttttacgc ttcgcagtat gtgatctgct
      241 atactaggtg gtcacgaggt gcttgtcaat ttaggtaaag cgctgcgagt tcgcccaaaa
      301 cgataaggcg ggctgatggc cgcgttccct ggcgctgact aaaagagtta atacgacgat
      361 gcagcgacgg gaaggtcgca catcgtcttg gttcgaggta atgcgtgtat ccaacgtgag
      421 gaaactatta catctctgaa ccacggcacg cccagaccac tggcgaaagt gtcttacggc
      481 aagcctgatg taatttagaa cgcgaccgca gtaagagcgt atgtacacat tctccctagg
      541 ttgcacatga agaatgagcc ggtatgcagg tctattggct gaagtccctc ttcacaaagc
      601 aatgcttcag ggcgaggagc acgaacgggc ctcaatgtat cttaaaacac tttttaatat
      661 atatccgtca ttgcctattt ccattgcagc tgccgttgaa agggtcccat ctctaaacct
      721 tcttcgagac gcaactcaac gaacgcctat cacacttcta tatgaacgat tggcctgaag
      781 gggcactgga atggctgcgt tacatgcgtc gtagcgcgct gaaaaggtaa tctctttggt
      841 cgtccccatt ccgagaactg gtgaaatcaa cacgcagagg tcaggtgttc attgtcgacg
      901 gagattgttt tgaaatactc tacctgggtc aactccccaa ccgtcagagc taaagttcac
      961 ttggtcatct cgataccgcc gcgcgtctaa accctttgcg accccattcg tgaggtggcg
     1021 tagtgacgta cagtcaagtc gtggtacgtc aataaacttt ggattggcga cgacaactcg
     1081 gggatatcga cttacacgat ctcggagtat tacaggctgc ttagatacct actcttctca
     1141 gctcaatcga cggttatgtg ccatgaatcg aagcgagcat gccagatcca cctgtagatt
     1201 gatagaggac gccatgtagc ataagggtta tatctgtcta agtggtggat agttagaagg
     1261 cacataagat catattagtg tcgtaatcta cgctagtagc tgattaaatt cgcattatcg
     1321 acgttttcga cccttgggac acacacaaga tgtcgggccg cccaatgaaa tatatcgtga
     1381 atttccttac atcccctcac gtctacccac gtttcggttc cgttctgcag ggccaataga
     1441 cgagcgatat tattggtgcc tctcgcagtc tggatagatg attgtggaaa gggggcttgg
     1501 gcgagagaat tattacggaa gttcacttag gatggaagta atgagcgcga gtggtggatg
     1561 gcgtagccac attctggatt aagaccgttg cggaatacca catttatgaa tagctgctgg
     1621 ggatgccaaa tatcagtggc acacactttg ggctatagac ccgccgctac tagcacgaag
     1681 agactccagg actagtactg atctctccat gcagtaaatt ccatcaccta gttaacgcag
     1741 cgtcttactc tcggcatttt cggtgcggac agtattcatt taatctacaa tacaaatcga
     1801 acgtacagca cgtctccata atcaggcccg ggcgcgcaga gaaccaacct gcgacccgat
     1861 gctccacgat cgaccgatga gatttcacgc acaccttcgt cgaggcgggt tcgctgctta
     1921 aagcttggaa tttctggcac ccccgatact atcggtgata tgcggactgg tctcctctgg
     1981 ttccgggttt ggtttttctc ccagaaagac tatacgaatg ttcaactggt atttcccttg
     2041 caacacgtac agagcttccg aaaaaaacgt gctctctcaa caccggagtt gattgatgtg
     2101 agtcgatgct gtacgttgat tggttagcat ccacggatca tatcactacc cacgtttttt
     2161 gcacaagcct gtccgacgtg tatatttggc gtctggagtc aagacaggca tctggctgat
     2221 ttacgagtag tcccggtcta gtcgcatatt cggggccttc aacgtgtcgg gccctagggc
     2281 tcatgtttct aaggtgatat ataacgcctt cgggggcaag taactgcctg agacatactc
     2341 gtgggaatca tcatgtcgct acttaagatt ggcgggttag aatgaattag tctttcacct
     2401 ctaggggcgc cccaaaggta aacgaaccgt tgcggtcaat cttgtcgcgg ctgatgaatt
     2461 tgaagcagtg gccgggagtg tgtgctcagg agttcgtccc atgacacgat agagagagaa
     2521 catcctgttg ggcttaatga tatagaattc cctcgcttgg atgagccata tagaccgcct
     2581 ctcgtcgtgt tgatctacct gacatgtctc tcgcgcgacc acccaggatt agactcatca
     2641 ttcgggtagt agacattata ttcgataccg tggtagccta gggtgttaac acccctataa
     2701 gttttatcgc ataatgatcg ctatctacct cctgtccgaa cgttcatgag aaacgcacag
     2761 aattacgatc ttacgactct gcatagaatt atttcgtcgt tgagtcctcg ggagacagta
     2821 gtcagttaca attagccctg gtgctggctg ggaggcccat tgggacatgg atgtctagta
     2881 gagaaaatcg agaactccat ttgataaaat tccctcgcga taatgatctt cagagctctg
     2941 tattcctgaa tctatcctcg ccaccacgcg gctctagagt acgctatttg cgactaattg
     3001 ctcttggagc cgcttagagt taagtattgg ccagcgtagc ctttgatgat cgtgtacact
     3061 ctccaaagca tgggccaggg gacggggcaa ttcaaggaaa gctaacctac gacagaaagc
     3121 tgcaaacgcc cctcacagat cagctaaatc aaagtttggc cgacacgttt ctcgttgatc
     3181 gagagacgta ccgccacaca gtcaaaagct gaggcactga cgagtgccac ggacatatgc
     3241 caaaacgagg ttaatccgga tattcaggat tctgttgagc gcctgtttgg gcacgccaag
     3301 ggtaatttga tcctagtcgt atatacgaca acggactcta agtcctgact ggatgagagc
     3361 gacgcttatg ccaaatggta tggagacgga acacgctcgc gcgaagatga tgtgggcgat
     3421 atctcaaaat aagtacaaaa cccacacttg agaattaact gtttcatata aaaggcccaa
     3481 gttatagcac ccgccgctct aattattcag aaagagttat tgatcacaca gattataccg
     3541 ttaatttgtg ttatctcagc tttgctcctc gagtgtgccg ctgtattatt tgacgctttg
     3601 aactgctgca tcttagaagt tgcttaggcg atatgcatgg cgtgctggtt tggtttaaag
     3661 tacggcgtga ctttacaaac ctggcagctt tggataataa cgttccgggc gtctgacgaa
     3721 acgctacttg caggcgtcga ttacgacata catgttccga catcctatag gttgtatcat
     3781 gctcagtacc agtgttatcg
//
""",
)

_CIRCOS_PLASTID_GENBANK = FixtureFile(
    relative_path="plastid.gb",
    content="""LOCUS       CPFIX                   3000 bp    DNA     circular UNK 01-JAN-1980
DEFINITION  Fixture plant chloroplast, complete genome.
ACCESSION   CPFIX
VERSION     CPFIX.1
KEYWORDS    .
SOURCE      .
  ORGANISM  Fixture plant
            .
FEATURES             Location/Qualifiers
     source          1..3000
                     /organelle="plastid:chloroplast"
                     /organism="Fixture plant"
     gene            complement(101..400)
                     /gene="psbA"
     CDS             complement(101..400)
                     /gene="psbA"
     gene            1301..1372
                     /gene="trnN-GUU"
     tRNA            1301..1372
                     /gene="trnN-GUU"
     gene            complement(2829..2900)
                     /gene="trnN-GUU"
     tRNA            complement(2829..2900)
                     /gene="trnN-GUU"
     gene            601..900
                     /gene="rbcL"
     CDS             601..900
                     /gene="rbcL"
ORIGIN
        1 tttcctcatg caattcaaaa ccatgtccgt aatgtaggcg aaatagtaaa ccattttacg
       61 gaggatacca aattcctcct tattcaggac ctaacctgag gtaaaccagg tctctccgcc
      121 cccttataaa agctgttgca cctagccaag ttcaacggca gctgcaatgg aaataggcaa
      181 tgacggatat atattaaaaa gtgttttaag atacattgag gcccgttcgt gctcctcgcc
      241 ctgaagcatt gctttgtgaa gagggacttc agccaataga cctgcatacc ggctcattct
      301 tcatgtgcaa cctagggaga atgtgtacat acgctcttac tgcggtcgcg tctaataata
      361 tacatttgct tcgttgacta gcaacccagg gctatagcta ttccccccgc ggcccaccca
      421 gtattcctaa cggagcataa atcccacccg aactaagttt gtcgaacctt ggtccaagat
      481 cgggactcgg tctccaggta agacgggctc attcataaac gttactaagg ggtataatct
      541 tctatttgtg ggtgggaaca cttagtagac ttgcaatcca attacagcag tcttgtgcgc
      601 ctaggggcgc cccaaaggta aacgaaccgt tgcggtcaat cttgtcgcgg ctgatgaatt
      661 tgaagcagtg gccgggagtg tgtgctcagg agttcgtccc atgacacgat agagagagaa
      721 catcctgttg ggcttaatga tatagaattc cctcgcttgg atgagccata tagaccgcct
      781 ctcgtcgtgt tgatctacct gacatgtctc tcgcgcgacc acccaggatt agactcatca
      841 ttcgggtagt agacattata ttcgataccg tggtagccta gggtgttaac acccctataa
      901 cacattagtc ccttgtatgc aggcggtatc ggacggcgcc cacaccttgg aggtatccag
      961 cgcaaggcgc catatccgta ccttactatc gcgcgaactt atgttgtttt aagttagagt
     1021 tggacatcta tacgtcagtc ctaaacatag cgagcatttc gcagatgggt ctccgacggt
     1081 accccaaggg tcgttaccga cgccgggacg ccgcatataa aggtacgccc gaccattata
     1141 caggtagcca tctgcgtctg acatcgcatt tgaaacccag taggtactgc cttagttgca
     1201 ctcctaactc atgttaacgg acttacgggc actagcttct tactgccctc tctgtttctc
     1261 ttaagggacg tcgagacgcc aagttatgga gtctacccac gtttcggttc cgttctgcag
     1321 ggccaataga cgagcgatat tattggtgcc tctcgcagtc tggatagatg attgtggaaa
     1381 gggggcttgg acaattagat tttacggtgt accgcgccat actagggaag ctccccgtgg
     1441 tggtccggcc aaagattact taggttgggg cgcctcgccc tgccatcggt gttcacaacg
     1501 gatgatcgag tgcttctcgc tcagttacga gcgtggcatc ggacaagaac gtccttatgt
     1561 acggcgctac acaaggagat acagagcttg atttgaaccg tgggtgggag aggcccacgc
     1621 cgaccggcta atatagcacg aagttcttcg atgcgactac gttaattttt ctaattgaag
     1681 ctgggcttac tacccaagga cagggtcatc tgcaattcat aacgcagagc gatctattaa
     1741 cgcttagggc cccctacgag gggcaacggt ccagtgtgtc aagtctagag atcttctcta
     1801 gtggtggaca tgcgttggaa atcagagaga ctagctgtac attcaaattc ctgctaaacg
     1861 tattcaggaa gtaagaacca gggccttact catcacccta taccatcgat atgattgacg
     1921 atgtccatgg gcgatttgtg taagactgtc agaggtctag taagcgggca gctagaacgg
     1981 tgtagaatcg gagccggata tacgacattg acatctttat gaagaatgac atgcacgtta
     2041 ttctttttac gcagcgtttt gcttgatcgg tagagtccta cttttaccag cagctgtctg
     2101 gaccccgacc cgggaggacg acggggcgta gaggctccac ggatgcttgg cggcaaagaa
     2161 acgggcaaca tcatcagtca tctcataacg ggcgcctatg cacaaaggat accaagactc
     2221 tggcgtacga gggtctcccc gttcgccgga cgcaggcaca actcatcgga atctcgctga
     2281 taatatatcc acctcggccc gacccctgga gcacgaaggc agtgaacaag ccgagttgtt
     2341 acctattagc actcaactta tacgacgagg gtggcgcttt ggtcctgcgc tcggaagtat
     2401 tagagaagat ctctagactt gacacactgg accgttgccc ctcgtagggg gccctaagcg
     2461 ttaatagatc gctctgcgtt atgaattgca gatgaccctg tccttgggta gtaagcccag
     2521 cttcaattag aaaaattaac gtagtcgcat cgaagaactt cgtgctatat tagccggtcg
     2581 gcgtgggcct ctcccaccca cggttcaaat caagctctgt atctccttgt gtagcgccgt
     2641 acataaggac gttcttgtcc gatgccacgc tcgtaactga gcgagaagca ctcgatcatc
     2701 cgttgtgaac accgatggca gggcgaggcg ccccaaccta agtaatcttt ggccggacca
     2761 ccacggggag cttccctagt atggcgcggt acaccgtaaa atctaattgt ccaagccccc
     2821 tttccacaat catctatcca gactgcgaga ggcaccaata atatcgctcg tctattggcc
     2881 ctgcagaacg gaaccgaaac gtgggtagac tccataactt ggcgtctcga cgtcccttaa
     2941 gagaaacaga gagggcagta agaagctagt gcccgtaagt ccgttaacat gagttaggag
//
""",
)

#: detect_mtpt fragments (LOSAT) for the two files above; the IR trnN copy has an HSP per IR arm.
_CIRCOS_FRAGMENTS = [
    {
        "seqid": "MTFIX.1",
        "start": 495,
        "end": 700,
        "length": 206,
        "identity": 99.52,
        "strand": "-",
        "hsps": [
            {
                "source_seqid": "CPFIX.1",
                "source_start": 151,
                "source_end": 355,
                "start": 495,
                "end": 700,
                "identity": 99.52,
                "strand": "-",
                "alignment_length": 206
            }
        ]
    },
    {
        "seqid": "MTFIX.1",
        "start": 1401,
        "end": 1500,
        "length": 100,
        "identity": 100.0,
        "strand": "+",
        "hsps": [
            {
                "source_seqid": "CPFIX.1",
                "source_start": 1291,
                "source_end": 1390,
                "start": 1401,
                "end": 1500,
                "identity": 100.0,
                "strand": "+",
                "alignment_length": 100
            },
            {
                "source_seqid": "CPFIX.1",
                "source_start": 2811,
                "source_end": 2910,
                "start": 1401,
                "end": 1500,
                "identity": 100.0,
                "strand": "-",
                "alignment_length": 100
            }
        ]
    },
    {
        "seqid": "MTFIX.1",
        "start": 2401,
        "end": 2700,
        "length": 300,
        "identity": 100.0,
        "strand": "+",
        "hsps": [
            {
                "source_seqid": "CPFIX.1",
                "source_start": 601,
                "source_end": 900,
                "start": 2401,
                "end": 2700,
                "identity": 100.0,
                "strand": "+",
                "alignment_length": 300
            }
        ]
    }
]

_ASSEMBLY_GFA = FixtureFile(
    relative_path="graph.gfa",
    content="H\tVN:Z:1.0\nS\tA\t*\tLN:i:40000\tdp:f:40.0\nS\tB\t*\tLN:i:25000\tdp:f:42.0\nS\tR\t*\tLN:i:3000\tdp:f:81.0\nL\tA\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tB\t+\tR\t+\t0M\nL\tR\t+\tA\t+\t0M\nP\tmt\tA+,R+,B+,R+\t*\n",
)

# Real subset of the Citrus mitochondrial pangenome (13 genomes, PGGB + vg deconstruct
# + vcfbub/vcfwave): NC_057143.1 features in 1-30000 without sequence, the VCF records
# there for four genomes, and the graph nodes the reference path visits there with
# each genome's sub-paths (sequences as '*' with LN:i).
_PAN_CIRCULAR_GENBANK = FixtureFile(
    relative_path='reference.gb',
    content=(
        'LOCUS       NC_057143.1            30000 bp    DNA     circular UNK 01-JAN-1980\n'
        'DEFINITION  Citrus maxima mitochondrion, OrganelleVerse fixture 1-30000.\n'
        'ACCESSION   NC_057143\n'
        'VERSION     NC_057143.1\n'
        'KEYWORDS    .\n'
        'SOURCE      .\n'
        '  ORGANISM  Citrus maxima\n'
        '            .\n'
        'FEATURES             Location/Qualifiers\n'
        '     gene            1026..2549\n'
        '                     /gene="atp1"\n'
        '                     /locus_tag="K4Z88_mgp01"\n'
        '     CDS             1026..2549\n'
        '                     /gene="atp1"\n'
        '                     /locus_tag="K4Z88_mgp01"\n'
        '     gene            19429..21854\n'
        '                     /gene="ccmFC"\n'
        '                     /locus_tag="K4Z88_mgp02"\n'
        '     CDS             join(19429..20199,21156..21854)\n'
        '                     /gene="ccmFC"\n'
        '                     /locus_tag="K4Z88_mgp02"\n'
        '     gene            27505..27578\n'
        '                     /gene="trnG-GCC"\n'
        '                     /locus_tag="K4Z88_mgt01"\n'
        '     tRNA            27505..27578\n'
        '                     /gene="trnG-GCC"\n'
        '                     /locus_tag="K4Z88_mgt01"\n'
        'ORIGIN\n'
        '//\n'
    ),
)

_PAN_CIRCULAR_VCF = FixtureFile(
    relative_path='variants.vcf',
    content=(
        '##fileformat=VCFv4.2\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNC_057142.1\tPQ636878.1\tNC_037463.1'
        '\tOZ535733.1\n'
        'NC_057143.1#1#1\t658\t>1505>1508\tC\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t725\t>1508>1511\tA\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t782\t>1511>1514\tT\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t789\t>1514>1518\tG\tT,A\t60\t.\tLV=0\tGT\t0\t2\t1\t1\n'
        'NC_057143.1#1#1\t800\t>1518>1521\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t937\t>1522>1524\tTCCTTGG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t955\t>1524>1527\tC\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t956\t>1527>1529\tC\tCTTTT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t958\t>1529>1532\tC\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t960\t>1532>1535_1\tAC\tTA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t968\t>1535>1538\tC\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t1031\t>1538>1541\tG\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t1046\t>1541>1544\tA\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t1074\t>1544>1547\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t2366\t>1583>1586_1\tCA\tAC\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t2599\t>1588>1591\tG\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t2630\t>1591>1594\tC\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t2828\t>1596>1599\tG\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t3002\t>1606>1609\tG\tC\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t3070\t>1611>1614\tC\tA\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t3296\t>1616>1619\tT\tC\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t3802\t>1621>1623\tGA\tG\t60\t.\tLV=0\tGT\t0\t.\t1\t0\n'
        'NC_057143.1#1#1\t4093\t>1623>1626\tT\tC\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t4934\t>1636>1639\tG\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t5161\t>1639>1642\tA\tC\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t5446\t>1642>1645\tA\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t5453\t>1645>1648\tA\tT\t60\t.\tLV=0\tGT\t0\t.\t1\t1\n'
        'NC_057143.1#1#1\t5672\t>1648>6_1\tCTTTT\tC\t60\t.\tLV=0\tGT\t0\t.\t1\t0\n'
        'NC_057143.1#1#1\t5674\t>1648>6_2\tTTT\tT\t60\t.\tLV=0\tGT\t0\t.\t0\t1\n'
        'NC_057143.1#1#1\t5851\t>9>12\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t6019\t>17>19\tGAACTAAAAT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t6039\t>19>22\tT\tA\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t6271\t>22>25\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t6347\t>25>30_3\tG\tGGG\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t6435\t>32>35\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t6725\t>37>40\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t6824\t>40>43\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t7320\t>69>72\tT\tC\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t7397\t>72>74\tGCGACCTCT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t7440\t>74>77\tA\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t7786\t>79>81\tC\tCCCTTT\t60\t.\tLV=0\tGT\t0\t0\t0\t1\n'
        'NC_057143.1#1#1\t7942\t>81>83\tT\tTG\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t8218\t>87>89\tTA\tT\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t8319\t>92>95_1\tGAAAGCTGGA\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t8496\t>95>98\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t8803\t>102>106_2\tT\tA\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t9704\t>115>117\tC\tCTTATGATA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t10176\t>119>122\tT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t10691\t>126>129\tT\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t11325\t>136>138\tC\tCCTTCTTCTT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t12049\t>140>143\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t12695\t>143>145\tGT\tG\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t14301\t>154>157\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t14720\t>157>160\tC\tT\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t14738\t>160>162\tG\tGTAGATAACA\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t15016\t>164>167\tC\tA\t60\t.\tLV=0\tGT\t0\t1\t0\t1\n'
        'NC_057143.1#1#1\t15243\t>167>171_3\tAA\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t15374\t>173>176\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t15627\t>176>178\tG\tGCAATCAGTT\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t15851\t>180>183\tT\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t15910\t>183>186\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t16020\t>188>190\tGAACATGGCACT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t16595\t>199>202\tT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t16714\t>204>206\tGGAAAAGGTT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t17165\t>208>210\tTA\tT\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t17173\t>210>212\tGA\tG\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t17180\t>212>214\tGT\tG\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t17453\t>218>221_1\tA\tATTTT\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t17473\t>221>224_1\tG\tGATGAG\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t17893\t>228>230\tT\tTA\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t17899\t>230>234_2\tAAAA\tA\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t17900\t>230>234_3\tAAA\tA\t60\t.\tLV=0\tGT\t0\t1\t0\t1\n'
        'NC_057143.1#1#1\t18130\t>234>238_1\tA\tAACAGCCCCCCCCTGTTGTTGATAGAGAGCTT\t60\t.\tLV=0\tGT'
        '\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t18518\t>238>241\tA\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t18527\t>241>244\tG\tT\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t21702\t>273>276\tC\tT\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t22368\t>281>284\tA\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t23274\t>289>291\tG\tGCGTAAAGT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t24128\t>293>296\tG\tA\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t24980\t>301>304\tT\tG\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t24982\t>304>307\tC\tA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t25281\t>309>312\tG\tT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t26408\t>325>327\tGTAGTCTTCC\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t26916\t>330>333_2\tCC\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t26922\t>333>335\tC\tCAATAAA\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t27380\t>340>342\tA\tAC\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t27975\t>342>345\tT\tG\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t28293\t>348>353_3\tCC\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t0\n'
        'NC_057143.1#1#1\t28626\t>355>358\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t28639\t>358>360\tCCTCTCATCT\tC\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t28674\t>362>365\tT\tG\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
        'NC_057143.1#1#1\t28735\t>365>367\tG\tGATTCT\t60\t.\tLV=0\tGT\t0\t1\t1\t1\n'
        'NC_057143.1#1#1\t29204\t>369>372\tA\tC\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t29313\t>372>374\tCGAAAGGAA\tC\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t29388\t>374>380_1\tTCCCTATCCC\tT\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t29396\t>374>380_2\tCC\tC\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t29397\t>374>380_3\tC\tCCC\t60\t.\tLV=0\tGT\t0\t0\t0\t1\n'
        'NC_057143.1#1#1\t29403\t>380>387_2\tC\tCCC\t60\t.\tLV=0\tGT\t0\t0\t1\t0\n'
        'NC_057143.1#1#1\t29403\t>380>387_3\tC\tCCT\t60\t.\tLV=0\tGT\t0\t1\t0\t1\n'
        'NC_057143.1#1#1\t29425\t>387>389\tGCGGGAGAAT\tG\t60\t.\tLV=0\tGT\t0\t1\t0\t0\n'
        'NC_057143.1#1#1\t29570\t>391>394\tG\tT\t60\t.\tLV=0\tGT\t0\t0\t1\t1\n'
    ),
)

_PAN_CIRCULAR_GFA = FixtureFile(
    relative_path='graph.gfa',
    content=(
        'H\tVN:Z:1.0\n'
        'S\t2\t*\tLN:i:288\n'
        'S\t5\t*\tLN:i:2\n'
        'S\t6\t*\tLN:i:3\n'
        'S\t8\t*\tLN:i:1\n'
        'S\t9\t*\tLN:i:170\n'
        'S\t10\t*\tLN:i:1\n'
        'S\t11\t*\tLN:i:1\n'
        'S\t12\t*\tLN:i:6\n'
        'S\t14\t*\tLN:i:2\n'
        'S\t15\t*\tLN:i:18\n'
        'S\t16\t*\tLN:i:1\n'
        'S\t17\t*\tLN:i:141\n'
        'S\t18\t*\tLN:i:9\n'
        'S\t19\t*\tLN:i:10\n'
        'S\t20\t*\tLN:i:1\n'
        'S\t21\t*\tLN:i:1\n'
        'S\t22\t*\tLN:i:231\n'
        'S\t23\t*\tLN:i:1\n'
        'S\t24\t*\tLN:i:1\n'
        'S\t25\t*\tLN:i:72\n'
        'S\t26\t*\tLN:i:1\n'
        'S\t27\t*\tLN:i:1\n'
        'S\t28\t*\tLN:i:2\n'
        'S\t29\t*\tLN:i:2\n'
        'S\t30\t*\tLN:i:42\n'
        'S\t31\t*\tLN:i:1\n'
        'S\t32\t*\tLN:i:44\n'
        'S\t33\t*\tLN:i:1\n'
        'S\t34\t*\tLN:i:1\n'
        'S\t35\t*\tLN:i:191\n'
        'S\t36\t*\tLN:i:1\n'
        'S\t37\t*\tLN:i:97\n'
        'S\t38\t*\tLN:i:1\n'
        'S\t39\t*\tLN:i:1\n'
        'S\t40\t*\tLN:i:98\n'
        'S\t41\t*\tLN:i:1\n'
        'S\t42\t*\tLN:i:1\n'
        'S\t43\t*\tLN:i:72\n'
        'S\t44\t*\tLN:i:84\n'
        'S\t45\t*\tLN:i:1\n'
        'S\t46\t*\tLN:i:1\n'
        'S\t47\t*\tLN:i:31\n'
        'S\t48\t*\tLN:i:1\n'
        'S\t49\t*\tLN:i:1\n'
        'S\t50\t*\tLN:i:44\n'
        'S\t51\t*\tLN:i:1\n'
        'S\t52\t*\tLN:i:1\n'
        'S\t53\t*\tLN:i:33\n'
        'S\t54\t*\tLN:i:1\n'
        'S\t55\t*\tLN:i:1\n'
        'S\t56\t*\tLN:i:10\n'
        'S\t57\t*\tLN:i:1\n'
        'S\t58\t*\tLN:i:1\n'
        'S\t59\t*\tLN:i:32\n'
        'S\t60\t*\tLN:i:1\n'
        'S\t61\t*\tLN:i:1\n'
        'S\t62\t*\tLN:i:11\n'
        'S\t63\t*\tLN:i:1\n'
        'S\t65\t*\tLN:i:1\n'
        'S\t66\t*\tLN:i:40\n'
        'S\t67\t*\tLN:i:1\n'
        'S\t68\t*\tLN:i:1\n'
        'S\t69\t*\tLN:i:129\n'
        'S\t70\t*\tLN:i:1\n'
        'S\t71\t*\tLN:i:1\n'
        'S\t72\t*\tLN:i:77\n'
        'S\t73\t*\tLN:i:8\n'
        'S\t74\t*\tLN:i:34\n'
        'S\t75\t*\tLN:i:1\n'
        'S\t76\t*\tLN:i:1\n'
        'S\t77\t*\tLN:i:8\n'
        'S\t78\t*\tLN:i:1\n'
        'S\t79\t*\tLN:i:337\n'
        'S\t80\t*\tLN:i:5\n'
        'S\t81\t*\tLN:i:156\n'
        'S\t82\t*\tLN:i:1\n'
        'S\t83\t*\tLN:i:7\n'
        'S\t84\t*\tLN:i:1\n'
        'S\t85\t*\tLN:i:36\n'
        'S\t86\t*\tLN:i:1\n'
        'S\t87\t*\tLN:i:231\n'
        'S\t88\t*\tLN:i:1\n'
        'S\t89\t*\tLN:i:30\n'
        'S\t90\t*\tLN:i:1\n'
        'S\t91\t*\tLN:i:1\n'
        'S\t92\t*\tLN:i:68\n'
        'S\t93\t*\tLN:i:9\n'
        'S\t94\t*\tLN:i:1\n'
        'S\t95\t*\tLN:i:166\n'
        'S\t96\t*\tLN:i:1\n'
        'S\t97\t*\tLN:i:1\n'
        'S\t98\t*\tLN:i:145\n'
        'S\t99\t*\tLN:i:1\n'
        'S\t100\t*\tLN:i:56\n'
        'S\t101\t*\tLN:i:1\n'
        'S\t102\t*\tLN:i:103\n'
        'S\t103\t*\tLN:i:1\n'
        'S\t104\t*\tLN:i:1\n'
        'S\t105\t*\tLN:i:3\n'
        'S\t106\t*\tLN:i:417\n'
        'S\t108\t*\tLN:i:1\n'
        'S\t109\t*\tLN:i:132\n'
        'S\t110\t*\tLN:i:1\n'
        'S\t111\t*\tLN:i:74\n'
        'S\t112\t*\tLN:i:1\n'
        'S\t113\t*\tLN:i:45\n'
        'S\t114\t*\tLN:i:1\n'
        'S\t115\t*\tLN:i:226\n'
        'S\t116\t*\tLN:i:8\n'
        'S\t117\t*\tLN:i:142\n'
        'S\t118\t*\tLN:i:1\n'
        'S\t119\t*\tLN:i:328\n'
        'S\t120\t*\tLN:i:1\n'
        'S\t121\t*\tLN:i:1\n'
        'S\t122\t*\tLN:i:139\n'
        'S\t123\t*\tLN:i:1\n'
        'S\t124\t*\tLN:i:109\n'
        'S\t125\t*\tLN:i:1\n'
        'S\t126\t*\tLN:i:264\n'
        'S\t127\t*\tLN:i:1\n'
        'S\t128\t*\tLN:i:1\n'
        'S\t129\t*\tLN:i:194\n'
        'S\t131\t*\tLN:i:1\n'
        'S\t132\t*\tLN:i:162\n'
        'S\t133\t*\tLN:i:1\n'
        'S\t134\t*\tLN:i:249\n'
        'S\t135\t*\tLN:i:1\n'
        'S\t136\t*\tLN:i:26\n'
        'S\t137\t*\tLN:i:9\n'
        'S\t138\t*\tLN:i:389\n'
        'S\t139\t*\tLN:i:1\n'
        'S\t140\t*\tLN:i:333\n'
        'S\t141\t*\tLN:i:1\n'
        'S\t142\t*\tLN:i:1\n'
        'S\t143\t*\tLN:i:646\n'
        'S\t144\t*\tLN:i:1\n'
        'S\t145\t*\tLN:i:245\n'
        'S\t146\t*\tLN:i:1\n'
        'S\t147\t*\tLN:i:253\n'
        'S\t148\t*\tLN:i:1\n'
        'S\t149\t*\tLN:i:406\n'
        'S\t151\t*\tLN:i:1\n'
        'S\t152\t*\tLN:i:691\n'
        'S\t153\t*\tLN:i:1\n'
        'S\t154\t*\tLN:i:5\n'
        'S\t155\t*\tLN:i:1\n'
        'S\t156\t*\tLN:i:1\n'
        'S\t157\t*\tLN:i:418\n'
        'S\t158\t*\tLN:i:1\n'
        'S\t159\t*\tLN:i:1\n'
        'S\t160\t*\tLN:i:18\n'
        'S\t161\t*\tLN:i:9\n'
        'S\t162\t*\tLN:i:125\n'
        'S\t163\t*\tLN:i:1\n'
        'S\t164\t*\tLN:i:151\n'
        'S\t165\t*\tLN:i:1\n'
        'S\t166\t*\tLN:i:1\n'
        'S\t167\t*\tLN:i:225\n'
        'S\t168\t*\tLN:i:1\n'
        'S\t169\t*\tLN:i:1\n'
        'S\t170\t*\tLN:i:1\n'
        'S\t171\t*\tLN:i:86\n'
        'S\t172\t*\tLN:i:1\n'
        'S\t173\t*\tLN:i:42\n'
        'S\t174\t*\tLN:i:1\n'
        'S\t175\t*\tLN:i:1\n'
        'S\t176\t*\tLN:i:253\n'
        'S\t177\t*\tLN:i:9\n'
        'S\t178\t*\tLN:i:216\n'
        'S\t179\t*\tLN:i:1\n'
        'S\t180\t*\tLN:i:6\n'
        'S\t181\t*\tLN:i:1\n'
        'S\t182\t*\tLN:i:1\n'
        'S\t183\t*\tLN:i:58\n'
        'S\t184\t*\tLN:i:1\n'
        'S\t185\t*\tLN:i:1\n'
        'S\t186\t*\tLN:i:6\n'
        'S\t187\t*\tLN:i:1\n'
        'S\t188\t*\tLN:i:103\n'
        'S\t189\t*\tLN:i:11\n'
        'S\t190\t*\tLN:i:123\n'
        'S\t191\t*\tLN:i:1\n'
        'S\t193\t*\tLN:i:21\n'
        'S\t194\t*\tLN:i:1\n'
        'S\t195\t*\tLN:i:368\n'
        'S\t196\t*\tLN:i:1\n'
        'S\t197\t*\tLN:i:37\n'
        'S\t198\t*\tLN:i:7\n'
        'S\t199\t*\tLN:i:4\n'
        'S\t200\t*\tLN:i:1\n'
        'S\t201\t*\tLN:i:1\n'
        'S\t202\t*\tLN:i:5\n'
        'S\t203\t*\tLN:i:1\n'
        'S\t204\t*\tLN:i:113\n'
        'S\t205\t*\tLN:i:9\n'
        'S\t206\t*\tLN:i:439\n'
        'S\t208\t*\tLN:i:3\n'
        'S\t209\t*\tLN:i:1\n'
        'S\t210\t*\tLN:i:7\n'
        'S\t211\t*\tLN:i:1\n'
        'S\t212\t*\tLN:i:6\n'
        'S\t213\t*\tLN:i:1\n'
        'S\t214\t*\tLN:i:21\n'
        'S\t215\t*\tLN:i:1\n'
        'S\t216\t*\tLN:i:219\n'
        'S\t217\t*\tLN:i:1\n'
        'S\t218\t*\tLN:i:30\n'
        'S\t219\t*\tLN:i:1\n'
        'S\t220\t*\tLN:i:3\n'
        'S\t221\t*\tLN:i:20\n'
        'S\t222\t*\tLN:i:4\n'
        'S\t223\t*\tLN:i:1\n'
        'S\t224\t*\tLN:i:43\n'
        'S\t225\t*\tLN:i:1\n'
        'S\t226\t*\tLN:i:94\n'
        'S\t227\t*\tLN:i:1\n'
        'S\t228\t*\tLN:i:281\n'
        'S\t229\t*\tLN:i:1\n'
        'S\t230\t*\tLN:i:5\n'
        'S\t231\t*\tLN:i:2\n'
        'S\t232\t*\tLN:i:1\n'
        'S\t233\t*\tLN:i:1\n'
        'S\t234\t*\tLN:i:228\n'
        'S\t235\t*\tLN:i:4\n'
        'S\t236\t*\tLN:i:2\n'
        'S\t237\t*\tLN:i:25\n'
        'S\t238\t*\tLN:i:387\n'
        'S\t239\t*\tLN:i:1\n'
        'S\t240\t*\tLN:i:1\n'
        'S\t241\t*\tLN:i:8\n'
        'S\t242\t*\tLN:i:1\n'
        'S\t243\t*\tLN:i:1\n'
        'S\t244\t*\tLN:i:32\n'
        'S\t245\t*\tLN:i:1\n'
        'S\t246\t*\tLN:i:1\n'
        'S\t247\t*\tLN:i:245\n'
        'S\t248\t*\tLN:i:1\n'
        'S\t249\t*\tLN:i:639\n'
        'S\t250\t*\tLN:i:1\n'
        'S\t251\t*\tLN:i:5\n'
        'S\t252\t*\tLN:i:1\n'
        'S\t253\t*\tLN:i:187\n'
        'S\t254\t*\tLN:i:1\n'
        'S\t255\t*\tLN:i:374\n'
        'S\t256\t*\tLN:i:1\n'
        'S\t257\t*\tLN:i:641\n'
        'S\t258\t*\tLN:i:1\n'
        'S\t259\t*\tLN:i:300\n'
        'S\t260\t*\tLN:i:1\n'
        'S\t261\t*\tLN:i:6\n'
        'S\t263\t*\tLN:i:1\n'
        'S\t264\t*\tLN:i:87\n'
        'S\t265\t*\tLN:i:1\n'
        'S\t266\t*\tLN:i:414\n'
        'S\t267\t*\tLN:i:1\n'
        'S\t268\t*\tLN:i:140\n'
        'S\t269\t*\tLN:i:1\n'
        'S\t270\t*\tLN:i:6\n'
        'S\t271\t*\tLN:i:1\n'
        'S\t273\t*\tLN:i:84\n'
        'S\t274\t*\tLN:i:1\n'
        'S\t275\t*\tLN:i:1\n'
        'S\t276\t*\tLN:i:82\n'
        'S\t277\t*\tLN:i:1\n'
        'S\t278\t*\tLN:i:520\n'
        'S\t279\t*\tLN:i:1\n'
        'S\t280\t*\tLN:i:1\n'
        'S\t281\t*\tLN:i:60\n'
        'S\t282\t*\tLN:i:1\n'
        'S\t283\t*\tLN:i:1\n'
        'S\t284\t*\tLN:i:305\n'
        'S\t285\t*\tLN:i:1\n'
        'S\t286\t*\tLN:i:159\n'
        'S\t288\t*\tLN:i:1\n'
        'S\t289\t*\tLN:i:440\n'
        'S\t290\t*\tLN:i:8\n'
        'S\t291\t*\tLN:i:472\n'
        'S\t292\t*\tLN:i:2\n'
        'S\t293\t*\tLN:i:379\n'
        'S\t294\t*\tLN:i:1\n'
        'S\t295\t*\tLN:i:1\n'
        'S\t296\t*\tLN:i:32\n'
        'S\t297\t*\tLN:i:1\n'
        'S\t298\t*\tLN:i:359\n'
        'S\t300\t*\tLN:i:1\n'
        'S\t301\t*\tLN:i:458\n'
        'S\t302\t*\tLN:i:1\n'
        'S\t303\t*\tLN:i:1\n'
        'S\t304\t*\tLN:i:1\n'
        'S\t305\t*\tLN:i:1\n'
        'S\t306\t*\tLN:i:1\n'
        'S\t307\t*\tLN:i:245\n'
        'S\t308\t*\tLN:i:2\n'
        'S\t309\t*\tLN:i:51\n'
        'S\t310\t*\tLN:i:1\n'
        'S\t311\t*\tLN:i:1\n'
        'S\t312\t*\tLN:i:365\n'
        'S\t313\t*\tLN:i:1\n'
        'S\t315\t*\tLN:i:180\n'
        'S\t316\t*\tLN:i:1\n'
        'S\t317\t*\tLN:i:192\n'
        'S\t318\t*\tLN:i:1\n'
        'S\t319\t*\tLN:i:114\n'
        'S\t320\t*\tLN:i:1\n'
        'S\t321\t*\tLN:i:178\n'
        'S\t322\t*\tLN:i:1\n'
        'S\t323\t*\tLN:i:16\n'
        'S\t324\t*\tLN:i:1\n'
        'S\t325\t*\tLN:i:76\n'
        'S\t326\t*\tLN:i:9\n'
        'S\t327\t*\tLN:i:221\n'
        'S\t328\t*\tLN:i:1\n'
        'S\t329\t*\tLN:i:1\n'
        'S\t330\t*\tLN:i:275\n'
        'S\t331\t*\tLN:i:1\n'
        'S\t332\t*\tLN:i:1\n'
        'S\t333\t*\tLN:i:5\n'
        'S\t334\t*\tLN:i:6\n'
        'S\t335\t*\tLN:i:2\n'
        'S\t336\t*\tLN:i:1\n'
        'S\t337\t*\tLN:i:445\n'
        'S\t338\t*\tLN:i:2\n'
        'S\t339\t*\tLN:i:1\n'
        'S\t340\t*\tLN:i:7\n'
        'S\t341\t*\tLN:i:1\n'
        'S\t342\t*\tLN:i:594\n'
        'S\t343\t*\tLN:i:1\n'
        'S\t344\t*\tLN:i:1\n'
        'S\t345\t*\tLN:i:26\n'
        'S\t347\t*\tLN:i:1\n'
        'S\t348\t*\tLN:i:287\n'
        'S\t350\t*\tLN:i:1\n'
        'S\t351\t*\tLN:i:3\n'
        'S\t352\t*\tLN:i:1\n'
        'S\t353\t*\tLN:i:168\n'
        'S\t354\t*\tLN:i:1\n'
        'S\t355\t*\tLN:i:162\n'
        'S\t356\t*\tLN:i:1\n'
        'S\t357\t*\tLN:i:1\n'
        'S\t358\t*\tLN:i:13\n'
        'S\t359\t*\tLN:i:9\n'
        'S\t360\t*\tLN:i:9\n'
        'S\t361\t*\tLN:i:1\n'
        'S\t362\t*\tLN:i:15\n'
        'S\t363\t*\tLN:i:1\n'
        'S\t364\t*\tLN:i:1\n'
        'S\t365\t*\tLN:i:61\n'
        'S\t366\t*\tLN:i:5\n'
        'S\t367\t*\tLN:i:6\n'
        'S\t368\t*\tLN:i:1\n'
        'S\t369\t*\tLN:i:461\n'
        'S\t370\t*\tLN:i:1\n'
        'S\t371\t*\tLN:i:1\n'
        'S\t372\t*\tLN:i:109\n'
        'S\t373\t*\tLN:i:8\n'
        'S\t374\t*\tLN:i:67\n'
        'S\t375\t*\tLN:i:6\n'
        'S\t376\t*\tLN:i:1\n'
        'S\t377\t*\tLN:i:1\n'
        'S\t378\t*\tLN:i:1\n'
        'S\t379\t*\tLN:i:2\n'
        'S\t380\t*\tLN:i:4\n'
        'S\t381\t*\tLN:i:2\n'
        'S\t383\t*\tLN:i:1\n'
        'S\t384\t*\tLN:i:1\n'
        'S\t385\t*\tLN:i:1\n'
        'S\t386\t*\tLN:i:1\n'
        'S\t387\t*\tLN:i:21\n'
        'S\t388\t*\tLN:i:9\n'
        'S\t389\t*\tLN:i:130\n'
        'S\t390\t*\tLN:i:1\n'
        'S\t391\t*\tLN:i:4\n'
        'S\t392\t*\tLN:i:1\n'
        'S\t393\t*\tLN:i:1\n'
        'S\t394\t*\tLN:i:475\n'
        'S\t842\t*\tLN:i:1\n'
        'S\t1463\t*\tLN:i:4\n'
        'S\t1464\t*\tLN:i:40\n'
        'S\t1465\t*\tLN:i:1\n'
        'S\t1467\t*\tLN:i:7\n'
        'S\t1469\t*\tLN:i:1\n'
        'S\t1470\t*\tLN:i:13\n'
        'S\t1472\t*\tLN:i:1\n'
        'S\t1473\t*\tLN:i:23\n'
        'S\t1475\t*\tLN:i:1\n'
        'S\t1476\t*\tLN:i:4\n'
        'S\t1478\t*\tLN:i:2\n'
        'S\t1479\t*\tLN:i:1\n'
        'S\t1480\t*\tLN:i:1\n'
        'S\t1482\t*\tLN:i:1\n'
        'S\t1484\t*\tLN:i:1\n'
        'S\t1485\t*\tLN:i:21\n'
        'S\t1487\t*\tLN:i:2\n'
        'S\t1488\t*\tLN:i:4\n'
        'S\t1490\t*\tLN:i:1\n'
        'S\t1491\t*\tLN:i:10\n'
        'S\t1493\t*\tLN:i:1\n'
        'S\t1494\t*\tLN:i:94\n'
        'S\t1496\t*\tLN:i:1\n'
        'S\t1497\t*\tLN:i:96\n'
        'S\t1498\t*\tLN:i:1\n'
        'S\t1505\t*\tLN:i:37\n'
        'S\t1506\t*\tLN:i:1\n'
        'S\t1507\t*\tLN:i:1\n'
        'S\t1508\t*\tLN:i:66\n'
        'S\t1509\t*\tLN:i:1\n'
        'S\t1510\t*\tLN:i:1\n'
        'S\t1511\t*\tLN:i:56\n'
        'S\t1512\t*\tLN:i:1\n'
        'S\t1513\t*\tLN:i:1\n'
        'S\t1514\t*\tLN:i:6\n'
        'S\t1515\t*\tLN:i:1\n'
        'S\t1516\t*\tLN:i:1\n'
        'S\t1517\t*\tLN:i:1\n'
        'S\t1518\t*\tLN:i:10\n'
        'S\t1519\t*\tLN:i:1\n'
        'S\t1520\t*\tLN:i:1\n'
        'S\t1521\t*\tLN:i:123\n'
        'S\t1522\t*\tLN:i:14\n'
        'S\t1523\t*\tLN:i:6\n'
        'S\t1524\t*\tLN:i:11\n'
        'S\t1525\t*\tLN:i:1\n'
        'S\t1526\t*\tLN:i:1\n'
        'S\t1527\t*\tLN:i:1\n'
        'S\t1528\t*\tLN:i:4\n'
        'S\t1529\t*\tLN:i:1\n'
        'S\t1530\t*\tLN:i:1\n'
        'S\t1531\t*\tLN:i:1\n'
        'S\t1532\t*\tLN:i:1\n'
        'S\t1533\t*\tLN:i:2\n'
        'S\t1534\t*\tLN:i:2\n'
        'S\t1535\t*\tLN:i:6\n'
        'S\t1536\t*\tLN:i:1\n'
        'S\t1537\t*\tLN:i:1\n'
        'S\t1538\t*\tLN:i:62\n'
        'S\t1539\t*\tLN:i:1\n'
        'S\t1540\t*\tLN:i:1\n'
        'S\t1541\t*\tLN:i:14\n'
        'S\t1542\t*\tLN:i:1\n'
        'S\t1543\t*\tLN:i:1\n'
        'S\t1544\t*\tLN:i:27\n'
        'S\t1545\t*\tLN:i:1\n'
        'S\t1546\t*\tLN:i:1\n'
        'S\t1547\t*\tLN:i:1087\n'
        'S\t1548\t*\tLN:i:1\n'
        'S\t1549\t*\tLN:i:3\n'
        'S\t1551\t*\tLN:i:1\n'
        'S\t1552\t*\tLN:i:1\n'
        'S\t1553\t*\tLN:i:1\n'
        'S\t1555\t*\tLN:i:1\n'
        'S\t1556\t*\tLN:i:1\n'
        'S\t1558\t*\tLN:i:3\n'
        'S\t1560\t*\tLN:i:1\n'
        'S\t1562\t*\tLN:i:3\n'
        'S\t1563\t*\tLN:i:3\n'
        'S\t1565\t*\tLN:i:1\n'
        'S\t1567\t*\tLN:i:2\n'
        'S\t1568\t*\tLN:i:1\n'
        'S\t1569\t*\tLN:i:5\n'
        'S\t1571\t*\tLN:i:1\n'
        'S\t1572\t*\tLN:i:1\n'
        'S\t1574\t*\tLN:i:1\n'
        'S\t1575\t*\tLN:i:1\n'
        'S\t1576\t*\tLN:i:1\n'
        'S\t1578\t*\tLN:i:1\n'
        'S\t1579\t*\tLN:i:13\n'
        'S\t1580\t*\tLN:i:140\n'
        'S\t1581\t*\tLN:i:2\n'
        'S\t1582\t*\tLN:i:2\n'
        'S\t1583\t*\tLN:i:15\n'
        'S\t1584\t*\tLN:i:2\n'
        'S\t1585\t*\tLN:i:2\n'
        'S\t1586\t*\tLN:i:110\n'
        'S\t1587\t*\tLN:i:1\n'
        'S\t1588\t*\tLN:i:120\n'
        'S\t1589\t*\tLN:i:1\n'
        'S\t1590\t*\tLN:i:1\n'
        'S\t1591\t*\tLN:i:30\n'
        'S\t1592\t*\tLN:i:1\n'
        'S\t1593\t*\tLN:i:1\n'
        'S\t1594\t*\tLN:i:141\n'
        'S\t1595\t*\tLN:i:9\n'
        'S\t1596\t*\tLN:i:47\n'
        'S\t1597\t*\tLN:i:1\n'
        'S\t1598\t*\tLN:i:1\n'
        'S\t1599\t*\tLN:i:28\n'
        'S\t1600\t*\tLN:i:1\n'
        'S\t1601\t*\tLN:i:112\n'
        'S\t1603\t*\tLN:i:2\n'
        'S\t1604\t*\tLN:i:3\n'
        'S\t1605\t*\tLN:i:1\n'
        'S\t1606\t*\tLN:i:26\n'
        'S\t1607\t*\tLN:i:1\n'
        'S\t1608\t*\tLN:i:1\n'
        'S\t1609\t*\tLN:i:26\n'
        'S\t1610\t*\tLN:i:1\n'
        'S\t1611\t*\tLN:i:40\n'
        'S\t1612\t*\tLN:i:1\n'
        'S\t1613\t*\tLN:i:1\n'
        'S\t1614\t*\tLN:i:160\n'
        'S\t1615\t*\tLN:i:1\n'
        'S\t1616\t*\tLN:i:64\n'
        'S\t1617\t*\tLN:i:1\n'
        'S\t1618\t*\tLN:i:1\n'
        'S\t1619\t*\tLN:i:473\n'
        'S\t1621\t*\tLN:i:33\n'
        'S\t1622\t*\tLN:i:1\n'
        'S\t1623\t*\tLN:i:289\n'
        'S\t1624\t*\tLN:i:1\n'
        'S\t1625\t*\tLN:i:1\n'
        'S\t1626\t*\tLN:i:3\n'
        'S\t1627\t*\tLN:i:1\n'
        'S\t1628\t*\tLN:i:184\n'
        'S\t1629\t*\tLN:i:1\n'
        'S\t1630\t*\tLN:i:325\n'
        'S\t1631\t*\tLN:i:1\n'
        'S\t1633\t*\tLN:i:320\n'
        'S\t1635\t*\tLN:i:1\n'
        'S\t1636\t*\tLN:i:4\n'
        'S\t1637\t*\tLN:i:1\n'
        'S\t1638\t*\tLN:i:1\n'
        'S\t1639\t*\tLN:i:226\n'
        'S\t1640\t*\tLN:i:1\n'
        'S\t1641\t*\tLN:i:1\n'
        'S\t1642\t*\tLN:i:284\n'
        'S\t1643\t*\tLN:i:1\n'
        'S\t1644\t*\tLN:i:1\n'
        'S\t1645\t*\tLN:i:6\n'
        'S\t1646\t*\tLN:i:1\n'
        'S\t1647\t*\tLN:i:1\n'
        'S\t1648\t*\tLN:i:219\n'
        'S\t1649\t*\tLN:i:1\n'
        'S\t1650\t*\tLN:i:1\n'
        'P\tNC_057143.1#1#1:1\t2+,1463+,1464+,1465+,1467+,1469+,1470+,1472+,1473+,1475+,147'
        '6+,1478+,1479+,1480+,1482+,1484+,1485+,1487+,1488+,1490+,1491+,1493+,1494+,1496+'
        ',1497+,1498+,1505+,1506+,1508+,1509+,1511+,1512+,1514+,1516+,1518+,1519+,1521+,1'
        '522+,1523+,1524+,1525+,1527+,1529+,1530+,1532+,1533+,1535+,1537+,1538+,1540+,154'
        '1+,1543+,1544+,1546+,1547+,1548+,1549+,1551+,1552+,1553+,1555+,1556+,1558+,1560+'
        ',1562+,1563+,1565+,1567+,1568+,1569+,1571+,1572+,1574+,1575+,1576+,1578+,1579+,1'
        '580+,1582+,1583+,1584+,1586+,1587+,1588+,1589+,1591+,1592+,1594+,1595+,1596+,159'
        '7+,1599+,1600+,1601+,1603+,1604+,1605+,1606+,1607+,1609+,1610+,1611+,1612+,1614+'
        ',1615+,1616+,1617+,1619+,1621+,1622+,1623+,1624+,1626+,1627+,1628+,1629+,1630+,1'
        '631+,1633+,1635+,1636+,1637+,1639+,1640+,1642+,1643+,1645+,1646+,1648+,1649+,165'
        '0+,5+,6+,8+,9+,10+,12+,14+,15+,16+,17+,18+,19+,21+,22+,23+,25+,28+,29+,30+,31+,3'
        '2+,34+,35+,36+,37+,39+,40+,42+,43+,44+,46+,47+,49+,50+,52+,53+,55+,56+,58+,59+,6'
        '1+,62+,63+,66+,67+,68+,69+,71+,72+,73+,74+,75+,77+,78+,79+,81+,83+,84+,85+,86+,8'
        '7+,88+,89+,90+,91+,92+,93+,94+,95+,96+,98+,99+,100+,101+,102+,103+,105+,106+,108'
        '+,109+,110+,111+,112+,113+,114+,115+,117+,118+,119+,121+,122+,123+,124+,125+,126'
        '+,127+,129+,131+,132+,133+,134+,135+,136+,138+,139+,140+,142+,143+,144+,145+,146'
        '+,147+,148+,149+,151+,152+,153+,154+,155+,157+,159+,160+,162+,163+,164+,166+,167'
        '+,168+,169+,170+,171+,172+,173+,174+,176+,178+,179+,180+,181+,183+,184+,186+,187'
        '+,188+,189+,190+,191+,193+,194+,195+,196+,197+,198+,199+,201+,202+,203+,204+,205'
        '+,206+,208+,209+,210+,211+,212+,213+,214+,215+,216+,217+,218+,221+,224+,225+,226'
        '+,227+,228+,230+,231+,232+,233+,234+,238+,239+,241+,242+,244+,245+,246+,247+,248'
        '+,249+,250+,251+,252+,253+,254+,255+,256+,257+,258+,259+,260+,261+,263+,264+,265'
        '+,266+,267+,268+,269+,270+,271+,273+,275+,276+,277+,278+,279+,280+,281+,282+,284'
        '+,285+,286+,288+,289+,291+,292+,293+,295+,296+,297+,298+,300+,301+,302+,304+,305'
        '+,307+,308+,309+,310+,312+,313+,315+,316+,317+,318+,319+,320+,321+,322+,323+,324'
        '+,325+,326+,327+,328+,329+,330+,331+,332+,333+,335+,336+,337+,338+,339+,340+,342'
        '+,343+,345+,347+,348+,350+,351+,352+,353+,354+,355+,356+,358+,359+,360+,361+,362'
        '+,363+,365+,367+,368+,369+,371+,372+,373+,374+,375+,378+,379+,380+,381+,386+,387'
        '+,388+,389+,390+,391+,393+,394+\t*\n'
        'P\tNC_057143.1#1#1:2\t63-,62-,60-,59-,57-,56-,54-,53-,51-,50-,48-,47-,45-,44-\t*\n'
        'P\tOZ535733.1#1#1:1\t1505+,1506+,1508+,1510+,1511+,1513+,1514+,1517+,1518+,1520+,1'
        '521+,1522+,1524+,1526+,1527+,1528+,1529+,1531+,1532+,1534+,1535+,1536+,1538+,153'
        '9+,1541+,1542+,1544+,1545+,1547+,1548+\t*\n'
        'P\tOZ535733.1#1#1:2\t66+,67+,68+,69+,71+,72+,73+,74+,76+,77+,78+,79+,80+,81+,82+,8'
        '3+,84+,85+,86+,87+,88+,89+,90+,91+,92+,93+,94+,95+,97+,98+,99+,100+,101+,102+,10'
        '4+,105+,106+,108+,109+,110+,111+,112+,113+,114+,115+,116+,117+,118+,119+,121+,12'
        '2+,123+,124+,125+,126+,128+,129+,131+,132+,133+,134+,135+,136+,137+,138+,139+,14'
        '0+,142+,143+,144+,145+,146+,147+,148+,149+,151+,152+,153+,154+,156+,157+,159+,16'
        '0+,161+,162+,163+,164+,165+,167+,169+,170+,171+,172+,173+,175+,176+,177+,178+,17'
        '9+,180+,182+,183+,185+,186+,187+,188+,189+,190+,191+,193+,194+,195+,196+,197+,19'
        '8+,199+,201+,202+,203+,204+,205+,206+,208+,209+,210+,211+,212+,213+,214+,215+,21'
        '6+,217+,218+,219+,220+,221+,222+,223+,224+,225+,226+,227+,228+,230+,232+,233+,23'
        '4+,235+,236+,237+,238+,240+,241+,243+,244+,245+,246+,247+,248+,249+,250+,251+,25'
        '2+,253+,254+,255+,256+,257+,258+,259+,260+,261+,263+,264+,265+,266+,267+,268+,26'
        '9+,270+,271+,273+,275+,276+,277+,278+,279+,280+,281+,282+,284+,285+,286+,288+,28'
        '9+,290+,291+,292+,293+,295+,296+,297+,298+,300+,301+,303+,304+,306+,307+,308+,30'
        '9+,311+,312+,313+,315+,316+,317+,318+,319+,320+,321+,322+,323+,324+,325+,326+,32'
        '7+,328+,329+,330+,331+,332+,333+,334+,335+,336+,337+,338+,339+,340+,342+,344+,34'
        '5+,347+,348+,350+,351+,352+,353+,354+,355+,357+,358+,359+,360+,361+,362+,364+,36'
        '5+,366+,367+,368+,369+,371+,372+,373+,374+,375+,376+,377+,378+,379+,380+,381+,38'
        '3+,385+,386+,387+,388+,389+,390+,391+,392+,394+\t*\n'
        'P\tOZ535733.1#1#1:3\t63-,62-,60-,59-,57-,56-,54-,53-,51-,50-,48-,47-,45-,44-,43-,4'
        '1-,40-,38-,37-,36-,35-,34-,32-,31-,30-,29-,28-,27-,26-,25-,24-,22-,21-,19-,18-,1'
        '7-,16-,15-,14-,12-,11-,9-,8-,6-,65-,1649-,1648-,1647-,1645-,1644-,1642-,1641-,16'
        '39-,1638-,1636-,1635-,1633-,1631-,1630-,1629-,1628-,1627-,1626-,1625-,1623-,1622'
        '-,1621-,1619-,1618-,1616-,1615-,1614-,1613-,1611-,1610-,1609-,1608-,1606-,1605-,'
        '1604-,1603-,1601-,1600-,1599-,1598-,1596-,1595-,1594-,1593-,1591-,1590-,1588-,15'
        '87-,1586-,1585-,1583-,1581-,1580-,1579-,1578-,1576-,1575-,1574-,1572-,1571-,1569'
        '-,1568-,1567-,1565-,1563-,1562-,1560-,1558-,1556-,1555-,1553-,1552-,1551-,1549-,'
        '1548-,1547-,1545-,1544-,1542-,1541-,1539-,1538-,1536-,1535-,1534-,1532-,1531-,15'
        '29-,1528-,1527-,1526-,1524-,1522-,1521-,1520-,1518-,1517-,1514-,1513-,1511-,1510'
        '-,1508-,1506-,1505-\t*\n'
        'P\tPQ636878.1#1#1:1\t5+,6+,8+,9+,11+,12+,14+,15+,16+,17+,19+,20+,22+,24+,25+,28+,2'
        '9+,30+,31+,32+,33+,35+,36+,37+,38+,40+,41+,43+,44+,46+,47+,49+,50+,52+,53+,55+,5'
        '6+,58+,59+,61+,62+,842+,66+,67+,68+,69+,70+,72+,74+,75+,77+,78+,79+,81+,82+,83+,'
        '84+,85+,86+,87+,88+,89+,90+,91+,92+,94+,95+,97+,98+,99+,100+,101+,102+,103+,105+'
        ',106+,108+,109+,110+,111+,112+,113+,114+,115+,116+,117+,118+,119+,120+,122+,123+'
        ',124+,125+,126+,128+,129+,131+,132+,133+,134+,135+,136+,137+,138+,139+,140+,141+'
        ',143+,144+,145+,146+,147+,148+,149+,151+,152+,153+,154+,156+,157+,158+,160+,162+'
        ',163+,164+,165+,167+,169+,170+,171+,172+,173+,175+,176+,178+,179+,180+,182+,183+'
        ',185+,186+,187+,188+,190+,191+,193+,194+,195+,196+,197+,198+,199+,200+,202+,203+'
        ',204+,206+,208+,209+,210+,211+,212+,213+,214+,215+,216+,217+,218+,221+,222+,223+'
        ',224+,225+,226+,227+,228+,230+,232+,233+,234+,235+,236+,237+,238+,239+,241+,242+'
        ',244+,245+,246+,247+,248+,249+,250+,251+,252+,253+,254+,255+,256+,257+,258+,259+'
        ',260+,261+,263+,264+,265+,266+,267+,268+,269+,270+,271+,273+,275+,276+,277+,278+'
        ',279+,280+,281+,282+,284+,285+,286+,288+,289+,290+,291+,292+,293+,294+,296+,297+'
        ',298+,300+,301+,302+,304+,306+,307+,308+,309+,311+,312+,313+,315+,316+,317+,318+'
        ',319+,320+,321+,322+,323+,324+,325+,327+,328+,329+,330+,331+,332+,333+,334+,335+'
        ',336+,337+,338+,339+,340+,342+,344+,345+,347+,348+,351+,352+,353+,354+,355+,357+'
        ',358+,360+,361+,362+,363+,365+,366+,367+,368+,369+,370+,372+,374+,380+,381+,383+'
        ',385+,386+,387+,389+,390+,391+,393+,394+\t*\n'
        'P\tPQ636878.1#1#1:2\t1582-,1580-,1579-,1578-,1576-,1575-,1574-,1572-,1571-,1569-,1'
        '568-,1567-,1565-,1563-,1562-,1560-,1558-,1556-,1555-,1553-,1552-,1551-,1549-,154'
        '8-,1547-,1545-,1544-,1542-,1541-,1539-,1538-,1536-,1535-,1534-,1532-,1531-,1529-'
        ',1528-,1527-,1526-,1524-,1522-,1521-,1520-,1518-,1515-,1514-,1513-,1511-,1509-,1'
        '508-,1507-,1505-\t*\n'
        'P\tNC_057142.1#1#1:1\t2+,1463+,1464+,1465+,1467+,1469+,1470+,1472+,1473+,1475+,147'
        '6+,1478+,1479+,1480+,1482+,1484+,1485+,1487+,1488+,1490+,1491+,1493+,1494+,1496+'
        ',1497+,1498+,1505+,1506+,1508+,1509+,1511+,1512+,1514+,1516+,1518+,1519+,1521+,1'
        '522+,1523+,1524+,1525+,1527+,1529+,1530+,1532+,1533+,1535+,1537+,1538+,1540+,154'
        '1+,1543+,1544+,1546+,1547+,1548+,1549+,1551+,1552+,1553+,1555+,1556+,1558+,1560+'
        ',1562+,1563+,1565+,1567+,1568+,1569+,1571+,1572+,1574+,1575+,1576+,1578+,1579+,1'
        '580+,1582+,1583+,1584+,1586+,1587+,1588+,1589+,1591+,1592+,1594+,1595+,1596+,159'
        '7+,1599+,1600+,1601+,1603+,1604+,1605+,1606+,1607+,1609+,1610+,1611+,1612+,1614+'
        ',1615+,1616+,1617+,1619+,1621+,1622+,1623+,1624+,1626+,1627+,1628+,1629+,1630+,1'
        '631+,1633+,1635+,1636+,1637+,1639+,1640+,1642+,1643+,1645+,1646+,1648+,1649+,165'
        '0+,5+,6+,8+,9+,10+,12+,14+,15+,16+,17+,18+,19+,21+,22+,23+,25+,28+,29+,30+,31+,3'
        '2+,34+,35+,36+,37+,39+,40+,42+,43+,44+,46+,47+,49+,50+,52+,53+,55+,56+,58+,59+,6'
        '1+,62+,63+,66+,67+,68+,69+,71+,72+,73+,74+,75+,77+,78+,79+,81+,83+,84+,85+,86+,8'
        '7+,88+,89+,90+,91+,92+,93+,94+,95+,96+,98+,99+,100+,101+,102+,103+,105+,106+,108'
        '+,109+,110+,111+,112+,113+,114+,115+,117+,118+,119+,121+,122+,123+,124+,125+,126'
        '+,127+,129+,131+,132+,133+,134+,135+,136+,138+,139+,140+,142+,143+,144+,145+,146'
        '+,147+,148+,149+,151+,152+,153+,154+,155+,157+,159+,160+,162+,163+,164+,166+,167'
        '+,168+,169+,170+,171+,172+,173+,174+,176+,178+,179+,180+,181+,183+,184+,186+,187'
        '+,188+,189+,190+,191+,193+,194+,195+,196+,197+,198+,199+,201+,202+,203+,204+,205'
        '+,206+,208+,209+,210+,211+,212+,213+,214+,215+,216+,217+,218+,221+,224+,225+,226'
        '+,227+,228+,230+,231+,232+,233+,234+,238+,239+,241+,242+,244+,245+,246+,247+,248'
        '+,249+,250+,251+,252+,253+,254+,255+,256+,257+,258+,259+,260+,261+,263+,264+,265'
        '+,266+,267+,268+,269+,270+,271+,273+,275+,276+,277+,278+,279+,280+,281+,282+,284'
        '+,285+,286+,288+,289+,291+,292+,293+,295+,296+,297+,298+,300+,301+,302+,304+,305'
        '+,307+,308+,309+,310+,312+,313+,315+,316+,317+,318+,319+,320+,321+,322+,323+,324'
        '+,325+,326+,327+,328+,329+,330+,331+,332+,333+,335+,336+,337+,338+,339+,340+,342'
        '+,343+,345+,347+,348+,350+,351+,352+,353+,354+,355+,356+,358+,359+,360+,361+,362'
        '+,363+,365+,367+,368+,369+,371+,372+,373+,374+,375+,378+,379+,380+,381+,386+,387'
        '+,388+,389+,390+,391+,393+,394+\t*\n'
        'P\tNC_057142.1#1#1:2\t63-,62-,60-,59-,57-,56-,54-,53-,51-,50-,48-,47-,45-,44-\t*\n'
        'P\tNC_037463.1#1#1:1\t66+,67+,68+,69+,71+,72+,73+,74+,76+,77+,78+,79+,81+,82+,83+,'
        '84+,85+,86+,87+,89+,90+,91+,92+,93+,94+,95+,97+,98+,99+,100+,101+,102+,104+,105+'
        ',106+,108+,109+,110+,111+,112+,113+,114+,115+,116+,117+,118+,119+,121+,122+,123+'
        ',124+,125+,126+,128+,129+,131+,132+,133+,134+,135+,136+,137+,138+,139+,140+,142+'
        ',143+,145+,146+,147+,148+,149+,151+,152+,153+,154+,156+,157+,159+,160+,161+,162+'
        ',163+,164+,166+,167+,169+,170+,171+,172+,173+,175+,176+,177+,178+,179+,180+,182+'
        ',183+,185+,186+,187+,188+,189+,190+,191+,193+,194+,195+,196+,197+,198+,199+,201+'
        ',202+,203+,204+,205+,206+,208+,210+,212+,214+,215+,216+,217+,218+,219+,220+,221+'
        ',222+,223+,224+,225+,226+,227+,228+,229+,230+,233+,234+,235+,236+,237+,238+,240+'
        ',241+,243+,244+,245+,246+,247+,248+,249+,250+,251+,252+,253+,254+,255+,256+,257+'
        ',258+,259+,260+,261+,263+,264+,265+,266+,267+,268+,269+,270+,271+,273+,274+,276+'
        ',277+,278+,279+,280+,281+,283+,284+,285+,286+,288+,289+,290+,291+,292+,293+,295+'
        ',296+,297+,298+,300+,301+,303+,304+,306+,307+,308+,309+,311+,312+,313+,315+,316+'
        ',317+,318+,319+,320+,321+,322+,323+,324+,325+,326+,327+,328+,329+,330+,332+,333+'
        ',334+,335+,336+,337+,338+,339+,340+,341+,342+,344+,345+,347+,348+,351+,352+,353+'
        ',354+,355+,357+,358+,359+,360+,361+,362+,364+,365+,366+,367+,368+,369+,371+,372+'
        ',373+,374+,375+,379+,380+,381+,383+,384+,386+,387+,388+,389+,390+,391+,392+,394+'
        '\t*\n'
        'P\tNC_037463.1#1#1:2\t63-,62-,60-,59-,57-,56-,54-,53-,51-,50-,48-,47-,45-,44-,43-,'
        '41-,40-,38-,37-,36-,35-,34-,32-,31-,30-,29-,28-,27-,26-,25-,24-,22-,21-,19-,18-,'
        '17-,16-,15-,14-,12-,11-,9-,8-,6-,1648-,1647-,1645-,1644-,1642-,1641-,1639-,1638-'
        ',1636-,1635-,1633-,1631-,1630-,1629-,1628-,1627-,1626-,1625-,1623-,1621-,1619-,1'
        '618-,1616-,1615-,1614-,1613-,1611-,1610-,1609-,1608-,1606-,1605-,1604-,1603-,160'
        '1-,1600-,1599-,1598-,1596-,1595-,1594-,1593-,1591-,1590-,1588-,1587-,1586-,1585-'
        ',1583-,1581-,1580-,1579-,1578-,1576-,1575-,1574-,1572-,1571-,1569-,1568-,1567-,1'
        '565-,1563-,1562-,1560-,1558-,1556-,1555-,1553-,1552-,1551-,1549-,1548-,1547-,154'
        '5-,1544-,1542-,1541-,1539-,1538-,1536-,1535-,1534-,1532-,1531-,1529-,1528-,1527-'
        ',1526-,1524-,1522-,1521-,1520-,1518-,1517-,1514-,1513-,1511-,1510-,1508-,1506-,1'
        '505-\t*\n'
        'P\tNC_037463.1#1#1:3\t1505+,1506+,1508+,1510+,1511+,1513+,1514+,1517+,1518+,1520+,'
        '1521+,1522+,1524+,1526+,1527+,1528+,1529+,1531+,1532+,1534+,1535+,1536+,1538+,15'
        '39+,1541+,1542+,1544+,1545+,1547+,1548+\t*\n'
    ),
)

FIXTURES = {
    "visualization.plot_transfer_circos": (
        FixtureCase(
            case="basic",
            files=(_CIRCOS_MITO_GENBANK, _CIRCOS_PLASTID_GENBANK),
            parameters={
                "transfers": _CIRCOS_FRAGMENTS,
                "recipient": "mito.gb",
                "donor": "plastid.gb",
            },
        ),
    ),
    "visualization.plot_assembly_graph": (
        FixtureCase(
            case="basic",
            files=(_ASSEMBLY_GFA,),
            parameters={"gfa": "graph.gfa"},
        ),
    ),
    "visualization.plot_pan_circular": (
        FixtureCase(
            case="basic",
            files=(_PAN_CIRCULAR_GENBANK, _PAN_CIRCULAR_VCF, _PAN_CIRCULAR_GFA),
            parameters={
                "reference_gbk": "reference.gb",
                "variants": "variants.vcf",
                "pangenome_gfa": "graph.gfa",
                "bins": 30,
                "graph_max_blocks": 12,
            },
        ),
    ),
    "visualization.plot_structure_map": (
        FixtureCase(
            case="basic",
            files=(_STRUCTURE_GENBANK,),
            parameters={
                "genbank_path": "structure.gb",
                "ssrs": [{"start": 1, "end": 10}],
                "tandem_repeats": [{"start": 20, "end": 50}],
                "dispersed_repeats": [{"positions": [70, 900], "length": 20}],
            },
        ),
    ),
    "visualization.plot_fst_scan": (
        FixtureCase(
            case="basic",
            files=(_POPGEN_VCF,),
            parameters={"vcf": "toy.vcf", "groups": {"W1": "wild", "W2": "wild", "W3": "wild", "W4": "wild", "C1": "crop", "C2": "crop", "C3": "crop", "C4": "crop"}, "pair": ["wild", "crop"], "estimator": "hudson", "window_size": 500, "step": 250, "highlight": [{"start": 601, "end": 900, "label": "SV"}]},
        ),
    ),
    "visualization.plot_pi_ratio": (
        FixtureCase(
            case="basic",
            files=(_POPGEN_VCF, _POPGEN_REF_FASTA, _POPGEN_GENBANK,),
            parameters={"groups": {"W1": "wild", "W2": "wild", "W3": "wild", "W4": "wild", "C1": "crop", "C2": "crop", "C3": "crop", "C4": "crop"}, "vcf": "toy.vcf", "reference_fasta": "ref.fasta", "genbank": "ref.gb", "pair": ["wild", "crop"], "window_size": 500, "step": 250, "min_sites": 100, "top_fraction": 0.1},
        ),
    ),
    "visualization.plot_sv_schematic": (
        FixtureCase(
            case="basic",
            files=(_POPGEN_GFA,),
            parameters={"gfa": "toy.gfa", "reference": "REF", "start": 1, "end": 2000, "samples": ["W1", "W4", "C1"]},
        ),
    ),
    "visualization.plot_sv_frequency": (
        FixtureCase(
            case="basic",
            files=(_POPGEN_GFA,),
            parameters={"groups": {"W1": "wild", "W2": "wild", "W3": "wild", "W4": "wild", "C1": "crop", "C2": "crop", "C3": "crop", "C4": "crop"}, "gfa": "toy.gfa", "reference": "REF", "min_length": 100},
        ),
    ),
    "visualization.plot_presence_tracks": (
        FixtureCase(
            case="basic",
            files=(_POPGEN_GFA, _POPGEN_GENBANK,),
            parameters={"gfa": "toy.gfa", "reference": "REF", "start": 1, "end": 2000, "groups": {"W1": "wild", "W2": "wild", "W3": "wild", "W4": "wild", "C1": "crop", "C2": "crop", "C3": "crop", "C4": "crop"}, "annotations": {"DNA source": {"W1": "Mt", "W2": "Pt", "W3": "Mt", "W4": "Mt", "C1": "Mt", "C2": "Mt", "C3": "Mt", "C4": "Mt"}}, "genbank": "ref.gb", "min_sv_length": 100},
        ),
    ),
}
