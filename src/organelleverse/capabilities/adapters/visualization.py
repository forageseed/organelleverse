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
  gets ``path`` (really opened); ``artifact`` (``OrganellePlot``). Defaults
  to the self-contained ``outer_method="ogdraw"`` backend, no external
  tool. Every other parameter (all optional file inputs / metadata / the
  Bandage GFA path) keeps its own default. ``side_effects =
  ["read_files"]``.
* ``visualization.plot_tree`` - resolved below: ``newick_path`` gets
  ``path`` (really opened); ``artifact``. ``method="auto"`` (toytree if
  installed, else Bio.Phylo + matplotlib - both self-contained, no
  subprocess). ``side_effects = ["read_files"]``.
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
        parameters=(ParameterOverride(name="reference_gbk", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
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
}
"""Every restored id not listed above is deliberately absent - see the
module docstring for the three exclusion families (custom-dataclass
params/returns, unresolvable ``PathInput``-alias unions, the
``Sequence``/``Mapping``-alias gap, and the charter's ``LEGACY_RESULT``
gap) and which specific capability falls into each.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]

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

FIXTURES = {
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
}
