"""Adapter decisions for the ``selection`` domain's restored capabilities.

**Two more genuinely ungated real external-tool calls, beyond
``phylogeny.align`` and ``transfer.detect_transfers_blast``.**
``selection.align_protein`` (``selection/codeml.py:177-192``) has no
``executor`` gate for its default ``method="mafft"`` path: it runs
``mafft --auto`` directly via ``subprocess.run`` whenever the binary is on
``PATH``. ``selection.kaks_calculator`` (a compute-only wrapper around
``run_kaks_calculator_workflow`` using an internal ``tempfile
.TemporaryDirectory()``, so no directory parameter is exposed) transitively
attempts to spawn the real KaKs_Calculator binary the same way -
``run_kaks_calculator``'s ``subprocess.run(cmd, ...)`` has no injectable
executor at all. ``mafft`` is installed in this environment, and
``selection.prepare_codeml`` (``use_mafft=True`` by default) reaches the
same MAFFT call transitively too. **``KaKs_Calculator`` itself is not
locatable here, discovered by real invocation while resolving Capability
Plan 05's ``run_kaks_calculator_workflow``**: ``check_kaks_calculator``
searches ``PATH`` for a binary literally named ``KaKs``
(``_KAKS_INFO["cli"] = "KaKs"``), not ``KaKs_Calculator`` - a real,
pre-existing name mismatch in the restored source, unrelated to any codec
or binding. A real ``KaKs_Calculator`` binary is on this machine's ``PATH``
under its own name, so the honest, always-reachable outcome for both
``kaks_calculator`` and ``run_kaks_calculator_workflow`` in this
environment is a real ``"failed"`` result (anomaly
``kaks_calculator_not_found``), which is what the invoke tests below for
both actually assert - not a fabricated success. The four codeml
"plan" wrappers (``branch_model``/``branch_site_model``/``clade_model``/
``site_model``) likewise default to ``run=True`` and, when the real
``codeml`` binary is installed (it is, here), spawn it for real via
``run_codeml`` - gated only by that one boolean, never an injectable
callable. The invoke tests below call ``align_protein`` for real (mafft)
and pass ``run=False`` to the codeml planners to keep test runtime bounded
and deterministic - the binding itself, not codeml's own runtime, is what
is being proven.

**A repeated destination-path gap, same shape as ``phylogeny.render_network``
and first generalized here across three capabilities.**
``selection.run_kaks_calculator``'s ``output_file: str | Path`` and
``selection.write_ctl_file``'s ``path: str | Path`` are both write
*destinations* the implementation creates, not existing artifacts: ``PATH``
demands pre-existing ``is_file()``, and this generator's own
``_looks_json_safe`` does not recognize ``Path`` as a JSON-safe name (only
``{str, int, float, bool}``), so neither codec resolves either parameter -
exactly the ``render_network`` finding, now confirmed to recur.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``selection.align_protein`` - resolved below: ``protein_fasta`` gets
  ``path`` (really opened via ``read_fasta``); ``canonical`` (a real
  ``OrganelleResult``). ``method``/``executor`` (``Any``, but optional so
  never bound) keep their own defaults. ``side_effects = ["read_files",
  "subprocess"]``.
* ``selection.branch_model`` / ``.branch_site_model`` / ``.clade_model`` /
  ``.site_model`` - resolved below: each's required ``alignment``/``tree``
  (both ``str | Path``, kwonly) get ``path``; ``run: bool`` (default
  ``True``) is also declared, covered by ``json``, so a caller can pass
  ``run=False`` explicitly to force the plan-only path deterministically -
  ``canonical``. Each uses an internal ``tempfile.TemporaryDirectory()``
  for its ``write_*`` counterpart's required ``output_dir``, so no
  directory parameter is exposed here (unlike the ``write_*`` siblings
  below). Every other parameter keeps its own default. ``side_effects =
  ["read_files", "subprocess"]`` - real with ``run=True`` (the function's
  own default) and ``codeml`` installed, per the module docstring above.
* ``selection.check_kaks_calculator`` - resolved below: zero parameters;
  the nested-dict return is finite JSON, so ``json_metric`` under
  ``"kaks_calculator_status"``. Really scans conda/micromamba environments
  (same ``morphology``/``population`` pattern), so ``side_effects =
  ["subprocess"]``.
* ``selection.install_hint`` - resolved below: zero parameters; the bare
  ``str`` return is finite JSON, so ``json_metric`` under ``"hint_text"``.
* ``selection.compute_kaks`` - resolved below: ``cds_fasta`` gets ``path``
  (really opened via ``read_fasta``); the ``dict`` return is finite JSON,
  so ``json_metric`` under ``"kaks"``. ``side_effects = ["read_files"]``.
* ``selection.fasta_to_axt`` - resolved below: ``fasta_path`` gets ``path``;
  the public service writes AXT to managed storage and returns its ``Path``.
  The Agent codec is ``artifact``; it never accepts an output path.
* ``selection.kaks`` - resolved below: ``cds_pairs_fasta`` gets ``path``;
  ``canonical``. ``jukes_cantor`` keeps its own default. ``side_effects =
  ["read_files"]``.
* ``selection.kaks_calculator`` - resolved below: ``cds_fasta`` gets
  ``path``; ``canonical`` - see the module docstring above for the real
  KaKs_Calculator subprocess call this transitively makes.
  ``side_effects = ["read_files", "subprocess"]``.
* ``selection.likelihood_ratio_test`` - resolved below: ``lnL_null``,
  ``lnL_alt`` (``float``), ``df`` (``int``) all get ``json``; the
  ``dict[str, float]`` return is finite JSON, so ``json_metric`` under
  ``"lrt"``.
* ``selection.pal2nal`` - resolved below: both required
  ``protein_alignment_fasta``/``cds_fasta`` get ``path``; ``canonical``.
  ``side_effects = ["read_files"]``.
* ``selection.parse_codeml_output`` - resolved below: ``out_path`` gets
  ``path`` (really opened via ``Path(out_path).read_text()``); the
  ``dict[str, Any]`` return is finite JSON at every real call (only
  numeric/string fields), so ``json_metric`` under ``"codeml_output"``.
  ``side_effects = ["read_files"]``.
* ``selection.parse_kaks_output`` - resolved below: ``output_file`` gets
  ``path`` (really opened via ``Path(output_file).read_text()``); the
  ``list[dict[str, Any]]`` return is finite JSON at every real call, so
  ``json_metric`` under ``"kaks_hits"``. ``side_effects = ["read_files"]``.
* ``selection.prepare_codeml`` - resolved below: ``cds_fasta`` gets
  ``path``; ``canonical``. Uses an internal ``tempfile.TemporaryDirectory()``
  for ``write_codeml_inputs``'s required ``output_dir``, so no directory
  parameter is exposed. ``require_start``/``use_mafft`` (``True`` by
  default) keep their own defaults. ``side_effects = ["read_files",
  "subprocess"]`` - real with the default ``use_mafft=True``.
* ``selection.to_paml`` - resolved below: ``alignment_fasta`` gets ``path``;
  ``canonical``. ``convert_t_to_u`` keeps its own default. ``side_effects =
  ["read_files"]``.
* ``selection.translate_cds`` - resolved below: ``cds_fasta`` gets ``path``;
  ``canonical``. ``strip_stop`` keeps its own default. ``side_effects =
  ["read_files"]``.
* ``selection.validate_cds`` - resolved below: ``cds_fasta`` gets ``path``;
  ``canonical``. ``require_start``/``allow_internal_stop`` keep their own
  defaults. ``side_effects = ["read_files"]``.
* ``selection.validate_cds_sequences`` - the ``json``-ish twin of
  ``validate_cds`` (returns a bare ``dict``, delegates to ``validate_cds``
  internally): resolved below: ``cds_fasta`` gets ``path``; the ``dict``
  return is finite JSON, so ``json_metric`` under ``"cds_validation"``.
  ``require_start`` keeps its own default. ``side_effects = ["read_files"]``.
* ``selection.batch_codeml`` - **excluded**: required ``jobs: list[dict]`` -
  the bare-``dict``-in-``list`` gap first documented in the ``phylogeny``
  adapter, since Capability Plan 03 Task 4 rejected by the generator's own
  ``_looks_json_safe`` at generation time too (it always failed at the real
  runtime binder). No entry here.
* ``selection.write`` - **excluded**: required ``obj: Any`` - explicitly
  typed ``Any``, refused by both ``PATH`` and ``JSON`` codecs outright
  (the same shape as ``transfer.detect``). No entry here.
* ``selection.write_branch_model`` / ``.write_branch_site_model`` /
  ``.write_clade_model`` / ``.write_site_model`` - resolved below: each's
  required ``alignment``/``tree`` (kwonly, both ``str | Path``) get ``path``;
  required ``output_dir: str | Path`` gets ``directory`` with ``path_role =
  "output"`` - each does ``out = Path(output_dir); out.mkdir(parents=True,
  exist_ok=True)`` unconditionally, the durable, caller-visible counterpart
  of ``branch_model``/``site_model``/``.``'s own internal
  ``tempfile.TemporaryDirectory()``. ``run: bool`` (default ``True``) is also
  declared, covered by ``json``, so a caller can force the plan-only path
  deterministically, same as the already-admitted plan wrappers.
  ``foreground_labels``/``clade1_labels``/``clade2_labels``/``models``/
  ``seqfile``/``treefile`` all keep their own defaults. ``canonical`` result
  (each already builds a real ``OrganelleResult``). ``side_effects =
  ["read_files", "write_files", "subprocess"]`` for all four: ``alignment``/
  ``tree`` are genuinely read (``_prepare_tree``, ``_ensure_paml_format``),
  the ctl/seq/tree files are genuinely written into ``output_dir`` (not an
  ephemeral tempdir here), and ``run_codeml`` genuinely spawns ``codeml``
  with its real default ``run=True`` and ``codeml`` installed (per this
  module's own docstring above). Verified by real invocation: each of the
  four hardcodes its own ``_contract.ok(...)`` op name (``"branch_model"``,
  ``"branch_site_model"``, ``"clade_model"``, ``"site_model"``) regardless
  of caller, so the returned ``OrganelleResult`` always carries the
  ``write_``-less sibling's ``operation_id`` - the same cross-function
  quirk already documented for ``population.detect_numt``.
* ``selection.write_codeml_inputs`` - resolved below: ``cds_fasta`` gets
  ``path`` (really opened via ``read_fasta``); required
  ``output_dir: str | Path`` gets ``directory`` with ``path_role =
  "output"`` - unconditionally created and where every pipeline stage's
  output file lands, the durable counterpart of ``prepare_codeml``'s own
  internal tempdir. ``require_start``/``use_mafft`` (``True`` by default)
  keep their own defaults. ``canonical`` result. ``side_effects =
  ["read_files", "write_files", "subprocess"]`` - real with the default
  ``use_mafft=True`` (mafft is installed, per this module's own docstring).
  Verified by real invocation: the returned ``operation_id`` reads
  ``"selection.prepare_codeml"``, not ``"selection.write_codeml_inputs"`` -
  the implementation's own ``_contract.ok("prepare_codeml", ...)`` call,
  same quirk shape.
* ``selection.run_kaks_calculator_workflow`` - resolved below: ``cds_fasta``
  gets ``path``; the public service allocates a managed output directory.
  ``method``/``genetic_code``/``kaks_path`` retain their defaults. The
  canonical result currently carries ``selection.kaks_calculator`` as its
  operation ID because the underlying workflow uses that name.
* ``selection.run_codeml`` - resolved below: ``ctl_path`` gets ``path``
  (genuinely read); the public service copies the control file and relative
  sequence/tree inputs into managed storage before invoking CodeML. The
  Python return is a parsed mapping or ``None``; the Agent binding wraps
  it as ``json_metric`` under ``"codeml_run"``.
* ``selection.run_kaks_calculator`` - **excluded**: required
  ``output_file: str | Path`` is a write destination, not an existing
  artifact - see the module docstring above (the repeated
  destination-path gap). No entry here.
* ``selection.write_ctl_file`` - **excluded**: required ``path: str |
  Path`` is a write destination - the same repeated destination-path gap.
  No entry here.
* ``selection.write_kaks`` - **excluded**: required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim. No entry here.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``run_kaks_calculator`` is INTERNAL (subprocess runner helper;
 Decision 004 item 4); ``write`` stays REJECT.
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
    # Zero-out batch: jobs rows are {ctl_path, work_dir, label?}, all str.
    "selection.batch_codeml": NamedParameterOverride(
        parameters=(ParameterOverride(name="jobs", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.align_protein": NamedParameterOverride(
        parameters=(ParameterOverride(name="protein_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.branch_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.branch_site_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.clade_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.site_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.check_kaks_calculator": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="kaks_calculator_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "selection.install_hint": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hint_text",
        side_effects=(),
    ),
    "selection.compute_kaks": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="kaks",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.fasta_to_axt": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    ),
    "selection.kaks": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_pairs_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.kaks_calculator": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.likelihood_ratio_test": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="lnL_null", codec=ParameterCodec.JSON),
            ParameterOverride(name="lnL_alt", codec=ParameterCodec.JSON),
            ParameterOverride(name="df", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="lrt",
        side_effects=(),
    ),
    "selection.pal2nal": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="protein_alignment_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.parse_codeml_output": NamedParameterOverride(
        parameters=(ParameterOverride(name="out_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="codeml_output",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.parse_kaks_output": NamedParameterOverride(
        parameters=(ParameterOverride(name="output_file", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="kaks_hits",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.prepare_codeml": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.to_paml": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.translate_cds": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.validate_cds": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.validate_cds_sequences": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="cds_validation",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "selection.write_branch_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(
                name="output_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
            ),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.write_branch_site_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(
                name="output_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
            ),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.write_clade_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(
                name="output_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
            ),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.write_site_model": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="alignment", codec=ParameterCodec.PATH),
            ParameterOverride(name="tree", codec=ParameterCodec.PATH),
            ParameterOverride(
                name="output_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
            ),
            ParameterOverride(name="run", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.write_codeml_inputs": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(
                name="output_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
            ),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
    "selection.run_kaks_calculator_workflow": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "selection.run_codeml": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="ctl_path", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="codeml_run",
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
    ),
}
"""``selection.batch_codeml``, ``.write``, ``.run_kaks_calculator``,
``.write_ctl_file``, and ``.write_kaks`` are deliberately absent - see the
module docstring for why each fails closed honestly.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
