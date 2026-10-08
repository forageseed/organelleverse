"""Adapter decisions for the ``morphology`` domain's restored capabilities.

Chosen as Capability Plan 03 Task 3's second domain and its **first** domain
containing a restoration-ledger ``execution_class`` of ``tool`` or ``model``
(``morphology.check_backend`` is ``tool``, ``morphology.measure`` is
``model``) - see the charter's instruction to start there because that path
(a genuine external-tool dependency, in principle reachable through
``LocalAdmissionEnvironment.observe_environment``'s ``DependencyKind.EXECUTABLE``
probe) had never been exercised by the ``format_conversion`` proof domain.

**What the tool/model path actually turned out to need - and not need.**
Reading every one of this domain's (and, for corroboration, several other
domains') ``tool``/``model`` implementations turned up two separate reasons
none of them ever reaches ``LocalAdmissionEnvironment.observe_environment``'s
``DependencyKind.EXECUTABLE`` branch (``admission.py:59-75``):

1. Every external-tool call in the restored source is either (a) gated behind
   a non-JSON-representable ``executor: Callable[[list[str]], Any] | None =
   None`` parameter - so an honest ``named_parameters`` binding, which can
   never expose a Python callable to an Agent as a JSON argument, can only
   ever invoke it with ``executor`` omitted, i.e. in "plan" mode, and the
   real subprocess is never spawned - or (b) self-probes via
   ``shutil.which``/a conda-env scan and degrades gracefully (no raise),
   reporting availability as a plain JSON field instead of demanding the
   admission layer enforce it. ``morphology.check_backend`` is case (b): it
   spawns a real, unconditional ``python -c "import mmseg"`` subprocess per
   candidate environment (no ``executor`` gate at all) and returns
   ``installed: False`` when none has it - a genuine, ungated tool
   invocation, just one that never raises.
2. The restoration ledger's own ``dependencies`` field - inspected across
   all 253 records, not just this domain's - only ever recorded Python
   ``import`` names (``numpy``, ``skimage``, ``Bio``, ...), never an OS
   executable name, for every ``tool``/``model`` record including this
   domain's two. The generator has no ledger-sourced signal that would ever
   produce a ``DependencyKind.EXECUTABLE`` dependency, and
   ``NamedParameterOverride`` (this module's own override type) has no field
   to declare one by hand - doing so honestly would first require extending
   ``PlannedBundle``/``render_capability_toml`` (both hardcode
   ``kind = "python"`` today), which is generator work, not a domain
   adapter's job, and was not attempted here.

A second constraint narrowed this domain further, until Capability Plan 05's
``ParameterCodec.DIRECTORY`` closed it: ``ParameterCodec.PATH``'s preflight
(``python_binding.py``'s ``_prepare_path_parameters``) requires
``candidate.is_file()`` - a directory always fails it - and
``ParameterCodec.JSON`` has no route for a ``str | Path``-typed parameter
either, because the generator's own pre-validation (``_looks_json_safe``)
treats a bare ``Path`` name as JSON-unsafe. A required *directory* parameter
was therefore unbindable by either codec. ``morphology.default_executor``
(``out_dir`` required), ``morphology.train``, ``morphology.training_code_snippet``,
and ``morphology.validate_dataset`` (``data_root`` required) all carry
exactly this shape and are resolved below with the new codec - see each
bullet for its declared ``path_role``. The same gap, read across domains,
also affected ``coevolution.run_orthofinder``/``run_coevolution``
(``proteome_dir``). The phylogeny tool runs now use managed output locations
and expose only their input FASTA paths.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``morphology.check_backend`` (``tool``) - resolved below: ``name: str`` is
  a plain JSON-safe scalar; the ``dict[str, Any]`` it returns is recursively
  finite JSON at every real call (``json_metric`` checks the actual runtime
  value, never the ``Any``-bearing annotation), so ``json_metric`` under
  ``"backend_status"`` is honest. Declares ``side_effects = ["subprocess"]``
  - it really does spawn one, per environment scanned.
* ``morphology.check_all_backends`` (``pure`` in the ledger, though it fans
  out to ``check_backend`` per registered backend and therefore spawns the
  same subprocess) - resolved below with zero declared parameters
  (``scan_envs`` keeps its own default; nothing required is left uncovered).
* ``morphology.install_hint`` - resolved below: ``name: str`` in, a plain
  ``str`` out (``install_hint() -> str``, itself finite JSON), so
  ``json_metric`` under ``"hint_text"``.
* ``morphology.measure`` (``model``) - resolved below: ``label_map: str |
  Path | np.ndarray`` gets the ``path`` codec. ``_resolve_path_parameter_plan``
  (``python_binding.py:482-507``) only needs *one* ``str``/``Path`` arm of a
  union to decide a target type - it does not require every arm to be a
  path type - so the ``np.ndarray`` arm is simply an unexposed calling
  convention, not a validation obstacle; the implementation's own
  ``_load_image`` already accepts a plain path string. The returned
  ``dict[str, Any]`` (per-object morphometrics) is recursively finite JSON
  at runtime (areas/angles are rounded floats or ``None``, never NaN), so
  ``json_metric`` under ``"morphometrics"``.
* ``morphology.overlay`` - resolved below: both ``image`` and ``label_map``
  get the ``path`` codec for the same reason as ``measure``. The return
  value is a raw ``np.ndarray`` (a colorized overlay image), which
  ``ResultCodec.ARTIFACT`` supports directly via
  ``codecs._try_encode_numpy_artifact`` (materializes it as a content-hashed
  ``.npy`` artifact) - no JSON conversion invented.
* ``morphology.segment`` - resolved below: ``images: str | Path | list[str | Path]`` gets the
  ``path`` codec by the same single-arm reasoning as ``measure``/``overlay``
  (only a single-image call is exposed to the Agent; the list-of-images
  calling convention is simply not surfaced). Every other parameter
  (``executor``, ``device``, ``config``, ``checkpoint``,
  ``orgseg_root``, ``measure_objects``, ``intensity_image``,
  ``pixel_size_um``) has a default and is left uncovered - with ``executor``
  never exposed, every real Agent call takes the "plan" branch without
  creating an output directory (``status=
  "warning"``, ``flags=("morph_planned", ...)``), which is still a real,
  fully-formed ``OrganelleResult`` the implementation itself produced, so
  ``result_codec = "canonical"`` with no invented wrapping.
* ``morphology.build_app`` / ``.serve`` - return ``Any`` (a live
  ``gradio.Blocks`` instance, and ``serve`` additionally blocks the calling
  thread to run a web server); no codec converts a live UI object or a
  running server into anything. No entry here.
* ``morphology.default_executor`` - resolved below: ``config``/``checkpoint``
  get ``path`` (existing files ``init_model`` opens); ``device: str`` and
  ``image_paths: list[str]`` get ``json``; ``out_dir`` gets ``directory`` with
  ``path_role = "output"`` - it is unconditionally ``mkdir(parents=True,
  exist_ok=True)``-created and is where every inferred label-map PNG is
  written, an Agent-chosen destination that need not exist beforehand, the
  textbook ``output`` shape. The real return, ``list[Path]``, is exactly the
  "sequence of ``Path``" shape ``ResultCodec.ARTIFACT`` already supports
  (``codecs._try_encode_numpy_artifact``'s sibling path-list handling), so
  ``result_codec = "artifact"``. ``side_effects = ["read_files",
  "write_files"]`` - real, unconditional, every call: ``mmseg`` (a local,
  in-function import, so importing this module never requires it) reads each
  submitted image and every label map is written to ``out_dir``. Declared as
  bindable and admitted, but **deliberately never invoked for real** in this
  repository's tests: ``mmseg`` is not installed in this environment (a
  restoration-ledger ``dependencies`` entry, not something admission enforces
  - ``LocalAdmissionEnvironment.observe_environment`` never probes a
  ``DependencyKind.PYTHON`` dependency, see this module's own survey above),
  so a real call would raise ``ModuleNotFoundError`` - a real environment gap,
  not a binding defect.
* ``morphology.train`` - resolved below: ``data_root`` gets ``directory``
  with ``path_role = "input"`` - a pre-existing fine-tune dataset tree
  (``image/``, ``label/``, ``splits/``) the implementation reads via its own
  ``validate_dataset(data_root)`` call before ever considering training.
  Every other parameter (``executor``, ``config``, ``load_from``,
  ``max_iters``, ``batch_size``, ``device``, ``orgseg_root``)
  keeps its own default; ``executor`` is never Agent-exposed (a
  ``Callable[[dict[str, Any]], None]`` cannot be a JSON argument), so every
  real call takes the plan-only branch (``status = "warning"``, a real,
  self-constructed ``OrganelleResult``), so ``result_codec = "canonical"``.
  ``side_effects = ["read_files", "subprocess"]`` - ``validate_dataset``
  genuinely walks the directory tree, and ``check_backend("orgseg",
  scan_envs=True)`` genuinely scans conda/micromamba environments, both
  unconditionally, on every call regardless of ``executor``.
* ``morphology.training_code_snippet`` - resolved below: ``data_root`` gets
  ``directory`` with ``path_role = "input"`` - the same dataset tree as
  ``train``, though this function only ever ``str()``-embeds it into the
  returned Python snippet text (never opened) - directory-input containment
  still applies (the tree must exist and is hashed for provenance) regardless
  of whether the implementation itself reads it, exactly as ``path``-codec
  precedent already established for a str()-embedded file path
  (``phylogeny.build_tree``, ``population.cytonuclear_gwas``). The bare
  ``str`` return is finite JSON, so ``json_metric`` under
  ``"training_snippet"``. ``side_effects = ()`` - no real filesystem access.
* ``morphology.validate_dataset`` - resolved below: ``data_root`` gets
  ``directory`` with ``path_role = "input"`` - genuinely walked
  (``img_dir.is_dir()``, ``lab_dir.is_dir()``, each ``splits/*.txt`` read) to
  validate the fine-tune dataset layout. The returned ``dict[str, Any]`` is
  recursively finite JSON at every real call (``valid: bool``, three
  ``int`` counts, ``problems: list[str]``, ``data_root: str`` - never an
  ``Any``-shaped value in practice, the same class of judgment call
  ``morphology.check_backend``'s own ``dict[str, Any]`` return already makes),
  so ``json_metric`` under ``"dataset_validation"``. ``side_effects =
  ["read_files"]``.
* ``morphology.summarize`` / ``.write_csv`` - both take
  ``per_object: list[dict[str, Any]] | dict[str, Any]`` as a required first
  parameter. The generator's own pre-validation (``_looks_json_safe``) is
  lenient enough to wave this through (it does not look inside a
  ``dict[str, Any]``'s value type), but the real runtime check the binder
  actually uses, ``operations.signature._is_supported_json_annotation``, is
  stricter: ``Any`` is explicitly rejected wherever it appears, including
  nested inside a declared container. Declaring ``json`` here would pass
  this generator's validation and then fail for real, at every invocation,
  which is worse than not generating a bundle at all - so no entry here,
  deliberately, even though nothing in this generator's own checks would
  have stopped it.
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
    """The full ``named_parameters`` binding plan for one restored capability.

    ``deterministic`` defaults to True (the generator's historical blanket
    value); ``morphology.train`` overrides it to False below - stochastic
    gradient descent is not deterministic, and the contract must say so.
    ``extra_keywords`` carries the image modality: the frozen OperationSpec
    validators reserve ``input_modalities`` / ``output_modalities`` for
    CoreKind.DATA core-object flows, while these bundles take image/dataset
    paths as named parameters (task brief 2026-08-17, constraints a+b), so
    the modality is expressed as a searchable contract keyword instead.
    """

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]
    deterministic: bool = True
    deterministic_reason: str = ""
    extra_keywords: tuple[str, ...] = ()


OVERRIDES: dict[str, NamedParameterOverride] = {
    # Zero-out batch: per-object measurement rows are primitive mappings
    # (float fields + class_name); the dict arm is the measure() envelope.
    "morphology.summarize": NamedParameterOverride(
        parameters=(ParameterOverride(name="per_object", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="morphology_summary",
        side_effects=(),
    ),
    "morphology.check_backend": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "morphology.check_all_backends": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "morphology.install_hint": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hint_text",
        side_effects=(),
    ),
    "morphology.measure": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="label_map", codec=ParameterCodec.PATH),
            # Optional extras so an Agent can request calibrated areas (µm²)
            # and stamp the label map's provenance (model / human_corrected /
            # llm / ...) on every per-object row, matching the desktop
            # surface's mask-layer parity.
            ParameterOverride(name="pixel_size_um", codec=ParameterCodec.JSON),
            ParameterOverride(name="label_source", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="morphometrics",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "morphology.overlay": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="image", codec=ParameterCodec.PATH),
            ParameterOverride(name="label_map", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    ),
    "morphology.segment": NamedParameterOverride(
        parameters=(ParameterOverride(name="images", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
        extra_keywords=("electron-micrograph",),
    ),
    "morphology.default_executor": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="config", codec=ParameterCodec.PATH),
            ParameterOverride(name="checkpoint", codec=ParameterCodec.PATH),
            ParameterOverride(name="device", codec=ParameterCodec.JSON),
            ParameterOverride(name="image_paths", codec=ParameterCodec.JSON),
            ParameterOverride(name="out_dir", codec=ParameterCodec.DIRECTORY, path_role="output"),
        ),
        result_codec=ResultCodec.ARTIFACT,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    ),
    "morphology.train": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="data_root", codec=ParameterCodec.DIRECTORY, path_role="input"),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
        deterministic=False,
        deterministic_reason="Model fine-tuning depends on random initialization and data ordering.",
        extra_keywords=("electron-micrograph",),
    ),
    "morphology.training_code_snippet": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="data_root", codec=ParameterCodec.DIRECTORY, path_role="input"),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="training_snippet",
        side_effects=(),
    ),
    "morphology.validate_dataset": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="data_root", codec=ParameterCodec.DIRECTORY, path_role="input"),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="dataset_validation",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""Every restored id this domain resolves outside the generic ``canonical_core``
path. ``build_app``, ``serve``, ``summarize``, and ``write_csv`` are
deliberately absent - see the module docstring for why each one fails closed
honestly rather than being forced through a codec that cannot actually carry
it.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
