"""Capability Plan 03, Task 4 + Capability Plan 05: generator/runtime agreement.

``scripts/capabilities/build_restored_bundles.py``'s ``_looks_json_safe`` is
the generator's AST-only pre-validation for ``ParameterCodec.JSON`` -
Task 3's adapter docstrings (``organelleverse.capabilities.adapters
.visualization``, ``.phylogeny``, ``.rna_editing``, ``.coevolution``,
``.selection``) repeatedly documented two ways it disagreed with the real
runtime binder, ``organelleverse.operations.signature
._is_supported_json_annotation``; Capability Plan 05 added a third
(withdrawing ``Path`` from the codec wall's JSON-safe set):

1. ``_looks_json_safe`` never recognized ``collections.abc``/``typing``
   aliases (``Sequence[...]``, ``Mapping[...]``) - only the bare spellings
   ``list``, ``dict``, ``tuple`` - even though the real runtime binder's
   ``_JSON_SEQUENCE_ORIGINS = {list, Sequence}`` and ``_JSON_MAPPING_ORIGINS
   = {dict, Mapping}`` (``operations/signature.py:50-51``) treat them
   identically to their bare counterparts. Fixed with real, element-wise
   checking (not just outer-name matching) - including the explicit trap:
   ``Mapping[str, Any]`` and ``Sequence[Mapping[str, Any]]`` must stay
   rejected by both, because ``Any`` is independently unsafe at the real
   binder (``_is_supported_json_annotation`` explicitly returns ``False``
   for ``Any``) - widening container-alias recognition must never also,
   accidentally, admit ``Any``.

2. A bare, unparameterized ``dict`` directly inside ``list[...]`` (e.g.
   ``list[dict]``) passed the old ``_looks_json_safe`` (it only ever
   inspected a ``Subscript``'s *outer* name) but is rejected by the real
   runtime binder: ``get_origin(dict) is None`` for an un-subscripted
   ``dict``, so it matches no branch of ``_is_supported_json_annotation``
   and falls through to ``False``. ``phylogeny.build_mjn``/``.build_msn``/
   ``.build_tcs_network``/``.pairwise_distances``,
   ``rna_editing.validate_edits``, ``coevolution.coevolution_network``, and
   ``selection.batch_codeml`` all have a required ``list[dict]`` parameter
   and stay excluded either way - this is not a rescue, it is closing the
   generator's false-positive so it fails closed *at generation time*
   instead of only at ``verify_capability`` time.

3. ``Path`` itself - bare or nested at any depth - was withdrawn from the
   JSON-safe set at the bundle codec wall (the bare-``Path`` JSON hole,
   decision 004 §5): ``_is_supported_json_annotation`` gained an
   ``allow_path`` keyword, the codec wall (``python_binding``'s JSON codec)
   calls it with ``allow_path=False``, and registry admission of the 20
   released operations keeps the lenient default - a deliberate wall split,
   not a relaxation. ``_looks_json_safe`` never recognized ``Path`` as a
   *name*, so it already matched the strict variant for bare or unioned
   ``Path``; making its legacy-container branch fully recursive (fixing
   defect 2's outer-name-only shortcut for good) is what withdraws ``Path``
   *nested in a container* (``list[str | Path]``) at generation time too.
   That recursion also made ``_is_bare_unparameterized_container`` dead
   code - a bare ``dict`` element now fails naturally as a ``Name`` outside
   ``_JSON_SAFE_NAMES`` - so it and ``_BARE_CONTAINER_NAMES`` were deleted.
   The one admitted user of the hole (``phylogeny.extract_shared_genes``)
   migrated to the PATH codec's new list form in the same change, so the
   admitted/skipped totals below are unchanged.

Every case below is cross-checked against the real
``_is_supported_json_annotation`` called with ``allow_path=False`` - the
strict codec-wall variant (not a hand-reasoned duplicate) on the
*equivalent real type object* - so a passing test here really does mean the
generator and the runtime codec wall agree, not just that the generator's
private opinion changed.

Measured (by generating every domain and inspecting every skipped
capability's real AST signature, not just re-reading the ``visualization``
adapter's own docstring): every ``Sequence[...]``/``Mapping[...]``-typed
required parameter among the 121 skipped ledger records has ``Any``
somewhere in its shape (verified across all 22 domains, not just the 21
``visualization`` enumerates by name), and every ``list[dict]``-typed one
(``phylogeny`` x4, ``rna_editing``, ``coevolution``, ``selection`` - 7
total) is genuinely unbindable at the real runtime too - so fixing both
disagreements changes *zero* domains' generated/skipped id sets. Both are
still the right fix: the generator's pre-validation now tells the truth
about ``Sequence``/``Mapping`` instead of refusing them for the wrong
reason, and it now also tells the truth about a bare ``dict`` inside
``list[...]`` instead of waving it through to fail later at bind time.
Confirmed against the real, unmodified domain build reports in
``test_no_domains_generated_or_skipped_sets_change``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from organelleverse.operations.signature import (
    _is_supported_json_annotation,  # pyright: ignore[reportPrivateUsage]
)
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _annotation_expr(source: str) -> ast.expr:
    """Parse ``source`` (a bare annotation, e.g. ``"Sequence[str]"``) to an AST expr."""
    module = ast.parse(source, mode="eval")
    assert isinstance(module, ast.Expression)
    return module.body


# Every case is (annotation source text, equivalent real type object,
# expected agreement between the generator's AST check and the real runtime
# binder). Both are evaluated and asserted equal to each other AND to the
# expected value - so a bug in either the test's own expectation or in one
# checker alone cannot silently pass.
_AGREEMENT_CASES: tuple[tuple[str, object, bool], ...] = (
    # --- defect 1: Sequence[...]/Mapping[...] aliases, now recognized -----
    ("Sequence[str]", AbcSequence[str], True),
    ("Sequence[int]", AbcSequence[int], True),
    ("Mapping[str, int]", AbcMapping[str, int], True),
    ("Mapping[str, str]", AbcMapping[str, str], True),
    ("Sequence[Mapping[str, int]]", AbcSequence[AbcMapping[str, int]], True),
    ("Mapping[str, Sequence[str]]", AbcMapping[str, AbcSequence[str]], True),
    # --- the explicit trap: Any poisons Mapping/Sequence either way -------
    ("Mapping[str, Any]", AbcMapping[str, Any], False),
    ("Sequence[Mapping[str, Any]]", AbcSequence[AbcMapping[str, Any]], False),
    ("Sequence[Any]", AbcSequence[Any], False),
    # a non-``str`` key is never safe, alias or not
    ("Mapping[int, str]", AbcMapping[int, str], False),
    # --- defect 2: a bare, unparameterized dict inside list[...] ----------
    ("list[dict]", list[dict], False),
    ("tuple[dict, ...]", tuple[dict, ...], False),
    # --- the bare-Path hole closed: Path is withdrawn at any nesting depth -
    ("list[str | Path]", list[str | Path], False),
    # --- regression guards: existing, currently-admitted shapes -----------
    ("list[tuple[str, str]]", list[tuple[str, str]], True),
    ("dict[str, dict[str, float]]", dict[str, dict[str, float]], True),
    ("list[str]", list[str], True),
    ("list[float]", list[float], True),
)


@pytest.mark.parametrize("source,real_type,expected", _AGREEMENT_CASES)
def test_looks_json_safe_agrees_with_the_real_runtime_binder(
    source: str, real_type: object, expected: bool
) -> None:
    """``_looks_json_safe`` (AST-only) must agree with ``_is_supported_json_annotation``.

    ``real_type`` is the actual Python type object the annotation source
    resolves to - so this is not two hand-written opinions being compared
    to each other, it is the generator's guess checked against the one
    real runtime function that decides whether ``ParameterCodec.JSON``
    actually binds, called with ``allow_path=False``: the strict variant
    the bundle codec wall itself uses.
    """
    generator = _load_generator()
    ast_result = generator._looks_json_safe(_annotation_expr(source))
    runtime_result = _is_supported_json_annotation(real_type, allow_path=False)

    assert runtime_result is expected, (
        f"test's own expectation for {source!r} disagrees with the real runtime binder - "
        f"the test is wrong, not the code"
    )
    assert ast_result == runtime_result, (
        f"generator._looks_json_safe({source!r}) = {ast_result}, but the real runtime "
        f"binder says {runtime_result} for the equivalent type {real_type!r} - "
        "a widened recognizer must never admit something that cannot honestly bind, "
        "and must not stay blind to a shape the runtime really accepts"
    )


def test_bare_bindable_str_and_none_are_unaffected() -> None:
    """Sanity: the pre-existing, always-safe leaves keep working."""
    generator = _load_generator()
    for source in ("str", "int", "float", "bool"):
        assert generator._looks_json_safe(_annotation_expr(source)) is True


def test_no_domains_generated_or_skipped_sets_change(tmp_path: Path) -> None:
    """The cheapest real proof that fixing the recognizer is safe: regenerate every domain.

    Every ``Sequence``/``Mapping``-typed required parameter among the
    ledger's then-121 skipped capabilities had ``Any`` somewhere in its
    shape (see the module docstring) - so *this* fix (the JSON-safety
    recognizer, Capability Plan 03 Task 4 / Plan 05's bare-``Path``
    withdrawal) changed zero domains' generated/skipped id sets on its own.

    The totals asserted below are no longer that fix's own baseline of
    132/121: Capability Plan 05's directory-parameter codec (``codec =
    "directory"``, mandatory ``path_role``) was regenerated into five
    domains afterwards - ``morphology`` (+4), ``population`` (+2),
    ``phylogeny`` (+2), ``selection`` (+7), ``coevolution`` (+3), 18 total -
    each pinned by id in that domain's own
    ``tests/capabilities/test_restored_<domain>_bundles.py``. This test
    still proves what it always proved (regenerating every domain from
    scratch reproduces exactly the committed bundle tree, and no skip ever
    silently becomes ``capability.adapter_override_invalid``), just against
    the current measured totals: 199 generated, 56 skipped - the zero-out batch followed - the sequence-of-core batch followed - the legacy_result batch followed - the output-file path_role followed - the ideogram pair followed - the nested-row
    batch (qc_dashboard/matrix_heatmap/splicing/localization nested at real
    depth, summarize_synteny both arms) followed - Plan 03's
    module-level type-alias resolution (reading ``PathInput = str | Path``
    from the already-parsed module AST, never an import) admitted the
    ``visualization`` ``PathInput``-union trio (``plot_collinearity``,
    ``plot_gbdraw``, ``plot_gene_structure``), each pinned by id in that
    domain's own test module - the P1 mVISTA identity plot followed
    (``visualization.plot_genome_identity``), pinned by id in that domain's
    own test module.
    """
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"
    with open(_LEDGER_PATH, "rb") as handle:
        ledger = tomllib.load(handle)
    domains = sorted({record["domain"] for record in ledger["capability"]})

    total_generated = 0
    total_skipped = 0
    skip_codes: set[str] = set()
    all_generated_dirs: set[str] = set()
    for domain in domains:
        report = generator.build_domain(
            domain=domain,
            ledger_path=_LEDGER_PATH,
            current_root=_CURRENT_ROOT,
            output_root=isolated_output_root,
        )
        total_generated += len(report.generated)
        total_skipped += len(report.skipped)
        skip_codes.update(item.code for item in report.skipped)
        all_generated_dirs.update(item.path.parent.name for item in report.generated)

    assert total_generated == 199
    assert total_skipped == 56
    assert skip_codes == {
        "capability.no_adapter_override",
        # Ruling 3 follow-up: the one explicit domain-adapter refusal
        "capability.excluded_by_domain_adapter",
    }

    # What is committed under src/organelleverse/capabilities is exactly
    # what today's generator produces, across every domain at once - proved
    # by real, byte-for-byte comparison against a fresh, isolated
    # regeneration of the entire tree, never by writing into the committed
    # tree to find out. This was previously the single largest self-healer
    # in the suite: it regenerated all 22 domains into the real tree on
    # every run.
    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, all_generated_dirs
    )
