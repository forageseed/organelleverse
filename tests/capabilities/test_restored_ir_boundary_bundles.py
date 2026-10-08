"""Capability Plan 03, Task 3: the restored ``ir_boundary`` bundles, for real.

Unlike every domain resolved so far, **zero** of ``ir_boundary``'s 3 ledger
records can be honestly bound - this is not a bug in the generator or an
oversight in an adapter, it is the fail-closed answer for this domain:

* ``ir_boundary.compute_ir_boundary`` - its sole parameter is annotated
  ``genomes: list`` (a bare, unparameterized ``list`` -
  ``ir_boundary/ir_boundary_core.py:8``, ``from __future__ import
  annotations`` makes this a string ``"list"`` at the AST level too). Not a
  path (it is not a file), and the generator's own ``_looks_json_safe``
  rejects a bare ``Name("list")`` outright (only ``{str, int, float, bool}``
  are JSON-safe *names*; ``"list"`` is only recognized as a *container*
  prefix on a ``Subscript``, and there is no subscript here at all).
* ``ir_boundary.ir_boundary`` - its sole parameter is
  ``genomes: list[OrganelleGenome]`` (``ir_boundary/ir_boundary.py:67-69``).
  This is a **new blocker class**, first encountered in this domain:
  a list of a core domain type has no bindable codec at all.
  ``canonical_core`` detection only recognizes a *bare* ``OrganelleGenome``/
  ``OrganelleData``/``OrganelleResult`` first parameter, never a list of
  one. ``ParameterCodec.PATH`` cannot apply (not a file path).
  ``ParameterCodec.JSON`` cannot apply either: ``organelleverse.core.genome
  .OrganelleGenome`` is a ``StrictFrozenModel``, not one of the two
  JSON-safe escape hatches ``operations.signature._is_supported_json_annotation``
  recognizes (a JSON primitive, or a registered
  ``operations.parameters.OperationParameterModel``/``OrganelleMetadata``) -
  so ``list[OrganelleGenome]`` is rejected by the *real* runtime check (the
  strict codec-wall variant, ``allow_path=False``), and - since the
  generator's ``_looks_json_safe`` became fully recursive over container
  elements - by the generator's AST check too: both walls now agree, where
  the original outer-name-only check would have waved it through, exactly
  the lenient-generator/strict-binder mismatch already documented in
  ``organelleverse.capabilities.adapters.phylogeny`` for ``list[dict]`` -
  just with a core Pydantic model as the list's element type instead of a
  bare ``dict``. (Not asserted here with a real bind attempt, since doing so
  would require generating a bundle this domain deliberately does not
  produce; recorded from direct inspection of
  ``_is_supported_json_annotation`` instead - see the module docstring of
  ``organelleverse.capabilities.adapters.comparative``, which resolves
  several capabilities with this exact shape and *does* prove the failure
  with a real bind attempt.)
* ``ir_boundary.write_boundary_svg`` - required
  ``result: OrganelleResult | Mapping[str, Any]`` - the charter's
  ``LEGACY_RESULT`` gap verbatim.

No domain adapter module exists for ``ir_boundary`` - none of its
capabilities is rescuable by one, so (per the charter's instruction not to
author an empty module for symmetry) none was created; the generator's
``_load_overrides`` falls back to an empty override table on
``ModuleNotFoundError`` exactly as it does for every other domain with no
module, and every record fails closed with
``capability.no_adapter_override``.

Because nothing is generated, there is nothing to discover, admit, or
invoke for this domain - this test file only pins the generator's report.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from tests._paths import PROJECT_ROOT

_EXCLUDED_IDS = (
    "ir_boundary.compute_ir_boundary",
    "ir_boundary.write_boundary_svg",
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


def test_the_generator_produces_no_ids_and_skips_every_record(tmp_path: Path) -> None:
    """Generates into an isolated ``tmp_path`` output root, never
    ``_BUNDLE_ROOT`` - even though nothing is generated for this domain
    today (every record is skipped, so ``build_domain`` never writes a
    bundle directory), keeping ``output_root`` isolated here too means this
    test cannot start silently rewriting tracked source the day a future
    adapter rescues even one of ``ir_boundary``'s records.
    """
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="ir_boundary",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == {"ir_boundary.ir_boundary"}
    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {
        **{excluded_id: "capability.no_adapter_override" for excluded_id in _EXCLUDED_IDS},
        # Zero-out ruling: explicit domain-adapter refusal (see the adapter note)
        "ir_boundary.compute_ir_boundary": "capability.excluded_by_domain_adapter",
    }

    # Nothing generated -> nothing to byte-compare; this checks the real,
    # committed tree directly (read-only) instead, which is the correct
    # target for "is anything for this domain actually committed".
    on_disk_dirs = {path.parent.name for path in _BUNDLE_ROOT.glob("ir_boundary-*/capability.toml")}
    assert on_disk_dirs == set()


def test_no_ir_boundary_capability_is_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every excluded id is simply absent from the real core channel - never rejected."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)

    index = discover_capabilities()
    discovered_ids = {entry.capability_id for entry in index.entries}
    for excluded_id in _EXCLUDED_IDS:
        assert excluded_id not in discovered_ids
