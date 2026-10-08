"""Restoration inventory plus native additions: schema, counts, and archive generator.

``docs/operations/restored-capabilities.toml`` is a migration ledger, not a
runtime contract: it must never register an ``OperationSpec`` or enter the
Registry (see ``tests/contracts/test_legacy_removed.py`` for the Registry's
own 20-operation and empty-``CONTRACTED_SUITES`` guarantees). This file only
checks that the ledger itself is exact, portable, and was produced by a
generator that fails closed on a bad count and is reproducible byte-for-byte.

The generator's own archive-vs-current-source resolution step is exercised
here with synthetic fixtures under ``tmp_path`` rather than the real external
archive: the real archive lives outside this repository
(``.archives/organelleverse-monorepo-20260728/``, passed via
``--archive-root``) and must never be hardcoded into a portable test.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
INVENTORY = ROOT / "docs" / "operations" / "restored-capabilities.toml"
GENERATOR_PATH = ROOT / "scripts" / "capabilities" / "build_restoration_inventory.py"

EXPECTED_DOMAIN_COUNTS = {
    "barcode": 4,
    "codon_composition": 8,
    "coevolution": 26,
    "comparative": 13,
    "composition": 3,
    "diversity": 6,
    "format_conversion": 4,
    "hgt": 4,
    "ir_boundary": 3,
    "localization": 10,
    "morphology": 14,
    "pangenome": 9,
    "phylogeny": 20,
    "population": 9,
    "rna_editing": 14,
    "selection": 32,
    "structure": 10,
    "trans_splicing": 2,
    "transfer": 10,
    "variation": 5,
    "visualization": 40,
    "phenotype": 7,
}


# Two native additions extend the live ledger, not the historical archive.
CURRENT_DOMAIN_COUNTS = {**EXPECTED_DOMAIN_COUNTS, "comparative": 14, "visualization": 41}


def _load_records() -> list[dict[str, Any]]:
    payload = tomllib.loads(INVENTORY.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload["capability"]
    return records


def _load_generator() -> ModuleType:
    """Load the standalone generator script as a module, without a package.

    Deliberately not imported as ``scripts.capabilities.build_restoration_inventory``:
    ``scripts/`` is a CLI-script directory, not an importable package, and
    pytest's configured ``pythonpath`` only covers ``src/``. Attribute access
    on the result is necessarily dynamic (pyright cannot know a
    file-loaded module's members statically); that is inherent to this
    loading pattern, not a typing gap worth suppressing line-by-line.
    """
    spec = importlib.util.spec_from_file_location(
        "build_restoration_inventory", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses (with `from __future__ import annotations`) resolves string
    # annotations via sys.modules[cls.__module__]; the module must be
    # registered before exec_module runs, exactly as a real import would.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- the frozen ledger itself -----------------------------------------------


def test_restoration_inventory_is_exact_and_includes_cms() -> None:
    records = _load_records()
    ids = [record["id"] for record in records]
    locators = [record["python_locator"] for record in records]
    domains = {record["domain"] for record in records}

    assert len(records) == 255
    assert len(ids) == len(set(ids))
    assert len(locators) == len(set(locators))
    assert len(domains) == 22
    assert "phenotype" in domains
    assert any(locator.endswith(".phenotype.cms.pipeline:cms") for locator in locators)
    assert all(
        record["execution_class"] in {"pure", "tool", "model", "network"}
        for record in records
    )
    assert all(
        record["result_shape"] in {"canonical", "legacy_result", "json", "artifact", "custom"}
        for record in records
    )


def test_inventory_matches_the_independently_verified_per_domain_counts() -> None:
    records = _load_records()
    actual = dict(Counter(record["domain"] for record in records))
    assert actual == CURRENT_DOMAIN_COUNTS
    base_total = sum(count for domain, count in CURRENT_DOMAIN_COUNTS.items() if domain != "phenotype")
    assert base_total == 248
    assert EXPECTED_DOMAIN_COUNTS["phenotype"] == 7


def test_inventory_records_have_the_minimum_required_fields() -> None:
    required = {
        "id",
        "domain",
        "public_name",
        "python_locator",
        "source_relpath",
        "execution_class",
        "result_shape",
        "dependencies",
    }
    for record in _load_records():
        assert required <= set(record), f"record missing fields: {record}"


def test_source_relpath_is_portable() -> None:
    """No absolute path, no traversal: relative to ``src/organelleverse`` only.

    This does not also forbid specific local-machine path fragments (a
    developer username, a disk-label directory) the way
    ``tests/test_portability.py`` does repo-wide: hardcoding those exact
    fragments as string literals here would itself trip that repo-wide scan.
    """
    for record in _load_records():
        relpath = record["source_relpath"]
        assert not relpath.startswith("/"), relpath
        assert Path(relpath).is_absolute() is False, relpath
        assert ".." not in Path(relpath).parts, relpath


def test_python_locator_matches_domain_and_source_relpath() -> None:
    for record in _load_records():
        relpath = Path(record["source_relpath"])
        module_parts = list(relpath.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        expected_module = "organelleverse." + ".".join(module_parts)
        expected_locator = f"{expected_module}:{record['public_name']}"
        assert record["python_locator"] == expected_locator
        assert record["source_relpath"].split("/")[0] == record["domain"]


def test_inventory_is_a_ledger_not_a_runtime_contract() -> None:
    """Task 1 freezes a migration ledger; it must not register operations."""
    import organelleverse as ov

    registered_ids = {spec.operation_id for spec in ov.operations.list()}
    ledger_locators = {record["python_locator"] for record in _load_records()}
    assert registered_ids.isdisjoint(ledger_locators)
    assert len(registered_ids) == 20


# --- the generator itself (synthetic fixtures only; no external archive) ---


def _write_domain_functions(root: Path, domain: str, relpath: str, names: list[str]) -> None:
    path = root / domain / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(f"def {name}():\n    return None" for name in names)
    path.write_text(body + "\n" if body else "", encoding="utf-8")


def _build_matched_fixture(archive_root: Path, current_root: Path) -> None:
    """A tiny synthetic archive + current-source pair with correct counts.

    Every domain gets its exact expected count of trivially-named functions,
    placed identically under both roots so the generator's archive-vs-current
    resolution step succeeds without ever touching the real repository
    source or the real external archive.
    """
    for domain, count in EXPECTED_DOMAIN_COUNTS.items():
        prefix = "cms_fn_" if domain == "phenotype" else "fn_"
        names = [f"{prefix}{index}" for index in range(count)]
        _write_domain_functions(archive_root, domain, "__init__.py", names)
        _write_domain_functions(current_root, domain, "__init__.py", names)
    # visualization must use a non-recursive glob: a nested file's functions
    # must never be counted, even though every other domain uses rglob.
    _write_domain_functions(archive_root, "visualization", "nested/extra.py", ["nested_only"])
    _write_domain_functions(current_root, "visualization", "nested/extra.py", ["nested_only"])
    # a non-visualization domain's nested file DOES count (rglob), matched by
    # temporarily removing one top-level function and adding it back nested.
    barcode_names = [f"fn_{index}" for index in range(EXPECTED_DOMAIN_COUNTS["barcode"] - 1)]
    _write_domain_functions(archive_root, "barcode", "__init__.py", barcode_names)
    _write_domain_functions(current_root, "barcode", "__init__.py", barcode_names)
    _write_domain_functions(archive_root, "barcode", "nested/extra.py", ["fn_nested"])
    _write_domain_functions(current_root, "barcode", "nested/extra.py", ["fn_nested"])


def test_generator_reproduces_matched_fixture_counts(tmp_path: Path) -> None:
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)

    records = generator.build_inventory_records(archive_root, current_root)
    assert len(records) == 253
    actual = dict(Counter(record["domain"] for record in records))
    assert actual == EXPECTED_DOMAIN_COUNTS


def test_generator_output_is_byte_identical_across_repeated_runs(tmp_path: Path) -> None:
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)

    first = generator.render_toml(generator.build_inventory_records(archive_root, current_root))
    second = generator.render_toml(generator.build_inventory_records(archive_root, current_root))
    assert first == second


def test_generator_fails_closed_on_domain_count_mismatch(tmp_path: Path) -> None:
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)
    # Corrupt one domain's count by adding an extra archive-only function.
    _write_domain_functions(archive_root, "hgt", "extra_module.py", ["one_function_too_many"])

    with pytest.raises(generator.InventoryValidationError):
        generator.build_inventory_records(archive_root, current_root)


def test_generator_cli_exits_non_zero_and_writes_nothing_on_mismatch(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)
    _write_domain_functions(archive_root, "hgt", "extra_module.py", ["one_function_too_many"])
    output = tmp_path / "broken.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR_PATH),
            "--archive-root",
            str(archive_root),
            "--output",
            str(output),
        ],
        cwd=current_root.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "Traceback" not in result.stderr, result.stderr
    assert "hgt" in result.stderr, result.stderr


def test_generator_resolves_a_moved_function_to_its_new_current_location(tmp_path: Path) -> None:
    """A function present in the archive but relocated in current source.

    The generator must not silently drop it, and must not keep pointing at
    its stale archived path: it must record the canonical *current* locator.
    """
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)

    # Move one function in current source only: delete it from its archived
    # location's counterpart and add it under a different file in the same
    # domain.
    remaining = [
        name
        for name in (f"fn_{index}" for index in range(EXPECTED_DOMAIN_COUNTS["barcode"] - 1))
        if name != "fn_0"
    ]
    _write_domain_functions(current_root, "barcode", "__init__.py", remaining)
    _write_domain_functions(current_root, "barcode", "relocated.py", ["fn_0"])

    records = generator.build_inventory_records(archive_root, current_root)
    barcode_fn_0 = next(r for r in records if r["domain"] == "barcode" and r["public_name"] == "fn_0")
    assert barcode_fn_0["source_relpath"] == "barcode/relocated.py"
    assert barcode_fn_0["python_locator"] == "organelleverse.barcode.relocated:fn_0"


def test_generator_fails_closed_when_a_capability_has_no_current_counterpart(tmp_path: Path) -> None:
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)

    remaining = [
        name
        for name in (f"fn_{index}" for index in range(EXPECTED_DOMAIN_COUNTS["barcode"] - 1))
        if name != "fn_0"
    ]
    _write_domain_functions(current_root, "barcode", "__init__.py", remaining)
    # fn_0 now exists nowhere under current_root's barcode domain.

    with pytest.raises(generator.InventoryValidationError):
        generator.build_inventory_records(archive_root, current_root)


def test_duplicate_public_name_within_a_domain_gets_a_module_qualified_id(tmp_path: Path) -> None:
    generator = _load_generator()
    archive_root = tmp_path / "archive"
    current_root = tmp_path / "current"
    _build_matched_fixture(archive_root, current_root)

    # rna_editing trades 2 of its baseline functions for two files that both
    # define the same function name, holding the domain (and base) total
    # steady so only the duplicate-name behavior is under test here.
    reduced = [f"fn_{index}" for index in range(EXPECTED_DOMAIN_COUNTS["rna_editing"] - 2)]
    _write_domain_functions(archive_root, "rna_editing", "__init__.py", reduced)
    _write_domain_functions(current_root, "rna_editing", "__init__.py", reduced)
    _write_domain_functions(archive_root, "rna_editing", "one.py", ["shared_name"])
    _write_domain_functions(archive_root, "rna_editing", "two.py", ["shared_name"])
    _write_domain_functions(current_root, "rna_editing", "one.py", ["shared_name"])
    _write_domain_functions(current_root, "rna_editing", "two.py", ["shared_name"])

    records = generator.build_inventory_records(archive_root, current_root)
    ids = [r["id"] for r in records if r["domain"] == "rna_editing" and r["public_name"] == "shared_name"]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert all(i != "rna_editing.shared_name" for i in ids)
