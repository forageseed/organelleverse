"""Dependency checks must not report present things as missing."""
from __future__ import annotations

import tomllib
from pathlib import Path

from organelleverse.operations.dependencies import DependencyState, _check_path, _check_python
from organelleverse.operations.spec import DependencyKind, DependencySpec

CAPABILITIES = Path(__file__).resolve().parents[2] / "src" / "organelleverse" / "capabilities"
# distribution name -> import name; the check uses importlib.util.find_spec, i.e. the import name
DISTRIBUTION_TO_IMPORT = {"biopython": "Bio", "pillow": "PIL", "scikit-learn": "sklearn", "pyyaml": "yaml", "opencv-python": "cv2"}


def test_python_dependencies_are_declared_by_import_name():
    wrong = []
    for toml in sorted(CAPABILITIES.glob("*/capability.toml")):
        data = tomllib.loads(toml.read_text(encoding="utf-8"))
        for dep in data.get("contract", {}).get("dependencies", []):
            if dep.get("kind") == "python" and dep.get("name", "").lower() in DISTRIBUTION_TO_IMPORT:
                wrong.append((toml.parent.name, dep["name"]))
    assert not wrong, f"declared by distribution name, the check can never find them: {wrong}"


def test_biopython_is_found_by_its_import_name():
    dep = DependencySpec(kind=DependencyKind.PYTHON, name="Bio", locator="biopython")
    assert _check_python(dep, _empty_spec()).state is DependencyState.PRESENT


def test_packaged_asset_locator_is_resolved_not_treated_as_a_path():
    dep = DependencySpec(
        kind=DependencyKind.DATABASE, name="cms_curated_reference",
        locator="ov-asset:phenotype/cms_curated_reference@1.0",
    )
    check = _check_path(dep, _empty_spec(), requires_directory=True)
    assert check.state is DependencyState.PRESENT
    assert check.resolved_locator == "ov-asset:phenotype/cms_curated_reference@1.0"


def test_unknown_asset_is_missing():
    dep = DependencySpec(kind=DependencyKind.DATABASE, name="x", locator="ov-asset:nothing/here@9.9")
    assert _check_path(dep, _empty_spec(), requires_directory=True).state is DependencyState.MISSING


def _empty_spec():
    from packaging.specifiers import SpecifierSet

    return SpecifierSet()
