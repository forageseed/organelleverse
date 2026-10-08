"""Clean-environment import checks for the restored scientific runtime.

These tests must pass against an environment created purely from the
project's own declared dependencies (``pip install -e '.[dev]'`` into a
fresh virtual environment) — not against whatever happens to already be on
``sys.path`` in a developer's shared conda environment. A package that a
restored suite imports unconditionally but that is missing from
``[project].dependencies`` will import correctly in a stale developer
environment and fail in a genuinely clean one; that gap is exactly what
these tests exist to catch.

Every subprocess below uses ``sys.executable`` so the check reflects
whichever interpreter is actually running pytest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests._paths import child_env

# The 22 domains restored in feat(suites) (commit 4f11c1a6), kept as an
# explicit literal here rather than imported from tests/contracts/ (which is
# not a regular package: no tests/contracts/__init__.py).
RESTORED_SUITES = (
    "barcode",
    "codon_composition",
    "coevolution",
    "comparative",
    "composition",
    "diversity",
    "format_conversion",
    "hgt",
    "ir_boundary",
    "localization",
    "morphology",
    "pangenome",
    "phenotype",
    "phylogeny",
    "population",
    "rna_editing",
    "selection",
    "structure",
    "trans_splicing",
    "transfer",
    "variation",
    "visualization",
)

# Packages a restored suite imports unconditionally at module scope, that
# must therefore be real project dependencies rather than something a
# developer's ambient environment happens to already provide.
RESTORED_RUNTIME_PACKAGES = ("numpy", "scipy", "matplotlib", "skimage")


def test_bare_import_stays_lightweight(tmp_path: Path) -> None:
    """``import organelleverse`` alone must not pull in any restored domain
    or any of its heavy runtime packages."""
    script = f"""
import json
import sys

before = set(sys.modules)
import organelleverse  # noqa: F401
after = set(sys.modules)

loaded_domains = sorted(
    name for name in {RESTORED_SUITES!r}
    if any(m == f"organelleverse.{{name}}" or m.startswith(f"organelleverse.{{name}}.")
           for m in after)
)
loaded_heavy = sorted(
    name for name in {RESTORED_RUNTIME_PACKAGES!r}
    if name in after and name not in before
)
print(json.dumps({{"loaded_domains": loaded_domains, "loaded_heavy": loaded_heavy}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env(),
    )
    result = json.loads(completed.stdout)
    assert result["loaded_domains"] == []
    assert result["loaded_heavy"] == []


def test_every_public_module_imports_after_declared_dependency_install(
    tmp_path: Path,
) -> None:
    """Every name in ``organelleverse._PUBLIC_MODULES`` must resolve.

    Sized dynamically off the real set rather than a hardcoded count, so this
    stays correct as the public surface grows; today that set has 31 entries.
    """
    script = """
import json
import sys

import organelleverse as ov

names = sorted(ov._PUBLIC_MODULES)
ok = []
failed = {}
for name in names:
    try:
        getattr(ov, name)
        ok.append(name)
    except Exception as error:
        failed[name] = f"{type(error).__name__}: {error}"

print(json.dumps({"total": len(names), "ok": ok, "failed": failed}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env(),
    )
    result = json.loads(completed.stdout)
    assert result["failed"] == {}
    assert len(result["ok"]) == result["total"]


def test_every_restored_domain_imports_after_declared_dependency_install(
    tmp_path: Path,
) -> None:
    """All 22 restored domains must import from declared dependencies alone."""
    script = f"""
import importlib
import json

failed = {{}}
ok = []
for name in {RESTORED_SUITES!r}:
    try:
        importlib.import_module(f"organelleverse.{{name}}")
        ok.append(name)
    except Exception as error:
        failed[name] = f"{{type(error).__name__}}: {{error}}"

print(json.dumps({{"ok": ok, "failed": failed}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env(),
    )
    result = json.loads(completed.stdout)
    assert result["failed"] == {}
    assert len(result["ok"]) == len(RESTORED_SUITES)
