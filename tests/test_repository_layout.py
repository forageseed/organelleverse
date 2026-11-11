"""Repository layout guards for domain tests and packaged reference data."""

import ast

from tests._paths import PROJECT_ROOT


def test_only_repository_wide_tests_live_at_the_test_root() -> None:
    root_tests = {path.name for path in (PROJECT_ROOT / "tests").glob("test_*.py")}

    assert root_tests == {
        "test_bundled_tools.py",
        "test_clean_install.py",
        "test_portability.py",
        "test_repository_layout.py",
    }


def test_capability_scripts_directory_holds_only_the_generator_scripts() -> None:
    scripts_dir = PROJECT_ROOT / "scripts" / "capabilities"
    scripts = {path.name for path in scripts_dir.glob("*.py")}

    assert scripts == {
        "build_restoration_inventory.py",
        "build_restored_bundles.py",
        "migrate_release_catalog.py",
        "triage_unmigrated_capabilities.py",
        "realdata_matrix.py",
    }


def test_mitochondrion_database_keeps_one_canonical_reference_tree() -> None:
    reference_root = (
        PROJECT_ROOT
        / "src"
        / "annotation"
        / "data"
        / "mitochondrion"
        / "blast_refs"
    )
    reference_groups = {path.name for path in reference_root.iterdir() if path.is_dir()}

    # Boundary proteins use a distinct group to avoid case-only filename collisions on Windows.
    assert reference_groups == {"exons", "pcg", "pcg_boundary", "rrna", "rrna_mito", "trna"}


RETIRED_APPLICATION_LAYER = frozenset(
    {"desktop", "agent", "agent_skills", "notebooks", "skills", "revisions", "results", "plugin_runs", "cli", "compute"}
)


def test_retired_application_layer_is_not_imported() -> None:
    """The desktop/agent layer moved to PhytoScience; science code must not reach back for it.

    Relative imports are resolved against each module's own package, so a
    subpackage's local ``cli`` or ``results`` module is not mistaken for one.
    """
    # The real package directory (Windows: src/organelleverse; server: src). A directory of
    # symlinks into src is not the package and is skipped below.
    package = next(
        p for p in (PROJECT_ROOT / "src" / "organelleverse", PROJECT_ROOT / "src")
        if (p / "operations").is_dir() and not (p / "operations").is_symlink()
    )
    assert (package / "pangenome").is_dir(), package
    offenders = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or relative.parts[0] == "organelleverse":
            continue
        # The package a module's relative imports resolve against ("pkg/__init__" and "pkg/mod" alike).
        current = ["organelleverse", *relative.with_suffix("").parts][:-1]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # Unparseable work in progress is the compile checks' concern, not this guard's.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = current[: len(current) - node.level + 1] if node.level else []
                target = ".".join([*base, *(node.module.split(".") if node.module else [])])
                names = [target] if node.module else [f"{target}.{alias.name}" for alias in node.names]
                if target == "organelleverse":
                    names = [f"organelleverse.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                pieces = name.split(".")
                if len(pieces) > 1 and pieces[0] == "organelleverse" and pieces[1] in RETIRED_APPLICATION_LAYER:
                    offenders.append(f"{relative}:{node.lineno} {name}")
    assert offenders == []
