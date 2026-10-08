"""Repository layout guards for domain tests and packaged reference data."""

from tests._paths import PROJECT_ROOT


def test_only_repository_wide_tests_live_at_the_test_root() -> None:
    root_tests = {path.name for path in (PROJECT_ROOT / "tests").glob("test_*.py")}

    assert root_tests == {
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
        / "organelleverse"
        / "annotation"
        / "data"
        / "mitochondrion"
        / "blast_refs"
    )
    reference_groups = {path.name for path in reference_root.iterdir() if path.is_dir()}

    assert reference_groups == {"exons", "pcg", "rrna", "rrna_mito", "trna"}
