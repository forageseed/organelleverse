from pathlib import Path


def test_backend_modules_match_the_released_implementation_set() -> None:
    root = Path(__file__).parents[2] / "src" / "organelleverse" / "assembly"
    backend_modules = {
        path.stem for path in (root / "backends").glob("*.py") if path.name != "__init__.py"
    }
    assert backend_modules == {
        "base",
        "getorganelle",
        "himt",
        "oatk",
        "pmat",
        "pmat_graph",
        "p2_cli",
        "tippo",
        "ptgaul",
        "novoplasty",
        "ovasm",
        "registry",
        "runtime",
        "spec",
    }

    environment_resources = {
        path.name
        for path in (root / "resources" / "environments").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert environment_resources == {"getorganelle", "himt", "oatk", "pmat", "novoplasty"}
