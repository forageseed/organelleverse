import tomllib
from pathlib import Path


def test_pydantic_dependency_matches_used_field_info_api() -> None:
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())

    assert "pydantic>=2.12.3" in project["project"]["dependencies"]
