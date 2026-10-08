"""Tests for the output-FILE path_role (Ruling 1, Decision 004 approval 2026-08-14).

``codec = "path"`` + ``path_role = "output"`` names a single destination
file validated under the same containment rules directory outputs established
(traversal, symlink components, wrong leaf kind, managed-run-store overlap) —
and binds the three previously output-FILE-blocked writers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations.python_binding import (
    _PathParameterPlan,
    _validate_file_output_target,
)
from organelleverse.operations.spec import (
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
)


def _plan(target: type = str) -> _PathParameterPlan:
    return _PathParameterPlan(target_type=target, optional=False, multiple=False,
                               path_kind="file_output")


def _validate(value: str, tmp: Path):
    import organelleverse.runtime as runtime

    original = runtime.managed_runs_root
    runtime.managed_runs_root = lambda: tmp / "runs"  # type: ignore[assignment]
    try:
        return _validate_file_output_target(
            value, plan=_plan(), name="output_html", operation_id="op.test"
        )
    finally:
        runtime.managed_runs_root = original  # type: ignore[assignment]


def test_plain_destination_inside_a_real_directory_passes(tmp_path: Path) -> None:
    out = _validate(str(tmp_path / "report.html"), tmp_path)
    assert out == str(tmp_path / "report.html")


def test_missing_leaf_under_real_directory_ancestor_passes(tmp_path: Path) -> None:
    # mirrors directory-output semantics: a not-yet-existing leaf under a real
    # directory ancestor is a valid destination (creation is the callable's act)
    _validate(str(tmp_path / "no" / "deeper" / "x.html"), tmp_path)


def test_parent_traversal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as exc:
        _validate(str(tmp_path / ".." / "x.html"), tmp_path)
    assert exc.value.code == "input.file_destination_traversal"


def test_existing_directory_leaf_is_refused(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    with pytest.raises(OrganelleInputError) as exc:
        _validate(str(tmp_path / "adir"), tmp_path)
    assert exc.value.code == "input.file_destination_not_file"


def test_existing_regular_file_leaf_passes(tmp_path: Path) -> None:
    f = tmp_path / "existing.html"
    f.write_text("x", encoding="utf-8")
    _validate(str(f), tmp_path)


def test_symlink_component_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(OrganelleInputError) as exc:
        _validate(str(link / "x.html"), tmp_path)
    assert exc.value.code == "input.file_destination_not_file"  # symlink ancestor: wrong-kind, mirroring directory


def test_managed_run_store_overlap_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as exc:
        _validate(str(tmp_path / "runs" / "x.html"), tmp_path)
    assert exc.value.code == "input.file_destination_run_store"


def test_path_role_is_now_valid_on_the_path_codec() -> None:
    # the wire contract accepts path_role on path since Ruling 1
    ParameterBindingSpec(
        name="output_html",
        codec=ParameterCodec.PATH,
        source=ParameterSource.AGENT,
        path_role="output",
    )


def test_path_role_still_rejected_on_json() -> None:
    with pytest.raises(ValueError):
        ParameterBindingSpec(
            name="x",
            codec=ParameterCodec.JSON,
            source=ParameterSource.AGENT,
            path_role="output",
        )
