from __future__ import annotations

from pathlib import Path
from typing import Never, Protocol, cast

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    DependencyKind,
    DependencySpec,
    OperationRegistry,
    OperationSpec,
)
from organelleverse.operations.dependencies import DependencyState, check_dependencies


def _with_dependencies(analyze_spec: OperationSpec, *dependencies: DependencySpec) -> OperationSpec:
    return analyze_spec.model_copy(update={"dependencies": dependencies})


def _must_not_run(*args: object, **kwargs: object) -> Never:
    raise AssertionError("must not execute")


def _missing_module(name: str) -> None:
    return None


def _found_module(name: str) -> object:
    return object()


def _version_one(name: str) -> str:
    return "1.0"


def _executable_locator(name: str) -> str | None:
    return {
        "available": "/tools/available",
        "absent": None,
        "versioned": "/tools/versioned",
    }[name]


class SyspathPrependMonkeyPatch(Protocol):
    def syspath_prepend(self, path: str) -> None: ...


def test_dependency_report_models_are_frozen_and_forbid_extra(analyze_spec: OperationSpec) -> None:
    report = check_dependencies(analyze_spec)

    with pytest.raises(ValidationError):
        report.checks = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        type(report)(  # pyright: ignore[reportCallIssue]
            operation_id="annotation.annotate",
            checks=(),
            ready=True,
            extra=True,  # pyright: ignore[reportCallIssue]
        )


def test_dependency_check_does_not_execute_subprocesses(
    monkeypatch: MonkeyPatch, tmp_path: Path, analyze_spec: OperationSpec
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        "subprocess.run",
        _must_not_run,
    )
    spec = _with_dependencies(
        analyze_spec,
        DependencySpec(kind=DependencyKind.PYTHON, name="pydantic"),
        DependencySpec(kind=DependencyKind.EXECUTABLE, name="definitely-missing-tool"),
        DependencySpec(kind=DependencyKind.MODEL, name="model", locator=str(model)),
    )

    report = check_dependencies(spec)

    assert [item.state for item in report.checks] == [
        DependencyState.PRESENT,
        DependencyState.MISSING,
        DependencyState.PRESENT,
    ]


def test_python_check_uses_find_spec_without_importing_target(
    monkeypatch: MonkeyPatch, tmp_path: Path, analyze_spec: OperationSpec
) -> None:
    module_name = "dependency_target_that_must_not_import"
    (tmp_path / f"{module_name}.py").write_text("raise AssertionError('imported')\n")
    cast(SyspathPrependMonkeyPatch, monkeypatch).syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "importlib.import_module",
        _must_not_run,
    )
    spec = _with_dependencies(
        analyze_spec,
        DependencySpec(kind=DependencyKind.PYTHON, name=module_name),
    )

    report = check_dependencies(spec)

    assert report.checks[0].state is DependencyState.PRESENT


def test_python_check_reports_missing_and_version_mismatch(
    monkeypatch: MonkeyPatch, analyze_spec: OperationSpec
) -> None:
    monkeypatch.setattr("organelleverse.operations.dependencies.find_spec", _missing_module)
    missing = check_dependencies(
        _with_dependencies(analyze_spec, DependencySpec(kind=DependencyKind.PYTHON, name="absent"))
    )

    monkeypatch.setattr("organelleverse.operations.dependencies.find_spec", _found_module)
    monkeypatch.setattr("organelleverse.operations.dependencies.version", _version_one)
    mismatched = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.PYTHON, name="demo", version_spec=">=2"),
        )
    )

    assert missing.checks[0].state is DependencyState.MISSING
    assert mismatched.checks[0].state is DependencyState.VERSION_MISMATCH
    assert mismatched.checks[0].observed_version == "1.0"


def test_python_check_reports_invalid_name_and_version_spec_as_unverified(
    analyze_spec: OperationSpec,
) -> None:
    report = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.PYTHON, name="unsafe.parent"),
            DependencySpec(kind=DependencyKind.PYTHON, name="pydantic", version_spec="not a spec"),
        )
    )

    assert [item.state for item in report.checks] == [
        DependencyState.UNVERIFIED,
        DependencyState.UNVERIFIED,
    ]
    assert [item.details for item in report.checks] == [
        "invalid_python_import_name",
        "invalid_version_spec",
    ]


def test_executable_check_reports_presence_missing_and_unverified_version(
    monkeypatch: MonkeyPatch, analyze_spec: OperationSpec
) -> None:
    monkeypatch.setattr(
        "organelleverse.operations.dependencies.which",
        _executable_locator,
    )
    report = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.EXECUTABLE, name="available"),
            DependencySpec(kind=DependencyKind.EXECUTABLE, name="absent"),
            DependencySpec(kind=DependencyKind.EXECUTABLE, name="versioned", version_spec=">=1"),
        )
    )

    assert [item.state for item in report.checks] == [
        DependencyState.PRESENT,
        DependencyState.MISSING,
        DependencyState.UNVERIFIED,
    ]
    assert report.checks[0].resolved_locator == "/tools/available"
    assert report.checks[2].details == "version_requires_execution"


def test_model_and_database_checks_use_expected_path_type(
    tmp_path: Path, analyze_spec: OperationSpec
) -> None:
    model = tmp_path / "model.bin"
    database = tmp_path / "database"
    model.write_bytes(b"model")
    database.mkdir()
    report = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.MODEL, name="model", locator=str(model)),
            DependencySpec(kind=DependencyKind.DATABASE, name="database", locator=str(database)),
            DependencySpec(
                kind=DependencyKind.DATABASE, name="missing", locator=str(tmp_path / "missing")
            ),
        )
    )

    assert [item.state for item in report.checks] == [
        DependencyState.PRESENT,
        DependencyState.PRESENT,
        DependencyState.MISSING,
    ]


def test_optional_non_present_dependencies_do_not_block_readiness(
    analyze_spec: OperationSpec,
) -> None:
    report = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.PYTHON, name="missing_required"),
            DependencySpec(kind=DependencyKind.PYTHON, name="missing_optional", optional=True),
        )
    )
    optional_only = check_dependencies(
        _with_dependencies(
            analyze_spec,
            DependencySpec(kind=DependencyKind.PYTHON, name="missing_optional", optional=True),
        )
    )

    assert report.ready is False
    assert optional_only.ready is True


def test_check_order_matches_declared_dependency_order(
    monkeypatch: MonkeyPatch, analyze_spec: OperationSpec
) -> None:
    seen: list[str] = []

    def fake_find_spec(name: str) -> object:
        seen.append(name)
        return object()

    monkeypatch.setattr("organelleverse.operations.dependencies.find_spec", fake_find_spec)
    spec = _with_dependencies(
        analyze_spec,
        DependencySpec(kind=DependencyKind.PYTHON, name="first"),
        DependencySpec(kind=DependencyKind.PYTHON, name="second"),
    )

    report = check_dependencies(spec)

    assert seen == ["first", "second"]
    assert [item.dependency.name for item in report.checks] == ["first", "second"]


def test_registry_delegates_dependency_check(analyze_spec: OperationSpec) -> None:
    local = OperationRegistry()

    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        raise AssertionError("dependency inspection must not invoke operations")

    spec = _with_dependencies(
        analyze_spec,
        DependencySpec(kind=DependencyKind.PYTHON, name="missing_from_registry"),
    )
    local.register(spec, annotate)

    report = local.check_dependencies("annotation.annotate")

    assert report.operation_id == "annotation.annotate"
    assert report.ready is False
